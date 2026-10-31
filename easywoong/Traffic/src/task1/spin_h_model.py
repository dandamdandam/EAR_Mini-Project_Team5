"""PyTorch-only port of the official SPIN-H imputation architecture.

Reference: Graph-Machine-Learning-Group/spin, MIT license, commit recorded in
``docs/spin-h.md``.  The original depends on PyG, torch_scatter and TSL, none
of which are installed in the Traffic environment.  This module implements the
same hierarchical hub-to-observation and observation-to-hub attention equations
with ordinary PyTorch tensor and scatter operations.  It uses the published
SPIN-H configuration: 32 observation channels, 128 hub channels, four hubs per
node, five layers, spatial softmax and eta=3.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _mlp(input_size: int, hidden_size: int, output_size: int | None,
         n_layers: int, *, activation: str = "relu") -> nn.Sequential:
    """Match TSL 0.1.1 MLP: n Dense blocks and an optional linear readout."""
    layers: list[nn.Module] = []
    for i in range(n_layers):
        layers.append(nn.Linear(input_size if i == 0 else hidden_size, hidden_size))
        layers.append(nn.PReLU() if activation == "prelu" else nn.ReLU())
    if output_size is not None:
        layers.append(nn.Linear(hidden_size, output_size))
    return nn.Sequential(*layers)


class TSLRelationNorm(nn.Module):
    """TSL 0.1.1 LayerNorm divides by std + eps, unlike torch LayerNorm."""

    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, value: Tensor) -> Tensor:
        mean = value.mean(dim=-1, keepdim=True)
        std = value.std(dim=-1, unbiased=False, keepdim=True)
        return (value - mean) / (std + 1e-5) * self.weight + self.bias


class TemporalAdditiveAttention(nn.Module):
    """Dense-in-time equivalent of the upstream TemporalAdditiveAttention."""

    def __init__(self, source_size: int, target_size: int, output_size: int,
                 *, reweight: str | None, norm: bool):
        super().__init__()
        if reweight not in ("softmax", None):
            raise ValueError(reweight)
        self.reweight = reweight
        self.lin_src = nn.Linear(source_size, output_size)
        self.lin_tgt = nn.Linear(target_size, output_size, bias=False)
        self.msg_nn = nn.Sequential(
            nn.PReLU(init=0.2),
            _mlp(output_size, output_size, output_size, 1, activation="prelu"),
        )
        self.msg_gate = nn.Linear(output_size, 1, bias=reweight is None)
        self.lin_skip = nn.Linear(target_size, output_size, bias=False)
        self.norm = nn.LayerNorm(output_size) if norm else nn.Identity()
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_tgt.weight)

    def forward(self, source: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
        # source [B,S,N,Cs], target [B,L,N,Ct], output [B,L,N,D].
        if source.shape[0] == 1 and target.shape[0] != 1:
            source = source.expand(target.shape[0], -1, -1, -1)
        message = self.msg_nn(
            self.lin_src(source).unsqueeze(1) + self.lin_tgt(target).unsqueeze(2)
        )
        logits = self.msg_gate(message)
        valid = mask.unsqueeze(1).bool() if mask is not None else None
        if self.reweight == "softmax":
            if valid is not None:
                logits = logits.float().masked_fill(~valid, -1e9)
            weights = torch.softmax(logits.float(), dim=2)
            if valid is not None:
                # A sensor can have no valid observations in a whole window.
                weights = weights * valid
        else:
            weights = torch.sigmoid(logits.float())
            if valid is not None:
                weights = weights * valid
        out = (message.float() * weights).sum(dim=2)
        return self.norm(out + self.lin_skip(target))


def _spatial_softmax(message: Tensor, destination: Tensor, n_nodes: int,
                     scorer: nn.Linear) -> Tensor:
    """Normalize incoming edge messages per target node, then sum them."""
    batch, steps, edges, channels = message.shape
    score = scorer(message).float()
    index_score = destination.view(1, 1, edges, 1).expand(batch, steps, -1, 1)
    maxima = torch.full((batch, steps, n_nodes, 1), -float("inf"),
                        dtype=score.dtype, device=score.device)
    maxima.scatter_reduce_(2, index_score, score, reduce="amax", include_self=True)
    shifted = score - maxima.gather(2, index_score)
    unnormalized = shifted.exp()
    normalizer = torch.zeros_like(maxima).scatter_add_(2, index_score, unnormalized)
    weights = unnormalized / normalizer.gather(2, index_score).clamp_min(1e-12)
    index_message = destination.view(1, 1, edges, 1).expand(batch, steps, -1, channels)
    return torch.zeros((batch, steps, n_nodes, channels), dtype=message.dtype,
                       device=message.device).scatter_add_(
        2, index_message, message * weights.to(message.dtype)
    )


class HierarchicalAttentionLayer(nn.Module):
    """SPIN-H layer with update_z_cross=False and spatial_aggr='softmax'."""

    def __init__(self, h_size: int, z_size: int):
        super().__init__()
        self.zh_self = TemporalAdditiveAttention(h_size, z_size, z_size,
                                                  reweight="softmax", norm=True)
        self.hz_self = TemporalAdditiveAttention(z_size, h_size, h_size,
                                                  reweight="softmax", norm=False)
        self.hz_cross = TemporalAdditiveAttention(z_size, h_size, h_size,
                                                   reweight=None, norm=False)
        self.lin_alpha_h = nn.Linear(h_size, 1, bias=False)
        self.h_skip = nn.Linear(h_size, h_size)
        self.z_skip = nn.Linear(z_size, z_size)
        self.h_norm = TSLRelationNorm(h_size)
        self.z_norm = TSLRelationNorm(z_size)

    def forward(self, h: Tensor, z: Tensor, edge_index: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        z_out = self.zh_self(h, z, mask=mask)
        h_self = self.hz_self(z_out, h)
        src, dst = edge_index[0], edge_index[1]
        cross = self.hz_cross(z_out.index_select(2, src), h_self.index_select(2, dst))
        h_out = _spatial_softmax(cross, dst, h.shape[2], self.lin_alpha_h)
        h_out = self.h_norm(h_out + self.h_skip(h))
        z_out = self.z_norm(z_out + self.z_skip(z))
        return h_out, z_out


class SPINH(nn.Module):
    """One shared SPIN-H model for all panels with global node embeddings."""

    def __init__(self, total_nodes: int, *, input_size: int = 2, h_size: int = 32,
                 z_size: int = 128, z_heads: int = 4, n_layers: int = 5, eta: int = 3):
        super().__init__()
        if not 0 <= eta < n_layers:
            raise ValueError("eta must index one of the hidden layers")
        self.h_size, self.z_size = h_size, z_size
        self.n_layers, self.eta = n_layers, eta
        self.v = nn.Embedding(total_nodes, h_size)
        self.lin_v = nn.Linear(h_size, z_size, bias=False)
        self.z = nn.Parameter(torch.empty(1, z_heads, total_nodes, z_size))
        self.z_norm = nn.LayerNorm(z_size)
        self.u_linear = nn.Linear(4, h_size)
        self.u_mlp = _mlp(h_size, h_size, h_size, 2)
        self.h_enc = _mlp(input_size, h_size, None, 2)
        self.h_norm = nn.LayerNorm(h_size)
        self.v1, self.m1 = nn.Embedding(total_nodes, h_size), nn.Embedding(total_nodes, h_size)
        self.v2, self.m2 = nn.Embedding(total_nodes, h_size), nn.Embedding(total_nodes, h_size)
        self.x_skip = nn.ModuleList(nn.Linear(input_size, h_size) for _ in range(n_layers))
        self.encoder = nn.ModuleList(HierarchicalAttentionLayer(h_size, z_size)
                                     for _ in range(n_layers))
        self.readout = nn.ModuleList(_mlp(h_size, z_size, input_size, 2)
                                     for _ in range(n_layers))
        for embedding in (self.v, self.v1, self.m1, self.v2, self.m2):
            nn.init.uniform_(embedding.weight, -1 / math.sqrt(h_size), 1 / math.sqrt(h_size))
        nn.init.uniform_(self.z, -1 / math.sqrt(z_size), 1 / math.sqrt(z_size))

    def forward(self, x: Tensor, u: Tensor, mask: Tensor, edge_index: Tensor,
                node_index: Tensor) -> tuple[Tensor, list[Tensor]]:
        if x.ndim != 4 or x.shape[-1] != 2 or mask.shape != (*x.shape[:3], 1):
            raise ValueError("Expected x [B,T,N,2] and mask [B,T,N,1]")
        if u.shape != (x.shape[0], x.shape[1], 4):
            raise ValueError("Expected time encoding [B,T,4]")
        if node_index.numel() != x.shape[2] or edge_index.shape[0] != 2:
            raise ValueError("Graph and input node counts differ")
        v_nodes = self.v(node_index)
        z = self.z.index_select(2, node_index) + self.lin_v(v_nodes)
        time_nodes = torch.nn.functional.leaky_relu(
            self.u_linear(u).unsqueeze(2) + v_nodes, negative_slope=0.01
        )
        q = self.u_mlp(time_nodes)
        position = torch.arange(x.shape[1], device=x.device, dtype=q.dtype).unsqueeze(1)
        scale = torch.exp(torch.arange(0, self.h_size, 2, device=x.device, dtype=q.dtype)
                          * (-math.log(10000.0) / self.h_size))
        pe = torch.zeros((x.shape[1], self.h_size), device=x.device, dtype=q.dtype)
        pe[:, 0::2] = torch.sin(position * scale)
        pe[:, 1::2] = torch.cos(position * scale)
        q = q + pe.view(1, x.shape[1], 1, self.h_size)
        observed = mask.bool()
        h = torch.where(observed, self.h_enc(x) + q + self.v1(node_index),
                        q + self.m1(node_index))
        h, z = self.h_norm(h), self.z_norm(z)
        imputations: list[Tensor] = []
        for layer in range(self.n_layers):
            if layer == self.eta:
                h = torch.where(observed, h + self.v2(node_index), h + self.m2(node_index))
            h = h + self.x_skip[layer](x) * mask
            h, z = self.encoder[layer](h, z, edge_index, mask)
            imputations.append(self.readout[layer](h))
        return imputations[-1], imputations[:-1]


def masked_l1_with_auxiliary(prediction: Tensor, auxiliary: list[Tensor],
                             truth: Tensor, target: Tensor) -> Tensor:
    """Published SPIN imputer loss: final L1 plus each intermediate L1."""
    target = target.bool().expand_as(truth)
    if not target.any():
        raise ValueError("Training batch has no hidden eligible cell")
    loss = (prediction.float()[target] - truth.float()[target]).abs().mean()
    for value in auxiliary:
        loss = loss + (value.float()[target] - truth.float()[target]).abs().mean()
    return loss
