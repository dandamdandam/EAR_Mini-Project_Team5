"""STAEformer-style spatial/temporal encoder for Task 1 residual imputation.

This is an independent adaptation of the architecture in the STAEformer paper
(https://arxiv.org/abs/2308.10425).  The published model forecasts future
steps; here the output is a correction to offline interpolation at the same
24 input steps.  Global node IDs make one model usable across all panels.
"""

from __future__ import annotations

import torch
from torch import nn


class AxisAttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float, axis: str):
        super().__init__()
        if axis not in {"time", "space"}:
            raise ValueError(axis)
        self.axis = axis
        self.norm_attn = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, width),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, width = x.shape
        if self.axis == "time":
            tokens = x.transpose(1, 2).reshape(batch * nodes, steps, width)
        else:
            tokens = x.reshape(batch * steps, nodes, width)
        normal = self.norm_attn(tokens)
        update, _ = self.attn(normal, normal, normal, need_weights=False)
        tokens = tokens + self.dropout(update)
        tokens = tokens + self.dropout(self.feedforward(self.norm_ff(tokens)))
        if self.axis == "time":
            return tokens.reshape(batch, nodes, steps, width).transpose(1, 2)
        return tokens.reshape(batch, steps, nodes, width)


class STAEformerResidual(nn.Module):
    """Inputs [B,24,N,5] -> residual in [25 km/h, 600 veh/h/lane] units."""

    def __init__(
        self,
        *,
        total_nodes: int,
        steps: int = 24,
        input_dim: int = 5,
        feature_dim: int = 32,
        tod_dim: int = 16,
        dow_dim: int = 8,
        adaptive_dim: int = 40,
        heads: int = 4,
        layers_each_axis: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        width = feature_dim + tod_dim + dow_dim + adaptive_dim
        if width % heads:
            raise ValueError("Model width must be divisible by number of heads")
        self.steps = steps
        self.total_nodes = total_nodes
        self.feature_proj = nn.Linear(input_dim, feature_dim)
        self.time_of_day = nn.Embedding(288, tod_dim)
        self.day_of_week = nn.Embedding(7, dow_dim)
        self.adaptive = nn.Embedding(total_nodes * steps, adaptive_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.temporal = nn.ModuleList(
            [AxisAttentionBlock(width, heads, dropout, "time") for _ in range(layers_each_axis)]
        )
        self.spatial = nn.ModuleList(
            [AxisAttentionBlock(width, heads, dropout, "space") for _ in range(layers_each_axis)]
        )
        self.output_norm = nn.LayerNorm(width)
        self.readout = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        tod: torch.Tensor,
        dow: torch.Tensor,
        global_node_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch, steps, nodes, _ = x.shape
        if steps != self.steps or global_node_ids.numel() != nodes:
            raise ValueError(f"Expected {self.steps} steps and one ID per node")
        if global_node_ids.min() < 0 or global_node_ids.max() >= self.total_nodes:
            raise ValueError("Global node ID out of range")
        node_time_ids = global_node_ids[None, :] * self.steps + torch.arange(
            self.steps, device=x.device
        )[:, None]
        pieces = [
            self.feature_proj(x),
            self.time_of_day(tod)[:, :, None, :].expand(-1, -1, nodes, -1),
            self.day_of_week(dow)[:, :, None, :].expand(-1, -1, nodes, -1),
            self.adaptive(node_time_ids)[None].expand(batch, -1, -1, -1),
        ]
        state = self.input_dropout(torch.cat(pieces, dim=-1))
        for block in self.temporal:
            state = block(state)
        for block in self.spatial:
            state = block(state)
        return self.readout(self.output_norm(state))


def reconstruction_loss(
    residual: torch.Tensor,
    base: torch.Tensor,
    truth: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Weighted MSE on synthetically masked eligible cells only.

    Flow and speed are measured against the same 600/25 scales used by the
    published Task 1 score.  Source truth may contain NaN outside target cells.
    """
    scale = residual.new_tensor([25.0, 600.0])
    pred = base + residual * scale
    safe_truth = torch.nan_to_num(truth, nan=0.0)
    error = (pred - safe_truth) / scale
    per_cell = 0.54 * error[..., 0].square() + 0.46 * error[..., 1].square()
    selected = per_cell[target_mask]
    if selected.numel() == 0:
        raise ValueError("No supervised masked cells in batch")
    return selected.mean()
