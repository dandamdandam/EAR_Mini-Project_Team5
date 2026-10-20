"""ImputeFormer attention blocks adapted to two-channel Task 1 imputation.

Architecture adapted from Tong Nie et al., *ImputeFormer: Low Rankness-Induced
Transformers for Generalizable Spatiotemporal Imputation* (KDD 2024),
https://github.com/tongnie/ImputeFormer, commit
1c3868cf499951ac43b4fc7280593fbb745f58c7. The authors' code is MIT
licensed; see IMPUTEFORMER_LICENSE next to this file.

The original model emits a direct imputation. Here its projected temporal
attention and embedded spatial attention are retained, while the readout emits
a correction to the released-data interpolation for Task 1. The model accepts
variable panel sizes through one global bank of position-by-node embeddings.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from src.task1.staeformer_data import (
    FLOW_LANE_SCALE,
    RESIDUAL_FLOW_LANE_SCALE,
    RESIDUAL_SPEED_SCALE,
    SPEED_SCALE,
)


class ProjectedTemporalBlock(nn.Module):
    """Cross-attend from K learned tokens to T steps and back, per node."""

    def __init__(self, width: int, heads: int, projection_tokens: int, dropout: float):
        super().__init__()
        self.projector = nn.Parameter(torch.randn(projection_tokens, width))
        self.compress = nn.MultiheadAttention(width, heads, batch_first=True)
        self.expand = nn.MultiheadAttention(width, heads, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, width)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        batch, nodes, steps, width = state.shape
        sequence = state.reshape(batch * nodes, steps, width)
        queries = self.projector.unsqueeze(0).expand(batch * nodes, -1, -1)
        compressed, _ = self.compress(queries, sequence, sequence, need_weights=False)
        returned, _ = self.expand(sequence, queries, compressed, need_weights=False)
        sequence = self.norm1(sequence + self.dropout(returned))
        sequence = self.norm2(sequence + self.dropout(self.feedforward(sequence)))
        return sequence.reshape(batch, nodes, steps, width)


class EmbeddedSpatialBlock(nn.Module):
    """Factorized spatial attention queried by learned time-by-node positions."""

    def __init__(self, width: int, embedding_dim: int, feed_forward_dim: int):
        super().__init__()
        self.query_key = nn.Linear(embedding_dim, width)
        self.value = nn.Linear(width, width)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, feed_forward_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, width),
        )

    def forward(self, state: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # state [B,N,T,D], positions [T,N,A]. Softmax over the latent axis
        # for queries and over nodes for keys follows the authors' factorization.
        sequence = state.transpose(1, 2)
        projected = self.query_key(positions)
        query = torch.softmax(projected, dim=-1)
        key = torch.softmax(projected.transpose(-1, -2), dim=-1)
        compressed = torch.matmul(key.unsqueeze(0), self.value(sequence))
        update = torch.matmul(query.unsqueeze(0), compressed)
        sequence = self.norm1(sequence + update)
        sequence = self.norm2(sequence + self.feedforward(sequence))
        return sequence.transpose(1, 2)


class ImputeFormerTask1(nn.Module):
    """One shared ImputeFormer encoder across all corridors; output residual units.

    Input features are [visible speed, visible flow/lane, visible mask,
    interpolated speed, interpolated flow/lane], each numeric channel scaled by
    the Task 1 data helper. Clock sin/cos terms use the original model's exogenous
    input interface. Output [B,T,N,2] is in 25 km/h and 600 veh/h/lane units.
    """

    def __init__(
        self,
        *,
        total_nodes: int,
        steps: int = 24,
        input_dim: int = 5,
        feature_dim: int = 128,
        adaptive_dim: int = 32,
        heads: int = 4,
        layers: int = 3,
        projection_tokens: int = 8,
        feed_forward_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        width = feature_dim + adaptive_dim
        if width % heads:
            raise ValueError("Feature width must be divisible by attention heads")
        if total_nodes < 1 or steps < 1 or projection_tokens < 1:
            raise ValueError("Node, step and projector counts must be positive")
        self.total_nodes = total_nodes
        self.steps = steps
        self.input_dim = input_dim
        self.input_proj = nn.Linear(input_dim + 4, feature_dim)
        self.adaptive = nn.Parameter(torch.empty(steps, total_nodes, adaptive_dim))
        nn.init.xavier_uniform_(self.adaptive)
        self.temporal = nn.ModuleList(
            ProjectedTemporalBlock(width, heads, projection_tokens, dropout)
            for _ in range(layers)
        )
        self.spatial = nn.ModuleList(
            EmbeddedSpatialBlock(width, adaptive_dim, feed_forward_dim)
            for _ in range(layers)
        )
        self.readout = nn.Sequential(
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 2),
        )
        # Starting from zero means the first prediction equals interpolation.
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        tod: torch.Tensor,
        dow: torch.Tensor,
        global_node_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch, steps, nodes, channels = x.shape
        if steps != self.steps or channels != self.input_dim:
            raise ValueError("Input time or feature dimension differs from model")
        if tod.shape != (batch, steps) or dow.shape != (batch, steps):
            raise ValueError("Expected clock arrays of shape [batch, steps]")
        if global_node_ids.ndim != 1 or global_node_ids.numel() != nodes:
            raise ValueError("Expected one global node ID per panel link")
        if global_node_ids.min() < 0 or global_node_ids.max() >= self.total_nodes:
            raise ValueError("Global node ID out of range")

        time_phase = (2.0 * math.pi / 288.0) * tod.float()
        week_phase = (2.0 * math.pi / 7.0) * dow.float()
        clock = torch.stack(
            [time_phase.sin(), time_phase.cos(), week_phase.sin(), week_phase.cos()],
            dim=-1,
        )
        model_input = torch.cat(
            [x, clock[:, :, None, :].expand(-1, -1, nodes, -1)], dim=-1
        )
        positions = self.adaptive[:, global_node_ids, :]
        state = self.input_proj(model_input)
        state = torch.cat(
            [state, positions.unsqueeze(0).expand(batch, -1, -1, -1)], dim=-1
        ).transpose(1, 2)
        for temporal, spatial in zip(self.temporal, self.spatial):
            state = temporal(state)
            state = spatial(state, positions)
        return self.readout(state.transpose(1, 2))


def imputeformer_objective(
    residual: torch.Tensor,
    base: torch.Tensor,
    truth: torch.Tensor,
    target_mask: torch.Tensor,
    features: torch.Tensor,
    fourier_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Task 1 weighted masked MSE plus ImputeFormer Fourier regularization.

    The paper uses masked MAE and a Fourier loss. We use Task 1's normalized
    squared error for the reconstruction term because its score is based on
    RMSE. Fourier loss is the mean magnitude of the orthonormal 2D FFT across
    time and topology-ordered nodes of the completed normalized input. The
    coefficient defaults to the authors' 0.01 setting, and is configurable.
    """
    if fourier_weight < 0:
        raise ValueError("Fourier loss weight cannot be negative")
    scale = residual.new_tensor([RESIDUAL_SPEED_SCALE, RESIDUAL_FLOW_LANE_SCALE])
    pred = base + residual * scale
    difference = (pred - torch.nan_to_num(truth, nan=0.0)) / scale
    weighted_error = 0.54 * difference[..., 0].square() + 0.46 * difference[..., 1].square()
    selected = weighted_error[target_mask]
    if selected.numel() == 0:
        raise ValueError("No artificially hidden target in batch")
    reconstruction = selected.mean()

    if fourier_weight:
        data_scale = residual.new_tensor([SPEED_SCALE, FLOW_LANE_SCALE])
        pred_normalized = pred / data_scale
        observed = features[..., :2]
        visible = features[..., 2:3] > 0.5
        completed = torch.where(visible, observed, pred_normalized).float()
        spectrum = torch.fft.rfftn(completed, dim=(1, 2), norm="ortho")
        fourier = spectrum.abs().mean()
    else:
        fourier = reconstruction.new_zeros(())
    return reconstruction + fourier_weight * fourier, reconstruction, fourier
