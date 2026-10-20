"""Leak-safe tabular features for Task 1 residual reconstruction.

Every dynamic feature is derived from the supplied visible mask.  In training,
synthetic targets are removed from that mask before this module is called.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import polars as pl

from src.io import paths
from src.task1.staeformer_data import Day, PanelInfo, model_inputs


CAT_FEATURES = ["panel_id", "node_id"]
NUM_FEATURES = [
    "base_speed", "base_flow_lane", "prev_speed", "prev_flow_lane",
    "next_speed", "next_flow_lane", "prev_gap_steps", "next_gap_steps",
    "left_speed", "left_flow_lane", "right_speed", "right_flow_lane",
    "left_gap_links", "right_gap_links", "left_gap_km", "right_gap_km",
    "upstream_speed", "upstream_flow_lane", "downstream_speed",
    "downstream_flow_lane", "upstream_visible_count", "downstream_visible_count",
    "visible_2h", "time_slot", "weekday", "mask_probability", "lanes",
    "free_speed_kmh", "capacity_per_lane", "length_km", "position_fraction",
    "on_ramp_count", "off_ramp_count",
]
FEATURES = CAT_FEATURES + NUM_FEATURES


@dataclass(frozen=True)
class NetworkFeatures:
    panel_id: int
    node_id: np.ndarray
    lanes: np.ndarray
    free_speed: np.ndarray
    capacity_per_lane: np.ndarray
    length_km: np.ndarray
    position_fraction: np.ndarray
    position_km: np.ndarray
    on_ramp_count: np.ndarray
    off_ramp_count: np.ndarray
    upstream: tuple[tuple[int, ...], ...]
    downstream: tuple[tuple[int, ...], ...]


def _items(value: str | None) -> list[str]:
    return [] if value is None else [part for part in value.split(";") if part]


def network_features(panel: str, info: PanelInfo, panel_id: int) -> NetworkFeatures:
    links = pl.read_csv(paths.network_dir(panel) / "links.csv")
    topo = pl.read_csv(paths.network_dir(panel) / "lwr_mainline_topology.csv")
    ordered = pl.DataFrame({"link_id": info.link_ids}).join(
        links, on="link_id", how="left", validate="1:1"
    ).join(
        topo.select("link_id", "incoming_link_ids", "outgoing_link_ids",
                    "on_ramp_link_ids", "off_ramp_link_ids"),
        on="link_id", how="left", validate="1:1",
    )
    if ordered.height != info.n_links or ordered["lanes"].null_count():
        raise ValueError(f"{panel}: invalid network features")
    index = {link: j for j, link in enumerate(info.link_ids)}
    upstream = tuple(
        tuple(index[link] for link in _items(value) if link in index)
        for value in ordered["incoming_link_ids"]
    )
    downstream = tuple(
        tuple(index[link] for link in _items(value) if link in index)
        for value in ordered["outgoing_link_ids"]
    )
    length = ordered["length_km"].to_numpy().astype(np.float32)
    position = np.cumsum(length, dtype=np.float32)
    return NetworkFeatures(
        panel_id=panel_id,
        node_id=info.global_node_ids.astype(np.int32),
        lanes=info.lanes,
        free_speed=info.free_speeds,
        capacity_per_lane=(ordered["capacity_vph"].to_numpy().astype(np.float32) / info.lanes),
        length_km=length,
        position_fraction=np.arange(info.n_links, dtype=np.float32) / max(info.n_links - 1, 1),
        position_km=position,
        on_ramp_count=np.array([len(_items(x)) for x in ordered["on_ramp_link_ids"]], dtype=np.float32),
        off_ramp_count=np.array([len(_items(x)) for x in ordered["off_ramp_link_ids"]], dtype=np.float32),
        upstream=upstream,
        downstream=downstream,
    )


def _connected_values(
    source: np.ndarray, visible: np.ndarray, neighbors: tuple[tuple[int, ...], ...]
) -> tuple[np.ndarray, np.ndarray]:
    values = np.full(source.shape, np.nan, dtype=np.float32)
    counts = np.zeros(source.shape, dtype=np.float32)
    for node, adjacent in enumerate(neighbors):
        if not adjacent:
            continue
        seen = visible[:, adjacent]
        count = seen.sum(axis=1).astype(np.float32)
        amount = np.where(seen, source[:, adjacent], 0.0).sum(axis=1)
        values[:, node] = np.divide(
            amount, count, out=np.full(source.shape[0], np.nan, dtype=np.float32),
            where=count > 0,
        )
        counts[:, node] = count
    return values, counts


def _nearest_indices(visible: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    length = visible.shape[axis]
    grid = np.arange(length, dtype=np.int32)
    if axis == 0:
        grid = grid[:, None]
    else:
        grid = grid[None, :]
    before = np.maximum.accumulate(np.where(visible, grid, -1), axis=axis)
    after = np.minimum.accumulate(
        np.flip(np.where(visible, grid, length), axis=axis), axis=axis
    )
    return before, np.flip(after, axis=axis)


def _observed_value(source: np.ndarray, indices: np.ndarray, axis: int) -> np.ndarray:
    looked_up = np.take_along_axis(source, np.clip(indices, 0, source.shape[axis] - 1), axis=axis)
    return np.where((indices >= 0) & (indices < source.shape[axis]), looked_up, np.nan)


def features_for_targets(
    day: Day,
    visible: np.ndarray,
    target: np.ndarray,
    network: NetworkFeatures,
    mask_probability: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return feature rows and [speed, flow/lane] interpolation for target cells."""
    if np.any(target & visible):
        raise ValueError("A supervised target is visible in feature input")
    t, n = np.nonzero(target)
    if not len(t):
        raise ValueError(f"{day.panel}/{day.day}: no target cells")
    _, base_grid = model_inputs(day, visible)
    n_steps, n_nodes = visible.shape
    time_grid = np.arange(n_steps, dtype=np.int32)[:, None]
    node_grid = np.arange(n_nodes, dtype=np.int32)[None, :]
    prev, nxt = _nearest_indices(visible, axis=0)
    left, right = _nearest_indices(visible, axis=1)

    # A centered, two-hour observability count uses only the input mask.
    cumulative = np.pad(np.cumsum(visible.astype(np.int16), axis=0), ((1, 0), (0, 0)))
    lo = np.maximum(np.arange(n_steps) - 12, 0)
    hi = np.minimum(np.arange(n_steps) + 13, n_steps)
    visible_2h = cumulative[hi] - cumulative[lo]

    upstream_speed, upstream_count = _connected_values(day.speed, visible, network.upstream)
    upstream_flow, _ = _connected_values(day.flow_lane, visible, network.upstream)
    downstream_speed, downstream_count = _connected_values(day.speed, visible, network.downstream)
    downstream_flow, _ = _connected_values(day.flow_lane, visible, network.downstream)

    def chosen(array: np.ndarray) -> np.ndarray:
        return np.asarray(array[t, n], dtype=np.float32)

    prev_speed = _observed_value(day.speed, prev, axis=0)
    prev_flow = _observed_value(day.flow_lane, prev, axis=0)
    next_speed = _observed_value(day.speed, nxt, axis=0)
    next_flow = _observed_value(day.flow_lane, nxt, axis=0)
    left_speed = _observed_value(day.speed, left, axis=1)
    left_flow = _observed_value(day.flow_lane, left, axis=1)
    right_speed = _observed_value(day.speed, right, axis=1)
    right_flow = _observed_value(day.flow_lane, right, axis=1)
    pos = network.position_km
    left_km = np.where(left >= 0, pos[node_grid] - pos[np.clip(left, 0, n_nodes - 1)], np.nan)
    right_km = np.where(right < n_nodes, pos[np.clip(right, 0, n_nodes - 1)] - pos[node_grid], np.nan)
    weekday = date.fromisoformat(day.day).weekday()
    columns = {
        "panel_id": np.full(len(t), network.panel_id, dtype=np.int32),
        "node_id": network.node_id[n],
        "base_speed": base_grid[t, n, 0],
        "base_flow_lane": base_grid[t, n, 1],
        "prev_speed": chosen(prev_speed),
        "prev_flow_lane": chosen(prev_flow),
        "next_speed": chosen(next_speed),
        "next_flow_lane": chosen(next_flow),
        "prev_gap_steps": chosen(np.where(prev >= 0, time_grid - prev, n_steps)),
        "next_gap_steps": chosen(np.where(nxt < n_steps, nxt - time_grid, n_steps)),
        "left_speed": chosen(left_speed),
        "left_flow_lane": chosen(left_flow),
        "right_speed": chosen(right_speed),
        "right_flow_lane": chosen(right_flow),
        "left_gap_links": chosen(np.where(left >= 0, node_grid - left, n_nodes)),
        "right_gap_links": chosen(np.where(right < n_nodes, right - node_grid, n_nodes)),
        "left_gap_km": chosen(left_km),
        "right_gap_km": chosen(right_km),
        "upstream_speed": chosen(upstream_speed),
        "upstream_flow_lane": chosen(upstream_flow),
        "downstream_speed": chosen(downstream_speed),
        "downstream_flow_lane": chosen(downstream_flow),
        "upstream_visible_count": chosen(upstream_count),
        "downstream_visible_count": chosen(downstream_count),
        "visible_2h": chosen(visible_2h),
        "time_slot": t.astype(np.float32),
        "weekday": np.full(len(t), weekday, dtype=np.float32),
        "mask_probability": np.full(len(t), mask_probability, dtype=np.float32),
        "lanes": network.lanes[n],
        "free_speed_kmh": network.free_speed[n],
        "capacity_per_lane": network.capacity_per_lane[n],
        "length_km": network.length_km[n],
        "position_fraction": network.position_fraction[n],
        "on_ramp_count": network.on_ramp_count[n],
        "off_ramp_count": network.off_ramp_count[n],
    }
    frame = pd.DataFrame({name: columns[name] for name in FEATURES})
    if frame.shape != (len(t), len(FEATURES)):
        raise ValueError("Feature table shape mismatch")
    return frame, base_grid[t, n].astype(np.float32)
