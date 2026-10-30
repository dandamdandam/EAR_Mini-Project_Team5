"""Daily TrafficFlowBench tensors for masked state reconstruction.

Only the caller's training path reads unmasked observations as features.  Official
evaluation loads the masked release for features and uses the unmasked layer in
the separate scoring function.  Every array has shape [288, N, C].
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import os
from pathlib import Path

import numpy as np
import polars as pl

from src.io import paths

STEPS_PER_DAY = 288
SPEED_SCALE = 120.0
FLOW_LANE_SCALE = 2000.0
RESIDUAL_SPEED_SCALE = 25.0
RESIDUAL_FLOW_LANE_SCALE = 600.0
REGIME_PROBABILITIES = (0.20, 0.30, 0.50)


@dataclass(frozen=True)
class PanelInfo:
    panel: str
    link_ids: tuple[str, ...]
    station_ids: tuple[str, ...]
    lanes: np.ndarray
    free_speeds: np.ndarray
    global_node_ids: np.ndarray

    @property
    def n_links(self) -> int:
        return len(self.link_ids)


@dataclass
class Day:
    panel: str
    day: str
    speed: np.ndarray  # [T,N], km/h, nan for absent measurements
    flow_lane: np.ndarray  # [T,N], veh/h/lane
    eligible: np.ndarray  # [T,N], original quality flag
    timestamps: tuple[str, ...]


def panel_infos() -> dict[str, PanelInfo]:
    """Map every released link to a unique embedding ID across all ten panels."""
    infos: dict[str, PanelInfo] = {}
    offset = 0
    for panel in paths.PANELS:
        root = paths.network_dir(panel)
        links = pl.read_csv(root / "links.csv")
        order = pl.read_csv(root / "lwr_mainline_topology.csv").select(
            "link_id", "order_index"
        )
        links = links.join(order, on="link_id", how="left").sort("order_index", "link_id")
        if links["order_index"].null_count() or links["link_id"].n_unique() != links.height:
            raise ValueError(f"{panel}: invalid topology-to-link mapping")

        first = sorted(paths.mainline_unmasked_dir(panel).rglob("*.parquet"))[0]
        stations = (
            pl.read_parquet(first, columns=["link_id", "station_id"])
            .unique()
            .select("link_id", "station_id")
        )
        if stations["link_id"].n_unique() != stations.height:
            raise ValueError(f"{panel}: more than one station per link")
        links = links.join(stations, on="link_id", how="left", validate="1:1")
        if links["station_id"].null_count():
            raise ValueError(f"{panel}: a network link has no released station")
        n = links.height
        infos[panel] = PanelInfo(
            panel=panel,
            link_ids=tuple(links["link_id"].to_list()),
            station_ids=tuple(links["station_id"].to_list()),
            lanes=links["lanes"].to_numpy().astype(np.float32),
            free_speeds=links["free_speed_kmh"].to_numpy().astype(np.float32),
            global_node_ids=np.arange(offset, offset + n, dtype=np.int64),
        )
        offset += n
    return infos


def all_train_dates(panel: str, start: str, end: str) -> list[str]:
    root = paths.mainline_unmasked_dir(panel)
    return sorted(
        day
        for file in root.rglob("synthetic_mainline_*.parquet")
        if start <= (day := file.stem.removeprefix("synthetic_mainline_").replace("_", "-")) <= end
    )


def _mainline_file(panel: str, split: str, day: str, masked: bool) -> Path:
    stem = f"synthetic_mainline_{day.replace('-', '_')}.parquet"
    if not masked:
        if split != "train":
            raise ValueError("Unmasked observations exist for train only")
        file = paths.mainline_unmasked_dir(panel) / f"year_month={day[:7]}" / stem
    else:
        candidates = list(paths.mainline_masked_dir(panel, split).glob(f"mask_regime=*/{stem}"))
        if len(candidates) != 1:
            raise FileNotFoundError(f"Expected one masked day: {panel}/{split}/{day}, got {candidates}")
        file = candidates[0]
    if not file.exists():
        raise FileNotFoundError(file)
    return file


def load_day(panel: str, split: str, day: str, info: PanelInfo, *, masked: bool) -> Day:
    file = _mainline_file(panel, split, day, masked)
    cache_root = os.environ.get("TFB_DAILY_CACHE")
    if cache_root:
        cache = Path(cache_root) / "mainline" / panel / split / ("masked" if masked else "raw") / f"{day}.npz"
        with np.load(cache, allow_pickle=False) as data:
            if (int(data["cache_version"]) != 1 or int(data["source_size"]) != file.stat().st_size
                    or int(data["source_mtime_ns"]) != file.stat().st_mtime_ns
                    or not np.array_equal(data["global_node_ids"], info.global_node_ids)
                    or not np.array_equal(data["lanes"], info.lanes)):
                raise ValueError(f"Stale or incompatible daily cache: {cache}")
            speed, flow, eligible = data["speed"], data["flow_lane"], data["eligible"]
        stamps = tuple(f"{day}T{slot // 12:02}:{slot % 12 * 5:02}:00Z" for slot in range(STEPS_PER_DAY))
        if speed.shape != (STEPS_PER_DAY, info.n_links) or flow.shape != speed.shape or eligible.shape != speed.shape:
            raise ValueError(f"Invalid daily cache shape: {cache}")
        return Day(panel, day, speed, flow, eligible, stamps)
    raw = pl.read_parquet(
        file,
        columns=[
            "timestamp", "station_id", "link_id", "speed_kmh", "flow_vph", "is_score_eligible"
        ],
    )
    index = pl.DataFrame(
        {
            "link_id": info.link_ids,
            "station_id_expected": info.station_ids,
            "node_index": np.arange(info.n_links),
        }
    )
    frame = raw.join(index, on="link_id", how="left").sort("timestamp", "node_index")
    if frame.height != STEPS_PER_DAY * info.n_links:
        raise ValueError(f"{file}: incomplete 288 x {info.n_links} grid")
    if frame["node_index"].null_count() or frame.filter(
        pl.col("station_id") != pl.col("station_id_expected")
    ).height:
        raise ValueError(f"{file}: station/link mapping changed")
    timestamps = tuple(frame["timestamp"].unique(maintain_order=True).to_list())
    if len(timestamps) != STEPS_PER_DAY:
        raise ValueError(f"{file}: expected 288 timestamps, got {len(timestamps)}")
    eligible = frame["is_score_eligible"].to_numpy().reshape(STEPS_PER_DAY, -1) == 1
    speed = frame["speed_kmh"].to_numpy().reshape(STEPS_PER_DAY, -1).astype(np.float32)
    flow = frame["flow_vph"].to_numpy().reshape(STEPS_PER_DAY, -1).astype(np.float32)
    flow /= info.lanes[None, :]
    if np.any(np.isfinite(speed) != np.isfinite(flow)):
        raise ValueError(f"{file}: speed and flow missingness differ")
    return Day(panel, day, speed, flow, eligible, timestamps)


def synthetic_target_mask(day: Day, probability: float, rng: np.random.Generator) -> np.ndarray:
    if not 0 < probability < 1:
        raise ValueError("Mask probability must lie in (0, 1)")
    targetable = day.eligible & np.isfinite(day.speed) & np.isfinite(day.flow_lane)
    return targetable & (rng.random(targetable.shape) < probability)


def interpolate_visible(speed: np.ndarray, flow_lane: np.ndarray, visible: np.ndarray) -> np.ndarray:
    """Interpolate after masking; return speed and flow per lane in physical units.

    The time interpolation sees the complete day's *visible* observations,
    which is allowed for offline Task 1.  Spatial interpolation uses topology
    order, and supplies a 15% blend where both estimates are available.
    """
    t_count, n = visible.shape
    if speed.shape != visible.shape or flow_lane.shape != visible.shape:
        raise ValueError("Expected equal [time, link] arrays")
    result = np.empty((t_count, n, 2), dtype=np.float32)
    x_time = np.arange(t_count)
    x_nodes = np.arange(n)
    for c, source in enumerate((speed, flow_lane)):
        temporal = np.full_like(source, np.nan, dtype=np.float32)
        spatial = np.full_like(source, np.nan, dtype=np.float32)
        for j in range(n):
            valid = visible[:, j] & np.isfinite(source[:, j])
            if valid.any():
                temporal[:, j] = np.interp(x_time, x_time[valid], source[valid, j])
        for t in range(t_count):
            valid = visible[t] & np.isfinite(source[t])
            if valid.any():
                spatial[t] = np.interp(x_nodes, x_nodes[valid], source[t, valid])
        mixed = np.where(
            np.isfinite(temporal) & np.isfinite(spatial),
            0.85 * temporal + 0.15 * spatial,
            np.where(np.isfinite(temporal), temporal, spatial),
        )
        fallback = np.nanmedian(source[visible & np.isfinite(source)])
        if not np.isfinite(fallback):
            raise ValueError("No visible observation to interpolate")
        mixed = np.where(np.isfinite(mixed), mixed, fallback)
        mixed[visible] = source[visible]
        result[..., c] = mixed
    result[..., 0] = np.clip(result[..., 0], 5.0, 135.0)
    result[..., 1] = np.maximum(result[..., 1], 0.0)
    return result


def model_inputs(day: Day, visible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Five input channels and 2-channel interpolation from public inputs."""
    base = interpolate_visible(day.speed, day.flow_lane, visible)
    observed = np.stack(
        [np.where(visible, day.speed, 0.0), np.where(visible, day.flow_lane, 0.0)], axis=-1
    )
    features = np.empty((*visible.shape, 5), dtype=np.float32)
    features[..., 0] = observed[..., 0] / SPEED_SCALE
    features[..., 1] = observed[..., 1] / FLOW_LANE_SCALE
    features[..., 2] = visible.astype(np.float32)
    features[..., 3] = base[..., 0] / SPEED_SCALE
    features[..., 4] = base[..., 1] / FLOW_LANE_SCALE
    if np.any(~np.isfinite(features)) or np.any(~np.isfinite(base)):
        raise ValueError("Nonfinite model input or interpolation")
    return features, base


def model_arrays(day: Day, visible: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Training input, base, and released training truth for synthetic masks."""
    if np.any(target & visible) or np.any(target & ~day.eligible):
        raise ValueError("Target must be eligible and hidden in the input")
    features, base = model_inputs(day, visible)
    truth = np.stack([day.speed, day.flow_lane], axis=-1)
    if np.any(~np.isfinite(truth[target])):
        raise ValueError("Target truth is nonfinite")
    return features, base, truth


def day_time_features(day: str) -> tuple[np.ndarray, np.ndarray]:
    """5-minute slot and synthetic-calendar weekday for the 288 daily steps."""
    weekday = date.fromisoformat(day).weekday()
    return np.arange(STEPS_PER_DAY, dtype=np.int64), np.full(STEPS_PER_DAY, weekday, dtype=np.int64)


def official_targets(panel: str, start: str, end: str) -> pl.DataFrame:
    """Authoritative train Task 1 coordinates, restricted to a held-out date span."""
    if not ("2030-06-01" <= start <= end <= "2031-02-28"):
        raise ValueError("Official local targets must be within train")
    return (
        pl.scan_csv(paths.task1_targets(panel, "train"))
        .filter((pl.col("timestamp").str.slice(0, 10) >= start) &
                (pl.col("timestamp").str.slice(0, 10) <= end))
        .select("timestamp", "station_id", "link_id", "mask_regime")
        .collect(engine="streaming")
    )


def targets_for_day(targets: pl.DataFrame, day: Day, info: PanelInfo) -> tuple[np.ndarray, str]:
    rows = targets.filter(pl.col("timestamp").str.starts_with(day.day))
    if rows.is_empty():
        raise ValueError(f"{info.panel}/{day.day}: official targets missing")
    regimes = rows["mask_regime"].unique().to_list()
    if len(regimes) != 1:
        raise ValueError(f"{info.panel}/{day.day}: multiple regimes")
    key_to_node = {(link, station): j for j, (link, station) in enumerate(zip(info.link_ids, info.station_ids))}
    time_to_index = {stamp: i for i, stamp in enumerate(day.timestamps)}
    mask = np.zeros((STEPS_PER_DAY, info.n_links), dtype=bool)
    for stamp, station, link in rows.select("timestamp", "station_id", "link_id").iter_rows():
        mask[time_to_index[stamp], key_to_node[(link, station)]] = True
    if mask.sum() != rows.height:
        raise ValueError(f"{info.panel}/{day.day}: duplicate or unmatched official targets")
    return mask, regimes[0]
