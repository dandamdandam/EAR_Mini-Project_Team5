"""Validate the release and cache daily grids without filling missing values.

Run from Traffic: python -m src.preprocess.run
The cache contains public inputs and training truth in separate directories.
Random training masks and interpolation are deliberately made by the trainer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "8")

import numpy as np
import polars as pl

from src.io import paths
from src.task1.staeformer_data import panel_infos

CACHE_VERSION = 1


def timestamps(day: str) -> tuple[str, ...]:
    return tuple(f"{day}T{slot // 12:02}:{slot % 12 * 5:02}:00Z" for slot in range(288))


def cache_path(root: Path, panel: str, split: str, day: str, masked: bool) -> Path:
    return root / "mainline" / panel / split / ("masked" if masked else "raw") / f"{day}.npz"


def sha256(file: Path) -> str:
    with file.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_grid(file: Path, day: str, info) -> tuple[dict, dict]:
    raw = pl.read_parquet(file)
    node_map = pl.DataFrame({"link_id": info.link_ids, "node_index": range(info.n_links),
                             "expected_station": info.station_ids})
    frame = raw.join(node_map, on="link_id", how="left", validate="m:1").sort("timestamp", "node_index")
    expected = timestamps(day)
    if frame.height != 288 * info.n_links or frame["node_index"].null_count():
        raise ValueError(f"Invalid daily dimensions: {file}")
    if frame.select(pl.struct("timestamp", "link_id").n_unique()).item() != frame.height:
        raise ValueError(f"Duplicate daily keys: {file}")
    if tuple(frame["timestamp"].unique(maintain_order=True)) != expected:
        raise ValueError(f"Not an exact UTC five-minute day: {file}")
    if frame.filter(pl.col("station_id") != pl.col("expected_station")).height:
        raise ValueError(f"Station/link mismatch: {file}")
    shape = (288, info.n_links)
    speed = frame["speed_kmh"].to_numpy().reshape(shape).astype(np.float32)
    flow = frame["flow_vph"].to_numpy().reshape(shape).astype(np.float32)
    occupancy = frame["occupancy"].to_numpy().reshape(shape).astype(np.float32)
    quality = frame["pct_observed"].to_numpy().reshape(shape)
    eligible = frame["is_score_eligible"].to_numpy().reshape(shape).astype(bool)
    observed = np.isfinite(speed) & np.isfinite(flow)
    if not np.array_equal(np.isfinite(speed), np.isfinite(flow)):
        raise ValueError(f"Different speed/flow availability: {file}")
    if np.isinf(speed).any() or np.isinf(flow).any() or (speed[observed] < 0).any() or (flow[observed] < 0).any():
        raise ValueError(f"Invalid physical measurement: {file}")
    if not np.isfinite(quality).all() or ((quality < 0) | (quality > 100)).any():
        raise ValueError(f"Invalid observation quality: {file}")
    if np.any(eligible & (quality < 75)):
        raise ValueError(f"Inconsistent score eligibility: {file}")
    arrays = {"speed": speed, "flow_lane": flow / info.lanes[None],
              "occupancy": occupancy, "pct_observed": quality.astype(np.uint8),
              "eligible": eligible, "observed": observed,
              "is_imputed": frame["is_imputed"].to_numpy().reshape(shape).astype(bool),
              "global_node_ids": info.global_node_ids, "lanes": info.lanes,
              "source_size": np.int64(file.stat().st_size),
              "source_mtime_ns": np.int64(file.stat().st_mtime_ns),
              "cache_version": np.int64(CACHE_VERSION)}
    stats = {"rows": frame.height, "observed_cells": int(observed.sum()),
             "eligible_cells": int(eligible.sum()), "missing_cells": int((~observed).sum())}
    return arrays, stats


def target_mask(rows: pl.DataFrame | None, day: str, info) -> np.ndarray:
    mask = np.zeros((288, info.n_links), dtype=bool)
    if rows is None or rows.is_empty():
        return mask
    keys = {(link, station): i for i, (link, station) in enumerate(zip(info.link_ids, info.station_ids))}
    ts = {stamp: i for i, stamp in enumerate(timestamps(day))}
    for stamp, link, station in rows.select("timestamp", "link_id", "station_id").iter_rows():
        mask[ts[stamp], keys[(link, station)]] = True
    if mask.sum() != rows.height:
        raise ValueError(f"Duplicate official target: {info.panel}/{day}")
    return mask


def small_inputs(root: Path, info) -> dict:
    """Keep static assets and task-specific public tables in their own schemas."""
    panel = info.panel
    net = root / "network" / panel
    net.mkdir(parents=True, exist_ok=True)
    for file in paths.network_dir(panel).glob("*.csv"):
        pl.read_csv(file).write_parquet(net / (file.stem + ".parquet"))
    fd = pl.DataFrame({"link_id": info.link_ids}).join(
        pl.read_csv(paths.network_dir(panel) / "fd_parameters.csv"), on="link_id", validate="1:1")
    if not np.array_equal(fd["lanes"].to_numpy(), info.lanes) or (info.lanes <= 0).any():
        raise ValueError(f"FD lanes differ from link metadata: {panel}")
    (net / "node_mapping.json").write_text(json.dumps({
        "panel": panel, "link_ids": info.link_ids, "station_ids": info.station_ids,
        "global_node_ids": info.global_node_ids.tolist(), "lanes": info.lanes.tolist(),
        "free_speed_kmh": info.free_speeds.tolist()}, indent=2))
    task2_windows = 0
    for split in paths.SPLITS:
        if panel in paths.TASK2_PANELS:
            src = paths.task2_dir(panel, split)
            dst = root / "task2" / panel / split
            dst.mkdir(parents=True, exist_ok=True)
            history = pl.read_parquet(src / "window_history.parquet")
            if history.select(pl.struct("window_id", "timestamp", "link_id").n_unique()).item() != history.height:
                raise ValueError(f"Duplicate Task 2 history: {panel}/{split}")
            index = pl.read_csv(src / "window_index.csv")
            task2_windows += index.height
            history = history.join(fd.select("link_id", "lanes"), on="link_id", validate="m:1").with_columns(
                (pl.col("flow_vph") / pl.col("lanes")).alias("flow_per_lane"),
                (pl.col("speed_kmh").is_not_null() & pl.col("flow_vph").is_not_null()).alias("observed_mask"))
            bound = history.join(index.select("window_id", "history_start", "history_end", "forecast_origin"), on="window_id")
            if bound.filter((pl.col("timestamp") < pl.col("history_start")) |
                            (pl.col("timestamp") >= pl.col("history_end")) |
                            (pl.col("timestamp") >= pl.col("forecast_origin"))).height:
                raise ValueError(f"Task 2 history contains an out-of-window observation: {panel}/{split}")
            history.sort("window_id", "timestamp", "link_id").write_parquet(dst / "history.parquet")
            index.write_parquet(dst / "index.parquet")
            pl.read_csv(src / "sample_submission_queue.csv").write_parquet(dst / "targets.parquet")
        src = paths.task4_dir(panel, split)
        dst = root / "task4" / panel / split
        dst.mkdir(parents=True, exist_ok=True)
        for name in ("synthetic_link_counts", "synthetic_weak_prior", "sample_submission_path_flow"):
            pl.read_csv(src / (name + ".csv")).write_parquet(dst / (name + ".parquet"))
    return {"task2_windows": task2_windows}


def prepare(root: Path, *, smoke: bool = False) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    infos = panel_infos()
    records, ramp_records = [], []
    task2_windows = 0
    for panel, info in infos.items():
        task2_windows += small_inputs(root, info)["task2_windows"]
        for split in paths.SPLITS:
            target = (pl.read_csv(paths.task1_targets(panel, split), columns=["timestamp", "link_id", "station_id", "mask_regime"])
                      .with_columns(pl.col("timestamp").str.slice(0, 10).alias("date")))
            by_date = target.partition_by("date", as_dict=True)
            layers = [(True, paths.mainline_masked_dir(panel, split))]
            if split == "train":
                layers.insert(0, (False, paths.mainline_unmasked_dir(panel)))
            for masked, source in layers:
                files = sorted(source.rglob("*.parquet"), key=lambda p: p.name)
                if smoke:
                    files = files[:1]
                for file in files:
                    day = file.stem.removeprefix("synthetic_mainline_").replace("_", "-")
                    arrays, stats = read_grid(file, day, info)
                    official = target_mask(by_date.get((day,)), day, info) if masked else np.zeros((288, info.n_links), bool)
                    if (official & arrays["observed"]).any() or (official & ~arrays["eligible"]).any():
                        raise ValueError(f"Visible/ineligible official target: {file}")
                    arrays["official_target"] = official
                    dest = cache_path(root, panel, split, day, masked)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(dest, **arrays)
                    if split != "train":
                        role = "public_inference" if split == "validation" else "private_inference"
                    else:
                        role = "fit" if day < "2031-01-01" else ("selection" if day < "2031-02-01" else "local_holdout")
                    records.append({"panel": panel, "split": split, "date": day, "layer": "masked" if masked else "raw",
                                    "local_role": role, "mask_regime": file.parent.name.removeprefix("mask_regime=") if masked else "",
                                    "official_targets": int(official.sum()), "source_sha256": sha256(file), **stats})
                if masked and not smoke:
                    emitted = sum(r["official_targets"] for r in records if r["panel"] == panel and r["split"] == split and r["layer"] == "masked")
                    if emitted != target.height:
                        raise ValueError(f"Official target coverage {emitted}/{target.height}: {panel}/{split}")
            for file in sorted(paths.ramp_dir(panel, split).rglob("*.parquet"))[:1 if smoke else None]:
                raw = pl.read_parquet(file)
                if raw.select(pl.struct("timestamp", "ramp_link_id").n_unique()).item() != raw.height:
                    raise ValueError(f"Duplicate ramp key: {file}")
                raw = raw.with_columns(pl.col("flow_vph").is_not_null().alias("observed_mask"))
                invalid = raw.filter((pl.col("flow_vph") < 0) | pl.col("flow_vph").is_infinite() | pl.col("flow_vph").is_nan()).height
                if invalid:
                    raise ValueError(f"Invalid ramp flow: {file}")
                dst = root / "ramp" / panel / split / (file.stem + ".parquet")
                dst.parent.mkdir(parents=True, exist_ok=True)
                raw.sort("timestamp", "ramp_link_id").write_parquet(dst)
                ramp_records.append({"panel": panel, "split": split, "file": file.name,
                                     "rows": raw.height, "missing_flow": raw["flow_vph"].null_count()})
        print(f"preprocessed {panel}: {info.n_links} links, {len(records)} daily grids so far", flush=True)
    manifest = pl.DataFrame(records)
    manifest.write_parquet(root / "manifest.parquet")
    manifest.group_by("panel", "split", "layer").agg(pl.len().alias("days"), pl.col("rows", "observed_cells", "eligible_cells", "missing_cells", "official_targets").sum()).sort("panel", "split", "layer").write_csv(root / "quality_summary.csv")
    pl.DataFrame(ramp_records).write_parquet(root / "ramp_manifest.parquet")
    settings = {"cache_version": CACHE_VERSION, "smoke": smoke, "panels": list(infos),
                "total_nodes": sum(i.n_links for i in infos.values()), "daily_grids": len(records),
                "task2_windows": task2_windows, "manifest_sha256": sha256(root / "manifest.parquet"),
                "source_contract_sha256": sha256(paths.DATA_ROOT / "config/synthetic_release_v1.json"),
                "task1_scaling": {"speed_divisor": 120.0, "flow_per_lane_divisor": 2000.0,
                                  "type": "fixed physical scales, same as selected STAEformer experiment"},
                "raw_values_modified": False, "missing_values_filled": False,
                "seconds": round(time.perf_counter() - started, 2)}
    (root / "SUCCESS.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(json.dumps(settings), flush=True)
    return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=paths.CACHE_ROOT / "preprocessed_v1")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    prepare(args.out, smoke=args.smoke)


if __name__ == "__main__":
    main()
