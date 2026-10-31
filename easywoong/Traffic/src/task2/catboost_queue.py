"""Causal 60-minute features -> six direct CatBoost queue forecasts.

Local labels are proxies from eligible unmasked TRAIN observations, not the
organizer's latent queue labels. January selects tree count/threshold, February
is reported once as a proxy holdout, then the model is refit on all nine months.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import time

os.environ.setdefault("POLARS_MAX_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier

from src.io import paths
from src.task1.staeformer_data import panel_infos

CAT_FEATURES = ["panel_id", "node_id"]


def metadata(panel, info, cache):
    fd = pl.DataFrame({"link_id": info.link_ids}).join(
        pl.read_parquet(cache / "network" / panel / "fd_parameters.parquet"),
        on="link_id", validate="1:1")
    return {"panel_id": paths.PANELS.index(panel), "node_id": info.global_node_ids,
            "lanes": info.lanes, "free_speed": info.free_speeds,
            "capacity_lane": fd["capacity_vph"].to_numpy() / info.lanes,
            "length": fd["length_km"].to_numpy(), "forward": panel.endswith(("N", "E"))}


def temporal_stats(values):
    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    safe = np.where(valid, values, 0.0)
    mean = np.divide(safe.sum(axis=0), count, out=np.full(values.shape[1], np.nan), where=count > 0)
    last_index = np.where(valid, np.arange(len(values))[:, None], -1).max(axis=0)
    last = np.where(last_index >= 0, values[np.maximum(last_index, 0), np.arange(values.shape[1])], np.nan)
    var = np.divide(np.where(valid, (values - mean) ** 2, 0.0).sum(axis=0), count,
                    out=np.full(values.shape[1], np.nan), where=count > 0)
    t = np.arange(len(values), dtype=float)[:, None]
    sum_t, sum_t2 = np.where(valid, t, 0).sum(axis=0), np.where(valid, t * t, 0).sum(axis=0)
    denom = count * sum_t2 - sum_t ** 2
    slope = np.divide(count * (safe * t).sum(axis=0) - sum_t * safe.sum(axis=0), denom,
                      out=np.full(values.shape[1], np.nan), where=denom > 0)
    lo = np.where(valid, values, np.inf).min(axis=0)
    lo[~(count > 0)] = np.nan
    return {"last": last, "mean": mean, "min": lo, "std": np.sqrt(var),
            "slope": slope, "count": count.astype(float), "age": len(values) - last_index}


def features(history, net, origin_slot, weekday):
    """Only the passed 12 history steps are accessible here. Horizon-major rows."""
    speed, flow, occ, quality = (np.asarray(history[k]) for k in ("speed", "flow_lane", "occupancy", "pct_observed"))
    if speed.shape[0] != 12 or any(a.shape != speed.shape for a in (flow, occ, quality)):
        raise ValueError("Queue features require exactly 12 aligned history steps")
    n = speed.shape[1]
    v = speed / net["free_speed"][None]
    q = flow / net["capacity_lane"][None]
    vs, qs, os_ = temporal_stats(v), temporal_stats(q), temporal_stats(occ)
    data = {"panel_id": np.full(n, net["panel_id"]), "node_id": net["node_id"]}
    for prefix, values in (("v", v), ("q", q), ("occ", occ)):
        for lag in (1, 2, 3, 6, 12):
            data[f"{prefix}_lag{lag}"] = values[-lag]
    for prefix, stats in (("v", vs), ("q", qs), ("occ", os_)):
        for key in ("last", "mean", "min", "std", "slope"):
            data[f"{prefix}_{key}"] = stats[key]
    for span in (3, 6):
        recent = temporal_stats(v[-span:])
        data[f"v_mean{span}"] = recent["mean"]
        data[f"v_slope{span}"] = recent["slope"]
    data.update({"observed_fraction": vs["count"] / 12, "last_age": vs["age"],
                 "quality_mean": quality.mean(axis=0) / 100.0,
                 "lanes": net["lanes"], "free_speed": net["free_speed"],
                 "capacity_lane": net["capacity_lane"], "length": net["length"],
                 "position": np.arange(n) / max(n - 1, 1)})
    for label, offset in (("up1", -1), ("down1", 1), ("up2", -2), ("down2", 2)):
        if not net["forward"]:
            offset = -offset
        j = np.arange(n) + offset
        good = (j >= 0) & (j < n)
        data[f"v_{label}"] = np.where(good, vs["last"][np.clip(j, 0, n - 1)], np.nan)
        data[f"slope_{label}"] = np.where(good, vs["slope"][np.clip(j, 0, n - 1)], np.nan)
    any_history_queue = bool(np.any(np.isfinite(v) & (v <= 0.6)))
    data["history_has_queue"] = np.full(n, float(any_history_queue))
    last_valid = np.isfinite(vs["last"])
    data["panel_last_queue_fraction"] = np.full(n, np.mean(vs["last"][last_valid] <= 0.6) if last_valid.any() else 0.0)
    data["panel_min_last_speed"] = np.full(n, np.min(vs["last"][last_valid]) if last_valid.any() else np.nan)
    for name, value in (("time_sin", np.sin(2 * np.pi * origin_slot / 288)),
                         ("time_cos", np.cos(2 * np.pi * origin_slot / 288)),
                         ("weekday_sin", np.sin(2 * np.pi * weekday / 7)),
                         ("weekday_cos", np.cos(2 * np.pi * weekday / 7))):
        data[name] = np.full(n, value)
    names = list(data) + ["horizon_steps"]
    base = np.column_stack(list(data.values())).astype(np.float32)
    result = np.column_stack([np.tile(base, (6, 1)), np.repeat(np.arange(1, 7), n)]).astype(np.float32)
    if np.isinf(result).any():
        raise ValueError("Infinite Task 2 feature")
    return result, names


def as_frame(matrix, names):
    frame = pd.DataFrame(matrix, columns=names)
    for name in CAT_FEATURES:
        frame[name] = frame[name].astype(np.int32)
    return frame


def build_samples(cache, infos, start, end, seed=42):
    rng = np.random.default_rng(seed)
    chunks, labels, groups, persisted, windows = [], [], [], [], []
    for panel in paths.TASK2_PANELS:
        info = infos[panel]
        net = metadata(panel, info, cache)
        for file in sorted((cache / "mainline" / panel / "train/raw").glob("*.npz")):
            if not start <= file.stem <= end:
                continue
            with np.load(file, allow_pickle=False) as saved:
                raw = {k: saved[k] for k in ("speed", "flow_lane", "occupancy", "pct_observed", "eligible")}
            observed = np.isfinite(raw["speed"]) & raw["eligible"]
            queue = observed & (raw["speed"] <= 0.6 * info.free_speeds[None])
            candidates = {0: [], 1: []}
            for origin in range(12, 282, 3):
                hist = slice(origin - 12, origin)
                future = slice(origin + 1, origin + 7)
                if raw["eligible"][hist].mean() < 0.70 or not queue[future].any():
                    continue
                condition = int(np.any(np.isfinite(raw["speed"][hist]) &
                                       (raw["speed"][hist] <= 0.6 * info.free_speeds[None])))
                candidates[condition].append(origin)
            for condition, origins in candidates.items():
                if not origins:
                    continue
                origin = int(rng.choice(origins))
                history = {k: v[origin - 12:origin] for k, v in raw.items() if k != "eligible"}
                x, names = features(history, net, origin, datetime.fromisoformat(file.stem).weekday())
                future = slice(origin + 1, origin + 7)
                valid = observed[future].ravel()
                y = queue[future].ravel()[valid].astype(np.uint8)
                window_id = len(windows)
                last = temporal_stats(history["speed"])["last"] <= 0.6 * info.free_speeds
                chunks.append(x[valid])
                labels.append(y)
                groups.append(np.full(len(y), window_id, dtype=np.int32))
                persisted.append(np.tile(last, 6)[valid])
                windows.append({"panel": panel, "date": file.stem, "origin_slot": origin,
                                "condition": "queue_ongoing" if condition else "queue_onset",
                                "rows": len(y), "positives": int(y.sum())})
        print(f"queue samples {start}..{end}: {panel}, {len(windows)} windows so far", flush=True)
    if not chunks:
        raise ValueError("No train-derived queue windows")
    return {"x": np.concatenate(chunks), "y": np.concatenate(labels), "groups": np.concatenate(groups),
            "persistence": np.concatenate(persisted), "windows": windows, "names": names}


def proxy_score(prediction, sample):
    truth, groups = sample["y"].astype(bool), sample["groups"]
    n = len(sample["windows"])
    inter = np.bincount(groups, weights=prediction & truth, minlength=n)
    union = np.bincount(groups, weights=prediction | truth, minlength=n)
    iou = np.divide(inter, union, out=np.ones(n), where=union > 0)
    records = [{**w, "iou": float(v)} for w, v in zip(sample["windows"], iou)]
    frame = pl.DataFrame(records)
    conditions = frame.group_by("panel", "condition").agg(pl.col("iou").mean())
    panels = conditions.group_by("panel").agg(pl.col("iou").mean())
    result = {"proxy_iou": float(panels["iou"].mean()), "windows": n,
              "known_label_rows": len(truth), "positive_fraction": float(truth.mean()),
              "by_panel_condition": conditions.sort("panel", "condition").to_dicts()}
    return result


def fit_and_predict(out, cache, *, iterations=400):
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    infos = panel_infos()
    train = build_samples(cache, infos, "2030-06-01", "2030-12-31")
    selection = build_samples(cache, infos, "2031-01-01", "2031-01-31", seed=43)
    params = {"iterations": iterations, "depth": 7, "learning_rate": 0.08,
              "loss_function": "Logloss", "random_seed": 42, "thread_count": 8,
              "allow_writing_files": False, "l2_leaf_reg": 5.0}
    model = CatBoostClassifier(**params)
    model.fit(as_frame(train["x"], train["names"]), train["y"], cat_features=CAT_FEATURES,
              eval_set=(as_frame(selection["x"], selection["names"]), selection["y"]),
              early_stopping_rounds=50, verbose=100)
    model.save_model(str(out / "selection.cbm"))
    probability = model.predict_proba(as_frame(selection["x"], selection["names"]))[:, 1]
    thresholds = np.arange(0.15, 0.76, 0.05)
    scores = {f"{threshold:.2f}": proxy_score(probability >= threshold, selection) for threshold in thresholds}
    winner = max(scores, key=lambda key: scores[key]["proxy_iou"])
    threshold = float(winner)
    chosen_iterations = int(model.tree_count_)
    pl.DataFrame(train["windows"]).write_parquet(out / "fit_windows.parquet")
    pl.DataFrame(selection["windows"]).write_parquet(out / "selection_windows.parquet")
    report = {"label_type": "Proxy: eligible noisy train speed <= 0.60 * free_speed; NOT official queue labels",
              "fit_dates": ["2030-06-01", "2030-12-31"], "selection_dates": ["2031-01-01", "2031-01-31"],
              "fit_rows": len(train["y"]), "fit_windows": len(train["windows"]),
              "selected_iterations": chosen_iterations, "selected_threshold": threshold,
              "threshold_scores": {key: result["proxy_iou"] for key, result in scores.items()},
              "selection": scores[winner], "selection_persistence": proxy_score(selection["persistence"], selection),
              "feature_names": train["names"], "params": params}
    (out / "selection_result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"queue January proxy IoU={scores[winner]['proxy_iou']:.6f}, threshold={threshold}, trees={chosen_iterations}", flush=True)
    del train, selection
    holdout = build_samples(cache, infos, "2031-02-01", "2031-02-28", seed=44)
    probability = model.predict_proba(as_frame(holdout["x"], holdout["names"]))[:, 1]
    test = {"label_type": report["label_type"], "model": proxy_score(probability >= threshold, holdout),
            "persistence": proxy_score(holdout["persistence"], holdout), "threshold": threshold,
            "independent_of_final_refit": True}
    (out / "proxy_holdout.json").write_text(json.dumps(test, indent=2) + "\n")
    del holdout, model
    full = build_samples(cache, infos, "2030-06-01", "2031-02-28")
    final = CatBoostClassifier(**{**params, "iterations": chosen_iterations})
    final.fit(as_frame(full["x"], full["names"]), full["y"], cat_features=CAT_FEATURES, verbose=100)
    final.save_model(str(out / "final.cbm"))
    config = {"model": "CatBoostClassifier direct six-horizon queue prediction", "train_dates": ["2030-06-01", "2031-02-28"],
              "history_steps": 12, "history_includes_origin": False, "horizon_steps": list(range(1, 7)),
              "threshold": threshold, "params": {**params, "iterations": chosen_iterations},
              "feature_names": full["names"], "cat_features": CAT_FEATURES,
              "train_rows": len(full["y"]), "train_windows": len(full["windows"]),
              "label_type": report["label_type"], "seconds_through_refit": round(time.perf_counter() - started, 2)}
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    del full
    predict(out, cache, model=final)


def predict(out, cache, model=None):
    config = json.loads((out / "config.json").read_text())
    if model is None:
        model = CatBoostClassifier()
        model.load_model(str(out / "final.cbm"))
    infos = panel_infos()
    for split in ("validation", "private"):
        parts = []
        for panel in paths.TASK2_PANELS:
            info = infos[panel]
            root = cache / "task2" / panel / split
            index, history, targets = (pl.read_parquet(root / f"{name}.parquet") for name in ("index", "history", "targets"))
            node = pl.DataFrame({"link_id": info.link_ids, "node": range(info.n_links)})
            net = metadata(panel, info, cache)
            for record in index.iter_rows(named=True):
                window = record["window_id"]
                origin = datetime.fromisoformat(record["forecast_origin"].replace("Z", "+00:00"))
                h = history.filter(pl.col("window_id") == window).join(node, on="link_id", how="left", validate="m:1").sort("timestamp", "node")
                expected = [(origin - timedelta(minutes=5 * lag)).strftime("%Y-%m-%dT%H:%M:%SZ") for lag in range(12, 0, -1)]
                if h.height != 12 * info.n_links or h["node"].null_count() or h["timestamp"].unique(maintain_order=True).to_list() != expected:
                    raise ValueError(f"Queue history not 12 complete causal steps: {window}")
                hist = {name: h[col].to_numpy().reshape(12, info.n_links) for name, col in
                        (("speed", "speed_kmh"), ("flow_lane", "flow_per_lane"),
                         ("occupancy", "occupancy"), ("pct_observed", "pct_observed"))}
                x, names = features(hist, net, origin.hour * 12 + origin.minute // 5, origin.weekday())
                if names != config["feature_names"]:
                    raise ValueError("Queue feature order changed")
                prob = model.predict_proba(as_frame(x, names))[:, 1]
                future = [(origin + timedelta(minutes=5 * step)).strftime("%Y-%m-%dT%H:%M:%SZ") for step in range(1, 7)]
                grid = pl.DataFrame({"timestamp": np.repeat(future, info.n_links), "link_id": list(info.link_ids) * 6,
                                     "probability": prob, "queue_pred": (prob >= config["threshold"]).astype(np.int64)})
                requested = targets.filter(pl.col("window_id") == window).drop("queue_pred").join(grid, on=["timestamp", "link_id"], how="left", validate="1:1")
                if requested["queue_pred"].null_count() or requested.height != 6 * info.n_links:
                    raise ValueError(f"Queue target coverage failed: {window}")
                parts.append(requested.with_columns(pl.lit(panel).alias("panel")))
            print(f"queue inference {split}/{panel}: {targets.height:,} targets", flush=True)
        pl.concat(parts).write_parquet(out / f"predictions_{split}.parquet")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/queue_catboost_003"))
    parser.add_argument("--cache", type=Path, default=paths.CACHE_ROOT / "preprocessed_v1")
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--predict-only", action="store_true")
    args = parser.parse_args()
    if args.predict_only:
        predict(args.out, args.cache)
    else:
        fit_and_predict(args.out, args.cache, iterations=args.iterations)


if __name__ == "__main__":
    main()
