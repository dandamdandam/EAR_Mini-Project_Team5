"""CPU CatBoost residual competitor for official Task 1 reconstruction.

Train on June--December synthetic masks, select residual blend on January's
official masked inputs/targets, and score February once after selection.

    conda run -n easypj python -m src.task1.catboost_residual --smoke
    conda run -n easypj python -m src.task1.catboost_residual
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

# Cap Polars as well as CatBoost on the shared CPU server.
os.environ.setdefault("POLARS_MAX_THREADS", "8")

import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostRegressor

from src.io import paths
from src.task1.catboost_features import CAT_FEATURES, FEATURES, features_for_targets, network_features
from src.task1.staeformer_data import (
    REGIME_PROBABILITIES,
    all_train_dates,
    load_day,
    official_targets,
    panel_infos,
    synthetic_target_mask,
    targets_for_day,
)


TRAIN_START, TRAIN_END = "2030-06-01", "2030-12-31"
SELECT_START, SELECT_END = "2031-01-01", "2031-01-31"
TEST_START, TEST_END = "2031-02-01", "2031-02-28"
BLENDS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)


def _train_samples(infos, networks, *, seed: int, samples_per_day: int, smoke: bool):
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []
    y_speed: list[np.ndarray] = []
    y_flow: list[np.ndarray] = []
    sample_counts = defaultdict(int)
    panel_days = 0
    for panel in paths.PANELS:
        dates = all_train_dates(panel, TRAIN_START, TRAIN_END)
        if smoke:
            dates = dates[:3]
        for index, day in enumerate(dates):
            rate = REGIME_PROBABILITIES[index % len(REGIME_PROBABILITIES)]
            regime = paths.REGIMES[index % len(paths.REGIMES)]
            raw = load_day(panel, "train", day, infos[panel], masked=False)
            synthetic = synthetic_target_mask(raw, rate, rng)
            possible = np.flatnonzero(synthetic)
            chosen = rng.choice(possible, size=min(len(possible), samples_per_day), replace=False)
            sampled = np.zeros(synthetic.size, dtype=bool)
            sampled[chosen] = True
            sampled = sampled.reshape(synthetic.shape)
            visible = np.isfinite(raw.speed) & np.isfinite(raw.flow_lane) & ~synthetic
            features, base = features_for_targets(
                raw, visible, sampled, networks[panel], rate
            )
            truth_speed = raw.speed[sampled]
            truth_flow = raw.flow_lane[sampled]
            if not (np.isfinite(truth_speed).all() and np.isfinite(truth_flow).all()):
                raise ValueError(f"Nonfinite synthetic truth: {panel}/{day}")
            frames.append(features)
            y_speed.append((truth_speed - base[:, 0]).astype(np.float32))
            y_flow.append((truth_flow - base[:, 1]).astype(np.float32))
            sample_counts[f"{panel}/{regime}"] += features.shape[0]
            panel_days += 1
        print(f"prepared {panel}: {len(dates)} training days", flush=True)
    x = pd.concat(frames, ignore_index=True)
    if list(x.columns) != FEATURES or x[CAT_FEATURES].isna().any().any():
        raise ValueError("Invalid CatBoost training feature table")
    return x, np.concatenate(y_speed), np.concatenate(y_flow), {
        "panel_days": panel_days,
        "sampled_rows": len(x),
        "samples_per_day_cap": samples_per_day,
        "per_panel_regime": dict(sorted(sample_counts.items())),
        "prepare_seconds": round(time.perf_counter() - started, 1),
    }


def _representative_days(panel: str, start: str, end: str) -> list[str]:
    dates = all_train_dates(panel, start, end)
    found = {}
    for day in dates:
        for regime in paths.REGIMES:
            path = (paths.mainline_masked_dir(panel, "train") / f"mask_regime={regime}"
                    / f"synthetic_mainline_{day.replace('-', '_')}.parquet")
            if path.exists():
                found.setdefault(regime, day)
    if set(found) != set(paths.REGIMES):
        raise ValueError(f"Missing official mask regime for {panel}/{start}..{end}")
    return sorted(found.values())


def _dates(start: str, end: str, *, smoke: bool) -> dict[str, list[str]]:
    if smoke:
        return {p: _representative_days(p, start, end) for p in paths.PANELS}
    return {p: all_train_dates(p, start, end) for p in paths.PANELS}


def _target_frames(dates_by_panel: dict[str, list[str]]) -> dict[str, pl.DataFrame]:
    return {
        panel: official_targets(panel, dates[0], dates[-1]).filter(
            pl.col("timestamp").str.slice(0, 10).is_in(dates)
        )
        for panel, dates in dates_by_panel.items()
    }


def _summary(accum, expected: int):
    panel_regime = defaultdict(dict)
    rows = []
    for (panel, regime), values in sorted(accum.items()):
        n = values["n"]
        if not n:
            raise ValueError(f"No scored targets: {panel}/{regime}")
        rv = (values["v2"] / n) ** 0.5
        rq = (values["q2"] / n) ** 0.5
        score = 0.54 * max(0.0, 1.0 - rv / 25.0) + 0.46 * max(0.0, 1.0 - rq / 600.0)
        rows.append({"panel": panel, "regime": regime, "n": n,
                     "rmse_speed": rv, "rmse_flow_per_lane": rq, "s_state": score})
        panel_regime[panel][regime] = score
    panel_scores = {}
    for panel in paths.PANELS:
        if set(panel_regime[panel]) != set(paths.REGIMES):
            raise ValueError(f"Missing score regime on {panel}: {panel_regime[panel]}")
        panel_scores[panel] = sum(panel_regime[panel].values()) / 3
    families = {
        name: sum(panel_scores[panel] for panel in members) / 2
        for name, members in paths.CORRIDOR_FAMILIES.items()
    }
    scored = sum(row["n"] for row in rows)
    if scored != expected:
        raise ValueError(f"Scored {scored} of {expected} official target rows")
    return {"s_state": sum(families.values()) / len(families), "rows": rows,
            "panel_scores": panel_scores, "family_scores": families, "scored_cells": scored}


def _score_official(models, infos, networks, target_frames, dates_by_panel, *, blends):
    """Use masked features to predict, then open originals only for scoring."""
    started = time.perf_counter()
    accum = {
        blend: defaultdict(lambda: {"v2": 0.0, "q2": 0.0, "n": 0})
        for blend in blends
    }
    expected = sum(frame.height for frame in target_frames.values())
    for panel in paths.PANELS:
        for day in dates_by_panel[panel]:
            masked = load_day(panel, "train", day, infos[panel], masked=True)
            target, regime = targets_for_day(target_frames[panel], masked, infos[panel])
            visible = np.isfinite(masked.speed) & np.isfinite(masked.flow_lane)
            if np.any(target & visible) or np.any(target & ~masked.eligible):
                raise ValueError(f"Invalid official target mask: {panel}/{day}")
            rate = REGIME_PROBABILITIES[paths.REGIMES.index(regime)]
            features, base = features_for_targets(masked, visible, target, networks[panel], rate)
            correction = np.column_stack([
                models["speed"].predict(features), models["flow_lane"].predict(features)
            ])
            if not np.isfinite(correction).all():
                raise ValueError(f"Nonfinite CatBoost residual: {panel}/{day}")

            # No unmasked January/February values have entered feature creation.
            truth = load_day(panel, "train", day, infos[panel], masked=False)
            if not (truth.timestamps == masked.timestamps and truth.eligible[target].all()
                    and np.isfinite(truth.speed[target]).all()
                    and np.isfinite(truth.flow_lane[target]).all()):
                raise ValueError(f"Official truth mismatch: {panel}/{day}")
            true_speed = truth.speed[target].astype(np.float64)
            true_flow = truth.flow_lane[target].astype(np.float64)
            node_ids = np.nonzero(target)[1]
            upper_speed = np.minimum(135.0, 1.05 * infos[panel].free_speeds[node_ids])
            for blend in blends:
                pred_speed = np.clip(base[:, 0] + blend * correction[:, 0], 5.0, upper_speed)
                pred_flow = np.maximum(base[:, 1] + blend * correction[:, 1], 0.0)
                stat = accum[blend][(panel, regime)]
                stat["v2"] += float(np.square(pred_speed - true_speed).sum())
                stat["q2"] += float(np.square(pred_flow - true_flow).sum())
                stat["n"] += len(true_speed)
        print(f"scored {panel}: {len(dates_by_panel[panel])} official masked days", flush=True)
    summaries = {str(blend): _summary(accum[blend], expected) for blend in blends}
    return summaries, round(time.perf_counter() - started, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-day", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--thread-count", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("runs/catboost_residual_001"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.thread_count < 1 or args.thread_count > 8:
        raise ValueError("CPU thread cap is 8")
    if args.samples_per_day < 1 or args.iterations < 1:
        raise ValueError("samples_per_day and iterations must be positive")
    if args.smoke:
        args.samples_per_day = min(args.samples_per_day, 64)
        args.iterations = min(args.iterations, 25)
        args.out = Path("runs/catboost_residual_smoke")
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    infos = panel_infos()
    networks = {
        panel: network_features(panel, infos[panel], panel_id)
        for panel_id, panel in enumerate(paths.PANELS)
    }
    settings = {
        "method": "CatBoost speed/flow-per-lane residuals over released-input interpolation",
        "train_dates": [TRAIN_START, TRAIN_END],
        "selection_dates": [SELECT_START, SELECT_END],
        "test_dates": [TEST_START, TEST_END] if not args.smoke else None,
        "seed": args.seed, "samples_per_day": args.samples_per_day,
        "iterations": args.iterations, "depth": args.depth,
        "learning_rate": args.learning_rate, "thread_count": args.thread_count,
        "feature_names": FEATURES, "categorical_features": CAT_FEATURES,
        "random_mask_rates": REGIME_PROBABILITIES,
        "catboost_version": __import__("catboost").__version__,
        "smoke": args.smoke,
    }
    (args.out / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    x, y_speed, y_flow, sample_report = _train_samples(
        infos, networks, seed=args.seed, samples_per_day=args.samples_per_day,
        smoke=args.smoke,
    )
    (args.out / "sample_report.json").write_text(json.dumps(sample_report, indent=2) + "\n")
    print(f"sampled {len(x):,} synthetic targets from {sample_report['panel_days']} panel-days", flush=True)

    models = {}
    fit_times = {}
    for name, labels in (("speed", y_speed), ("flow_lane", y_flow)):
        model = CatBoostRegressor(
            iterations=args.iterations,
            depth=args.depth,
            learning_rate=args.learning_rate,
            loss_function="RMSE",
            l2_leaf_reg=5.0,
            random_seed=args.seed,
            thread_count=args.thread_count,
            allow_writing_files=False,
            verbose=False,
        )
        fit_started = time.perf_counter()
        model.fit(x, labels, cat_features=CAT_FEATURES)
        fit_times[name] = round(time.perf_counter() - fit_started, 1)
        model.save_model(str(args.out / f"{name}.cbm"))
        models[name] = model
        print(f"fit {name}: {fit_times[name]:.1f}s", flush=True)
    del x, y_speed, y_flow

    select_dates = _dates(SELECT_START, SELECT_END, smoke=args.smoke)
    select_targets = _target_frames(select_dates)
    selection, selection_seconds = _score_official(
        models, infos, networks, select_targets, select_dates, blends=BLENDS
    )
    winning = max(selection, key=lambda blend: selection[blend]["s_state"])
    selected_blend = float(winning)
    selected_score = selection[winning]["s_state"]
    (args.out / "selection_best.json").write_text(json.dumps({
        "selected_blend": selected_blend, "selection_s_state": selected_score,
        "selection_scores": {b: report["s_state"] for b, report in selection.items()},
        "selected_summary": selection[winning], "selection_seconds": selection_seconds,
    }, indent=2) + "\n")
    print(f"January official selection: S_state={selected_score:.6f} at blend {winning}", flush=True)

    if args.smoke:
        print("smoke complete; February remains untouched", flush=True)
        return

    # February is first opened after both models and the blend are fixed.
    test_dates = _dates(TEST_START, TEST_END, smoke=False)
    test_targets = _target_frames(test_dates)
    test, test_seconds = _score_official(
        models, infos, networks, test_targets, test_dates,
        blends=tuple(dict.fromkeys((0.0, selected_blend))),
    )
    output = {
        "selected_epoch": None,
        "selected_blend": selected_blend,
        "selection_s_state": selected_score,
        "test": test,
        "test_target_rows": sum(frame.height for frame in test_targets.values()),
        "train_sampled_rows": sample_report["sampled_rows"],
        "cpu_seconds": {
            "prepare": sample_report["prepare_seconds"], "fit_speed": fit_times["speed"],
            "fit_flow_lane": fit_times["flow_lane"], "selection": selection_seconds,
            "test": test_seconds, "total": round(time.perf_counter() - started, 1),
        },
    }
    (args.out / "test_result.json").write_text(json.dumps(output, indent=2) + "\n")
    print(
        f"February official test: S_state={test[winning]['s_state']:.6f}; "
        f"interpolation-only={test['0.0']['s_state']:.6f}; "
        f"rows={output['test_target_rows']:,}; total={output['cpu_seconds']['total']:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
