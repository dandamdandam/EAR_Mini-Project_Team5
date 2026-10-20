"""Score a masked-input-only interpolation baseline on February 2031 Task 1.

Run from the Traffic directory with the easypj environment::

    python -m src.task1.compare_baseline_feb

Only the released February masked files feed the predictor. The official Task 1
train CSV supplies the requested keys; the February unmasked files are opened
only after predictions have been generated, to compute the local score.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import polars as pl

from src.io import paths
from src.eval import score_task1


MONTH = "2031-02"
KEYS = ["timestamp", "station_id", "link_id"]
FINE_KEYS = ["station_id", "link_id", "weekday", "slot"]
MID_KEYS = ["station_id", "link_id", "slot"]
COARSE_KEYS = ["station_id", "link_id"]
W_TEMPORAL = 0.85
W_SPATIAL = 0.15


def _feb_files(panel: str) -> dict[str, Path]:
    root = paths.mainline_masked_dir(panel, "train")
    files = list(root.glob("mask_regime=*/synthetic_mainline_2031_02_*.parquet"))
    by_date = {path.stem[-10:].replace("_", "-"): path for path in files}
    if len(files) != 28 or len(by_date) != 28:
        raise ValueError(f"{panel}: expected 28 distinct February masked daily files, got {len(files)}")
    return dict(sorted(by_date.items()))


def _time_features(frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    return (
        frame.with_columns(
            pl.col("timestamp")
            .str.to_datetime(format="%Y-%m-%dT%H:%M:%SZ", time_unit="us")
            .alias("_time")
        )
        .with_columns(
            pl.col("_time").dt.weekday().alias("weekday"),
            (pl.col("_time").dt.hour() * 12 + pl.col("_time").dt.minute() // 5).alias("slot"),
        )
        .drop("_time")
    )


def _feb_targets(panel: str) -> dict[str, pl.DataFrame]:
    targets = (
        pl.scan_csv(paths.task1_targets(panel, "train"))
        .select("timestamp", "station_id", "link_id", "mask_regime")
        .filter(pl.col("timestamp").str.starts_with(MONTH))
        .with_columns(pl.col("timestamp").str.slice(0, 10).alias("date"))
        .collect()
    )
    if targets.is_empty() or targets.select(KEYS).is_duplicated().any():
        raise ValueError(f"{panel}: empty or duplicate official February Task 1 keys")
    return {row[0]: group.drop("date") for row, group in targets.partition_by("date", as_dict=True).items()}


def _feb_profile(files: dict[str, Path]) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, float, float]:
    """Fit fallback values using visible February masked observations only."""
    observed = _time_features(
        pl.scan_parquet([str(path) for path in files.values()])
        .filter(pl.col("speed_kmh").is_not_null() & pl.col("flow_vph").is_not_null())
        .select("timestamp", "station_id", "link_id", "speed_kmh", "flow_vph")
    )
    fine = observed.group_by(FINE_KEYS).agg(
        pl.col("speed_kmh").sum().alias("speed_sum"),
        pl.col("flow_vph").sum().alias("flow_sum"),
        pl.len().alias("n"),
    ).collect()
    if fine.is_empty():
        raise ValueError("February masked input has no visible speed/flow pairs")

    mid = fine.group_by(MID_KEYS).agg(
        pl.col("speed_sum").sum(), pl.col("flow_sum").sum(), pl.col("n").sum()
    )
    coarse = fine.group_by(COARSE_KEYS).agg(
        pl.col("speed_sum").sum(), pl.col("flow_sum").sum(), pl.col("n").sum()
    )
    speed_global = fine["speed_sum"].sum() / fine["n"].sum()
    flow_global = fine["flow_sum"].sum() / fine["n"].sum()

    def means(frame: pl.DataFrame, suffix: str, keys: list[str]) -> pl.DataFrame:
        return frame.with_columns(
            (pl.col("speed_sum") / pl.col("n")).alias(f"speed_{suffix}"),
            (pl.col("flow_sum") / pl.col("n")).alias(f"flow_{suffix}"),
        ).select(keys + [f"speed_{suffix}", f"flow_{suffix}"])

    return (
        means(fine, "fine", FINE_KEYS),
        means(mid, "mid", MID_KEYS),
        means(coarse, "coarse", COARSE_KEYS),
        float(speed_global),
        float(flow_global),
    )


def _predict_day(
    masked_path: Path,
    targets: pl.DataFrame,
    links: pl.DataFrame,
    profile: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, float, float],
) -> pl.DataFrame:
    fine, mid, coarse, speed_global, flow_global = profile
    masked = (
        pl.read_parquet(masked_path, columns=KEYS + ["milepost", "speed_kmh", "flow_vph"])
        .join(links, on="link_id", how="left")
        .with_columns((pl.col("flow_vph") / pl.col("lanes")).alias("flow_lane"))
    )
    if masked["lanes"].null_count():
        raise ValueError(f"{masked_path}: unknown link_id in masked input")

    temporal = masked.sort("timestamp").with_columns(
        pl.col("speed_kmh").interpolate().over("station_id", "link_id").alias("t_speed"),
        pl.col("flow_lane").interpolate().over("station_id", "link_id").alias("t_flow"),
    )
    spatial = temporal.sort("milepost").with_columns(
        pl.col("speed_kmh").interpolate().over("timestamp").alias("s_speed"),
        pl.col("flow_lane").interpolate().over("timestamp").alias("s_flow"),
    )
    blended = spatial.with_columns(
        pl.coalesce(
            W_TEMPORAL * pl.col("t_speed") + W_SPATIAL * pl.col("s_speed"),
            pl.col("t_speed"),
            pl.col("s_speed"),
        ).alias("blend_speed"),
        pl.coalesce(
            W_TEMPORAL * pl.col("t_flow") + W_SPATIAL * pl.col("s_flow"),
            pl.col("t_flow"),
            pl.col("s_flow"),
        ).alias("blend_flow_lane"),
    ).select(KEYS + ["lanes", "free_speed_kmh", "blend_speed", "blend_flow_lane"])

    predicted = (
        _time_features(targets)
        .join(blended, on=KEYS, how="left")
        .join(fine, on=FINE_KEYS, how="left")
        .join(mid, on=MID_KEYS, how="left")
        .join(coarse, on=COARSE_KEYS, how="left")
        .with_columns(
            pl.coalesce(
                "blend_speed", "speed_fine", "speed_mid", "speed_coarse", pl.lit(speed_global)
            ).alias("speed_kmh"),
            pl.coalesce(
                pl.col("blend_flow_lane") * pl.col("lanes"),
                "flow_fine", "flow_mid", "flow_coarse", pl.lit(flow_global),
            ).alias("flow_vph"),
        )
        .with_columns(
            pl.col("speed_kmh")
            .clip(
                lower_bound=5.0,
                upper_bound=pl.min_horizontal(pl.lit(135.0), pl.col("free_speed_kmh") * 1.05),
            ),
            pl.col("flow_vph").clip(lower_bound=0.0),
        )
        .select(KEYS + ["mask_regime", "speed_kmh", "flow_vph", "lanes"])
    )
    if predicted.height != targets.height or predicted.select("speed_kmh", "flow_vph", "lanes").null_count().sum_horizontal().sum() != 0:
        raise ValueError(f"{masked_path}: missing or duplicated predictions for official targets")
    return predicted


def _score_day(predicted: pl.DataFrame, truth_path: Path) -> dict[str, tuple[int, float, float]]:
    """This function alone opens the unmasked ground truth."""
    truth = pl.read_parquet(
        truth_path, columns=KEYS + ["speed_kmh", "flow_vph", "is_score_eligible"]
    ).rename({"speed_kmh": "true_speed", "flow_vph": "true_flow"})
    joined = predicted.join(truth, on=KEYS, how="left")
    if joined.height != predicted.height or joined["true_speed"].null_count() or joined["true_flow"].null_count():
        raise ValueError(f"{truth_path}: missing or duplicate ground truth for official targets")
    if not joined["is_score_eligible"].eq(1).all():
        raise ValueError(f"{truth_path}: official target is not score eligible")
    if not joined.select("speed_kmh", "flow_vph").to_numpy().dtype.kind in "fi":
        raise ValueError("predictions are not numeric")
    return {
        regime: (int(n), float(sse_speed), float(sse_flow))
        for regime, n, sse_speed, sse_flow in joined.group_by("mask_regime").agg(
            pl.len().alias("n"),
            ((pl.col("speed_kmh") - pl.col("true_speed")) ** 2).sum().alias("sse_speed"),
            (
                ((pl.col("flow_vph") - pl.col("true_flow")) / pl.col("lanes")) ** 2
            ).sum().alias("sse_flow_lane"),
        ).iter_rows()
    }


def compare_panel(panel: str) -> dict:
    files = _feb_files(panel)
    targets_by_date = _feb_targets(panel)
    if set(targets_by_date) != set(files):
        raise ValueError(f"{panel}: February target days and masked-input days differ")

    links = pl.read_csv(paths.network_dir(panel) / "links.csv").select(
        "link_id", pl.col("lanes").cast(pl.Float64), "free_speed_kmh"
    )
    profile = _feb_profile(files)
    totals = defaultdict(lambda: [0, 0.0, 0.0])
    for date, masked_path in files.items():
        predicted = _predict_day(masked_path, targets_by_date[date], links, profile)
        truth_path = paths.mainline_unmasked_dir(panel) / f"year_month={MONTH}" / masked_path.name
        for regime, (n, sse_speed, sse_flow) in _score_day(predicted, truth_path).items():
            totals[regime][0] += n
            totals[regime][1] += sse_speed
            totals[regime][2] += sse_flow

    regimes = {}
    for regime in paths.REGIMES:
        n, speed_sse, flow_sse = totals[regime]
        if not n:
            raise ValueError(f"{panel}: no official February targets for {regime}")
        rmse_speed = math.sqrt(speed_sse / n)
        rmse_flow_lane = math.sqrt(flow_sse / n)
        s_speed = max(0.0, 1.0 - rmse_speed / score_task1.SPEED_SCALE)
        s_flow = max(0.0, 1.0 - rmse_flow_lane / score_task1.FLOW_PER_LANE_SCALE)
        regimes[regime] = {
            "n": n,
            "rmse_speed_kmh": rmse_speed,
            "rmse_flow_vph_per_lane": rmse_flow_lane,
            "s_speed": s_speed,
            "s_flow": s_flow,
            "s_state": score_task1.W_SPEED * s_speed + score_task1.W_FLOW * s_flow,
        }
    return {
        "panel": panel,
        "n_targets": sum(item["n"] for item in regimes.values()),
        "s_state": sum(item["s_state"] for item in regimes.values()) / len(regimes),
        "per_regime": regimes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=paths.PROJECT_ROOT / "runs" / "staeformer_baseline_feb.json")
    parser.add_argument("--panels", nargs="+", choices=paths.PANELS, default=paths.PANELS)
    args = parser.parse_args()

    panels = [compare_panel(panel) for panel in args.panels]
    overall = score_task1.aggregate_panels({row["panel"]: row["s_state"] for row in panels})
    report = {
        "method": "February masked-only 0.85 temporal / 0.15 spatial linear interpolation with February masked-only profile fallback",
        "month": MONTH,
        "input": "corridors/<panel>/train/mainline_states_masked, February only",
        "targets": "task1/<panel>/train/sample_submission_state.csv, February keys only",
        "truth_for_scoring_only": "corridors/<panel>/train/mainline_states, February only",
        "n_targets": sum(row["n_targets"] for row in panels),
        "s_state": overall["s_state"],
        "per_family": overall["per_family"],
        "per_panel": {row["panel"]: {k: v for k, v in row.items() if k != "panel"} for row in panels},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out}: {report['n_targets']:,} targets, S_state={report['s_state']:.4f}")
    for row in panels:
        print(f"  {row['panel']:<12} {row['n_targets']:>9,}  {row['s_state']:.4f}")


if __name__ == "__main__":
    main()
