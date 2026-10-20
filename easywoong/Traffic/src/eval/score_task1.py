"""Local Task 1 scoring on train, where unmasked ground truth is released.

Implements the published metric exactly (spec §2):

    S_v(r)     = max(0, 1 - RMSE_speed(r) / 25)            km/h
    S_q(r)     = max(0, 1 - RMSE_flow_per_lane(r) / 600)   veh/h/lane
    S_state(r) = 0.54 * S_v(r) + 0.46 * S_q(r)
    S_state    = mean over regimes R1, R2, R3

Two things are easy to get wrong and are handled here:
  - flow error is normalised per lane before the RMSE, so corridors with
    different lane counts are comparable;
  - the target list comes from task1/<PANEL>/<split>/sample_submission_state.csv,
    never from re-deriving the mask.

This is the only local feedback loop in the competition. Task 3 rides entirely
on these same predictions, so improving this score is also the only lever on
physics consistency.
"""

from __future__ import annotations

import polars as pl

from src.io import loaders, paths

SPEED_SCALE = 25.0
FLOW_PER_LANE_SCALE = 600.0
W_SPEED = 0.54
W_FLOW = 0.46

PRED_COLUMNS = ("timestamp", "station_id", "link_id", "speed_kmh", "flow_vph")


def _regime_score(rmse_speed: float, rmse_flow_per_lane: float) -> dict[str, float]:
    s_v = max(0.0, 1.0 - rmse_speed / SPEED_SCALE)
    s_q = max(0.0, 1.0 - rmse_flow_per_lane / FLOW_PER_LANE_SCALE)
    return {
        "rmse_speed": rmse_speed,
        "rmse_flow_per_lane": rmse_flow_per_lane,
        "s_v": s_v,
        "s_q": s_q,
        "s_state": W_SPEED * s_v + W_FLOW * s_q,
    }


def score_panel(
    predictions: pl.DataFrame | pl.LazyFrame,
    panel: str,
    split: str = "train",
) -> dict:
    """Score one panel's Task 1 predictions against released ground truth.

    predictions needs columns timestamp / station_id / link_id / speed_kmh / flow_vph.
    Only train can be scored locally; validation and private have no unmasked layer.
    """
    if split != "train":
        raise ValueError(
            f"split={split!r} has no released ground truth; only 'train' can be scored locally"
        )

    pred = predictions.lazy().select(
        pl.col("timestamp"),
        pl.col("station_id"),
        pl.col("link_id"),
        pl.col("speed_kmh").cast(pl.Float64).alias("pred_speed"),
        pl.col("flow_vph").cast(pl.Float64).alias("pred_flow"),
    )

    targets = loaders.load_task1_targets(panel, split).select(
        "timestamp", "station_id", "link_id", "mask_regime"
    )

    truth = loaders.load_mainline_unmasked(panel).select(
        "timestamp",
        "station_id",
        "link_id",
        pl.col("speed_kmh").alias("true_speed"),
        pl.col("flow_vph").alias("true_flow"),
        "is_score_eligible",
    )

    lanes = loaders.load_links(panel).lazy().select("link_id", "lanes")

    joined = (
        targets.join(truth, on=list(loaders.JOIN_KEYS), how="inner")
        .filter(pl.col("is_score_eligible") == 1)
        .join(pred, on=list(loaders.JOIN_KEYS), how="left")
        .join(lanes, on="link_id", how="left")
    )

    stats = (
        joined.group_by("mask_regime")
        .agg(
            pl.len().alias("n"),
            pl.col("pred_speed").is_null().sum().alias("n_missing_pred"),
            ((pl.col("pred_speed") - pl.col("true_speed")) ** 2).mean().sqrt().alias("rmse_speed"),
            (((pl.col("pred_flow") - pl.col("true_flow")) / pl.col("lanes")) ** 2)
            .mean()
            .sqrt()
            .alias("rmse_flow_per_lane"),
        )
        .collect()
    )

    per_regime: dict[str, dict[str, float]] = {}
    for row in stats.iter_rows(named=True):
        entry = _regime_score(row["rmse_speed"], row["rmse_flow_per_lane"])
        entry["n"] = row["n"]
        entry["n_missing_pred"] = row["n_missing_pred"]
        per_regime[row["mask_regime"]] = entry

    missing = [r for r in paths.REGIMES if r not in per_regime]
    if missing:
        raise ValueError(f"{panel}: no scored rows for regime(s) {missing}")

    return {
        "panel": panel,
        "s_state": sum(per_regime[r]["s_state"] for r in paths.REGIMES) / len(paths.REGIMES),
        "per_regime": per_regime,
    }


def aggregate_panels(panel_scores: dict[str, float]) -> dict:
    """Panel scores -> corridor-family averages -> equal average over families.

    Having both directions of a family does not make that corridor count twice.
    """
    families: dict[str, float] = {}
    for family, members in paths.CORRIDOR_FAMILIES.items():
        present = [panel_scores[p] for p in members if p in panel_scores]
        if present:
            families[family] = sum(present) / len(present)
    overall = sum(families.values()) / len(families) if families else 0.0
    return {"s_state": overall, "per_family": families}


def format_report(results: list[dict], overall: dict | None = None) -> str:
    lines = [
        f"{'panel':<12} {'R1':>7} {'R2':>7} {'R3':>7} {'S_state':>8}  "
        f"{'RMSE_v':>7} {'RMSE_q/ln':>9}",
        "-" * 64,
    ]
    for r in results:
        pr = r["per_regime"]
        mean_rmse_v = sum(pr[x]["rmse_speed"] for x in paths.REGIMES) / 3
        mean_rmse_q = sum(pr[x]["rmse_flow_per_lane"] for x in paths.REGIMES) / 3
        lines.append(
            f"{r['panel']:<12} "
            + " ".join(f"{pr[x]['s_state']:>7.4f}" for x in paths.REGIMES)
            + f" {r['s_state']:>8.4f}  {mean_rmse_v:>7.2f} {mean_rmse_q:>9.2f}"
        )
    if overall is not None:
        lines.append("-" * 64)
        for family, score in overall["per_family"].items():
            lines.append(f"  {family:<10} {score:.4f}")
        lines.append(f"{'S_state (5 corridor mean)':<30} {overall['s_state']:.4f}")
    return "\n".join(lines)
