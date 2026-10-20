"""Persistence baseline for Task 2 (queue forecasting).

Takes the last observed speed for each link in a window's 60-minute history and
carries its queue state forward across all six horizon steps.

Predicting all zeros scores exactly 0: every released window is constructed to
contain at least one queued cell in its horizon, so the intersection is always
empty. Persistence costs nothing and recovers the `queue_ongoing` windows, where
a queue is already present at the forecast origin.

The label rule is speed <= 0.60 * free-flow speed (spec §3).
"""

from __future__ import annotations

import polars as pl

from src.io import loaders, paths

QUEUE_SPEED_RATIO = 0.60


def predict(panel: str, split: str) -> pl.DataFrame:
    """Return window_id / timestamp / link_id / queue_pred for one panel."""
    history = pl.read_parquet(paths.task2_dir(panel, split) / "window_history.parquet")
    targets = pl.read_csv(paths.task2_dir(panel, split) / "sample_submission_queue.csv")
    free_speed = loaders.load_links(panel).select("link_id", "free_speed_kmh")

    last_seen = (
        history.filter(pl.col("speed_kmh").is_not_null())
        .sort("timestamp")
        .group_by("window_id", "link_id")
        .agg(pl.col("speed_kmh").last())
        .join(free_speed, on="link_id", how="left")
        .with_columns(
            (pl.col("speed_kmh") <= QUEUE_SPEED_RATIO * pl.col("free_speed_kmh"))
            .cast(pl.Int64)
            .alias("persisted")
        )
        .select("window_id", "link_id", "persisted")
    )

    return (
        targets.drop("queue_pred")
        .join(last_seen, on=["window_id", "link_id"], how="left")
        # A link never observed in the history window carries no evidence of a queue.
        .with_columns(pl.col("persisted").fill_null(0).alias("queue_pred"))
        .select("window_id", "timestamp", "link_id", "queue_pred")
    )
