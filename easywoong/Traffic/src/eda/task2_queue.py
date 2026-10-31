"""Task 2 diagnostics: queue kinematics in the 60-minute history windows.

The queue label is speed <= 0.60 * free-flow speed, and free-flow speed is
per link: across the release it runs 104.9-128.1 km/h, so the threshold itself
runs 62.9-76.9 km/h. Using one constant for every link (63 km/h, say, which is
0.6 * the lowest free-flow speed in the set) makes the test far too strict on
fast links and quietly suppresses queue predictions — the same under-prediction
that then looks mysterious downstream.

Two things are measured per window:
  margin  (v_last - threshold) / threshold   how close the link sits to queueing
  decel   (v_last - v_lag3) / 3              recent slope, km/h per 5-min step

For an onset window, by construction no link is queued anywhere in the history,
so every margin is positive; how small the minimum margin gets, and how hard the
link is decelerating, is what a forecast has to work with.
"""

from __future__ import annotations

import polars as pl

from src.io import loaders, paths

QUEUE_SPEED_RATIO = 0.60
LAG_STEPS = 3


def window_features(panel: str, split: str = "validation") -> pl.DataFrame:
    """Per (window, link): last observed speed, its margin to the threshold, recent slope."""
    history = pl.read_parquet(paths.task2_dir(panel, split) / "window_history.parquet")
    index = pl.read_csv(paths.task2_dir(panel, split) / "window_index.csv").select(
        "window_id", "condition", "forecast_origin"
    )
    thresholds = loaders.load_links(panel).select(
        "link_id",
        (QUEUE_SPEED_RATIO * pl.col("free_speed_kmh")).alias("threshold_kmh"),
    )

    observed = history.filter(pl.col("speed_kmh").is_not_null()).sort("timestamp")

    per_link = (
        observed.group_by("window_id", "link_id")
        .agg(
            pl.col("speed_kmh").last().alias("v_last"),
            pl.col("speed_kmh").shift(LAG_STEPS).last().alias("v_lag"),
            pl.len().alias("n_obs"),
        )
        .join(thresholds, on="link_id", how="left")
        .with_columns(
            ((pl.col("v_last") - pl.col("threshold_kmh")) / pl.col("threshold_kmh")).alias("margin"),
            ((pl.col("v_last") - pl.col("v_lag")) / LAG_STEPS).alias("decel"),
            (pl.col("v_last") <= pl.col("threshold_kmh")).alias("queued_at_origin"),
        )
    )
    return per_link.join(index, on="window_id", how="left").with_columns(
        pl.lit(panel).alias("panel")
    )


def run(panels=None, split: str = "validation") -> pl.DataFrame:
    panels = panels or paths.TASK2_PANELS
    features = pl.concat([window_features(p, split) for p in panels], how="vertical")

    print("=== Task 2: window inventory ===")
    print(
        features.group_by("panel", "condition")
        .agg(pl.col("window_id").n_unique().alias("windows"))
        .sort("panel", "condition")
    )

    print("\n=== Threshold spread (why a single constant fails) ===")
    print(
        features.group_by("panel")
        .agg(
            pl.col("threshold_kmh").min().round(1).alias("thr_min"),
            pl.col("threshold_kmh").max().round(1).alias("thr_max"),
        )
        .sort("panel")
    )

    print("\n=== Per-window closeness to queueing, by condition ===")
    per_window = features.group_by("panel", "condition", "window_id").agg(
        pl.col("margin").min().alias("min_margin"),
        pl.col("decel").min().alias("strongest_decel"),
        pl.col("queued_at_origin").sum().alias("links_queued_at_origin"),
    )
    print(
        per_window.group_by("condition")
        .agg(
            pl.col("min_margin").mean().round(3).alias("min_margin_mean"),
            pl.col("min_margin").min().round(3).alias("min_margin_min"),
            pl.col("strongest_decel").mean().round(2).alias("decel_mean"),
            pl.col("links_queued_at_origin").mean().round(2).alias("queued_links_mean"),
        )
        .sort("condition")
    )

    onset_leak = per_window.filter(
        (pl.col("condition") == "queue_onset") & (pl.col("links_queued_at_origin") > 0)
    ).height
    print(
        f"\nonset windows already showing a queued link at the origin: {onset_leak}"
        "\n(expected 0 — onset is defined as no queue anywhere in the history, so a"
        "\n non-zero count would mean the threshold or the join is wrong)"
    )
    print(
        "\nWhat this implies for the persistence baseline: it can only fire where a link"
        "\nis already queued at the forecast origin, so it scores nothing on onset windows"
        "\n— half of them. Closing that half needs the margin and slope above, not persistence."
    )
    return features
