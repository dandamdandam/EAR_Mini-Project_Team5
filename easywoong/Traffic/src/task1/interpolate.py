"""Spatio-temporal interpolation for Task 1.

The historical profile in baseline.py ignores the two strongest signals in the
data: the same link five minutes earlier, and the neighbouring link right now.
This fills blanked cells from both instead.

    temporal   linear interpolation along time within a link
    spatial    linear interpolation along the corridor (ordered by milepost)
               at a fixed timestamp
    blend      w_t * temporal + w_s * spatial, falling back to the historical
               profile where neither neighbour exists

Fitted on the masked layer only, like baseline.py, so nothing leaks and the same
path runs on validation and private.

Flow is interpolated per lane and scaled back afterwards. Interpolating raw
flow across links would mix 3-lane and 6-lane sections and invent traffic at
every lane-count change.
"""

from __future__ import annotations

import polars as pl

from src.io import loaders
from src.task1 import baseline

W_TEMPORAL = 0.85
W_SPATIAL = 0.15
SPEED_BOUNDS = (5.0, 135.0)
MIN_FLOW = 0.0


def fit_predict(
    panel: str,
    split: str = "train",
    w_temporal: float = W_TEMPORAL,
    w_spatial: float = W_SPATIAL,
) -> pl.DataFrame:
    lanes = loaders.load_links(panel).lazy().select("link_id", "lanes", "free_speed_kmh")

    masked = (
        loaders.load_mainline_masked(panel, split)
        .select("timestamp", "station_id", "link_id", "milepost", "speed_kmh", "flow_vph")
        .join(lanes, on="link_id", how="left")
        .with_columns((pl.col("flow_vph") / pl.col("lanes")).alias("flow_lane"))
    )

    # Temporal: walk each link forward in time and fill the gaps between observations.
    temporal = masked.sort("timestamp").with_columns(
        pl.col("speed_kmh").interpolate().over("station_id", "link_id").alias("t_speed"),
        pl.col("flow_lane").interpolate().over("station_id", "link_id").alias("t_flow"),
    )

    # Spatial: at a fixed timestamp, walk along the corridor by milepost.
    spatial = temporal.sort("milepost").with_columns(
        pl.col("speed_kmh").interpolate().over("timestamp").alias("s_speed"),
        pl.col("flow_lane").interpolate().over("timestamp").alias("s_flow"),
    )

    blended = spatial.with_columns(
        pl.coalesce(
            w_temporal * pl.col("t_speed") + w_spatial * pl.col("s_speed"),
            pl.col("t_speed"),
            pl.col("s_speed"),
        ).alias("blend_speed"),
        pl.coalesce(
            w_temporal * pl.col("t_flow") + w_spatial * pl.col("s_flow"),
            pl.col("t_flow"),
            pl.col("s_flow"),
        ).alias("blend_flow"),
    ).select("timestamp", "station_id", "link_id", "lanes", "free_speed_kmh", "blend_speed", "blend_flow")

    targets = loaders.load_task1_targets(panel, split)

    profile = baseline.fit_predict(panel, split).lazy().select(
        "timestamp",
        "station_id",
        "link_id",
        pl.col("speed_kmh").alias("prof_speed"),
        pl.col("flow_vph").alias("prof_flow"),
    )

    joined = (
        targets.join(blended, on=list(loaders.JOIN_KEYS), how="left")
        .join(profile, on=list(loaders.JOIN_KEYS), how="left")
        .with_columns(
            pl.coalesce("blend_speed", "prof_speed").alias("speed_kmh"),
            pl.coalesce(
                pl.col("blend_flow") * pl.col("lanes"),
                pl.col("prof_flow"),
            ).alias("flow_vph"),
        )
        .with_columns(
            pl.col("speed_kmh").clip(
                lower_bound=SPEED_BOUNDS[0],
                upper_bound=pl.min_horizontal(
                    pl.lit(SPEED_BOUNDS[1]), pl.col("free_speed_kmh") * 1.05
                ),
            ),
            pl.col("flow_vph").clip(lower_bound=MIN_FLOW),
        )
        .select("timestamp", "station_id", "link_id", "speed_kmh", "flow_vph")
    )
    return joined.collect()
