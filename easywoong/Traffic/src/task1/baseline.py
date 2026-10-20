"""Historical-profile baseline for Task 1.

The profile is fitted on the *masked* layer only — the cells that remain visible
after blanking. That keeps it leak-free by construction (target cells are blank
there) and means the identical code path works on validation and private, where
no unmasked layer exists.

Prediction for a blanked cell is the mean of observed values at the same
station/link, weekday and 5-minute slot, falling back to coarser keys when that
cell was never observed.
"""

from __future__ import annotations

import polars as pl

from src.io import loaders

PROFILE_KEYS = ("station_id", "link_id", "weekday", "slot")
FALLBACK_KEYS = ("station_id", "link_id", "slot")
COARSE_KEYS = ("station_id", "link_id")


def _profile(observed: pl.LazyFrame, keys: tuple[str, ...], suffix: str) -> pl.LazyFrame:
    return observed.group_by(list(keys)).agg(
        pl.col("speed_kmh").mean().alias(f"speed_{suffix}"),
        pl.col("flow_vph").mean().alias(f"flow_{suffix}"),
    )


def fit_predict(panel: str, split: str = "train") -> pl.DataFrame:
    """Predict every Task 1 target cell for one panel.

    Returns timestamp / station_id / link_id / speed_kmh / flow_vph, which is
    what src.eval.score_task1.score_panel expects.
    """
    observed = loaders.add_time_features(
        loaders.load_mainline_masked(panel, split).filter(
            pl.col("speed_kmh").is_not_null() & pl.col("flow_vph").is_not_null()
        )
    )

    fine = _profile(observed, PROFILE_KEYS, "fine")
    mid = _profile(observed, FALLBACK_KEYS, "mid")
    coarse = _profile(observed, COARSE_KEYS, "coarse")
    overall = observed.select(
        pl.col("speed_kmh").mean().alias("speed_all"),
        pl.col("flow_vph").mean().alias("flow_all"),
    )

    targets = loaders.add_time_features(loaders.load_task1_targets(panel, split))

    predicted = (
        targets.join(fine, on=list(PROFILE_KEYS), how="left")
        .join(mid, on=list(FALLBACK_KEYS), how="left")
        .join(coarse, on=list(COARSE_KEYS), how="left")
        .join(overall, how="cross")
        .select(
            "timestamp",
            "station_id",
            "link_id",
            pl.coalesce("speed_fine", "speed_mid", "speed_coarse", "speed_all").alias("speed_kmh"),
            pl.coalesce("flow_fine", "flow_mid", "flow_coarse", "flow_all").alias("flow_vph"),
        )
    )
    return predicted.collect()
