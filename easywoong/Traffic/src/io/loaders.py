"""Polars loaders for the TrafficFlowBench release.

Everything returns LazyFrames where the underlying data is large; callers decide
when to collect. The 643 MB submission_key.csv is never read eagerly.
"""

from __future__ import annotations

import polars as pl

from . import paths

# Columns common to mainline observation files.
MAINLINE_MEASURED = ("speed_kmh", "flow_vph", "occupancy", "density_occ_linear_vehpkm")
JOIN_KEYS = ("timestamp", "station_id", "link_id")


def _parse_timestamp(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Timestamps arrive as '2030-06-02T00:00:00Z' strings."""
    return lf.with_columns(
        pl.col("timestamp").str.to_datetime(format="%Y-%m-%dT%H:%M:%SZ", time_unit="us")
    )


def load_mainline_unmasked(panel: str, months: list[str] | None = None) -> pl.LazyFrame:
    """Ground-truth mainline states. Train only — this is what local scoring joins against."""
    root = paths.mainline_unmasked_dir(panel)
    if months is None:
        lf = pl.scan_parquet(root / "**" / "*.parquet")
    else:
        lf = pl.concat(
            [pl.scan_parquet(root / f"year_month={m}" / "*.parquet") for m in months],
            how="vertical",
        )
    return _parse_timestamp(lf)


def load_mainline_masked(
    panel: str, split: str, regimes: tuple[str, ...] | None = None
) -> pl.LazyFrame:
    """Masked mainline states — the only mainline view available for validation/private.

    Each calendar date belongs to exactly one mask regime, so reading all three
    regimes yields each date once.
    """
    root = paths.mainline_masked_dir(panel, split)
    regimes = regimes or paths.REGIMES
    lf = pl.concat(
        [pl.scan_parquet(root / f"mask_regime={r}" / "*.parquet") for r in regimes],
        how="vertical",
    )
    return _parse_timestamp(lf)


def load_ramp(panel: str, split: str) -> pl.LazyFrame:
    """Ramp counts. No speed column — ramp loops only count vehicles crossing them."""
    return _parse_timestamp(pl.scan_parquet(paths.ramp_dir(panel, split) / "**" / "*.parquet"))


def load_task1_targets(panel: str, split: str) -> pl.LazyFrame:
    """The authoritative target list for Task 1.

    speed_kmh / flow_vph arrive empty, so they are read as strings and dropped;
    deriving the mask from the masked files instead would yield a larger set
    (Task 2 window cells are excluded from Task 1).
    """
    lf = pl.scan_csv(
        paths.task1_targets(panel, split),
        schema_overrides={"speed_kmh": pl.String, "flow_vph": pl.String},
    ).drop("speed_kmh", "flow_vph")
    return _parse_timestamp(lf)


def _normalise_lanes(df: pl.DataFrame) -> pl.DataFrame:
    """`lanes` ships as Int64 on some panels and Float64 on others; concatenating
    frames across panels fails on that mismatch, so pin it to Float64 here."""
    return df.with_columns(pl.col("lanes").cast(pl.Float64))


def load_links(panel: str) -> pl.DataFrame:
    """link_id, length_km, lanes, free_speed_kmh, capacity_vph."""
    return _normalise_lanes(pl.read_csv(paths.network_dir(panel) / "links.csv"))


def load_fd_parameters(panel: str) -> pl.DataFrame:
    """links.csv columns plus critical_density and k_jam (triangular FD)."""
    return _normalise_lanes(pl.read_csv(paths.network_dir(panel) / "fd_parameters.csv"))


def load_topology(panel: str) -> pl.DataFrame:
    """Which mainline link flows into which, and which ramps attach where."""
    return pl.read_csv(paths.network_dir(panel) / "lwr_mainline_topology.csv")


def load_ramp_attachment(panel: str) -> pl.DataFrame:
    """ramp_link_id -> nearest mainline link, with ramp_type OR/FR."""
    return pl.read_csv(paths.network_dir(panel) / "ramp_attachment_map.csv")


def scan_submission_key(
    task: str | None = None, split: str | None = None, panel: str | None = None
) -> pl.LazyFrame:
    """Filtered scan of the 643 MB key. Always prune before collecting."""
    lf = pl.scan_csv(paths.SUBMISSION_KEY)
    if task is not None:
        lf = lf.filter(pl.col("task") == task)
    if split is not None:
        lf = lf.filter(pl.col("split") == split)
    if panel is not None:
        lf = lf.filter(pl.col("panel") == panel)
    return lf


def add_time_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Weekday and 5-minute-of-day slot, the backbone of any historical profile."""
    return lf.with_columns(
        pl.col("timestamp").dt.weekday().alias("weekday"),
        (pl.col("timestamp").dt.hour() * 12 + pl.col("timestamp").dt.minute() // 5).alias("slot"),
    )
