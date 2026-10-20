"""Weak-prior passthrough for Task 4 (path-flow / OD estimation).

Submits the released weak prior unchanged. This is a valid, non-degenerate
starting point that exercises the submission path, but it is deliberately weak:
S_dev compares how far the submission moved from the prior against how far the
reference moved, so a zero-deviation answer gives that term away (spec §5).

Replacing this with the non-negative least squares fit

    min ||A f - c||^2 + lambda ||f - f_prior||^2   s.t.  f >= 0

is the intended next step; keeping this module fixed in the meantime is what
makes the Task 3 sensitivity back-calculation in CLAUDE.md §5 valid, since that
requires Task 2 and Task 4 to stay constant while Task 1 varies.
"""

from __future__ import annotations

import polars as pl

from src.io import paths


def predict(panel: str, split: str) -> pl.DataFrame:
    """Return panel / departure_time / path_id / path_flow for one panel."""
    prior = pl.read_csv(paths.task4_dir(panel, split) / "synthetic_weak_prior.csv")
    targets = pl.read_csv(paths.task4_dir(panel, split) / "sample_submission_path_flow.csv")

    return (
        targets.drop("path_flow")
        .join(
            prior.select("panel", "departure_time", "path_id", "path_flow"),
            on=["panel", "departure_time", "path_id"],
            how="left",
        )
        # path_flow must be finite and non-negative for every required row.
        .with_columns(pl.col("path_flow").fill_null(0.0).clip(lower_bound=0.0))
        .select("panel", "departure_time", "path_id", "path_flow")
    )
