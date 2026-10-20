"""Task 1 / Task 3 diagnostics: where observations sit relative to the FD.

Density is computed as k = q / v, matching how the evaluator derives it. The
released density_occ_linear_vehpkm column is occupancy-derived and will not line
up with the physics score, so it is only carried here for comparison.

Against the triangular fundamental diagram

    q_FD(k) = v_f * k            for k <= k_crit
    q_FD(k) = w  * (k_jam - k)   for k >  k_crit,   w = q_cap / (k_jam - k_crit)

the gap |q - q_FD(k)| / q_cap says how far a speed/flow pair sits from a
combination the road can actually produce. Task 3 rewards closing that gap, and
since Task 3 has no submission rows of its own, the only way to move it is to
change these same Task 1 answers.
"""

from __future__ import annotations

import polars as pl

from src.io import loaders, paths

PERCENTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def fd_table(panel: str) -> pl.DataFrame:
    """Per-link triangular FD parameters, per lane."""
    fd = loaders.load_fd_parameters(panel)
    return fd.with_columns(
        (pl.col("capacity_vph") / pl.col("lanes")).alias("q_cap_lane"),
        (pl.col("critical_density") / pl.col("lanes")).alias("k_crit_lane"),
        (pl.col("k_jam") / pl.col("lanes")).alias("k_jam_lane"),
    ).with_columns(
        (pl.col("q_cap_lane") / (pl.col("k_jam_lane") - pl.col("k_crit_lane"))).alias("w_lane")
    )


def observed_vs_fd(panel: str, months: list[str] | None = None) -> pl.DataFrame:
    """Join released observations to their link's FD and measure the gap."""
    months = months or ["2030-06"]
    obs = (
        loaders.load_mainline_unmasked(panel, months)
        .filter(
            pl.col("speed_kmh").is_not_null()
            & pl.col("flow_vph").is_not_null()
            & (pl.col("speed_kmh") > 0)
            & (pl.col("is_score_eligible") == 1)
        )
        .select("timestamp", "link_id", "speed_kmh", "flow_vph", "density_occ_linear_vehpkm")
        .collect()
    )

    joined = (
        obs.join(fd_table(panel), on="link_id", how="inner")
        .with_columns(
            # k = q / v, per lane — the evaluator's definition, not the released column.
            (pl.col("flow_vph") / pl.col("speed_kmh") / pl.col("lanes")).alias("k_lane"),
            (pl.col("flow_vph") / pl.col("lanes")).alias("q_lane"),
        )
        .with_columns(
            pl.when(pl.col("k_lane") <= pl.col("k_crit_lane"))
            .then(pl.col("free_speed_kmh") * pl.col("k_lane"))
            .otherwise(pl.col("w_lane") * (pl.col("k_jam_lane") - pl.col("k_lane")))
            .clip(lower_bound=0.0)
            .alias("q_fd_lane")
        )
        .with_columns(
            ((pl.col("q_lane") - pl.col("q_fd_lane")).abs() / pl.col("q_cap_lane")).alias("fd_gap"),
            (pl.col("k_lane") > pl.col("k_crit_lane")).alias("congested"),
        )
    )
    return joined.with_columns(pl.lit(panel).alias("panel"))


def run(panels=None, months: list[str] | None = None) -> pl.DataFrame:
    panels = panels or list(paths.PANELS)
    frames = [observed_vs_fd(p, months) for p in panels]
    data = pl.concat(frames, how="vertical")

    print("=== Task 1/3: observed states vs the triangular FD ===")
    print(
        data.group_by("panel")
        .agg(
            pl.len().alias("cells"),
            pl.col("k_lane").quantile(0.50).round(1).alias("k_med"),
            pl.col("speed_kmh").quantile(0.05).round(1).alias("v_p05"),
            pl.col("speed_kmh").quantile(0.50).round(1).alias("v_med"),
            pl.col("congested").mean().round(4).alias("congested_frac"),
            pl.col("fd_gap").median().round(4).alias("fd_gap_med"),
            pl.col("fd_gap").quantile(0.95).round(4).alias("fd_gap_p95"),
        )
        .sort("panel")
    )

    print("\n=== q/v density vs the released occupancy-derived column ===")
    comparison = data.select(
        pl.col("k_lane").mul(pl.col("lanes")).alias("k_from_qv"),
        pl.col("density_occ_linear_vehpkm").alias("k_released"),
    ).drop_nulls()
    corr = comparison.select(pl.corr("k_from_qv", "k_released")).item()
    print(
        comparison.select(
            pl.col("k_from_qv").median().round(2).alias("qv_median"),
            pl.col("k_released").median().round(2).alias("released_median"),
        )
    )
    print(f"correlation: {corr:.4f}")
    print(
        "\nThe two agree closely at the median and correlate ~0.98, so they are not\n"
        "wildly different quantities — but that residual 2% is concentrated exactly\n"
        "where occupancy and q/v diverge, which is congestion, which is where the\n"
        "physics score is decided. Normalise on q/v regardless (CLAUDE.md rule 2):\n"
        "it is what the evaluator recomputes, so it is the only definition that can\n"
        "line up with the score."
    )

    print("\n=== How much of the data is congested ===")
    print(
        "The congested share is small, but it is where speed reconstruction fails and\n"
        "where over-smoothing erases the queues that S_LWR is looking for. A model that\n"
        "lowers RMSE by flattening these cells can take the physics score to zero."
    )
    return data
