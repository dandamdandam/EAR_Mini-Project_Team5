"""Task 4 diagnostics: what the released data can and cannot tell us.

S_ODME = 0.45*S_od + 0.25*S_link + 0.15*S_dev + 0.15*S_attr, and only one of
those four terms is computable offline:

  S_link  yes   c_hat = A f is compared against the released link counts
  S_od    no    needs the withheld reference path flow
  S_dev   no    needs how far the reference moved from the prior
  S_attr  no    needs the reference destination-attraction distribution

So this module reports S_link honestly and everything else as a labelled
*proxy*. The distinction matters: a proxy of the form 1 - |f - prior|/|prior|
scores "stay exactly on the prior" as perfect, which is the first of the two
degenerate answers S_dev exists to punish (spec §5). Tuning against it would
drive lambda up and land on the prior, feeling like progress while giving away
0.15 of the ODME weight.

The release has no base_od.csv, so the prior is the only anchor available and
there is nothing to blend it against; the sweep is over lambda alone.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import scipy.sparse as sp
from scipy.optimize import lsq_linear

from src.io import paths

# Each solve is a ~4400x4300 bounded least squares and costs roughly a minute, so
# the default grid is coarse on purpose. It can afford to be: S_link turns out to
# sit at ~0.99 for every lambda in this range, so a finer sweep buys no signal.
DEFAULT_LAMBDAS = (0.05, 1.0, 20.0)


def _incidence(panel: str) -> tuple[sp.csr_matrix, list[str], list[str]]:
    """Sparse link-by-path incidence matrix A, with its row and column labels."""
    inc = pl.read_csv(paths.network_dir(panel) / "path_link_incidence.csv")
    link_ids = inc["link_id"].unique(maintain_order=True).to_list()
    path_ids = (
        pl.read_csv(paths.network_dir(panel) / "path_set.csv")["path_id"]
        .unique(maintain_order=True)
        .to_list()
    )
    link_pos = {v: i for i, v in enumerate(link_ids)}
    path_pos = {v: i for i, v in enumerate(path_ids)}

    rows = inc["link_id"].replace_strict(link_pos, return_dtype=pl.Int64).to_numpy()
    cols = inc["path_id"].replace_strict(path_pos, return_dtype=pl.Int64, default=-1).to_numpy()
    keep = cols >= 0
    data = np.ones(keep.sum())
    A = sp.csr_matrix(
        (data, (rows[keep], cols[keep])), shape=(len(link_ids), len(path_ids))
    )
    return A, link_ids, path_ids


def solve(
    A: sp.csr_matrix,
    counts: np.ndarray,
    measured: np.ndarray,
    anchor: np.ndarray,
    lam: float,
) -> np.ndarray:
    """min ||A f - c||^2 + lambda ||f - anchor||^2  subject to  f >= 0."""
    n_paths = A.shape[1]
    design = sp.vstack([A[measured], np.sqrt(lam) * sp.identity(n_paths, format="csr")]).tocsr()
    target = np.concatenate([counts[measured], np.sqrt(lam) * anchor])
    fit = lsq_linear(design, target, bounds=(0.0, np.inf), tol=1e-8, max_iter=200)
    return fit.x


def sweep_panel(panel: str, split: str = "validation", lambdas=DEFAULT_LAMBDAS) -> pl.DataFrame:
    A, link_ids, path_ids = _incidence(panel)

    counts_df = pl.read_csv(paths.task4_dir(panel, split) / "synthetic_link_counts.csv")
    counts_map = dict(zip(counts_df["link_id"], counts_df["count"]))
    counts = np.array([counts_map.get(link, 0.0) for link in link_ids])
    measured = np.array([link in counts_map for link in link_ids])

    prior_df = pl.read_csv(paths.task4_dir(panel, split) / "synthetic_weak_prior.csv")
    prior_map = dict(zip(prior_df["path_id"], prior_df["path_flow"]))
    prior = np.array([prior_map.get(p, 0.0) for p in path_ids])

    rows = []
    for lam in lambdas:
        f = solve(A, counts, measured, prior, lam)
        loaded = A @ f

        # Real: released link counts are ground truth for this term.
        link_err = np.abs(loaded[measured] - counts[measured]).sum()
        s_link = max(0.0, 1.0 - link_err / max(counts[measured].sum(), 1e-9))

        # Diagnostic only — how far the solution travelled from the prior.
        # NOT S_od and NOT S_dev; the reference those compare against is withheld.
        deviation = np.abs(f - prior).sum() / max(prior.sum(), 1e-9)

        rows.append(
            {
                "panel": panel,
                "lambda": lam,
                "s_link": s_link,
                "deviation_from_prior": deviation,
                "total_flow": float(f.sum()),
                "prior_flow": float(prior.sum()),
                "n_paths": len(path_ids),
                "n_measured_links": int(measured.sum()),
            }
        )
    return pl.DataFrame(rows)


def run(panels=None, split: str = "validation", lambdas=DEFAULT_LAMBDAS) -> pl.DataFrame:
    panels = panels or paths.PANELS
    out = pl.concat([sweep_panel(p, split, lambdas) for p in panels], how="vertical")

    print("=== Task 4: lambda sweep (S_link is real; deviation is a diagnostic) ===")
    summary = (
        out.group_by("lambda")
        .agg(
            pl.col("s_link").mean().alias("s_link_mean"),
            pl.col("deviation_from_prior").mean().alias("deviation_mean"),
        )
        .sort("lambda")
    )
    print(summary)
    print(
        "\nReading this: small lambda fits the link counts almost exactly and moves far\n"
        "from the prior; large lambda stays near the prior and fits counts worse.\n"
        "S_od / S_dev / S_attr cannot be evaluated here, so the balance between them\n"
        "has to be settled on the leaderboard, not in this table."
    )
    return out
