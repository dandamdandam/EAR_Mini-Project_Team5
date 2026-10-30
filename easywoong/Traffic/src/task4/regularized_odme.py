"""Fit public link counts with nonnegative, prior-regularized path flows.

The fixed lambda=0.05 follows the organizer's published NNLS baseline.
Only S_link can be computed locally; the reference-dependent terms are hidden.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("POLARS_MAX_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import polars as pl
import scipy.sparse as sp
from scipy.linalg import solve as solve_linear

from src.io import paths


def solve(A, counts, prior, lam=0.05):
    """Minimize ||A f-c||² + lam*||f-prior||² with f>=0, checking convergence."""
    A = sp.csr_matrix(A, dtype=np.float64)
    counts, prior = np.asarray(counts, float), np.asarray(prior, float)
    if A.shape != (len(counts), len(prior)) or lam <= 0:
        raise ValueError("Invalid NNLS dimensions or regularization")
    if not np.isfinite(counts).all() or not np.isfinite(prior).all() or (counts < 0).any() or (prior < 0).any():
        raise ValueError("Counts and prior must be finite and nonnegative")
    transpose = A.T.tocsr()

    def objective(x):
        residual, deviation = A @ x - counts, x - prior
        value = 0.5 * (residual @ residual + lam * (deviation @ deviation))
        return float(value), transpose @ residual + lam * deviation

    # Solve the smooth, strongly convex dual in measured-link dimensions, rather
    # than the much larger path space. Eliminating f gives f=max(0,b-A.T@y/lam).
    # Its generalized Hessian is I + A_active @ A_active.T / lam. Armijo search
    # handles changes of the active nonnegative constraints.
    def dual(y):
        f = np.maximum(prior - transpose @ y / lam, 0.0)
        value = 0.5 * (y @ y) + counts @ y + 0.5 * lam * (f @ f)
        return float(value), y + counts - A @ f, f

    started = time.perf_counter()
    y = np.zeros(len(counts))
    converged = False
    for iteration in range(1, 201):
        value, gradient_dual, prediction = dual(y)
        if np.max(np.abs(gradient_dual)) < 1e-7:
            converged = True
            break
        active = A[:, prediction > 0]
        hessian = (active @ active.T).toarray() / lam + np.eye(len(counts))
        direction = solve_linear(hessian, -gradient_dual, assume_a="pos", check_finite=False)
        descent = float(gradient_dual @ direction)
        if descent >= 0:
            raise RuntimeError("NNLS dual Newton direction is not a descent direction")
        step = 1.0
        rounding = 5 * np.finfo(float).eps * max(abs(value), 1.0)
        for _ in range(60):
            if dual(y + step * direction)[0] <= value + 1e-4 * step * descent + rounding:
                break
            step *= 0.5
        else:
            raise RuntimeError("NNLS dual line search did not converge")
        y += step * direction
    dual_value, gradient_dual, prediction = dual(y)
    gradient = objective(prediction)[1]
    projected = np.where(prediction > 1e-8, gradient, np.minimum(gradient, 0))
    tolerance = max(1e-3, 1e-7 * float(np.max(np.abs(transpose @ counts))))
    final_value, _ = objective(prediction)
    prior_value, _ = objective(prior)
    if not converged or not np.isfinite(prediction).all() or (prediction < 0).any():
        raise RuntimeError(f"NNLS dual did not converge: gradient={np.max(np.abs(gradient_dual))}")
    if final_value > prior_value + 1e-6 or np.max(np.abs(projected)) > tolerance:
        raise RuntimeError("NNLS failed objective/KKT validation")
    return prediction, {"solver": "dual semismooth Newton with Armijo search", "success": converged,
                        "message": "dual gradient and primal nonnegative KKT conditions satisfied",
                        "iterations": iteration, "dual_gradient_inf": float(np.max(np.abs(gradient_dual))),
                        "primal_dual_gap": float(final_value + dual_value - 0.5 * lam * (prior @ prior)),
                        "objective_prior": 2 * prior_value, "objective_fit": 2 * final_value,
                        "projected_gradient_inf": float(np.max(np.abs(projected))),
                        "kkt_tolerance": tolerance, "seconds": round(time.perf_counter() - started, 3)}


def predict_panel(cache, panel, split, lam=0.05):
    root = cache / "task4" / panel / split
    count_table = pl.read_parquet(root / "synthetic_link_counts.parquet")
    prior_table = pl.read_parquet(root / "synthetic_weak_prior.parquet")
    targets = pl.read_parquet(root / "sample_submission_path_flow.parquet")
    path_table = pl.read_parquet(cache / "network" / panel / "path_set.parquet")
    incidence = pl.read_parquet(cache / "network" / panel / "path_link_incidence.parquet")
    if incidence.select(pl.struct("link_id", "path_id").n_unique()).item() != incidence.height:
        raise ValueError(f"Duplicate incidence entry: {panel}")
    parts, reports = [], []
    for (departure,), requested in targets.partition_by("departure_time", as_dict=True).items():
        keys = ["panel", "departure_time", "path_id", "origin_zone", "destination_zone"]
        aligned = requested.drop("path_flow").join(
            prior_table, on=keys, how="left", validate="1:1")
        if aligned["path_flow"].null_count() or aligned["path_id"].n_unique() != len(aligned):
            raise ValueError(f"Missing/duplicate path prior: {panel}/{split}")
        if aligned.select("path_id", "origin_zone", "destination_zone").join(
                path_table.select("path_id", "origin_zone", "destination_zone"),
                on=["path_id", "origin_zone", "destination_zone"], how="anti").height:
            raise ValueError(f"Path zone mapping does not match the network: {panel}")
        counts = count_table
        if "departure_time" in counts.columns:
            counts = counts.filter(pl.col("departure_time") == departure)
        elif targets["departure_time"].n_unique() != 1:
            raise ValueError("Multiple scenarios require scenario-specific link counts")
        if counts["link_id"].n_unique() != len(counts):
            raise ValueError(f"Duplicate measured link: {panel}/{split}")
        if counts["panel"].unique().to_list() != [panel] or aligned["panel"].unique().to_list() != [panel]:
            raise ValueError("Panel mismatch")
        links = pl.DataFrame({"link_id": counts["link_id"], "row": np.arange(len(counts))})
        columns = pl.DataFrame({"path_id": aligned["path_id"], "col": np.arange(len(aligned))})
        mapped = incidence.join(links, on="link_id", how="inner", validate="m:1").join(columns, on="path_id", how="inner", validate="m:1")
        if mapped["link_id"].n_unique() != len(counts):
            raise ValueError(f"Measured link has no path coverage: {panel}")
        A = sp.csr_matrix((np.ones(len(mapped)), (mapped["row"].to_numpy(), mapped["col"].to_numpy())),
                          shape=(len(counts), len(aligned)))
        c, prior = counts["count"].to_numpy(), aligned["path_flow"].to_numpy()
        estimate, diagnostic = solve(A, c, prior, lam)
        denominator = max(float(c.sum()), 1e-9)
        s_link = lambda f: max(0.0, 1.0 - float(np.abs(A @ f - c).sum()) / denominator)
        reports.append({"panel": panel, "split": split, "departure_time": departure,
                        "lambda": lam, "paths": len(prior), "measured_links": len(c),
                        "s_link_prior": s_link(prior), "s_link_fit": s_link(estimate),
                        "relative_change_from_prior": float(np.abs(estimate - prior).sum() / max(prior.sum(), 1e-9)),
                        "minimum_path_flow": float(estimate.min()), **diagnostic})
        parts.append(aligned.select(keys).with_columns(pl.Series("path_flow", estimate)))
    return pl.concat(parts), reports


def run(out, cache, lam=0.05):
    out.mkdir(parents=True, exist_ok=True)
    reports = []
    for split in ("validation", "private"):
        parts = []
        for panel in paths.PANELS:
            prediction, diagnostics = predict_panel(cache, panel, split, lam)
            parts.append(prediction)
            reports.extend(diagnostics)
            print(f"odme {split}/{panel}: {len(prediction):,} paths, S_link={diagnostics[0]['s_link_fit']:.6f}", flush=True)
        pl.concat(parts).write_parquet(out / f"predictions_{split}.parquet")
    result = {"method": "nonnegative least squares + weak-prior L2 regularization", "lambda": lam,
              "lambda_source": "Organizer published BASELINES.md, fixed before submission",
              "source_url": "https://github.com/jacky850/trafficflowbench-public/blob/main/docs/BASELINES.md",
              "metric_note": "S_link only; S_od, S_dev, S_attr and complete S_ODME require hidden reference labels",
              "by_panel_split": reports}
    (out / "diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/odme_003"))
    parser.add_argument("--cache", type=Path, default=paths.CACHE_ROOT / "preprocessed_v1")
    parser.add_argument("--lam", type=float, default=0.05)
    args = parser.parse_args()
    run(args.out, args.cache, args.lam)


if __name__ == "__main__":
    main()
