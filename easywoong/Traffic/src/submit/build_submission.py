"""Assemble the single upload CSV from per-task predictions, then validate it.

    conda activate easypj
    python -m src.submit.build_submission --out submissions/001_baseline

The evaluator rejects a file outright for a missing cell or an unknown
submission_id, and zeroes a whole task if even one of its required rows is
absent, so this writes every row of the template and checks that invariant
before the file is allowed to exist.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import polars as pl

from src.io import loaders, paths
from src.task1 import baseline as task1_baseline
from src.task1 import interpolate as task1_interpolate
from src.task2 import baseline as task2_baseline
from src.task4 import baseline as task4_baseline

OUTPUT_COLUMNS = ("submission_id", "task", "speed_kmh", "flow_vph", "queue_pred", "path_flow")
PREDICT_SPLITS = ("validation", "private")

TASK1_METHODS = {
    "profile": (task1_baseline.fit_predict, "historical profile fitted on the masked layer"),
    "interpolate": (
        task1_interpolate.fit_predict,
        "spatio-temporal interpolation (0.85 temporal / 0.15 spatial), profile fallback",
    ),
}


def _key(task: str, columns: list[str]) -> pl.DataFrame:
    return (
        loaders.scan_submission_key(task=task)
        .select(["submission_id", "split"] + columns)
        .collect()
    )


def build_state(method: str = "interpolate") -> pl.DataFrame:
    fit_predict, _ = TASK1_METHODS[method]
    key = _key("state", ["panel", "timestamp", "station_id", "link_id"]).with_columns(
        pl.col("timestamp").str.to_datetime(format="%Y-%m-%dT%H:%M:%SZ", time_unit="us")
    )

    predictions = []
    for split in PREDICT_SPLITS:
        for panel in paths.PANELS:
            predictions.append(
                fit_predict(panel, split).with_columns(
                    pl.lit(panel).alias("panel"), pl.lit(split).alias("split")
                )
            )
    pred = pl.concat(predictions, how="vertical")

    return key.join(
        pred, on=["split", "panel", "timestamp", "station_id", "link_id"], how="left"
    ).select("submission_id", "speed_kmh", "flow_vph")


def build_queue() -> pl.DataFrame:
    key = _key("queue", ["panel", "window_id", "timestamp", "link_id"])

    predictions = []
    for split in PREDICT_SPLITS:
        for panel in paths.TASK2_PANELS:
            predictions.append(
                task2_baseline.predict(panel, split).with_columns(pl.lit(split).alias("split"))
            )
    pred = pl.concat(predictions, how="vertical")

    return key.join(
        pred, on=["split", "window_id", "timestamp", "link_id"], how="left"
    ).select("submission_id", "queue_pred")


def build_odme() -> pl.DataFrame:
    key = _key("odme", ["panel", "departure_time", "path_id"])

    predictions = []
    for split in PREDICT_SPLITS:
        for panel in paths.PANELS:
            predictions.append(
                task4_baseline.predict(panel, split).with_columns(pl.lit(split).alias("split"))
            )
    pred = pl.concat(predictions, how="vertical")

    return key.join(
        pred, on=["split", "panel", "departure_time", "path_id"], how="left"
    ).select("submission_id", "path_flow")


def assemble(task1_method: str = "interpolate") -> pl.DataFrame:
    template = pl.read_csv(paths.SAMPLE_SUBMISSION, columns=["submission_id", "task"])

    return (
        template.join(build_state(task1_method), on="submission_id", how="left")
        .join(build_queue(), on="submission_id", how="left")
        .join(build_odme(), on="submission_id", how="left")
        .with_columns(
            pl.col("speed_kmh").fill_null(0.0),
            pl.col("flow_vph").fill_null(0.0),
            pl.col("queue_pred").fill_null(0).cast(pl.Int64),
            pl.col("path_flow").fill_null(0.0),
        )
        .select(OUTPUT_COLUMNS)
    )


def validate(submission: pl.DataFrame) -> dict:
    """Raise on anything the evaluator would reject; return a summary otherwise."""
    template = pl.read_csv(paths.SAMPLE_SUBMISSION, columns=["submission_id", "task"])

    problems = []
    if submission.height != template.height:
        problems.append(f"row count {submission.height} != template {template.height}")
    if submission["submission_id"].n_unique() != submission.height:
        problems.append("duplicate submission_id")
    if not submission["submission_id"].sort().equals(template["submission_id"].sort()):
        problems.append("submission_id set differs from template")

    nulls = {c: n for c, n in zip(submission.columns, submission.null_count().row(0)) if n}
    if nulls:
        problems.append(f"null cells: {nulls}")

    bad_queue = submission.filter(~pl.col("queue_pred").is_in([0, 1])).height
    if bad_queue:
        problems.append(f"{bad_queue} queue_pred values outside {{0, 1}}")

    for column in ("speed_kmh", "flow_vph", "path_flow"):
        bad = submission.filter(
            pl.col(column).is_infinite() | pl.col(column).is_nan() | (pl.col(column) < 0)
        ).height
        if bad:
            problems.append(f"{bad} {column} values non-finite or negative")

    # Every task must be complete; a single missing row zeroes the whole task.
    filled = submission.join(
        loaders.scan_submission_key().select("submission_id", "task").collect(),
        on="submission_id",
        how="left",
        suffix="_key",
    )
    coverage = {}
    for task, column in (("state", "speed_kmh"), ("queue", "queue_pred"), ("odme", "path_flow")):
        rows = filled.filter(pl.col("task_key") == task)
        nonzero = rows.filter(pl.col(column) != 0).height
        coverage[task] = {"rows": rows.height, "nonzero": nonzero}
        if rows.height and nonzero == 0:
            problems.append(f"task '{task}' is entirely zero — it would score 0")

    if problems:
        raise ValueError("submission failed validation:\n  - " + "\n  - ".join(problems))
    return coverage


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="submissions/<id> directory")
    parser.add_argument("--task1", choices=sorted(TASK1_METHODS), default="interpolate")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    submission = assemble(args.task1)
    coverage = validate(submission)

    path = out_dir / "submission.csv"
    submission.write_csv(path)

    # Reproducibility snapshot — required before any prize is confirmed (CLAUDE.md §6).
    metadata = {
        "git_commit": git_commit(),
        "rows": submission.height,
        "coverage": coverage,
        "task1": TASK1_METHODS[args.task1][1],
        "task2": "persistence from last observed speed, threshold 0.60 * free-flow",
        "task4": "released weak prior, unchanged",
        "seconds": round(time.perf_counter() - started, 1),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"wrote {path}  ({submission.height:,} rows, {path.stat().st_size / 1e6:.0f} MB)")
    for task, stats in coverage.items():
        print(f"  {task:<6} {stats['rows']:>9,} rows  {stats['nonzero']:>9,} non-zero")


if __name__ == "__main__":
    main()
