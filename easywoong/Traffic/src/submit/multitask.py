"""Strictly join three complete prediction tables to the official upload template."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault("POLARS_MAX_THREADS", "8")

import polars as pl

from src.io import loaders, paths
from src.preprocess.run import sha256

OUTPUT_COLUMNS = ["submission_id", "task", "speed_kmh", "flow_vph", "queue_pred", "path_flow"]
TASK_SCHEMA = {
    "state": (["split", "panel", "timestamp", "station_id", "link_id", "mask_regime"], ["speed_kmh", "flow_vph"]),
    "queue": (["split", "panel", "window_id", "timestamp", "link_id"], ["queue_pred"]),
    "odme": (["split", "panel", "departure_time", "path_id", "origin_zone", "destination_zone"], ["path_flow"]),
}


def align_predictions(key, prediction, task):
    """Reject missing, extra or duplicate predictions BEFORE filling irrelevant fields."""
    keys, values = TASK_SCHEMA[task]
    key, prediction = key.select("submission_id", *keys), prediction.select(*keys, *values)
    for name, frame in (("key", key), ("prediction", prediction)):
        if any(frame.null_count().row(0)):
            raise ValueError(f"{task}: null cells in {name}")
        if frame.select(pl.struct(keys).n_unique()).item() != frame.height:
            raise ValueError(f"{task}: duplicate {name} coordinates")
    if key["submission_id"].n_unique() != len(key):
        raise ValueError(f"{task}: duplicate submission_id")
    missing = key.join(prediction.select(keys), on=keys, how="anti").height
    extra = prediction.select(keys).join(key.select(keys), on=keys, how="anti").height
    if missing or extra:
        raise ValueError(f"{task}: {missing} missing and {extra} extra predictions")
    aligned = key.join(prediction, on=keys, how="left", validate="1:1").select("submission_id", *values)
    for column in values:
        if aligned.filter(~pl.col(column).is_finite() | (pl.col(column) < 0)).height:
            raise ValueError(f"{task}: invalid numeric prediction in {column}")
    if task == "queue" and aligned.filter(~pl.col("queue_pred").is_in([0, 1])).height:
        raise ValueError("queue: prediction must be 0 or 1")
    aligned = aligned.with_columns(pl.lit(task).alias("task"))
    for column in OUTPUT_COLUMNS[2:]:
        dtype = pl.Int64 if column == "queue_pred" else pl.Float64
        expr = pl.col(column) if column in values else pl.lit(0)
        aligned = aligned.with_columns(expr.cast(dtype).alias(column))
    return aligned.select(OUTPUT_COLUMNS)


def validate_frame(submission, template):
    if submission.columns != OUTPUT_COLUMNS:
        raise ValueError("Unexpected submission columns/order")
    if len(submission) != len(template) or submission["submission_id"].n_unique() != len(template):
        raise ValueError("Incorrect row count or duplicate IDs")
    if not submission.select("submission_id", "task").equals(template.select("submission_id", "task")):
        raise ValueError("Submission IDs, order, or task mapping differs from template")
    if any(submission.null_count().row(0)):
        raise ValueError("Submission contains nulls")
    for column in OUTPUT_COLUMNS[2:]:
        if submission.filter(~pl.col(column).is_finite() | (pl.col(column) < 0)).height:
            raise ValueError(f"Submission contains invalid {column}")
    if submission["queue_pred"].dtype not in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
        raise ValueError("queue_pred must be an integer column")
    if submission.filter(~pl.col("queue_pred").is_in([0, 1])).height:
        raise ValueError("queue_pred outside {0, 1}")
    coverage = {}
    for task, (_, relevant) in TASK_SCHEMA.items():
        group = submission.filter(pl.col("task") == task)
        for column in set(OUTPUT_COLUMNS[2:]) - set(relevant):
            if group.filter(pl.col(column) != 0).height:
                raise ValueError(f"Nonzero irrelevant column {column} in {task}")
        coverage[task] = {"rows": len(group), **{
            column: {"min": group[column].min(), "max": group[column].max(),
                     "nonzero": group.filter(pl.col(column) != 0).height} for column in relevant}}
    return coverage


def assemble(run_dirs):
    parts, coverage = [], []
    for task, directory in run_dirs.items():
        keys, values = TASK_SCHEMA[task]
        key = loaders.scan_submission_key(task=task).select("submission_id", *keys).collect()
        prediction = pl.concat([
            pl.read_parquet(directory / f"predictions_{split}.parquet").select(*[k for k in keys if k != "split"], *values)
            .with_columns(pl.lit(split).alias("split")) for split in ("validation", "private")], how="vertical_relaxed")
        parts.append(align_predictions(key, prediction, task))
        coverage.extend(key.group_by("split", "panel").len().with_columns(pl.lit(task).alias("task")).to_dicts())
        print(f"verified {task}: {len(key):,} complete predictions", flush=True)
    template = pl.read_csv(paths.SAMPLE_SUBMISSION, columns=["submission_id", "task"])
    predicted = pl.concat(parts)
    if predicted["submission_id"].n_unique() != len(predicted) or len(predicted) != len(template):
        raise ValueError("Combined task predictions have duplicate/missing IDs")
    result = template.join(predicted.rename({"task": "predicted_task"}),
                           on="submission_id", how="left", validate="1:1", maintain_order="left")
    if result.filter(pl.col("task") != pl.col("predicted_task")).height or result["predicted_task"].null_count():
        raise ValueError("Predicted task does not match official ID")
    result = result.select(OUTPUT_COLUMNS)
    return result, validate_frame(result, template), coverage


def snapshot(out, cache, run_dirs):
    source_hashes = {}
    files = list((paths.PROJECT_ROOT / "src").rglob("*.py"))
    files += list((paths.PROJECT_ROOT / "tests").glob("*.py"))
    files += [paths.PROJECT_ROOT / "requirements.txt"]
    files += list((paths.PROJECT_ROOT / "configs").glob("*.json"))
    files += list((paths.PROJECT_ROOT / "src").rglob("*LICENSE*"))
    for file in sorted(files):
        relative = file.relative_to(paths.PROJECT_ROOT)
        destination = out / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, destination)
        source_hashes[str(relative)] = sha256(file)
    input_hashes = {str(p.relative_to(paths.DATA_ROOT)): sha256(p) for p in (
        paths.SAMPLE_SUBMISSION, paths.SUBMISSION_KEY, paths.DATA_ROOT / "config/synthetic_release_v1.json")}
    # The daily manifest already hashes every original mainline parquet. Hash the
    # smaller network/task tables here, including the exact official target lists.
    for panel in paths.PANELS:
        small = list(paths.network_dir(panel).glob("*.csv"))
        for split in paths.SPLITS:
            small.append(paths.task1_targets(panel, split))
            if panel in paths.TASK2_PANELS:
                small += list(paths.task2_dir(panel, split).glob("*"))
            small += list(paths.task4_dir(panel, split).glob("*"))
        for file in small:
            if file.is_file():
                input_hashes[str(file.relative_to(paths.DATA_ROOT))] = sha256(file)
    (out / "source_hashes.json").write_text(json.dumps(source_hashes, indent=2) + "\n")
    (out / "input_hashes.json").write_text(json.dumps(input_hashes, indent=2) + "\n")
    artifacts = {}
    for task, directory in run_dirs.items():
        artifacts[task] = {"run_directory": str(directory), "files": {}}
        for file in sorted(directory.rglob("*")):
            if file.suffix in (".json", ".pt", ".cbm", ".parquet"):
                relative = file.relative_to(directory)
                artifacts[task]["files"][str(relative)] = {"bytes": file.stat().st_size, "sha256": sha256(file)}
                if file.suffix == ".json":
                    destination = out / "reports" / task / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file, destination)
    for name in ("SUCCESS.json", "quality_summary.csv", "manifest.parquet"):
        shutil.copy2(cache / name, out / name)
    historical = paths.PROJECT_ROOT / "runs/staeformer_residual_001"
    reference = out / "reports/historical_task1"
    reference.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "selection_best.json", "test_result.json"):
        shutil.copy2(historical / name, reference / name)
    return artifacts


def compare_frozen_tasks(submission, reference):
    """An exact check after CSV serialization of the frozen queue/ODME outputs."""
    previous = pl.read_csv(Path(reference) / "submission.csv")
    if not previous.select("submission_id", "task").equals(submission.select("submission_id", "task")):
        raise ValueError("Reference and candidate use different submission IDs/order")
    nonstate = pl.col("task") != "state"
    if not previous.filter(nonstate).equals(submission.filter(nonstate)):
        raise ValueError("Task 2 or Task 4 changed in a state-only comparison")
    a, b = previous.filter(~nonstate), submission.filter(~nonstate)
    changes = {col: {"changed_rows": int((a[col] != b[col]).sum()),
                     "mean_absolute_change": float((a[col] - b[col]).abs().mean())}
               for col in ("speed_kmh", "flow_vph")}
    if not any(v["changed_rows"] for v in changes.values()):
        raise ValueError("Candidate is identical to the previous submission")
    return {"reference": str(reference), "reference_sha256": sha256(Path(reference) / "submission.csv"),
            "queue_and_odme_identical": True, "unchanged_nonstate_rows": previous.filter(nonstate).height,
            "state_changes": changes}


def run(out, cache, run_dirs, *, context=None, reference=None):
    context = context or {}
    out.mkdir(parents=True, exist_ok=True)
    if (out / "submission.csv").exists():
        raise FileExistsError(f"Preserving existing submission: {out / 'submission.csv'}")
    started = time.perf_counter()
    submission, coverage, detailed_coverage = assemble(run_dirs)
    temporary = out / "submission.pending.csv"
    submission.write_csv(temporary)
    # Validate serialized data, not just the in-memory table.
    reread = pl.read_csv(temporary)
    validate_frame(reread, submission.select("submission_id", "task"))
    if not reread.equals(submission):
        raise ValueError("CSV serialization changed prediction values")
    controlled_comparison = compare_frozen_tasks(reread, reference) if reference else None
    temporary.rename(out / "submission.csv")
    artifacts = snapshot(out, cache, run_dirs)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.PROJECT_ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--short"], cwd=paths.PROJECT_ROOT,
                            check=True, capture_output=True, text=True).stdout
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
                "git_status": status, "rows": len(submission), "coverage": coverage,
                "coverage_by_panel_split": detailed_coverage,
                "submission_sha256": sha256(out / "submission.csv"),
                "preprocessing": json.loads((cache / "SUCCESS.json").read_text()),
                "models": {"task1": context.get("state_model", "Interpolation + STAEformer residual, 5 epochs, all nine train months"),
                           "task2": "CatBoost, causal 12-step history, six horizons; train-observation proxy labels",
                           "task3": "Same state predictions, no separate output or hidden-label calibration",
                           "task4": "Nonnegative least squares, weak-prior L2 lambda=0.05"},
                "artifacts": artifacts, "python": sys.version,
                "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "polars", "scipy", "torch", "catboost", "kaggle")},
                "seed": 42, "external_training_data": None,
                "metric_note": "Full-train refits have no independent February score. Queue local metrics use proxy labels.",
                "reproduce": context.get("reproduce", "conda run --no-capture-output -n easypj python -m src.submit.run_multitask --id 007"),
                "seconds": round(time.perf_counter() - started, 2)}
    if context:
        metadata["experiment"] = context
    if context.get("submission_description"):
        metadata["submission_description"] = context["submission_description"]
    if controlled_comparison:
        metadata["controlled_comparison"] = controlled_comparison
        (out / "verification.json").write_text(json.dumps(controlled_comparison, indent=2) + "\n")
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {out / 'submission.csv'}: {len(submission):,} validated rows", flush=True)
    print(json.dumps(coverage, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("submissions/003_multitask"))
    parser.add_argument("--cache", type=Path, default=paths.CACHE_ROOT / "preprocessed_v1")
    parser.add_argument("--state", type=Path, default=Path("runs/staeformer_final_003"))
    parser.add_argument("--queue", type=Path, default=Path("runs/queue_catboost_003"))
    parser.add_argument("--odme", type=Path, default=Path("runs/odme_003"))
    args = parser.parse_args()
    run(args.out, args.cache, {"state": args.state, "queue": args.queue, "odme": args.odme})


if __name__ == "__main__":
    main()
