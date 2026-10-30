"""Reproduce the two Task 1 upgrades and at most three controlled submissions.

All three payloads and the January ensemble choice are frozen before the first
upload. Re-running submit checks an existing receipt instead of uploading twice.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from src.io import paths
from src.preprocess.run import sha256
from src.submit import kaggle_submit, multitask
from src.task1.upgrade import CONFIG, write_json
from src.task1.upgrade_ensemble import run as make_ensemble


def descriptors(common):
    result = []
    for kind, cfg in common["models"].items():
        root = Path(cfg["run"])
        selected = json.loads((root / "local/selection_best.json").read_text())
        label = f"{kind} {cfg['window']} steps, epoch {selected['epoch']}, residual {selected['blend']}"
        result.append({"kind": kind, "run": root, "out": Path(cfg["submission"]),
                       "label": label, "local_selection": selected,
                       "state_model": "Interpolation + " + label + "; final refit on all nine training months"})
    ens = json.loads((Path(common["ensemble_run"]) / "selection_best.json").read_text())
    label = f"ensemble STAEformer={ens['staeformer_weight']}, ImputeFormer={ens['imputeformer_weight']}"
    result.append({"kind": "ensemble", "run": Path(common["ensemble_run"]),
                   "out": Path(common["ensemble_submission"]), "label": label,
                   "local_selection": ens, "state_model": label + "; components refitted on all nine train months"})
    if len(result) != 3 or common["max_additional_submissions"] != 3:
        raise ValueError("This experiment is authorized for exactly three candidate payloads")
    if len({str(r["out"]) for r in result}) != 3:
        raise ValueError("Submission destinations must be distinct")
    return result


def verify_manifest(manifest, config):
    if manifest["config_sha256"] != sha256(config):
        raise ValueError("The experiment config changed after payloads were frozen")
    if len(manifest["payloads"]) != 3 or manifest["max_additional_submissions"] != 3:
        raise ValueError("Invalid three-submission budget")
    for item in manifest["payloads"]:
        out = Path(item["directory"])
        meta = json.loads((out / "metadata.json").read_text())
        if sha256(out / "submission.csv") != item["sha256"] or meta["submission_sha256"] != item["sha256"]:
            raise ValueError(f"Frozen payload changed: {out}")
        if not meta["controlled_comparison"]["queue_and_odme_identical"]:
            raise ValueError("Missing frozen-task verification")
    return manifest


def prepare(common, config, manifest_file):
    if manifest_file.exists():
        return verify_manifest(json.loads(manifest_file.read_text()), config)
    make_ensemble(common)
    payloads = []
    for spec in descriptors(common):
        out = spec["out"]
        context = {"state_model": spec["state_model"], "submission_description": spec["label"] + "; queue/ODME fixed from 003",
                   "config": str(config), "config_sha256": sha256(config),
                   "selection": spec["local_selection"], "max_additional_submissions": 3,
                   "reproduce": f"conda run --no-capture-output -n easypj python -m src.submit.run_task1_upgrade --config {config}",
                   "reproduction_note": "Preserve submitted files. For a fresh run copy the config and change output paths. Warm seeds and frozen Task 2/4 artifacts are required; see docs/task1-upgrade-004-006.md."}
        if not (out / "metadata.json").exists():
            multitask.run(out, Path(common["cache"]), {"state": spec["run"], "queue": Path(common["fixed_queue_run"]),
                          "odme": Path(common["fixed_odme_run"])}, context=context, reference=Path(common["reference_submission"]))
        metadata = json.loads((out / "metadata.json").read_text())
        if metadata.get("experiment", {}).get("config_sha256") != sha256(config):
            raise ValueError(f"Existing candidate was created with a different config: {out}")
        payloads.append({"kind": spec["kind"], "directory": str(out), "sha256": metadata["submission_sha256"],
                         "rows": metadata["rows"]})
    if len({p["sha256"] for p in payloads}) != 3:
        raise ValueError("Refusing to spend submissions on duplicate payloads")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "config_sha256": sha256(config),
                "max_additional_submissions": 3, "reference_submission": common["reference_submission"],
                "payloads": payloads, "all_payloads_prepared_before_upload": True}
    verify_manifest(manifest, config)
    write_json(manifest_file, manifest)
    return manifest


def submit_or_status(manifest, config, *, status_only):
    verify_manifest(manifest, config)
    results = []
    for item in manifest["payloads"]:
        out = Path(item["directory"])
        attempted = (out / "upload_intent.json").exists()
        if status_only and not attempted:
            results.append({"directory": str(out), "status": "NOT_SUBMITTED"})
            continue
        result = kaggle_submit.run(out, status_only=status_only or attempted)
        results.append({"directory": str(out), **result})
    return results


def wait_for_training(common):
    """Wait for each model's final completion artifacts before preparation."""
    required = ["local/SUCCESS.json", "final/SUCCESS.json", "selection_result.json", "february_result.json",
                "predictions_selection.parquet", "predictions_february.parquet", "predictions_validation.parquet",
                "predictions_private.parquet", "prediction_diagnostics.json"]
    while True:
        missing = {kind: [name for name in required if not (Path(cfg["run"]) / name).exists()]
                   for kind, cfg in common["models"].items()}
        if not any(missing.values()):
            return
        for kind, cfg in common["models"].items():
            log = Path(cfg["run"]) / "run.log"
            if missing[kind] and log.exists() and "Traceback (most recent call last)" in log.read_text():
                raise RuntimeError(f"Training failed for {kind}; inspect {log}. No upload was attempted.")
        print("Waiting for completed model artifacts:", {k: len(v) for k, v in missing.items()}, flush=True)
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--stages", nargs="+", choices=("train", "prepare", "submit", "status"), default=["train", "prepare"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wait-for-training", action="store_true", help="Wait for separately launched model runs before prepare")
    args = parser.parse_args()
    common = json.loads(args.config.read_text())
    experiment_root = Path(common["ensemble_run"])
    manifest_file = experiment_root / "submission_manifest.json"
    for stage in args.stages:
        if stage == "train":
            for kind in common["models"]:
                command = [sys.executable, "-m", "src.task1.upgrade", "--model", kind, "--config", str(args.config)]
                if args.resume:
                    command.append("--resume")
                subprocess.run(command, cwd=paths.PROJECT_ROOT, check=True)
        elif stage == "prepare":
            if args.wait_for_training:
                wait_for_training(common)
            prepare(common, args.config, manifest_file)
        else:
            manifest = json.loads(manifest_file.read_text())
            results = submit_or_status(manifest, args.config, status_only=stage == "status")
            write_json(experiment_root / "kaggle_results.json", results)


if __name__ == "__main__":
    main()
