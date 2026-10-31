"""One entry point for preprocessing, fitting and validated submission generation.

Run in easypj. Default generates a CSV without uploading it. Choose a new --id
to reproduce without overwriting this run's checkpoints or submitted payload.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

from src.io import paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", default="003")
    parser.add_argument("--cache", type=Path, default=paths.CACHE_ROOT / "preprocessed_v1")
    parser.add_argument("--stages", nargs="+", choices=("preprocess", "state", "queue", "odme", "assemble"),
                        default=["preprocess", "state", "queue", "odme", "assemble"])
    parser.add_argument("--predict-only", action="store_true", help="For state/queue, use existing final checkpoints")
    parser.add_argument("--submit", action="store_true", help="Upload the validated CSV to Kaggle once")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9]{3}", args.id):
        parser.error("--id must be three digits")
    state, queue, odme = (Path("runs") / name for name in (
        f"staeformer_final_{args.id}", f"queue_catboost_{args.id}", f"odme_{args.id}"))
    out = Path("submissions") / f"{args.id}_multitask"
    settings = json.loads((paths.PROJECT_ROOT / "configs/multitask_003.json").read_text())
    commands = {
        "preprocess": ["src.preprocess.run", "--out", str(args.cache)],
        "state": ["src.task1.final_staeformer", "--out", str(state), "--cache", str(args.cache),
                  "--epochs", str(settings["task1"]["epochs"])],
        "queue": ["src.task2.catboost_queue", "--out", str(queue), "--cache", str(args.cache),
                  "--iterations", str(settings["task2"]["max_iterations"])],
        "odme": ["src.task4.regularized_odme", "--out", str(odme), "--cache", str(args.cache),
                 "--lam", str(settings["task4"]["lambda"])],
        "assemble": ["src.submit.multitask", "--out", str(out), "--cache", str(args.cache),
                     "--state", str(state), "--queue", str(queue), "--odme", str(odme)],
    }
    if args.predict_only:
        commands["state"] += ["--mode", "predict"]
        commands["queue"] += ["--predict-only"]
    for stage in args.stages:
        if stage == "state" and not args.predict_only and (paths.PROJECT_ROOT / state / "last.pt").exists():
            raise FileExistsError(f"Preserving existing checkpoint: {state}; choose a new --id")
        if stage == "queue" and not args.predict_only and (paths.PROJECT_ROOT / queue / "selection.cbm").exists():
            raise FileExistsError(f"Preserving existing checkpoint: {queue}; choose a new --id")
        command = [sys.executable, "-m", *commands[stage]]
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, cwd=paths.PROJECT_ROOT, check=True)
    if "assemble" in args.stages:
        print(f"Validated upload file: {out / 'submission.csv'}", flush=True)
    if args.submit:
        subprocess.run([sys.executable, "-m", "src.submit.kaggle_submit", "--out", str(out)],
                       cwd=paths.PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
