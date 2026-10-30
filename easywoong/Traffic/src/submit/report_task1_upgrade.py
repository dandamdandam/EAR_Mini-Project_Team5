"""Collect saved local metrics and actual Kaggle receipts; plot local comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def run(config):
    common = read(config)
    root = Path(common["ensemble_run"])
    previous = read("runs/staeformer_residual_001/test_result.json")
    previous_score = previous["test"][str(previous["selected_blend"])]["s_state"]
    baseline_public = float(read(Path(common["reference_submission"]) / "result.json")["public_score"])
    rows = [{"name": "Previous STAEformer", "epoch": 5, "selection_s_state": previous["selection_s_state"],
             "february_s_state": previous_score, "february_delta": 0., "public_score": baseline_public}]
    weight = read(root / "selection_best.json")["staeformer_weight"]
    ensemble_label = f"{weight:.0%} STAE + {1 - weight:.0%} Impute"
    entries = [("Improved STAEformer", common["models"]["staeformer"]),
               ("ImputeFormer", common["models"]["imputeformer"]),
               (ensemble_label, {"run": common["ensemble_run"], "submission": common["ensemble_submission"]})]
    for name, cfg in entries:
        directory = Path(cfg["run"])
        jan, feb = read(directory / "selection_result.json"), read(directory / "february_result.json")
        row = {"name": name, "run": str(directory), "epoch": jan.get("epoch"),
               "selection_s_state": jan["score"]["s_state"], "february_s_state": feb["score"]["s_state"],
               "february_delta": feb["score"]["s_state"] - previous_score,
               "january_targets": jan["score"]["scored_cells"], "february_targets": feb["score"]["scored_cells"],
               "february_regimes": {r: sum(x["s_state"] for x in feb["score"]["rows"] if x["regime"] == r) / 10
                                    for r in ("R1", "R2", "R3")}}
        result_file = Path(cfg["submission"]) / "result.json"
        if result_file.exists():
            result = read(result_file)
            row.update(kaggle_ref=result.get("ref"), kaggle_status=result["status"], public_score=result.get("public_score"))
            if result.get("public_score") is not None:
                row["public_score"] = float(result["public_score"])
                row["public_delta_from_003"] = row["public_score"] - baseline_public
        rows.append(row)
    # Preserve the local-only report already included in submission snapshots.
    filename = "comparison_with_kaggle.json" if any("kaggle_status" in r for r in rows) else "comparison.json"
    (root / filename).write_text(json.dumps(rows, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    for kind, label, color in (("staeformer", "STAEformer (5-epoch warm start)", "#2563eb"),
                               ("imputeformer", "ImputeFormer", "#ea580c")):
        history = read(Path(common["models"][kind]["run"]) / "local/history.json")
        scores = [max(v for k, v in r["selection_scores"].items() if k != "0.0") for r in history]
        axes[0].plot([r["epoch"] for r in history], scores, marker=".", label=label, color=color)
    axes[0].set(title="January model selection", xlabel="Cumulative epoch", ylabel="Task 1 S_state")
    axes[0].set_xticks([1, 5, 10, 15, 20])
    axes[0].grid(alpha=.2)
    axes[0].legend(fontsize=8)
    values = [r["february_s_state"] for r in rows]
    axes[1].scatter(values, range(4), s=65, c=["#9ca3af", "#2563eb", "#ea580c", "#16a34a"])
    axes[1].set_yticks(range(4), [r["name"] for r in rows])
    axes[1].invert_yaxis()
    axes[1].set_ylim(3.3, -0.35)
    axes[1].set(title="February reporting set", xlabel="Task 1 S_state (higher is better)", xlim=(.930, .953))
    axes[1].grid(axis="x", alpha=.2)
    for y, value in enumerate(values):
        axes[1].annotate(f"{value:.6f}", (value, y), xytext=(5, 8), textcoords="offset points", fontsize=8)
    fig.savefig(root / "local_comparison.png", dpi=160)
    plt.close(fig)
    print(json.dumps(rows, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/task1_upgrade_004_006.json"))
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
