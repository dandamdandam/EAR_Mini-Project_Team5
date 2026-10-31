"""Check Task 1 experiment coverage and print a like-for-like score table.

Run from Traffic after all experiments finish::

    python -m src.task1.compare_experiments

The comparison verifies that every model scored the same panel/regime target
counts before reporting an overall number.  Each model must have used its own
January selection; this script only reads frozen February results.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from src.io import paths


DEFAULT_RESULTS = {
    "STAEformer residual": Path("runs/staeformer_residual_001/test_result.json"),
    "CatBoost residual": Path("runs/catboost_residual_001/test_result.json"),
    "ImputeFormer": Path("runs/imputeformer_001/test_result.json"),
    "SPIN-H": Path("runs/spin_h_001/test_result.json"),
}


def _selected_summary(result: dict) -> dict:
    test = result["test"]
    blend = result.get("selected_blend")
    if blend is None:
        if "selected" in test:
            return test["selected"]
        raise ValueError("Result has no selected_blend or test.selected")
    for key, summary in test.items():
        try:
            if math.isclose(float(key), float(blend), abs_tol=1e-12):
                return summary
        except ValueError:
            continue
    raise ValueError(f"Selected blend {blend!r} is absent from test summaries")


def _validate(name: str, result: dict) -> tuple[dict, dict]:
    summary = _selected_summary(result)
    rows = summary["rows"]
    expected_keys = {(p, r) for p in paths.PANELS for r in paths.REGIMES}
    keyed = {(row["panel"], row["regime"]): row for row in rows}
    if len(rows) != len(keyed) or set(keyed) != expected_keys:
        raise ValueError(f"{name}: expected all 30 unique panel/regime rows")
    counts = {key: int(row["n"]) for key, row in keyed.items()}
    if any(n <= 0 for n in counts.values()):
        raise ValueError(f"{name}: empty panel/regime row")
    total = sum(counts.values())
    if total != int(result["test_target_rows"]) or total != int(summary["scored_cells"]):
        raise ValueError(f"{name}: scored target count mismatch")

    by_panel = {}
    for panel in paths.PANELS:
        scores = []
        for regime in paths.REGIMES:
            row = keyed[(panel, regime)]
            score = (
                0.54 * max(0.0, 1.0 - float(row["rmse_speed"]) / 25.0)
                + 0.46 * max(0.0, 1.0 - float(row["rmse_flow_per_lane"]) / 600.0)
            )
            if not math.isclose(score, float(row["s_state"]), abs_tol=1e-8):
                raise ValueError(f"{name}: score mismatch for {panel}/{regime}")
            scores.append(score)
        by_panel[panel] = sum(scores) / 3
    by_family = {
        family: sum(by_panel[p] for p in members) / len(members)
        for family, members in paths.CORRIDOR_FAMILIES.items()
    }
    overall = sum(by_family.values()) / len(by_family)
    if not math.isclose(overall, float(summary["s_state"]), abs_tol=1e-8):
        raise ValueError(f"{name}: overall score does not match regime/family aggregation")
    return summary, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/task1_model_comparison.json"))
    args = parser.parse_args()
    results = {}
    reference_counts = None
    reference_interpolation = None
    for name, file in DEFAULT_RESULTS.items():
        result = json.loads(file.read_text())
        summary, counts = _validate(name, result)
        if reference_counts is None:
            reference_counts = counts
        elif counts != reference_counts:
            raise ValueError(f"{name}: panel/regime target counts differ from STAEformer")
        if "0.0" in result["test"]:
            interpolation = float(result["test"]["0.0"]["s_state"])
            if reference_interpolation is None:
                reference_interpolation = interpolation
            elif not math.isclose(interpolation, reference_interpolation, abs_tol=1e-6):
                raise ValueError(f"{name}: interpolation-only score differs from STAEformer")
        results[name] = {
            "s_state": summary["s_state"],
            "interpolation_s_state": result["test"].get("0.0", {}).get("s_state"),
            "selected_epoch": result.get("selected_epoch"),
            "selected_blend": result.get("selected_blend"),
            "test_target_rows": result["test_target_rows"],
            "family_scores": summary["family_scores"],
            "regime_panel_means": {
                regime: sum(row["s_state"] for row in summary["rows"]
                            if row["regime"] == regime) / len(paths.PANELS)
                for regime in paths.REGIMES
            },
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Validated {next(iter(results.values()))['test_target_rows']:,} same official targets")
    print(f"{'Model':<24} {'S_state':>10} {'R1':>10} {'R2':>10} {'R3':>10}")
    for name, result in sorted(results.items(), key=lambda item: -item[1]["s_state"]):
        regime = result["regime_panel_means"]
        print(f"{name:<24} {result['s_state']:>10.6f} "
              f"{regime['R1']:>10.6f} {regime['R2']:>10.6f} {regime['R3']:>10.6f}")


if __name__ == "__main__":
    main()
