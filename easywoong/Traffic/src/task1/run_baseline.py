"""Fit the historical-profile baseline and score it locally on train.

    conda activate easypj
    python -m src.task1.run_baseline --panels D7_I10_E
    python -m src.task1.run_baseline --all
"""

from __future__ import annotations

import argparse
import json
import time

from src.eval import score_task1
from src.io import paths
from src.task1 import baseline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", nargs="+", default=["D7_I10_E"])
    parser.add_argument("--all", action="store_true", help="score all 10 panels")
    parser.add_argument("--out", type=str, default=None, help="write per-panel scores as JSON")
    args = parser.parse_args()

    panels = list(paths.PANELS) if args.all else args.panels

    results = []
    for panel in panels:
        started = time.perf_counter()
        predictions = baseline.fit_predict(panel, "train")
        result = score_task1.score_panel(predictions, panel, "train")
        result["seconds"] = round(time.perf_counter() - started, 1)
        results.append(result)
        print(
            f"{panel:<12} S_state={result['s_state']:.4f}  "
            f"({result['seconds']}s, {sum(r['n'] for r in result['per_regime'].values()):,} cells)",
            flush=True,
        )

    overall = score_task1.aggregate_panels({r["panel"]: r["s_state"] for r in results})
    print()
    print(score_task1.format_report(results, overall if len(results) > 1 else None))

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"panels": results, "overall": overall}, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
