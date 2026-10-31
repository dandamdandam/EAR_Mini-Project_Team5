"""Run the EDA diagnostics.

    conda activate easypj
    python -m src.eda.run                  # all three sections
    python -m src.eda.run --only task2
    python -m src.eda.run --only task4 --panels D7_I10_E D7_I10_W
"""

from __future__ import annotations

import argparse
import time

from src.eda import task1_fd, task2_queue, task4_odme
from src.io import paths

SECTIONS = ("task4", "task2", "task1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=SECTIONS, default=None)
    parser.add_argument("--panels", nargs="+", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument(
        "--months", nargs="+", default=["2030-06"], help="train months for the FD section"
    )
    args = parser.parse_args()

    sections = [args.only] if args.only else list(SECTIONS)

    for name in sections:
        started = time.perf_counter()
        print("\n" + "=" * 78)
        if name == "task4":
            panels = args.panels or list(paths.PANELS)
            task4_odme.run(panels=panels, split=args.split)
        elif name == "task2":
            panels = args.panels or list(paths.TASK2_PANELS)
            task2_queue.run(panels=panels, split=args.split)
        elif name == "task1":
            panels = args.panels or list(paths.PANELS)
            task1_fd.run(panels=panels, months=args.months)
        print(f"\n[{name} finished in {time.perf_counter() - started:.1f}s]")


if __name__ == "__main__":
    main()
