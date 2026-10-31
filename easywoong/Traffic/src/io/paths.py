"""Path helpers for the TrafficFlowBench 2026 data release.

Data root is read from the TFB_DATA_ROOT environment variable so the same code
runs unchanged on another machine or in a Kaggle notebook (CLAUDE.md §6).
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("TFB_DATA_ROOT", PROJECT_ROOT / "data" / "kaggle_public"))
CACHE_ROOT = Path(os.environ.get("TFB_CACHE_ROOT", PROJECT_ROOT / "cache"))

SPLITS = ("train", "validation", "private")
REGIMES = ("R1", "R2", "R3")

# 5 corridor families x 2 directions. Scoring averages the two directions of a
# family first, then averages the 5 families equally (spec §1).
CORRIDOR_FAMILIES: dict[str, tuple[str, str]] = {
    "D7_I10": ("D7_I10_E", "D7_I10_W"),
    "D7_I210": ("D7_I210_E", "D7_I210_W"),
    "D7_I405": ("D7_I405_N", "D7_I405_S"),
    "D12_I5": ("D12_I5_N", "D12_I5_S"),
    "D12_I405": ("D12_I405_N", "D12_I405_S"),
}
PANELS: tuple[str, ...] = tuple(p for pair in CORRIDOR_FAMILIES.values() for p in pair)

# D12_I405 has no Task 2 windows; those panels are scored on tasks 1/3/4 only.
TASK2_PANELS: tuple[str, ...] = tuple(p for p in PANELS if not p.startswith("D12_I405"))


def corridor_dir(panel: str, split: str) -> Path:
    return DATA_ROOT / "corridors" / panel / split


def network_dir(panel: str) -> Path:
    return DATA_ROOT / "corridors" / panel / "network"


def mainline_unmasked_dir(panel: str) -> Path:
    """Ground truth. Only exists for train."""
    return DATA_ROOT / "corridors" / panel / "train" / "mainline_states"


def mainline_masked_dir(panel: str, split: str) -> Path:
    return corridor_dir(panel, split) / "mainline_states_masked"


def ramp_dir(panel: str, split: str) -> Path:
    return corridor_dir(panel, split) / "ramp_states"


def task1_targets(panel: str, split: str) -> Path:
    """The authoritative Task 1 target list — never derive the mask yourself."""
    return DATA_ROOT / "task1" / panel / split / "sample_submission_state.csv"


def task2_dir(panel: str, split: str) -> Path:
    return DATA_ROOT / "task2" / panel / split


def task4_dir(panel: str, split: str) -> Path:
    return DATA_ROOT / "task4" / panel / split


SUBMISSION_KEY = DATA_ROOT / "submission_key.csv"
SAMPLE_SUBMISSION = DATA_ROOT / "sample_submission.csv"
