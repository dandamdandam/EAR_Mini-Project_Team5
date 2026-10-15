import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocessing import OUTPUT_DIR, PANELS, RELEASE_ROOT
from .task4_pseudo_label import PSEUDO_DIR, load_labels

SPLITS = ['train', 'validation', 'private']
SPLIT_STYLE = {
    'train': ('#2a78d6', '-'),
    'validation': ('#eb6834', '--'),
    'private': ('#4a3aa7', ':'),
}
PSEUDO_COLOR = '#898781'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e1e0d9'
BINS = 40


def pseudo_counts(lab):
    B = lab['B']
    return B[~np.isnan(B)]


def split_counts(release, panel):
    out = {}
    for split in SPLITS:
        p = release / 'task4' / panel / split / 'synthetic_link_counts.csv'
        if p.exists():
            out[split] = pd.read_csv(p)['count'].to_numpy(dtype=float)
    return out


def share_hist(ax, x, bins, **kw):
    ax.hist(x, bins=bins, weights=np.full(len(x), 100 / len(x)), **kw)


def plot_panel(ax, panel, pseudo, splits):
    bins = np.histogram_bin_edges(np.concatenate([pseudo, *splits.values()]), bins=BINS)
    share_hist(ax, pseudo, bins, color=PSEUDO_COLOR, alpha=0.45, edgecolor='white', linewidth=0.5,
               label='pseudo (day × link)')
    for split, x in splits.items():
        color, ls = SPLIT_STYLE[split]
        share_hist(ax, x, bins, histtype='step', color=color, linestyle=ls, linewidth=2, label=split)
    ax.set_title(f'{panel}  (pseudo n={len(pseudo):,}, split n={len(next(iter(splits.values())))})',
                 fontsize=9, color=INK, loc='left')
    ax.grid(axis='y', color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color('#c3c2b7')
    ax.tick_params(colors=MUTED, labelsize=8)


def run(release, panels, label_dir, out_path, ncols):
    data = []
    for panel in panels:
        lab = load_labels(label_dir, panel)
        splits = split_counts(release, panel)
        if lab is None or not splits:
            print(f'[{panel}] pseudo labels or synthetic_link_counts missing -> skipped.')
            continue
        data.append((panel, pseudo_counts(lab), splits))
    if not data:
        return

    nrows = -(-len(data) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows), squeeze=False)
    for ax, (panel, pseudo, splits) in zip(axes.flat, data):
        plot_panel(ax, panel, pseudo, splits)
    for ax in axes.flat[len(data):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel('share of values (%)', fontsize=8, color=MUTED)
    for ax in axes[-1]:
        ax.set_xlabel('link count (veh/h)', fontsize=8, color=MUTED)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f'saved -> {out_path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--panels', nargs='+', default=PANELS)
    ap.add_argument('--labels', type=Path, default=PSEUDO_DIR)
    ap.add_argument('--out', type=Path, default=OUTPUT_DIR / 'task4_link_count_hist.png')
    ap.add_argument('--ncols', type=int, default=5)
    args = ap.parse_args()

    run(RELEASE_ROOT, args.panels, args.labels, args.out, args.ncols)
