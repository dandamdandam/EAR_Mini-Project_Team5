import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from preprocessing import OUTPUT_DIR, PANELS, RELEASE_ROOT, SUBMIT_SPLITS
from .nnls import format_records, load_corridor_network, load_split_inputs, write_records
from .task4_hf import load_tabpfn
from .task4_pseudo_label import PSEUDO_DIR, load_labels, zone_geometry

ODME_PSEUDO_PATH = OUTPUT_DIR / 'odme_submission_pseudo.csv'
N_CONTEXT = 10_000
N_EVAL_DAYS = 5


def path_features(b_ord, o, d, n, start, end):
    obs = np.flatnonzero(~np.isnan(b_ord))
    vals = b_ord[obs]

    def before(p, inclusive):
        i = np.searchsorted(obs, p, side='right' if inclusive else 'left') - 1
        return np.where(i >= 0, vals[np.clip(i, 0, len(obs) - 1)], np.nan)

    def after(p, inclusive):
        i = np.searchsorted(obs, p, side='left' if inclusive else 'right')
        return np.where(i < len(obs), vals[np.clip(i, 0, len(obs) - 1)], np.nan)

    ps, pe = start[o], end[d]
    csum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(b_ord))])
    cnt = np.concatenate([[0], np.cumsum(~np.isnan(b_ord))])
    lo, hi = ps.astype(int), pe.astype(int) + 1
    n_obs = cnt[hi] - cnt[lo]
    seg_mean = np.where(n_obs > 0, (csum[hi] - csum[lo]) / np.maximum(n_obs, 1), np.nan)
    return np.column_stack([o / n, d / n, (d - o) / n,
                            before(ps, False), after(ps, True),
                            before(pe, True), after(pe, False),
                            seg_mean, n_obs])


def s_od(pred, true):
    return max(0.0, 1 - np.abs(pred - true).sum() / max(true.sum(), 1e-9))


def run(release, panels, splits, label_dir, out_path, device, seed=0):
    model = load_tabpfn(device)
    rng = np.random.default_rng(seed)
    records, summary = [], []
    for panel in panels:
        net = load_corridor_network(release, panel)
        lab = load_labels(label_dir, panel)
        if net is None or lab is None:
            print(f'[{panel}] network or pseudo labels missing -> skipped.')
            continue
        t0 = time.time()
        paths = net['paths']
        order, pos, o, d, n, start, end = zone_geometry(paths)
        B, F = lab['B'], lab['F']

        days = rng.permutation(len(B))
        tr_days, ev_days = days[:int(0.8 * len(days))], days[int(0.8 * len(days)):][:N_EVAL_DAYS]
        clim = F[tr_days].mean(axis=0)

        def feats(b_ord):
            return np.column_stack([path_features(b_ord, o, d, n, start, end), clim])

        def predict(x):
            return np.maximum(clim + model.predict(x), 0.0)

        X = np.stack([feats(b) for b in B])
        Xtr = X[tr_days].reshape(-1, X.shape[-1])
        ytr = (F[tr_days] - clim).reshape(-1)
        pick = rng.choice(len(ytr), size=min(N_CONTEXT, len(ytr)), replace=False)
        model.fit(Xtr[pick], ytr[pick])

        sc_tab = np.mean([s_od(predict(X[t]), F[t]) for t in ev_days])
        sc_clim = np.mean([s_od(clim, F[t]) for t in ev_days])
        print(f'[{panel}] held-out S_od  TabPFN {sc_tab:.3f}  vs 날짜평균 {sc_clim:.3f} | '
              f'{time.time() - t0:.0f}s', flush=True)
        summary.append({'panel': panel, 'S_od_tabpfn': sc_tab, 'S_od_climatology': sc_clim})

        for split in splits:
            inp = load_split_inputs(release, panel, split, net['link_ids'], net['path_ids'])
            if inp is None:
                continue
            c = dict(zip(net['link_ids'], inp['counts']))
            meas = {l for l, m in zip(net['link_ids'], inp['measured']) if m}
            b_ord = np.array([c[l] if l in meas else np.nan for l in order])
            records.append(format_records(paths, panel, inp['dep_time'], predict(feats(b_ord))))

    if summary:
        print('\n' + pd.DataFrame(summary).set_index('panel').round(3).to_string())
    return write_records(records, out_path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', nargs='+', default=SUBMIT_SPLITS)
    ap.add_argument('--panels', nargs='+', default=PANELS)
    ap.add_argument('--labels', type=Path, default=PSEUDO_DIR)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', type=Path, default=ODME_PSEUDO_PATH)
    args = ap.parse_args()

    run(RELEASE_ROOT, args.panels, args.splits, args.labels, args.out, args.device)
