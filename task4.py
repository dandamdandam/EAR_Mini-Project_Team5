"""Task 4: Prior-Regularized NNLS (lambda = 20.0)
  min ||A f - b||^2 + lambda * ||f - anchor||^2   s.t.  f >= 0
  A      : path-link incidence (link x path, 0/1)
  b      : synthetic_link_counts.csv  (관측 링크만, PM 첨두 veh/h)
  anchor : synthetic_weak_prior.csv 의 path_flow  (weak OD prior)
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from preprocessing import ODME_PATH, PANELS, RELEASE_ROOT, SUBMIT_SPLITS

REG_LAMBDA = 20.0

PRIOR_SCALE_CALIBRATION = False

ODME_COLUMNS = ['panel', 'departure_time', 'path_id', 'origin_zone', 'destination_zone', 'path_flow']


def solve_path_flow(A, counts, anchor, reg_lambda=REG_LAMBDA):
    aa = np.vstack([A, np.sqrt(reg_lambda) * np.eye(A.shape[1])])
    bb = np.concatenate([counts, np.sqrt(reg_lambda) * anchor])
    f, _ = nnls(aa, bb)
    return f


def load_corridor_network(release, panel):
    """path 집합과 link x path incidence 행렬. 없으면 None."""
    network = release / 'corridors' / panel / 'network'
    paths_csv = network / 'path_set.csv'
    incidence_csv = network / 'path_link_incidence.csv'

    if not (paths_csv.exists() and incidence_csv.exists()):
        print(f'[{panel}] Task 4: path_set/incidence missing -> skipped.')
        return None

    paths = pd.read_csv(paths_csv)
    paths['path_id'] = paths.path_id.astype(str)
    path_ids = paths.path_id.tolist()
    path_index = {x: i for i, x in enumerate(path_ids)}

    incidence = pd.read_csv(incidence_csv)
    incidence['path_id'] = incidence.path_id.astype(str)
    incidence['link_id'] = incidence.link_id.astype(str)
    link_ids = incidence.link_id.drop_duplicates().tolist()
    link_index = {x: i for i, x in enumerate(link_ids)}

    rr = incidence.link_id.map(link_index).to_numpy(dtype=np.int64)
    cc = incidence.path_id.map(path_index).to_numpy(dtype=np.int64)
    A = np.zeros((len(link_ids), len(path_ids)), dtype=np.float64)
    A[rr, cc] = 1.0
    return {'paths': paths, 'path_ids': path_ids, 'link_ids': link_ids, 'A': A}


def load_split_inputs(release, panel, split, link_ids, path_ids):
    """관측 link count, 관측 마스크, weak prior(anchor), departure_time. 없으면 None."""
    counts_path = release / 'task4' / panel / split / 'synthetic_link_counts.csv'
    prior_path = release / 'task4' / panel / split / 'synthetic_weak_prior.csv'

    if not counts_path.exists():
        print(f'[{panel}/{split}] Task 4: synthetic_link_counts.csv missing -> skipped.')
        return None
    if not prior_path.exists():
        print(f'[{panel}/{split}] Task 4: synthetic_weak_prior.csv missing -> skipped.')
        return None

    counts_df = pd.read_csv(counts_path, dtype={'link_id': str})
    counts = counts_df.set_index('link_id').reindex(link_ids).fillna(0.0)['count'].to_numpy(dtype=float)
    observed = set(counts_df.link_id)
    # A 의 행에는 본선 외에 RON-Z*/ROFF-Z* 가상 링크가 섞여 있다.
    # link_counts 는 본선만 주므로 관측 행만 추려서 푼다 (관측비 19~75%).
    measured = np.array([l in observed for l in link_ids], dtype=bool)

    pr = pd.read_csv(prior_path, dtype={'path_id': str})
    anchor = np.nan_to_num(
        pr.set_index('path_id').reindex(path_ids)['path_flow'].to_numpy(dtype=float), nan=0.0)
    dep_time = (str(pr.departure_time.iloc[0]) if 'departure_time' in pr.columns and len(pr)
                else f'SYN_PM_{split.upper()}')
    return {'counts': counts, 'measured': measured, 'anchor': anchor, 'dep_time': dep_time}


def format_records(paths, panel, dep_time, path_flow):
    ps = paths[['path_id', 'origin_zone', 'destination_zone']].copy()
    ps['panel'] = panel
    ps['departure_time'] = dep_time
    ps['path_flow'] = path_flow
    return ps[ODME_COLUMNS]


def write_records(records, out_path):
    if not records:
        pd.DataFrame(columns=ODME_COLUMNS).to_csv(out_path, index=False)
        print('Task 4: no records produced -> empty file written.')
        return 0
    out_df = pd.concat(records, ignore_index=True)
    out_df.to_csv(out_path, index=False)
    print(f'Task 4 complete: {len(out_df):,} entries exported -> {out_path}')
    return len(out_df)


def run_task4(release, panels, splits, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for panel in panels:
        net = load_corridor_network(release, panel)
        if net is None:
            continue
        A, link_ids, path_ids = net['A'], net['link_ids'], net['path_ids']

        for split in splits:
            inp = load_split_inputs(release, panel, split, link_ids, path_ids)
            if inp is None:
                continue
            val_counts, measured, anchor = inp['counts'], inp['measured'], inp['anchor']

            scale = 1.0
            if PRIOR_SCALE_CALIBRATION:
                implied = float((A[measured] @ anchor).sum())
                if implied > 0:
                    scale = float(val_counts[measured].sum()) / implied
                    anchor = anchor * scale

            t0 = time.time()
            solved = solve_path_flow(A[measured], val_counts[measured], anchor)

            # 관측 링크 재현 오차 — train 스플릿에서는 이 값이 S_link 의 대리 지표가 된다.
            pred_links = A[measured] @ solved
            obs_links = val_counts[measured]
            rel = np.abs(pred_links - obs_links) / np.maximum(obs_links, 1e-9)
            print(f'[{panel}/{split}] paths={len(path_ids)} links={measured.sum()}/{len(link_ids)} '
                  f'dep={inp["dep_time"]} scale={scale:.3f} | link 상대오차 중앙값 {np.median(rel) * 100:.2f}% '
                  f'| sum(f)={solved.sum():,.0f} (prior {anchor.sum():,.0f}) | {time.time() - t0:.1f}s')

            records.append(format_records(net['paths'], panel, inp['dep_time'], solved))

    return write_records(records, out_path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--splits', nargs='+', default=SUBMIT_SPLITS)
    ap.add_argument('--panels', nargs='+', default=PANELS)
    ap.add_argument('--out', type=Path, default=ODME_PATH)
    args = ap.parse_args()

    run_task4(RELEASE_ROOT, args.panels, args.splits, args.out)
