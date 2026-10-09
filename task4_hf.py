"""Task 4 (HF 모델 버전): TabPFN 으로 미관측 링크 count 를 추론한 뒤 가중 prior-NNLS.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

from preprocessing import OUTPUT_DIR, PANELS, RELEASE_ROOT, SUBMIT_SPLITS
from task4 import REG_LAMBDA, format_records, load_corridor_network, load_split_inputs, write_records

ODME_HF_PATH = OUTPUT_DIR / 'odme_submission_hf.csv'

# 보간된 링크 행의 가중치 (관측 링크 = 1). holdout 스윕 결과:
#   w     0     0.25   0.5    1.0    2.0
#   ho%   1.80  1.28   1.05   1.02   1.03     (held-out 링크 오차 중앙값, 패널 평균)
#   dev   .140  .149   .154   .161   .167     (||f - f0|| / ||f0||)
IMPUTED_WEIGHT = 0.5


def load_tabpfn(device):
    # 첫 호출 때 HF Hub(Prior-Labs/TabPFN-v2-reg)에서 받아 ~/.cache/tabpfn 에 캐시한다.
    return TabPFNRegressor.create_default_for_version(ModelVersion.V2, device=device)


def solve_weighted(A, b, w, anchor, reg_lambda=REG_LAMBDA):
    """min 0.5 ||sqrt(w)(A f - b)||^2 + 0.5 lambda ||f - anchor||^2,  f >= 0."""
    def fun_grad(f):
        r = A @ f - b
        wr = w * r
        d = f - anchor
        return 0.5 * (r @ wr) + 0.5 * reg_lambda * (d @ d), A.T @ wr + reg_lambda * d

    res = minimize(fun_grad, anchor.copy(), jac=True, method='L-BFGS-B',
                   bounds=[(0, None)] * len(anchor),
                   options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-8})
    return np.maximum(res.x, 0.0)


def corridor_order(paths, link_ids):
    """본선 링크의 A 행 인덱스를 상류 -> 하류 순서로."""
    longest = max(paths.link_seq.astype(str).str.split(';'), key=len)
    row = {l: k for k, l in enumerate(link_ids)}
    return np.array([row[l] for l in longest if l in row], dtype=np.int64)


def link_features(known, counts, prior_flow, n_paths):
    """코리도 순서로 정렬된 본선 링크별 특징. 최근접 관측 이웃은 자기 자신을 제외한다."""
    n = len(known)
    idx = np.arange(n)
    k_idx = np.flatnonzero(known)
    last = len(k_idx) - 1
    up_pos = np.searchsorted(k_idx, idx, side='left') - 1     # k_idx[up_pos] < i
    dn_pos = np.searchsorted(k_idx, idx, side='right')        # k_idx[dn_pos] > i
    has_up = up_pos >= 0
    has_dn = dn_pos <= last
    u = k_idx[np.clip(up_pos, 0, last)]
    d = k_idx[np.clip(dn_pos, 0, last)]

    uv = np.where(has_up, counts[u], np.nan)
    dv = np.where(has_dn, counts[d], np.nan)
    ud = np.where(has_up, idx - u, np.nan)
    dd = np.where(has_dn, d - idx, np.nan)
    ur = uv / np.maximum(prior_flow[u], 1e-6)
    dr = dv / np.maximum(prior_flow[d], 1e-6)
    between = has_up & has_dn
    interp = np.where(between, uv + (dv - uv) * ud / np.where(between, d - u, 1), np.where(has_up, uv, dv))

    return np.column_stack([idx / max(n - 1, 1), prior_flow, n_paths, uv, dv, ud, dd, ur, dr, interp])


def impute_counts(model, known, counts, prior_flow, n_paths):
    """known 링크를 context 로 TabPFN in-context 회귀. 전 링크에 대한 예측(>=0)을 돌려준다."""
    X = link_features(known, counts, prior_flow, n_paths)
    model.fit(X[known], counts[known])
    return np.maximum(model.predict(X), 0.0)


def build_system(A, ml_rows, known, counts, imputed, measured, weight):
    """관측 행 + (미관측 본선) 보간 행을 쌓은 (A, b, w)."""
    obs_rows = np.flatnonzero(measured)
    unk = ~known
    if imputed is None or not unk.any():
        return A[obs_rows], counts[obs_rows], np.ones(len(obs_rows))
    rows = np.concatenate([obs_rows, ml_rows[unk]])
    b = np.concatenate([counts[obs_rows], imputed[unk]])
    w = np.concatenate([np.ones(len(obs_rows)), np.full(unk.sum(), weight)])
    return A[rows], b, w


def run_task4_hf(release, panels, splits, out_path, weight, device):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = load_tabpfn(device)
    records = []
    for panel in panels:
        net = load_corridor_network(release, panel)
        if net is None:
            continue
        A, link_ids, path_ids = net['A'], net['link_ids'], net['path_ids']
        ml_rows = corridor_order(net['paths'], link_ids)
        n_paths = A[ml_rows].sum(axis=1)

        for split in splits:
            inp = load_split_inputs(release, panel, split, link_ids, path_ids)
            if inp is None:
                continue
            counts, measured, anchor = inp['counts'], inp['measured'], inp['anchor']
            ml_counts = counts[ml_rows]
            known = measured[ml_rows]
            prior_flow = A[ml_rows] @ anchor

            t0 = time.time()
            imputed = None
            if known.any() and not known.all():
                imputed = impute_counts(model, known, ml_counts, prior_flow, n_paths)
            A_sys, b_sys, w_sys = build_system(A, ml_rows, known, counts, imputed, measured, weight)
            solved = solve_weighted(A_sys, b_sys, w_sys, anchor)

            obs_links = counts[measured]
            rel = np.abs(A[measured] @ solved - obs_links) / np.maximum(obs_links, 1e-9)
            dev = np.linalg.norm(solved - anchor) / max(np.linalg.norm(anchor), 1e-9)
            print(f'[{panel}/{split}] paths={len(path_ids)} obs={known.sum()}/{len(known)} '
                  f'imputed={(~known).sum() if imputed is not None else 0} | '
                  f'관측 link 상대오차 중앙값 {np.median(rel) * 100:.2f}% | '
                  f'||f-f0||/||f0||={dev:.3f} | sum(f)={solved.sum():,.0f} (prior {anchor.sum():,.0f}) '
                  f'| {time.time() - t0:.1f}s')

            records.append(format_records(net['paths'], panel, inp['dep_time'], solved))

    return write_records(records, out_path)


def run_holdout(release, panels, weight, device, n_folds=5, split='train', seed=0):
    """관측 본선 링크를 fold 로 가리고, 가린 링크의 count 재현 오차를 비교한다.

    nnls   : 남은 관측 링크만으로 prior-NNLS (= task4.py)
    tabpfn : TabPFN 보간값 자체
    hybrid : 남은 관측 + TabPFN 보간 (가중치 weight) 으로 prior-NNLS (= 이 스크립트)
    """
    model = load_tabpfn(device)
    rng = np.random.default_rng(seed)
    rows = []
    for panel in panels:
        net = load_corridor_network(release, panel)
        if net is None:
            continue
        A, link_ids = net['A'], net['link_ids']
        inp = load_split_inputs(release, panel, split, link_ids, net['path_ids'])
        if inp is None:
            continue
        counts, measured, anchor = inp['counts'], inp['measured'], inp['anchor']
        ml_rows = corridor_order(net['paths'], link_ids)
        n_paths = A[ml_rows].sum(axis=1)
        prior_flow = A[ml_rows] @ anchor
        ml_counts = counts[ml_rows]
        obs = measured[ml_rows]

        err = {'nnls': [], 'tabpfn': [], 'hybrid': []}
        dev = {'nnls': [], 'hybrid': []}
        for fold in np.array_split(rng.permutation(np.flatnonzero(obs)), n_folds):
            known = obs.copy()
            known[fold] = False
            meas = measured.copy()
            meas[ml_rows[fold]] = False
            imputed = impute_counts(model, known, ml_counts, prior_flow, n_paths)
            target = ml_counts[fold]

            def rel(pred):
                return np.abs(pred - target) / np.maximum(target, 1e-9)

            f_nnls = solve_weighted(*build_system(A, ml_rows, known, counts, None, meas, weight), anchor)
            f_hyb = solve_weighted(*build_system(A, ml_rows, known, counts, imputed, meas, weight), anchor)
            err['nnls'].append(rel(A[ml_rows[fold]] @ f_nnls))
            err['tabpfn'].append(rel(imputed[fold]))
            err['hybrid'].append(rel(A[ml_rows[fold]] @ f_hyb))
            for k, f in (('nnls', f_nnls), ('hybrid', f_hyb)):
                dev[k].append(np.linalg.norm(f - anchor) / max(np.linalg.norm(anchor), 1e-9))

        r = {'panel': panel, 'obs': f'{obs.sum()}/{len(obs)}'}
        r.update({f'{k}_err%': np.median(np.concatenate(v)) * 100 for k, v in err.items()})
        r.update({f'{k}_dev': np.mean(v) for k, v in dev.items()})
        rows.append(r)
        print(f'[{panel}] ' + ' '.join(f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}'
                                       for k, v in r.items() if k != 'panel'), flush=True)

    table = pd.DataFrame(rows).set_index('panel')
    print(f'\nholdout ({split}, {n_folds}-fold, weight={weight}) — err% = held-out 링크 상대오차 중앙값')
    print(table.round(3).to_string())
    print(table.select_dtypes('number').mean().round(3).to_string())
    return table


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--splits', nargs='+', default=SUBMIT_SPLITS)
    ap.add_argument('--panels', nargs='+', default=PANELS)
    ap.add_argument('--weight', type=float, default=IMPUTED_WEIGHT, help='보간 링크 행 가중치')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', type=Path, default=ODME_HF_PATH)
    ap.add_argument('--cv', action='store_true', help='train holdout 비교만 하고 제출 파일은 만들지 않는다')
    args = ap.parse_args()

    if args.cv:
        run_holdout(RELEASE_ROOT, args.panels, args.weight, args.device)
    else:
        run_task4_hf(RELEASE_ROOT, args.panels, args.splits, args.out, args.weight, args.device)
