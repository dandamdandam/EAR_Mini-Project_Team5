import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import torch

import preprocess
from _main import iou_scores, to_device
from features import assemble_pool
from model import QueueNet
from ode import DEVICE, DayCtx, ODERNN, build_link_index, dow_array
from utils import (CACHE_DIR, DAY_STEPS, HIST, HOR, OUT_DIR, QUEUE_RATIO, SCORED_PANELS, SEQ,
                   load_ode_feats, load_panel, split_day_range)

TAU_GRID = np.round(np.arange(0.05, 0.951, 0.025), 3)


def panel_climatology(z):
    x = (z['s_raw'] / z['vf'][None]).astype(np.float32)
    lo, hi = split_day_range('train')
    xt = x[lo * DAY_STEPS:(hi + 1) * DAY_STEPS]
    clim = np.nanmean(xt.reshape(-1, DAY_STEPS, xt.shape[1]), axis=0)
    return np.nan_to_num(clim, nan=1.0).astype(np.float32)


def build_pools(rng, splits=('train', 'val', 'test'), no_ode=False):
    offsets, _ = build_link_index()
    ck = torch.load(CACHE_DIR / 'ode_models.pt', weights_only=False)
    day_model = ODERNN(ck['n_links'], 8).to(DEVICE)
    day_model.load_state_dict(ck['day'])
    day_ctx = DayCtx().to(DEVICE)
    day_ctx.load_state_dict(ck['day_ctx'])
    day_model.eval()
    dows = dow_array()

    pools = {s: [] for s in splits}
    for p in SCORED_PANELS:
        z = load_panel(p)
        wy_feats = load_ode_feats(p)
        fill_clim = panel_climatology(z) if no_ode else None
        for split in splits:
            t0s, cond = preprocess.select_windows(p, z, split, rng)
            mask_rng = np.random.default_rng(zlib.crc32(f'{p}-{split}'.encode()))
            pool = assemble_pool(p, z, wy_feats, t0s, cond, mask_rng, day_model, day_ctx, offsets[p], dows, fill_clim)
            pools[split].append(pool)
            print(f'[pool] {p} {split}: {len(t0s)} windows', flush=True)
    return pools


def add_cond_feature(pools, legacy=False):
    for split in pools:
        for pool in pools[split]:
            c = pool['cond']
            if legacy:
                pool['cnd'] = np.stack([c == 0, c == 1], -1).astype(np.float32)
            else:
                pool['cnd'] = np.stack([np.isin(c, (0, 2)), np.isin(c, (1, 3))], -1).astype(np.float32)


def to_device_ex(pool, idx, use_cond=False):
    b = to_device(pool, idx)
    if use_cond:
        b['cnd'] = torch.tensor(pool['cnd'][idx], device=DEVICE)
    if 'prp' in pool:
        b['prp'] = torch.tensor(pool['prp'][idx], device=DEVICE)
    return b


@torch.no_grad()
def collect_entries(model, pools, use_cond=False):
    model.eval()
    entries = []
    for pool in pools:
        N = len(pool['qt'])
        probs_list = []
        for s in range(0, N, 16):
            idx = np.arange(s, min(s + 16, N))
            b = to_device_ex(pool, idx, use_cond)
            logits, _ = model(b, teacher_prob=0.0)
            probs_list.append(torch.sigmoid(logits[:, HIST - 1:SEQ]).cpu().numpy())
        pers = np.repeat((pool['sph'][:, HIST - 1] <= QUEUE_RATIO)[:, None, :].astype(np.float32), HOR, axis=1)
        entries.append({'panel': pool['panel'], 'probs': np.concatenate(probs_list),
                        'qt': pool['qt'], 'cond': pool['cond'], 'pers': pers})
    return entries


def evaluate_general(model, pools, tau_grid, use_cond=False):
    entries = collect_entries(model, pools, use_cond)
    cond = np.concatenate([e['cond'] for e in entries])
    panels = np.concatenate([np.full(len(e['cond']), e['panel']) for e in entries])

    def all_scores(tau, key='probs'):
        return np.concatenate([iou_scores(e[key], e['qt'], tau) for e in entries])

    best = (-1.0, 0.5)
    for tau in tau_grid:
        sc = all_scores(tau)
        eq = np.mean([sc[cond == c].mean() for c in [0, 1] if (cond == c).any()])
        if eq > best[0]:
            best = (eq, float(tau))
    eq, tau = best
    sc = all_scores(tau)
    pers_sc = all_scores(0.5, 'pers')
    return {
        'equal_mean_iou': float(eq), 'tau': float(tau),
        'onset_iou': float(sc[cond == 0].mean()) if (cond == 0).any() else None,
        'ongoing_iou': float(sc[cond == 1].mean()) if (cond == 1).any() else None,
        'persistence_equal_mean': float(np.mean([pers_sc[cond == c].mean() for c in [0, 1] if (cond == c).any()])),
        'per_panel': {p: float(sc[panels == p].mean()) for p in np.unique(panels)},
        'n_windows': int(len(sc)),
    }


def score_matrix(entries, tau_grid=TAU_GRID):
    cond = np.concatenate([e['cond'] for e in entries])
    cols = [np.concatenate([iou_scores(e['probs'], e['qt'], t) for e in entries]) for t in tau_grid]
    return np.stack(cols, -1), cond


def bootstrap_tau(S_c, tau_grid=TAU_GRID, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(S_c)
    idx = rng.integers(0, n, (B, n))
    taus = tau_grid[S_c[idx].mean(1).argmax(1)]
    med = np.median(taus)
    snap = float(tau_grid[np.argmin(np.abs(tau_grid - med))])
    return snap, float(np.quantile(taus, 0.25)), float(np.quantile(taus, 0.75))


def single_tau(S, cond, tau_grid=TAU_GRID):
    eq = np.array([np.mean([S[cond == c, i].mean() for c in (0, 1)]) for i in range(len(tau_grid))])
    i = int(eq.argmax())
    return float(tau_grid[i]), float(eq[i])


def percond_eval(S, cond, taus, tau_grid=TAU_GRID):
    per = {}
    for c in (0, 1):
        i = int(np.argmin(np.abs(tau_grid - taus[c])))
        per[c] = float(S[cond == c, i].mean())
    return float((per[0] + per[1]) / 2), per


def oracle_percond(S, cond, tau_grid=TAU_GRID):
    taus = {c: float(tau_grid[int(S[cond == c].mean(0).argmax())]) for c in (0, 1)}
    eq, per = percond_eval(S, cond, taus, tau_grid)
    return taus, eq, per


def analyze_ckpt(ck_path, pools):
    sd = torch.load(ck_path, weights_only=False)['model']
    c_in = int(sd['proj.0.weight'].shape[1])
    use_cond = c_in in (23, 20)
    model = QueueNet(c_in=c_in, no_ode=c_in in (20, 18)).to(DEVICE)
    model.load_state_dict(sd)
    Sv, cv = score_matrix(collect_entries(model, pools['val'], use_cond))
    St, ct = score_matrix(collect_entries(model, pools['test'], use_cond))

    tau1, val_eq1 = single_tau(Sv, cv)
    test_eq1, test_per1 = percond_eval(St, ct, {0: tau1, 1: tau1})
    tau1_or, test_eq1_or = single_tau(St, ct)

    taus, iqr = {}, {}
    for c in (0, 1):
        t, q25, q75 = bootstrap_tau(Sv[cv == c])
        taus[c] = t
        iqr[c] = [q25, q75]
    val_eq2, val_per2 = percond_eval(Sv, cv, taus)
    test_eq2, test_per2 = percond_eval(St, ct, taus)
    or_taus, test_eq2_or, _ = oracle_percond(St, ct)

    return {'ckpt': Path(ck_path).stem, 'c_in': c_in, 'cond_feature': use_cond,
            'single': {'tau': tau1, 'val_eq': val_eq1, 'test_eq': test_eq1,
                       'test_onset': test_per1[0], 'test_ongoing': test_per1[1],
                       'test_oracle_tau': tau1_or, 'test_oracle_eq': test_eq1_or,
                       'transfer_gap': test_eq1_or - test_eq1},
            'percond': {'tau_onset': taus[0], 'tau_ongoing': taus[1],
                        'iqr_onset': iqr[0], 'iqr_ongoing': iqr[1],
                        'val_eq': val_eq2, 'val_onset': val_per2[0], 'val_ongoing': val_per2[1],
                        'test_eq': test_eq2, 'test_onset': test_per2[0], 'test_ongoing': test_per2[1],
                        'test_oracle_taus': or_taus, 'test_oracle_eq': test_eq2_or,
                        'transfer_gap': test_eq2_or - test_eq2}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpts', nargs='+', type=Path)
    ap.add_argument('--out', type=Path, default=OUT_DIR / 'sweep' / 'tau_report.json')
    ap.add_argument('--no-ode', action='store_true')
    args = ap.parse_args()

    pools = build_pools(np.random.default_rng(0), splits=('val', 'test'), no_ode=args.no_ode)
    add_cond_feature(pools)

    rows = [analyze_ckpt(ck, pools) for ck in args.ckpts]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))

    print('\nckpt            mode     tau(on/og)      val_eq  test_eq  t_onset  t_ongoing  oracle_gap')
    for r in rows:
        s, p = r['single'], r['percond']
        print(f"{r['ckpt']:<15} single   {s['tau']:.3f}/{s['tau']:.3f}     "
              f"{s['val_eq']:.4f}  {s['test_eq']:.4f}   {s['test_onset']:.4f}   {s['test_ongoing']:.4f}     {s['transfer_gap']:+.4f}")
        print(f"{'':<15} percond  {p['tau_onset']:.3f}/{p['tau_ongoing']:.3f}     "
              f"{p['val_eq']:.4f}  {p['test_eq']:.4f}   {p['test_onset']:.4f}   {p['test_ongoing']:.4f}     {p['transfer_gap']:+.4f}")
    print(f'\nreport: {args.out}', flush=True)


if __name__ == '__main__':
    main()
