import argparse
import json
import zlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import preprocess
from features import prop_map, static_features
from model import QueueNet
from ode import DEVICE, dow_array
from tau_select import (bootstrap_tau, collect_entries, evaluate_general,
                        panel_climatology, percond_eval, score_matrix, to_device_ex)
from utils import (CACHE_DIR, DAY_STEPS, EXTRA_DROP, HIST, OUT_DIR, QUEUE_RATIO, SCORED_PANELS,
                   SEQ, STEP_OFFSETS, date_to_day_index, interp_fill, load_panel, split_day_range)

SEED = 123
EXTRA_PANELS = ['D12_I405_N', 'D12_I405_S']
DOWS = dow_array()
ARGS = None


def select_train_windows(z, day_lo, day_hi, rng):
    x = (z['s_raw'] / z['vf'][None]).astype(np.float32)
    est, hq = preprocess.classify_origins(x, z['q'])
    t_all = np.arange(x.shape[0])
    slot = t_all % DAY_STEPS
    day = t_all // DAY_STEPS
    base = (slot >= 12) & (slot <= 281) & (day >= day_lo) & (day <= day_hi)
    onset = t_all[base & hq & ~est]
    ongoing = t_all[base & hq & est]
    neg = t_all[base & ~hq]
    cap_on = ARGS.on_cap
    cap_og = int(len(onset) * ARGS.og_frac) if ARGS.og_frac else ARGS.og_cap
    cap_ng = int(len(onset) * ARGS.ng_frac) if ARGS.ng_frac else ARGS.ng_cap
    on_p = preprocess.pick_spaced(onset, cap_on, 0, rng)
    og_p = preprocess.pick_spaced(ongoing, cap_og, 0, rng)
    ng_p = preprocess.pick_spaced(neg, cap_ng, 0, rng)
    pools = [on_p, og_p, ng_p[~est[ng_p]], ng_p[est[ng_p]]]
    t0s = np.concatenate(pools)
    cond = np.concatenate([np.full(len(p), i, np.int64) for i, p in enumerate(pools)])
    order = np.argsort(t0s)
    return t0s[order], cond[order]


def assemble_noode(panel, z, t0s, cond, rng, fill_clim):
    vf = z['vf']
    x = (z['s_raw'] / vf[None]).astype(np.float32)
    fc = (z['f_raw'] / z['cap'][None]).astype(np.float32)
    strue = (z['s_true'] / vf[None]).astype(np.float32)
    q = z['q']
    L = x.shape[1]
    N = len(t0s)
    ht = t0s[:, None] + np.arange(-12, 1)[None, :]
    sph = x[ht].copy()
    flh = fc[ht].copy()
    extra = rng.random(sph.shape) < EXTRA_DROP
    sph[extra] = np.nan
    flh[extra] = np.nan
    obs = np.isfinite(sph).astype(np.float32)
    sph = interp_fill(sph.transpose(1, 0, 2).reshape(HIST, -1)).reshape(HIST, N, L).transpose(1, 0, 2)
    flh = interp_fill(flh.transpose(1, 0, 2).reshape(HIST, -1)).reshape(HIST, N, L).transpose(1, 0, 2)
    st = strue[t0s[:, None] + np.arange(-11, 7)[None, :]]
    qt = q[t0s[:, None] + np.arange(1, 7)[None, :]].astype(np.float32)
    slots = (t0s % DAY_STEPS).astype(np.int64)
    days = (t0s // DAY_STEPS).astype(np.int64)
    step_slot = slots[:, None] + STEP_OFFSETS[None, :]
    ang_t = 2 * np.pi * step_slot / DAY_STEPS
    ang_d = 2 * np.pi * (DOWS[days][:, None] + step_slot * 5.0 / 1440.0) / 7.0
    tf = np.stack([np.sin(ang_t), np.cos(ang_t), np.sin(ang_d), np.cos(ang_d)], axis=-1).astype(np.float32)
    nanmask = ~np.isfinite(sph)
    slot_h = (t0s[:, None] + np.arange(-12, 1)[None, :]) % DAY_STEPS
    sph[nanmask] = fill_clim[slot_h][nanmask]
    flh[~np.isfinite(flh)] = 0.4
    pool = {'sph': sph.astype(np.float32), 'obs': obs, 'flh': flh.astype(np.float32), 'tf': tf,
            'odd': np.zeros((N, SEQ, L), np.float32), 'wy': np.zeros((N, L, 2), np.float32),
            'stat': static_features(z), 'st': st.astype(np.float32),
            'qt': qt, 'cond': cond, 'panel': panel}
    if 'wave' in z:
        pool['prp'] = prop_map(sph[:, -1] <= QUEUE_RATIO, z['milepost'], z['wave'])
    return pool


def build_all_pools(rng):
    for p in EXTRA_PANELS:
        if not (CACHE_DIR / f'{p}.npz').exists():
            print(f'building cache for {p}', flush=True)
            preprocess.build_panel(p)

    pools = {'train': [], 'val': [], 'test': []}
    tr_lo, tr_hi = split_day_range('train')
    full_hi = date_to_day_index('2031-02-28')

    for p in SCORED_PANELS:
        z = load_panel(p)
        fill_clim = panel_climatology(z)
        t0s, cond = select_train_windows(z, tr_lo, tr_hi, rng)
        mask_rng = np.random.default_rng(zlib.crc32(f'{p}-train'.encode()))
        pools['train'].append(assemble_noode(p, z, t0s, cond, mask_rng, fill_clim))
        print(f'[pool] {p} train: {len(t0s)} windows '
              f'(onset={int((cond == 0).sum())}, ongoing={int((cond == 1).sum())}, neg={int((cond >= 2).sum())})', flush=True)
        for split in ['val', 'test']:
            t0s, cond = preprocess.select_windows(p, z, split, rng)
            mask_rng = np.random.default_rng(zlib.crc32(f'{p}-{split}'.encode()))
            pools[split].append(assemble_noode(p, z, t0s, cond, mask_rng, fill_clim))
            print(f'[pool] {p} {split}: {len(t0s)} windows', flush=True)

    for p in EXTRA_PANELS:
        z = load_panel(p)
        fill_clim = panel_climatology(z)
        t0s, cond = select_train_windows(z, tr_lo, full_hi, rng)
        mask_rng = np.random.default_rng(zlib.crc32(f'{p}-train'.encode()))
        pools['train'].append(assemble_noode(p, z, t0s, cond, mask_rng, fill_clim))
        print(f'[pool] {p} train(full-range): {len(t0s)} windows '
              f'(onset={int((cond == 0).sum())}, ongoing={int((cond == 1).sum())}, neg={int((cond >= 2).sum())})', flush=True)

    for split in pools:
        for pool in pools[split]:
            c = pool['cond']
            pool['cnd'] = np.stack([np.isin(c, (0, 2)), np.isin(c, (1, 3))], -1).astype(np.float32)
    return pools


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-name', default='maxdata_s123')
    ap.add_argument('--on-cap', type=int, default=10 ** 9)
    ap.add_argument('--og-cap', type=int, default=3000)
    ap.add_argument('--ng-cap', type=int, default=1200)
    ap.add_argument('--og-frac', type=float, default=None)
    ap.add_argument('--ng-frac', type=float, default=None)
    ap.add_argument('--pe-freqs', type=int, default=0)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--no-nb', action='store_true')
    ap.add_argument('--phys', action='store_true')
    ARGS = ap.parse_args()
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    pools = build_all_pools(rng)
    if ARGS.no_nb:
        for split in pools:
            for pool in pools[split]:
                pool['stat'] = pool['stat'][:, :6]

    qt_all = np.concatenate([pl['qt'].reshape(-1) for pl in pools['train']])
    p_pos = qt_all.mean()
    pos_weight = torch.tensor(min((1 - p_pos) / max(p_pos, 1e-4), 4.0), device=DEVICE)
    n_train = sum(len(pl['qt']) for pl in pools['train'])
    print(f'train windows={n_train} positive rate={p_pos:.4f} pos_weight={pos_weight.item():.2f}', flush=True)

    model = QueueNet(c_in=(16 if ARGS.no_nb else 20) + 2 * ARGS.pe_freqs, no_ode=True,
                     pe_freqs=ARGS.pe_freqs, no_nb=ARGS.no_nb, phys=ARGS.phys).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    tau_grid = np.arange(0.2, 0.81, 0.05)

    batches = []
    for pi, pool in enumerate(pools['train']):
        N = len(pool['qt'])
        for s in range(0, N, ARGS.batch):
            batches.append((pi, np.arange(s, min(s + ARGS.batch, N))))

    best_val, best_state, best_tau, best_epoch, patience = -1.0, None, 0.5, 0, 0
    for ep in range(ARGS.epochs):
        model.train()
        tf_prob = max(0.0, 0.7 * (1 - ep / 12))
        rng.shuffle(batches)
        tot, cnt = 0.0, 0
        for pi, idx in batches:
            b = to_device_ex(pools['train'][pi], idx, True)
            logits, spreds = model(b, teacher_prob=tf_prob)
            bce = F.binary_cross_entropy_with_logits(logits[:, HIST - 1:SEQ], b['qt'], pos_weight=pos_weight)
            w = torch.ones(SEQ, device=DEVICE)
            w[HIST - 1:] = 2.0
            mse = (w[None, :, None] * (spreds - b['st']) ** 2).mean()
            loss = bce + 0.3 * mse
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            tot += loss.item()
            cnt += 1
        val = evaluate_general(model, pools['val'], tau_grid, True)
        print(f'epoch {ep + 1}: loss={tot / cnt:.4f} tf={tf_prob:.2f} '
              f'val_eq_iou={val["equal_mean_iou"]:.4f} (tau={val["tau"]:.2f}, '
              f'onset={val["onset_iou"]:.3f}, ongoing={val["ongoing_iou"]:.3f})', flush=True)
        if val['equal_mean_iou'] > best_val:
            best_val = val['equal_mean_iou']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_tau = val['tau']
            best_epoch = ep + 1
            patience = 0
        else:
            patience += 1
            if patience >= ARGS.patience:
                print('early stop', flush=True)
                break

    model.load_state_dict(best_state)
    Sv, cv = score_matrix(collect_entries(model, pools['val'], True))
    St, ct = score_matrix(collect_entries(model, pools['test'], True))
    taus = {c: bootstrap_tau(Sv[cv == c])[0] for c in (0, 1)}
    val_eq, val_per = percond_eval(Sv, cv, taus)
    test_eq, test_per = percond_eval(St, ct, taus)
    report = {'run': ARGS.run_name, 'seed': SEED, 'best_epoch': best_epoch,
              'pe_freqs': ARGS.pe_freqs, 'batch': ARGS.batch, 'max_epochs': ARGS.epochs,
              'no_nb': ARGS.no_nb, 'phys': ARGS.phys,
              'alpha': float(torch.sigmoid(model.alpha)) if ARGS.phys else None,
              'train_windows': n_train, 'train_positive_rate': float(p_pos),
              'caps': {'onset': ARGS.on_cap, 'ongoing': ARGS.og_cap, 'neg': ARGS.ng_cap,
                       'og_frac': ARGS.og_frac, 'ng_frac': ARGS.ng_frac},
              'extra_panels': EXTRA_PANELS,
              'val': {'equal_mean_iou': val_eq, 'onset_iou': val_per[0], 'ongoing_iou': val_per[1]},
              'test': {'equal_mean_iou': test_eq, 'onset_iou': test_per[0], 'ongoing_iou': test_per[1]},
              'tau_onset': taus[0], 'tau_ongoing': taus[1], 'best_epoch_tau': float(best_tau)}
    sweep_dir = OUT_DIR / 'sweep'
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / f'{ARGS.run_name}.json').write_text(json.dumps(report, indent=2))
    torch.save({'model': best_state, 'tau': best_tau, 'no_nb': ARGS.no_nb, 'phys': ARGS.phys,
                'tau_onset': taus[0], 'tau_ongoing': taus[1]}, sweep_dir / f'{ARGS.run_name}.pt')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
