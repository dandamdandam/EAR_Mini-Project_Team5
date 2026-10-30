import json
import zlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ode as ode_pipeline
import preprocess
import submission
from features import assemble_pool
from model import QueueNet
from ode import DEVICE, DayCtx, ODERNN, build_link_index, dow_array
from utils import (CACHE_DIR, HIST, HOR, OUT_DIR, QUEUE_RATIO, SCORED_PANELS, SEQ,
                   load_ode_feats, load_panel)

SEED = 123


def ensure_artifacts():
    missing = [p for p in SCORED_PANELS if not (CACHE_DIR / f'{p}.npz').exists()]
    if missing:
        print(f'cache missing for {missing}: running preprocess', flush=True)
        preprocess.build_all()
    ode_missing = not (CACHE_DIR / 'ode_models.pt').exists() or \
        any(not (CACHE_DIR / f'ode_feats_{p}.npz').exists() for p in SCORED_PANELS)
    if ode_missing:
        print('ode models/features missing: training ode models', flush=True)
        ode_pipeline.main()


def to_device(pool, idx):
    out = {}
    for k in ['sph', 'obs', 'flh', 'tf', 'odd', 'wy', 'st', 'qt']:
        out[k] = torch.tensor(pool[k][idx], device=DEVICE)
    out['stat'] = torch.tensor(pool['stat'], device=DEVICE)
    return out


def iou_scores(probs, qt, tau):
    pred = probs >= tau
    tgt = qt > 0.5
    inter = (pred & tgt).sum(axis=(1, 2))
    union = (pred | tgt).sum(axis=(1, 2))
    return np.where(union > 0, inter / np.maximum(union, 1), 1.0)


@torch.no_grad()
def evaluate(model, pools, tau_grid):
    model.eval()
    entries = []
    for pool in pools:
        N = len(pool['qt'])
        probs_list = []
        for s in range(0, N, 16):
            idx = np.arange(s, min(s + 16, N))
            b = to_device(pool, idx)
            if 'cnd' in pool:
                b['cnd'] = torch.tensor(pool['cnd'][idx], device=DEVICE)
            logits, _ = model(b, teacher_prob=0.0)
            probs_list.append(torch.sigmoid(logits[:, HIST - 1:SEQ]).cpu().numpy())
        pers = np.repeat((pool['sph'][:, HIST - 1] <= QUEUE_RATIO)[:, None, :].astype(np.float32), HOR, axis=1)
        entries.append({'panel': pool['panel'], 'probs': np.concatenate(probs_list),
                        'qt': pool['qt'], 'cond': pool['cond'], 'pers': pers})
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
    detail = {
        'equal_mean_iou': float(eq), 'tau': float(tau),
        'onset_iou': float(sc[cond == 0].mean()) if (cond == 0).any() else None,
        'ongoing_iou': float(sc[cond == 1].mean()) if (cond == 1).any() else None,
        'persistence_equal_mean': float(np.mean([pers_sc[cond == c].mean() for c in [0, 1] if (cond == c).any()])),
        'per_panel': {p: float(sc[panels == p].mean()) for p in np.unique(panels)},
        'n_windows': int(len(sc)),
    }
    return detail


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    ensure_artifacts()
    offsets, n_links = build_link_index()
    ck = torch.load(CACHE_DIR / 'ode_models.pt', weights_only=False)
    day_model = ODERNN(ck['n_links'], 8).to(DEVICE)
    day_model.load_state_dict(ck['day'])
    day_ctx = DayCtx().to(DEVICE)
    day_ctx.load_state_dict(ck['day_ctx'])
    day_model.eval()
    dows = dow_array()

    from tau_select import panel_climatology
    pools = {'train': [], 'val': [], 'test': []}
    for p in SCORED_PANELS:
        z = load_panel(p)
        wy_feats = load_ode_feats(p)
        fill_clim = panel_climatology(z)
        for split in ['train', 'val', 'test']:
            t0s, cond = preprocess.select_windows(p, z, split, rng)
            mask_rng = np.random.default_rng(zlib.crc32(f'{p}-{split}'.encode()))
            pool = assemble_pool(p, z, wy_feats, t0s, cond, mask_rng, day_model, day_ctx, offsets[p], dows, fill_clim)
            pool['cnd'] = np.stack([np.isin(cond, (0, 2)), np.isin(cond, (1, 3))], -1).astype(np.float32)
            pools[split].append(pool)
            print(f'[pool] {p} {split}: {len(t0s)} windows '
                  f'(onset={int((cond == 0).sum())}, ongoing={int((cond == 1).sum())}, neg={int((cond >= 2).sum())})', flush=True)

    qt_all = np.concatenate([pl['qt'].reshape(-1) for pl in pools['train']])
    p_pos = qt_all.mean()
    pos_weight = torch.tensor(min((1 - p_pos) / max(p_pos, 1e-4), 4.0), device=DEVICE)
    print(f'train target positive rate={p_pos:.4f} pos_weight={pos_weight.item():.2f}', flush=True)

    model = QueueNet(c_in=18, no_ode=True).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    tau_grid = np.arange(0.2, 0.81, 0.05)

    batches = []
    for pi, pool in enumerate(pools['train']):
        N = len(pool['qt'])
        for s in range(0, N, 8):
            batches.append((pi, np.arange(s, min(s + 8, N))))

    best_val, best_state, best_tau, patience = -1.0, None, 0.5, 0
    for ep in range(30):
        model.train()
        tf_prob = max(0.0, 0.7 * (1 - ep / 12))
        rng.shuffle(batches)
        tot, cnt = 0.0, 0
        for pi, idx in batches:
            pool = pools['train'][pi]
            b = to_device(pool, idx)
            b['cnd'] = torch.tensor(pool['cnd'][idx], device=DEVICE)
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
        val = evaluate(model, pools['val'], tau_grid)
        print(f'epoch {ep + 1}: loss={tot / cnt:.4f} tf={tf_prob:.2f} '
              f'val_eq_iou={val["equal_mean_iou"]:.4f} (tau={val["tau"]:.2f}, onset={val["onset_iou"]:.3f}, ongoing={val["ongoing_iou"]:.3f})', flush=True)
        if val['equal_mean_iou'] > best_val:
            best_val = val['equal_mean_iou']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_tau = val['tau']
            patience = 0
        else:
            patience += 1
            if patience >= 8:
                print('early stop', flush=True)
                break

    model.load_state_dict(best_state)
    from tau_select import bootstrap_tau, collect_entries, percond_eval, score_matrix
    Sv, cv = score_matrix(collect_entries(model, pools['val'], True))
    St, ct = score_matrix(collect_entries(model, pools['test'], True))
    taus = {c: bootstrap_tau(Sv[cv == c])[0] for c in (0, 1)}
    val_eq, val_per = percond_eval(Sv, cv, taus)
    test_eq, test_per = percond_eval(St, ct, taus)
    report = {'val': {'equal_mean_iou': val_eq, 'onset_iou': val_per[0], 'ongoing_iou': val_per[1]},
              'test': {'equal_mean_iou': test_eq, 'onset_iou': test_per[0], 'ongoing_iou': test_per[1]},
              'tau_onset': taus[0], 'tau_ongoing': taus[1], 'best_epoch_tau': float(best_tau),
              'train_positive_rate': float(p_pos),
              'holdout': {'train': '2030-06-01..2030-12-31', 'val': '2031-01-01..2031-01-31', 'test': '2031-02-01..2031-02-28'}}
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / 't2_report.json').write_text(json.dumps(report, indent=2))
    torch.save({'model': model.state_dict(), 'tau': best_tau,
                'tau_onset': taus[0], 'tau_ongoing': taus[1]}, CACHE_DIR / 't2_model.pt')
    print(json.dumps(report, indent=2), flush=True)
    submission.generate()


if __name__ == '__main__':
    main()
