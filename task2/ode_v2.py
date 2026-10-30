import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from ode import (DEVICE, DayCtx, ODERNN, dow_array, dow_array_val3, doy_array,
                 gather_days)
from utils import CACHE_DIR, SCORED_PANELS, load_ode_feats, load_panel, n_days, split_day_range

SEED = 42
MIN_DAY_SEQS = 10
WEEK_LAGS = np.arange(-7, 0)
YEAR_LAGS = np.arange(-90, 0, 3)
EVAL_CUT_HOUR = 17
OUT_CKPT = CACHE_DIR / 'ode_models_v2.pt'


def sd_cpu(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def train_link_day_model(x_seqs, dow_seq, steps, rng):
    model = ODERNN(1, 8).to(DEVICE)
    ctx_mod = DayCtx().to(DEVICE)
    params = list(model.parameters()) + list(ctx_mod.parameters())
    opt = torch.optim.AdamW(params, lr=3e-3)
    x = torch.tensor(x_seqs, device=DEVICE)
    dw = torch.tensor(dow_seq, device=DEVICE, dtype=torch.long)
    li = torch.zeros(len(x_seqs), device=DEVICE, dtype=torch.long)
    karr = torch.arange(24, device=DEVICE).unsqueeze(0)
    m = torch.isfinite(x)
    tgt = torch.nan_to_num(x)
    last = 0.0
    for _ in range(steps):
        cut = torch.tensor(rng.integers(6, 23, size=len(x_seqs)), device=DEVICE)
        obs = m & (karr < cut.unsqueeze(1))
        w = torch.where(karr >= cut.unsqueeze(1), 3.0, 1.0).expand_as(x)
        pred, _, _ = model.scan(li, ctx_mod(dw), x, obs, 1.0 / 24)
        loss = (w[m] * (pred[m] - tgt[m]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, 2.0)
        opt.step()
        last = loss.item()
    return model, ctx_mod, last


def train_panel_day_models(panel, z, dows, dows3, tr_lo, va_hi, steps, rng):
    H = np.concatenate([z['hourly'][tr_lo:va_hi + 1], z['hourly3']]).astype(np.float32)
    DW = np.concatenate([dows[tr_lo:va_hi + 1], dows3])
    L = H.shape[2]
    states = []
    t0 = time.time()
    losses = []
    for l in range(L):
        xs = H[:, :, l]
        ok = np.isfinite(xs).sum(axis=1) >= 8
        if ok.sum() < MIN_DAY_SEQS:
            states.append(None)
            continue
        model, ctx_mod, last = train_link_day_model(xs[ok], DW[ok], steps, rng)
        states.append({'model': sd_cpu(model), 'ctx': sd_cpu(ctx_mod)})
        losses.append(last)
    n_tr = sum(s is not None for s in states)
    print(f'[day:{panel}] trained {n_tr}/{L} link models in {time.time() - t0:.0f}s '
          f'(mean final wmse={np.mean(losses):.5f})', flush=True)
    return states


def train_corridor_span(panel, daily, lags, ctx_kind, name, epochs, rng, dows, doys, tr_lo, va_hi):
    L = daily.shape[1]
    ctx_dim = 8 if ctx_kind == 'dow' else 2
    model = ODERNN(L, ctx_dim).to(DEVICE)
    ctx_mod = DayCtx().to(DEVICE) if ctx_kind == 'dow' else None
    params = list(model.parameters()) + (list(ctx_mod.parameters()) if ctx_mod else [])
    opt = torch.optim.AdamW(params, lr=3e-3)
    d_min = tr_lo + max(0, -int(lags.min()))
    days = np.arange(d_min, va_hi + 1)
    for ep in range(epochs):
        rng.shuffle(days)
        tot, cnt = 0.0, 0
        for d in days:
            xs = gather_days(daily, np.array([d]), lags)[0].T
            tgt = daily[d]
            keep = np.isfinite(xs).sum(axis=1) >= max(3, len(lags) // 3)
            keep &= np.isfinite(tgt)
            if keep.sum() < 5:
                continue
            idx = np.where(keep)[0]
            x_in = np.concatenate([xs[idx], np.full((len(idx), 1), np.nan, np.float32)], axis=1)
            t_in = np.concatenate([xs[idx], tgt[idx][:, None]], axis=1).astype(np.float32)
            x = torch.tensor(x_in, device=DEVICE)
            tgt_t = torch.tensor(t_in, device=DEVICE)
            li = torch.tensor(idx, device=DEVICE, dtype=torch.long)
            if ctx_kind == 'dow':
                ctx = ctx_mod(torch.full((len(idx),), dows[d], device=DEVICE, dtype=torch.long))
            else:
                ang = 2 * np.pi * doys[d] / 365.25
                ctx = torch.tensor(np.tile([np.sin(ang), np.cos(ang)], (len(idx), 1)),
                                   device=DEVICE, dtype=torch.float32)
            obs = torch.isfinite(x)
            w = torch.ones_like(x)
            w[:, -1] = 4.0
            pred, _, _ = model.scan(li, ctx, x, obs, 1.0 / x.shape[1])
            m = torch.isfinite(tgt_t)
            loss = (w[m] * (pred[m] - torch.nan_to_num(tgt_t)[m]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 2.0)
            opt.step()
            tot += loss.item()
            cnt += 1
        print(f'[{name}:{panel}] epoch {ep + 1}/{epochs} wmse={tot / max(cnt, 1):.5f}', flush=True)
    return model, ctx_mod


@torch.no_grad()
def export_wy_v2(panel, daily, week, week_ctx, year, dows, doys):
    D = n_days()
    L = daily.shape[1]
    wy = np.zeros((D, L, 2), dtype=np.float32)
    li = torch.arange(L, device=DEVICE, dtype=torch.long)
    for d in range(D):
        for j, (model, ctx_mod, lags) in enumerate([(week, week_ctx, WEEK_LAGS), (year, None, YEAR_LAGS)]):
            xs = gather_days(daily, np.array([d]), lags)[0].T
            x_in = np.concatenate([xs, np.full((L, 1), np.nan, np.float32)], axis=1)
            x = torch.tensor(x_in, device=DEVICE)
            if ctx_mod is not None:
                ctx = ctx_mod(torch.full((L,), dows[d], device=DEVICE, dtype=torch.long))
            else:
                ang = 2 * np.pi * doys[d] / 365.25
                ctx = torch.tensor(np.tile([np.sin(ang), np.cos(ang)], (L, 1)),
                                   device=DEVICE, dtype=torch.float32)
            pred, _, _ = model.scan(li, ctx, x, torch.isfinite(x), 1.0 / x.shape[1])
            wy[d, :, j] = pred[:, -1].float().cpu().numpy()
    wy = np.clip(wy, 0.0, 1.5)
    np.savez_compressed(CACHE_DIR / f'ode_feats_v2_{panel}.npz', wy=wy)
    return wy


@torch.no_grad()
def eval_day_mse(z, day_states, v1_day, v1_ctx, offset, dows, te_lo, te_hi):
    H = z['hourly']
    nd = te_hi - te_lo + 1
    L = H.shape[2]
    X = H[te_lo:te_hi + 1].transpose(0, 2, 1).reshape(nd * L, 24).astype(np.float32)
    dw = np.repeat(dows[te_lo:te_hi + 1], L)
    fin = np.isfinite(X)
    usable = (fin[:, :EVAL_CUT_HOUR].sum(1) >= 8) & fin[:, EVAL_CUT_HOUR:].any(1)
    x = torch.tensor(X, device=DEVICE)
    karr = torch.arange(24, device=DEVICE)[None, :]
    obs = torch.isfinite(x) & (karr < EVAL_CUT_HOUR)
    dw_t = torch.tensor(dw, device=DEVICE, dtype=torch.long)
    li = torch.tensor(offset + np.tile(np.arange(L), nd), device=DEVICE, dtype=torch.long)
    pred_v1, _, _ = v1_day.scan(li, v1_ctx(dw_t), x, obs, 1.0 / 24)
    pred_v1 = pred_v1.float().cpu().numpy()

    pred_v2 = np.full_like(X, np.nan)
    for l in range(L):
        st = day_states[l]
        if st is None:
            continue
        model = ODERNN(1, 8).to(DEVICE)
        model.load_state_dict(st['model'])
        model.eval()
        ctx_mod = DayCtx().to(DEVICE)
        ctx_mod.load_state_dict(st['ctx'])
        rows = np.arange(l, nd * L, L)
        p, _, _ = model.scan(torch.zeros(len(rows), device=DEVICE, dtype=torch.long),
                             ctx_mod(dw_t[rows]), x[rows], obs[rows], 1.0 / 24)
        pred_v2[rows] = p.float().cpu().numpy()

    tail = slice(EVAL_CUT_HOUR, 24)
    t = X[:, tail]
    m = usable[:, None] & np.isfinite(t) & np.isfinite(pred_v2[:, tail])
    mse_v1 = float(((pred_v1[:, tail] - np.nan_to_num(t))[m] ** 2).mean())
    mse_v2 = float(((pred_v2[:, tail] - np.nan_to_num(t))[m] ** 2).mean())
    return mse_v1, mse_v2


def eval_wy_mse(daily, wy_v1, wy_v2, te_lo, te_hi):
    t = daily[te_lo:te_hi + 1]
    m = np.isfinite(t)
    out = {}
    for j, name in ((0, 'week'), (1, 'year')):
        out[f'{name}_v1'] = float(((wy_v1[te_lo:te_hi + 1, :, j][m] - t[m]) ** 2).mean())
        out[f'{name}_v2'] = float(((wy_v2[te_lo:te_hi + 1, :, j][m] - t[m]) ** 2).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day-steps', type=int, default=60)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    tr_lo, _ = split_day_range('train')
    _, va_hi = split_day_range('val')
    te_lo, te_hi = split_day_range('test')
    dows = dow_array()
    dows3 = dow_array_val3()
    doys = doy_array()

    v1 = torch.load(CACHE_DIR / 'ode_models.pt', weights_only=False)
    v1_day = ODERNN(v1['n_links'], 8).to(DEVICE)
    v1_day.load_state_dict(v1['day'])
    v1_day.eval()
    v1_ctx = DayCtx().to(DEVICE)
    v1_ctx.load_state_dict(v1['day_ctx'])
    offsets = v1['offsets']

    ck = {'day': {}, 'week': {}, 'year': {},
          'week_lags': WEEK_LAGS, 'year_lags': YEAR_LAGS,
          'day_steps': args.day_steps, 'min_day_seqs': MIN_DAY_SEQS}
    report = {}
    for p in SCORED_PANELS:
        z = load_panel(p)
        daily = z['daily']

        day_states = train_panel_day_models(p, z, dows, dows3, tr_lo, va_hi, args.day_steps, rng)
        week_m, week_c = train_corridor_span(p, daily, WEEK_LAGS, 'dow', 'week', 5, rng, dows, doys, tr_lo, va_hi)
        year_m, _ = train_corridor_span(p, daily, YEAR_LAGS, 'doy', 'year', 4, rng, dows, doys, tr_lo, va_hi)

        ck['day'][p] = day_states
        ck['week'][p] = {'model': sd_cpu(week_m), 'ctx': sd_cpu(week_c)}
        ck['year'][p] = {'model': sd_cpu(year_m)}
        torch.save(ck, OUT_CKPT)

        wy_v2 = export_wy_v2(p, daily, week_m, week_c, year_m, dows, doys)
        wy_v1 = load_ode_feats(p)['wy']
        day_v1, day_v2 = eval_day_mse(z, day_states, v1_day, v1_ctx, offsets[p], dows, te_lo, te_hi)
        r = {'day_v1': day_v1, 'day_v2': day_v2}
        r.update(eval_wy_mse(daily, wy_v1, wy_v2, te_lo, te_hi))
        report[p] = r
        print(f'[eval:{p}] test MSE  day(cut17h): v1={day_v1:.5f} v2={day_v2:.5f}  '
              f'week: v1={r["week_v1"]:.5f} v2={r["week_v2"]:.5f}  '
              f'year: v1={r["year_v1"]:.5f} v2={r["year_v2"]:.5f}', flush=True)

    print('\npanel        day_v1   day_v2   week_v1  week_v2  year_v1  year_v2')
    for p, r in report.items():
        print(f'{p:<12} {r["day_v1"]:.5f}  {r["day_v2"]:.5f}  {r["week_v1"]:.5f}  '
              f'{r["week_v2"]:.5f}  {r["year_v1"]:.5f}  {r["year_v2"]:.5f}')
    mean = {k: float(np.mean([r[k] for r in report.values()]))
            for k in ('day_v1', 'day_v2', 'week_v1', 'week_v2', 'year_v1', 'year_v2')}
    print(f'{"MEAN":<12} {mean["day_v1"]:.5f}  {mean["day_v2"]:.5f}  {mean["week_v1"]:.5f}  '
          f'{mean["week_v2"]:.5f}  {mean["year_v1"]:.5f}  {mean["year_v2"]:.5f}')
    print(f'saved: {OUT_CKPT}', flush=True)


if __name__ == '__main__':
    main()
