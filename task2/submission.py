import argparse

import numpy as np
import pandas as pd
import torch

from features import ode_day_features, static_features
from model import QueueNet
from ode import DEVICE, DayCtx, ODERNN, build_link_index, gather_days
from utils import (CACHE_DIR, DAY_STEPS, HIST, HOR, KROOT, SCORED_PANELS, SEQ,
                   STEP_OFFSETS, TASK_DIR, interp_fill, load_panel)

SPLIT_START = {'validation': '2031-03-01', 'private': '2031-04-01'}
SPLIT_NDAYS = {'validation': 31, 'private': 30}
TRAIN_DAYS = 273
DEFAULT_OUT = TASK_DIR.parent / 'outputs' / 'queue_submission.csv'


def month_grid(split):
    start = pd.Timestamp(SPLIT_START[split], tz='UTC')
    return pd.date_range(start, periods=SPLIT_NDAYS[split] * DAY_STEPS, freq='5min')


def masked_month(panel, split, links, vf):
    d = pd.read_parquet(KROOT / 'corridors' / panel / split / 'mainline_states_masked' / 'mask_regime=R2',
                        columns=['timestamp', 'link_id', 'speed_kmh'])
    d['timestamp'] = pd.to_datetime(d['timestamp'], utc=True)
    p = d.pivot_table(index='timestamp', columns='link_id', values='speed_kmh', aggfunc='mean')
    p = p.reindex(index=month_grid(split), columns=links)
    x = p.to_numpy(np.float32) / vf[None, :]
    D = SPLIT_NDAYS[split]
    hourly = np.nanmean(x.reshape(D, 24, DAY_STEPS // 24, -1), axis=2).astype(np.float32)
    daily = np.nanmean(x.reshape(D, DAY_STEPS, -1), axis=1).astype(np.float32)
    return hourly, daily


@torch.no_grad()
def wy_for_days(ck, week, week_ctx, year, offset, daily_ext, g_days, dows, doys, L):
    out = np.zeros((len(g_days), L, 2), np.float32)
    for i, g in enumerate(g_days):
        for j, (model, ctx_mod, lags) in enumerate([(week, week_ctx, ck['week_lags']), (year, None, ck['year_lags'])]):
            xs = gather_days(daily_ext, np.array([g]), lags)[0].T
            x_in = np.concatenate([xs, np.full((L, 1), np.nan, np.float32)], axis=1)
            x = torch.tensor(x_in, device=DEVICE)
            li = torch.tensor(offset + np.arange(L), device=DEVICE, dtype=torch.long)
            if ctx_mod is not None:
                ctx = ctx_mod(torch.full((L,), int(dows[i]), device=DEVICE, dtype=torch.long))
            else:
                ang = 2 * np.pi * doys[i] / 365.25
                ctx = torch.tensor(np.tile([np.sin(ang), np.cos(ang)], (L, 1)), device=DEVICE, dtype=torch.float32)
            obs = torch.isfinite(x)
            pred, _, _ = model.scan(li, ctx, x, obs, 1.0 / x.shape[1])
            out[i, :, j] = pred[:, -1].float().cpu().numpy()
    return np.clip(out, 0.0, 1.5)


def build_panel_split(panel, z, split, models, qmodel, val_daily, tau):
    day_model, day_ctx, week_model, week_ctx, year_model, ck, offset = models
    links = [str(l) for l in z['links']]
    L = len(links)
    vf = z['vf']
    cap = z['cap']
    tdir = KROOT / 'task2' / panel / split
    wi = pd.read_csv(tdir / 'window_index.csv')
    hist = pd.read_parquet(tdir / 'window_history.parquet')
    hist['timestamp'] = pd.to_datetime(hist['timestamp'], utc=True)
    hist['link_id'] = hist['link_id'].astype(str)
    tmpl = pd.read_csv(tdir / 'sample_submission_queue.csv', dtype={'window_id': str, 'link_id': str})

    if split == 'validation':
        hourly_m = z['hourly3']
        g_base = TRAIN_DAYS
        daily_ext = np.concatenate([z['daily'], val_daily], axis=0)
    else:
        hourly_m, daily_m = masked_month(panel, split, links, vf)
        g_base = TRAIN_DAYS + SPLIT_NDAYS['validation']
        daily_ext = np.concatenate([z['daily'], val_daily, daily_m], axis=0)

    N = len(wi)
    origins = pd.to_datetime(wi['forecast_origin'], utc=True)
    month_start = pd.Timestamp(SPLIT_START[split], tz='UTC')
    day_idx = ((origins - month_start).dt.days).to_numpy()
    slots = ((origins - origins.dt.normalize()).dt.total_seconds() // 300).to_numpy(np.int64)
    dows = origins.dt.dayofweek.to_numpy()
    doys = origins.dt.dayofyear.to_numpy()

    sph = np.full((N, HIST, L), np.nan, np.float32)
    flh = np.full((N, HIST, L), np.nan, np.float32)
    hourly_win = np.empty((N, 24, L), np.float32)
    for n, (wid, T) in enumerate(zip(wi['window_id'], origins)):
        grid13 = pd.date_range(T - pd.Timedelta(minutes=60), T, freq='5min')
        h = hist[hist.window_id == wid]
        ps = h.pivot_table(index='timestamp', columns='link_id', values='speed_kmh', aggfunc='mean').reindex(index=grid13, columns=links)
        pf = h.pivot_table(index='timestamp', columns='link_id', values='flow_vph', aggfunc='mean').reindex(index=grid13, columns=links)
        sph[n] = ps.to_numpy(np.float32) / vf[None, :]
        flh[n] = pf.to_numpy(np.float32) / cap[None, :]
        hd = hourly_m[day_idx[n]].copy()
        hrs = ((grid13 - grid13.normalize()).total_seconds() // 3600).astype(int)
        for hh in np.unique(hrs):
            vals = np.nanmean(sph[n][hrs == hh], axis=0)
            fill = np.isnan(hd[hh]) & np.isfinite(vals)
            hd[hh][fill] = vals[fill]
        hourly_win[n] = hd

    obs = np.isfinite(sph).astype(np.float32)
    sph = interp_fill(sph.transpose(1, 0, 2).reshape(HIST, -1)).reshape(HIST, N, L).transpose(1, 0, 2)
    flh = interp_fill(flh.transpose(1, 0, 2).reshape(HIST, -1)).reshape(HIST, N, L).transpose(1, 0, 2)

    step_slot = slots[:, None] + STEP_OFFSETS[None, :]
    ang_t = 2 * np.pi * step_slot / DAY_STEPS
    ang_d = 2 * np.pi * (dows[:, None] + step_slot * 5.0 / 1440.0) / 7.0
    tf = np.stack([np.sin(ang_t), np.cos(ang_t), np.sin(ang_d), np.cos(ang_d)], axis=-1).astype(np.float32)

    odd = ode_day_features(day_model, day_ctx, offset, hourly_win, np.arange(N), slots, L, dows)
    nanmask = ~np.isfinite(sph)
    sph[nanmask] = odd[:, :HIST, :][nanmask]
    flh[~np.isfinite(flh)] = 0.4

    wy = wy_for_days(ck, week_model, week_ctx, year_model, offset, daily_ext, g_base + day_idx, dows, doys, L)

    b = {'sph': torch.tensor(sph, device=DEVICE), 'obs': torch.tensor(obs, device=DEVICE),
         'flh': torch.tensor(flh, device=DEVICE), 'tf': torch.tensor(tf, device=DEVICE),
         'odd': torch.tensor(odd, device=DEVICE), 'wy': torch.tensor(wy, device=DEVICE),
         'stat': torch.tensor(static_features(z), device=DEVICE),
         'st': torch.zeros((N, SEQ, L), device=DEVICE)}
    with torch.no_grad():
        logits, _ = qmodel(b, teacher_prob=0.0)
        probs = torch.sigmoid(logits[:, HIST - 1:SEQ]).cpu().numpy()
    preds = (probs >= tau).astype(np.int64)

    rows = []
    for n, (wid, T) in enumerate(zip(wi['window_id'], origins)):
        for k in range(HOR):
            ts = (T + pd.Timedelta(minutes=5 * (k + 1))).strftime('%Y-%m-%dT%H:%M:%SZ')
            rows.append(pd.DataFrame({'window_id': wid, 'timestamp': ts, 'link_id': links, 'queue_pred': preds[n, k]}))
    pred_df = pd.concat(rows, ignore_index=True)
    out = tmpl[['window_id', 'timestamp', 'link_id']].merge(pred_df, on=['window_id', 'timestamp', 'link_id'], how='left')
    n_miss = int(out.queue_pred.isna().sum())
    out['queue_pred'] = out['queue_pred'].fillna(0).astype(int)
    print(f'[{panel} {split}] windows={N} rows={len(out)} pos_rate={out.queue_pred.mean():.4f} unmatched={n_miss}', flush=True)
    return out


def generate(splits=('validation', 'private'), out=str(DEFAULT_OUT)):
    offsets, n_links = build_link_index()
    ck = torch.load(CACHE_DIR / 'ode_models.pt', weights_only=False)
    day_model = ODERNN(ck['n_links'], 8).to(DEVICE)
    day_model.load_state_dict(ck['day'])
    day_model.eval()
    day_ctx = DayCtx().to(DEVICE)
    day_ctx.load_state_dict(ck['day_ctx'])
    week_model = ODERNN(ck['n_links'], 8).to(DEVICE)
    week_model.load_state_dict(ck['week'])
    week_model.eval()
    week_ctx = DayCtx().to(DEVICE)
    week_ctx.load_state_dict(ck['week_ctx'])
    year_model = ODERNN(ck['n_links'], 2).to(DEVICE)
    year_model.load_state_dict(ck['year'])
    year_model.eval()

    ck2 = torch.load(CACHE_DIR / 't2_model.pt', weights_only=False)
    qmodel = QueueNet().to(DEVICE)
    qmodel.load_state_dict(ck2['model'])
    qmodel.eval()
    tau = ck2['tau']
    print(f'loaded model tau={tau:.2f}', flush=True)

    parts = []
    for panel in SCORED_PANELS:
        z = load_panel(panel)
        val_daily = np.nanmean(z['hourly3'], axis=1).astype(np.float32)
        models = (day_model, day_ctx, week_model, week_ctx, year_model, ck, offsets[panel])
        for split in splits:
            parts.append(build_panel_split(panel, z, split, models, qmodel, val_daily, tau))
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(out, index=False)
    print(f'saved {out}: {len(df)} rows, pos_rate={df.queue_pred.mean():.4f}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', nargs='+', default=['validation', 'private'])
    ap.add_argument('--out', default=str(DEFAULT_OUT))
    args = ap.parse_args()
    generate(args.splits, args.out)


if __name__ == '__main__':
    main()
