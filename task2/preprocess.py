import warnings

import numpy as np
import pandas as pd

from utils import (CACHE_DIR, DAY_STEPS, KROOT, QUEUE_RATIO, SCORED_PANELS,
                   interp_fill, n_days, split_day_range, time_grid)

warnings.filterwarnings('ignore')

MONTHS = ['2030-06', '2030-07', '2030-08', '2030-09', '2030-10', '2030-11', '2030-12', '2031-01', '2031-02']
COLS = ['timestamp', 'link_id', 'speed_kmh', 'flow_vph', 'milepost']


def pivot_to_grid(df, grid, links, value_col):
    p = df.pivot_table(index='timestamp', columns='link_id', values=value_col, aggfunc='mean')
    p = p.reindex(index=grid, columns=links)
    return p.to_numpy(dtype=np.float32)


def build_panel(panel):
    net_dir = KROOT / 'corridors' / panel / 'network'
    fd = pd.read_csv(net_dir / 'fd_parameters.csv', dtype={'link_id': str}).drop_duplicates('link_id').set_index('link_id')
    topo = pd.read_csv(net_dir / 'lwr_mainline_topology.csv', dtype=str).drop_duplicates('mainline_link_id').set_index('mainline_link_id')
    mainline = set(topo.index)

    def count_conn(l, col):
        if l not in mainline:
            return 0.0
        return float(sum(1 for e in str(topo.at[l, col]).split(';') if e.startswith('CONN')))

    grid = time_grid()

    s_parts = []
    milepost_map = {}
    for m in MONTHS:
        df = pd.read_parquet(KROOT / 'corridors' / panel / 'train' / 'mainline_states', filters=[('year_month', '=', m)], columns=COLS)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        for lid, mp in df.drop_duplicates('link_id')[['link_id', 'milepost']].itertuples(index=False):
            milepost_map.setdefault(lid, mp)
        s_parts.append(df)

    df = pd.concat(s_parts, ignore_index=True)
    del s_parts

    links = sorted(set(milepost_map) & set(fd.index), key=lambda l: milepost_map[l])
    links_arr = np.array(links)

    s_raw = pivot_to_grid(df, grid, links, 'speed_kmh')
    f_raw = pivot_to_grid(df, grid, links, 'flow_vph')
    del df

    vf = fd.loc[links, 'free_speed_kmh'].to_numpy(np.float32)
    cap = fd.loc[links, 'capacity_vph'].to_numpy(np.float32)
    lanes = fd.loc[links, 'lanes'].to_numpy(np.float32)
    length = fd.loc[links, 'length_km'].to_numpy(np.float32)
    milepost = np.array([milepost_map[l] for l in links], dtype=np.float32)
    n_or = np.array([count_conn(l, 'incoming_link_ids') for l in links], dtype=np.float32)
    n_fr = np.array([count_conn(l, 'outgoing_link_ids') for l in links], dtype=np.float32)

    s_true = interp_fill(s_raw)
    dead = ~np.isfinite(s_true).all(axis=0)
    s_true[:, dead] = vf[dead]
    q = s_true <= QUEUE_RATIO * vf[None, :]

    D = n_days()
    x = s_raw / vf[None, :]
    hourly = np.nanmean(x.reshape(D, 24, DAY_STEPS // 24, -1), axis=2).astype(np.float32)
    daily = np.nanmean(x.reshape(D, DAY_STEPS, -1), axis=1).astype(np.float32)

    val3_dir = KROOT / 'corridors' / panel / 'validation' / 'mainline_states_masked' / 'mask_regime=R2'
    dv = pd.read_parquet(val3_dir, columns=['timestamp', 'link_id', 'speed_kmh'])
    dv['timestamp'] = pd.to_datetime(dv['timestamp'], utc=True)
    grid3 = pd.date_range('2031-03-01', '2031-03-31 23:55', freq='5min', tz='UTC')
    s3 = pivot_to_grid(dv, grid3, links, 'speed_kmh')
    x3 = s3 / vf[None, :]
    hourly3 = np.nanmean(x3.reshape(31, 24, DAY_STEPS // 24, -1), axis=2).astype(np.float32)

    CACHE_DIR.mkdir(exist_ok=True)
    np.savez_compressed(
        CACHE_DIR / f'{panel}.npz',
        links=links_arr, vf=vf, cap=cap, lanes=lanes, length=length, milepost=milepost,
        n_or=n_or, n_fr=n_fr, s_raw=s_raw, f_raw=f_raw, s_true=s_true, q=q,
        hourly=hourly, daily=daily, hourly3=hourly3,
    )
    print(f'[{panel}] links={len(links)} grid={s_raw.shape[0]} nan_speed={np.isnan(s_raw).mean():.3f} queue_rate={q.mean():.4f}', flush=True)


def build_all():
    for p in SCORED_PANELS:
        build_panel(p)
    print('cache build complete')


def classify_origins(x, q):
    T, L = x.shape
    qraw = np.nan_to_num(x, nan=9.9) <= QUEUE_RATIO
    c = np.cumsum(qraw, axis=0, dtype=np.int32)
    r = c.copy()
    r[13:] -= c[:-13]
    est = r.max(axis=1) >= 2
    qa = q.any(axis=1)
    hq = np.zeros(T, bool)
    for k in range(1, 7):
        hq[:T - k] |= qa[k:]
    return est, hq


def pick_spaced(cands, cap, spacing, rng=None):
    if rng is not None:
        cands = cands.copy()
        rng.shuffle(cands)
        return np.sort(cands[:cap])
    out, last = [], -10 ** 9
    for t in cands:
        if t - last >= spacing:
            out.append(t)
            last = t
            if len(out) >= cap:
                break
    return np.array(out, dtype=np.int64)


def select_windows(panel, z, split, rng):
    vf = z['vf']
    x = (z['s_raw'] / vf[None]).astype(np.float32)
    est, hq = classify_origins(x, z['q'])
    lo, hi = split_day_range(split)
    T = x.shape[0]
    t_all = np.arange(T)
    slot = t_all % DAY_STEPS
    day = t_all // DAY_STEPS
    base = (slot >= 12) & (slot <= 281) & (day >= lo) & (day <= hi)
    onset = t_all[base & hq & ~est]
    ongoing = t_all[base & hq & est]
    neg = t_all[base & ~hq]
    if split == 'train':
        on_p = pick_spaced(onset, 1500, 0, rng)
        og_p = pick_spaced(ongoing, 750, 0, rng)
        ng_p = pick_spaced(neg, 300, 0, rng)
        pools = [on_p, og_p, ng_p[~est[ng_p]], ng_p[est[ng_p]]]
    else:
        pools = [pick_spaced(onset, 30, 24), pick_spaced(ongoing, 30, 24)]
    t0s = np.concatenate(pools)
    cond = np.concatenate([np.full(len(p), i, np.int64) for i, p in enumerate(pools)])
    order = np.argsort(t0s)
    return t0s[order], cond[order]


if __name__ == '__main__':
    build_all()
