import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figures import BLUE, GREY, INK, ORANGE, style_ax
from utils import KROOT, OUT_DIR, QUEUE_RATIO, SCORED_PANELS, load_panel, time_grid

R2_PANELS = ['D7_I10_E', 'D7_I405_N', 'D12_I5_S']
R2_DAYS = 12


def window_stats():
    offs, rates = [], []
    split_rate = {}
    for p in SCORED_PANELS:
        for split in ['train', 'validation', 'private']:
            h = pd.read_parquet(KROOT / 'task2' / p / split / 'window_history.parquet',
                                columns=['window_id', 'timestamp', 'link_id', 'speed_kmh'])
            h['timestamp'] = pd.to_datetime(h.timestamp, utc=True)
            wi = pd.read_csv(KROOT / 'task2' / p / split / 'window_index.csv')
            org = pd.to_datetime(wi.set_index('window_id').forecast_origin, utc=True)
            off = ((h.timestamp - h.window_id.map(org)).dt.total_seconds() / 60).astype(int)
            miss = h.speed_kmh.isna()
            offs.append(off.to_numpy())
            rates.append(miss.to_numpy())
            split_rate.setdefault(split, []).append(miss.mean())
    off = np.concatenate(offs)
    miss = np.concatenate(rates)
    by_off = {o: miss[off == o].mean() for o in sorted(np.unique(off))}
    return by_off, {k: float(np.mean(v)) for k, v in split_rate.items()}


def corridor_stats():
    link_rates = []
    hour = np.zeros((24, 2))
    markov = np.zeros((2, 2))
    dec_proxy = np.zeros((10, 2))
    dec_lastobs = np.zeros((10, 2))
    q_nat = np.zeros((2, 2))
    run_counts = np.zeros(80)
    for p in SCORED_PANELS:
        z = load_panel(p)
        s = z['s_raw']
        vf = z['vf']
        miss = np.isnan(s)
        link_rates.append(miss.mean(axis=0))
        hrs = (np.arange(s.shape[0]) // 12) % 24
        for hh in range(24):
            m = hrs == hh
            hour[hh] += [miss[m].sum(), m.sum() * s.shape[1]]
        prev = miss[:-1]
        cur = miss[1:]
        markov[0] += [cur[prev].sum(), prev.sum()]
        markov[1] += [cur[~prev].sum(), (~prev).sum()]
        xt = z['s_true'] / vf[None]
        dec = np.clip((xt * 10).astype(int), 0, 9)
        for d in range(10):
            m = dec == d
            dec_proxy[d] += [miss[m].sum(), m.sum()]
        xo = np.where(miss, np.nan, xt)
        last = np.vstack([np.full((1, s.shape[1]), np.nan), xo[:-1]])
        ok = np.isfinite(last)
        dl = np.clip(np.nan_to_num(last * 10), 0, 9).astype(int)
        for d in range(10):
            m = ok & (dl == d)
            dec_lastobs[d] += [miss[m].sum(), m.sum()]
        qd = z['q']
        q_nat[0] += [miss[qd].sum(), qd.sum()]
        q_nat[1] += [miss[~qd].sum(), (~qd).sum()]
        col = np.vstack([miss, np.zeros((1, miss.shape[1]), bool)]).T.reshape(-1)
        d = np.diff(col.astype(np.int8))
        starts = np.flatnonzero(d == 1) + 1
        ends = np.flatnonzero(d == -1) + 1
        if col[0]:
            starts = np.concatenate([[0], starts])
        lens = ends - starts
        for L_ in lens:
            if L_ < 80:
                run_counts[L_] += 1
    return (np.concatenate(link_rates), hour[:, 0] / hour[:, 1],
            markov[:, 0] / markov[:, 1], dec_proxy, dec_lastobs,
            q_nat[:, 0] / q_nat[:, 1], run_counts)


def r2_stats():
    grid = time_grid()
    tidx = pd.Series(np.arange(len(grid)), index=grid)
    dec = np.zeros((10, 2))
    q = np.zeros((2, 2))
    for p in R2_PANELS:
        z = load_panel(p)
        s = z['s_raw']
        vf = z['vf']
        links = [str(l) for l in z['links']]
        fs = sorted(glob.glob(str(KROOT / 'corridors' / p / 'train' / 'mainline_states_masked' / 'mask_regime=R2' / '*.parquet')))[:R2_DAYS]
        d = pd.concat([pd.read_parquet(f, columns=['timestamp', 'link_id', 'speed_kmh']) for f in fs])
        d['timestamp'] = pd.to_datetime(d.timestamp, utc=True)
        pv = d.pivot_table(index='timestamp', columns='link_id', values='speed_kmh', aggfunc='mean').reindex(columns=links)
        rows = tidx.loc[pv.index].to_numpy()
        sm = pv.to_numpy(np.float32)
        su = s[rows]
        extra = np.isnan(sm) & np.isfinite(su)
        known = np.isfinite(su)
        xv = su / vf[None]
        dd = np.clip(np.nan_to_num(xv * 10), 0, 9).astype(int)
        for k in range(10):
            m = known & (dd == k)
            dec[k] += [extra[m].sum(), m.sum()]
        qq = known & (xv <= QUEUE_RATIO)
        q[0] += [extra[qq].sum(), qq.sum()]
        nq = known & (xv > QUEUE_RATIO)
        q[1] += [extra[nq].sum(), nq.sum()]
    return dec, q[:, 0] / q[:, 1]


def main():
    by_off, by_split = window_stats()
    link_rates, hour_rate, markov, dec_proxy, dec_lastobs, q_nat, run_counts = corridor_stats()
    dec_r2, q_r2 = r2_stats()
    p_marg = float(np.mean(link_rates))

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), dpi=150)

    ax = axes[0, 0]
    xs = sorted(by_off)
    ax.plot(xs, [by_off[o] for o in xs], color=BLUE, lw=2.2, marker='o', ms=4, label='windows, by step offset (min)')
    ax.plot(range(24), hour_rate, color=ORANGE, lw=2.2, label='corridor, by hour of day')
    ax.axhline(p_marg, color=GREY, lw=1, ls=':')
    ax.set_ylim(0, 0.36)
    ax.set_xlabel('step offset (min) / hour of day')
    ax.set_ylabel('P(missing)')
    style_ax(ax, 'P(missing) by window step and hour of day')
    ax.legend(frameon=False, fontsize=9, loc='lower right')

    ax = axes[0, 1]
    ks = np.arange(1, 31)
    obs_frac = run_counts[1:31] / run_counts.sum()
    geo = (p_marg ** (ks - 1)) * (1 - p_marg)
    ax.semilogy(ks, obs_frac, color=BLUE, lw=2.2, marker='o', ms=4, label='observed')
    ax.semilogy(ks, geo, color=GREY, lw=1.6, ls='--', label='geometric (iid) null')
    ax.set_ylim(1e-7, 1.5)
    ax.set_xlabel('missing-run length (steps of 5 min)')
    ax.set_ylabel('fraction of runs')
    style_ax(ax, 'Missing-run length distribution')
    ax.legend(frameon=False, fontsize=9, loc='upper right')

    ax = axes[1, 0]
    xc = np.arange(10) / 10 + 0.05
    r_last = dec_lastobs[:, 0] / np.maximum(dec_lastobs[:, 1], 1)
    r_proxy = dec_proxy[:, 0] / np.maximum(dec_proxy[:, 1], 1)
    r_r2 = dec_r2[:, 0] / np.maximum(dec_r2[:, 1], 1)
    ok_l = dec_lastobs[:, 1] > 2e4
    ok_p = dec_proxy[:, 1] > 2e4
    ok_r = dec_r2[:, 1] > 2e4
    ax.plot(xc[ok_r], r_r2[ok_r], color=ORANGE, lw=2.2, marker='o', ms=4, label='R2 mask layer (true value)')
    ax.plot(xc[ok_l], r_last[ok_l], color=BLUE, lw=2.2, marker='o', ms=4, label='natural, last observed value')
    ax.plot(xc[ok_p], r_proxy[ok_p], color=GREY, lw=1.6, ls='--', marker='o', ms=3, label='natural, interpolated value')
    ax.axvline(QUEUE_RATIO, color=GREY, lw=1, ls=':')
    ax.set_ylim(0, 0.45)
    ax.set_xlabel('speed / free-flow')
    ax.set_ylabel('P(missing)')
    style_ax(ax, 'P(missing) by speed value')
    ax.legend(frameon=False, fontsize=9, loc='upper right')

    ax = axes[1, 1]
    labels = ['natural\nqueued', 'natural\nfree-flow', 'R2 layer\nqueued', 'R2 layer\nfree-flow']
    vals = [q_nat[0], q_nat[1], q_r2[0], q_r2[1]]
    bars = ax.bar(labels, vals, color=[BLUE, BLUE, ORANGE, ORANGE], width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f'{v:.3f}', ha='center', fontsize=9, color=INK)
    ax.set_ylim(0, 0.36)
    ax.set_ylabel('P(missing)')
    style_ax(ax, 'P(missing): queued vs free-flow')

    fig.tight_layout()
    out = OUT_DIR / 'fig_missingness.png'
    fig.savefig(out, bbox_inches='tight')
    print('saved', out)
    print('splits:', by_split)
    print('link rate: mean %.4f std %.4f range %.3f..%.3f' % (p_marg, link_rates.std(), link_rates.min(), link_rates.max()))
    print('markov:', markov, 'queued/free nat:', q_nat, 'r2:', q_r2)


if __name__ == '__main__':
    main()
