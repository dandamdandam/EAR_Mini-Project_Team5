import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ode import DEVICE, DayCtx, ODERNN, build_link_index, dow_array, doy_array, rk4
from utils import (CACHE_DIR, DATA_START, OUT_DIR, SCORED_PANELS,
                   date_to_day_index, load_panel, split_day_range)

SURF, INK, MUT, GRID_C, BASE_C = '#fcfcfb', '#0b0b0b', '#898781', '#e1e0d9', '#c3c2b7'
C_TRUTH = '#2a78d6'
C_CURVES = ['#f0b08c', '#eb6834', '#9c3d14']
WEEK_CUTS = [3, 5, 7]
YEAR_CUTS = [10, 20, 30]


def style_ax(ax):
    ax.set_facecolor(SURF)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(BASE_C)
    ax.tick_params(colors=MUT, labelsize=8)
    ax.yaxis.grid(True, color=GRID_C, lw=0.8)
    ax.set_axisbelow(True)


def load_v1():
    ck = torch.load(CACHE_DIR / 'ode_models.pt', weights_only=False)
    day = ODERNN(ck['n_links'], 8).to(DEVICE)
    day.load_state_dict(ck['day'])
    day_ctx = DayCtx().to(DEVICE)
    day_ctx.load_state_dict(ck['day_ctx'])
    week = ODERNN(ck['n_links'], 8).to(DEVICE)
    week.load_state_dict(ck['week'])
    week_ctx = DayCtx().to(DEVICE)
    week_ctx.load_state_dict(ck['week_ctx'])
    year = ODERNN(ck['n_links'], 2).to(DEVICE)
    year.load_state_dict(ck['year'])
    for m in (day, week, year):
        m.eval()
    return ck, day, day_ctx, week, week_ctx, year


def load_v2_corridor(ck2, panel, L):
    week = ODERNN(L, 8).to(DEVICE)
    week.load_state_dict(ck2['week'][panel]['model'])
    week_ctx = DayCtx().to(DEVICE)
    week_ctx.load_state_dict(ck2['week'][panel]['ctx'])
    year = ODERNN(L, 2).to(DEVICE)
    year.load_state_dict(ck2['year'][panel]['model'])
    for m in (week, year):
        m.eval()
    return week, week_ctx, year


def load_v2_link_day(ck2, panel, link):
    st = ck2['day'][panel][link]
    if st is None:
        return None, None
    model = ODERNN(1, 8).to(DEVICE)
    model.load_state_dict(st['model'])
    model.eval()
    ctx_mod = DayCtx().to(DEVICE)
    ctx_mod.load_state_dict(st['ctx'])
    return model, ctx_mod


@torch.no_grad()
def encode_then_solve(model, link_idx, ctx, x, obs, n_obs, substeps):
    K = x.shape[1]
    dt = 1.0 / K
    h, e = model.init_state(link_idx, ctx)
    for k in range(n_obs):
        h = rk4(model.f, h, dt)
        upd = model.gru(torch.nan_to_num(x[:, k:k + 1]), h)
        h = torch.where(obs[:, k:k + 1], upd, h)

    def dec(hh):
        return model.dec(torch.cat([hh, e], -1)).squeeze(-1).float().cpu().numpy()

    ts, ys = [n_obs * dt], [dec(h)]
    for s in range((K - n_obs) * substeps):
        h = rk4(model.f, h, dt / substeps)
        ts.append(dt * (n_obs + (s + 1) / substeps))
        ys.append(dec(h))
    return np.array(ts), np.clip(np.stack(ys, 0), 0.0, 1.5)


@torch.no_grad()
def dense_solve(model, link_idx, ctx, x, obs, substeps):
    K = x.shape[1]
    dt = 1.0 / K
    h, e = model.init_state(link_idx, ctx)

    def dec(hh):
        return model.dec(torch.cat([hh, e], -1)).squeeze(-1).float().cpu().numpy()

    ts, ys = [0.0], [dec(h)]
    for k in range(K):
        for s in range(substeps):
            h = rk4(model.f, h, dt / substeps)
            ts.append(dt * (k + (s + 1) / substeps))
            ys.append(dec(h))
        upd = model.gru(torch.nan_to_num(x[:, k:k + 1]), h)
        h = torch.where(obs[:, k:k + 1], upd, h)
        ts.append(dt * (k + 1))
        ys.append(dec(h))
    return np.array(ts), np.clip(np.stack(ys, 0), 0.0, 1.5)


def pick_day(z, lo, hi):
    H = z['hourly']
    best_d, best_v = lo, np.inf
    for d in range(lo, hi + 1):
        hd = H[d]
        if np.isfinite(hd).mean() < 0.5:
            continue
        v = np.nanmin(hd)
        if v < best_v:
            best_d, best_v = d, v
    return best_d


def pick_links(z, d):
    hd = z['hourly'][d]
    ok = np.isfinite(hd).sum(axis=0) >= 20
    m = np.where(ok, np.nanmin(np.where(np.isfinite(hd), hd, np.inf), axis=0), np.inf)
    order = np.argsort(m)
    n_ok = int(ok.sum())
    return [int(order[0]), int(order[n_ok // 2]), int(order[n_ok - 1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', default='D7_I10_E', choices=SCORED_PANELS)
    ap.add_argument('--date', default=None)
    ap.add_argument('--hour', type=float, default=17.0)
    ap.add_argument('--links', type=int, nargs=3, default=None)
    ap.add_argument('--v2', action='store_true')
    ap.add_argument('--odernn', action='store_true')
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()
    stem = 'fig_ode_curves' + ('_v2' if args.v2 else '') + ('_odernn' if args.odernn else '')
    out = args.out or OUT_DIR / f'{stem}.png'

    dows = dow_array()
    doys = doy_array()
    z = load_panel(args.panel)
    L = z['hourly'].shape[2]

    lo, hi = split_day_range('test')
    d = date_to_day_index(args.date) if args.date else pick_day(z, lo, hi)
    auto = args.links is None
    links = pick_links(z, d) if auto else args.links
    tags = ['most congested', 'median', 'least congested'] if auto else ['', '', '']
    date_str = (DATA_START + pd.Timedelta(days=int(d))).strftime('%Y-%m-%d (%a)')
    mode = ('v2 per-link/per-corridor' if args.v2 else 'v1 global') + \
           (', ODE-RNN filter' if args.odernn else ', latent ODE')
    print(f'panel={args.panel} day={date_str} (idx {int(d)}) links={links} hour={args.hour} [{mode}]', flush=True)

    if args.v2:
        ck2 = torch.load(CACHE_DIR / 'ode_models_v2.pt', weights_only=False)
        week, week_ctx, year = load_v2_corridor(ck2, args.panel, L)
        wl = np.asarray(ck2['week_lags'])
        yl = np.asarray(ck2['year_lags'])
        li_span = torch.tensor(links, device=DEVICE, dtype=torch.long)
    else:
        offsets, _ = build_link_index()
        ck, day, day_ctx, week, week_ctx, year = load_v1()
        off = offsets[args.panel]
        wl = np.asarray(ck['week_lags'])
        yl = np.asarray(ck['year_lags'])
        li_span = torch.tensor([off + l for l in links], device=DEVICE, dtype=torch.long)

    n_full = int(round(args.hour * 12)) // 12
    day_cuts = sorted(set([8, 12, n_full]))
    B = len(links)
    dow_d = int(dows[d])
    hours_grid = torch.arange(24, device=DEVICE)[None, :]
    colors = [C_CURVES[1]] if args.odernn else C_CURVES

    day_curves = []
    for j, l in enumerate(links):
        if args.v2:
            dm, dc = load_v2_link_day(ck2, args.panel, l)
            if dm is None:
                day_curves.append(None)
                continue
            li_l = torch.zeros(1, device=DEVICE, dtype=torch.long)
        else:
            dm, dc = day, day_ctx
            li_l = li_span[j:j + 1]
        x_l = torch.tensor(z['hourly'][d][:, l][None, :].astype(np.float32), device=DEVICE)
        ctx_l = dc(torch.full((1,), dow_d, device=DEVICE, dtype=torch.long))
        if args.odernn:
            obs_l = torch.isfinite(x_l) & (hours_grid < n_full)
            day_curves.append([(dense_solve(dm, li_l, ctx_l, x_l, obs_l, 12),
                                f'ODE-RNN ({n_full}h observed)')])
        else:
            day_curves.append([(encode_then_solve(dm, li_l, ctx_l, x_l,
                                                  torch.isfinite(x_l) & (hours_grid < cut), cut, 12),
                                f'latent ODE ({cut}h observed)') for cut in day_cuts])

    daily = z['daily']
    xs_w = daily[d + wl][:, links].T.astype(np.float32)
    xw = torch.tensor(np.concatenate([xs_w, np.full((B, 1), np.nan, np.float32)], axis=1), device=DEVICE)
    ctx_w = week_ctx(torch.full((B,), dow_d, device=DEVICE, dtype=torch.long))
    if args.odernn:
        week_curves = [(dense_solve(week, li_span, ctx_w, xw, torch.isfinite(xw), 10), 'ODE-RNN')]
    else:
        week_curves = [(encode_then_solve(week, li_span, ctx_w, xw, torch.isfinite(xw), cut, 10),
                        f'latent ODE ({cut}d observed)') for cut in WEEK_CUTS]

    xs_y = daily[d + yl][:, links].T.astype(np.float32)
    xy = torch.tensor(np.concatenate([xs_y, np.full((B, 1), np.nan, np.float32)], axis=1), device=DEVICE)
    ang = 2 * np.pi * doys[d] / 365.25
    ctx_y = torch.tensor(np.tile([np.sin(ang), np.cos(ang)], (B, 1)), device=DEVICE, dtype=torch.float32)
    if args.odernn:
        year_curves = [(dense_solve(year, li_span, ctx_y, xy, torch.isfinite(xy), 4), 'ODE-RNN')]
    else:
        year_curves = [(encode_then_solve(year, li_span, ctx_y, xy, torch.isfinite(xy), cut, 4),
                        f'latent ODE ({cut}pt observed)') for cut in YEAR_CUTS]

    fig, axes = plt.subplots(3, B, figsize=(4.6 * B, 10.2), facecolor=SURF)
    hh = np.arange(24)
    for j in range(B):
        ax = axes[0, j]
        style_ax(ax)
        truth = z['hourly'][d][:, links[j]]
        tm = np.isfinite(truth)
        if day_curves[j] is not None:
            for color, ((t, y), label) in zip(colors, day_curves[j]):
                ax.plot(24 * t, y[:, 0], color=color, lw=1.6, label=label)
        ax.plot((hh + 1)[tm], truth[tm], 'o', color=C_TRUTH, ms=4.5, ls='none', label='true values')
        ax.set_xlim(0, 24)
        ax.set_ylim(0, 1.35)
        ax.set_title(f'link {links[j]} ({tags[j]}) — day ODE', color=INK, fontsize=9.5)
        ax.set_xlabel('hour of day', color=MUT, fontsize=8)
        if j == 0:
            ax.set_ylabel('speed ratio', color=MUT, fontsize=8)
            ax.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc='lower left')

        ax = axes[1, j]
        style_ax(ax)
        for color, ((t, y), label) in zip(colors, week_curves):
            ax.plot(-8 + 8 * t, y[:, j], color=color, lw=1.6, label=label)
        fin = np.isfinite(xs_w[j])
        ax.plot(wl[fin], xs_w[j][fin], 'o', color=C_TRUTH, ms=4.5, ls='none')
        if np.isfinite(daily[d, links[j]]):
            ax.plot(0, daily[d, links[j]], 'o', color=C_TRUTH, ms=4.5)
        ax.set_xlim(-8.1, 0.5)
        ax.set_ylim(0, 1.35)
        ax.set_title(f'link {links[j]} — week ODE', color=INK, fontsize=9.5)
        ax.set_xlabel('days before target', color=MUT, fontsize=8)
        if j == 0:
            ax.set_ylabel('speed ratio (daily mean)', color=MUT, fontsize=8)
            ax.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc='lower left')

        ax = axes[2, j]
        style_ax(ax)
        for color, ((t, y), label) in zip(colors, year_curves):
            ax.plot(-93 + 93 * t, y[:, j], color=color, lw=1.6, label=label)
        fin = np.isfinite(xs_y[j])
        ax.plot(yl[fin], xs_y[j][fin], 'o', color=C_TRUTH, ms=3.5, ls='none')
        if np.isfinite(daily[d, links[j]]):
            ax.plot(0, daily[d, links[j]], 'o', color=C_TRUTH, ms=3.5)
        ax.set_xlim(-95, 4)
        ax.set_ylim(0, 1.35)
        ax.set_title(f'link {links[j]} — year ODE', color=INK, fontsize=9.5)
        ax.set_xlabel('days before target', color=MUT, fontsize=8)
        if j == 0:
            ax.set_ylabel('speed ratio (daily mean)', color=MUT, fontsize=8)
            ax.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc='lower left')

    fig.suptitle(f'ODE curves [{mode}] — {args.panel}, {date_str}', color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=SURF)
    plt.close(fig)
    print(f'saved: {out}', flush=True)


if __name__ == '__main__':
    main()
