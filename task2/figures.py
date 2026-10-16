import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import DATA_START, OUT_DIR, SCORED_PANELS, load_panel, n_days

BLUE = '#2a78d6'
ORANGE = '#eb6834'
GREY = '#9a9a9a'
INK = '#222222'


def style_ax(ax, title):
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    ax.grid(True, color='#e6e6e6', linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=11, loc='left', pad=8)


def plot_day(ax, z, dows):
    hc = np.nanmean(z['hourly'], axis=2)
    hours = np.arange(24)
    wd = dows < 5
    for d in range(hc.shape[0]):
        ax.plot(hours, hc[d], color=GREY, alpha=0.06, lw=0.7, zorder=1)
    prof_wd = np.nanmean(hc[wd], axis=0)
    prof_we = np.nanmean(hc[~wd], axis=0)
    ax.plot(hours, prof_wd, color=BLUE, lw=2.2, zorder=3, label='weekday mean')
    ax.plot(hours, prof_we, color=ORANGE, lw=2.2, zorder=3, label='weekend mean')
    pred = np.where(wd[:, None], prof_wd[None, :], prof_we[None, :])
    r2 = 1 - np.nansum((hc - pred) ** 2) / np.nansum((hc - np.nanmean(hc)) ** 2)
    ax.set_xlim(0, 23)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xlabel('hour of day')
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    return r2


def plot_week(ax, z, dows):
    dc = np.nanmean(z['daily'], axis=1)
    D = len(dc)
    first_monday = int(np.where(dows == 0)[0][0])
    W = np.array([dc[s:s + 7] for s in range(first_monday, D - 6, 7)])
    x = np.arange(7)
    for w in W:
        ax.plot(x, w, color=GREY, alpha=0.18, lw=0.8, zorder=1)
    prof = np.nanmean(W, axis=0)
    ax.plot(x, prof, color=BLUE, lw=2.2, zorder=3, label='mean week')
    r2 = 1 - np.nansum((W - prof[None, :]) ** 2) / np.nansum((W - np.nanmean(W)) ** 2)
    s = pd.Series(dc)
    ax.set_xticks(x)
    ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    ax.set_xlabel('day of week')
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    return r2, s.autocorr(1), s.autocorr(7)


def plot_year(ax, z):
    dc = np.nanmean(z['daily'], axis=1)
    dates = pd.date_range(DATA_START.tz_localize(None), periods=len(dc))
    ax.scatter(dates, dc, s=7, color=GREY, alpha=0.5, lw=0, zorder=2, label='daily mean')
    roll = pd.Series(dc).rolling(14, center=True, min_periods=7).mean()
    ax.plot(dates, roll, color=INK, lw=1.8, zorder=3, label='14-day rolling mean')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.set_xlabel('date')
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    return float(np.nanmax(roll) - np.nanmin(roll))


def main():
    OUT_DIR.mkdir(exist_ok=True)
    dows = np.array([(DATA_START + pd.Timedelta(days=i)).dayofweek for i in range(n_days())])
    n = len(SCORED_PANELS)
    fig, axes = plt.subplots(n, 3, figsize=(14, 4.2 * n), dpi=150, squeeze=False)
    for i, panel in enumerate(SCORED_PANELS):
        z = load_panel(panel)
        r2d = plot_day(axes[i, 0], z, dows)
        style_ax(axes[i, 0], f'{panel} — day span')
        axes[i, 0].set_ylabel('corridor mean speed / free-flow')
        r2w, ac1, ac7 = plot_week(axes[i, 1], z, dows)
        style_ax(axes[i, 1], f'{panel} — week span')
        drift = plot_year(axes[i, 2], z)
        style_ax(axes[i, 2], f'{panel} — year span')
        print(panel, {'r2_day': float(r2d), 'r2_week': float(r2w),
                      'autocorr_lag1': float(ac1), 'autocorr_lag7': float(ac7),
                      'year_drift': drift})
    fig.tight_layout()
    out = OUT_DIR / 'fig_speed_overlap.png'
    fig.savefig(out, bbox_inches='tight')
    print('saved', out)


if __name__ == '__main__':
    main()
