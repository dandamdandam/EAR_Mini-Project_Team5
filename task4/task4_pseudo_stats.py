import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing import PANELS, RELEASE_ROOT
from .nnls import load_corridor_network
from .task4_pseudo_label import PSEUDO_DIR, load_labels, zone_geometry

WEEKDAYS = ['월', '화', '수', '목', '금', '토', '일']


def s_od(pred, true):
    return max(0.0, 1 - np.abs(pred - true).sum() / max(true.sum(), 1e-9))


def daily_table(lab, o, d, n):
    F = lab['F']
    trips = F.sum(axis=1)
    through = F[:, np.flatnonzero((o == 0) & (d == n - 1))].sum(axis=1)
    return pd.DataFrame({
        'b_mean': np.nanmean(lab['B'], axis=1),
        'q0': lab['q0'],
        'ramp_in': lab['O'][:, 1:n - 1].sum(axis=1),
        'ramp_out': lab['D'][:, 1:n - 1].sum(axis=1),
        'trips': trips,
        'through%': through / trips * 100,
    }, index=pd.to_datetime(lab['dates']))


def group_mean(t, key):
    g = t.groupby(key)
    out = g.mean()
    out.insert(0, 'days', g.size())
    return out


def path_variability(F, dates):
    mean, std = F.mean(axis=0), F.std(axis=0)
    big = mean >= 1
    wd = dates.dayofweek.to_numpy()
    wd_mean = {w: F[wd == w].mean(axis=0) for w in np.unique(wd)}
    months = dates.strftime('%Y-%m').to_numpy()
    month_mean = np.stack([F[months == m].mean(axis=0) for m in np.unique(months)])
    return {
        'path_cv_weighted': std.sum() / mean.sum(),
        'path_cv_median': np.median(std[big] / mean[big]),
        'S_od_day_by_mean': np.mean([s_od(mean, f) for f in F]),
        'S_od_day_by_weekday': np.mean([s_od(wd_mean[w], f) for w, f in zip(wd, F)]),
        'S_od_month_by_mean': np.mean([s_od(mean, f) for f in month_mean]),
    }, mean, std


def task4_count_means(release, panel, links):
    out = {}
    for split in ['train', 'validation', 'private']:
        p = release / 'task4' / panel / split / 'synthetic_link_counts.csv'
        if p.exists():
            c = pd.read_csv(p, dtype={'link_id': str}).set_index('link_id')['count']
            out[split] = c.reindex(links).mean()
    return out


def report_panel(release, panel, lab, paths):
    order, pos, o, d, n, start, end = zone_geometry(paths)
    t = daily_table(lab, o, d, n)
    dates = t.index
    print(f'\n===== {panel}  ({len(t)}일, {dates.min():%Y-%m-%d} ~ {dates.max():%Y-%m-%d}, '
          f'존 {n}개, 경로 {len(paths)}개)  단위 veh/h')

    wd = group_mean(t, dates.dayofweek)
    wd.index = [WEEKDAYS[i] for i in wd.index]
    print('\n[요일별 평균]')
    print(wd.round(1).to_string())

    print('\n[월별 평균]')
    print(group_mean(t, dates.strftime('%Y-%m')).round(1).to_string())

    links = lab['order'][~np.isnan(lab['B']).all(axis=0)]
    cm = task4_count_means(release, panel, links)
    month_b = t.b_mean.groupby(dates.strftime('%Y-%m')).mean()
    print('\n[task4 synthetic_link_counts 평균 (같은 관측 링크)]  '
          + '  '.join(f'{k} {v:.1f}' for k, v in cm.items())
          + f'  | train 월평균 b 범위 {month_b.min():.1f} ~ {month_b.max():.1f}')

    stats, mean, std = path_variability(lab['F'], dates)
    print('\n[경로별 날짜 변동]  '
          f'CV(가중) {stats["path_cv_weighted"]:.3f}  CV(중앙값, 평균≥1) {stats["path_cv_median"]:.3f}')
    print(f'  날짜평균으로 하루 맞히기 S_od {stats["S_od_day_by_mean"]:.3f} | '
          f'요일평균으로 하루 맞히기 {stats["S_od_day_by_weekday"]:.3f} | '
          f'날짜평균으로 월평균 맞히기 {stats["S_od_month_by_mean"]:.3f}')
    top = np.argsort(mean)[::-1][:5]
    print(pd.DataFrame({'path_id': paths.path_id.to_numpy()[top],
                        'od': (paths.origin_zone + '->' + paths.destination_zone).to_numpy()[top],
                        'mean': mean[top], 'std': std[top], 'cv': std[top] / mean[top]})
          .round(3).to_string(index=False))

    trips = t.trips
    month_trips = trips.groupby(dates.strftime('%Y-%m')).mean()
    return {'panel': panel, 'days': len(t), 'trips_mean': trips.mean(),
            'trips_cv_day': trips.std() / trips.mean(),
            'trips_cv_month': month_trips.std() / month_trips.mean(),
            'weekday_max/min': wd.trips.max() / wd.trips.min(),
            'through%': t['through%'].mean(), **stats}


def run(release, panels, label_dir):
    rows = []
    for panel in panels:
        net = load_corridor_network(release, panel)
        lab = load_labels(label_dir, panel)
        if net is None or lab is None:
            print(f'[{panel}] network or pseudo labels missing -> skipped.')
            continue
        rows.append(report_panel(release, panel, lab, net['paths']))
    if rows:
        print('\n===== 요약 (corridor별)')
        print(pd.DataFrame(rows).set_index('panel').round(3).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--panels', nargs='+', default=PANELS)
    ap.add_argument('--labels', type=Path, default=PSEUDO_DIR)
    args = ap.parse_args()

    run(RELEASE_ROOT, args.panels, args.labels)
