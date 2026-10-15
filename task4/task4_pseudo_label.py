import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing import OUTPUT_DIR, PANELS, RELEASE_ROOT
from .nnls import load_corridor_network

PSEUDO_DIR = OUTPUT_DIR / 'pseudo_labels'
PM_START, PM_END = 15, 19 # we only use data generated between 3pm and 7pm


def daily_mean(state_dir, id_col):
    daily_means = []
    for parquet_path in sorted(state_dir.glob('*/*.parquet')):
        day_df = pd.read_parquet(parquet_path, columns=['date', 'timestamp', id_col, 'flow_vph'])
        hour = day_df.timestamp.str[11:13].astype(int)
        day_df = day_df[(hour >= PM_START) & (hour < PM_END)]
        daily_means.append(day_df.groupby(['date', id_col]).flow_vph.mean())
    return pd.concat(daily_means).unstack(id_col)


def zone_geometry(paths):
    link_seqs = paths.link_seq.astype(str).str.split(';')
    main_stream = max(link_seqs, key=len) # get longest path sequence
    link_pos = {link: i for i, link in enumerate(main_stream)}

    origin_zone_idx = paths.origin_zone.str[1:].astype(int).to_numpy()
    destination_zone_idx = paths.destination_zone.str[1:].astype(int).to_numpy()
    zone_n = max(origin_zone_idx.max(), destination_zone_idx.max()) + 1

    first_links = link_seqs.str[0].map(link_pos).to_numpy(dtype=float)
    last_links = link_seqs.str[-1].map(link_pos).to_numpy(dtype=float)
    zone_start_pos = pd.Series(first_links).groupby(origin_zone_idx).min().reindex(range(zone_n)).to_numpy()
    zone_end_pos = pd.Series(last_links).groupby(destination_zone_idx).max().reindex(range(zone_n)).to_numpy()
    return main_stream, link_pos, origin_zone_idx, destination_zone_idx, zone_n, zone_start_pos, zone_end_pos


def ramp_zone_map(ramp_attach, link_pos, zone_start_pos, zone_end_pos):
    ramp_to_zone = {}
    for ramp in ramp_attach.itertuples():
        attach_pos = link_pos.get(ramp.nearest_mainline_link_id)
        if attach_pos is None:
            continue
        if ramp.ramp_type == 'OR':
            candidate_zones = np.flatnonzero(zone_start_pos <= attach_pos)
        else:
            candidate_zones = np.flatnonzero(zone_end_pos >= attach_pos)
        if len(candidate_zones):
            zone = candidate_zones.max() if ramp.ramp_type == 'OR' else candidate_zones.min()
            ramp_to_zone[ramp.ramp_link_id] = (ramp.ramp_type, zone)
    return ramp_to_zone


def pseudo_od(zone_n, upstream_inflow, zone_in, zone_out):
    od_flow = np.zeros((zone_n, zone_n))
    mainline_by_origin = np.zeros(zone_n)
    mainline_by_origin[0] = upstream_inflow
    for zone in range(1, zone_n):
        mainline_total = mainline_by_origin.sum()
        if zone == zone_n - 1:
            exit_ratio = 1.0
        elif mainline_total > 0:
            exit_ratio = min(zone_out[zone] / mainline_total, 1.0)
        else:
            exit_ratio = 0.0
        od_flow[:, zone] = mainline_by_origin * exit_ratio
        mainline_by_origin = mainline_by_origin * (1 - exit_ratio)
        mainline_by_origin[zone] += zone_in[zone]
    return od_flow


def corridor_incidence(paths, main_stream, link_pos):
    incidence = np.zeros((len(main_stream), len(paths)))
    for path_idx, link_seq in enumerate(paths.link_seq.astype(str).str.split(';')):
        incidence[[link_pos[link] for link in link_seq if link in link_pos], path_idx] = 1.0
    return incidence


def build_panel(release, panel, paths, observed_links):
    """
    @param observed_links: the links that we interested in.
    """
    (main_stream, link_pos, origin_zone_idx, destination_zone_idx,
     zone_n, zone_start_pos, zone_end_pos) = zone_geometry(paths)
    train_dir = release / 'corridors' / panel / 'train'
    mainline_daily = daily_mean(train_dir / 'mainline_states', 'link_id')
    ramp_daily = daily_mean(train_dir / 'ramp_states', 'ramp_link_id').reindex(mainline_daily.index)
    ramp_daily = ramp_daily.fillna(ramp_daily.mean())
    day_n = len(mainline_daily)

    ramp_attach = pd.read_csv(release / 'corridors' / panel / 'network' / 'ramp_attachment_map.csv')
    ramp_to_zone = ramp_zone_map(ramp_attach, link_pos, zone_start_pos, zone_end_pos)

    observed_main_links = [link for link in main_stream
                           if link in observed_links and link in mainline_daily.columns]
    link_flow = np.full((day_n, len(main_stream)), np.nan)
    link_flow[:, [link_pos[link] for link in observed_main_links]] = mainline_daily[observed_main_links].to_numpy()

    zone_in = np.zeros((day_n, zone_n))
    zone_out = np.zeros((day_n, zone_n))
    for ramp_id, (ramp_type, zone) in ramp_to_zone.items():
        if ramp_id in ramp_daily.columns:
            (zone_in if ramp_type == 'OR' else zone_out)[:, zone] += ramp_daily[ramp_id].to_numpy()

    upstream_inflow = np.array([day_flow[~np.isnan(day_flow)][0] for day_flow in link_flow])
    path_flow = np.stack([
        pseudo_od(zone_n, upstream_inflow[day], zone_in[day], zone_out[day])[origin_zone_idx, destination_zone_idx]
        for day in range(day_n)])

    observed_pos = np.flatnonzero(~np.isnan(link_flow).all(axis=0))
    loaded_link_flow = path_flow @ corridor_incidence(paths, main_stream, link_pos)[observed_pos].T
    link_rel_err = np.nanmedian(np.abs(loaded_link_flow - link_flow[:, observed_pos]) / link_flow[:, observed_pos])

    zones_with_on_ramp = len({zone for ramp_type, zone in ramp_to_zone.values() if ramp_type == 'OR'})
    zones_with_off_ramp = len({zone for ramp_type, zone in ramp_to_zone.values() if ramp_type == 'FR'})
    print(f'[{panel}] days={day_n} ramps mapped {len(ramp_to_zone)}/{len(ramp_attach)} '
          f'(OR->{zones_with_on_ramp} zones, FR->{zones_with_off_ramp} zones of {zone_n}) '
          f'obs links {len(observed_main_links)} | '
          f'가상정답 A·f vs 본선 b 상대오차 중앙값 {link_rel_err * 100:.1f}%', flush=True)

    return {'dates': mainline_daily.index.to_numpy().astype(str), 'order': np.array(main_stream),
            'path_ids': paths.path_id.to_numpy(dtype=str),
            'B': link_flow, 'F': path_flow, 'O': zone_in, 'D': zone_out, 'q0': upstream_inflow}


def load_labels(label_dir, panel):
    label_path = label_dir / f'{panel}.npz'
    return dict(np.load(label_path)) if label_path.exists() else None


def run(release, panels, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for panel in panels:
        network = load_corridor_network(release, panel)
        if network is None:
            continue
        counts_csv = release / 'task4' / panel / 'train' / 'synthetic_link_counts.csv'
        observed_links = set(pd.read_csv(counts_csv).link_id)
        labels = build_panel(release, panel, network['paths'], observed_links)
        np.savez_compressed(out_dir / f'{panel}.npz', **labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--panels', nargs='+', default=PANELS)
    parser.add_argument('--out-dir', type=Path, default=PSEUDO_DIR)
    args = parser.parse_args()

    run(RELEASE_ROOT, args.panels, args.out_dir)
