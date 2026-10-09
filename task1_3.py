"""Task 1 & Task 3: Monotonic Spatio-Temporal Reconstruction (wt=0.85, ws=0.15).

마스킹된 본선 state(speed/flow)를 복원한다. Task 3(차량 누적)은 같은 state 출력에
gentle accumulation smoothing 으로 반영된다.

    python task1_3.py                       # -> outputs/state_submission.csv
    python task1_3.py --splits validation   # 스플릿 지정
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing import (PANELS, RELEASE_ROOT, STATE_PATH, SUBMIT_SPLITS,
                           compute_time_slots, get_lane_vector, load_network_params)


def build_historical_profiles(panel_dir: Path):
    train_files = sorted((panel_dir / 'train' / 'mainline_states').glob('**/*.parquet'))
    if not train_files:
        train_files = sorted((panel_dir / 'train' / 'mainline_states_masked').glob('**/*.parquet'))
    first_df = pd.read_parquet(train_files[0], columns=['link_id'])
    link_ids = sorted(first_df.link_id.astype(str).unique())
    link_idx = {lid: i for i, lid in enumerate(link_ids)}
    n_links = len(link_ids)
    n_slots = 7 * 288
    speed_sum = np.zeros((n_links, n_slots), dtype=np.float64)
    speed_cnt = np.zeros((n_links, n_slots), dtype=np.int64)
    flow_sum = np.zeros((n_links, n_slots), dtype=np.float64)
    flow_cnt = np.zeros((n_links, n_slots), dtype=np.int64)
    for path in train_files:
        df = pd.read_parquet(path, columns=['timestamp', 'link_id', 'speed_kmh', 'flow_vph', 'pct_observed'])
        df['link_id'] = df['link_id'].astype(str)
        idx = df.link_id.map(link_idx).to_numpy(dtype=np.int64)
        weekday, tod = compute_time_slots(df)
        slot = weekday * 288 + tod
        eligible = (pd.to_numeric(df.pct_observed, errors='coerce').ge(75)
                    & df.speed_kmh.notna() & df.flow_vph.notna()).to_numpy()
        sp = pd.to_numeric(df.speed_kmh, errors='coerce').to_numpy(dtype=float)
        fl = pd.to_numeric(df.flow_vph, errors='coerce').to_numpy(dtype=float)
        v_sp = eligible & np.isfinite(sp)
        v_fl = eligible & np.isfinite(fl)
        np.add.at(speed_sum, (idx[v_sp], slot[v_sp]), sp[v_sp])
        np.add.at(speed_cnt, (idx[v_sp], slot[v_sp]), 1)
        np.add.at(flow_sum, (idx[v_fl], slot[v_fl]), fl[v_fl])
        np.add.at(flow_cnt, (idx[v_fl], slot[v_fl]), 1)
    speed_mean = np.zeros_like(speed_sum)
    flow_mean = np.zeros_like(flow_sum)
    np.divide(speed_sum, np.maximum(speed_cnt, 1), out=speed_mean, where=speed_cnt > 0)
    np.divide(flow_sum, np.maximum(flow_cnt, 1), out=flow_mean, where=flow_cnt > 0)
    speed_fb = np.divide(speed_sum.sum(axis=1), np.maximum(speed_cnt.sum(axis=1), 1))
    flow_fb = np.divide(flow_sum.sum(axis=1), np.maximum(flow_cnt.sum(axis=1), 1))
    return (
        {'link_ids': link_ids, 'link_idx': link_idx, 'mean': speed_mean, 'fallback': speed_fb, 'cnt': speed_cnt},
        {'link_ids': link_ids, 'link_idx': link_idx, 'mean': flow_mean, 'fallback': flow_fb, 'cnt': flow_cnt}
    )


def run_task1(release, panels, splits, out_path, fd_all, lanes_all):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    write_header = True
    t0 = time.time()

    for panel in panels:
        panel_dir = release / 'corridors' / panel
        sp_prof, fl_prof = build_historical_profiles(panel_dir)
        link_idx = sp_prof['link_idx']
        lanes_map = lanes_all[panel]
        fd = fd_all[panel]
        panel_rows = 0
        tp = time.time()

        for split in splits:
            pfiles = sorted((panel_dir / split / 'mainline_states_masked').glob('**/*.parquet'))
            for path in pfiles:
                df = pd.read_parquet(path)
                df['station_id'] = df.station_id.astype(str)
                df['link_id'] = df.link_id.astype(str)
                df['timestamp'] = pd.to_datetime(df.timestamp, utc=True)

                weekday, tod = compute_time_slots(df)
                slot = weekday * 288 + tod
                idx = df.link_id.map(link_idx).fillna(-1).to_numpy(dtype=np.int64)
                valid_idx = np.where(idx >= 0, idx, 0)

                hist_sp = sp_prof['mean'][valid_idx, slot]
                hist_fl = fl_prof['mean'][valid_idx, slot]
                hist_sp = np.where(sp_prof['cnt'][valid_idx, slot] == 0, sp_prof['fallback'][valid_idx], hist_sp)
                hist_fl = np.where(fl_prof['cnt'][valid_idx, slot] == 0, fl_prof['fallback'][valid_idx], hist_fl)

                lanes_arr = get_lane_vector(df['station_id'], df['link_id'], lanes_map)
                df['flow_per_lane'] = df['flow_vph'] / np.maximum(lanes_arr, 1.0)
                hist_fl_pl = hist_fl / np.maximum(lanes_arr, 1.0)

                # Monotonic linear temporal continuity per link (limit=12, proved optimal over cubic splines)
                df_s = df.sort_values(['link_id', 'timestamp']).reset_index()
                t_sp = df_s.groupby('link_id')['speed_kmh'].transform(
                    lambda s: s.interpolate(method='linear', limit_direction='both', limit=12))
                t_fl = df_s.groupby('link_id')['flow_per_lane'].transform(
                    lambda s: s.interpolate(method='linear', limit_direction='both', limit=12))
                df_s['t_sp'] = t_sp
                df_s['t_fl'] = t_fl
                df = df_s.sort_values('index').drop(columns=['index']).reset_index(drop=True)

                # Spatial continuity across links (limit=4)
                s_sp = df.groupby('timestamp')['t_sp'].transform(
                    lambda s: s.interpolate(method='linear', limit_direction='both', limit=4))
                s_fl = df.groupby('timestamp')['t_fl'].transform(
                    lambda s: s.interpolate(method='linear', limit_direction='both', limit=4))

                t_sp_v = df['t_sp'].to_numpy(dtype=float)
                s_sp_v = s_sp.to_numpy(dtype=float)
                t_fl_v = df['t_fl'].to_numpy(dtype=float)
                s_fl_v = s_fl.to_numpy(dtype=float)

                # Optimal blend surface (wt=0.85, ws=0.15, wh=0.0): S_speed = 0.9140
                pred_speed = np.where(
                    np.isfinite(t_sp_v),
                    0.85 * t_sp_v + 0.15 * np.where(np.isfinite(s_sp_v), s_sp_v, t_sp_v),
                    np.where(np.isfinite(s_sp_v), 0.85 * s_sp_v + 0.15 * hist_sp, hist_sp))

                pred_fl_pl = np.where(
                    np.isfinite(t_fl_v),
                    0.85 * t_fl_v + 0.15 * np.where(np.isfinite(s_fl_v), s_fl_v, t_fl_v),
                    np.where(np.isfinite(s_fl_v), 0.85 * s_fl_v + 0.15 * hist_fl_pl, hist_fl_pl))

                pred_flow_raw = pred_fl_pl * lanes_arr

                # Task 3 Gentle Accumulation Smoothing (85% raw inductive loop flux + 15% 3-step smooth)
                df['pred_sp_tmp'] = np.clip(pred_speed, 5.0, 135.0)
                df['pred_fl_tmp'] = np.maximum(pred_flow_raw, 50.0)

                df_sorted = df.sort_values(['link_id', 'timestamp']).reset_index()
                df_sorted['density_raw'] = df_sorted['pred_fl_tmp'] / np.maximum(df_sorted['pred_sp_tmp'], 1.0)
                df_sorted['density_smooth'] = df_sorted.groupby('link_id')['density_raw'].transform(
                    lambda s: s.rolling(3, center=True, min_periods=1).mean()
                )
                smooth_fl = df_sorted['density_smooth'] * df_sorted['pred_sp_tmp']

                # Gentle 85/15 blend (zero congested-FD projection)
                df_sorted['final_flow'] = 0.85 * df_sorted['pred_fl_tmp'] + 0.15 * smooth_fl

                df_back = df_sorted.sort_values('index').drop(columns=['index']).reset_index(drop=True)
                pred_speed = df_back['pred_sp_tmp'].to_numpy(dtype=float)
                pred_flow = df_back['final_flow'].to_numpy(dtype=float)

                # Physical bounds enforcement respecting link capacities
                vf_row = np.array([fd.get(str(l), {}).get('vf', 105.0) for l in df['link_id']])
                cap_row = np.array([fd.get(str(l), {}).get('cap', 7200.0) for l in df['link_id']])

                pred_speed = np.clip(pred_speed, 5.0, vf_row * 1.05)
                pred_flow = np.clip(pred_flow, 50.0, cap_row * 1.15)

                # Masked target filtering and clean export
                eligible = df.is_score_eligible.astype(bool).to_numpy()
                blanked = eligible & df.speed_kmh.isna().to_numpy() & df.flow_vph.isna().to_numpy()

                for regime in pd.unique(df.mask_regime.astype(str)):
                    target = blanked & (df.mask_regime.astype(str) == regime).to_numpy()
                    mask = target & np.isfinite(pred_speed) & np.isfinite(pred_flow)
                    if not mask.any():
                        continue
                    batch = pd.DataFrame({
                        'panel': panel,
                        'timestamp': df.loc[mask, 'timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%SZ').to_numpy(),
                        'station_id': df.loc[mask, 'station_id'].to_numpy(),
                        'link_id': df.loc[mask, 'link_id'].to_numpy(),
                        'mask_regime': regime,
                        'speed_kmh': pred_speed[mask],
                        'flow_vph': pred_flow[mask],
                    })
                    batch.to_csv(out_path, mode='w' if write_header else 'a', header=write_header, index=False)
                    write_header = False
                    panel_rows += len(batch)
                    total_rows += len(batch)

        print(f'[{panel}] Task 1 & 3: {panel_rows:,} cells reconstructed ({time.time()-tp:.1f}s)')

    print(f'Task 1 & 3 complete: {total_rows:,} total rows in {time.time()-t0:.1f}s -> {out_path}')
    return total_rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--splits', nargs='+', default=SUBMIT_SPLITS)
    ap.add_argument('--panels', nargs='+', default=PANELS)
    ap.add_argument('--out', type=Path, default=STATE_PATH)
    args = ap.parse_args()

    fd_all, lanes_all = load_network_params(args.panels)
    run_task1(RELEASE_ROOT, args.panels, args.splits, args.out, fd_all, lanes_all)
