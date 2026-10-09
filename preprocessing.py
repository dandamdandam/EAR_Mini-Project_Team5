"""
python preprocessing.py

Dataset root: /home/dandamdandam/.cache/kagglehub/competitions/2026-ieee-big-data-traffic-flow-bench/kaggle_public
Active corridors (10): ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S', 'D12_I5_N', 'D12_I5_S', 'D12_I405_N', 'D12_I405_S']
[D7_I10_E] Network ready: 146 links, vf=113.3 km/h, w=24.6 km/h
[D7_I10_W] Network ready: 107 links, vf=114.3 km/h, w=24.6 km/h
[D7_I210_E] Network ready: 297 links, vf=113.5 km/h, w=24.8 km/h
[D7_I210_W] Network ready: 81 links, vf=114.5 km/h, w=24.0 km/h
[D7_I405_N] Network ready: 352 links, vf=113.6 km/h, w=23.6 km/h
[D7_I405_S] Network ready: 109 links, vf=113.3 km/h, w=25.5 km/h
[D12_I5_N] Network ready: 258 links, vf=117.4 km/h, w=25.7 km/h
[D12_I5_S] Network ready: 100 links, vf=117.8 km/h, w=25.4 km/h
[D12_I405_N] Network ready: 89 links, vf=115.4 km/h, w=26.5 km/h
[D12_I405_S] Network ready: 44 links, vf=115.2 km/h, w=27.2 km/h
"""
import json
import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / 'outputs'

load_dotenv(PROJECT_DIR / '.env')

STATE_PATH = OUTPUT_DIR / 'state_submission.csv'    # Task 1 & 3
QUEUE_PATH = OUTPUT_DIR / 'queue_submission.csv'    # Task 2
ODME_PATH = OUTPUT_DIR / 'odme_submission.csv'      # Task 4
FINAL_PATH = OUTPUT_DIR / 'submission.csv'

SUBMIT_SPLITS = ['validation', 'private']


def locate_release_root() -> Path:
    candidates = [
        os.getenv('FILE_PATH'),
        '/home/dandamdandam/.cache/kagglehub/competitions/2026-ieee-big-data-traffic-flow-bench/kaggle_public',
    ]
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        if (p / 'config' / 'corridors.json').exists() or (p / 'corridors').exists():
            return p
        if p.name == 'corridors' and p.parent.exists():
            return p.parent
    for p in Path('/kaggle/input').rglob('submission_key.csv'):
        return p.parent
    raise FileNotFoundError('Failed to locate competition dataset directory.')


RELEASE_ROOT = locate_release_root()

manifest_path = RELEASE_ROOT / 'config' / 'corridors.json'
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    PANELS = [p['corridor_id'] for p in manifest['panels']]
else:
    PANELS = sorted([p.name for p in (RELEASE_ROOT / 'corridors').iterdir() if p.is_dir()])
# 5개 freeway 에 각각 2개의 corridor(방향)가 존재


def load_lane_counts(panel_dir: Path) -> dict:
    lanes = {}
    links_path = panel_dir / 'network' / 'links.csv'
    fd_path = panel_dir / 'network' / 'fd_parameters.csv'
    if links_path.exists():
        df = pd.read_csv(links_path, dtype={'link_id': str})
        if 'lanes' in df.columns:
            lns = pd.to_numeric(df['lanes'], errors='coerce')
            valid = lns.notna() & (lns >= 1) & (lns <= 16)
            for lid, l in zip(df['link_id'][valid], lns[valid]):
                lanes[str(lid)] = float(l)
    if fd_path.exists():
        df = pd.read_csv(fd_path, dtype={'station_id': str, 'link_id': str})
        if 'lanes' in df.columns:
            lns = pd.to_numeric(df['lanes'], errors='coerce')
            valid = lns.notna() & (lns >= 1) & (lns <= 16)
            for _, row in df[valid].iterrows():
                if pd.notna(row.get('station_id')):
                    lanes[str(row['station_id'])] = float(row['lanes'])
                if pd.notna(row.get('link_id')):
                    lanes[str(row['link_id'])] = float(row['lanes'])
    return lanes


def load_fd_params(panel_dir: Path) -> dict:
    links_path = panel_dir / 'network' / 'links.csv'
    fd_path = panel_dir / 'network' / 'fd_parameters.csv'
    params = {}
    if links_path.exists():
        df = pd.read_csv(links_path, dtype={'link_id': str})
        for _, r in df.iterrows():
            lid = str(r['link_id'])
            vf = float(pd.to_numeric(r.get('free_speed_kmh'), errors='coerce') or 105.0)
            cap = float(pd.to_numeric(r.get('capacity_vph'), errors='coerce') or 0)
            ln = float(pd.to_numeric(r.get('lanes'), errors='coerce') or 4.0)
            ln = max(ln, 1.0)
            length = float(pd.to_numeric(r.get('length_km'), errors='coerce') or 1.0)
            if cap <= 0:
                cap = 1800.0 * ln
            params[lid] = {'vf': vf, 'cap': cap, 'lanes': ln, 'length': length}
    if fd_path.exists():
        fd = pd.read_csv(fd_path, dtype={'link_id': str, 'station_id': str})
        for lid, g in fd.groupby('link_id'):
            lid = str(lid)
            if lid not in params:
                params[lid] = {'vf': 105.0, 'cap': 7200.0, 'lanes': 4.0, 'length': 1.0}
            if 'free_speed_kmh' in g.columns:
                v = g['free_speed_kmh'].dropna().mean()
                if pd.notna(v) and v > 0:
                    params[lid]['vf'] = float(v)
            if 'capacity_vph' in g.columns:
                v = g['capacity_vph'].dropna().mean()
                if pd.notna(v) and v > 0:
                    params[lid]['cap'] = float(v)
            if 'lanes' in g.columns:
                v = g['lanes'].dropna().mean()
                if pd.notna(v) and v >= 1:
                    params[lid]['lanes'] = float(v)
            if 'k_jam' in g.columns:
                v = g['k_jam'].dropna().mean()
                if pd.notna(v) and v > 0:
                    params[lid]['k_jam_raw'] = float(v)
            if 'v_cut' in g.columns:
                v = g['v_cut'].dropna().mean()
                if pd.notna(v) and v > 0:
                    params[lid]['v_cut_raw'] = float(v)
    for lid, p in params.items():
        p['k_crit'] = p['cap'] / max(p['vf'], 1.0)
        p['k_jam'] = p.get('k_jam_raw', 100.0 * max(p['lanes'], 1.0))
        p['k_jam'] = max(p['k_jam'], p['k_crit'] * 1.05)
        p['wave_speed'] = p['cap'] / max(p['k_jam'] - p['k_crit'], 1e-9)
        p['v_cut'] = p.get('v_cut_raw', p['vf'] * 0.60)
    return params


def get_lane_vector(stations, links, lanes_map):
    result = np.ones(len(stations), dtype=float)
    for i, (s, l) in enumerate(zip(stations, links)):
        s, l = str(s), str(l)
        if s in lanes_map:
            result[i] = lanes_map[s]
        elif l in lanes_map:
            result[i] = lanes_map[l]
    return result


def compute_time_slots(frame):
    ts = pd.to_datetime(frame['timestamp'], utc=True)
    weekday = ts.dt.weekday.to_numpy(dtype=np.int16)
    tod = (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy(dtype=np.int16)
    return weekday, tod


def load_network_params(panels=PANELS, verbose=True):
    """코리도별 FD 파라미터와 차선 수. 노트북의 FD_ALL / LANES_ALL."""
    fd_all, lanes_all = {}, {}
    for panel in panels:
        p_dir = RELEASE_ROOT / 'corridors' / panel
        fd_all[panel] = load_fd_params(p_dir)
        lanes_all[panel] = load_lane_counts(p_dir)
        if verbose:
            n = len(fd_all[panel])
            mean_vf = np.mean([v['vf'] for v in fd_all[panel].values()])
            mean_w = np.mean([v['wave_speed'] for v in fd_all[panel].values()])
            print(f'[{panel}] Network ready: {n} links, vf={mean_vf:.1f} km/h, w={mean_w:.1f} km/h')
    return fd_all, lanes_all


if __name__ == '__main__':
    print(f'Dataset root: {RELEASE_ROOT}')
    print(f'Active corridors ({len(PANELS)}): {PANELS}')
    load_network_params()
