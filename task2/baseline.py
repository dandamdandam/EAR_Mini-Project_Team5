"""Task 2: Delayed Onset Bottleneck Dynamics.
    python task2.py                         # -> outputs/queue_submission.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocessing import PANELS, QUEUE_PATH, RELEASE_ROOT, SUBMIT_SPLITS, load_network_params

ONSET_ACTIVE_LAST_STEPS = 1   # 1 = 마지막 스텝만 (= 기존 dt_min>=29.5)

# Top-2 empirical recurrent bottlenecks identified across all 8 corridors from ground truth training data
# (D12_I405_N / D12_I405_S 는 Task 2 윈도가 없어 제외 — contract 의 excluded_panels)
EMPIRICAL_TOP2_BOTTLENECKS = {
    'D12_I5_N': ['L5N-059', 'L5N-104'],
    'D12_I5_S': ['L5S-152', 'L5S-249'],
    'D7_I10_E': ['L10E-043', 'L10E-125'],
    'D7_I10_W': ['L10W-013', 'L10W-212'],
    'D7_I210_E': ['L210E-132', 'L210E-261'],
    'D7_I210_W': ['L210W-190', 'L210W-246'],
    'D7_I405_N': ['L405N-145', 'L405N-041'],
    'D7_I405_S': ['L405S-264', 'L405S-127']
}


def run_task2(release, splits, out_path, fd_all, onset_last_steps=ONSET_ACTIVE_LAST_STEPS):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wi_files = sorted(release.glob('task2/**/window_index.csv')) or sorted(release.glob('**/window_index.csv'))
    hist_files = sorted(release.glob('task2/**/window_history.parquet')) or sorted(release.glob('**/window_history.parquet'))
    tmpl_files = sorted(release.glob('task2/**/sample_submission_queue.csv')) or sorted(release.glob('**/sample_submission_queue.csv'))

    if not (wi_files and hist_files and tmpl_files):
        pd.DataFrame(columns=['window_id', 'timestamp', 'link_id', 'queue_pred']).to_csv(out_path, index=False)
        return 0

    windows = pd.concat([pd.read_csv(f) for f in wi_files], ignore_index=True)
    windows = windows[windows['split'].isin(splits)].copy()

    history = pd.concat([pd.read_parquet(f) for f in hist_files], ignore_index=True)
    history['window_id'] = history.window_id.astype(str)
    history['link_id'] = history.link_id.astype(str)
    history['timestamp'] = pd.to_datetime(history.timestamp, utc=True)
    history['speed_kmh'] = pd.to_numeric(history.speed_kmh, errors='coerce')

    template = pd.concat([pd.read_csv(f) for f in tmpl_files], ignore_index=True)
    template['window_id'] = template.window_id.astype(str)
    template['link_id'] = template.link_id.astype(str)
    template['timestamp'] = pd.to_datetime(template.timestamp, utc=True)

    panel_map = windows.set_index('window_id').panel.astype(str).to_dict()
    cond_map = windows.set_index('window_id').condition.astype(str).to_dict() if 'condition' in windows.columns else {}

    targets = []
    horizon_steps_seen = set()
    for window_id, h in tqdm(history.groupby('window_id', sort=True), desc='Task 2 Delayed Onset Dynamics'):
        wid = str(window_id)
        panel = panel_map.get(wid)
        if panel is None:
            continue
        cond = cond_map.get(wid, 'queue_ongoing')
        fd = fd_all.get(panel, {})
        t_last = h.timestamp.max()

        # Determine current queue status at T0
        # 주의: v_cut 은 릴리스에 공개되지 않는다. fd_parameters.csv 에 v_cut 컬럼이 없어
        #       load_fd_params 의 폴백 0.60 * vf 가 항상 쓰인다 (약 68 km/h). 추정값이다.
        at_t0 = h[h.timestamp == t_last].copy()
        v_cuts = {lid: fd.get(lid, {}).get('v_cut', 63.0) for lid in at_t0['link_id'].unique()}
        at_t0['is_elig'] = at_t0['is_score_eligible'].astype(bool) if 'is_score_eligible' in at_t0.columns else True
        at_t0['q_now'] = (at_t0['speed_kmh'] <= at_t0['link_id'].map(v_cuts).fillna(63.0)) & at_t0['is_elig']
        queue_now_map = at_t0.groupby('link_id')['q_now'].any().to_dict()

        # Top-2 recurrent bottlenecks for this corridor
        bottlenecks = EMPIRICAL_TOP2_BOTTLENECKS.get(panel, [])

        tgt = template[template.window_id == wid][['window_id', 'timestamp', 'link_id']].copy()

        # 지평 길이를 템플릿에서 읽어 "마지막 N스텝"의 시작 시각을 구한다.
        step_times = np.sort(tgt.timestamp.unique())
        horizon_steps_seen.add(len(step_times))
        k = min(max(onset_last_steps, 1), len(step_times))
        onset_from = step_times[len(step_times) - k]

        preds = []
        for row in tgt.itertuples():
            lid = str(row.link_id)

            if 'ongoing' in cond.lower():
                # Ongoing windows: 30분 지평 전체에 T0 상태를 그대로 유지 (persistence).
                # 단 contract 의 maximum_persistence_iou=0.9 로 순수 persistence 가 통하는
                # 윈도는 애초에 선정에서 제외되므로 상한이 막혀 있다.
                preds.append(1 if queue_now_map.get(lid, False) else 0)
            else:
                # Onset windows: 정체 형성에 축적 지연이 필요하므로 지평 뒷부분에만 활성화.
                # Top-2 상습 병목 링크로 공간을 한정해 초기 오경보를 제거한다.
                preds.append(1 if (lid in bottlenecks and row.timestamp >= onset_from) else 0)

        tgt['queue_pred'] = preds
        targets.append(tgt)

    out_df = pd.concat(targets, ignore_index=True).drop_duplicates(['window_id', 'timestamp', 'link_id'])
    out_df['queue_pred'] = out_df['queue_pred'].astype(int)
    out_df.to_csv(out_path, index=False)
    print(f'Task 2 complete: {len(out_df):,} window predictions exported. '
          f'horizon steps={sorted(horizon_steps_seen)}, onset active last {onset_last_steps} step(s), '
          f'positive rate={out_df.queue_pred.mean():.4f}')
    return len(out_df)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--splits', nargs='+', default=SUBMIT_SPLITS)
    ap.add_argument('--onset-last-steps', type=int, default=ONSET_ACTIVE_LAST_STEPS)
    ap.add_argument('--out', type=Path, default=QUEUE_PATH)
    args = ap.parse_args()

    fd_all, _ = load_network_params(PANELS)
    run_task2(RELEASE_ROOT, args.splits, args.out, fd_all, args.onset_last_steps)