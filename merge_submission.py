"""
python merge_submission.py --odme outputs/odme_submission_hf.csv
"""
import argparse
from pathlib import Path

import pandas as pd

from preprocessing import FINAL_PATH, ODME_PATH, QUEUE_PATH, RELEASE_ROOT, STATE_PATH

CHUNK_SIZE = 500_000
OUT_COLUMNS = ['submission_id', 'task', 'speed_kmh', 'flow_vph', 'queue_pred', 'path_flow']
KEYS = {
    'state': ['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime'],
    'queue': ['window_id', 'timestamp', 'link_id'],
    'odme': ['panel', 'departure_time', 'path_id']
}
VALUES = {
    'state': ['speed_kmh', 'flow_vph'],
    'queue': ['queue_pred'],
    'odme': ['path_flow']
}


def index_table(path, task):
    frame = pd.read_csv(path)
    missing = [c for c in KEYS[task] + VALUES[task] if c not in frame.columns]
    if missing:
        raise ValueError(f'{path}: {task} 결과에 컬럼 {missing} 이 없다.')
    frame = frame[KEYS[task] + VALUES[task]].copy()
    for col in KEYS[task]:
        frame[col] = pd.to_datetime(frame[col], utc=True) if col == 'timestamp' else frame[col].astype(str)
    for col in VALUES[task]:
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
    return frame.drop_duplicates(KEYS[task]).set_index(KEYS[task])


def locate_key_file(release):
    key_file = release / 'submission_key.csv'
    if not key_file.exists():
        found = list(release.glob('**/submission_key.csv'))
        if found:
            key_file = found[0]
    assert key_file.exists(), 'Missing submission_key.csv'
    return key_file


def merge(task_paths, key_file, out_path):
    print('Indexing prediction tables...')
    tables = {}
    for task, path in task_paths.items():
        if path is None or not Path(path).exists():
            print(f'  [{task}] 결과 파일 없음 ({path}) -> 0 으로 채움')
            continue
        tables[task] = index_table(path, task)
        print(f'  [{task}] {len(tables[task]):,} rows <- {path}')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    key_rows = {t: 0 for t in KEYS}
    matched = {t: 0 for t in KEYS}
    first_chunk = True
    total = 0
    for chunk in pd.read_csv(key_file, chunksize=CHUNK_SIZE, dtype=str):
        chunk['timestamp'] = pd.to_datetime(chunk.timestamp, utc=True, errors='coerce')
        out = pd.DataFrame({
            'submission_id': chunk.submission_id.astype('int64'),
            'task': chunk.task.astype(str)
        })
        for col in ('speed_kmh', 'flow_vph', 'queue_pred', 'path_flow'):
            out[col] = 0.0
        for task in KEYS:
            rows = (out.task == task)
            key_rows[task] += int(rows.sum())
            if task not in tables or not rows.any():
                continue
            wanted = pd.MultiIndex.from_frame(chunk.loc[rows, KEYS[task]])
            found = tables[task].reindex(wanted)
            matched[task] += int(found[VALUES[task]].notna().all(axis=1).sum())
            for col in VALUES[task]:
                out.loc[rows, col] = pd.Series(found[col].to_numpy()).fillna(0.0).to_numpy()

        # 청크 단위 검증 (노트북 diagnostics 의 값 검사를 전체 행에 적용)
        assert not out[OUT_COLUMNS].isna().any().any(), 'blank value in submission'
        assert (out['speed_kmh'] >= 0).all() and (out['flow_vph'] >= 0).all()
        assert out['queue_pred'].isin([0, 1]).all(), 'queue_pred must be 0/1'
        assert (out['path_flow'] >= 0).all()

        out[OUT_COLUMNS].to_csv(out_path, mode='w' if first_chunk else 'a', header=first_chunk, index=False)
        first_chunk = False
        total += len(out)
        print(f'\rProcessed: {total:,} rows', end='', flush=True)
    print(f'\nSubmission complete: {total:,} rows -> {out_path}')

    print('Key coverage (예측이 매칭된 비율, 나머지는 0):')
    for task in KEYS:
        n = key_rows[task]
        rate = matched[task] / n * 100 if n else float('nan')
        print(f'  {task:5s}: {matched[task]:>10,} / {n:>10,} ({rate:6.2f}%)')
    return total


def diagnostics(out_path):
    print('Running submission diagnostics...')
    assert out_path.exists()
    sz = out_path.stat().st_size / (1024 * 1024)
    print(f'Size: {sz:.2f} MB')
    sample = pd.read_csv(out_path, nrows=5000)
    assert sample.columns.tolist() == OUT_COLUMNS
    print('All diagnostics passed.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--state', type=Path, default=STATE_PATH, help='Task 1&3 결과 CSV')
    ap.add_argument('--queue', type=Path, default=QUEUE_PATH, help='Task 2 결과 CSV')
    ap.add_argument('--odme', type=Path, default=ODME_PATH, help='Task 4 결과 CSV')
    ap.add_argument('--out', type=Path, default=FINAL_PATH)
    args = ap.parse_args()

    merge({'state': args.state, 'queue': args.queue, 'odme': args.odme},
          locate_key_file(RELEASE_ROOT), args.out)
    diagnostics(args.out)
