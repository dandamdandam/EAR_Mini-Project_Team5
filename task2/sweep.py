import argparse
import itertools
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
SWEEP_DIR = TASK_DIR / 'outputs' / 'sweep'
LOG_DIR = SWEEP_DIR / 'logs'

LRS = [5e-4, 1e-3, 2e-3, 4e-3]
EPOCHS = [15, 30, 45]


def load_env():
    env = os.environ.copy()
    env_file = TASK_DIR.parent / '.env'
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                env.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3])
    ap.add_argument('--project', default='task2-queue-sweep')
    args = ap.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    base_env = load_env()
    base_env['WANDB_DIR'] = str(SWEEP_DIR)

    q = queue.Queue()
    for lr, ep in itertools.product(LRS, EPOCHS):
        q.put({'lr': lr, 'epochs': ep})

    results = []
    lock = threading.Lock()

    def worker(gpu):
        while True:
            try:
                r = q.get_nowait()
            except queue.Empty:
                return
            name = f"lr{r['lr']:g}_ep{r['epochs']}"
            env = dict(base_env, CUDA_VISIBLE_DEVICES=str(gpu))
            print(f"[gpu{gpu}] start {name}", flush=True)
            with open(LOG_DIR / f'{name}.log', 'w') as f:
                p = subprocess.run(
                    [sys.executable, 'sweep_train.py', '--lr', str(r['lr']),
                     '--epochs', str(r['epochs']), '--run-name', name,
                     '--project', args.project],
                    cwd=TASK_DIR, env=env, stdout=f, stderr=subprocess.STDOUT)
            with lock:
                results.append((name, p.returncode))
            print(f"[gpu{gpu}] done {name} rc={p.returncode}", flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    failed = [n for n, rc in results if rc != 0]
    rows = []
    for rpt in sorted(SWEEP_DIR.glob('lr*_ep*.json')):
        d = json.loads(rpt.read_text())
        rows.append((d['val']['equal_mean_iou'], d['test']['equal_mean_iou'],
                     d['lr'], d['max_epochs'], d['best_epoch'], d['best_tau']))
    rows.sort(reverse=True)
    print('\nval_eq_iou  test_eq_iou  lr      max_ep  best_ep  tau')
    for v, t, lr, me, be, tau in rows:
        print(f'{v:.4f}      {t:.4f}       {lr:<7g} {me:<7d} {be:<8d} {tau:.2f}')
    if failed:
        print(f'\nfailed runs: {failed}')


if __name__ == '__main__':
    main()
