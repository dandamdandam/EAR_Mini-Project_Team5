import argparse
import queue
import subprocess
import sys
import threading
from pathlib import Path

from sweep import LOG_DIR, SWEEP_DIR, TASK_DIR, load_env

RUN_SETS = {
    'cond': [
        {'name': 'base_s124', 'seed': 124, 'cond': False},
        {'name': 'base_s125', 'seed': 125, 'cond': False},
        {'name': 'cond_s123', 'seed': 123, 'cond': True},
        {'name': 'cond_s124', 'seed': 124, 'cond': True},
        {'name': 'cond_s125', 'seed': 125, 'cond': True},
    ],
    'noode': [
        {'name': 'noode_s123', 'seed': 123, 'cond': True, 'noode': True},
        {'name': 'noode_s124', 'seed': 124, 'cond': True, 'noode': True},
        {'name': 'noode_s125', 'seed': 125, 'cond': True, 'noode': True},
    ],
    'negcnd': [
        {'name': 'negcnd_s123', 'seed': 123, 'cond': True, 'noode': True},
        {'name': 'negcnd_s124', 'seed': 124, 'cond': True, 'noode': True},
        {'name': 'negcnd_s125', 'seed': 125, 'cond': True, 'noode': True},
    ],
    'big': [
        {'name': 'big_s123', 'seed': 123, 'cond': True, 'noode': True, 'epochs': 30},
        {'name': 'big_s124', 'seed': 124, 'cond': True, 'noode': True, 'epochs': 30},
        {'name': 'big_s125', 'seed': 125, 'cond': True, 'noode': True, 'epochs': 30},
    ],
    'big_noode': [
        {'name': 'big_noode_s123', 'seed': 123, 'cond': True, 'noode': True, 'epochs': 30, 'legacy': True},
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='cond', choices=sorted(RUN_SETS))
    args = ap.parse_args()
    RUNS = RUN_SETS[args.set]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    base_env = load_env()
    base_env['WANDB_DIR'] = str(SWEEP_DIR)

    q = queue.Queue()
    for r in RUNS:
        q.put(r)

    def worker(gpu):
        while True:
            try:
                r = q.get_nowait()
            except queue.Empty:
                return
            cmd = [sys.executable, 'sweep_train.py', '--lr', '0.002', '--epochs', str(r.get('epochs', 45)),
                   '--seed', str(r['seed']), '--run-name', r['name']]
            if r['cond']:
                cmd.append('--cond-feature')
            if r.get('noode'):
                cmd.append('--no-ode')
            if r.get('legacy'):
                cmd.append('--legacy-cnd')
            env = dict(base_env, CUDA_VISIBLE_DEVICES=str(gpu))
            print(f"[gpu{gpu}] start {r['name']}", flush=True)
            with open(LOG_DIR / f"{r['name']}.log", 'w') as f:
                p = subprocess.run(cmd, cwd=TASK_DIR, env=env, stdout=f, stderr=subprocess.STDOUT)
            print(f"[gpu{gpu}] done {r['name']} rc={p.returncode}", flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == '__main__':
    main()
