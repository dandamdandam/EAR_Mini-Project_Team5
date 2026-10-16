import os
from pathlib import Path

import numpy as np
import pandas as pd

KROOT = Path(os.getenv('FILE_PATH') or '/home/dandamdandam/.cache/kagglehub/competitions/2026-ieee-big-data-traffic-flow-bench/kaggle_public')
TASK_DIR = Path(__file__).resolve().parent
CACHE_DIR = TASK_DIR / 'cache'
OUT_DIR = TASK_DIR / 'outputs'

SCORED_PANELS = ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S', 'D12_I5_N', 'D12_I5_S']

STEP_MIN = 5
DAY_STEPS = 288
HIST = 13
SEQ = 18
HOR = 6
STEP_OFFSETS = np.arange(-12, 6)
EXTRA_DROP = 0.20

DATA_START = pd.Timestamp('2030-06-01', tz='UTC')
DATA_END = pd.Timestamp('2031-02-28 23:55', tz='UTC')
TRAIN_RANGE = ('2030-06-01', '2030-12-31')
VAL_RANGE = ('2031-01-01', '2031-01-31')
TEST_RANGE = ('2031-02-01', '2031-02-28')

QUEUE_RATIO = 0.60


def time_grid():
    return pd.date_range(DATA_START, DATA_END, freq='5min')


def n_days():
    return (DATA_END.normalize() - DATA_START).days + 1


def date_to_day_index(date_str):
    return (pd.Timestamp(date_str, tz='UTC') - DATA_START).days


def split_day_range(name):
    r = {'train': TRAIN_RANGE, 'val': VAL_RANGE, 'test': TEST_RANGE}[name]
    return date_to_day_index(r[0]), date_to_day_index(r[1])


def interp_fill(a):
    T, L = a.shape
    m = np.isfinite(a)
    ar = np.arange(T)[:, None]
    prev = np.where(m, ar, -1)
    prev = np.maximum.accumulate(prev, axis=0)
    nxt = np.where(m, ar, T)
    nxt = np.minimum.accumulate(nxt[::-1], axis=0)[::-1]
    has_prev = prev >= 0
    has_next = nxt < T
    pi = np.clip(prev, 0, T - 1)
    ni = np.clip(nxt, 0, T - 1)
    col = np.arange(L)[None, :]
    pv = a[pi, col]
    nv = a[ni, col]
    both = has_prev & has_next
    denom = np.maximum(ni - pi, 1).astype(np.float32)
    w = (ar - pi) / denom
    fill = np.where(both, pv * (1 - w) + nv * w, np.where(has_prev, pv, np.where(has_next, nv, np.nan)))
    return np.where(m, a, fill).astype(np.float32)


def load_panel(panel):
    z = np.load(CACHE_DIR / f'{panel}.npz', allow_pickle=False)
    return {k: z[k] for k in z.files}


def load_ode_feats(panel):
    z = np.load(CACHE_DIR / f'ode_feats_{panel}.npz', allow_pickle=False)
    return {k: z[k] for k in z.files}
