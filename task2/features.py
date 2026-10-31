import numpy as np
import torch

from ode import DEVICE
from utils import EXTRA_DROP, HIST, HOR, QUEUE_RATIO, SEQ, STEP_OFFSETS, DAY_STEPS, interp_fill


@torch.no_grad()
def ode_day_features(day_model, day_ctx, offset, hourly, days, slots, L, dows):
    N = len(days)
    rows = N * L
    x_seq = hourly[days].transpose(0, 2, 1).reshape(rows, 24)
    n_full = (slots // 12).astype(np.int64)
    n_full_rows = np.repeat(n_full, L)
    li = offset + np.tile(np.arange(L), N)
    dw = np.repeat(dows[days], L)
    out = np.zeros((rows, SEQ), dtype=np.float32)
    q_h = (slots[:, None] + STEP_OFFSETS[None, :]).astype(np.float32) * 5.0 / 60.0
    dt_all = (q_h / 24.0 - (n_full / 24.0)[:, None]).astype(np.float32)
    for s in range(0, rows, 32768):
        e = min(s + 32768, rows)
        xb = torch.tensor(x_seq[s:e], device=DEVICE)
        ob = torch.isfinite(xb) & (torch.arange(24, device=DEVICE)[None, :] < torch.tensor(n_full_rows[s:e], device=DEVICE)[:, None])
        snap, emb = day_model.scan_snapshot(
            torch.tensor(li[s:e], device=DEVICE, dtype=torch.long),
            day_ctx(torch.tensor(dw[s:e], device=DEVICE, dtype=torch.long)),
            xb, ob, 1.0 / 24, torch.tensor(n_full_rows[s:e], device=DEVICE))
        dt_rows = np.repeat(dt_all, L, axis=0)[s:e]
        for i in range(SEQ):
            dtv = torch.tensor(dt_rows[:, i:i + 1], device=DEVICE)
            out[s:e, i] = day_model.decode_at(snap, emb, dtv).float().cpu().numpy()
    return np.clip(out.reshape(N, L, SEQ).transpose(0, 2, 1), 0.0, 1.5)


def static_features(z):
    vf = z['vf']
    mp = z['milepost']
    gap = np.diff(mp).astype(np.float32)
    gap_prev = np.concatenate([[0.0], gap])
    gap_next = np.concatenate([gap, [0.0]])
    return np.stack([z['lanes'] / 5.0, z['length'], vf / 120.0,
                     (mp - mp.min()) / max(float(np.ptp(mp)), 1e-3),
                     z['n_or'], z['n_fr'], gap_prev, gap_next], axis=-1).astype(np.float32)


def prop_map(q0, milepost, wave):
    N, L = q0.shape
    arr = np.where(q0, 0.0, np.inf).astype(np.float32)
    if L > 1:
        gap = np.diff(milepost).astype(np.float32)
        wc = np.clip(wave, 10.0, 40.0)
        wgap = 0.5 * (wc[:-1] + wc[1:])
        step_min = gap / np.maximum(wgap, 1e-3) * 60.0
        for j in range(L - 2, -1, -1):
            arr[:, j] = np.minimum(arr[:, j], arr[:, j + 1] + step_min[j])
    t_k = np.arange(1, HOR + 1, dtype=np.float32) * 5.0
    return np.clip((t_k[None, :, None] + 2.5 - arr[:, None, :]) / 5.0, 0.0, 1.0).astype(np.float32)


def assemble_pool(panel, z, wy_feats, t0s, cond, rng, day_model, day_ctx, offset, dows, fill_clim=None):
    vf = z['vf']
    x = (z['s_raw'] / vf[None]).astype(np.float32)
    fc = (z['f_raw'] / z['cap'][None]).astype(np.float32)
    strue = (z['s_true'] / vf[None]).astype(np.float32)
    q = z['q']
    L = x.shape[1]
    N = len(t0s)
    ht = t0s[:, None] + np.arange(-12, 1)[None, :]
    sph = x[ht].copy()
    flh = fc[ht].copy()
    extra = rng.random(sph.shape) < EXTRA_DROP
    sph[extra] = np.nan
    flh[extra] = np.nan
    obs = np.isfinite(sph).astype(np.float32)
    sph = interp_fill(sph.transpose(1, 0, 2).reshape(HIST, -1)).reshape(HIST, N, L).transpose(1, 0, 2)
    flh = interp_fill(flh.transpose(1, 0, 2).reshape(HIST, -1)).reshape(HIST, N, L).transpose(1, 0, 2)
    st = strue[t0s[:, None] + np.arange(-11, 7)[None, :]]
    qt = q[t0s[:, None] + np.arange(1, 7)[None, :]].astype(np.float32)
    slots = (t0s % DAY_STEPS).astype(np.int64)
    days = (t0s // DAY_STEPS).astype(np.int64)
    step_slot = slots[:, None] + STEP_OFFSETS[None, :]
    ang_t = 2 * np.pi * step_slot / DAY_STEPS
    dow = dows[days]
    ang_d = 2 * np.pi * (dow[:, None] + step_slot * 5.0 / 1440.0) / 7.0
    tf = np.stack([np.sin(ang_t), np.cos(ang_t), np.sin(ang_d), np.cos(ang_d)], axis=-1).astype(np.float32)
    odd = ode_day_features(day_model, day_ctx, offset, z['hourly'], days, slots, L, dows)
    nanmask = ~np.isfinite(sph)
    if fill_clim is None:
        sph[nanmask] = odd[:, :HIST, :][nanmask]
    else:
        slot_h = (t0s[:, None] + np.arange(-12, 1)[None, :]) % DAY_STEPS
        sph[nanmask] = fill_clim[slot_h][nanmask]
    flh[~np.isfinite(flh)] = 0.4
    wy = wy_feats['wy'][days]
    stat = static_features(z)
    return {'sph': sph.astype(np.float32), 'obs': obs, 'flh': flh.astype(np.float32), 'tf': tf,
            'odd': odd, 'wy': wy.astype(np.float32), 'stat': stat, 'st': st.astype(np.float32),
            'qt': qt, 'cond': cond, 'panel': panel}
