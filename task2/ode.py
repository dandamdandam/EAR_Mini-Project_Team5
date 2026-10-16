import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from utils import CACHE_DIR, DATA_START, SCORED_PANELS, load_panel, n_days, split_day_range

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
H_DIM = 48
EMB_DIM = 16
SEED = 42


def rk4(f, h, dt):
    k1 = f(h)
    k2 = f(h + 0.5 * dt * k1)
    k3 = f(h + 0.5 * dt * k2)
    k4 = f(h + dt * k3)
    return h + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


class ODERNN(nn.Module):
    def __init__(self, n_links, ctx_dim, h_dim=H_DIM, emb_dim=EMB_DIM):
        super().__init__()
        self.emb = nn.Embedding(n_links, emb_dim)
        self.h0 = nn.Linear(emb_dim + ctx_dim, h_dim)
        self.f = nn.Sequential(nn.Linear(h_dim, 64), nn.Tanh(), nn.Linear(64, h_dim), nn.Tanh())
        self.gru = nn.GRUCell(1, h_dim)
        self.dec = nn.Sequential(nn.Linear(h_dim + emb_dim, 64), nn.SiLU(), nn.Linear(64, 1))

    def init_state(self, link_idx, ctx):
        e = self.emb(link_idx)
        h = torch.tanh(self.h0(torch.cat([e, ctx], -1)))
        return h, e

    def scan(self, link_idx, ctx, x, obs, dt):
        h, e = self.init_state(link_idx, ctx)
        preds = []
        for k in range(x.shape[1]):
            h = rk4(self.f, h, dt)
            preds.append(self.dec(torch.cat([h, e], -1)))
            xk = torch.nan_to_num(x[:, k:k + 1])
            upd = self.gru(xk, h)
            h = torch.where(obs[:, k:k + 1], upd, h)
        return torch.cat(preds, -1), h, e

    def scan_snapshot(self, link_idx, ctx, x, obs, dt, n_last):
        h, e = self.init_state(link_idx, ctx)
        snap = h.clone()
        for k in range(x.shape[1]):
            h = rk4(self.f, h, dt)
            xk = torch.nan_to_num(x[:, k:k + 1])
            upd = self.gru(xk, h)
            h = torch.where(obs[:, k:k + 1], upd, h)
            snap = torch.where((n_last == k + 1).unsqueeze(-1), h, snap)
        return snap, e

    def decode_at(self, snap, e, dt_vec, substeps=2):
        h = snap
        step = dt_vec / substeps
        for _ in range(substeps):
            h = rk4(self.f, h, step)
        return self.dec(torch.cat([h, e], -1)).squeeze(-1)


def build_link_index():
    offsets, links_count = {}, 0
    for p in SCORED_PANELS:
        z = np.load(CACHE_DIR / f'{p}.npz')
        offsets[p] = links_count
        links_count += len(z['links'])
    return offsets, links_count


def dow_array():
    return np.array([(DATA_START + pd.Timedelta(days=i)).dayofweek for i in range(n_days())])


def dow_array_val3():
    s = pd.Timestamp('2031-03-01', tz='UTC')
    return np.array([(s + pd.Timedelta(days=i)).dayofweek for i in range(31)])


def doy_array():
    return np.array([(DATA_START + pd.Timedelta(days=i)).dayofyear for i in range(n_days())])


def make_day_dataset():
    offsets, _ = build_link_index()
    dows = dow_array()
    dows3 = dow_array_val3()
    tr_lo, _ = split_day_range('train')
    _, va_hi = split_day_range('val')
    rows_x, rows_link, rows_dow = [], [], []
    for p in SCORED_PANELS:
        z = load_panel(p)
        H = z['hourly']
        for d in range(tr_lo, va_hi + 1):
            hd = H[d]
            ok = np.isfinite(hd).sum(axis=0) >= 8
            idx = np.where(ok)[0]
            rows_x.append(hd[:, idx].T)
            rows_link.append(offsets[p] + idx)
            rows_dow.append(np.full(len(idx), dows[d]))
        H3 = z['hourly3']
        for d in range(31):
            hd = H3[d]
            ok = np.isfinite(hd).sum(axis=0) >= 8
            idx = np.where(ok)[0]
            rows_x.append(hd[:, idx].T)
            rows_link.append(offsets[p] + idx)
            rows_dow.append(np.full(len(idx), dows3[d]))
    X = np.concatenate(rows_x)
    LI = np.concatenate(rows_link)
    DW = np.concatenate(rows_dow)
    return X, LI, DW


def make_daily_series():
    offsets, _ = build_link_index()
    series = {}
    for p in SCORED_PANELS:
        z = load_panel(p)
        series[p] = z['daily']
    return offsets, series


class DayCtx(nn.Module):
    def __init__(self):
        super().__init__()
        self.dow = nn.Embedding(7, 8)

    def forward(self, d):
        return self.dow(d)


def train_day_model(n_links):
    X, LI, DW = make_day_dataset()
    print(f'[day] sequences={len(X)}', flush=True)
    rng = np.random.default_rng(SEED)
    model = ODERNN(n_links, 8).to(DEVICE)
    ctx_mod = DayCtx().to(DEVICE)
    params = list(model.parameters()) + list(ctx_mod.parameters())
    opt = torch.optim.AdamW(params, lr=3e-3)
    n = len(X)
    sub = min(n, 140000)
    for ep in range(6):
        order = rng.choice(n, size=sub, replace=False)
        tot, cnt = 0.0, 0
        for i in range(0, sub, 4096):
            b = order[i:i + 4096]
            x = torch.tensor(X[b], device=DEVICE)
            li = torch.tensor(LI[b], device=DEVICE, dtype=torch.long)
            dw = torch.tensor(DW[b], device=DEVICE, dtype=torch.long)
            cut = torch.tensor(rng.integers(6, 23, size=len(b)), device=DEVICE)
            karr = torch.arange(24, device=DEVICE).unsqueeze(0)
            obs = torch.isfinite(x) & (karr < cut.unsqueeze(1))
            w = torch.where(karr >= cut.unsqueeze(1), 3.0, 1.0).expand_as(x)
            pred, _, _ = model.scan(li, ctx_mod(dw), x, obs, 1.0 / 24)
            m = torch.isfinite(x)
            loss = (w[m] * (pred[m] - torch.nan_to_num(x)[m]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 2.0)
            opt.step()
            tot += loss.item() * m.sum().item()
            cnt += m.sum().item()
        print(f'[day] epoch {ep + 1}/6 wmse={tot / max(cnt, 1):.5f}', flush=True)
    return model, ctx_mod


def gather_days(daily, d_targets, lags):
    D, L = daily.shape
    idx = d_targets[:, None] + lags[None, :]
    valid = idx >= 0
    idx = np.clip(idx, 0, D - 1)
    out = daily[idx, :]
    out[~valid] = np.nan
    return out


def train_span_model(n_links, offsets, series, lags, ctx_kind, name, epochs, batch):
    rng = np.random.default_rng(SEED + len(lags))
    dows = dow_array()
    doys = doy_array()
    _, va_hi = split_day_range('val')
    tr_lo, _ = split_day_range('train')
    samples = []
    for p in SCORED_PANELS:
        d_min = tr_lo + max(0, -int(lags.min()))
        for d in range(d_min, va_hi + 1):
            samples.append((p, d))
    ctx_dim = 8 if ctx_kind == 'dow' else 2
    model = ODERNN(n_links, ctx_dim).to(DEVICE)
    ctx_mod = DayCtx().to(DEVICE) if ctx_kind == 'dow' else None
    params = list(model.parameters()) + (list(ctx_mod.parameters()) if ctx_mod else [])
    opt = torch.optim.AdamW(params, lr=3e-3)
    for ep in range(epochs):
        rng.shuffle(samples)
        tot, cnt = 0.0, 0
        for p, d in samples:
            daily = series[p]
            xs = gather_days(daily, np.array([d]), lags)[0].T
            tgt = daily[d]
            keep = np.isfinite(xs).sum(axis=1) >= max(3, len(lags) // 3)
            keep &= np.isfinite(tgt)
            if keep.sum() < 5:
                continue
            idx = np.where(keep)[0]
            if len(idx) > batch:
                idx = rng.choice(idx, size=batch, replace=False)
            x_in = np.concatenate([xs[idx], np.full((len(idx), 1), np.nan, np.float32)], axis=1)
            t_in = np.concatenate([xs[idx], tgt[idx][:, None]], axis=1).astype(np.float32)
            x = torch.tensor(x_in, device=DEVICE)
            tgt_t = torch.tensor(t_in, device=DEVICE)
            li = torch.tensor(offsets[p] + idx, device=DEVICE, dtype=torch.long)
            if ctx_kind == 'dow':
                ctx = ctx_mod(torch.full((len(idx),), dows[d], device=DEVICE, dtype=torch.long))
            else:
                ang = 2 * np.pi * doys[d] / 365.25
                ctx = torch.tensor(np.tile([np.sin(ang), np.cos(ang)], (len(idx), 1)), device=DEVICE, dtype=torch.float32)
            obs = torch.isfinite(x)
            K = x.shape[1]
            w = torch.ones_like(x)
            w[:, -1] = 4.0
            pred, _, _ = model.scan(li, ctx, x, obs, 1.0 / K)
            m = torch.isfinite(tgt_t)
            loss = (w[m] * (pred[m] - torch.nan_to_num(tgt_t)[m]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 2.0)
            opt.step()
            tot += loss.item()
            cnt += 1
        print(f'[{name}] epoch {ep + 1}/{epochs} wmse={tot / max(cnt, 1):.5f}', flush=True)
    return model, ctx_mod


@torch.no_grad()
def export_wy_features(offsets, series, week, week_ctx, year, year_ctx, week_lags, year_lags):
    dows = dow_array()
    doys = doy_array()
    D = n_days()
    for p in SCORED_PANELS:
        daily = series[p]
        L = daily.shape[1]
        wy = np.zeros((D, L, 2), dtype=np.float32)
        for d in range(D):
            for j, (model, ctx_mod, lags) in enumerate([(week, week_ctx, week_lags), (year, year_ctx, year_lags)]):
                xs = gather_days(daily, np.array([d]), lags)[0].T
                x_in = np.concatenate([xs, np.full((L, 1), np.nan, np.float32)], axis=1)
                x = torch.tensor(x_in, device=DEVICE)
                li = torch.tensor(offsets[p] + np.arange(L), device=DEVICE, dtype=torch.long)
                if ctx_mod is not None:
                    ctx = ctx_mod(torch.full((L,), dows[d], device=DEVICE, dtype=torch.long))
                else:
                    ang = 2 * np.pi * doys[d] / 365.25
                    ctx = torch.tensor(np.tile([np.sin(ang), np.cos(ang)], (L, 1)), device=DEVICE, dtype=torch.float32)
                obs = torch.isfinite(x)
                pred, _, _ = model.scan(li, ctx, x, obs, 1.0 / x.shape[1])
                wy[d, :, j] = pred[:, -1].float().cpu().numpy()
        np.savez_compressed(CACHE_DIR / f'ode_feats_{p}.npz', wy=np.clip(wy, 0.0, 1.5))
        print(f'[export] {p} wy shape={wy.shape} mean={np.nanmean(wy):.3f}', flush=True)


def main():
    torch.manual_seed(SEED)
    offsets, n_links = build_link_index()
    day_model, day_ctx = train_day_model(n_links)
    _, series = make_daily_series()
    week_lags = np.arange(-7, 0)
    year_lags = np.arange(-90, 0, 3)
    week_model, week_ctx = train_span_model(n_links, offsets, series, week_lags, 'dow', 'week', 5, 2048)
    year_model, year_ctx = train_span_model(n_links, offsets, series, year_lags, 'doy', 'year', 4, 1024)
    torch.save({
        'offsets': offsets, 'n_links': n_links,
        'day': day_model.state_dict(), 'day_ctx': day_ctx.state_dict(),
        'week': week_model.state_dict(), 'week_ctx': week_ctx.state_dict(),
        'year': year_model.state_dict(),
        'week_lags': week_lags, 'year_lags': year_lags,
    }, CACHE_DIR / 'ode_models.pt')
    export_wy_features(offsets, series, week_model, week_ctx, year_model, year_ctx, week_lags, year_lags)
    print('ode training complete')


if __name__ == '__main__':
    main()
