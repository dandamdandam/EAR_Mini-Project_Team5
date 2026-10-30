import json

import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import QueueNet
from ode import DEVICE
from tau_select import (add_cond_feature, bootstrap_tau, build_pools,
                        collect_entries, score_matrix)
from utils import HIST, HOR, OUT_DIR, QUEUE_RATIO

SURF, INK, MUT, GRID_C, BASE_C = '#fcfcfb', '#0b0b0b', '#898781', '#e1e0d9', '#c3c2b7'
C_MODEL, C_ODE, C_PERS = '#2a78d6', '#eb6834', '#898781'
C_ONSET, C_ONGO = '#eb6834', '#1baf7a'
C_REAL, C_FREE, C_CONG = '#2a78d6', '#1baf7a', '#eb6834'
ODE_STEP_IDX = [13, 14, 15, 16, 17, 17]
THRESHOLDS = np.round(np.arange(0.30, 1.001, 0.025), 3)
GROUPS = {
    'cond': ['outputs/sweep/cond_s123.pt', 'outputs/sweep/cond_s124.pt', 'outputs/sweep/cond_s125.pt'],
    'base': ['outputs/sweep/lr0.002_ep45.pt', 'outputs/sweep/base_s124.pt', 'outputs/sweep/base_s125.pt'],
}
FIG_CKPT = {'cond': 'outputs/sweep/cond_s123.pt', 'base': 'outputs/sweep/lr0.002_ep45.pt'}


def iou_from_pred(pred, qt):
    tgt = qt > 0.5
    inter = (pred & tgt).sum(axis=(1, 2))
    union = (pred | tgt).sum(axis=(1, 2))
    return np.where(union > 0, inter / np.maximum(union, 1), 1.0)


def pool_arrays(p):
    return {'cond': p['cond'], 'qt': p['qt'],
            'ode_sp': p['odd'][:, ODE_STEP_IDX, :],
            'ode_in': p['odd'][:, HIST - 1:HIST + 5, :],
            'pers': np.repeat((p['sph'][:, HIST - 1] <= QUEUE_RATIO)[:, None, :], HOR, axis=1)}


def by_cond(scores, cond):
    return {'onset': float(scores[cond == 0].mean()), 'ongoing': float(scores[cond == 1].mean())}


def iou_by_cond(preds, arrs):
    sc = np.concatenate([iou_from_pred(pr, a['qt']) for pr, a in zip(preds, arrs)])
    cond = np.concatenate([a['cond'] for a in arrs])
    return by_cond(sc, cond)


def model_iou_by_cond(probs_list, arrs, taus):
    preds = []
    for pr, a in zip(probs_list, arrs):
        tau_w = np.where(a['cond'] == 0, taus[0], taus[1])[:, None, None]
        preds.append(pr >= tau_w)
    return iou_by_cond(preds, arrs)


def cells(arrs, probs_list):
    oc = np.concatenate([a['ode_in'].ravel() for a in arrs])
    pc = np.concatenate([pr.ravel() for pr in probs_list])
    tc = np.concatenate([a['qt'].ravel() for a in arrs])
    cnd = np.concatenate([np.repeat(a['cond'], a['qt'].shape[1] * a['qt'].shape[2]) for a in arrs])
    return oc, pc, tc, cnd


def spearman(a, b):
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def load_model(path):
    sd = torch.load(path, weights_only=False)['model']
    c_in = int(sd['proj.0.weight'].shape[1])
    model = QueueNet(c_in=c_in, no_ode=c_in in (18, 16)).to(DEVICE)
    model.load_state_dict(sd)
    return model, c_in in (21, 18)


def model_probs(model, pools_split, use_cond):
    return [e['probs'] for e in collect_entries(model, pools_split, use_cond)]


def val_taus(model, pools, use_cond):
    Sv, cv = score_matrix(collect_entries(model, pools['val'], use_cond))
    return {c: bootstrap_tau(Sv[cv == c])[0] for c in (0, 1)}


def set_odd(pools_split, value):
    bak = []
    for p in pools_split:
        bak.append(p['odd'])
        o = p['odd'].copy()
        o[:, HIST:, :] = value
        p['odd'] = o
    return bak


def restore_odd(pools_split, bak):
    for p, o in zip(pools_split, bak):
        p['odd'] = o


def style_ax(ax):
    ax.set_facecolor(SURF)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(BASE_C)
    ax.tick_params(colors=MUT, labelsize=9)
    ax.yaxis.grid(True, color=GRID_C, lw=0.8)
    ax.set_axisbelow(True)


def grouped_bars(ax, labels, series, colors, names):
    x = np.arange(len(labels))
    n = len(series)
    w = 0.8 / n
    for i, (s, c, nm) in enumerate(zip(series, colors, names)):
        pos = x - 0.4 + w * (i + 0.5)
        ax.bar(pos, s, w * 0.88, color=c, label=nm)
        for p, v in zip(pos, s):
            ax.text(p, v + 0.012, f'{v:.2f}', ha='center', fontsize=8, color=MUT)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=INK, fontsize=10)
    ax.set_ylim(0, 1.0)


def fig_h1(h1, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), facecolor=SURF)
    for ax, split in zip(axes, ('val', 'test')):
        style_ax(ax)
        d = h1[split]
        grouped_bars(ax, ['onset', 'ongoing'],
                     [[d['model'][k] for k in ('onset', 'ongoing')],
                      [d['ode06'][k] for k in ('onset', 'ongoing')],
                      [d['pers'][k] for k in ('onset', 'ongoing')]],
                     [C_MODEL, C_ODE, C_PERS],
                     ['QueueNet (cond)', 'ODE day model (spd≤0.60)', 'Persistence'])
        ax.set_title(split, color=INK, fontsize=11)
        ax.set_ylabel('mean IoU', color=MUT, fontsize=9)
    axes[0].legend(loc='upper left', frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle('H1 — ODE output as a queue predictor fails on onset windows', color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=160, facecolor=SURF)
    plt.close(fig)


def fig_h1_threshold(sweeps, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.8), facecolor=SURF)
    style_ax(ax)
    for split, ls in (('val', '--'), ('test', '-')):
        ax.plot(THRESHOLDS, sweeps[split]['onset'], ls, color=C_ONSET, lw=2,
                label=f'onset ({split})')
        ax.plot(THRESHOLDS, sweeps[split]['ongoing'], ls, color=C_ONGO, lw=2,
                label=f'ongoing ({split})')
    ax.axvline(QUEUE_RATIO, color=MUT, lw=1, ls=':')
    ax.text(QUEUE_RATIO + 0.01, 0.92, 'v_cut=0.60', color=MUT, fontsize=8)
    ax.set_xlabel('speed-ratio threshold on ODE output', color=MUT, fontsize=9)
    ax.set_ylabel('mean IoU', color=MUT, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    ax.set_title('H1 — no threshold rescues ODE on onset', color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURF)
    plt.close(fig)


def fig_h2(curves, interv, rho, path):
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), facecolor=SURF)
    for r, g in enumerate(('cond', 'base')):
        ax = axes[r, 0]
        style_ax(ax)
        cv = curves[g]
        for ck, color in (('onset', C_ONSET), ('ongoing', C_ONGO)):
            ax.plot(cv[ck]['x'], cv[ck]['prob'], '-', color=color, lw=2, label=f'{ck}: model prob')
            ax.plot(cv[ck]['x'], cv[ck]['truth'], '--', color=color, lw=1.2, alpha=0.6,
                    label=f'{ck}: true queue rate')
        ax.axvline(QUEUE_RATIO, color=MUT, lw=1, ls=':')
        ax.set_ylim(0, 1.0)
        ax.set_xlabel('ODE predicted speed ratio (model input)', color=MUT, fontsize=9)
        ax.set_ylabel('mean queue probability', color=MUT, fontsize=9)
        ax.set_title(f'{g} model — prob vs ODE input  '
                     f'(ρ onset={rho[g]["onset"]:.2f}, ongoing={rho[g]["ongoing"]:.2f})',
                     color=INK, fontsize=10)
        if r == 0:
            ax.legend(frameon=False, fontsize=7.5, labelcolor=INK)

        ax = axes[r, 1]
        style_ax(ax)
        iv = interv[g]
        grouped_bars(ax, ['onset', 'ongoing'],
                     [[iv['real'][k] for k in ('onset', 'ongoing')],
                      [iv['free'][k] for k in ('onset', 'ongoing')],
                      [iv['cong'][k] for k in ('onset', 'ongoing')]],
                     [C_REAL, C_FREE, C_CONG],
                     ['ODE input: real', 'ODE → free-flow (1.0)', 'ODE → congested (0.5)'])
        ax.set_ylabel('mean queue probability (horizon)', color=MUT, fontsize=9)
        ax.set_title(f'{g} model — counterfactual ODE input', color=INK, fontsize=10)
        if r == 0:
            ax.legend(loc='upper left', frameon=False, fontsize=7.5, labelcolor=INK)
    fig.suptitle('H2 — how much do QueueNet predictions depend on the ODE input? (test set)', color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=160, facecolor=SURF)
    plt.close(fig)


def main():
    pools = build_pools(np.random.default_rng(0), splits=('val', 'test'))
    add_cond_feature(pools)
    arrs = {s: [pool_arrays(p) for p in pools[s]] for s in ('val', 'test')}

    fig_model, fig_use_cond = load_model(FIG_CKPT['cond'])
    fig_taus = val_taus(fig_model, pools, fig_use_cond)

    h1, sweeps = {}, {}
    for s in ('val', 'test'):
        aa = arrs[s]
        h1[s] = {
            'ode06': iou_by_cond([a['ode_sp'] <= QUEUE_RATIO for a in aa], aa),
            'pers': iou_by_cond([a['pers'] > 0.5 for a in aa], aa),
            'model': model_iou_by_cond(model_probs(fig_model, pools[s], fig_use_cond), aa, fig_taus),
        }
        sw = {'onset': [], 'ongoing': []}
        for thr in THRESHOLDS:
            d = iou_by_cond([a['ode_sp'] <= thr for a in aa], aa)
            sw['onset'].append(d['onset'])
            sw['ongoing'].append(d['ongoing'])
        sweeps[s] = sw
        h1[s]['ode_best'] = {k: {'thr': float(THRESHOLDS[int(np.argmax(sw[k]))]), 'iou': float(max(sw[k]))}
                             for k in ('onset', 'ongoing')}

    h2, curves, interv, rho = {}, {}, {}, {}
    for g, ckpts in GROUPS.items():
        h2[g] = {}
        for ck in ckpts:
            model, use_cond = load_model(ck)
            taus = val_taus(model, pools, use_cond)
            stats = {'taus': taus}
            for s in ('val', 'test'):
                aa = arrs[s]
                probs_list = model_probs(model, pools[s], use_cond)
                oc, pc, tc, cnd_cells = cells(aa, probs_list)
                rh = {k: spearman(oc[cnd_cells == c], pc[cnd_cells == c]) for k, c in (('onset', 0), ('ongoing', 1))}
                iv = {}
                for name, value in (('free', 1.0), ('cong', 0.5)):
                    bak = set_odd(pools[s], value)
                    pcf_list = model_probs(model, pools[s], use_cond)
                    restore_odd(pools[s], bak)
                    pcf = np.concatenate([pr.ravel() for pr in pcf_list])
                    iv[name] = {'prob': {k: float(pcf[cnd_cells == c].mean()) for k, c in (('onset', 0), ('ongoing', 1))},
                                'iou': model_iou_by_cond(pcf_list, aa, taus)}
                stats[s] = {'spearman': rh,
                            'real': {'prob': {k: float(pc[cnd_cells == c].mean()) for k, c in (('onset', 0), ('ongoing', 1))},
                                     'iou': model_iou_by_cond(probs_list, aa, taus)},
                            'interv': iv}
                if ck == FIG_CKPT[g] and s == 'test':
                    rho[g] = rh
                    interv[g] = {'real': stats[s]['real']['prob'],
                                 'free': iv['free']['prob'], 'cong': iv['cong']['prob']}
                    bins = np.arange(0.30, 1.21, 0.05)
                    cv = {}
                    for k, c in (('onset', 0), ('ongoing', 1)):
                        m = cnd_cells == c
                        bi = np.digitize(oc[m], bins)
                        xs, ps, ts = [], [], []
                        for bidx in range(1, len(bins)):
                            sel = bi == bidx
                            if sel.sum() >= 200:
                                xs.append((bins[bidx - 1] + bins[bidx]) / 2)
                                ps.append(float(pc[m][sel].mean()))
                                ts.append(float(tc[m][sel].mean()))
                        cv[k] = {'x': xs, 'prob': ps, 'truth': ts}
                    curves[g] = cv
            h2[g][ck.split('/')[-1].replace('.pt', '')] = stats
            print(f'[h2] {ck} done', flush=True)

    OUT_DIR.mkdir(exist_ok=True)
    fig_h1(h1, OUT_DIR / 'fig_h1_ode.png')
    fig_h1_threshold(sweeps, OUT_DIR / 'fig_h1_threshold.png')
    fig_h2(curves, interv, rho, OUT_DIR / 'fig_h2_follow_ode.png')
    (OUT_DIR / 'sweep' / 'ode_analysis.json').write_text(json.dumps({'h1': h1, 'h2': h2}, indent=2))

    print(json.dumps({'h1': h1}, indent=2), flush=True)
    for g in GROUPS:
        r = h2[g][FIG_CKPT[g].split('/')[-1].replace('.pt', '')]
        print(f"[{g}] test spearman={r['test']['spearman']} "
              f"real_prob={r['test']['real']['prob']} free={r['test']['interv']['free']['prob']} "
              f"cong={r['test']['interv']['cong']['prob']}", flush=True)


if __name__ == '__main__':
    main()
