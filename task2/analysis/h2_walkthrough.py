import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import QueueNet
from ode import DEVICE
from tau_select import add_cond_feature, build_pools, collect_entries
from utils import HIST, OUT_DIR

SURF, INK, MUT, GRID_C, BASE_C = '#fcfcfb', '#0b0b0b', '#2c2c2c', '#e1e0d9', '#c3c2b7'
C_ONSET, C_ONGO = '#eb6834', '#1baf7a'
C_FREE, C_CONG = '#1baf7a', '#eb6834'
COND_NAMES = {0: 'onset', 1: 'ongoing'}
ODE_BINS = np.arange(0.30, 1.21, 0.05)
INTERVENTIONS = (('free_flow', 1.0), ('congested', 0.5))


def spearman(a, b):
    rank_a = a.argsort().argsort().astype(np.float64)
    rank_b = b.argsort().argsort().astype(np.float64)
    rank_a -= rank_a.mean()
    rank_b -= rank_b.mean()
    return float((rank_a * rank_b).sum() / np.sqrt((rank_a ** 2).sum() * (rank_b ** 2).sum()))


def load_model(ckpt_path):
    state = torch.load(ckpt_path, weights_only=False)['model']
    in_dim = int(state['proj.0.weight'].shape[1])
    model = QueueNet(c_in=in_dim, no_ode=in_dim in (18, 16)).to(DEVICE)
    model.load_state_dict(state)
    return model, in_dim in (21, 18)


def predict_probs(model, pools, use_cond):
    return [e['probs'] for e in collect_entries(model, pools, use_cond)]


def flatten_horizon_cells(pools, probs_per_pool):
    ode_speed_in = np.concatenate([p['odd'][:, HIST - 1:HIST + 5, :].ravel() for p in pools])
    model_prob = np.concatenate([pr.ravel() for pr in probs_per_pool])
    true_queue = np.concatenate([p['qt'].ravel() for p in pools])
    cond_label = np.concatenate([np.repeat(p['cond'], p['qt'].shape[1] * p['qt'].shape[2]) for p in pools])
    return ode_speed_in, model_prob, true_queue, cond_label


def force_future_ode(pools, forced_speed):
    originals = [p['odd'] for p in pools]
    for p in pools:
        forced = p['odd'].copy()
        forced[:, HIST:, :] = forced_speed
        p['odd'] = forced
    return originals


def restore_ode(pools, originals):
    for p, odd in zip(pools, originals):
        p['odd'] = odd


def bin_curve(ode_speed_in, cell_values, cell_mask, min_cells=200):
    bin_idx = np.digitize(ode_speed_in[cell_mask], ODE_BINS)
    centers, means, counts = [], [], []
    for b in range(1, len(ODE_BINS)):
        in_bin = bin_idx == b
        if in_bin.sum() >= min_cells:
            centers.append((ODE_BINS[b - 1] + ODE_BINS[b]) / 2)
            means.append(float(cell_values[cell_mask][in_bin].mean()))
            counts.append(int(in_bin.sum()))
    return np.array(centers), np.array(means), np.array(counts)


def style_ax(ax):
    ax.set_facecolor(SURF)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(BASE_C)
    ax.tick_params(colors=MUT, labelsize=8)
    ax.yaxis.grid(True, color=GRID_C, lw=0.8)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=Path, default=OUT_DIR / 'sweep' / 'cond_s123.pt')
    ap.add_argument('--out', type=Path, default=OUT_DIR / 'fig_h2_walkthrough.png')
    args = ap.parse_args()

    pools = build_pools(np.random.default_rng(0), splits=('test',))
    add_cond_feature(pools)
    test_pools = pools['test']
    model, use_cond = load_model(args.ckpt)
    print(f'ckpt={args.ckpt.name} use_cond={use_cond}', flush=True)

    print('--- step 1: observed correlation ---')
    probs_with_real_ode = predict_probs(model, test_pools, use_cond)
    ode_speed_in, model_prob, true_queue, cond_label = flatten_horizon_cells(test_pools, probs_with_real_ode)
    for c in (0, 1):
        in_cond = cond_label == c
        print(f'[{COND_NAMES[c]:>7}] rho(ode_speed_in, model_prob)={spearman(ode_speed_in[in_cond], model_prob[in_cond]):+.3f}  '
              f'rho(ode_speed_in, true_queue)={spearman(ode_speed_in[in_cond], true_queue[in_cond]):+.3f}')

    print('--- step 2: counterfactual intervention on future ode input ---')
    ode_future_mean = float(np.concatenate([p['odd'][:, HIST:, :].ravel() for p in test_pools]).mean())
    prob_with_forced_ode, prob_delta = {}, {}
    for interv_name, forced_speed in INTERVENTIONS:
        originals = force_future_ode(test_pools, forced_speed)
        probs_forced = predict_probs(model, test_pools, use_cond)
        restore_ode(test_pools, originals)
        _, forced_prob_cells, _, _ = flatten_horizon_cells(test_pools, probs_forced)
        prob_with_forced_ode[interv_name] = forced_prob_cells
        prob_delta[interv_name] = forced_prob_cells - model_prob
    for c in (0, 1):
        in_cond = cond_label == c
        forced_means = '  '.join(f'do(ode={speed})={prob_with_forced_ode[name][in_cond].mean():.3f}'
                                 for name, speed in INTERVENTIONS)
        print(f'[{COND_NAMES[c]:>7}] mean_prob: real={model_prob[in_cond].mean():.3f}  {forced_means}')
    for interv_name, forced_speed in INTERVENTIONS:
        abs_delta = np.abs(prob_delta[interv_name])
        print(f'do(ode={forced_speed}): ode_input mean {ode_future_mean:.2f} -> {forced_speed:.1f}  '
              f'|d_prob| mean={abs_delta.mean():.4f} p95={np.quantile(abs_delta, 0.95):.4f} max={abs_delta.max():.3f}')

    print('--- step 3: model_prob vs true_queue, binned by ode input ---')
    for c in (0, 1):
        in_cond = cond_label == c
        bin_centers, prob_curve, bin_counts = bin_curve(ode_speed_in, model_prob, in_cond)
        _, truth_curve, _ = bin_curve(ode_speed_in, true_queue, in_cond)
        print(f'[{COND_NAMES[c]}]  ode_bin  model_prob  true_queue_rate          n')
        for center, prob_mean, truth_mean, n_cells in zip(bin_centers, prob_curve, truth_curve, bin_counts):
            print(f'           {center:.3f}      {prob_mean:.3f}            {truth_mean:.3f}  {n_cells:>9,}')

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), facecolor=SURF)
    ax = axes[0]
    style_ax(ax)
    for c, color in ((0, C_ONSET), (1, C_ONGO)):
        in_cond = cond_label == c
        bin_centers, prob_curve, _ = bin_curve(ode_speed_in, model_prob, in_cond)
        _, truth_curve, _ = bin_curve(ode_speed_in, true_queue, in_cond)
        ax.plot(bin_centers, prob_curve, '-', color=color, lw=2, label=f'{COND_NAMES[c]}: model prob')
        ax.plot(bin_centers, truth_curve, '--', color=color, lw=1.2, alpha=0.6, label=f'{COND_NAMES[c]}: ground truth')
    ax.set_xlabel('ODE predicted speed/free-flow speed', color=MUT, fontsize=9)
    ax.set_ylabel('P(queue)', color=MUT, fontsize=9)
    ax.set_title('prob of queuing given ODE input',
                 color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK)

    ax = axes[1]
    style_ax(ax)
    delta_bins = np.linspace(-0.25, 0.25, 101)
    for (interv_name, forced_speed), color in zip(INTERVENTIONS, (C_FREE, C_CONG)):
        delta = prob_delta[interv_name]
        ax.hist(delta, bins=delta_bins, histtype='step', lw=1.8, color=color, log=True,
                label=f'do(ODE={forced_speed})  p95 |Δ|={np.quantile(np.abs(delta), 0.95):.3f}')
    ax.axvline(0, color=MUT, lw=1, ls=':')
    ax.set_xlabel('Δ P(queue) (forced - real ODE input)', color=MUT, fontsize=9)
    ax.set_ylabel('cell count (log)', color=MUT, fontsize=9)
    ax.set_title('change in P(queue) under forced ODE input', color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK)

    fig.suptitle('H2 walkthrough (test set)', color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, facecolor=SURF)
    plt.close(fig)
    print(f'saved: {args.out}', flush=True)


if __name__ == '__main__':
    main()
