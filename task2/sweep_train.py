import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from _main import ensure_artifacts
from model import QueueNet
from ode import DEVICE
from tau_select import add_cond_feature, build_pools, evaluate_general, to_device_ex
from utils import HIST, OUT_DIR, SEQ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lr', type=float, required=True)
    ap.add_argument('--epochs', type=int, required=True)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--project', default='task2-queue-sweep')
    ap.add_argument('--run-name', default=None)
    ap.add_argument('--cond-feature', action='store_true')
    ap.add_argument('--no-ode', action='store_true')
    ap.add_argument('--legacy-cnd', action='store_true')
    args = ap.parse_args()

    run_name = args.run_name or f'lr{args.lr:g}_ep{args.epochs}'
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    ensure_artifacts()
    pools = build_pools(rng, no_ode=args.no_ode)
    if args.cond_feature:
        add_cond_feature(pools, legacy=args.legacy_cnd)

    qt_all = np.concatenate([pl['qt'].reshape(-1) for pl in pools['train']])
    p_pos = qt_all.mean()
    pos_weight = torch.tensor(min((1 - p_pos) / max(p_pos, 1e-4), 4.0), device=DEVICE)

    run = wandb.init(project=args.project, name=run_name, config={
        'lr': args.lr, 'max_epochs': args.epochs, 'seed': args.seed,
        'weight_decay': args.weight_decay, 'patience': args.patience,
        'cond_feature': args.cond_feature, 'no_ode': args.no_ode, 'legacy_cnd': args.legacy_cnd,
        'pos_weight': float(pos_weight), 'train_positive_rate': float(p_pos)})

    c_in = 19 + (2 if args.cond_feature else 0) - (3 if args.no_ode else 0)
    model = QueueNet(c_in=c_in, no_ode=args.no_ode).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tau_grid = np.arange(0.2, 0.81, 0.05)

    batches = []
    for pi, pool in enumerate(pools['train']):
        N = len(pool['qt'])
        for s in range(0, N, 8):
            batches.append((pi, np.arange(s, min(s + 8, N))))

    best_val, best_state, best_tau, best_epoch, patience = -1.0, None, 0.5, 0, 0
    for ep in range(args.epochs):
        model.train()
        tf_prob = max(0.0, 0.7 * (1 - ep / 12))
        rng.shuffle(batches)
        tot, cnt = 0.0, 0
        for pi, idx in batches:
            b = to_device_ex(pools['train'][pi], idx, args.cond_feature)
            logits, spreds = model(b, teacher_prob=tf_prob)
            bce = F.binary_cross_entropy_with_logits(logits[:, HIST - 1:SEQ], b['qt'], pos_weight=pos_weight)
            w = torch.ones(SEQ, device=DEVICE)
            w[HIST - 1:] = 2.0
            mse = (w[None, :, None] * (spreds - b['st']) ** 2).mean()
            loss = bce + 0.3 * mse
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            tot += loss.item()
            cnt += 1
        val = evaluate_general(model, pools['val'], tau_grid, args.cond_feature)
        run.log({'epoch': ep + 1, 'train_loss': tot / cnt, 'teacher_prob': tf_prob,
                 'val_equal_mean_iou': val['equal_mean_iou'], 'val_tau': val['tau'],
                 'val_onset_iou': val['onset_iou'], 'val_ongoing_iou': val['ongoing_iou']})
        print(f'epoch {ep + 1}: loss={tot / cnt:.4f} tf={tf_prob:.2f} '
              f'val_eq_iou={val["equal_mean_iou"]:.4f} (tau={val["tau"]:.2f})', flush=True)
        if val['equal_mean_iou'] > best_val:
            best_val = val['equal_mean_iou']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_tau = val['tau']
            best_epoch = ep + 1
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                print('early stop', flush=True)
                break

    model.load_state_dict(best_state)
    val = evaluate_general(model, pools['val'], [best_tau], args.cond_feature)
    test = evaluate_general(model, pools['test'], [best_tau], args.cond_feature)
    run.summary['best_val_equal_mean_iou'] = best_val
    run.summary['best_epoch'] = best_epoch
    run.summary['best_tau'] = float(best_tau)
    run.summary['test_equal_mean_iou'] = test['equal_mean_iou']
    run.summary['test_onset_iou'] = test['onset_iou']
    run.summary['test_ongoing_iou'] = test['ongoing_iou']

    report = {'lr': args.lr, 'max_epochs': args.epochs, 'best_epoch': best_epoch,
              'seed': args.seed, 'cond_feature': args.cond_feature, 'no_ode': args.no_ode,
              'best_tau': float(best_tau), 'val': val, 'test': test,
              'train_positive_rate': float(p_pos)}
    sweep_dir = OUT_DIR / 'sweep'
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / f'{run_name}.json').write_text(json.dumps(report, indent=2))
    torch.save({'model': best_state, 'tau': best_tau}, sweep_dir / f'{run_name}.pt')
    print(json.dumps({'run': run_name, 'best_epoch': best_epoch,
                      'val_eq_iou': val['equal_mean_iou'], 'test_eq_iou': test['equal_mean_iou']}, indent=2), flush=True)
    run.finish()


if __name__ == '__main__':
    main()
