"""Train an ImputeFormer Task 1 competitor over all ten corridor panels.

Protocol matches train_staeformer.py: Jun-Dec synthetic masks for fitting,
January released Task 1 masks for checkpoint/blend selection, and February
released Task 1 masks for one final local test. Raw Jan/Feb observations are
opened only inside the shared scorer, after predictions from masked files.

Run from Traffic/ with:
  conda run -n easypj python -m src.task1.train_imputeformer
  conda run -n easypj python -m src.task1.train_imputeformer --smoke

The normal run uses physical GPU 9 only. --smoke is a CPU shape/backward check
and does not calculate an official Task 1 score.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

# Set device visibility before importing torch, as required on this server.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "9"

import numpy as np
import torch

from src.io import paths
from src.task1.imputeformer_model import ImputeFormerTask1, imputeformer_objective
from src.task1.staeformer_data import (
    REGIME_PROBABILITIES,
    all_train_dates,
    load_day,
    model_arrays,
    panel_infos,
    synthetic_target_mask,
)
from src.task1.train_staeformer import (
    BLENDS,
    SELECT_END,
    SELECT_START,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    WINDOW,
    _date_lists,
    _target_frames,
    _tensor_batch,
    _window_batches,
    score_official,
)

SOURCE_REPOSITORY = "https://github.com/tongnie/ImputeFormer"
SOURCE_COMMIT = "1c3868cf499951ac43b4fc7280593fbb745f58c7"


def seed_all(seed: int, *, cuda: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def settings_dict(args, infos: dict) -> dict:
    return {
        "architecture": "ImputeFormer attention + Task 1 residual readout",
        "source": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "source_license": "MIT; src/task1/IMPUTEFORMER_LICENSE",
        "source_adaptations": [
            "one shared model with global time-by-node embedding IDs for varying panel sizes",
            "five scaled speed/flow/mask/interpolation inputs plus four clock exogenous values",
            "two-channel residual head initialized to interpolation",
            "Task 1 weighted masked MSE instead of paper masked MAE",
            "orthonormal 2D time/node Fourier regularizer on normalized completed values",
        ],
        "epochs": args.epochs,
        "seed": args.seed,
        "window": WINDOW,
        "random_mask_rates": REGIME_PROBABILITIES,
        "train_dates": [TRAIN_START, TRAIN_END],
        "selection_dates": [SELECT_START, SELECT_END],
        "test_dates": [TEST_START, TEST_END],
        "model": {
            "feature_dim": args.feature_dim,
            "adaptive_dim": args.adaptive_dim,
            "heads": args.heads,
            "layers": args.layers,
            "projection_tokens": args.projection_tokens,
            "feed_forward_dim": args.feed_forward_dim,
            "dropout": args.dropout,
        },
        "fourier_weight": args.fourier_weight,
        "optimizer": "Adam",
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "large_batch_size": args.large_batch_size,
        "large_panel_threshold": args.large_panel_threshold,
        "eval_batch_size": args.eval_batch_size,
        "amp_bfloat16": args.amp,
        "visible_physical_gpu": 9 if not args.smoke else None,
        "torch": torch.__version__,
        "panel_nodes": {panel: info.n_links for panel, info in infos.items()},
        "smoke": args.smoke,
    }


def train_epoch(model, optimizer, infos, days_by_panel, epoch, args, device):
    model.train()
    rng = np.random.default_rng(args.seed + epoch * 1009)
    pairs = [(panel, day) for panel in paths.PANELS for day in days_by_panel.get(panel, ())]
    rng.shuffle(pairs)
    totals = {"loss": 0.0, "reconstruction": 0.0, "fourier": 0.0, "cells": 0, "batches": 0}
    started = time.perf_counter()

    for index, (panel, day) in enumerate(pairs):
        info = infos[panel]
        raw = load_day(panel, "train", day, info, masked=False)
        probability = REGIME_PROBABILITIES[index % len(REGIME_PROBABILITIES)]
        target = synthetic_target_mask(raw, probability, rng)
        visible = np.isfinite(raw.speed) & np.isfinite(raw.flow_lane) & ~target
        features, base, truth = model_arrays(raw, visible, target)
        starts = np.arange(0, 288, WINDOW)
        rng.shuffle(starts)
        batch_size = args.batch_size if info.n_links <= args.large_panel_threshold else args.large_batch_size
        node_ids = torch.as_tensor(info.global_node_ids, device=device)

        for batch_starts in _window_batches(starts, batch_size):
            x, b, y, mask, tod, dow = _tensor_batch(
                features, base, truth, target, day, batch_starts, device
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=args.amp and device.type == "cuda",
            ):
                residual = model(x, tod, dow, node_ids)
                loss, reconstruction, fourier = imputeformer_objective(
                    residual.float(), b, y, mask, x, args.fourier_weight
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at {panel}/{day}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = int(mask.sum())
            totals["loss"] += float(loss.detach()) * count
            totals["reconstruction"] += float(reconstruction.detach()) * count
            totals["fourier"] += float(fourier.detach()) * count
            totals["cells"] += count
            totals["batches"] += 1

        if (index + 1) % args.progress_days == 0 or index + 1 == len(pairs):
            print(
                f"epoch {epoch}: {index + 1}/{len(pairs)} panel-days, "
                f"loss={totals['loss'] / totals['cells']:.5f}, "
                f"elapsed={time.perf_counter() - started:.0f}s",
                flush=True,
            )

    return {
        "loss": totals["loss"] / totals["cells"],
        "reconstruction": totals["reconstruction"] / totals["cells"],
        "fourier": totals["fourier"] / totals["cells"],
        "supervised_cells": totals["cells"],
        "batches": totals["batches"],
        "seconds": round(time.perf_counter() - started, 1),
    }


def smoke(model, optimizer, infos, args, device):
    """Exercise one backward pass and the smallest/largest panel shapes on CPU."""
    rng = np.random.default_rng(args.seed)
    by_size = sorted(infos, key=lambda p: infos[p].n_links)
    report = []
    for panel in (by_size[0], by_size[-1]):
        info = infos[panel]
        day = all_train_dates(panel, TRAIN_START, TRAIN_END)[0]
        raw = load_day(panel, "train", day, info, masked=False)
        target = synthetic_target_mask(raw, 0.3, rng)
        visible = np.isfinite(raw.speed) & np.isfinite(raw.flow_lane) & ~target
        features, base, truth = model_arrays(raw, visible, target)
        x, b, y, mask, tod, dow = _tensor_batch(
            features, base, truth, target, day, np.array([0]), device
        )
        node_ids = torch.as_tensor(info.global_node_ids, device=device)
        optimizer.zero_grad(set_to_none=True)
        output = model(x, tod, dow, node_ids)
        loss, reconstruction, fourier = imputeformer_objective(
            output, b, y, mask, x, args.fourier_weight
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite CPU smoke loss: {panel}")
        if panel == by_size[0]:
            loss.backward()
            optimizer.step()
        report.append({
            "panel": panel, "nodes": info.n_links,
            "output_shape": list(output.shape), "target_cells": int(mask.sum()),
            "loss": float(loss.detach()),
            "reconstruction": float(reconstruction.detach()),
            "fourier": float(fourier.detach()),
        })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--large-batch-size", type=int, default=8)
    parser.add_argument("--large-panel-threshold", type=int, default=160)
    parser.add_argument("--eval-batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fourier-weight", type=float, default=0.01)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--adaptive-dim", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--projection-tokens", type=int, default=8)
    parser.add_argument("--feed-forward-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--progress-days", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("runs/imputeformer_001"))
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke", action="store_true", help="CPU shape and backward check")
    parser.add_argument(
        "--gpu-benchmark", action="store_true",
        help="one training day per panel on GPU 9; no official scoring",
    )
    args = parser.parse_args()
    if args.smoke and args.gpu_benchmark:
        raise ValueError("Choose one of --smoke and --gpu-benchmark")
    if args.gpu_benchmark:
        args.out = Path("runs/imputeformer_gpu_benchmark")
    if args.epochs < 1 or min(args.batch_size, args.large_batch_size, args.eval_batch_size) < 1:
        raise ValueError("Epoch and batch counts must be positive")
    if args.fourier_weight < 0:
        raise ValueError("Fourier loss weight cannot be negative")

    if args.smoke:
        args.out = Path("runs/imputeformer_cpu_smoke")
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expected only physical GPU 9 visible as cuda:0")
        device = torch.device("cuda:0")
    seed_all(args.seed, cuda=device.type == "cuda")
    infos = panel_infos()
    total_nodes = sum(info.n_links for info in infos.values())
    model = ImputeFormerTask1(
        total_nodes=total_nodes,
        steps=WINDOW,
        feature_dim=args.feature_dim,
        adaptive_dim=args.adaptive_dim,
        heads=args.heads,
        layers=args.layers,
        projection_tokens=args.projection_tokens,
        feed_forward_dim=args.feed_forward_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.out.mkdir(parents=True, exist_ok=True)
    settings = settings_dict(args, infos)
    (args.out / "config.json").write_text(json.dumps(settings, indent=2))

    if args.smoke:
        torch.set_num_threads(4)
        report = smoke(model, optimizer, infos, args, device)
        (args.out / "smoke_result.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({"cpu_smoke": report}), flush=True)
        return

    if args.gpu_benchmark:
        settings["gpu_benchmark"] = True
        (args.out / "config.json").write_text(json.dumps(settings, indent=2))
        sample_dates = {
            panel: all_train_dates(panel, TRAIN_START, TRAIN_END)[:1]
            for panel in paths.PANELS
        }
        result = train_epoch(model, optimizer, infos, sample_dates, 1, args, device)
        torch.cuda.synchronize()
        (args.out / "benchmark_result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({"gpu_benchmark": result}), flush=True)
        return

    train_dates = _date_lists(TRAIN_START, TRAIN_END)
    select_dates = _date_lists(SELECT_START, SELECT_END)
    test_dates = _date_lists(TEST_START, TEST_END)
    if {d for dates in train_dates.values() for d in dates} & {
        d for dates in select_dates.values() for d in dates
    }:
        raise ValueError("Training and selection dates overlap")
    select_targets = _target_frames(select_dates)
    test_targets = _target_frames(test_dates)
    print(json.dumps({
        "device": str(device), "total_nodes": total_nodes,
        "train_panel_days": sum(map(len, train_dates.values())),
        "selection_panel_days": sum(map(len, select_dates.values())),
        "test_panel_days": sum(map(len, test_dates.values())),
    }), flush=True)

    best_score = -float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        training = train_epoch(model, optimizer, infos, train_dates, epoch, args, device)
        selected = score_official(
            model, infos, select_targets, select_dates, device,
            amp=args.amp, batch_size=args.eval_batch_size, blends=BLENDS,
        )
        winning_blend = max(selected, key=lambda key: selected[key]["s_state"])
        selection_score = selected[winning_blend]["s_state"]
        history.append({
            "epoch": epoch,
            "train": training,
            "selection_scores": {blend: result["s_state"] for blend, result in selected.items()},
            "best_blend_this_epoch": winning_blend,
        })
        (args.out / "history.json").write_text(json.dumps(history, indent=2))
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": epoch, "settings": settings,
        }, args.out / "last.pt")
        if selection_score > best_score:
            best_score = selection_score
            torch.save({
                "model": model.state_dict(), "epoch": epoch,
                "blend": float(winning_blend), "settings": settings,
            }, args.out / "best.pt")
            (args.out / "selection_best.json").write_text(json.dumps({
                "epoch": epoch, "blend": float(winning_blend), "scores": selected,
            }, indent=2))
        print(
            f"epoch {epoch}: train_loss={training['loss']:.5f}; "
            f"Jan S_state={selection_score:.6f} at blend {winning_blend}; "
            f"best={best_score:.6f}", flush=True,
        )

    checkpoint = torch.load(args.out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    blend = float(checkpoint["blend"])
    final = score_official(
        model, infos, test_targets, test_dates, device,
        amp=args.amp, batch_size=args.eval_batch_size, blends=(0.0, blend),
    )
    result = {
        "selected_epoch": checkpoint["epoch"],
        "selected_blend": blend,
        "selection_s_state": best_score,
        "test": final,
        "test_target_rows": sum(frame.height for frame in test_targets.values()),
    }
    (args.out / "test_result.json").write_text(json.dumps(result, indent=2))
    print(
        f"February official-mask test: S_state={final[str(blend)]['s_state']:.6f}; "
        f"interpolation-only={final['0.0']['s_state']:.6f}; "
        f"rows={result['test_target_rows']:,}", flush=True,
    )


if __name__ == "__main__":
    main()
