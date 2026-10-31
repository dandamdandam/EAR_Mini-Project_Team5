"""Train interpolation + STAEformer residual on all TrafficFlowBench panels.

Evaluation protocol:
  train: 2030-06 .. 2030-12, synthetic 20/30/50% masks on eligible raw cells
  select: 2031-01, official released Task 1 mask and target keys
  test: 2031-02, official released Task 1 mask and target keys, once at the end

The unmasked January/February layer is opened inside score_official() only,
after predictions have been generated from the masked layer.  The held-out
February metric is not used to select a checkpoint or residual blend.

Run with: CUDA_VISIBLE_DEVICES=9 conda run -n easypj python -m src.task1.train_staeformer
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

# The shared-server rule requires physical GPU 9, which becomes cuda:0 here.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "9")

import numpy as np
import polars as pl
import torch

from src.io import paths
from src.task1.staeformer_data import (
    RESIDUAL_FLOW_LANE_SCALE,
    RESIDUAL_SPEED_SCALE,
    REGIME_PROBABILITIES,
    all_train_dates,
    day_time_features,
    load_day,
    model_arrays,
    model_inputs,
    official_targets,
    panel_infos,
    synthetic_target_mask,
    targets_for_day,
)
from src.task1.staeformer_residual import STAEformerResidual, reconstruction_loss

WINDOW = 24
BLENDS = (0.0, 0.25, 0.5, 0.75, 1.0)
TRAIN_START, TRAIN_END = "2030-06-01", "2030-12-31"
SELECT_START, SELECT_END = "2031-01-01", "2031-01-31"
TEST_START, TEST_END = "2031-02-01", "2031-02-28"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _window_batches(starts: np.ndarray, batch_size: int):
    for j in range(0, len(starts), batch_size):
        yield starts[j : j + batch_size]


def _tensor_batch(
    features: np.ndarray,
    base: np.ndarray,
    truth: np.ndarray,
    target: np.ndarray,
    day: str,
    starts: np.ndarray,
    device: torch.device,
):
    tod, dow = day_time_features(day)
    slices = [slice(int(s), int(s) + WINDOW) for s in starts]
    x = torch.from_numpy(np.stack([features[sl] for sl in slices])).to(device)
    b = torch.from_numpy(np.stack([base[sl] for sl in slices])).to(device)
    y = torch.from_numpy(np.stack([truth[sl] for sl in slices])).to(device)
    mask = torch.from_numpy(np.stack([target[sl] for sl in slices])).to(device)
    tt = torch.from_numpy(np.stack([tod[sl] for sl in slices])).to(device)
    dd = torch.from_numpy(np.stack([dow[sl] for sl in slices])).to(device)
    return x, b, y, mask, tt, dd


def train_epoch(model, optimizer, scaler, infos, days_by_panel, epoch, args, device):
    model.train()
    rng = np.random.default_rng(args.seed + epoch * 1009)
    pairs = [(panel, day) for panel in paths.PANELS for day in days_by_panel[panel]]
    rng.shuffle(pairs)
    losses = []
    counts = []
    started = time.perf_counter()
    for index, (panel, day) in enumerate(pairs):
        info = infos[panel]
        raw = load_day(panel, "train", day, info, masked=False)
        probability = REGIME_PROBABILITIES[index % 3]
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
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.amp):
                residual = model(x, tod, dow, node_ids)
                loss = reconstruction_loss(residual.float(), b, y, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite train loss: {panel}/{day}")
            if scaler is None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            losses.append(float(loss.detach()))
            counts.append(int(mask.sum()))
        if (index + 1) % args.progress_days == 0 or index + 1 == len(pairs):
            print(
                f"epoch {epoch}: {index + 1}/{len(pairs)} panel-days, "
                f"loss={np.average(losses, weights=counts):.4f}, "
                f"elapsed={time.perf_counter()-started:.0f}s",
                flush=True,
            )
    return {
        "loss": float(np.average(losses, weights=counts)),
        "supervised_cells": sum(counts),
        "batches": len(losses),
        "seconds": round(time.perf_counter() - started, 1),
    }


@torch.no_grad()
def predict_masked_day(model, masked_day, info, device, *, amp: bool, batch_size: int):
    model.eval()
    visible = np.isfinite(masked_day.speed) & np.isfinite(masked_day.flow_lane)
    features, base = model_inputs(masked_day, visible)
    all_starts = np.arange(0, 288, WINDOW)
    tod, dow = day_time_features(masked_day.day)
    correction = np.zeros_like(base)
    node_ids = torch.as_tensor(info.global_node_ids, device=device)
    for starts in _window_batches(all_starts, batch_size):
        slices = [slice(int(s), int(s) + WINDOW) for s in starts]
        x = torch.from_numpy(np.stack([features[sl] for sl in slices])).to(device)
        tt = torch.from_numpy(np.stack([tod[sl] for sl in slices])).to(device)
        dd = torch.from_numpy(np.stack([dow[sl] for sl in slices])).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            output = model(x, tt, dd, node_ids)
        values = output.float().cpu().numpy()
        for sl, value in zip(slices, values):
            correction[sl] = value
    return base, correction


def _summarize_scores(accum, panels):
    by_panel = defaultdict(dict)
    rows = []
    for (panel, regime), stat in sorted(accum.items()):
        if stat["n"] == 0:
            raise ValueError(f"No scored target cells for {panel}/{regime}")
        rv = (stat["v2"] / stat["n"]) ** 0.5
        rq = (stat["q2"] / stat["n"]) ** 0.5
        score = 0.54 * max(0, 1 - rv / 25) + 0.46 * max(0, 1 - rq / 600)
        rows.append(
            {"panel": panel, "regime": regime, "n": stat["n"], "rmse_speed": rv,
             "rmse_flow_per_lane": rq, "s_state": score}
        )
        by_panel[panel][regime] = score
    for panel in panels:
        if set(by_panel[panel]) != {"R1", "R2", "R3"}:
            raise ValueError(f"Missing regime on {panel}: {by_panel[panel]}")
    panel_scores = {p: sum(by_panel[p].values()) / 3 for p in panels}
    families = {
        family: sum(panel_scores[p] for p in members) / len(members)
        for family, members in paths.CORRIDOR_FAMILIES.items()
        if all(p in panel_scores for p in members)
    }
    return {
        "s_state": sum(families.values()) / len(families),
        "rows": rows,
        "panel_scores": panel_scores,
        "family_scores": families,
        "scored_cells": sum(r["n"] for r in rows),
    }


def score_official(
    model, infos, target_frames, days_by_panel, device, *, amp, batch_size, blends=BLENDS
):
    """Predict from masked inputs first, then open original observations to score."""
    accum = {b: defaultdict(lambda: {"v2": 0.0, "q2": 0.0, "n": 0}) for b in blends}
    expected = sum(frame.height for frame in target_frames.values())
    for panel in paths.PANELS:
        info = infos[panel]
        for day in days_by_panel[panel]:
            masked = load_day(panel, "train", day, info, masked=True)
            target, regime = targets_for_day(target_frames[panel], masked, info)
            if np.any(target & ~masked.eligible):
                raise ValueError(f"Official target is not score eligible: {panel}/{day}")
            visible = np.isfinite(masked.speed) & np.isfinite(masked.flow_lane)
            if (target & visible).any():
                raise ValueError(f"Official target is visible in input: {panel}/{day}")
            base, correction = predict_masked_day(
                model, masked, info, device, amp=amp, batch_size=batch_size
            )
            # This read is deliberately *after* model prediction.
            truth = load_day(panel, "train", day, info, masked=False)
            if not (
                truth.timestamps == masked.timestamps
                and truth.eligible[target].all()
                and np.isfinite(truth.speed[target]).all()
                and np.isfinite(truth.flow_lane[target]).all()
            ):
                raise ValueError(f"Official truth mismatch: {panel}/{day}")
            for blend in blends:
                pred = base + blend * correction * np.array(
                    [RESIDUAL_SPEED_SCALE, RESIDUAL_FLOW_LANE_SCALE], dtype=np.float32
                )
                pred[..., 0] = np.clip(
                    pred[..., 0], 5.0, np.minimum(135.0, 1.05 * info.free_speeds)[None]
                )
                pred[..., 1] = np.maximum(pred[..., 1], 0)
                if not np.isfinite(pred[target]).all():
                    raise ValueError(f"Nonfinite predictions: {panel}/{day}")
                sv = pred[..., 0][target] - truth.speed[target]
                sq = pred[..., 1][target] - truth.flow_lane[target]
                stat = accum[blend][(panel, regime)]
                stat["v2"] += float(np.square(sv.astype(np.float64)).sum())
                stat["q2"] += float(np.square(sq.astype(np.float64)).sum())
                stat["n"] += int(target.sum())
    summaries = {str(blend): _summarize_scores(stats, paths.PANELS) for blend, stats in accum.items()}
    for summary in summaries.values():
        if summary["scored_cells"] != expected:
            raise ValueError(f"Scored {summary['scored_cells']} of {expected} official target rows")
    return summaries


def _date_lists(start, end, *, limit=None):
    dates = {panel: all_train_dates(panel, start, end) for panel in paths.PANELS}
    if limit is not None:
        if limit == "regimes":
            selected = {}
            for panel, days in dates.items():
                by_regime = {}
                for day in days:
                    for regime in paths.REGIMES:
                        file = (paths.mainline_masked_dir(panel, "train") /
                                f"mask_regime={regime}" /
                                f"synthetic_mainline_{day.replace('-', '_')}.parquet")
                        if file.exists():
                            by_regime.setdefault(regime, day)
                if set(by_regime) != set(paths.REGIMES):
                    raise ValueError(f"Missing representative regime in {panel}: {by_regime}")
                selected[panel] = sorted(by_regime.values())
            dates = selected
        else:
            dates = {p: days[:limit] for p, days in dates.items()}
    if any(not days for days in dates.values()):
        raise ValueError(f"Missing dates in {start}..{end}")
    return dates


def _target_frames(dates_by_panel):
    return {
        p: official_targets(p, dates[0], dates[-1]).filter(
            pl.col("timestamp").str.slice(0, 10).is_in(dates)
        )
        for p, dates in dates_by_panel.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--large-batch-size", type=int, default=2)
    parser.add_argument("--large-panel-threshold", type=int, default=160)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--progress-days", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("runs/staeformer_residual_001"))
    parser.add_argument("--smoke", action="store_true", help="One day/panel and one epoch")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.smoke:
        args.epochs = 1
        args.out = Path("runs/staeformer_residual_smoke")
    args.out.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected physical GPU 9 masked as the only CUDA device")
    device = torch.device("cuda:0")
    infos = panel_infos()
    n_nodes = sum(info.n_links for info in infos.values())
    train_dates = _date_lists(TRAIN_START, TRAIN_END, limit=1 if args.smoke else None)
    select_dates = _date_lists(SELECT_START, SELECT_END, limit="regimes" if args.smoke else None)
    test_dates = _date_lists(TEST_START, TEST_END, limit="regimes" if args.smoke else None)
    if set(d for days in train_dates.values() for d in days) & set(
        d for days in select_dates.values() for d in days
    ):
        raise ValueError("Train and selection dates overlap")
    select_targets = _target_frames(select_dates)
    test_targets = _target_frames(test_dates)

    model = STAEformerResidual(total_nodes=n_nodes, steps=WINDOW).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = None  # bfloat16 does not require GradScaler
    settings = {
        "epochs": args.epochs, "seed": args.seed, "window": WINDOW, "architecture": "STAEformer-style residual",
        "random_mask_rates": REGIME_PROBABILITIES,
        "model": {"feature_dim": 32, "tod_dim": 16, "dow_dim": 8, "adaptive_dim": 40,
                  "heads": 4, "layers_each_axis": 3, "dropout": 0.1},
        "train_dates": [TRAIN_START, TRAIN_END], "selection_dates": [SELECT_START, SELECT_END],
        "test_dates": [TEST_START, TEST_END], "device": str(device),
        "visible_physical_gpu": 9, "torch": torch.__version__,
        "panel_nodes": {p: info.n_links for p, info in infos.items()},
        "batch_size": args.batch_size, "large_batch_size": args.large_batch_size,
        "large_panel_threshold": args.large_panel_threshold, "lr": args.lr, "amp_bfloat16": args.amp,
        "smoke": args.smoke,
    }
    (args.out / "config.json").write_text(json.dumps(settings, indent=2))
    print(json.dumps({"device": str(device), "node_count": n_nodes,
                      "train_panel_days": sum(map(len, train_dates.values())),
                      "selection_panel_days": sum(map(len, select_dates.values())),
                      "test_panel_days": sum(map(len, test_dates.values()))}), flush=True)

    history = []
    best = -float("inf")
    selected_blend = None
    for epoch in range(1, args.epochs + 1):
        train_result = train_epoch(model, optimizer, scaler, infos, train_dates, epoch, args, device)
        selected = score_official(
            model, infos, select_targets, select_dates, device,
            amp=args.amp, batch_size=args.batch_size,
        )
        winning_blend = max(selected, key=lambda b: selected[b]["s_state"])
        score = selected[winning_blend]["s_state"]
        record = {"epoch": epoch, "train": train_result, "selection_scores":
                  {b: v["s_state"] for b, v in selected.items()},
                  "best_blend_this_epoch": winning_blend}
        history.append(record)
        (args.out / "history.json").write_text(json.dumps(history, indent=2))
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch, "settings": settings}, args.out / "last.pt")
        if score > best:
            best, selected_blend = score, winning_blend
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "blend": selected_blend, "settings": settings}, args.out / "best.pt")
            (args.out / "selection_best.json").write_text(json.dumps(
                {"epoch": epoch, "blend": selected_blend, "scores": selected}, indent=2
            ))
        print(f"epoch {epoch}: train_loss={train_result['loss']:.5f}; "
              f"Jan S_state={score:.6f} at residual blend {winning_blend}; best={best:.6f}",
              flush=True)

    checkpoint = torch.load(args.out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    blend = float(checkpoint["blend"])
    result = score_official(
        model, infos, test_targets, test_dates, device,
        amp=args.amp, batch_size=args.batch_size, blends=(0.0, blend)
    )
    output = {
        "selected_epoch": checkpoint["epoch"], "selected_blend": blend,
        "selection_s_state": best, "test": result,
        "test_target_rows": sum(frame.height for frame in test_targets.values()),
    }
    (args.out / "test_result.json").write_text(json.dumps(output, indent=2))
    print(f"February official-mask test: S_state={result[str(blend)]['s_state']:.6f}; "
          f"interpolation-only={result['0.0']['s_state']:.6f}; "
          f"rows={output['test_target_rows']:,}", flush=True)


if __name__ == "__main__":
    main()
