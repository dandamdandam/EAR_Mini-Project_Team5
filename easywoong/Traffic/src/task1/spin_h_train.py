"""Train the official-style SPIN-H competitor for Task 1.

Full protocol: Jun-Dec synthetic hidden cells for training, January released
Task 1 targets for checkpoint/blend selection, February released targets once
for final testing.  One neural network is shared by all ten panels.  Only the
masked layer is loaded before each official prediction; unmasked observations
are opened afterwards inside the score function.

Run from Traffic with GPU 9 reserved:
    CUDA_VISIBLE_DEVICES=9 conda run -n easypj python -m src.task1.spin_h_train

CPU-only implementation check (no CUDA work):
    conda run -n easypj python -m src.task1.spin_h_train --cpu-smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# Shared-server constraint: only physical GPU 9 may be visible to this process.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "9"

import numpy as np
import torch

from src.io import paths
from src.task1.spin_h_graph import SensorGraph, build_sensor_graph
from src.task1.spin_h_model import SPINH, masked_l1_with_auxiliary
from src.task1.staeformer_data import (
    FLOW_LANE_SCALE, REGIME_PROBABILITIES, SPEED_SCALE, Day, PanelInfo,
    interpolate_visible, load_day, panel_infos, targets_for_day,
)
from src.task1.train_staeformer import _date_lists, _summarize_scores, _target_frames

WINDOW = 24
BLENDS = (0.0, 0.25, 0.5, 0.75, 1.0)
TRAIN_START, TRAIN_END = "2030-06-01", "2030-12-31"
SELECT_START, SELECT_END = "2031-01-01", "2031-01-31"
TEST_START, TEST_END = "2031-02-01", "2031-02-28"

@dataclass(frozen=True)
class ChannelScaler:
    mean: np.ndarray  # [2] speed and flow/lane
    std: np.ndarray   # [2]

    def to_json(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def fit_channel_scaler(infos: dict[str, PanelInfo],
                       days_by_panel: dict[str, list[str]]) -> ChannelScaler:
    """Fit the official-style StandardScaler on Jun-Dec observed values only."""
    sums = np.zeros(2, dtype=np.float64)
    squares = np.zeros(2, dtype=np.float64)
    counts = np.zeros(2, dtype=np.int64)
    for panel in paths.PANELS:
        for day_name in days_by_panel[panel]:
            day = load_day(panel, "train", day_name, infos[panel], masked=False)
            for channel, value in enumerate((day.speed, day.flow_lane)):
                valid = value[np.isfinite(value)].astype(np.float64)
                sums[channel] += valid.sum()
                squares[channel] += np.square(valid).sum()
                counts[channel] += valid.size
    if np.any(counts == 0):
        raise ValueError("No Jun-Dec observed values for channel scaler")
    mean = sums / counts
    std = np.sqrt(np.maximum(squares / counts - np.square(mean), 1e-6))
    return ChannelScaler(mean.astype(np.float32), std.astype(np.float32))


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def time_features(day: str, starts: np.ndarray) -> np.ndarray:
    """Four periodic day/week coordinates used by the official SPIN runner."""
    slots = starts[:, None] + np.arange(WINDOW)[None, :]
    daily = 2 * math.pi * slots / 288.0
    weekly = 2 * math.pi * (date.fromisoformat(day).weekday() + slots / 288.0) / 7.0
    return np.stack((np.sin(daily), np.cos(daily), np.sin(weekly), np.cos(weekly)),
                    axis=-1).astype(np.float32)


def _window_values(day: Day, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = starts[:, None] + np.arange(WINDOW)[None, :]
    value = np.stack((day.speed[indices], day.flow_lane[indices]), axis=-1)
    eligible = day.eligible[indices]
    finite = np.isfinite(value).all(axis=-1)
    return value, eligible, finite


def regime_for_visit(panel: str, day_name: str, cycle: int) -> tuple[str, float]:
    """Balanced day-level regime: each date sees all rates over three visits."""
    if cycle < 0:
        raise ValueError("cycle must be nonnegative")
    index = (date.fromisoformat(day_name).toordinal()
             + paths.PANELS.index(panel) + cycle) % 3
    return paths.REGIMES[index], REGIME_PROBABILITIES[index]


def train_tensors(day: Day, starts: np.ndarray, rng: np.random.Generator,
                  device: torch.device, scaler: ChannelScaler,
                  mask_probability: float) -> tuple[torch.Tensor, ...]:
    value, eligible, finite = _window_values(day, starts)
    if mask_probability not in REGIME_PROBABILITIES:
        raise ValueError(f"Invalid day-level regime probability: {mask_probability}")
    # One panel-day has exactly one regime. Every sampled window from this day
    # uses the same probability; only the individual hidden cells vary.
    target = eligible & finite & (rng.random(finite.shape) < mask_probability)
    visible = finite & ~target
    standardized = (value - scaler.mean) / scaler.std
    x = np.where(visible[..., None], standardized, 0.0)
    truth = np.where(finite[..., None], standardized, 0.0)
    if not target.any() or not np.isfinite(x).all() or not np.isfinite(truth).all():
        raise ValueError(f"Invalid synthetic training masks for {day.panel}/{day.day}")
    return (
        torch.from_numpy(x.astype(np.float32)).to(device),
        torch.from_numpy(time_features(day.day, starts)).to(device),
        torch.from_numpy(visible[..., None].astype(np.float32)).to(device),
        torch.from_numpy(truth.astype(np.float32)).to(device),
        torch.from_numpy(target[..., None]).to(device),
    )


def visible_tensors(day: Day, starts: np.ndarray, device: torch.device,
                    scaler: ChannelScaler) -> tuple[torch.Tensor, ...]:
    value, _, finite = _window_values(day, starts)
    x = np.where(finite[..., None], (value - scaler.mean) / scaler.std, 0.0)
    if not np.isfinite(x).all():
        raise ValueError(f"Nonfinite input after masking: {day.panel}/{day.day}")
    return (
        torch.from_numpy(x.astype(np.float32)).to(device),
        torch.from_numpy(time_features(day.day, starts)).to(device),
        torch.from_numpy(finite[..., None].astype(np.float32)).to(device),
    )


def graph_tensors(infos: dict[str, PanelInfo], graphs: dict[str, SensorGraph],
                  device: torch.device) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    return {
        panel: (
            torch.as_tensor(info.global_node_ids, dtype=torch.long, device=device),
            torch.as_tensor(graphs[panel].edge_index, dtype=torch.long, device=device),
        )
        for panel, info in infos.items()
    }


def train_epoch(model: SPINH, optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler.LRScheduler,
                infos: dict[str, PanelInfo], days_by_panel: dict[str, list[str]],
                graph: dict[str, tuple[torch.Tensor, torch.Tensor]],
                epoch: int, args: argparse.Namespace, device: torch.device,
                scaler: ChannelScaler) -> dict:
    model.train()
    rng = np.random.default_rng(args.seed + epoch * 1009)
    pairs = [(panel, day_name) for panel in paths.PANELS
             for day_name in days_by_panel[panel]]
    permutations: dict[int, np.ndarray] = {}
    stats: list[tuple[float, int]] = []
    regime_batches = {"R1": 0, "R2": 0, "R3": 0}
    start_time = time.perf_counter()
    for batch_idx in range(args.batches_per_epoch):
        # A shuffled cycle visits every Jun-Dec panel-day before repeating.
        # This preserves the upstream 300-batch epoch budget while ensuring
        # complete date exposure after ceil(2140 / 300) epochs.
        position = (epoch - 1) * args.batches_per_epoch + batch_idx
        cycle, offset = divmod(position, len(pairs))
        if cycle not in permutations:
            permutations[cycle] = np.random.default_rng(
                args.seed + cycle * 7919
            ).permutation(len(pairs))
        panel, day_name = pairs[int(permutations[cycle][offset])]
        # Adjacent panel-days start at different regimes. Repeated visits to
        # the same day rotate R1 -> R2 -> R3, balancing all three per day.
        regime, probability = regime_for_visit(panel, day_name, cycle)
        regime_batches[regime] += 1
        raw = load_day(panel, "train", day_name, infos[panel], masked=False)
        batch_size = args.batch_size if infos[panel].n_links <= args.large_panel_threshold else args.large_batch_size
        starts = rng.integers(0, 288 - WINDOW + 1, size=batch_size)
        x, u, mask, truth, target = train_tensors(
            raw, starts, rng, device, scaler, probability
        )
        node_index, edge_index = graph[panel]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=args.amp and device.type == "cuda"):
            prediction, auxiliary = model(x, u, mask, edge_index, node_index)
            loss = masked_l1_with_auxiliary(prediction, auxiliary, truth, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite SPIN-H loss: {panel}/{day_name}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        stats.append((float(loss.detach()), int(target.sum())))
        if (batch_idx + 1) % args.progress_batches == 0:
            print(f"epoch {epoch}: {batch_idx + 1}/{args.batches_per_epoch} batches, "
                  f"loss={np.average([s[0] for s in stats], weights=[s[1] for s in stats]):.5f}, "
                  f"elapsed={time.perf_counter()-start_time:.0f}s", flush=True)
    scheduler.step()
    return {
        "loss": float(np.average([s[0] for s in stats], weights=[s[1] for s in stats])),
        "supervised_cells": sum(s[1] for s in stats),
        "batches": len(stats), "seconds": round(time.perf_counter() - start_time, 1),
        "learning_rate": optimizer.param_groups[0]["lr"],
        "regime_batches": regime_batches,
    }


@torch.no_grad()
def predict_masked_day(model: SPINH, masked: Day, info: PanelInfo,
                       graph: tuple[torch.Tensor, torch.Tensor], device: torch.device,
                       scaler: ChannelScaler, *, amp: bool,
                       batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    visible = np.isfinite(masked.speed) & np.isfinite(masked.flow_lane)
    interpolation = interpolate_visible(masked.speed, masked.flow_lane, visible)
    node_index, edge_index = graph
    output = np.empty((*masked.speed.shape, 2), dtype=np.float32)
    starts = np.arange(0, 288, WINDOW)
    for j in range(0, len(starts), batch_size):
        group = starts[j:j + batch_size]
        x, u, mask = visible_tensors(masked, group, device, scaler)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=amp and device.type == "cuda"):
            prediction, _ = model(x, u, mask, edge_index, node_index)
        physical = prediction.float().cpu().numpy() * scaler.std + scaler.mean
        for start, values in zip(group, physical):
            output[int(start):int(start) + WINDOW] = values
    if not np.isfinite(output).all():
        raise ValueError(f"Nonfinite SPIN-H output: {masked.panel}/{masked.day}")
    return interpolation, output


def _clip_prediction(prediction: np.ndarray, info: PanelInfo) -> np.ndarray:
    prediction[..., 0] = np.clip(prediction[..., 0], 5.0,
                                 np.minimum(135.0, 1.05 * info.free_speeds)[None])
    prediction[..., 1] = np.maximum(prediction[..., 1], 0.0)
    return prediction


@torch.no_grad()
def score_official(model: SPINH, infos: dict[str, PanelInfo],
                   graph: dict[str, tuple[torch.Tensor, torch.Tensor]],
                   target_frames, days_by_panel, device: torch.device,
                   scaler: ChannelScaler,
                   *, amp: bool, batch_size: int, large_batch_size: int,
                   large_panel_threshold: int, blends=BLENDS) -> dict:
    """Score only authoritative template cells; open truth after prediction."""
    accum = {blend: defaultdict(lambda: {"v2": 0.0, "q2": 0.0, "n": 0}) for blend in blends}
    expected = sum(frame.height for frame in target_frames.values())
    for panel in paths.PANELS:
        info = infos[panel]
        eval_batch = batch_size if info.n_links <= large_panel_threshold else large_batch_size
        for day_name in days_by_panel[panel]:
            masked = load_day(panel, "train", day_name, info, masked=True)
            target, regime = targets_for_day(target_frames[panel], masked, info)
            if np.any(target & ~masked.eligible):
                raise ValueError(f"Ineligible official target: {panel}/{day_name}")
            visible = np.isfinite(masked.speed) & np.isfinite(masked.flow_lane)
            if np.any(target & visible):
                raise ValueError(f"Official target visible to SPIN-H: {panel}/{day_name}")
            interpolation, direct = predict_masked_day(
                model, masked, info, graph[panel], device, scaler,
                amp=amp, batch_size=eval_batch
            )
            # Truth is deliberately opened only after masked-only prediction.
            truth = load_day(panel, "train", day_name, info, masked=False)
            if not (truth.timestamps == masked.timestamps and truth.eligible[target].all()
                    and np.isfinite(truth.speed[target]).all()
                    and np.isfinite(truth.flow_lane[target]).all()):
                raise ValueError(f"Official truth mismatch: {panel}/{day_name}")
            for blend in blends:
                predicted = _clip_prediction(interpolation + blend * (direct - interpolation), info)
                sv = predicted[..., 0][target] - truth.speed[target]
                sq = predicted[..., 1][target] - truth.flow_lane[target]
                stat = accum[blend][(panel, regime)]
                stat["v2"] += float(np.square(sv.astype(np.float64)).sum())
                stat["q2"] += float(np.square(sq.astype(np.float64)).sum())
                stat["n"] += int(target.sum())
    summaries = {str(blend): _summarize_scores(stats, paths.PANELS)
                 for blend, stats in accum.items()}
    for summary in summaries.values():
        if summary["scored_cells"] != expected:
            raise ValueError(f"Scored {summary['scored_cells']} of {expected} official targets")
    return summaries


def _magic_schedule(optimizer: torch.optim.Optimizer, epochs: int):
    """The official SPIN-H warmup/cosine scheduler, stepped once per epoch."""
    def factor(step: int) -> float:
        if step < 12:
            return max(0.1, step / 12.0)
        progress = (step - 12) / max(1, epochs - 12)
        if progress >= 1.0:
            return 0.0
        cycle = (max(1, epochs // 100) * progress) % 1.0
        return max(0.1, 0.5 * (1 + math.cos(math.pi * cycle)) * (1 - 0.67 * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def cpu_smoke(model: SPINH, infos: dict[str, PanelInfo],
              graphs: dict[str, SensorGraph], args: argparse.Namespace) -> None:
    """One real-data forward/backward and one masked-only official inference."""
    device = torch.device("cpu")
    torch.set_num_threads(2)
    panel = "D12_I405_S"
    graph = graph_tensors(infos, graphs, device)[panel]
    day_name = _date_lists(TRAIN_START, TRAIN_END, limit=1)[panel][0]
    raw = load_day(panel, "train", day_name, infos[panel], masked=False)
    scaler = ChannelScaler(np.zeros(2, dtype=np.float32),
                           np.asarray([SPEED_SCALE, FLOW_LANE_SCALE], dtype=np.float32))
    rng = np.random.default_rng(args.seed)
    smoke_probability = REGIME_PROBABILITIES[1]
    visit_probabilities = [regime_for_visit(panel, day_name, cycle)[1]
                           for cycle in range(3)]
    if set(visit_probabilities) != set(REGIME_PROBABILITIES):
        raise ValueError("Day-level regime rotation is not balanced")
    x, u, mask, truth, target = train_tensors(
        raw, np.asarray([0]), rng, device, scaler, smoke_probability
    )
    model.train()
    prediction, auxiliary = model(x, u, mask, graph[1], graph[0])
    loss = masked_l1_with_auxiliary(prediction, auxiliary, truth, target)
    loss.backward()
    grad_finite = all(p.grad is None or torch.isfinite(p.grad).all().item()
                      for p in model.parameters())
    if not grad_finite:
        raise FloatingPointError("Nonfinite CPU smoke gradient")
    jan_days = _date_lists(SELECT_START, SELECT_END, limit=1)
    jan_day = jan_days[panel][0]
    targets = _target_frames({panel: [jan_day]})[panel]
    masked = load_day(panel, "train", jan_day, infos[panel], masked=True)
    target_mask, regime = targets_for_day(targets, masked, infos[panel])
    model.eval()
    with torch.no_grad():
        x, u, mask = visible_tensors(masked, np.asarray([0]), device, scaler)
        official_prediction, _ = model(x, u, mask, graph[1], graph[0])
    if not torch.isfinite(official_prediction).all() or not target_mask.any():
        raise ValueError("CPU smoke official-mask inference failed")
    report = {
        "device": "cpu", "panel": panel, "train_day": day_name, "selection_day": jan_day,
        "training_window_shape": list(prediction.shape), "training_hidden_cells": int(target.sum()),
        "training_mask_probability": smoke_probability,
        "same_day_visit_probabilities": visit_probabilities,
        "training_loss": float(loss), "finite_gradients": grad_finite,
        "official_target_cells_on_day": int(target_mask.sum()),
        "regime": regime, "official_window_prediction_finite": True,
        "graph_edges": graphs[panel].n_edges,
        "smoke_scaler": scaler.to_json(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cpu_smoke.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


def gpu_benchmark(model: SPINH, infos: dict[str, PanelInfo],
                  graphs: dict[str, SensorGraph], args: argparse.Namespace) -> None:
    """Measure one largest-panel train batch and one full-day masked inference."""
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected only physical GPU 9 masked as cuda:0")
    device = torch.device("cuda:0")
    model.to(device)
    panel = max(infos, key=lambda name: infos[name].n_links)
    node_index, edge_index = graph_tensors(infos, graphs, device)[panel]
    day_name = _date_lists(TRAIN_START, TRAIN_END, limit=1)[panel][0]
    raw = load_day(panel, "train", day_name, infos[panel], masked=False)
    values = np.stack((raw.speed, raw.flow_lane), axis=-1)
    channel_means = np.nanmean(values, axis=(0, 1))
    channel_stds = np.nanstd(values, axis=(0, 1))
    scaler = ChannelScaler(channel_means.astype(np.float32),
                           np.maximum(channel_stds, 1e-3).astype(np.float32))
    starts = np.arange(args.large_batch_size, dtype=np.int64)
    x, u, mask, truth, target = train_tensors(
        raw, starts, np.random.default_rng(args.seed), device, scaler,
        REGIME_PROBABILITIES[1]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    model.train()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.amp):
        prediction, auxiliary = model(x, u, mask, edge_index, node_index)
        loss = masked_l1_with_auxiliary(prediction, auxiliary, truth, target)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - start
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    jan_day = _date_lists(SELECT_START, SELECT_END, limit=1)[panel][0]
    masked = load_day(panel, "train", jan_day, infos[panel], masked=True)
    start = time.perf_counter()
    _, output = predict_masked_day(model, masked, infos[panel],
                                   (node_index, edge_index), device, scaler,
                                   amp=args.amp, batch_size=args.large_eval_batch_size)
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - start
    report = {
        "physical_gpu": 9, "panel": panel, "n_nodes": infos[panel].n_links,
        "n_edges": graphs[panel].n_edges, "batch_size": args.large_batch_size,
        "train_batch_seconds": train_seconds,
        "jan_day_inference_seconds": inference_seconds,
        "peak_gpu_memory_gib": peak_gib,
        "training_loss": float(loss.detach()),
        "prediction_finite": bool(np.isfinite(output).all()),
        "note": "Single largest-panel batch; full-run scaler fitting and varying panel sizes are excluded.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "gpu_benchmark.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batches-per-epoch", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--large-batch-size", type=int, default=4)
    parser.add_argument("--large-panel-threshold", type=int, default=160)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--large-eval-batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-batches", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--gpu-benchmark", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("runs/spin_h_001"))
    args = parser.parse_args()
    if args.epochs < 1 or args.batches_per_epoch < 1 or args.batch_size < 1:
        parser.error("epochs, batches-per-epoch and batch-size must be positive")
    seed_all(args.seed)
    infos = panel_infos()
    graphs = {panel: build_sensor_graph(info) for panel, info in infos.items()}
    n_nodes = sum(info.n_links for info in infos.values())
    model = SPINH(n_nodes)
    if args.cpu_smoke:
        cpu_smoke(model, infos, graphs, args)
        return
    if args.gpu_benchmark:
        gpu_benchmark(model, infos, graphs, args)
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected only physical GPU 9 masked as cuda:0")
    device = torch.device("cuda:0")
    model = model.to(device)
    graph = graph_tensors(infos, graphs, device)
    train_dates = _date_lists(TRAIN_START, TRAIN_END)
    selection_dates = _date_lists(SELECT_START, SELECT_END)
    test_dates = _date_lists(TEST_START, TEST_END)
    scaler = fit_channel_scaler(infos, train_dates)
    selection_targets = _target_frames(selection_dates)
    if any(set(train_dates[p]) & (set(selection_dates[p]) | set(test_dates[p]))
           for p in paths.PANELS):
        raise ValueError("Training dates overlap selection/test")
    settings = {
        "architecture": "SPIN-H, PyTorch-only port of official model", "seed": args.seed,
        "source_url": "https://github.com/Graph-Machine-Learning-Group/spin",
        "window": WINDOW, "random_mask_rates": REGIME_PROBABILITIES,
        "model": {"h_size": 32, "z_size": 128, "z_heads": 4, "n_layers": 5,
                  "eta": 3, "message_layers": 1, "reweight": "softmax",
                  "spatial_aggr": "softmax", "update_z_cross": False},
        "train_dates": [TRAIN_START, TRAIN_END],
        "training_sampling": "shuffled cycle over all Jun-Dec panel-days; one rotated R1/R2/R3 probability per panel-day visit; random stride-1 24-step windows per visit",
        "train_panel_days": sum(len(days) for days in train_dates.values()),
        "selection_dates": [SELECT_START, SELECT_END],
        "test_dates": [TEST_START, TEST_END],
        "epochs": args.epochs, "batches_per_epoch": args.batches_per_epoch,
        "batch_size": args.batch_size, "large_batch_size": args.large_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "large_eval_batch_size": args.large_eval_batch_size,
        "large_panel_threshold": args.large_panel_threshold,
        "lr": args.lr, "patience": args.patience, "amp_bfloat16": args.amp,
        "scaler": scaler.to_json(),
        "device": str(device), "visible_physical_gpu": 9, "torch": torch.__version__,
        "panel_nodes": {panel: info.n_links for panel, info in infos.items()},
        "graph": {panel: {"edges": g.n_edges, "ordered_edges": g.ordered_edges,
                           "explicit_edges": g.explicit_edges, "symmetric": g.symmetric}
                  for panel, g in graphs.items()},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(settings, indent=2))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = _magic_schedule(optimizer, args.epochs)
    best = -float("inf")
    best_epoch = None
    selected_blend = None
    history = []
    for epoch in range(1, args.epochs + 1):
        trained = train_epoch(model, optimizer, scheduler, infos, train_dates,
                              graph, epoch, args, device, scaler)
        selected = score_official(
            model, infos, graph, selection_targets, selection_dates, device, scaler,
            amp=args.amp, batch_size=args.eval_batch_size,
            large_batch_size=args.large_eval_batch_size,
            large_panel_threshold=args.large_panel_threshold,
        )
        winner = max(selected, key=lambda b: selected[b]["s_state"])
        score = selected[winner]["s_state"]
        history.append({"epoch": epoch, "train": trained,
                        "selection_scores": {b: v["s_state"] for b, v in selected.items()},
                        "best_blend_this_epoch": float(winner)})
        (args.out / "history.json").write_text(json.dumps(history, indent=2))
        if score > best:
            best, best_epoch, selected_blend = score, epoch, float(winner)
            torch.save({"model": model.state_dict(), "epoch": epoch, "blend": selected_blend,
                        "settings": settings}, args.out / "best.pt")
            (args.out / "selection_best.json").write_text(json.dumps(
                {"epoch": epoch, "blend": selected_blend, "scores": selected}, indent=2
            ))
        print(f"epoch {epoch}: train loss={trained['loss']:.5f}; "
              f"Jan S_state={score:.6f} at blend {winner}; best={best:.6f}", flush=True)
        if (epoch - best_epoch >= args.patience
                and epoch * args.batches_per_epoch >= settings["train_panel_days"]):
            print(f"Early stopping after {epoch} epochs", flush=True)
            break
    checkpoint = torch.load(args.out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    blend = float(checkpoint["blend"])
    test_targets = _target_frames(test_dates)
    tested = score_official(
        model, infos, graph, test_targets, test_dates, device, scaler,
        amp=args.amp, batch_size=args.eval_batch_size,
        large_batch_size=args.large_eval_batch_size,
        large_panel_threshold=args.large_panel_threshold, blends=tuple(sorted({0.0, blend})),
    )
    result = {"selected_epoch": checkpoint["epoch"], "selected_blend": blend,
              "selection_s_state": best, "test": tested,
              "test_target_rows": sum(frame.height for frame in test_targets.values())}
    (args.out / "test_result.json").write_text(json.dumps(result, indent=2))
    print(f"February official-mask test: S_state={tested[str(blend)]['s_state']:.6f}; "
          f"interpolation-only={tested['0.0']['s_state']:.6f}; "
          f"rows={result['test_target_rows']:,}", flush=True)


if __name__ == "__main__":
    main()
