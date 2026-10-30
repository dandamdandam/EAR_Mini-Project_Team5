"""Selected Task 1 upgrades: chronological selection, full refit, public inference.

All model choices use January only. February is a reporting set already used
in earlier experiments, not a fresh untouched benchmark. Public/private labels
are never available. The all-train 003 checkpoint is only a FINAL refit seed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
os.environ.setdefault("POLARS_MAX_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import polars as pl
import torch
from torch import nn

from src.io import paths
from src.preprocess.run import cache_path, sha256
from src.task1.imputeformer_model import ImputeFormerTask1, imputeformer_objective
from src.task1.staeformer_data import (
    day_time_features, load_day, model_inputs, panel_infos,
    synthetic_target_mask, targets_for_day,
)
from src.task1.staeformer_residual import STAEformerResidual, reconstruction_loss
from src.task1.train_staeformer import _date_lists, _target_frames, _summarize_scores, seed_all

CONFIG = Path("configs/task1_upgrade_004_006.json")
SCALE = np.array([25.0, 600.0], dtype=np.float32)


def write_json(file, value):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_suffix(file.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(file)


def extended_inputs(day, visible, quality=None):
    features, base = model_inputs(day, visible)
    if quality is None:
        return features, base
    if quality.shape != visible.shape or not np.isfinite(quality).all():
        raise ValueError("Invalid observation quality shape/value")
    steps = visible.shape[0]
    index = np.arange(steps)[:, None]
    previous = np.maximum.accumulate(np.where(visible, index, -steps), axis=0)
    following = np.minimum.accumulate(np.where(visible, index, 2 * steps)[::-1], axis=0)[::-1]
    before = np.log1p(np.minimum(index - previous, steps)) / np.log1p(steps)
    after = np.log1p(np.minimum(following - index, steps)) / np.log1p(steps)
    extra = np.stack([np.where(visible, quality / 100.0, 0.0), before, after], axis=-1)
    return np.concatenate([features, extra.astype(np.float32)], axis=-1), base


def input_arrays(day, visible, cfg, cache, split, masked):
    quality = None
    if cfg["input_dim"] == 8:
        with np.load(cache_path(cache, day.panel, split, day.day, masked), allow_pickle=False) as saved:
            quality = saved["pct_observed"]
    return extended_inputs(day, visible, quality)


def make_model(kind, cfg, total_nodes):
    kwargs = dict(total_nodes=total_nodes, steps=cfg["window"], input_dim=cfg["input_dim"], **cfg["architecture"])
    if kind == "staeformer":
        return STAEformerResidual(**kwargs)
    model = ImputeFormerTask1(**kwargs)
    # The encoder retains ImputeFormer's projected temporal and factorized
    # spatial attention. A smooth residual head is an experimental change to
    # the previous unstable adaptation (not a paper reproduction).
    if cfg.get("residual_head_activation") == "GELU":
        model.readout[1], model.readout[3] = nn.GELU(), nn.GELU()
    return model


def warm_start(model, file, cfg, phase):
    saved = torch.load(file, map_location="cpu", weights_only=False)
    settings = saved["settings"]
    dates = settings.get("train_dates")
    if phase == "local" and (not dates or dates[1] > "2030-12-31"):
        raise ValueError("Warm start would leak January/February into local fitting")
    if saved["epoch"] != cfg["warm_epochs"]:
        raise ValueError("Warm checkpoint epoch differs from experiment contract")
    old, target = saved["model"], model.state_dict()
    for key in target:
        if target[key].shape == old[key].shape:
            target[key] = old[key]
        elif key == "feature_proj.weight":
            target[key].zero_()
            target[key][:, :old[key].shape[1]] = old[key]
        elif key == "adaptive.weight":
            old_steps = int(settings["window"])
            if cfg["window"] % old_steps:
                raise ValueError("Warm position embeddings need an integer window expansion")
            target[key] = old[key].reshape(model.total_nodes, old_steps, -1).repeat(
                1, cfg["window"] // old_steps, 1).reshape_as(target[key])
        else:
            raise ValueError(f"Cannot transfer parameter {key}")
    model.load_state_dict(target)
    return {"path": str(file), "sha256": sha256(Path(file)), "trained_dates": dates,
            "epoch": saved["epoch"], "optimizer": "new AdamW for the upgrade stage"}


def tensor_windows(features, base, truth, target, date, starts, window, device):
    tod, dow = day_time_features(date)
    slices = [slice(int(start), int(start) + window) for start in starts]
    arrays = [np.stack([array[sl] for sl in slices]) for array in (features, base, truth, target, tod, dow)]
    return [torch.from_numpy(array).to(device) for array in arrays]


def train_epoch(model, optimizer, infos, dates, cfg, common, epoch, kind, cache, device):
    model.train()
    rng = np.random.default_rng(common["seed"] + epoch * 1009)
    pairs = [(p, d) for p in paths.PANELS for d in dates[p]]
    rng.shuffle(pairs)
    start_time = time.perf_counter()
    total, reconstruction_total, fourier_total, cells, batches = 0., 0., 0., 0, 0
    for index, (panel, date) in enumerate(pairs):
        info = infos[panel]
        raw = load_day(panel, "train", date, info, masked=False)
        target = synthetic_target_mask(raw, (0.2, 0.3, 0.5)[index % 3], rng)
        visible = np.isfinite(raw.speed) & np.isfinite(raw.flow_lane) & ~target
        features, base = input_arrays(raw, visible, cfg, cache, "train", False)
        truth = np.stack([raw.speed, raw.flow_lane], axis=-1)
        starts = np.arange(0, 288, cfg["window"])
        rng.shuffle(starts)
        batch_size = cfg["batch_size"] if info.n_links <= 160 else cfg["large_batch_size"]
        node_ids = torch.as_tensor(info.global_node_ids, device=device)
        for offset in range(0, len(starts), batch_size):
            x, b, y, mask, tod, dow = tensor_windows(features, base, truth, target, date,
                starts[offset:offset + batch_size], cfg["window"], device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                residual = model(x, tod, dow, node_ids).float()
                if kind == "imputeformer":
                    loss, reconstruction, fourier = imputeformer_objective(residual, b, y, mask, x, cfg["fourier_weight"])
                else:
                    loss = reconstruction_loss(residual, b, y, mask)
                    reconstruction, fourier = loss, loss.new_zeros(())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss: {panel}/{date}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = int(mask.sum())
            total += float(loss.detach()) * count
            reconstruction_total += float(reconstruction.detach()) * count
            fourier_total += float(fourier.detach()) * count
            cells += count
            batches += 1
        if (index + 1) % 400 == 0 or index + 1 == len(pairs):
            print(f"{kind} epoch {epoch}: {index + 1}/{len(pairs)} days, loss={total / cells:.6f}, elapsed={time.perf_counter()-start_time:.0f}s", flush=True)
    return {"loss": total / cells, "reconstruction": reconstruction_total / cells,
            "fourier": fourier_total / cells, "supervised_cells": cells, "batches": batches,
            "seconds": round(time.perf_counter() - start_time, 2)}


def window_starts(window, stride):
    starts = list(range(0, 288 - window + 1, stride))
    if starts[-1] != 288 - window:
        starts.append(288 - window)
    return starts


@torch.no_grad()
def infer_day(model, day, info, cfg, cache, split, device):
    model.eval()
    visible = np.isfinite(day.speed) & np.isfinite(day.flow_lane)
    features, base = input_arrays(day, visible, cfg, cache, split, True)
    tod, dow = day_time_features(day.day)
    starts = window_starts(cfg["window"], cfg["inference_stride"])
    correction = np.zeros_like(base)
    count = np.zeros((288, 1, 1), dtype=np.float32)
    node_ids = torch.as_tensor(info.global_node_ids, device=device)
    batch_size = cfg["batch_size"] if info.n_links <= 160 else cfg["large_batch_size"]
    for offset in range(0, len(starts), batch_size):
        slices = [slice(s, s + cfg["window"]) for s in starts[offset:offset + batch_size]]
        x, tt, dd = [torch.from_numpy(np.stack([arr[sl] for sl in slices])).to(device) for arr in (features, tod, dow)]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(x, tt, dd, node_ids).float().cpu().numpy()
        for sl, values in zip(slices, output):
            correction[sl] += values
            count[sl] += 1
    if (count == 0).any():
        raise ValueError("Inference windows failed to cover the whole day")
    return base, correction / count


def physical_prediction(base, correction, blend, info):
    pred = base + float(blend) * correction * SCALE
    pred[..., 0] = np.clip(pred[..., 0], 5.0, np.minimum(135.0, 1.05 * info.free_speeds)[None])
    pred[..., 1] = np.maximum(pred[..., 1], 0.0)
    if not np.isfinite(pred).all():
        raise ValueError("Nonfinite prediction")
    return pred


def target_frame(day, info, target, regime, prediction, truth=None):
    t, n = np.where(target)
    columns = {"panel": [info.panel] * len(t), "timestamp": np.asarray(day.timestamps)[t],
               "station_id": np.asarray(info.station_ids)[n], "link_id": np.asarray(info.link_ids)[n],
               "mask_regime": [regime] * len(t), "speed_kmh": prediction[t, n, 0].astype(np.float64),
               "flow_vph": (prediction[t, n, 1] * info.lanes[n]).astype(np.float64)}
    if truth is not None:
        columns.update(truth_speed=truth.speed[t, n].astype(np.float64),
                       truth_flow_lane=truth.flow_lane[t, n].astype(np.float64), lanes=info.lanes[n].astype(np.float64))
    return pl.DataFrame(columns)


def evaluate(model, infos, dates, targets, cfg, cache, device, blends, save=None):
    if save is not None and len(blends) != 1:
        raise ValueError("Save only the frozen selected blend")
    accum = {float(b): defaultdict(lambda: {"v2": 0., "q2": 0., "n": 0}) for b in blends}
    records = []
    for panel in paths.PANELS:
        info = infos[panel]
        for date in dates[panel]:
            masked = load_day(panel, "train", date, info, masked=True)
            target, regime = targets_for_day(targets[panel], masked, info)
            if (target & np.isfinite(masked.speed)).any() or (target & ~masked.eligible).any():
                raise ValueError("Official target visibility/eligibility mismatch")
            base, correction = infer_day(model, masked, info, cfg, cache, "train", device)
            # Truth is read only after the complete prediction from masked inputs.
            truth = load_day(panel, "train", date, info, masked=False)
            if truth.timestamps != masked.timestamps or not np.isfinite(truth.speed[target]).all() or not np.isfinite(truth.flow_lane[target]).all():
                raise ValueError("Official truth alignment failed")
            for blend in blends:
                pred = physical_prediction(base, correction, blend, info)
                stat = accum[float(blend)][(panel, regime)]
                stat["v2"] += float(np.square(pred[..., 0][target].astype(float) - truth.speed[target]).sum())
                stat["q2"] += float(np.square(pred[..., 1][target].astype(float) - truth.flow_lane[target]).sum())
                stat["n"] += int(target.sum())
                if save is not None:
                    records.append(target_frame(masked, info, target, regime, pred, truth))
    summaries = {str(b): _summarize_scores(stats, paths.PANELS) for b, stats in accum.items()}
    expected = sum(len(frame) for frame in targets.values())
    if any(s["scored_cells"] != expected for s in summaries.values()):
        raise ValueError("Incomplete official target coverage")
    if save is not None:
        pl.concat(records).write_parquet(save)
    return summaries


def learning_rate(cfg, common, epoch):
    first = cfg["warm_epochs"] + 1
    progress = (epoch - first) / max(common["max_epochs"] - first, 1)
    return cfg["lr"] * (0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2)


def checkpoint(model, optimizer, epoch, settings, best_score=None, stale=0):
    return {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
            "settings": settings, "best_score": best_score, "stale": stale,
            "rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state()}


def fit(kind, common, cfg, cache, infos, device, phase, resume=False):
    root = Path(cfg["run"])
    out = root / phase
    out.mkdir(parents=True, exist_ok=True)
    if (out / "last.pt").exists() and not resume:
        raise FileExistsError(f"Preserving {out}; use --resume for interrupted work")
    seed_all(common["seed"])
    model = make_model(kind, cfg, sum(i.n_links for i in infos.values()))
    warm = warm_start(model, cfg[f"warm_{phase}"], cfg, phase) if cfg["warm_epochs"] else None
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    settings = {"kind": kind, "model": cfg, "common": common, "phase": phase, "warm_start": warm,
                "visible_physical_gpu": 9, "amp": "bfloat16", "torch": str(torch.__version__)}
    write_json(out / "config.json", settings)
    dates = _date_lists(*(common["fit_dates"] if phase == "local" else common["final_dates"]))
    start_epoch, history, best_score, stale = cfg["warm_epochs"] + 1, [], -math.inf, 0
    if phase == "local":
        select_dates = _date_lists(*common["selection_dates"])
        select_targets = _target_frames(select_dates)
        limit = common["max_epochs"]
    else:
        selected = json.loads((root / "local/selection_best.json").read_text())
        limit = selected["epoch"]
        settings["selected_epoch"], settings["selected_blend"] = limit, selected["blend"]
        write_json(out / "config.json", settings)
    if resume and (out / "last.pt").exists():
        saved = torch.load(out / "last.pt", map_location=device, weights_only=False)
        if saved["settings"]["model"] != cfg or saved["settings"]["common"] != common:
            raise ValueError("Resume settings differ from the saved experiment")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng_state"].cpu())
        torch.cuda.set_rng_state(saved["cuda_rng_state"].cpu())
        start_epoch = saved["epoch"] + 1
        best_score = saved.get("best_score") if saved.get("best_score") is not None else -math.inf
        stale = saved.get("stale", 0)
        history = json.loads((out / "history.json").read_text())
    for epoch in range(start_epoch, limit + 1):
        if phase == "local" and stale >= common["early_stopping_patience"] and epoch > common["minimum_epochs"]:
            break
        lr = learning_rate(cfg, common, epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr
        train = train_epoch(model, optimizer, infos, dates, cfg, common, epoch, kind, cache, device)
        record = {"epoch": epoch, "lr": lr, "train": train}
        if phase == "local":
            scores = evaluate(model, infos, select_dates, select_targets, cfg, cache, device, [0.0, *common["residual_blends"]])
            blend = max(common["residual_blends"], key=lambda b: scores[str(b)]["s_state"])
            score = scores[str(blend)]["s_state"]
            record.update(selection_scores={b: s["s_state"] for b, s in scores.items()}, selected_blend=blend)
            if score > best_score + common["minimum_improvement"]:
                best_score, stale = score, 0
                torch.save({"model": model.state_dict(), "epoch": epoch, "blend": blend, "settings": settings}, out / "best.pt")
                write_json(out / "selection_best.json", {"epoch": epoch, "blend": blend, "s_state": score, "scores": scores})
            else:
                stale += 1
            print(f"{kind} local epoch={epoch} Jan={score:.6f}, best={best_score:.6f}, stale={stale}", flush=True)
        else:
            print(f"{kind} final epoch={epoch}/{limit}, loss={train['loss']:.6f}", flush=True)
        history.append(record)
        write_json(out / "history.json", history)
        torch.save(checkpoint(model, optimizer, epoch, settings, best_score if phase == "local" else None, stale), out / "last.pt")
    if phase == "final":
        torch.save({"model": model.state_dict(), "epoch": limit, "blend": selected["blend"], "settings": settings}, out / "final.pt")
        write_json(out / "SUCCESS.json", {"epoch": limit, "blend": selected["blend"], "independent_local_test": False})
    else:
        chosen = json.loads((out / "selection_best.json").read_text())
        write_json(out / "SUCCESS.json", {"selected_epoch": chosen["epoch"], "selected_blend": chosen["blend"],
                   "selection_s_state": chosen["s_state"], "trained_to_epoch": history[-1]["epoch"],
                   "stop_reason": "patience" if stale >= common["early_stopping_patience"] else "max_epochs"})


def load_fitted(kind, cfg, infos, device, phase):
    filename = "best.pt" if phase == "local" else "final.pt"
    saved = torch.load(Path(cfg["run"]) / phase / filename, map_location=device, weights_only=False)
    model = make_model(kind, cfg, sum(i.n_links for i in infos.values())).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    return model, saved


def export_local(kind, common, cfg, cache, infos, device):
    model, saved = load_fitted(kind, cfg, infos, device, "local")
    root = Path(cfg["run"])
    for name, date_key in (("selection", "selection_dates"), ("february", "report_dates")):
        dates = _date_lists(*common[date_key])
        scores = evaluate(model, infos, dates, _target_frames(dates), cfg, cache, device,
                          [saved["blend"]], save=root / f"predictions_{name}.parquet")
        write_json(root / f"{name}_result.json", {"epoch": saved["epoch"], "blend": saved["blend"],
                   "dates": common[date_key], "score": scores[str(saved["blend"])],
                   "note": "January selects. February is a previously used reporting set, not fresh model selection."})
        print(f"{kind} {name}: S_state={scores[str(saved['blend'])]['s_state']:.6f}", flush=True)


def predict(kind, cfg, cache, infos, device):
    model, saved = load_fitted(kind, cfg, infos, device, "final")
    root, diagnostics = Path(cfg["run"]), []
    for split in ("validation", "private"):
        records = []
        for panel, info in infos.items():
            target = pl.read_csv(paths.task1_targets(panel, split), columns=["timestamp", "station_id", "link_id", "mask_regime"])
            by_date = target.with_columns(pl.col("timestamp").str.slice(0, 10).alias("date")).partition_by("date", as_dict=True)
            for (date,), rows in sorted(by_date.items()):
                day = load_day(panel, split, date, info, masked=True)
                mask, regime = targets_for_day(rows.drop("date"), day, info)
                if (mask & np.isfinite(day.speed)).any() or (mask & ~day.eligible).any():
                    raise ValueError("Invalid official public target")
                base, correction = infer_day(model, day, info, cfg, cache, split, device)
                pred = physical_prediction(base, correction, saved["blend"], info)
                records.append(target_frame(day, info, mask, regime, pred))
            print(f"{kind} inference {split}/{panel}: {target.height:,} targets", flush=True)
        frame = pl.concat(records)
        frame.write_parquet(root / f"predictions_{split}.parquet")
        for g in frame.partition_by(["panel", "mask_regime"]):
            diagnostics.append({"split": split, "panel": g["panel"][0], "regime": g["mask_regime"][0],
                                "rows": len(g), "speed_min": g["speed_kmh"].min(), "speed_max": g["speed_kmh"].max(),
                                "flow_min": g["flow_vph"].min(), "flow_below_50_fraction": g.filter(pl.col("flow_vph") < 50).height / len(g)})
    write_json(root / "prediction_diagnostics.json", diagnostics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("staeformer", "imputeformer"), required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--stage", choices=("local", "export-local", "final", "predict", "all", "benchmark"), default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    common = json.loads(args.config.read_text())
    cfg, cache = common["models"][args.model], Path(common["cache"])
    if not (cache / "SUCCESS.json").exists() or json.loads((cache / "SUCCESS.json").read_text())["smoke"]:
        raise ValueError("Full validated cache is required")
    os.environ["TFB_DAILY_CACHE"] = str(cache.resolve())
    torch.set_num_threads(8)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected only physical GPU 9")
    device, infos = torch.device("cuda:0"), panel_infos()
    Path(cfg["run"]).mkdir(parents=True, exist_ok=True)
    write_json(Path(cfg["run"]) / "config.json", {"kind": args.model, "model": cfg, "experiment": common})
    if args.stage == "benchmark":
        seed_all(common["seed"])
        model = make_model(args.model, cfg, sum(i.n_links for i in infos.values())).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
        dates = _date_lists(*common["fit_dates"], limit=1)
        report = train_epoch(model, opt, infos, dates, cfg, common, 1, args.model, cache, device)
        report["peak_cuda_memory_mb"] = torch.cuda.max_memory_allocated() / 1024 ** 2
        write_json(Path(cfg["run"]) / "benchmark.json", report)
        print(json.dumps(report), flush=True)
        return
    for stage in ("local", "export-local", "final", "predict") if args.stage == "all" else (args.stage,):
        if stage in ("local", "final"):
            fit(args.model, common, cfg, cache, infos, device, stage, args.resume)
        elif stage == "export-local":
            export_local(args.model, common, cfg, cache, infos, device)
        else:
            predict(args.model, cfg, cache, infos, device)


if __name__ == "__main__":
    main()
