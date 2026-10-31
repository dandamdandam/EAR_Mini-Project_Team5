"""Refit the selected STAEformer configuration on all nine train months.

Five epochs and blend=1 were selected in the earlier January experiment.
This is a submission refit; it does not report an independent February score.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
os.environ.setdefault("POLARS_MAX_THREADS", "8")

import numpy as np
import polars as pl
import torch

from src.io import paths
from src.task1.staeformer_data import load_day, panel_infos, RESIDUAL_SPEED_SCALE, RESIDUAL_FLOW_LANE_SCALE
from src.task1.staeformer_residual import STAEformerResidual
from src.task1.train_staeformer import _date_lists, predict_masked_day, seed_all, train_epoch


def train(out: Path, cache: Path, *, epochs: int = 5, resume: bool = False):
    out.mkdir(parents=True, exist_ok=True)
    os.environ["TFB_DAILY_CACHE"] = str(cache.resolve())
    torch.set_num_threads(8)
    seed_all(42)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Physical GPU 9 must be the only visible CUDA device")
    device = torch.device("cuda:0")
    infos = panel_infos()
    dates = _date_lists("2030-06-01", "2031-02-28")
    args = SimpleNamespace(seed=42, batch_size=4, large_batch_size=2,
                           large_panel_threshold=160, amp=True, progress_days=200)
    model = STAEformerResidual(total_nodes=sum(i.n_links for i in infos.values()), steps=24).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    settings = {"train_dates": ["2030-06-01", "2031-02-28"], "epochs": epochs, "seed": 42,
                "window": 24, "blend": 1.0, "model": "Interpolation + STAEformer residual",
                "preprocessing": json.loads((cache / "SUCCESS.json").read_text()),
                "selection_reference": "runs/staeformer_residual_001/selection_best.json",
                "independent_test": False, "note": "All nine training months are used for submission refit.",
                "panel_nodes": {p: i.n_links for p, i in infos.items()},
                "batch_size": 4, "large_batch_size": 2, "lr": 5e-4,
                "visible_physical_gpu": 9, "amp_bfloat16": True, "torch": str(torch.__version__)}
    (out / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    start, history = 1, []
    if resume:
        saved = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start = saved["epoch"] + 1
        history = json.loads((out / "history.json").read_text())
        if "rng_state" in saved:
            torch.set_rng_state(saved["rng_state"].cpu())
            torch.cuda.set_rng_state(saved["cuda_rng_state"].cpu(), device)
    for epoch in range(start, epochs + 1):
        result = train_epoch(model, optimizer, None, infos, dates, epoch, args, device)
        history.append({"epoch": epoch, "train": result})
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch, "settings": settings, "rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state(device)}, out / "last.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(f"final refit epoch {epoch}/{epochs}: {result}", flush=True)
    torch.save({"model": model.state_dict(), "epoch": epochs, "blend": 1.0, "settings": settings}, out / "final.pt")


def predict(out: Path, cache: Path):
    os.environ["TFB_DAILY_CACHE"] = str(cache.resolve())
    torch.set_num_threads(8)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected GPU 9 only")
    device = torch.device("cuda:0")
    saved = torch.load(out / "final.pt", map_location=device, weights_only=False)
    infos = panel_infos()
    model = STAEformerResidual(total_nodes=sum(i.n_links for i in infos.values()), steps=24).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    scale = np.array([RESIDUAL_SPEED_SCALE, RESIDUAL_FLOW_LANE_SCALE], dtype=np.float32)
    diagnostics = []
    for split in ("validation", "private"):
        parts = []
        for panel, info in infos.items():
            target = pl.read_csv(paths.task1_targets(panel, split), columns=["timestamp", "station_id", "link_id", "mask_regime"])
            mapping = pl.DataFrame({"link_id": info.link_ids, "station_id": info.station_ids, "node": range(info.n_links)})
            target = target.join(mapping, on=["link_id", "station_id"], how="left", validate="m:1").with_columns(
                pl.col("timestamp").str.slice(0, 10).alias("date"),
                (pl.col("timestamp").str.slice(11, 2).cast(pl.Int32) * 12 +
                 pl.col("timestamp").str.slice(14, 2).cast(pl.Int32) // 5).alias("slot"))
            if target["node"].null_count() or target.select(pl.struct("timestamp", "link_id", "station_id").n_unique()).item() != target.height:
                raise ValueError(f"Invalid state target mapping: {panel}/{split}")
            for (day,), rows in target.partition_by("date", as_dict=True).items():
                masked = load_day(panel, split, day, info, masked=True)
                t, n = rows["slot"].to_numpy(), rows["node"].to_numpy()
                if np.isfinite(masked.speed[t, n]).any() or not masked.eligible[t, n].all():
                    raise ValueError(f"State target is visible/ineligible: {panel}/{day}")
                base, correction = predict_masked_day(model, masked, info, device, amp=True, batch_size=4)
                pred = base + float(saved["blend"]) * correction * scale
                pred[..., 0] = np.clip(pred[..., 0], 5.0, np.minimum(135.0, 1.05 * info.free_speeds)[None])
                pred[..., 1] = np.maximum(pred[..., 1], 0.0)
                v, q = pred[t, n, 0], pred[t, n, 1] * info.lanes[n]
                if not np.isfinite(v).all() or not np.isfinite(q).all():
                    raise ValueError(f"Nonfinite state predictions: {panel}/{day}")
                parts.append(rows.select("timestamp", "station_id", "link_id", "mask_regime").with_columns(
                    pl.lit(panel).alias("panel"), pl.Series("speed_kmh", v), pl.Series("flow_vph", q)))
            print(f"state inference {split}/{panel}: {target.height:,} official targets", flush=True)
        predicted = pl.concat(parts)
        predicted.write_parquet(out / f"predictions_{split}.parquet")
        for group in predicted.partition_by(["panel", "mask_regime"]):
            diagnostics.append({"split": split, "panel": group["panel"][0], "regime": group["mask_regime"][0],
                                "rows": group.height, "speed_min": group["speed_kmh"].min(),
                                "speed_max": group["speed_kmh"].max(), "flow_min": group["flow_vph"].min(),
                                "fraction_flow_below_50": group.filter(pl.col("flow_vph") < 50).height / group.height})
    (out / "prediction_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/staeformer_final_003"))
    parser.add_argument("--cache", type=Path, default=paths.CACHE_ROOT / "preprocessed_v1")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--mode", choices=("train", "predict", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.mode in ("train", "all"):
        train(args.out, args.cache, epochs=args.epochs, resume=args.resume)
    if args.mode in ("predict", "all"):
        predict(args.out, args.cache)


if __name__ == "__main__":
    main()
