"""Choose one global convex weight on January, then freeze it for submissions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from src.io import paths
from src.preprocess.run import sha256
from src.task1.train_staeformer import _summarize_scores
from src.task1.upgrade import CONFIG, write_json

KEYS = ["panel", "timestamp", "station_id", "link_id", "mask_regime"]
VALUES = ["speed_kmh", "flow_vph"]
TRUTH = ["truth_speed", "truth_flow_lane", "lanes"]


def align(a, b, with_truth):
    for frame in (a, b):
        if frame.select(pl.struct(KEYS).n_unique()).item() != len(frame):
            raise ValueError("Duplicate component prediction coordinates")
        if any(frame.null_count().row(0)):
            raise ValueError("Null component prediction")
    if len(a) != len(b) or a.join(b.select(KEYS), on=KEYS, how="anti").height:
        raise ValueError("Component prediction coordinates differ")
    columns = [*VALUES, *(TRUTH if with_truth else [])]
    joined = a.join(b.select(*KEYS, *columns), on=KEYS, how="left", validate="1:1", suffix="_other", maintain_order="left")
    if with_truth:
        for column in TRUTH:
            if joined.filter(pl.col(column) != pl.col(column + "_other")).height:
                raise ValueError(f"Component truth/lanes differ: {column}")
    return joined


def blend(joined, weight, with_truth=False):
    if not 0 < weight < 1:
        raise ValueError("Both independent models must contribute")
    result = joined.with_columns([
        (weight * pl.col(c) + (1 - weight) * pl.col(c + "_other")).alias(c) for c in VALUES
    ]).select(*KEYS, *VALUES, *(TRUTH if with_truth else []))
    if result.filter(pl.any_horizontal([~pl.col(c).is_finite() | (pl.col(c) < 0) for c in VALUES])).height:
        raise ValueError("Invalid ensemble predictions")
    return result


def score(frame):
    grouped = frame.group_by("panel", "mask_regime").agg(
        (pl.col("speed_kmh") - pl.col("truth_speed")).pow(2).sum().alias("v2"),
        (pl.col("flow_vph") / pl.col("lanes") - pl.col("truth_flow_lane")).pow(2).sum().alias("q2"),
        pl.len().alias("n"))
    return _summarize_scores({(r["panel"], r["mask_regime"]): {c: r[c] for c in ("v2", "q2", "n")}
                              for r in grouped.iter_rows(named=True)}, paths.PANELS)


def run(common):
    root = Path(common["ensemble_run"])
    root.mkdir(parents=True, exist_ok=True)
    components = [Path(common["models"][name]["run"]) for name in ("staeformer", "imputeformer")]
    files = [p / "predictions_selection.parquet" for p in components]
    hashes = {str(p): sha256(p) for p in files}
    selected_path = root / "selection_best.json"
    if selected_path.exists():
        selected = json.loads(selected_path.read_text())
        if selected["selection_prediction_hashes"] != hashes:
            raise ValueError("Selection inputs changed after the ensemble weight was frozen")
    else:
        joined = align(*(pl.read_parquet(f) for f in files), with_truth=True)
        scores = {str(w): score(blend(joined, w, True)) for w in common["ensemble_staeformer_weights"]}
        weight = max(common["ensemble_staeformer_weights"], key=lambda w: scores[str(w)]["s_state"])
        selected = {"staeformer_weight": weight, "imputeformer_weight": 1 - weight,
                    "selection_dates": common["selection_dates"], "scores": scores,
                    "selection_prediction_hashes": hashes,
                    "selection_policy": "one global weight selected using January only, before any new Kaggle upload"}
        write_json(selected_path, selected)
    weight = selected["staeformer_weight"]
    write_json(root / "config.json", {"staeformer_weight": weight, "imputeformer_weight": 1 - weight,
               "components": [str(p) for p in components], "experiment": common})
    component_hashes = {}
    component_hashes["checkpoints"] = {str(root / phase / filename): sha256(root / phase / filename)
        for root in components for phase, filename in (("local", "best.pt"), ("final", "final.pt"))}
    for name in ("selection", "february", "validation", "private"):
        files = [p / f"predictions_{name}.parquet" for p in components]
        component_hashes[name] = {str(p): sha256(p) for p in files}
        with_truth = name in ("selection", "february")
        joined = align(*(pl.read_parquet(p) for p in files), with_truth=with_truth)
        result = blend(joined, weight, with_truth)
        result.write_parquet(root / f"predictions_{name}.parquet")
        if with_truth:
            report = score(result)
            write_json(root / f"{name}_result.json", {"score": report, "staeformer_weight": weight,
                       "note": "February was already used in earlier work and is reported only; it does not select the weight."})
            print(f"ensemble {name}: S_state={report['s_state']:.6f}, STAEformer weight={weight}", flush=True)
        else:
            print(f"ensemble {name}: {len(result):,} targets", flush=True)
    write_json(root / "component_hashes.json", component_hashes)
    write_json(root / "SUCCESS.json", {"staeformer_weight": weight, "imputeformer_weight": 1 - weight})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()))


if __name__ == "__main__":
    main()
