import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from preprocessing import OUTPUT_DIR, PANELS, RELEASE_ROOT, SUBMIT_SPLITS
from task4.task4_hf import load_tabpfn
from task4.task4_pseudo_label import PSEUDO_DIR, load_labels, zone_geometry
from task4.task4_pseudo_train import N_CONTEXT, N_EVAL_DAYS, path_features, s_od
from task4.nnls import format_records, load_corridor_network, load_split_inputs, write_records

ODME_RAW_PATH = OUTPUT_DIR / 'odme_submission_pseudo_raw.csv'


def run(release, panels, splits, label_dir, out_path, device, seed=0):
    model = load_tabpfn(device)
    rng = np.random.default_rng(seed)
    records, summary = [], []
    for panel in panels:
        network = load_corridor_network(release, panel)
        labels = load_labels(label_dir, panel)
        if network is None or labels is None:
            print(f'[{panel}] network or pseudo labels missing -> skipped.')
            continue
        t0 = time.time()
        paths = network['paths']
        (main_stream, _, origin_zone_idx, destination_zone_idx,
         zone_n, zone_start_pos, zone_end_pos) = zone_geometry(paths)
        link_flow, path_flow = labels['B'], labels['F']

        day_order = rng.permutation(len(link_flow))
        split_at = int(0.8 * len(day_order))
        train_days, eval_days = day_order[:split_at], day_order[split_at:][:N_EVAL_DAYS]
        mean_flow = path_flow[train_days].mean(axis=0)

        def features(day_link_flow):
            return path_features(day_link_flow, origin_zone_idx, destination_zone_idx,
                                 zone_n, zone_start_pos, zone_end_pos)

        def predict(x):
            return np.maximum(model.predict(x), 0.0)

        X = np.stack([features(day_link_flow) for day_link_flow in link_flow])
        X_train = X[train_days].reshape(-1, X.shape[-1])
        y_train = path_flow[train_days].reshape(-1)
        pick = rng.choice(len(y_train), size=min(N_CONTEXT, len(y_train)), replace=False)
        model.fit(X_train[pick], y_train[pick])

        score_model = np.mean([s_od(predict(X[day]), path_flow[day]) for day in eval_days])
        score_mean = np.mean([s_od(mean_flow, path_flow[day]) for day in eval_days])
        print(f'[{panel}] held-out S_od  TabPFN(raw) {score_model:.3f}  vs 날짜평균 {score_mean:.3f} | '
              f'{time.time() - t0:.0f}s', flush=True)
        summary.append({'panel': panel, 'S_od_tabpfn_raw': score_model, 'S_od_climatology': score_mean})

        for split in splits:
            inputs = load_split_inputs(release, panel, split, network['link_ids'], network['path_ids'])
            if inputs is None:
                continue
            count_by_link = dict(zip(network['link_ids'], inputs['counts']))
            measured_links = {link for link, m in zip(network['link_ids'], inputs['measured']) if m}
            split_link_flow = np.array([count_by_link[link] if link in measured_links else np.nan
                                        for link in main_stream])
            records.append(format_records(paths, panel, inputs['dep_time'],
                                          predict(features(split_link_flow))))

    if summary:
        print('\n' + pd.DataFrame(summary).set_index('panel').round(3).to_string())
    return write_records(records, out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits', nargs='+', default=SUBMIT_SPLITS)
    parser.add_argument('--panels', nargs='+', default=PANELS)
    parser.add_argument('--labels', type=Path, default=PSEUDO_DIR)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out', type=Path, default=ODME_RAW_PATH)
    args = parser.parse_args()

    run(RELEASE_ROOT, args.panels, args.splits, args.labels, args.out, args.device)
