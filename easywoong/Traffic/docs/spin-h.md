# SPIN-H Task 1 competitor

This run adapts the [official SPIN repository](https://github.com/Graph-Machine-Learning-Group/spin)
at commit `7349ba31da7306e7e96c13668a3f1f0a4df90902`. Its original code is MIT
licensed; the notice is in `src/task1/spin_h_LICENSE`. The corresponding paper is
[Marisca, Cini and Alippi, NeurIPS 2022](https://openreview.net/pdf?id=YXHoPO33rk).

## Architecture and data

`src/task1/spin_h_model.py` is a PyTorch-only port of the released
`SPINHierarchicalModel` and `HierarchicalTemporalGraphAttention` equations.
The upstream dependencies PyG, `torch_scatter`, and TSL are absent from the
`easypj` environment. The model retains the published SPIN-H setting:

| Setting | Value |
| --- | ---: |
| Observation hidden size | 32 |
| Hub hidden size | 128 |
| Hubs per sensor | 4 |
| Hierarchical layers | 5 |
| Late mask-conditioning layer (`eta`) | 3 |
| Temporal weighting | masked softmax for observation-to-hub and hub-to-observation |
| Spatial weighting | softmax across incoming sensor messages |
| Training loss | masked L1 on final and all four intermediate readouts |

Each input is `[batch, 24, panel sensor count, 2]` containing standardized
speed and flow **per lane**. Missing measurements and synthetically hidden
cells are zero after standardization and have a separate visibility mask.
The four temporal coordinates are day/week sine and cosine. A single model is
trained across all ten panels; sensor embeddings use unique global IDs.

The graph in `src/task1/spin_h_graph.py` uses only released static network
metadata. Consecutive `order_index` sensors are connected in travel order
(N/E increasing, S/W decreasing), crossing unsensed connector gaps. Explicit
sensor links listed in `incoming_link_ids` and `outgoing_link_ids` are added.
Each physical connection gets two directed messages, as the official SPIN
runner uses `force_symmetric=True`. The graph is fixed before training and
does not use traffic values or target labels.

## Evaluation protocol

- Training: 2030-06-01 through 2030-12-31 unmasked observations, with random
  target masks at competition rates 20%, 30%, and 50% on eligible cells. All
  sampled windows of one panel-day visit use one regime. A deterministic
  panel/date offset balances regimes across days; each revisit rotates to the
  next regime, so three visits expose that day to all three rates.
- Standardization: one mean and standard deviation per channel, fitted only
  on available June–December training observations across all panels.
- Sampling: a shuffled cycle visits all 2,140 panel-days before repeating.
  Each visit draws 4 or 8 stride-one 24-step windows from the day's 265
  possible windows, depending on panel size. At 300 batches per epoch, every
  training day is visited by the end of epoch 8. The default 40-epoch budget
  visits each day at least five times unless a later cycle is interrupted.
- Selection: complete January official Task 1 target template, with five
  interpolation/model blends. The best epoch and blend are chosen by the exact
  corridor-weighted `S_state` metric.
- Final test: complete February official target template, opened after
  checkpoint and blend selection. February unmasked values are read only after
  predictions from masked inputs have been produced.

The full run writes `runs/spin_h_001/{config,history,selection_best,test_result}.json`
and an ignored `best.pt`. The test result includes selected epoch/blend,
per-panel/regime RMSE and score, and a target-row coverage assertion. The
expected February target count in this release is 3,443,944 rows.

## Differences from the paper experiment

The paper used single-channel benchmark datasets, sensor-distance graphs,
20/50/80% synthetic training masks, and up to 300 epochs. This task uses two
TrafficFlowBench channels, the released freeway topology, its 20/30/50%
regimes, and January official masks to choose the checkpoint. We retain the
published core architecture but replace the PyG/TSL message-passing runtime
with equivalent dense temporal and indexed spatial operations in PyTorch.
An interpolation blend is selected only for the competition score; the
network itself predicts both channels directly.

## Commands

From `Traffic/` in `easypj`:

```bash
conda run -n easypj python -m src.task1.spin_h_train --cpu-smoke
CUDA_VISIBLE_DEVICES=9 conda run -n easypj python -m src.task1.spin_h_train --gpu-benchmark
CUDA_VISIBLE_DEVICES=9 conda run -n easypj python -m src.task1.spin_h_train
```

The benchmark and full run are for physical GPU 9 only. Use the benchmark to
measure actual memory and time before choosing a full training budget on the
shared server.
