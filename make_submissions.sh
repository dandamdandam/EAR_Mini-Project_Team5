#!/usr/bin/env bash
set -euo pipefail

# 대회 데이터셋 경로 설정
export TFB_DATA_ROOT="$HOME/.cache/kagglehub/competitions/2026-ieee-big-data-traffic-flow-bench/kaggle_public"

export FILE_PATH="$TFB_DATA_ROOT"
export CUDA_VISIBLE_DEVICES=0
RUN="easywoong/Traffic/runs/ensemble_006r"
CFG="configs/task1_upgrade_004r_006r.json"
mkdir -p outputs

[ -f easywoong/Traffic/cache/preprocessed_v1/SUCCESS.json ] || (cd easywoong/Traffic && python3 -m src.preprocess.run)
(cd easywoong/Traffic && python3 -m src.task1.upgrade --model imputeformer --stage all --config "$CFG")
(cd easywoong/Traffic && python3 -m src.task1.upgrade --model staeformer --stage all --config "$CFG")
(cd easywoong/Traffic && python3 -m src.task1.upgrade_ensemble --config "$CFG")
python3 - "$RUN" outputs/state_submission.csv <<'EOF'
import sys
import pandas as pd
run, out = sys.argv[1], sys.argv[2]
df = pd.concat([pd.read_parquet(f'{run}/predictions_validation.parquet'),
                pd.read_parquet(f'{run}/predictions_private.parquet')], ignore_index=True)
df['speed_kmh'] = df.speed_kmh.clip(lower=0)
df['flow_vph'] = df.flow_vph.clip(lower=0)
df.to_csv(out, index=False)
print(f'saved {out}: {len(df):,} rows')
EOF

[ -f task2/cache/D7_I10_E.npz ] || (cd task2 && python3 preprocess.py)
[ -f task2/cache/ode_models.pt ] || (cd task2 && python3 ode.py)
[ -f task2/outputs/sweep/pe_b128_s123.pt ] || (cd task2 && python3 train_maxdata.py \
    --run-name pe_b128_s123 --stat6 --pe-freqs 4 --batch 128 --epochs 150 \
    --on-cap 1500 --og-cap 750 --ng-cap 300)
cp task2/outputs/sweep/pe_b128_s123.pt task2/cache/t2_model.pt
(cd task2 && python3 submission.py --out ../outputs/queue_submission.csv)

python3 -m task4.task4_hf --out outputs/odme_submission.csv

python3 merge_submission.py
