#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="/home/data/lilab06/wj/minimind-v-reasoning"
cd "$PROJECT_DIR"
source /home/data/lilab06/anaconda3/etc/profile.d/conda.sh
conda activate opengame

exec python dataset/distill_multigpu.py \
  --input dataset/sft_i2t.parquet \
  --output dataset/sft_i2t_cot_distilled.parquet \
  --work-dir dataset/distill_multigpu \
  --endpoints http://127.0.0.1:8000/v1 http://127.0.0.1:8011/v1 http://127.0.0.1:8002/v1 http://127.0.0.1:8003/v1 \
  --candidate-count 360000 \
  --min-target 100000 \
  --max-target 300000 \
  --benchmark-size 1000 \
  --deadline-hours 9 \
  --concurrency-per-endpoint 6 \
  --part-size 500 \
  --max-tokens 64
