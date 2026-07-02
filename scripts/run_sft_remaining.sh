#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
run_dir="experiment_runs/p2_sft_remaining"
mkdir -p "$run_dir" out checkpoints
echo "running" > "$run_dir/STATUS"
trap 'echo "failed" > "$run_dir/STATUS"' ERR

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  torchrun --standalone --nproc_per_node=4 \
  trainer/train_sft_vlm.py \
  --data_path dataset/sft_i2t_remaining_2303k.parquet \
  --val_data_path dataset/sft_i2t_val_1k.parquet \
  --from_weight sft_vlm_600k \
  --epochs 1 \
  --batch_size 4 \
  --accumulation_steps 4 \
  --num_workers 4 \
  --learning_rate 2e-6 \
  --max_seq_len 768 \
  --log_interval 20 \
  --eval_interval 1000 \
  --eval_batches 16 \
  --save_interval 4000 \
  --save_weight sft_vlm_full \
  --save_dir out \
  --checkpoint_dir checkpoints \
  2>&1 | tee -a "$run_dir/train.log"

sha256sum out/sft_vlm_full_768.pth checkpoints/sft_vlm_full_768_resume.pth > "$run_dir/SHA256SUMS"
echo "completed" > "$run_dir/STATUS"
