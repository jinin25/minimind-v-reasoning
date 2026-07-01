#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
run_dir="experiment_runs/p1_pretrain"
mkdir -p "$run_dir" out checkpoints logs

echo "running" > "$run_dir/STATUS"
trap 'echo "failed" > "$run_dir/STATUS"' ERR
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  torchrun --standalone --nproc_per_node=4 \
  trainer/train_pretrain_vlm.py \
  --epochs 1 \
  --batch_size 8 \
  --num_workers 4 \
  --learning_rate 4e-4 \
  --max_seq_len 360 \
  --log_interval 50 \
  --save_interval 1000 \
  --save_weight pretrain_vlm \
  --save_dir out \
  --checkpoint_dir checkpoints \
  2>&1 | tee "$run_dir/train.log"

python scripts/evaluate_visual_ablation.py \
  --weight pretrain_vlm \
  --samples 256 \
  --output "$run_dir/visual_ablation.json" \
  2>&1 | tee "$run_dir/visual_ablation.log"

sha256sum out/pretrain_vlm_768.pth checkpoints/pretrain_vlm_768_resume.pth \
  > "$run_dir/SHA256SUMS"
echo "completed" > "$run_dir/STATUS"
