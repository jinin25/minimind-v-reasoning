#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 OMP_NUM_THREADS=4

run_one() {
  local drop="$1" name="$2"
  mkdir -p "experiment_runs/$name" "checkpoints/$name"
  torchrun --standalone --nproc_per_node=4 trainer/train_sft_vlm.py \
    --data_path dataset/cot_sft_mix_25.parquet \
    --val_data_path dataset/cot_sft_val_1k.parquet \
    --from_weight sft_vlm_full --save_weight "$name" \
    --checkpoint_dir "checkpoints/$name" --epochs 2 --max_steps -1 \
    --batch_size 4 --accumulation_steps 4 --max_seq_len 1024 \
    --learning_rate 0.000002 --reasoning_drop_ratio "$drop" --cot_trim_ratio 0 \
    --log_interval 20 --eval_interval 1000 --eval_batches 16 --save_interval 2000 \
    --num_workers 4 --use_swanlab 2>&1 | tee "experiment_runs/$name/train.log"
}

run_one 0.0 cot_sft_rd0
run_one 0.2 cot_sft_rd02
