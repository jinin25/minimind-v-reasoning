#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 OMP_NUM_THREADS=4
mkdir -p experiment_runs/grpo_g0 checkpoints/grpo_g0

torchrun --standalone --nproc_per_node=4 trainer/train_grpo_vlm.py \
  --data_dir dataset/RL_Innovator-VL \
  --init_ckpt out/cot_sft_rd02_768.pth \
  --save_dir out --save_weight grpo_g0_rd02 \
  --checkpoint_dir checkpoints/grpo_g0 \
  --epochs 1 --max_train_samples 1000 --max_val_samples 200 \
  --batch_size 1 --accumulation_steps 4 --num_workers 2 \
  --max_prompt_len 512 --max_gen_len 192 --num_generations 4 \
  --learning_rate 0.000001 --beta 0.02 \
  --format_weight 0.3 --tag_weight 0.1 --answer_weight 0.6 \
  --temperature 0.8 --top_p 0.9 --top_k 50 --repetition_penalty 1.05 \
  --log_interval 5 --save_interval 50 --val_steps 20 \
  --use_swanlab 2>&1 | tee experiment_runs/grpo_g0/train.log
