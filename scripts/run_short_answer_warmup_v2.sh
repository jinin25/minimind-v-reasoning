#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=experiment_runs/short_answer_warmup_v2
mkdir -p "$RUN"
echo running > "$RUN/STATUS"
trap 'code=$?; echo $code > "$RUN/exit"; if [ $code -ne 0 ]; then echo failed > "$RUN/STATUS"; fi' EXIT

while pgrep -f 'scripts/evaluate_generation.py.*short_answer_warmup_rd02' >/dev/null; do sleep 10; done
python scripts/repair_short_answer_5k.py | tee "$RUN/repair.log"
python scripts/audit_short_answer_5k.py --data dataset/short_answer_5k_sft_v2.parquet --output-dir "$RUN/audit" | tee "$RUN/audit.log"
python -m unittest tests.test_short_answer_warmup_v2 -v | tee "$RUN/tests.log"

export CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 OMP_NUM_THREADS=4
torchrun --standalone --nproc_per_node=4 trainer/train_sft_vlm.py \
  --data_path dataset/short_answer_5k_sft_v2.parquet --from_weight cot_sft_rd02 \
  --save_weight short_answer_warmup_v2 --checkpoint_dir checkpoints/short_answer_warmup_v2 \
  --epochs 3 --batch_size 5 --accumulation_steps 3 --max_seq_len 768 --learning_rate 0.000003 \
  --answer_loss_weight 4 --reasoning_drop_ratio 0 --cot_trim_ratio 0 \
  --log_interval 10 --eval_interval 0 --save_interval 334 --num_workers 4 --use_swanlab \
  2>&1 | tee "$RUN/train.log"

for epoch in 1 2 3; do
  mkdir -p "$RUN/epoch$epoch"
  CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_rl_by_task.py \
    --weight "short_answer_warmup_v2_epoch$epoch" --output "$RUN/epoch$epoch/rl_tasks.json" \
    > "$RUN/epoch$epoch/rl_tasks.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_generation.py \
    --weight "short_answer_warmup_v2_epoch$epoch" --data dataset/general_generation_eval.parquet \
    --output "$RUN/epoch$epoch/general.json" --reasoning off --max_new_tokens 128 \
    > "$RUN/epoch$epoch/general.log" 2>&1
done
python scripts/summarize_warmup_v2.py --run-dir "$RUN" | tee "$RUN/admission.log"
echo completed > "$RUN/STATUS"
