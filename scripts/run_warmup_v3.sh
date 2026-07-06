#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."; RUN=experiment_runs/warmup_v3; mkdir -p "$RUN"; echo running > "$RUN/STATUS"
trap 'c=$?; echo $c > "$RUN/exit"; [ $c -eq 0 ] || echo failed > "$RUN/STATUS"' EXIT
if [ ! -f dataset/manifests/warmup_v3.json ]; then python scripts/prepare_warmup_v3.py | tee "$RUN/prepare.log"; fi
export CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 OMP_NUM_THREADS=4
torchrun --standalone --nproc_per_node=4 trainer/train_sft_vlm.py \
 --data_path dataset/warmup_v3_mixed.parquet --from_weight cot_sft_rd02 --save_weight warmup_v3 \
 --checkpoint_dir checkpoints/warmup_v3 --epochs 2 --batch_size 5 --accumulation_steps 3 --max_seq_len 768 \
 --learning_rate 0.000001 --answer_loss_weight 2 --reasoning_drop_ratio 0.2 --cot_trim_ratio 0 \
 --freeze_llm 3 --log_interval 20 --save_interval 1000 --num_workers 4 --use_swanlab 2>&1 | tee "$RUN/train.log"
weights=(cot_sft_rd02 warmup_v3_epoch1 warmup_v3_epoch2)
for i in 0 1 2; do CUDA_VISIBLE_DEVICES=$i python scripts/evaluate_warmup_v3_diagnostics.py --weight ${weights[$i]} --output "$RUN/${weights[$i]}_diagnostics.json" > "$RUN/${weights[$i]}_diagnostics.log" 2>&1 & done
wait
for e in 1 2; do CUDA_VISIBLE_DEVICES=3 python scripts/evaluate_generation.py --weight warmup_v3_epoch$e --data dataset/general_generation_eval.parquet --output "$RUN/epoch${e}_general.json" --reasoning off --max_new_tokens 128 > "$RUN/epoch${e}_general.log" 2>&1; done
python scripts/summarize_warmup_v3.py | tee "$RUN/summary.log"
echo completed > "$RUN/STATUS"
