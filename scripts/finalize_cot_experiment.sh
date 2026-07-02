#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p experiment_runs/cot_eval_final

run_eval() {
  local weight="$1" data="$2" mode="$3" samples="$4" name="$5"
  CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_generation.py \
    --weight "$weight" --data "$data" --reasoning "$mode" --samples "$samples" \
    --max_new_tokens 128 --output "experiment_runs/cot_eval_final/$name.json" \
    > "experiment_runs/cot_eval_final/$name.log" 2>&1
}

for variant in rd0 rd02; do
  weight="cot_sft_${variant}"
  run_eval "$weight" dataset/general_generation_eval.parquet off -1 "${variant}_general_off"
  run_eval "$weight" dataset/cot_sft_val_1k.parquet on 200 "${variant}_cot_on"
  run_eval "$weight" dataset/cot_sft_val_1k.parquet off 200 "${variant}_cot_off"
done

python - <<'PY'
import glob, json, os
summary = {}
for path in sorted(glob.glob("experiment_runs/cot_eval_final/*.json")):
    result = json.load(open(path, encoding="utf-8"))
    summary[os.path.basename(path)] = {
        "weight": result["weight"], "reasoning": result["reasoning"],
        "overall": result["overall"], "by_category": result["by_category"],
    }
open("experiment_runs/cot_eval_final/summary.json", "w", encoding="utf-8").write(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
)
PY
