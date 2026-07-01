#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="/home/data/lilab06/wj/minimind-v-reasoning"
MODEL_DIR="$PROJECT_DIR/model/qwen2.5-vl-7b-instruct"
RUNTIME_DIR="$PROJECT_DIR/.runtime/vllm_multi"
mkdir -p "$RUNTIME_DIR"

source /home/data/lilab06/anaconda3/etc/profile.d/conda.sh
conda activate opengame

for gpu in 0 1 2 3; do
  ports=(8000 8011 8002 8003)
  port="${ports[$gpu]}"
  if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    echo "GPU $gpu / port $port 已就绪"
    continue
  fi
  CUDA_VISIBLE_DEVICES="$gpu" nohup setsid python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --served-model-name qwen2.5-vl-7b-instruct \
    --host 127.0.0.1 \
    --port "$port" \
    --tensor-parallel-size 1 \
    --dtype half \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt image=0,video=0 \
    --generation-config vllm \
    --trust-remote-code \
    </dev/null >"$RUNTIME_DIR/gpu${gpu}.log" 2>&1 &
  echo $! >"$RUNTIME_DIR/gpu${gpu}.launcher.pid"
  echo "已启动 GPU $gpu / port $port"
done

for gpu in 0 1 2 3; do
  ports=(8000 8011 8002 8003)
  port="${ports[$gpu]}"
  for _ in $(seq 1 180); do
    curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break
    sleep 2
  done
  curl -fsS "http://127.0.0.1:$port/health" >/dev/null
  echo "port $port health=ok"
done
