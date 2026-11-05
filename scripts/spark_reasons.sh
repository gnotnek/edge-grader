#!/usr/bin/env bash
# Teacher reasons on the lab's DGX Spark, inside the edge-grader:spark image: Qwen3-14B explains the
# gold grade of every train answer in $TRAIN_SETS, as notebooks/kaggle_reasons.ipynb does for
# SciEntsBank (same model file, prompt and flags). A stopped run resumes from its .part file.
#
#   docker compose up -d reasons-beetle-saf
#   N=5 OUT=data/smoke TRAIN_SETS="beetle saf" scripts/spark_reasons.sh   # 5 answers per set
#
# Guide: docs/guides/spark-docker.md
set -euo pipefail

SETS=${TRAIN_SETS:?TRAIN_SETS: the sets to write reasons for, e.g. "beetle saf"}
N=${N:-0}
OUT=${OUT:-results/finetune}
GGUF=data/models/Qwen3-14B-Q4_K_M.gguf
mkdir -p data/logs data/models
LOG=data/logs/reasons-${SETS// /-}-$(date -u +%Y%m%d-%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "== $(date -u +%FT%TZ) reasons for $SETS: repo $(git rev-parse --short HEAD), llama.cpp $(cat /opt/llama.cpp/bin/COMMIT), n $N, out $OUT"

[ -f "$GGUF" ] || uv run --frozen python -c \
  "from huggingface_hub import hf_hub_download; hf_hub_download('Qwen/Qwen3-14B-GGUF', 'Qwen3-14B-Q4_K_M.gguf', local_dir='data/models')"
SERVER_LOG=data/logs/server-teacher.log
llama-server -m "$GGUF" -c 8192 --host 127.0.0.1 --port 8080 -ngl 99 \
  -rea off --reasoning-budget -1 --cache-ram 0 > "$SERVER_LOG" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 150); do
  curl -sf http://127.0.0.1:8080/health > /dev/null && break
  kill -0 $SERVER 2>/dev/null || { tail -40 "$SERVER_LOG"; exit 1; }
  sleep 2
done
uv run --frozen edge-grader-reasons --base-url http://127.0.0.1:8080/v1 --sets $SETS --n "$N" --out "$OUT"
echo "== $(date -u +%FT%TZ) done"
