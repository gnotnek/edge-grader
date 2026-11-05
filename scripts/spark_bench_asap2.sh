#!/usr/bin/env bash
# Stage 5 on the lab's DGX Spark, inside the edge-grader:spark image: ASAP 2.0 on the fixed 600-essay
# sample, one version after the other: the untouched E2B and E4B, then their adapters. Each version
# gets its own llama-server with the lab's baseline flags. A version that fails is logged and skipped.
#
#   docker compose up -d bench-asap2
#   docker logs -f edge-grader-bench-asap2
#   N=5 OUT=data/smoke scripts/spark_bench_asap2.sh   # minutes: 5 essays each, nothing in results/
#
# The adapters were trained on SciEntsBank's three labels, not ASAP's 1-6 scores, so `unparsed` is
# the number to read first. Reason-first isn't here: it writes a reason first and ASAP stops at 8 tokens.
# Guide: docs/guides/spark-docker.md
set -uo pipefail

N=${N:-0}
OUT=${OUT:-results/spark}
mkdir -p data/logs data/models
LOG=data/logs/bench-asap2-$(date -u +%Y%m%d-%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "== $(date -u +%FT%TZ) ASAP 2.0: repo $(git rev-parse --short HEAD), llama.cpp $(cat /opt/llama.cpp/bin/COMMIT), n $N, out $OUT"

base_gguf() {  # e2b -> its GGUF in data/models, downloaded the first time
  local name repo
  case $1 in
    e2b) repo=unsloth/gemma-4-E2B-it-qat-GGUF; name=gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf ;;
    e4b) repo=unsloth/gemma-4-E4B-it-qat-GGUF; name=gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf ;;
  esac
  [ -f "data/models/$name" ] || uv run --frozen python -c \
    "from huggingface_hub import hf_hub_download; hf_hub_download('$repo', '$name', local_dir='data/models')" >&2
  echo "data/models/$name"
}

run() {  # run <e2b|e4b> <label suffix> [adapter gguf]
  local model=$1 suffix=$2 adapter=${3:-} label="$1-spark-gb10$2" server_log gguf server status=0
  echo "== $(date -u +%FT%TZ) $label ${adapter:-(no adapter)}"
  if [ -n "$adapter" ] && [ ! -f "$adapter" ]; then echo "SKIPPED $label: no $adapter"; return; fi
  gguf=$(base_gguf "$model")
  server_log=data/logs/server-asap2-$label.log
  llama-server -m "$gguf" -c 8192 --host 127.0.0.1 --port 8080 -ngl 99 -rea off --reasoning-budget -1 \
    --cache-ram 0 ${adapter:+--lora "$adapter"} > "$server_log" 2>&1 &
  server=$!
  for _ in $(seq 150); do
    curl -sf http://127.0.0.1:8080/health > /dev/null && break
    kill -0 $server 2>/dev/null || { echo "FAILED $label: server stopped"; tail -20 "$server_log"; return; }
    sleep 2
  done
  uv run --frozen edge-grader-asap run --base-url http://127.0.0.1:8080/v1 --label "$label" --n "$N" --out "$OUT" \
    || status=$?
  kill $server; wait $server 2>/dev/null
  [ $status = 0 ] || echo "FAILED $label: exit $status"
}

run e2b ""
run e4b ""
run e2b -ft data/adapters/e2b-ft.gguf
run e2b -ft-lf data/adapters/e2b-ft-lf.gguf
run e4b -ft data/adapters/e4b-ft.gguf
run e4b -ft-lf data/adapters/e4b-ft-lf.gguf
echo "== $(date -u +%FT%TZ) done, summaries in $OUT"
