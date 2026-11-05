#!/usr/bin/env bash
# SciEntsBank on the lab's DGX Spark, inside the edge-grader:spark image: every version on all three
# test splits, one after the other, each with its own llama-server and the lab's baseline flags:
# E2B and E4B untouched, their Oct 28 adapters, and the adapters trained on Beetle and SAF only.
# A version that fails or has no adapter file is logged and skipped.
#
#   docker compose up -d bench-scientsbank
#   docker logs -f edge-grader-bench-scientsbank
#   N=5 OUT=data/smoke scripts/spark_bench_scientsbank.sh   # minutes: 5 answers per split, nothing in results/
#
# Guide: docs/guides/spark-docker.md
set -uo pipefail

N=${N:-0}
OUT=${OUT:-results/spark}
mkdir -p data/logs data/models
LOG=data/logs/bench-scientsbank-$(date -u +%Y%m%d-%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "== $(date -u +%FT%TZ) SciEntsBank: repo $(git rev-parse --short HEAD), llama.cpp $(cat /opt/llama.cpp/bin/COMMIT), n $N, out $OUT"

run() {  # run <e2b|e4b> <label suffix> [adapter gguf]
  local model=$1 suffix=$2 adapter=${3:-} label="$1-spark-gb10$2" gguf server server_log split
  echo "== $(date -u +%FT%TZ) $label ${adapter:-(no adapter)}"
  if [ -n "$adapter" ] && [ ! -f "$adapter" ]; then echo "SKIPPED $label: no $adapter"; return; fi
  case $model in
    e2b) gguf=data/models/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf ;;
    e4b) gguf=data/models/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf ;;
  esac
  server_log=data/logs/server-scientsbank-$label.log
  llama-server -m "$gguf" -c 8192 --host 127.0.0.1 --port 8080 -ngl 99 -rea off --reasoning-budget -1 \
    --cache-ram 0 ${adapter:+--lora "$adapter"} > "$server_log" 2>&1 &
  server=$!
  for _ in $(seq 150); do
    curl -sf http://127.0.0.1:8080/health > /dev/null && break
    kill -0 $server 2>/dev/null || { echo "FAILED $label: server stopped"; tail -20 "$server_log"; return; }
    sleep 2
  done
  for split in test_ua test_uq test_ud; do
    uv run --frozen edge-grader-baseline --base-url http://127.0.0.1:8080/v1 --label "$label" --split "$split" \
      --n "$N" --out "$OUT" || echo "FAILED $label $split"
  done
  kill $server; wait $server 2>/dev/null
}

run e2b ""
run e4b ""
run e2b -ft data/adapters/e2b-ft.gguf
run e4b -ft data/adapters/e4b-ft.gguf
run e2b -ft-beetle-saf data/adapters/e2b-ft-beetle-saf.gguf
run e4b -ft-beetle-saf data/adapters/e4b-ft-beetle-saf.gguf
echo "== $(date -u +%FT%TZ) done, summaries in $OUT"
