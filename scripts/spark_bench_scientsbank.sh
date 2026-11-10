#!/usr/bin/env bash
# SciEntsBank on the lab's DGX Spark, inside the edge-grader:spark image: every version on all three
# test splits, one after the other, each with its own llama-server and the lab's baseline flags:
# the adapters, trained on SciEntsBank or on Beetle and SAF only, with the label, label-first and
# reason-first replies, or with BASELINE=1 E2B and E4B untouched, or with EXAMPLES=1 the graded-examples
# runs (spec 2026-11-10: E4B untouched, + kNN examples, + random examples, adapter + kNN examples). A version with no adapter file yet
# is skipped, and a split already in $OUT isn't graded again, so running this again picks up what's new.
#
#   docker compose up -d bench-scientsbank
#   docker logs -f edge-grader-bench-scientsbank
#   N=5 OUT=data/smoke scripts/spark_bench_scientsbank.sh   # minutes: 5 answers per split, nothing in results/
#   SOUP="0 1 2 3 4" scripts/spark_bench_scientsbank.sh     # E4B untouched, those seeds' adapters and their soup
#
# Guide: docs/guides/spark-docker.md
set -uo pipefail

N=${N:-0}
OUT=${OUT:-results/spark}
mkdir -p data/logs data/models
LOG=data/logs/bench-scientsbank-$(date -u +%Y%m%d-%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "== $(date -u +%FT%TZ) SciEntsBank: repo $(git rev-parse --short HEAD), llama.cpp $(cat /opt/llama.cpp/bin/COMMIT), n $N, out $OUT"

run() {  # run <e2b|e4b> <label suffix> [adapter gguf] [reply] [examples file]
  local model=$1 suffix=$2 adapter=${3:-} reply=${4:-label} examples=${5:-} label="$1-spark-gb10$2" gguf server server_log split
  local stage=${examples:+examples}; stage=${stage:-baseline}
  echo "== $(date -u +%FT%TZ) $label ${adapter:-(no adapter)}"
  if [ -n "$adapter" ] && [ ! -f "$adapter" ]; then echo "SKIPPED $label: no $adapter"; return; fi
  todo=()
  for split in test_ua test_uq test_ud; do
    # a whole-split run counts as done; the 100-answer checks after training don't
    grep -qs '"n_requested": 0' "$OUT"/"$stage"-"$label"-"$split"-3way-*.json || todo+=("$split")
  done
  if [ ${#todo[@]} = 0 ]; then echo "already graded: $label"; return; fi
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
  for split in "${todo[@]}"; do
    uv run --frozen edge-grader-baseline --base-url http://127.0.0.1:8080/v1 --label "$label" --split "$split" \
      --n "$N" --out "$OUT" --reply "$reply" ${examples:+--examples "$examples" --k 4} || echo "FAILED $label $split"
  done
  kill $server; wait $server 2>/dev/null
}

if [ "${EXAMPLES:-0}" = 1 ]; then  # main pair first, the random control last
  knn=results/examples/examples-knn-bge-small-en-k4.json
  run e4b ""
  run e4b -ex4 "" label $knn
  run e4b -ft-ex4 data/adapters/e4b-ft.gguf label $knn
  run e4b -rand4 "" label results/examples/examples-random-k4.json
elif [ "${BASELINE:-0}" = 1 ]; then
  run e2b ""
  run e4b ""
elif [ -n "${SOUP:-}" ]; then  # adapter soup, spec 2026-11-10 §5.4
  run e4b ""
  for s in $SOUP; do run e4b -ft-s$s data/adapters/e4b-ft-s$s.gguf; done
  run e4b -ft-soup data/adapters/e4b-ft-soup.gguf
else
  for m in e2b e4b; do
    for trained in "" -beetle-saf; do  # on SciEntsBank, or on Beetle and SAF only
      run $m -ft$trained data/adapters/$m-ft$trained.gguf
      run $m -ft-lf$trained data/adapters/$m-ft-lf$trained.gguf label-first
      run $m -ft-rf$trained data/adapters/$m-ft-rf$trained.gguf reason-first
    done
  done
fi
echo "== $(date -u +%FT%TZ) done, summaries in $OUT"
