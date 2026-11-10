#!/usr/bin/env bash
# One fine-tuning run on the lab's DGX Spark, inside the edge-grader:spark image: train the adapter,
# convert it to GGUF, and grade 100 SciEntsBank test_ua answers with it through llama-server --lora.
# The same steps and settings as notebooks/kaggle_finetune.ipynb with SMOKE = False. Every file it
# writes stays in the project: data/ (adapters, models, Hugging Face cache, logs) and results/.
#
#   docker compose up -d finetune-e4b-lf     # runs: scripts/spark_finetune.sh e4b label-first
#   docker logs -f edge-grader-finetune-e4b-lf
#   SMOKE=1 scripts/spark_finetune.sh e4b label-first   # minutes: 20 steps, 10 answers, all output in data/smoke
#   TRAIN_SETS="beetle saf" scripts/spark_finetune.sh e4b label   # other sets' train splits: e4b-ft-beetle-saf.gguf
#   SEED=3 scripts/spark_finetune.sh e4b label                     # seed 3: data/adapters/e4b-s3, e4b-ft-s3.gguf
#   SOUP="0 1 2 3 4" scripts/spark_finetune.sh e4b label           # average those seeds instead of training: e4b-ft-soup.gguf
#
# Guide: docs/guides/spark-docker.md
set -euo pipefail

MODEL=${1:?model: e2b or e4b}
REPLY=${2:?reply: label, reason-first or label-first}
case $REPLY in
  label) TAG="" ;;
  reason-first) TAG="-rf" ;;
  label-first) TAG="-lf" ;;
  *) echo "reply must be label, reason-first or label-first, not $REPLY" >&2; exit 1 ;;
esac
case $MODEL in
  e2b) GGUF_REPO=unsloth/gemma-4-E2B-it-qat-GGUF; BASE_GGUF=gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf ;;
  e4b) GGUF_REPO=unsloth/gemma-4-E4B-it-qat-GGUF; BASE_GGUF=gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf ;;
  *) echo "model must be e2b or e4b, not $MODEL" >&2; exit 1 ;;
esac
SETS_ARGS=()
if [ -n "${TRAIN_SETS:-}" ]; then
  SETS_ARGS=(--train-sets $TRAIN_SETS)
  TAG=$TAG-${TRAIN_SETS// /-}
fi
# Adapter soup (spec 2026-11-10): SEED trains one ingredient, SOUP averages the named seeds' adapters.
SEED_ARGS=() SMOKE_SFX=""
[ "${SMOKE:-0}" = 1 ] && SMOKE_SFX=-smoke
[ -n "${SEED:-}" ] && [ -n "${SOUP:-}" ] && { echo "SEED trains, SOUP averages: set one" >&2; exit 1; }
[ -n "${SEED:-}" ] && { SEED_ARGS=(--seed "$SEED"); TAG=$TAG-s$SEED; }
INGREDIENTS=()
for s in ${SOUP:-}; do INGREDIENTS+=("data/adapters/$MODEL$TAG-s$s$SMOKE_SFX/adapter"); done
[ -n "${SOUP:-}" ] && TAG=$TAG-soup
RUN=$MODEL$TAG
TRAIN_ARGS=() N=100 RESULTS=results/finetune GRADED=results/spark
if [ "${SMOKE:-0}" = 1 ]; then  # the Day 1 check: does it train, convert and load? Nothing goes in results/
  RUN=$RUN-smoke TRAIN_ARGS=(--max-steps 20 --val-n 20) N=10 RESULTS=data/smoke GRADED=data/smoke
fi
ADAPTER=data/adapters/$RUN
FT_GGUF=data/adapters/${RUN/$MODEL/$MODEL-ft}.gguf  # e4b-ft-lf.gguf, as on Kaggle
LLAMA_CPP=/opt/llama.cpp
mkdir -p data/logs data/adapters data/models
LOG=data/logs/finetune-$RUN-$(date -u +%Y%m%d-%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

echo "== $(date -u +%FT%TZ) $MODEL $REPLY: repo $(git rev-parse --short HEAD), llama.cpp $(cat $LLAMA_CPP/bin/COMMIT), log $LOG"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

rm -rf "$ADAPTER"  # a re-run must not find the last run's adapter
if [ -n "${SOUP:-}" ]; then
  echo "== 1. average seeds $SOUP"
  uv run --frozen --group finetune python -m edge_grader.soup "$ADAPTER/adapter" "${INGREDIENTS[@]}" \
    --results "$RESULTS"
else
  echo "== 1. train"
  REPLY_ARGS=()
  REASONS=results/finetune/raw/reasons-qwen3-14b-q4_k_m${TRAIN_SETS:+-${TRAIN_SETS// /-}}.jsonl
  [ "$REPLY" = label ] || REPLY_ARGS=(--reply "$REPLY" --reasons "$REASONS")
  uv run --frozen --group finetune edge-grader-finetune --model "$MODEL" --out "$ADAPTER" --results "$RESULTS" \
    "${TRAIN_ARGS[@]}" "${REPLY_ARGS[@]}" "${SETS_ARGS[@]}" "${SEED_ARGS[@]}"
fi
test -f "$ADAPTER/adapter/adapter_model.safetensors"

echo "== 2. convert to GGUF"
# The converter's own venv with llama.cpp's requirements, as on Kaggle, and the base model's config
# files only (no weights, no shard index: it would make the converter look for the weights).
BASE_CONFIG=data/base-config-$MODEL
uv run --frozen python - "$MODEL" "$BASE_CONFIG" <<'EOF'
import sys
from huggingface_hub import snapshot_download
from edge_grader.finetune import BASES
repo, revision = BASES[sys.argv[1]]
snapshot_download(repo, revision=revision, allow_patterns=["*.json", "*.jinja"],
                  ignore_patterns=["*.index.json"], local_dir=sys.argv[2])
EOF
CONVERT_VENV=data/convert-venv-$RUN  # one per run, so two runs never rebuild the same venv
uv venv -q --clear "$CONVERT_VENV"
uv pip install -q --python "$CONVERT_VENV/bin/python" -r $LLAMA_CPP/requirements/requirements-convert_lora_to_gguf.txt
"$CONVERT_VENV/bin/python" $LLAMA_CPP/convert_lora_to_gguf.py "$ADAPTER/adapter" --base "$BASE_CONFIG" \
  --outfile "$FT_GGUF" --outtype f16
sha256sum "$FT_GGUF"

echo "== 3. grade $N test_ua answers with the adapter"
[ -f "data/models/$BASE_GGUF" ] || uv run --frozen python -c \
  "from huggingface_hub import hf_hub_download; hf_hub_download('$GGUF_REPO', '$BASE_GGUF', local_dir='data/models')"
SERVER_LOG=data/logs/server-$RUN.log
# The lab's baseline flags plus the adapter, on the container's own port: nothing else changes.
llama-server -m "data/models/$BASE_GGUF" -c 8192 --host 127.0.0.1 --port 8080 -ngl 99 \
  -rea off --reasoning-budget -1 --cache-ram 0 --lora "$FT_GGUF" > "$SERVER_LOG" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 150); do
  curl -sf http://127.0.0.1:8080/health > /dev/null && break
  kill -0 $SERVER 2>/dev/null || { tail -40 "$SERVER_LOG"; exit 1; }
  sleep 2
done
uv run --frozen edge-grader-baseline --base-url http://127.0.0.1:8080/v1 --label "$MODEL-spark-gb10-ft$TAG" \
  --split test_ua --n "$N" --out "$GRADED" --reply "$REPLY"
echo "== $(date -u +%FT%TZ) done: $FT_GGUF, summary in $RESULTS, graded run in $GRADED"
