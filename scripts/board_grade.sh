#!/usr/bin/env bash
# Grading baseline on a lab board (Pi 5 or Jetson), with the same llama-server
# flags as agentic-edge's MMLU-Pro baseline, wrapped in its run_measured.sh so
# energy is recorded the same way.
#
#   scripts/board_grade.sh e4b                  # test_ua, all 540 answers, measured
#   scripts/board_grade.sh e2b test_ud 200      # 200 answers of unseen domains
#   MEASURE=0 scripts/board_grade.sh e4b        # no telemetry (e.g. a quick check,
#                                               #   or a job-queue "raw" job, which
#                                               #   already wraps in run_measured.sh)
#
# Env overrides: AE (agentic-edge/benchmark dir), LLAMA_SERVER, MODEL_DIR, PORT.
#
# Everything a run produces stays inside this repo, never in agentic-edge's
# folders, so it can be committed as the record of the run:
#   results/board/<model>-<board>-<split>-<stamp>/
#     baseline-*.json      grading summary (metrics, settings, server model path)
#     raw/*.jsonl          one line per graded answer
#     server.log           llama-server log (timestamped)
#     telemetry-*/         run_measured.sh output: telemetry.csv, requests.csv,
#                          meta.json, summary.json (energy, J/token), command.log
#
# What it does to the board, and undoes on exit:
#   Pi     stops the va-llm service (it holds the model and port 8080), runs its
#          own llama-server, then starts va-llm again. Same sudo -n rights the
#          lab's std_mmlupro.sh uses.
#   Jetson kills any "llama-server -m" and runs its own (as std_mmlupro_jetson.sh).
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
SELF="$HERE/scripts/$(basename "$0")"  # absolute: run_measured.sh cd's to its own dir
MODEL_KEY="${1:?usage: $0 e2b|e4b [split] [n]}"
SPLIT="${2:-test_ua}"
N="${3:-0}"
MEASURE="${MEASURE:-1}"
PORT="${PORT:-8080}"

if [ -f /etc/nv_tegra_release ]; then
  BOARD=jetson
  AE="${AE:-$HOME/research/agentic-edge/benchmark}"
  LLAMA_SERVER="${LLAMA_SERVER:-$HOME/build/llama.cpp/build/bin/llama-server}"
  MODEL_DIR="${MODEL_DIR:-$HOME/research/models}"
  NGL=99
elif grep -qi raspberry /proc/device-tree/model 2>/dev/null; then
  BOARD=pi5
  AE="${AE:-$HOME/Research/agentic-edge/benchmark}"
  LLAMA_SERVER="${LLAMA_SERVER:-/home/mitlab/llama.cpp/build/bin/llama-server}"
  MODEL_DIR="${MODEL_DIR:-/home/mitlab/models}"
  NGL=0
else
  echo "not a Pi 5 or Jetson; on other machines start llama-server yourself and run" >&2
  echo "  uv run edge-grader-baseline --label <name> --split $SPLIT --n $N" >&2
  exit 1
fi

case "$MODEL_KEY" in
  e2b) MODEL=$MODEL_DIR/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf ;;
  e4b) MODEL=$MODEL_DIR/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf ;;
  *) echo "unknown model key: $MODEL_KEY (e2b|e4b)" >&2; exit 1 ;;
esac
# The outer call picks the run directory and hands it to the inner call.
RUN_DIR="${EG_RUN_DIR:-$HERE/results/board/$MODEL_KEY-$BOARD-$SPLIT-$(date +%Y%m%d-%H%M%S)}"
SRVLOG=$RUN_DIR/server.log

# Fail before touching the board if anything is missing.
for f in "$LLAMA_SERVER" "$MODEL"; do
  [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }
done
command -v uv >/dev/null || { echo "uv not installed on this board (see README)" >&2; exit 1; }
[ "$MEASURE" = 0 ] || [ -x "$AE/run_measured.sh" ] || {
  echo "run_measured.sh not found in AE=$AE; set AE or use MEASURE=0" >&2; exit 1; }

# Outer call: re-run this script inside run_measured.sh. MEASURED_ROOT keeps its
# output in our run directory (not agentic-edge's measured/), and SRVLOG lets it
# read per-request timings from our server log instead of the journal.
if [ "$MEASURE" = 1 ] && [ -z "${EG_INNER:-}" ]; then
  mkdir -p "$RUN_DIR"
  exec env MEASURED_ROOT="$RUN_DIR" SRVLOG="$SRVLOG" "$AE/run_measured.sh" telemetry -- \
    env EG_INNER=1 EG_RUN_DIR="$RUN_DIR" MEASURE=1 "$SELF" "$MODEL_KEY" "$SPLIT" "$N"
fi

mkdir -p "$RUN_DIR"
restore() {
  pkill -f "llama-server -m $MODEL" 2>/dev/null
  if [ "$BOARD" = pi5 ]; then
    sudo -n systemctl start va-llm && echo "$(date) va-llm restarted"
  fi
}
trap restore EXIT

if [ "$BOARD" = pi5 ]; then
  sudo -n systemctl stop va-llm || { echo "cannot stop va-llm (sudo -n)" >&2; exit 1; }
else
  pkill -f "llama-server -m" 2>/dev/null
fi
sleep 2

# stamp.py timestamps each log line so parse_llama_log.py can read it.
STAMP="python3 $AE/stamp.py"
[ -f "$AE/stamp.py" ] || STAMP=cat

# The agentic-edge baseline flags (baselines.json "mmlupro-baseline"): thinking off.
echo "$(date) starting llama-server on $BOARD: $(basename "$MODEL")"
nohup sh -c "$LLAMA_SERVER -m $MODEL -c 8192 --host 127.0.0.1 --port $PORT -ngl $NGL \
  -rea off --reasoning-budget -1 --cache-ram 0 2>&1 | $STAMP" \
  > "$SRVLOG" 2>&1 < /dev/null &
for _ in $(seq 150); do curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && break; sleep 2; done
curl -sf "localhost:$PORT/health" >/dev/null || { echo "llama-server did not come up; see $SRVLOG" >&2; exit 1; }
ps -o args= -C llama-server | head -1

cd "$HERE" || exit 1
uv run edge-grader-baseline --base-url "http://127.0.0.1:$PORT/v1" \
  --label "$MODEL_KEY-$BOARD" --split "$SPLIT" --n "$N" --out "$RUN_DIR"
RC=$?
echo "$(date) grading finished rc=$RC; everything for this run is in $RUN_DIR"
echo "commit it: git add results/board && git commit -m 'Board run: $MODEL_KEY $BOARD $SPLIT'"
exit $RC
