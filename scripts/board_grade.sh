#!/usr/bin/env bash
# The baseline on a lab board (Pi 5 or Jetson), with the same llama-server
# flags as agentic-edge's MMLU-Pro baseline, wrapped in its run_measured.sh so
# energy is recorded the same way. BENCH picks the benchmark (default: grading).
#
#   scripts/board_grade.sh e4b                  # SciEntsBank test_ua, all 540 answers, measured
#   scripts/board_grade.sh e2b test_ud 200      # 200 answers of unseen domains
#   BENCH=gsm scripts/board_grade.sh e2b gsm8k  # the 100 GSM8K originals
#   BENCH=gsm INSTANCES=1 scripts/board_grade.sh e2b symbolic   # first GSM-Symbolic variant set
#   BENCH=news scripts/board_grade.sh e4b       # LiveNewsBench, 200 (judge on Kaggle afterwards)
#   MEASURE=0 scripts/board_grade.sh e4b        # no telemetry (a quick check)
#   DRY_RUN=1 EG_BOARD=pi5 BENCH=gsm scripts/board_grade.sh e2b symbolic
#                                               # print the run folder and commands, touch nothing
#
# Run it directly on an idle board, never through agentic-edge's job queue:
# this project's runs must not land in the lab's measured/ or stdbench/. It
# refuses if started from the queue (or any run_measured.sh), and refuses to
# start while an agentic-edge run has the board.
#
# Env overrides: AE (agentic-edge/benchmark dir), LLAMA_SERVER, MODEL_DIR, PORT.
#
# Everything a run produces stays inside this repo, never in agentic-edge's
# folders, so it can be committed as the record of the run:
#   results/board/<model>-<board>-<split>-<stamp>/          SciEntsBank (stage-1 names)
#   results/board/<bench>-<model>-<board>-<set>-<stamp>/    gsm (set gsm8k | symbolic-i<N>), news (set test)
#     <runner summary>.json  baseline-*, gsm-* or news-*: metrics, settings, server model path
#     raw/*.jsonl          one line per question, including the model's raw reply
#     server.log           llama-server log (timestamped)
#     board-state.txt      CPU governor and clocks (and the Jetson's devfreq, nvpmodel) at the start
#     telemetry-*/         run_measured.sh output: telemetry.csv, requests.csv,
#                          meta.json, summary.json (energy, J/token), command.log
#
# What it does to the board, and undoes on exit:
#   Pi     stops the va-llm service (it holds the model and port 8080), runs its
#          own llama-server, then starts va-llm again if it was running before.
#          Same sudo -n rights the lab's std_mmlupro.sh uses.
#   Jetson kills any "llama-server -m" and runs its own (as std_mmlupro_jetson.sh).
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
SELF="$HERE/scripts/$(basename "$0")"  # absolute: run_measured.sh cd's to its own dir
MODEL_KEY="${1:?usage: [BENCH=scientsbank|gsm|news] [INSTANCES=n] $0 e2b|e4b [split|set] [n]}"
BENCH="${BENCH:-scientsbank}"
case "$BENCH" in
  scientsbank) SETS="test_ua test_uq test_ud" ;;
  gsm) SETS="gsm8k symbolic" ;;
  news) SETS="test" ;;
  *) echo "unknown BENCH: $BENCH (scientsbank|gsm|news)" >&2; exit 1 ;;
esac
SPLIT="${2:-${SETS%% *}}"
case " $SETS " in
  *" $SPLIT "*) ;;
  *) echo "unknown split/set for $BENCH: $SPLIT (one of: $SETS)" >&2; exit 1 ;;
esac
N="${3:-0}"
case "$N" in ''|*[!0-9]*) echo "n must be a whole number: $N" >&2; exit 1 ;; esac
INSTANCES="${INSTANCES:-10}"
case "$INSTANCES" in
  ''|*[!0-9]*|0) echo "INSTANCES must be a positive whole number: $INSTANCES" >&2; exit 1 ;;
esac
# Phase B1: THINKING=on serves with agentic-edge's thinking-on flags (grading only).
THINKING="${THINKING:-off}"
case "$THINKING" in
  off) ;;
  on) [ "$BENCH" = scientsbank ] || { echo "THINKING=on is for BENCH=scientsbank only" >&2; exit 1; } ;;
  *) echo "THINKING must be on or off: $THINKING" >&2; exit 1 ;;
esac
# Phase B2: VARIANT picks the engine setup (Jetson, grading only); RAG adds course material.
VARIANT="${VARIANT:-base}"
case "$VARIANT" in base|swafull|lg|lg-reuse) ;;
  *) echo "unknown VARIANT: $VARIANT (base|swafull|lg|lg-reuse)" >&2; exit 1 ;;
esac
RAG="${RAG:-}"
if [ "$VARIANT" != base ] || [ -n "$RAG" ]; then
  [ "$BENCH" = scientsbank ] || { echo "VARIANT/RAG are for BENCH=scientsbank only" >&2; exit 1; }
fi
MEASURE="${MEASURE:-1}"
PORT="${PORT:-8080}"
# deploy.sh installs uv here; non-interactive ssh shells do not have it on PATH.
export PATH="$HOME/.local/bin:$PATH"

# A dry run may pretend to be a board (EG_BOARD), so the tests can run anywhere.
DETECTED=""
if [ "${DRY_RUN:-0}" = 1 ] && [ -n "${EG_BOARD:-}" ]; then
  DETECTED=$EG_BOARD
elif [ -f /etc/nv_tegra_release ]; then
  DETECTED=jetson
elif grep -qi raspberry /proc/device-tree/model 2>/dev/null; then
  DETECTED=pi5
fi
case "$DETECTED" in
  jetson)
    BOARD=jetson
    AE="${AE:-$HOME/research/agentic-edge/benchmark}"
    LLAMA_SERVER="${LLAMA_SERVER:-$HOME/build/llama.cpp/build/bin/llama-server}"
    MODEL_DIR="${MODEL_DIR:-$HOME/research/models}"
    NGL=99
    THREADS=""
    # ~/research is an exFAT SSD (no symlinks); deploy.sh puts the venv here.
    export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/venvs/edge-grader}"
    ;;
  pi5)
    BOARD=pi5
    AE="${AE:-$HOME/Research/agentic-edge/benchmark}"
    LLAMA_SERVER="${LLAMA_SERVER:-/home/mitlab/llama.cpp/build/bin/llama-server}"
    MODEL_DIR="${MODEL_DIR:-/home/mitlab/models}"
    NGL=0
    # Every agentic-edge Pi run serves with -t 3 (va-llm, std_mmlupro.sh); without
    # it llama.cpp takes all 4 cores and the Pi's speed and energy stop matching.
    THREADS="-t 3"
    ;;
  *)
    echo "not a Pi 5 or Jetson; on other machines start llama-server yourself and run" >&2
    echo "  uv run edge-grader-baseline --label <name> --split $SPLIT --n $N" >&2
    exit 1
    ;;
esac

case "$VARIANT" in lg|lg-reuse|swafull)
  [ "$BOARD" = jetson ] || { echo "VARIANT=$VARIANT runs on the Jetson only" >&2; exit 1; } ;;
esac
# agentic-edge's S9 build (pin + ae.patch): the one with -reuse. With no flags it reproduces
# the plain little-gemma build byte for byte (agentic-edge's ae gate), so lg and lg-reuse share it.
LG_ENGINE="${LG_ENGINE:-$HOME/build/little-gemma-ae/build/run-cuda-i8}"

case "$MODEL_KEY" in
  e2b) MODEL=$MODEL_DIR/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf ;;
  e4b) MODEL=$MODEL_DIR/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf ;;
  *) echo "unknown model key: $MODEL_KEY (e2b|e4b)" >&2; exit 1 ;;
esac
# The outer call picks the run directory and hands it to the inner call.
# SciEntsBank keeps the stage-1 names; the other benchmarks lead with their name.
if [ "$BENCH" = scientsbank ]; then
  NAME="$MODEL_KEY-$BOARD-$SPLIT"
  [ "$THINKING" = on ] && NAME="think-$NAME"
  [ "$VARIANT" != base ] && NAME="$VARIANT-$NAME"
  [ -n "$RAG" ] && NAME="rag-$NAME"
elif [ "$SPLIT" = symbolic ]; then
  NAME="$BENCH-$MODEL_KEY-$BOARD-symbolic-i$INSTANCES"
else
  NAME="$BENCH-$MODEL_KEY-$BOARD-$SPLIT"
fi
RUN_DIR="${EG_RUN_DIR:-$HERE/results/board/$NAME-$(date +%Y%m%d-%H%M%S)}"
SRVLOG=$RUN_DIR/server.log

# The agentic-edge baseline flags (baselines.json "mmlupro-baseline"): thinking off,
# in the same order as the lab's command line. The same for every benchmark.
# THINKING=on swaps in "mmlupro-thinking-on"'s reasoning flags and nothing else.
# VARIANT=swafull adds --swa-full; lg and lg-reuse serve through agentic-edge's shim instead
# (its flags stand in for llama-server's: lg_openai_shim.py -h).
server_cmd() {
  case "$VARIANT" in
    lg|lg-reuse)
      echo "python3 $AE/lg_openai_shim.py --engine $LG_ENGINE -m $MODEL" \
           "--thinking off --think -1 --port $PORT$( [ "$VARIANT" = lg-reuse ] && echo ' --engine-flags=-reuse')" ;;
    *)
      echo "$LLAMA_SERVER -m $MODEL $THREADS -c 8192 --host 127.0.0.1 --port $PORT -ngl $NGL" \
           "$( [ "$THINKING" = on ] && echo '-rea on --reasoning-budget 320 --reasoning-format none' \
               || echo '-rea off --reasoning-budget -1' ) --cache-ram 0$( [ "$VARIANT" = swafull ] && echo ' --swa-full')" ;;
  esac
}

# The benchmark's runner. $1 is "echo" for a dry run and "" for the real one.
runner() {
  local run=$1 url="http://127.0.0.1:$PORT/v1" label="$MODEL_KEY-$BOARD"
  case "$BENCH" in
    scientsbank) $run uv run edge-grader-baseline --base-url "$url" --label "$label" \
                   --split "$SPLIT" --n "$N" --out "$RUN_DIR" $( [ "$THINKING" = on ] && echo --thinking ) \
                   $( [ -n "$RAG" ] && echo "--rag $RAG --k 3" ) ;;
    gsm) $run uv run edge-grader-gsm --base-url "$url" --label "$label" \
           --set "$SPLIT" --instances "$INSTANCES" --n "$N" --out "$RUN_DIR" ;;
    news) $run uv run edge-grader-news --base-url "$url" --label "$label" \
            --n "$N" --out "$RUN_DIR" ;;
  esac
}

# This project's runs are not agentic-edge's. A job in agentic-edge's queue is
# wrapped in the queue's own run_measured.sh, which writes telemetry into the
# lab's measured/ and needs a marker in its stdbench/, so the run lands in the
# lab's data and on its dashboard. The outer call refuses if anything above it
# is the queue runner or a run_measured.sh; the inner call is ours only if its
# telemetry root is inside this repo's results/.
under_lab_run() {
  local pid=$PPID cmd
  while [ -n "$pid" ] && [ "$pid" -gt 1 ]; do
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
    case "$cmd" in *queue_runner.py*|*run_measured.sh*) echo "$cmd"; return 0 ;; esac
    pid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
  done
  return 1
}
if [ -z "${EG_INNER:-}" ]; then
  if LAB=$(under_lab_run 2>/dev/null); then
    echo "refusing: started from agentic-edge's queue or run_measured.sh ($LAB)" >&2
    echo "edge-grader runs stay out of agentic-edge's data; run this script directly on the board" >&2
    exit 1
  fi
elif [ "$MEASURE" = 1 ]; then
  case "${MEASURED_ROOT:-}" in
    "$HERE"/results/*) ;;
    *) echo "refusing: telemetry would go to '${MEASURED_ROOT:-agentic-edge measured/}', not $HERE/results" >&2
       exit 1 ;;
  esac
fi
case "$RUN_DIR" in
  "$HERE"/results/*) ;;
  *) echo "refusing: run directory $RUN_DIR is outside $HERE/results" >&2; exit 1 ;;
esac

if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "run_dir: $RUN_DIR"
  echo "server: $(server_cmd)"
  printf 'runner: '
  runner echo
  exit 0
fi

# One job per board (agentic-edge AGENTS §4.9): a second run spoils both runs'
# telemetry, and on the Jetson this script kills any llama-server. So wait for
# the lab's run to finish rather than share the board with it. The Pi's va-llm
# server is the deployed one, which this script stops and restores. The lab's
# queue in turn waits while this run's run_measured.sh is alive.
board_busy() {
  local found="" deployed pid
  pgrep -f "[l]m_eval" >/dev/null && found="$found lm_eval"
  pgrep -f "[r]un_measured" >/dev/null && found="$found run_measured.sh"
  pgrep -f "[r]un-cuda-i8 -m" >/dev/null && found="$found little-gemma"
  deployed=$(systemctl show -p MainPID --value va-llm 2>/dev/null)
  for pid in $(pgrep -f "[l]lama-server"); do
    [ "$pid" = "$deployed" ] || { found="$found llama-server"; break; }
  done
  echo "${found# }"
}
if [ -z "${EG_INNER:-}" ]; then
  BUSY=$(board_busy)
  if [ -n "$BUSY" ]; then
    echo "refusing: the board is busy ($BUSY); wait for that run to finish" >&2
    exit 1
  fi
fi

# Fail before touching the board if anything is missing.
ENGINE_BIN=$LLAMA_SERVER
case "$VARIANT" in lg|lg-reuse) ENGINE_BIN=$LG_ENGINE
  [ -f "$AE/lg_openai_shim.py" ] || { echo "missing: $AE/lg_openai_shim.py" >&2; exit 1; } ;; esac
for f in "$ENGINE_BIN" "$MODEL"; do
  [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }
done
command -v uv >/dev/null || { echo "uv not installed on this board (run scripts/deploy.sh from the Mac)" >&2; exit 1; }
[ "$MEASURE" = 0 ] || [ -x "$AE/run_measured.sh" ] || {
  echo "run_measured.sh not found in AE=$AE; set AE or use MEASURE=0" >&2; exit 1; }

# Outer call: re-run this script inside run_measured.sh. MEASURED_ROOT keeps its
# output in our run directory (not agentic-edge's measured/), and SRVLOG lets it
# read per-request timings from our server log instead of the journal.
if [ "$MEASURE" = 1 ] && [ -z "${EG_INNER:-}" ]; then
  mkdir -p "$RUN_DIR"
  exec env MEASURED_ROOT="$RUN_DIR" SRVLOG="$SRVLOG" "$AE/run_measured.sh" telemetry -- \
    env EG_INNER=1 EG_RUN_DIR="$RUN_DIR" MEASURE=1 BENCH="$BENCH" INSTANCES="$INSTANCES" THINKING="$THINKING" \
    VARIANT="$VARIANT" RAG="$RAG" LG_ENGINE="$LG_ENGINE" \
    "$SELF" "$MODEL_KEY" "$SPLIT" "$N"
fi

mkdir -p "$RUN_DIR"
# Put the Pi's deployed server back the way it was found: restart it only if it was running.
VA_LLM_WAS=""
[ "$BOARD" = pi5 ] && VA_LLM_WAS=$(systemctl is-active va-llm 2>/dev/null)
stop_engines() {   # bracketed patterns never match this script or an ssh command line
  pkill -f "[l]lama-server -m $MODEL" 2>/dev/null
  pkill -f "[l]g_openai_shim.py" 2>/dev/null
  sleep 2
  pkill -f "[r]un-cuda-i8 -m" 2>/dev/null
}
restore() {
  stop_engines
  if [ "$BOARD" = pi5 ] && [ "$VA_LLM_WAS" = active ]; then
    sudo -n systemctl start va-llm && echo "$(date) va-llm restarted"
  fi
}
trap restore EXIT

# What the board was set to, so a run with locked clocks (agentic-edge RESULTS.md §7.7)
# can be told apart from one without. Read-only: nothing here changes a setting.
board_state() {
  local c=/sys/devices/system/cpu/cpu0/cpufreq d
  echo "time: $(date '+%Y-%m-%dT%H:%M:%S%z')"
  echo "kernel: $(uname -r)"
  [ -r "$c/scaling_governor" ] && echo "cpu0: governor $(cat $c/scaling_governor)" \
    "min $(cat $c/scaling_min_freq) max $(cat $c/scaling_max_freq)" \
    "cur $(cat $c/scaling_cur_freq) (kHz)"
  for d in /sys/class/devfreq/*; do
    [ -r "$d/min_freq" ] && echo "devfreq $(basename "$d"): min $(cat "$d/min_freq")" \
      "max $(cat "$d/max_freq") cur $(cat "$d/cur_freq") (Hz)"
  done
  command -v nvpmodel >/dev/null && echo "nvpmodel: $(nvpmodel -q 2>/dev/null | tr '\n' ' ')"
  [ -f "$HERE/DEPLOYED.json" ] && echo "deployed: $(cat "$HERE/DEPLOYED.json")"
  [ "$BOARD" = pi5 ] && echo "va-llm: ${VA_LLM_WAS:-unknown}"
  echo "variant: $VARIANT"
  case "$VARIANT" in
    lg|lg-reuse) echo "engine: $LG_ENGINE ($(head -1 "$(dirname "$LG_ENGINE")/BUILD_INFO" 2>/dev/null))" ;;
    *) echo "engine: $LLAMA_SERVER" ;;
  esac
  return 0
}
board_state > "$RUN_DIR/board-state.txt" 2>&1

if [ "$BOARD" = pi5 ]; then
  sudo -n systemctl stop va-llm || { echo "cannot stop va-llm (sudo -n)" >&2; exit 1; }
else
  pkill -f "[l]lama-server -m" 2>/dev/null; stop_engines
fi
sleep 2

# stamp.py timestamps each log line so parse_llama_log.py can read it.
STAMP="python3 $AE/stamp.py"
[ -f "$AE/stamp.py" ] || STAMP=cat

echo "$(date) starting llama-server on $BOARD: $(basename "$MODEL")"
nohup sh -c "$(server_cmd) 2>&1 | $STAMP" > "$SRVLOG" 2>&1 < /dev/null &
for _ in $(seq 150); do curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && break; sleep 2; done
curl -sf "localhost:$PORT/health" >/dev/null || { echo "llama-server did not come up; see $SRVLOG" >&2; exit 1; }
ps -o args= -C llama-server | head -1

cd "$HERE" || exit 1
runner ""
RC=$?
echo "$(date) $BENCH finished rc=$RC; everything for this run is in $RUN_DIR"
echo "commit it: git add results/board && git commit -m 'Board run: $BENCH $MODEL_KEY $BOARD $SPLIT'"
exit $RC
