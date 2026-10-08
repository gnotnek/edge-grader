# Phase 0: the baseline on both boards, implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure every missing baseline cell on the Pi 5 and the Jetson, accuracy and cost: SciEntsBank test_uq/test_ud, GSM8K + GSM-Symbolic and news. Check each against its Kaggle run.

**Architecture:**
- **`scripts/board_grade.sh`** learns a `BENCH` (`scientsbank` | `gsm` | `news`). It drives the existing runners (`edge-grader-baseline`, `-gsm`, `-news`) behind the same `llama-server` and telemetry. It records the board's clock state per run.
- **`edge-grader-board-check`** learns to read GSM and news run folders and pairs them with Kaggle on question ids. For news it reads the Kaggle judge's grades.
- **The Kaggle news-judge notebook** learns to judge news runs that sit in `results/board/`.
- **Then the runs:** Jetson in full, Pi with a trimmed GSM-Symbolic set. Then the board-check, and one baseline table.

**Tech Stack:** bash (board side, POSIX-ish, must run on bash 3.2 for the Mac tests), Python 3 via uv, pytest, scipy (already a dependency), Jupyter notebook JSON (Kaggle).

**Spec:** [`docs/superpowers/specs/2026-10-08-baseline-variations-design.md`](../specs/2026-10-08-baseline-variations-design.md) §4 (Phase 0). Read `CLAUDE.md` and `docs/research-outline.md` first.

## Global Constraints

**Tooling and serving:**
- Python via uv only: `uv sync`, `uv run pytest`, `uv run ruff check .`. Never pip.
- The Mac doesn't run Gemma. Board runs happen on the boards over ssh (`MITLAB-EDGE` = Pi 5, `MITLAB-JETSON` = Jetson).
- Server flags unchanged:
  - common: `-c 8192 --host 127.0.0.1 --port 8080 -rea off --reasoning-budget -1 --cache-ram 0`
  - Jetson: `-ngl 99`
  - Pi: `-ngl 0 -t 3`
- Board builds unchanged (Pi `661643e`, Jetson `a894dae`), recorded per run through the summary's `server` block.

**Separation from agentic-edge:**
- Board runs go through `scripts/board_grade.sh` directly, never through agentic-edge's job queue.
- Nothing is written into agentic-edge's `measured/`, `stdbench/` or `findings/`.

**Results:**
- Every run writes a summary JSON to `results/` with its settings. Never hand-edit a result file.
- Existing SciEntsBank folder names stay `<model>-<board>-<split>-<stamp>`.
- New folders are `<benchmark>-<model>-<board>-<set>-<stamp>`, with `<set>` = `symbolic-i<instances>` for GSM-Symbolic.

**Scope:**

| Board | SciEntsBank | GSM | News |
|---|---|---|---|
| Jetson | test_uq, test_ud | gsm8k (100) + symbolic `--instances 10` (1,000) | 200 |
| Pi 5 | test_uq, test_ud | gsm8k (100) + symbolic `--instances 1` (100) | 200 |

E2B and E4B on both boards.

**Judging and metrics:**
- News is judged by Qwen3-14B Q4_K_M with LiveNewsBench's SimpleQA prompt, on Kaggle (`notebooks/kaggle_news_judge.ipynb`). The judged file tag is `qwen3-14b-q4_k_m`.
- SciEntsBank metrics are accuracy and F1, not QWK.

**Gate:** nothing new runs on the boards until this phase's board-check shows board accuracy matching Kaggle.

**Commits:** end every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

These are the five things most likely to go wrong. Each line has a test in the task named.

1. **Pi's trimmed GSM-Symbolic (100 ids) against Kaggle's or the Jetson's 1,000.** Accuracy must be compared on the shared ids, not on the full runs. Task 2: `test_main_pairs_trimmed_symbolic_on_shared_ids`.
2. **A news run that isn't judged yet.** The board-check must still report its cost and mark accuracy as pending, not crash. Task 2: `test_news_accuracy_waits_for_the_judge`, and the e2e test.
3. **The Jetson's clocks left locked** (agentic-edge `findings/RESULTS.md` §7.7). Energy would then not be comparable with the 2026-09-30 test_ua runs. It must be recorded per run and flagged. Task 1 writes `board-state.txt`; Task 2: `test_read_board_state`; Task 4 checks it before any run.
4. **`deploy.sh` runs the test suite on the boards, where `results/` isn't copied.** A test that reads committed results must skip there, not fail. Task 2: the `skipif` on `test_stage1_jetson_run_reads_as_before`.
5. **A leftover `BENCH`/`INSTANCES` in the shell** (e.g. `export BENCH=news` from an earlier command) could silently run the wrong benchmark. The default with no `BENCH` must be SciEntsBank, and the run folder name must carry the benchmark. Task 1: `test_default_is_scientsbank_with_stage1_folder_name`; the run loops in Tasks 5–6 set both on every line.

---

## File map

| File | Change | Responsibility |
|---|---|---|
| `scripts/board_grade.sh` | Modify | `BENCH`/`INSTANCES`, argument checks, `DRY_RUN`, folder naming, runner dispatch, `board-state.txt` |
| `scripts/deploy.sh` | Modify (1 line) | Treat GSM and news runs as "busy" too |
| `tests/test_board_grade.py` | Create | Dry-run tests of the script |
| `src/edge_grader/board_check.py` | Modify | Read GSM and news runs; pair on shared ids; judged news; board state |
| `tests/test_board_check.py` | Modify | Tests for the above, and a regression test on a stage-1 run |
| `notebooks/kaggle_news_judge.ipynb` | Modify (cells `j00`, `j01`, `j11`, `j13`, `j15`, `j16`) | Judge board news runs; download only judged files |
| `results/README.md` | Modify | Document the new folder names and `board-state.txt` |
| `docs/results/baselines.md` | Modify (Task 7) | The phase-0 table |
| `docs/plan.md` | Modify (Task 7) | Tick phase 0 |

---

### Task 1: `board_grade.sh` runs any benchmark

**Files:**
- Modify: `scripts/board_grade.sh` (header lines 1–30; arguments lines 31–36; board detection lines 41–64; run dir line 72; the inner `exec` line; server start; grading call at the end)
- Modify: `scripts/deploy.sh` (the `busy=` line)
- Modify: `results/README.md` (the `board/` block)
- Test: `tests/test_board_grade.py`

**Interfaces:**
- Consumes: the CLIs `edge-grader-baseline --split --n --out --label --base-url`, `edge-grader-gsm --set --instances --n --out --label --base-url`, `edge-grader-news --n --out --label --base-url` (all exist).
- Produces:
  - **Environment variables:**
    - `BENCH` = `scientsbank` (default) | `gsm` | `news`
    - `INSTANCES` (default 10)
    - `DRY_RUN=1` with `EG_BOARD=pi5|jetson` prints and exits
  - **Positional arguments:** `$0 e2b|e4b [split|set] [n]`. Default split/set: `test_ua` / `gsm8k` / `test`.
  - **Run folders:**
    - `results/board/<model>-<board>-<split>-<stamp>/` (SciEntsBank)
    - `results/board/<bench>-<model>-<board>-<tag>-<stamp>/` (GSM, news), with `tag` = `gsm8k` | `symbolic-i<INSTANCES>` | `test`
  - **`board-state.txt`** in each new run folder, with a line `cpu0: governor <g> min <khz> max <khz> cur <khz> (kHz)`. Task 2 parses it.
  - **Dry-run output:** exactly three lines, `run_dir: <path>`, `server: <llama-server command>`, `runner: <uv run … command>`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_board_grade.py`:

```python
"""scripts/board_grade.sh in dry-run mode: argument checks, run-folder names, the runner it
would call, and the llama-server command (which must not change between benchmarks)."""

import os
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "board_grade.sh"
STAMP = r"\d{8}-\d{6}"


def dry(*args: str, board: str = "jetson", **env: str) -> subprocess.CompletedProcess:
    """Run the script with DRY_RUN=1 on a pretend board; BENCH etc. only if given."""
    e = {k: v for k, v in os.environ.items()
         if k not in {"BENCH", "INSTANCES", "EG_RUN_DIR", "EG_INNER", "MEASURE"}}
    e |= {"DRY_RUN": "1", "EG_BOARD": board, **env}
    return subprocess.run(["bash", str(SCRIPT), *args], env=e, capture_output=True, text=True)


def fields(proc: subprocess.CompletedProcess) -> dict[str, str]:
    assert proc.returncode == 0, proc.stderr
    return dict(line.split(": ", 1) for line in proc.stdout.strip().splitlines())


def test_default_is_scientsbank_with_stage1_folder_name():
    f = fields(dry("e2b"))
    assert re.fullmatch(rf"{REPO}/results/board/e2b-jetson-test_ua-{STAMP}", f["run_dir"])
    assert f["runner"] == (
        "uv run edge-grader-baseline --base-url http://127.0.0.1:8080/v1 --label e2b-jetson "
        f"--split test_ua --n 0 --out {f['run_dir']}"
    )


def test_scientsbank_split_and_n_pass_through():
    f = fields(dry("e4b", "test_ud", "200", board="pi5"))
    assert re.fullmatch(rf"{REPO}/results/board/e4b-pi5-test_ud-{STAMP}", f["run_dir"])
    assert "--split test_ud --n 200" in f["runner"]


def test_gsm_symbolic_trimmed_on_the_pi():
    f = fields(dry("e4b", "symbolic", board="pi5", BENCH="gsm", INSTANCES="1"))
    assert re.fullmatch(rf"{REPO}/results/board/gsm-e4b-pi5-symbolic-i1-{STAMP}", f["run_dir"])
    assert f["runner"] == (
        "uv run edge-grader-gsm --base-url http://127.0.0.1:8080/v1 --label e4b-pi5 "
        f"--set symbolic --instances 1 --n 0 --out {f['run_dir']}"
    )


def test_gsm_defaults_to_gsm8k_and_ten_instances():
    f = fields(dry("e2b", BENCH="gsm"))
    assert re.fullmatch(rf"{REPO}/results/board/gsm-e2b-jetson-gsm8k-{STAMP}", f["run_dir"])
    assert "--set gsm8k --instances 10" in f["runner"]


def test_news():
    f = fields(dry("e2b", BENCH="news"))
    assert re.fullmatch(rf"{REPO}/results/board/news-e2b-jetson-test-{STAMP}", f["run_dir"])
    assert f["runner"] == (
        "uv run edge-grader-news --base-url http://127.0.0.1:8080/v1 --label e2b-jetson "
        f"--n 0 --out {f['run_dir']}"
    )


def test_server_flags_are_the_same_for_every_benchmark():
    for board, extra in (("jetson", "-ngl 99"), ("pi5", "-t 3")):
        servers = {fields(dry("e2b", board=board, BENCH=b))["server"]
                   for b in ("scientsbank", "gsm", "news")}
        assert len(servers) == 1
        (server,) = servers
        assert extra in server
        assert server.endswith("-c 8192 --host 127.0.0.1 --port 8080 -ngl "
                               f"{'99' if board == 'jetson' else '0'} "
                               "-rea off --reasoning-budget -1 --cache-ram 0")


def test_rejects_bad_arguments():
    cases = [
        ({"BENCH": "mmlu"}, ("e2b",), "unknown BENCH"),
        ({"BENCH": "gsm"}, ("e2b", "test_ua"), "unknown split/set"),
        ({}, ("e2b", "symbolic"), "unknown split/set"),
        ({"BENCH": "gsm", "INSTANCES": "0"}, ("e2b", "symbolic"), "INSTANCES"),
        ({"BENCH": "gsm", "INSTANCES": "x"}, ("e2b", "symbolic"), "INSTANCES"),
        ({}, ("e2b", "test_ua", "abc"), "n must be"),
        ({}, ("e9b",), "unknown model key"),
        ({"EG_RUN_DIR": "/tmp/elsewhere"}, ("e2b",), "outside"),
    ]
    for env, args, message in cases:
        p = dry(*args, **env)
        assert p.returncode != 0, (env, args)
        assert message in p.stderr, (env, args, p.stderr)
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_board_grade.py -v`
Expected: FAIL. On the Mac the script exits with "not a Pi 5 or Jetson" because `EG_BOARD`/`DRY_RUN` aren't supported yet, so every `fields()` assertion fails on `returncode`.

- [ ] **Step 3: Implement the arguments, board override and dry run**

In `scripts/board_grade.sh`, replace the header's usage lines (2–9) with:

```bash
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
```

Update the folder layout comment in the header to:

```bash
#   results/board/<model>-<board>-<split>-<stamp>/          SciEntsBank (stage-1 names)
#   results/board/<bench>-<model>-<board>-<set>-<stamp>/    gsm (set gsm8k | symbolic-i<N>), news (set test)
#     <runner summary>.json  baseline-*, gsm-* or news-*: metrics, settings, server model path
#     raw/*.jsonl          one line per question, including the model's raw reply
#     server.log           llama-server log (timestamped)
#     board-state.txt      CPU governor and clocks (and the Jetson's devfreq, nvpmodel) at the start
#     telemetry-*/         run_measured.sh output: telemetry.csv, requests.csv,
#                          meta.json, summary.json (energy, J/token), command.log
```

Replace lines 31–33 (`MODEL_KEY=` … `N=`) with:

```bash
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
```

Replace the board detection `if [ -f /etc/nv_tegra_release ]; then … else … fi` block (lines 41–64) with this. The settings inside each branch are copied unchanged:

```bash
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
```

Replace the `RUN_DIR=` line (line 72) with:

```bash
# The outer call picks the run directory and hands it to the inner call.
# SciEntsBank keeps the stage-1 names; the other benchmarks lead with their name.
if [ "$BENCH" = scientsbank ]; then
  NAME="$MODEL_KEY-$BOARD-$SPLIT"
elif [ "$SPLIT" = symbolic ]; then
  NAME="$BENCH-$MODEL_KEY-$BOARD-symbolic-i$INSTANCES"
else
  NAME="$BENCH-$MODEL_KEY-$BOARD-$SPLIT"
fi
RUN_DIR="${EG_RUN_DIR:-$HERE/results/board/$NAME-$(date +%Y%m%d-%H%M%S)}"
```

Right after that, before the `under_lab_run()` function, add the two command builders:

```bash
# The agentic-edge baseline flags (baselines.json "mmlupro-baseline"): thinking off,
# in the same order as the lab's command line. The same for every benchmark.
server_cmd() {
  echo "$LLAMA_SERVER -m $MODEL $THREADS -c 8192 --host 127.0.0.1 --port $PORT -ngl $NGL" \
       "-rea off --reasoning-budget -1 --cache-ram 0"
}

# The benchmark's runner. $1 is "echo" for a dry run and "" for the real one.
runner() {
  local run=$1 url="http://127.0.0.1:$PORT/v1" label="$MODEL_KEY-$BOARD"
  case "$BENCH" in
    scientsbank) $run uv run edge-grader-baseline --base-url "$url" --label "$label" \
                   --split "$SPLIT" --n "$N" --out "$RUN_DIR" ;;
    gsm) $run uv run edge-grader-gsm --base-url "$url" --label "$label" \
           --set "$SPLIT" --instances "$INSTANCES" --n "$N" --out "$RUN_DIR" ;;
    news) $run uv run edge-grader-news --base-url "$url" --label "$label" \
            --n "$N" --out "$RUN_DIR" ;;
  esac
}
```

**About `THREADS`:** when `THREADS` is empty (Jetson), `server_cmd` prints a double space after the model path. The test compares by `endswith`, and `sh -c` splits words, so the process gets the same argv as before.

Note that the Pi line is `-ngl 0` **after** `-t 3`. The test's `endswith` covers everything from `-c 8192` on, and `extra in server` covers `-t 3`.

Directly after the existing `case "$RUN_DIR" in "$HERE"/results/*) ;; … esac` guard, add:

```bash
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "run_dir: $RUN_DIR"
  echo "server: $(server_cmd)"
  printf 'runner: '
  runner echo
  exit 0
fi
```

- [ ] **Step 4: Use the builders in the real run, pass BENCH/INSTANCES inward, record board state**

**Inner call.** Replace the inner `exec` line with:

```bash
  exec env MEASURED_ROOT="$RUN_DIR" SRVLOG="$SRVLOG" "$AE/run_measured.sh" telemetry -- \
    env EG_INNER=1 EG_RUN_DIR="$RUN_DIR" MEASURE=1 BENCH="$BENCH" INSTANCES="$INSTANCES" \
    "$SELF" "$MODEL_KEY" "$SPLIT" "$N"
```

**Board state.** After `trap restore EXIT`, before stopping va-llm, add:

```bash
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
  return 0
}
board_state > "$RUN_DIR/board-state.txt" 2>&1
```

**Server start.** Replace the `# The agentic-edge baseline flags …` comment and the `nohup sh -c "…"` command (keep the `echo "$(date) starting llama-server …"` line above it) with:

```bash
nohup sh -c "$(server_cmd) 2>&1 | $STAMP" > "$SRVLOG" 2>&1 < /dev/null &
```

**Runner call.** Replace the final `uv run edge-grader-baseline …` call (two lines) with `runner ""`.

**Commit hint.** Change the hint to:

```bash
echo "commit it: git add results/board && git commit -m 'Board run: $BENCH $MODEL_KEY $BOARD $SPLIT'"
```

- [ ] **Step 5: deploy.sh counts GSM and news runs as busy**

In `scripts/deploy.sh`, change

```bash
  busy=$(ssh "$H" 'pgrep -fa "[b]oard_grade|[e]dge-grader-baseline" | cut -c1-100')
```

to

```bash
  busy=$(ssh "$H" 'pgrep -fa "[b]oard_grade|[e]dge-grader-(baseline|gsm|news)" | cut -c1-100')
```

- [ ] **Step 6: Run the tests and lint**

Run: `uv run pytest tests/test_board_grade.py -v && uv run pytest -q && uv run ruff check .`
Expected: the 7 new tests PASS. The full suite passes (95 before this task + 7). Ruff is clean. Also run `bash -n scripts/board_grade.sh scripts/deploy.sh` (no output means the syntax is OK).

- [ ] **Step 7: Document the new folders**

In `results/README.md`, replace the `board/` block of the tree with:

```
  board/       runs on the Pi 5 / Jetson (from scripts/board_grade.sh), one folder per run:
    <model>-<board>-<split>-<stamp>/          SciEntsBank grading (stage 1 and phase 0)
    <bench>-<model>-<board>-<set>-<stamp>/    phase 0: gsm (set gsm8k | symbolic-i<variants>), news (set test)
      baseline-*.json | gsm-*.json | news-*.json   the runner's summary: metrics, settings,
                        server model path and build, token totals
      raw/*.jsonl       one line per question, including the model's raw reply
      news-*.judged-qwen3-14b-q4_k_m.json, raw/*.judged-*.jsonl   news only: the Kaggle judge's grades
      server.log        llama-server log
      board-state.txt   CPU governor and clocks (Jetson: devfreq, nvpmodel) at the start of the run
      telemetry-*/      agentic-edge's run_measured.sh output: telemetry.csv, requests.csv,
                        meta.json, summary.json (energy, J/token), command.log
```

- [ ] **Step 8: Commit**

```bash
git add scripts/board_grade.sh scripts/deploy.sh tests/test_board_grade.py results/README.md
git commit -m "board_grade.sh: BENCH=gsm|news, dry run, board state per run

Phase 0 runs GSM8K/GSM-Symbolic and news on the boards with the same server
flags and telemetry as grading. SciEntsBank folder names are unchanged.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: the board-check reads GSM and news runs

**Files:**
- Modify: `src/edge_grader/board_check.py`
- Test: `tests/test_board_check.py`

**Interfaces:**
- Consumes:
  - Task 1's run folders and `board-state.txt`
  - the runners' summaries:
    - grading: no `benchmark` key, `split`, `metrics.{n,accuracy,macro_f1,unparsed,errors,median_latency_s}`
    - gsm: `benchmark: "gsm"`, `set`, `generation.instances` (symbolic only), `metrics.{n,accuracy,errors,median_latency_s}`
    - news: `benchmark: "news"`, `metrics.{n,errors,median_latency_s}`, no accuracy
  - raw rows:
    - grading: `id, gold, pred, prompt_tokens`
    - gsm: `id, flexible, correct, prompt_tokens`
    - news: `id, pred, prompt_tokens`
  - the judge's `<stem>.judged-<tag>.json` (`metrics.accuracy`) and `raw/<stem>.judged-<tag>.jsonl` (`id, grade`)
- Produces (module `edge_grader.board_check`):
  - `JUDGE = "qwen3-14b-q4_k_m"`
  - `find_summary(run_dir: Path) -> Path`
  - `benchmark(summary: dict) -> str`
  - `split_of(summary: dict) -> str`
  - `outcomes(summary_path: Path, judge: str = JUDGE) -> dict[str, dict] | None`: `{id: {"answer": …, "right": bool}}`, or None for an unjudged news run
  - `compare(a, b) -> dict`: now takes **outcome dicts**. Keys: `n, same_grade, first_accuracy, second_accuracy, mcnemar`
  - `kaggle_run(directory: str, bench: str, model: str, split: str, n: int) -> Path | None`
  - `read_board_state(path: Path) -> dict | None`
  - `read_run(run_dir: Path, judge: str = JUDGE) -> dict`, with new keys `benchmark`, `instances`, `board_state`; `accuracy` may be None (news)
  - **CLI:** `--judge` (default `JUDGE`)
  - **Output file:** `vs_kaggle` has `kaggle_accuracy` (Kaggle **on the shared ids**) and `board_accuracy_paired`. For a stage-1 run the shared ids are all of them, so `kaggle_accuracy` keeps its old value.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_board_check.py`, and extend its import line to:

```python
from edge_grader.board_check import (
    compare,
    find_summary,
    kaggle_run,
    main,
    mcnemar,
    outcomes,
    parse_server_log,
    read_board_state,
    read_run,
    threads,
    window_energy,
)
```

Add at the top, after `import pytest`:

```python
import json
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
T0 = "2026-10-10T10:00:00+08:00"


def epoch(offset_s: float) -> float:
    return datetime.fromisoformat(T0).timestamp() + offset_s


def stamp(offset_s: int) -> str:
    return f"2026-10-10T10:00:{offset_s:02d}+0800"


def make_run(root: Path, name: str, summary: dict, rows: list[dict],
             judged: list[dict] | None = None, state: str | None = None) -> Path:
    """A board run folder as board_grade.sh leaves it: one request per row, each 1 s long
    (500 ms prompt + 500 ms decode), ending 2 s apart from t=2; 5 W throughout."""
    run = root / name
    (run / "raw").mkdir(parents=True)
    stem = f"{summary['_stem']}"
    body = {k: v for k, v in summary.items() if k != "_stem"}
    (run / f"{stem}.json").write_text(json.dumps(body))
    (run / "raw" / f"{stem}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    log = []
    for i, r in enumerate(rows):
        end = stamp(2 + 2 * i)
        log.append(f"{end} b llama-server[0]: I slot print_timing: id  0 | task {i} | "
                   f"prompt eval time =     500.00 ms / {r.get('prompt_tokens', 50):5d} tokens")
        log.append(f"{end} b llama-server[0]: I slot print_timing: id  0 | task {i} | "
                   f"       eval time =     500.00 ms /    10 tokens")
    (run / "server.log").write_text("\n".join(log) + "\n")
    tele = run / "edge-grader-x"
    tele.mkdir()
    (tele / "meta.json").write_text(json.dumps({"server_args_after": "llama-server -m m.gguf -t 3"}))
    (tele / "summary.json").write_text(json.dumps({"power": {"idle_w": 2.0}}))
    lines = ["ts_epoch,power_w"] + [f"{epoch(s)},5.0" for s in range(0, 2 * len(rows) + 3)]
    (tele / "telemetry.csv").write_text("\n".join(lines) + "\n")
    if judged is not None:
        acc = sum(g["grade"] == "correct" for g in judged) / len(judged)
        (run / f"{stem}.judged-qwen3-14b-q4_k_m.json").write_text(
            json.dumps({"metrics": {"n": len(judged), "accuracy": acc}}))
        (run / "raw" / f"{stem}.judged-qwen3-14b-q4_k_m.jsonl").write_text(
            "".join(json.dumps(g) + "\n" for g in judged))
    if state is not None:
        (run / "board-state.txt").write_text(state)
    return run


def gsm_summary(stem: str, label: str, set_: str, n: int, acc: float, instances=None) -> dict:
    gen = {"max_tokens": 1024} | ({"instances": instances} if instances else {})
    return {"_stem": stem, "stage": "topics", "benchmark": "gsm", "set": set_, "label": label,
            "generation": gen,
            "metrics": {"n": n, "accuracy": acc, "errors": 0, "median_latency_s": 1.0}}


def gsm_rows(results: dict[str, bool]) -> list[dict]:
    return [{"id": i, "flexible": "1" if ok else "2", "correct": ok,
             "prompt_tokens": 50 + 10 * k} for k, (i, ok) in enumerate(results.items())]


def write_kaggle(kdir: Path, stem: str, summary: dict, rows: list[dict]) -> None:
    (kdir / "raw").mkdir(parents=True, exist_ok=True)
    (kdir / f"{stem}.json").write_text(json.dumps(summary))
    (kdir / "raw" / f"{stem}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_read_run_gsm(tmp_path):
    rows = gsm_rows({"1/0": True, "2/0": False})
    rows.append({"id": "3/0", "error": "APIError()", "flexible": "[invalid]", "correct": False})
    run = make_run(tmp_path, "gsm-e2b-jetson-symbolic-i10-20261010-100000",
                   gsm_summary("gsm-symbolic-e2b-jetson-20261010-020000", "e2b-jetson",
                               "symbolic", 3, 0.3333, instances=10), rows)
    r = read_run(run)
    assert (r["benchmark"], r["model"], r["board"], r["split"], r["instances"]) == (
        "gsm", "e2b", "jetson", "symbolic", 10)
    assert r["accuracy"] == 0.3333 and r["macro_f1"] is None
    assert r["mean_prompt_tokens_sent"] == 55.0  # the errored row has no prompt_tokens
    # window: first request starts at t=1, last ends at t=6 -> 5 s at 5 W = 25 J over 3 questions
    assert r["grading_window_s"] == 5.0
    assert r["j_per_answer"] == pytest.approx(25 / 3, abs=0.01)
    assert r["board_state"] is None


def test_news_accuracy_waits_for_the_judge(tmp_path):
    summary = {"_stem": "news-baseline-e2b-pi5-20261010-020000", "stage": "baseline",
               "benchmark": "news", "label": "e2b-pi5",
               "metrics": {"n": 2, "answered": 1, "errors": 0, "median_latency_s": 1.0}}
    rows = [{"id": "news-000", "pred": None, "prompt_tokens": 60},
            {"id": "news-001", "pred": "Paris", "prompt_tokens": 70}]
    run = make_run(tmp_path, "news-e2b-pi5-test-20261010-100000", summary, rows)
    r = read_run(run)
    assert r["accuracy"] is None and r["split"] == "test"
    assert outcomes(find_summary(run)) is None

    judged = [{"id": "news-000", "grade": "not_attempted"}, {"id": "news-001", "grade": "correct"}]
    run2 = make_run(tmp_path, "news-e2b-pi5-test-20261010-110000", summary, rows, judged=judged)
    assert read_run(run2)["accuracy"] == 0.5
    assert outcomes(find_summary(run2)) == {
        "news-000": {"answer": "not_attempted", "right": False},
        "news-001": {"answer": "correct", "right": True},
    }


def test_compare_pairs_on_shared_ids():
    a = {"x": {"answer": 1, "right": True}, "y": {"answer": 2, "right": False}}
    b = {"x": {"answer": 1, "right": True}, "y": {"answer": 3, "right": True},
         "z": {"answer": 4, "right": False}}
    c = compare(a, b)
    assert (c["n"], c["same_grade"], c["first_accuracy"], c["second_accuracy"]) == (2, 1, 0.5, 1.0)
    assert (c["mcnemar"]["only_first_right"], c["mcnemar"]["only_second_right"]) == (0, 1)


def test_find_summary_ignores_judged_files_and_wants_exactly_one(tmp_path):
    (tmp_path / "news-baseline-e2b-pi5-1.json").write_text("{}")
    (tmp_path / "news-baseline-e2b-pi5-1.judged-qwen3-14b-q4_k_m.json").write_text("{}")
    assert find_summary(tmp_path).name == "news-baseline-e2b-pi5-1.json"
    (tmp_path / "gsm-gsm8k-e2b-pi5-1.json").write_text("{}")
    with pytest.raises(SystemExit):
        find_summary(tmp_path)


def test_kaggle_run_finds_each_benchmark(tmp_path):
    names = [
        "baseline-e2b-kaggle-t4-test_uq-3way-20261005-1.json",
        "gsm-gsm8k-e2b-kaggle-t4-20261008-1.json",
        "gsm-symbolic-e2b-kaggle-t4-20261008-1.json",
        "gsm-tiny-e2b-kaggle-t4-20261008-1.json",
        "news-baseline-e2b-kaggle-t4-20261007-1.json",
        "news-baseline-e2b-kaggle-t4-20261007-1.judged-qwen3-14b-q4_k_m.json",
        "news-rag-e2b-kaggle-t4-en-k3-20261008-1.json",
    ]
    for n in names:
        (tmp_path / n).write_text(json.dumps({"metrics": {"n": 733}}))
    d = str(tmp_path)
    assert kaggle_run(d, "scientsbank", "e2b", "test_uq", 733).name == names[0]
    assert kaggle_run(d, "scientsbank", "e2b", "test_uq", 100) is None
    assert kaggle_run(d, "gsm", "e2b", "gsm8k", 100).name == names[1]
    assert kaggle_run(d, "gsm", "e2b", "symbolic", 100).name == names[2]  # n may differ (Pi trimmed)
    assert kaggle_run(d, "news", "e2b", "test", 200).name == names[4]
    assert kaggle_run(d, "gsm", "e4b", "gsm8k", 100) is None


def test_read_board_state(tmp_path):
    assert read_board_state(tmp_path / "missing.txt") is None
    p = tmp_path / "board-state.txt"
    p.write_text("time: x\ncpu0: governor schedutil min 729600 max 1728000 cur 1113600 (kHz)\n")
    assert read_board_state(p) == {"governor": "schedutil", "cpu_min_khz": 729600,
                                   "cpu_max_khz": 1728000, "cpu_clock_locked": False}
    p.write_text("cpu0: governor schedutil min 1728000 max 1728000 cur 1728000 (kHz)\n")
    assert read_board_state(p)["cpu_clock_locked"] is True
    p.write_text("kernel: 6.1\n")
    assert read_board_state(p) == {}


def test_main_pairs_trimmed_symbolic_on_shared_ids(tmp_path):
    kdir = tmp_path / "kaggle"
    kaggle_rows = gsm_rows({"1/0": True, "1/1": True, "2/0": False, "2/1": True})
    write_kaggle(kdir, "gsm-symbolic-e2b-kaggle-t4-20261008-1",
                 gsm_summary("x", "e2b-kaggle-t4", "symbolic", 4, 0.75, instances=10), kaggle_rows)
    write_kaggle(kdir, "news-baseline-e2b-kaggle-t4-1",  # exists, so the unjudged board run is "pending"
                 {"benchmark": "news", "label": "e2b-kaggle-t4", "metrics": {"n": 1}},
                 [{"id": "news-000", "pred": None}])
    runs = tmp_path / "board"
    pi = make_run(runs, "gsm-e2b-pi5-symbolic-i1-20261010-100000",
                  gsm_summary("gsm-symbolic-e2b-pi5-1", "e2b-pi5", "symbolic", 2, 0.5, instances=1),
                  gsm_rows({"1/0": True, "2/0": True}),
                  state="cpu0: governor ondemand min 1500000 max 2400000 cur 2400000 (kHz)\n")
    jet = make_run(runs, "gsm-e2b-jetson-symbolic-i10-20261010-100000",
                   gsm_summary("gsm-symbolic-e2b-jetson-1", "e2b-jetson", "symbolic", 4, 0.75,
                               instances=10), kaggle_rows)
    news = make_run(runs, "news-e2b-pi5-test-20261010-100000",
                    {"_stem": "news-baseline-e2b-pi5-1", "benchmark": "news", "label": "e2b-pi5",
                     "metrics": {"n": 1, "errors": 0, "median_latency_s": 1.0}},
                    [{"id": "news-000", "pred": None, "prompt_tokens": 60}])
    est = tmp_path / "est.json"
    est.write_text(json.dumps({"estimates": []}))
    out = tmp_path / "out"
    main([str(pi), str(jet), str(news), "--estimate", str(est), "--kaggle", str(kdir),
          "--out", str(out)])
    data = json.loads(next(out.glob("board-check-*.json")).read_text())
    by = {r["run"]: r for r in data["runs"]}
    pi_vs = by[pi.name]["vs_kaggle"]
    assert (pi_vs["n"], pi_vs["kaggle_accuracy"], pi_vs["board_accuracy_paired"]) == (2, 0.5, 1.0)
    assert by[pi.name]["board_state"]["cpu_clock_locked"] is False
    assert by[news.name]["accuracy"] is None
    assert by[news.name]["vs_kaggle"]["pending"].startswith("judge")
    (pair,) = data["board_pairs"]
    assert (pair["first"], pair["second"], pair["n"]) == (pi.name, jet.name, 2)


STAGE1 = REPO / "results/board/e2b-jetson-test_ua-20260930-224702"


@pytest.mark.skipif(not STAGE1.exists(), reason="results/ isn't deployed to the boards")
def test_stage1_jetson_run_reads_as_before():
    """The committed board-check-20261005-042603.json values for this run, unchanged."""
    r = read_run(STAGE1)
    assert (r["benchmark"], r["split"], r["n"], r["accuracy"]) == ("scientsbank", "test_ua", 540, 0.5519)
    assert (r["j_per_answer"], r["median_latency_s"], r["mean_prompt_tokens_processed"],
            r["decode_ms_per_token"]) == (3.95, 0.42, 165.2, 24.8)
    k = kaggle_run(str(REPO / "results/kaggle"), "scientsbank", "e2b", "test_ua", 540)
    assert k.name == "baseline-e2b-kaggle-t4-test_ua-3way-20260929-102126.json"
    c = compare(outcomes(find_summary(STAGE1)), outcomes(k))
    assert (c["n"], c["same_grade"], c["second_accuracy"]) == (540, 507, 0.5519)
    assert (c["mcnemar"]["only_first_right"], c["mcnemar"]["only_second_right"]) == (15, 15)
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_board_check.py -v`
Expected: ImportError (`cannot import name 'find_summary'`). Every test in the file errors at collection.

- [ ] **Step 3: Implement**

In `src/edge_grader/board_check.py`, first replace the module docstring's opening and the "Per run folder" list. The first line, the "Per run folder in results/board/ this reads" list and the first sentence after it become:

```python
"""Check board runs against their Kaggle runs and the agentic-edge cost estimate.

Stage 1 estimated board cost from agentic-edge's MMLU-Pro request rates (cost.py) and
assumed board accuracy equals Kaggle's. A measured run on a board (scripts/board_grade.sh)
tests both; phase 0 extends that to GSM8K/GSM-Symbolic and news. Per run folder in
results/board/ this reads

    baseline-*.json | gsm-*.json | news-*.json, raw/*.jsonl   accuracy and each answer
    news-*.judged-<judge>.json, raw/*.judged-*.jsonl          news only: the Kaggle judge's grades
    board-state.txt                server.log...
```

Keep the remaining lines. Then add `GSM and news runs are paired with Kaggle on the question ids both answered, so the Pi's trimmed GSM-Symbolic set compares on its 100 ids.` after the sentence about McNemar.

Below `BOARDS = …`, add:

```python
SUMMARY_GLOBS = ("baseline-*.json", "gsm-*.json", "news-*.json")
JUDGE = "qwen3-14b-q4_k_m"  # the news judge's file tag (docs/results/topics.md)
CPU_STATE = re.compile(r"^cpu0: governor (\S+) min (\d+) max (\d+)")
```

Replace `compare` and `read_answers` with:

```python
def compare(a: dict[str, dict], b: dict[str, dict]) -> dict:
    """Two runs' outcomes, paired on the question ids both have."""
    ids = sorted(a.keys() & b.keys())

    def acc(x: dict[str, dict]) -> float | None:
        return round(sum(x[i]["right"] for i in ids) / len(ids), 4) if ids else None

    return {
        "n": len(ids),
        "same_grade": sum(a[i]["answer"] == b[i]["answer"] for i in ids),
        "first_accuracy": acc(a),
        "second_accuracy": acc(b),
        "mcnemar": mcnemar([a[i]["right"] for i in ids], [b[i]["right"] for i in ids]),
    }


def read_answers(summary_path: Path) -> dict[str, dict]:
    raw = summary_path.parent / "raw" / summary_path.with_suffix(".jsonl").name
    rows = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
    return {r["id"]: r for r in rows}


def find_summary(run_dir: Path) -> Path:
    """The runner's summary in a run folder (a judge's file is not one)."""
    found = sorted(p for g in SUMMARY_GLOBS for p in run_dir.glob(g) if ".judged-" not in p.name)
    if len(found) != 1:
        raise SystemExit(f"{run_dir}: expected one run summary, found {[p.name for p in found]}")
    return found[0]


def benchmark(summary: dict) -> str:
    return summary.get("benchmark", "scientsbank")  # grading summaries have no "benchmark" key


def split_of(summary: dict) -> str:
    return {"gsm": summary.get("set"), "news": "test"}.get(benchmark(summary), summary.get("split"))


def outcomes(summary_path: Path, judge: str = JUDGE) -> dict[str, dict] | None:
    """Per question, the answer given and whether it was right; None for an unjudged news run.

    Grading: the label. GSM: the flexible-extract number. News: the judge's grade."""
    bench = benchmark(json.loads(summary_path.read_text()))
    if bench == "news":
        judged = summary_path.parent / "raw" / f"{summary_path.stem}.judged-{judge}.jsonl"
        if not judged.exists():
            return None
        rows = [json.loads(line) for line in judged.read_text().splitlines() if line.strip()]
        return {r["id"]: {"answer": r["grade"], "right": r["grade"] == "correct"} for r in rows}
    rows = read_answers(summary_path).values()
    if bench == "gsm":
        return {r["id"]: {"answer": r["flexible"], "right": bool(r["correct"])} for r in rows}
    return {r["id"]: {"answer": r["pred"], "right": r["pred"] == r["gold"]} for r in rows}


def read_board_state(path: Path) -> dict | None:
    """CPU governor and frequency range from board-state.txt; None for runs made before it."""
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        m = CPU_STATE.match(line)
        if m:
            gov, lo, hi = m.groups()
            return {"governor": gov, "cpu_min_khz": int(lo), "cpu_max_khz": int(hi),
                    "cpu_clock_locked": lo == hi}
    return {}
```

In `read_run`:
1. Change its signature to `def read_run(run_dir: Path, judge: str = JUDGE) -> dict:`.
2. Replace `summary_path = next(run_dir.glob("baseline-*.json"))` with `summary_path = find_summary(run_dir)`.
3. After loading `summary`, add:

```python
    bench = benchmark(summary)
    m = summary["metrics"]
    accuracy = m.get("accuracy")
    if bench == "news":  # the runner doesn't score news; the Kaggle judge does
        judged = summary_path.with_name(f"{summary_path.stem}.judged-{judge}.json")
        accuracy = json.loads(judged.read_text())["metrics"]["accuracy"] if judged.exists() else None
```

4. Delete the later `m = summary["metrics"]` line.
5. In the returned dict:
   - Replace `"split": summary["split"],` with `"benchmark": bench, "split": split_of(summary), "instances": summary.get("generation", {}).get("instances"),`.
   - Replace `"accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "unparsed": m["unparsed"],` with `"accuracy": accuracy, "macro_f1": m.get("macro_f1"), "unparsed": m.get("unparsed"),`.
   - Replace the `"mean_prompt_tokens_sent"` value with:

```python
        "mean_prompt_tokens_sent": round(statistics.fmean(
            a["prompt_tokens"] for a in read_answers(summary_path).values()
            if a.get("prompt_tokens") is not None), 1),
```

   - Add `"board_state": read_board_state(run_dir / "board-state.txt"),` before `"_summary"`.

Replace `kaggle_run` with:

```python
def kaggle_run(directory: str, bench: str, model: str, split: str, n: int) -> Path | None:
    """The Kaggle run a board run is checked against. Grading: same split and n (as in stage 1).
    GSM and news: the latest Kaggle baseline run of the set; pairing is on shared ids."""
    d = Path(directory)
    if bench == "scientsbank":
        for path in sorted(d.glob(f"baseline-{model}-kaggle-*-{split}-*.json")):
            if json.loads(path.read_text())["metrics"]["n"] == n:
                return path
        return None
    pattern = (f"gsm-{split}-{model}-kaggle-*.json" if bench == "gsm"
               else f"news-baseline-{model}-kaggle-*.json")
    found = sorted(p for p in d.glob(pattern) if ".judged-" not in p.name)
    return found[-1] if found else None
```

In `main`:
1. Add `ap.add_argument("--judge", default=JUDGE, help="news judge file tag")`.
2. Make `runs = [read_run(d, args.judge) for d in sorted(args.runs)]`.
3. Replace the per-run loop body from `e = next(` through the `r["vs_kaggle"] = …` block with:

```python
        e = next(
            (x for x in est
             if r["benchmark"] == "scientsbank"
             and (x["model"], x["board"], x["split"]) == (r["model"], r["board"], r["split"])),
            None,
        )
        r["estimate"] = e and {
            "median_latency_s": e["median_latency_s"],
            "j_per_answer": e["mean_j_per_answer"],
            "measured_over_estimate_latency": round(r["median_latency_s"] / e["median_latency_s"], 2),
            "measured_over_estimate_energy": round(r["j_per_answer"] / e["mean_j_per_answer"], 2),
        }
        k = kaggle_run(args.kaggle, r["benchmark"], r["model"], r["split"], r["n"])
        mine, theirs = outcomes(r["_summary"], args.judge), k and outcomes(k, args.judge)
        if k and (mine is None or theirs is None):
            r["vs_kaggle"] = {"kaggle_run": k.name, "pending": "judge the news run first"}
        elif k:
            c = compare(mine, theirs)
            r["vs_kaggle"] = {"kaggle_run": k.name, "kaggle_accuracy": c.pop("second_accuracy"),
                              "board_accuracy_paired": c.pop("first_accuracy"), **c}
        else:
            r["vs_kaggle"] = None
```

Replace the `pairs` loop with:

```python
    pairs = []
    for a in runs:
        for b in runs:
            same = (a["benchmark"], a["model"], a["split"]) == (b["benchmark"], b["model"], b["split"])
            if not (same and (a["board"], b["board"]) == ("pi", "jetson")):
                continue
            oa, ob = outcomes(a["_summary"], args.judge), outcomes(b["_summary"], args.judge)
            pairs.append({
                "first": a["run"],
                "second": b["run"],
                "latency_ratio": round(a["median_latency_s"] / b["median_latency_s"], 1),
                "energy_ratio": round(a["j_per_answer"] / b["j_per_answer"], 1),
                **(compare(oa, ob) if oa is not None and ob is not None
                   else {"pending": "judge the news runs first"}),
            })
```

In the `settings.method` string, append: `" GSM/news: paired with Kaggle and between boards on the question ids both runs have; news correctness is the judge's grade."`

Replace the per-run `print` with:

```python
    def num(x, f=".3f"):
        return "n/a" if x is None else format(x, f)

    for r in runs:
        e, k = r["estimate"] or {}, r["vs_kaggle"] or {}
        print(
            f"{r['run']:46} t={r['threads']} acc {num(r['accuracy'])} (kaggle "
            f"{num(k.get('kaggle_accuracy'))} on {k.get('n')}, same {k.get('same_grade')}"
            f", p {k.get('mcnemar', {}).get('p')}{', ' + k['pending'] if 'pending' in k else ''})"
            f"  median {r['median_latency_s']:.2f} s (est {e.get('median_latency_s')})  "
            f"{r['j_per_answer']:.1f} J/question (est {e.get('j_per_answer')})"
            f"{'  CPU CLOCK LOCKED' if (r['board_state'] or {}).get('cpu_clock_locked') else ''}"
        )
```

In the pairs `print`, use `p.get('same_grade')`, `p.get('n')` and `p.get('mcnemar', {}).get('p')` so pending pairs print too.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_board_check.py -v`
Expected: all PASS, including `test_stage1_jetson_run_reads_as_before`. If that test fails on `j_per_answer` or `accuracy`, the refactor changed stage-1 numbers. Fix the code, not the test.

- [ ] **Step 5: Full suite, lint, and the stage-1 check end to end**

Run:

```bash
uv run pytest -q && uv run ruff check .
uv run edge-grader-board-check results/board/e*-pi5-test_ua-20261004-2[23]* results/board/e*-jetson-test_ua-* --out "$TMPDIR/bc"
```

Expected: tests pass and ruff is clean. The board-check prints the same accuracy, same-grade, p and J/answer for the four stage-1 runs as `results/estimates/board-check-20261005-042603.json`: Jetson E2B 3.95 J and 507/540, p 1.0. The output goes to a temp dir, so nothing is added to `results/`.

- [ ] **Step 6: Commit**

```bash
git add src/edge_grader/board_check.py tests/test_board_check.py
git commit -m "board-check: GSM and news runs, paired with Kaggle on shared ids

News accuracy comes from the Kaggle judge's files and is pending until they
exist. Records each run's CPU clock state from board-state.txt. Stage-1
numbers unchanged (regression test on the committed Jetson run).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: the news-judge notebook judges board runs

**Files:**
- Modify: `notebooks/kaggle_news_judge.ipynb` (cells `j00`, `j01`, `j11`, `j13`, `j15`, `j16`)

**Interfaces:**
- Consumes: `edge-grader-news-judge <run path>`. It already writes its output next to the run it's given, so a board run's grades land in that run's folder. No code change.
- Produces: `results/board/news-*/news-*.judged-qwen3-14b-q4_k_m.json` and `raw/*.judged-qwen3-14b-q4_k_m.jsonl` (what Task 2's `outcomes`/`read_run` read). Also `results-judged.zip` with only the judged files, at their repo paths.

- [ ] **Step 1: Edit the cells with a script** (keeps the notebook JSON valid; ids unchanged)

Save as `$SCRATCH/edit_judge_nb.py` (any scratch directory outside the repo) and run it with `uv run python $SCRATCH/edit_judge_nb.py notebooks/kaggle_news_judge.ipynb`:

```python
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
nb = json.loads(path.read_text())
cells = {c.get("id"): c for c in nb["cells"]}


def src(cell_id: str) -> str:
    return "".join(cells[cell_id]["source"])


def put(cell_id: str, text: str) -> None:
    cells[cell_id]["source"] = text.splitlines(keepends=True)


def swap(cell_id: str, old: str, new: str) -> None:
    s = src(cell_id)
    assert s.count(old) == 1, (cell_id, old)
    put(cell_id, s.replace(old, new))


swap("j00", "Grades the LiveNewsBench answers in `results/kaggle/news-*.json`",
     "Grades the LiveNewsBench answers in `results/kaggle/news-*.json` and the boards' "
     "`results/board/news-*/news-*.json` (phase 0)")
swap("j01", "RUNS = None   # None = every news run without this judge's grades yet; "
            "or a list of file names",
     "RUNS = None   # None = every news run (results/kaggle/, results/board/news-*/) without "
     "this judge's grades yet;\n             # or paths under results/, "
     "e.g. [\"board/news-e2b-pi5-test-…/news-baseline-e2b-pi5-….json\"]")
swap("j11", 'RESULTS = Path("/kaggle/working/edge-grader/results/kaggle")',
     'REPO_DIR = Path("/kaggle/working/edge-grader")\nRESULTS = REPO_DIR / "results"')
swap("j11",
     'runs = RUNS or sorted(f.name for f in RESULTS.glob("news-*.json")\n'
     '                      if ".judged-" not in f.name\n'
     '                      and not (RESULTS / f"{f.stem}.judged-{JUDGE_NAME}.json").exists())\n'
     'print(f"{len(runs)} runs to judge with {JUDGE[1]}:", *runs, sep="\\n  ")',
     'def unjudged(f):\n'
     '    return ".judged-" not in f.name and not f.with_name(f"{f.stem}.judged-{JUDGE_NAME}.json").exists()\n'
     '\n\n'
     '# Kaggle runs and the boards\' news runs; each is judged where it lies, so a board run\'s\n'
     '# grades land in its own run folder.\n'
     'found = [*RESULTS.glob("kaggle/news-*.json"), *RESULTS.glob("board/news-*/news-*.json")]\n'
     'runs = [RESULTS / r for r in RUNS] if RUNS else sorted(f for f in found if unjudged(f))\n'
     'print(f"{len(runs)} runs to judge with {JUDGE[1]}:", *(r.relative_to(REPO_DIR) for r in runs),\n'
     '      sep="\\n  ")')
swap("j11",
     '        print(f"== {run} ==")\n'
     '        !cd /kaggle/working/edge-grader && uv run edge-grader-news-judge results/kaggle/{run}',
     '        rel = run.relative_to(REPO_DIR)\n'
     '        print(f"== {rel} ==")\n'
     '        !cd /kaggle/working/edge-grader && uv run edge-grader-news-judge {rel}')
swap("j13",
     'for f in sorted(RESULTS.glob(f"news-*.judged-{JUDGE_NAME}.json")):',
     'for f in sorted([*RESULTS.glob(f"kaggle/news-*.judged-{JUDGE_NAME}.json"),\n'
     '                 *RESULTS.glob(f"board/news-*/news-*.judged-{JUDGE_NAME}.json")]):')
swap("j13", 'rows.append({"stage": r["run_stage"], "label": r["run_label"],',
     'rows.append({"where": f.parent.name, "stage": r["run_stage"], "label": r["run_label"],')
put("j15",
    "## 6. Download the results\n"
    "Kaggle only downloads single files, so this zips **only the judged files** (Kaggle's and the "
    "boards'), at their paths in the repo. Then, on your machine, from the repo root:\n"
    "```bash\n"
    "unzip -n ~/Downloads/results-judged.zip -d .\n"
    "git add results/kaggle results/board && git commit -m \"News judged by Qwen3-14B\" && git push\n"
    "```\n"
    "`-n` never overwrites a file that's already there.")
put("j16",
    "import zipfile\n\n"
    "from IPython.display import FileLink\n\n"
    "out = Path(\"/kaggle/working/results-judged.zip\")\n"
    "with zipfile.ZipFile(out, \"w\", zipfile.ZIP_DEFLATED) as z:\n"
    "    for f in sorted(RESULTS.rglob(f\"*.judged-{JUDGE_NAME}.json*\")):\n"
    "        z.write(f, f.relative_to(REPO_DIR))\n"
    "print(*zipfile.ZipFile(out).namelist(), sep=\"\\n\")\n"
    "FileLink(\"results-judged.zip\")")
path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
```

If an `assert` fires, the cell text differs from what this plan quotes. Open the cell, adjust `old` to match exactly, and re-run. Don't loosen the assert.

- [ ] **Step 2: Check the notebook still parses and the selection finds board runs**

Run:

```bash
uv run python -c "import json; nb=json.load(open('notebooks/kaggle_news_judge.ipynb')); print(len(nb['cells']), 'cells')"
git diff --stat notebooks/kaggle_news_judge.ipynb
```

Expected: `17 cells`, and only that file changed. Check that `git diff` shows only the six cells above: `json.dumps(indent=1)` matches nbformat's layout. If the whole file shows as changed, the original used a different indent. Redo with that indent so the diff stays small.

Then simulate the selection against a fake board run:

```bash
mkdir -p "$TMPDIR/nb/results/board/news-e2b-pi5-test-1" "$TMPDIR/nb/results/kaggle"
touch "$TMPDIR/nb/results/board/news-e2b-pi5-test-1/news-baseline-e2b-pi5-1.json" \
      "$TMPDIR/nb/results/kaggle/news-baseline-e2b-kaggle-t4-1.json" \
      "$TMPDIR/nb/results/kaggle/news-baseline-e2b-kaggle-t4-1.judged-qwen3-14b-q4_k_m.json"
uv run python - "$TMPDIR/nb" <<'EOF'
import sys
from pathlib import Path
REPO_DIR = Path(sys.argv[1]); RESULTS = REPO_DIR / "results"; JUDGE_NAME = "qwen3-14b-q4_k_m"; RUNS = None
def unjudged(f):
    return ".judged-" not in f.name and not f.with_name(f"{f.stem}.judged-{JUDGE_NAME}.json").exists()
found = [*RESULTS.glob("kaggle/news-*.json"), *RESULTS.glob("board/news-*/news-*.json")]
runs = [RESULTS / r for r in RUNS] if RUNS else sorted(f for f in found if unjudged(f))
print([str(r.relative_to(REPO_DIR)) for r in runs])
EOF
```

Expected: `['results/board/news-e2b-pi5-test-1/news-baseline-e2b-pi5-1.json']`. The Kaggle run is already judged, so it's skipped.

- [ ] **Step 3: Commit**

```bash
git add notebooks/kaggle_news_judge.ipynb
git commit -m "News judge notebook: also judge the boards' news runs; download only judged files

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: deploy, preflight and smoke runs on both boards

This task needs the boards over ssh. The owner runs anything that needs a password. **Nothing here writes into `results/board/`.**

**Files:** none changed. Outputs go to `results/local/` on the boards and are deleted after.

**Interfaces:**
- Consumes: Tasks 1–3 committed; `scripts/deploy.sh`; ssh aliases `MITLAB-EDGE`, `MITLAB-JETSON`.
- Produces: both boards on the new commit (`DEPLOYED.json`), with known clock state and a passing smoke run for each new path.

- [ ] **Step 1: Push and deploy**

```bash
git push
scripts/deploy.sh
```

Expected, for each board: `tests: N passed` (the stage-1 regression test shows as skipped, since `results/` isn't deployed) and `deployed <commit>`. If a board says `busy`, an agentic-edge or edge-grader run is going. Wait for it; don't deploy under it.

- [ ] **Step 2: Check that no agentic-edge run is due on the boards**

```bash
ssh MITLAB-JETSON 'cd ~/research/agentic-edge/benchmark && python3 queue_ctl.py --status 2>/dev/null | head -20'
ssh MITLAB-EDGE   'cd ~/Research/agentic-edge/benchmark && python3 queue_ctl.py --status 2>/dev/null | head -20'
```

`board_grade.sh` refuses to start while a lab run holds the board, but a queued lab job could start between our runs. If jobs are queued, ask the owner to agree a window with the lab first. agentic-edge's §10 lists 8 stale blocked Jetson jobs, which can be cancelled; that's for the owner to decide, not this plan.

- [ ] **Step 3: Check the Jetson's clocks (Review Focus 3)**

```bash
ssh MITLAB-JETSON 'c=/sys/devices/system/cpu/cpu0/cpufreq; echo "cpu0 min $(cat $c/scaling_min_freq) max $(cat $c/scaling_max_freq)"; for d in /sys/class/devfreq/*; do echo "$(basename $d) min $(cat $d/min_freq) max $(cat $d/max_freq)"; done; nvpmodel -q 2>/dev/null'
```

**If CPU min == max**, the clocks are still locked from agentic-edge's 2026-10-05..07 runs (its `findings/RESULTS.md` §7.7). Stop and tell the owner. Restoring needs their password: `sudo jetson_clocks --restore`, or a reboot. The owner then chooses:
- **(a)** restore first. This keeps energy comparable with the 2026-09-30 test_ua runs, and is the recommended option.
- **(b)** run locked, and note in `docs/results/baselines.md` that phase-0 Jetson energy isn't comparable with the test_ua runs.

Either way, `board-state.txt` records the state per run.

- [ ] **Step 4: Dry run on each board** (the real board detection, no `EG_BOARD`)

```bash
ssh MITLAB-JETSON 'cd ~/research/edge-grader && DRY_RUN=1 BENCH=gsm INSTANCES=10 scripts/board_grade.sh e2b symbolic'
ssh MITLAB-EDGE   'cd ~/Research/edge-grader && DRY_RUN=1 BENCH=gsm INSTANCES=1 scripts/board_grade.sh e2b symbolic'
```

Expected:
- Jetson: `run_dir: …/results/board/gsm-e2b-jetson-symbolic-i10-<stamp>`, with a `server:` line ending in `-ngl 99 -rea off --reasoning-budget -1 --cache-ram 0`.
- Pi: `…gsm-e2b-pi5-symbolic-i1-<stamp>` and `-t 3 … -ngl 0 …`.
- Neither creates a folder (`ls results/board | grep symbolic` is empty).

- [ ] **Step 5: Smoke runs, unmeasured, into results/local**

On each board (Jetson shown; on the Pi use `~/Research/edge-grader`):

```bash
ssh MITLAB-JETSON 'cd ~/research/edge-grader &&
  MEASURE=0 EG_RUN_DIR=$PWD/results/local/smoke-gsm  BENCH=gsm  scripts/board_grade.sh e2b gsm8k 3 &&
  MEASURE=0 EG_RUN_DIR=$PWD/results/local/smoke-sym  BENCH=gsm INSTANCES=1 scripts/board_grade.sh e2b symbolic 2 &&
  MEASURE=0 EG_RUN_DIR=$PWD/results/local/smoke-news BENCH=news scripts/board_grade.sh e2b test 2 &&
  cat results/local/smoke-*/board-state.txt | head -5 &&
  grep -h "\"n\"\|accuracy\|answered\|errors" results/local/smoke-*/*.json'
```

Expected:
- GSM: `n` 3 and 2, `errors 0`, accuracy is some fraction.
- News: `n` 2, `errors 0`.
- `board-state.txt` has a `cpu0:` line.

On the Pi, each run also stops and restarts `va-llm`. Check it's back afterwards: `ssh MITLAB-EDGE 'systemctl is-active va-llm'` should say `active`.

Then delete the smoke folders (results/local isn't fetched, but keep the boards clean):

```bash
ssh MITLAB-JETSON 'rm -rf ~/research/edge-grader/results/local/smoke-*'
ssh MITLAB-EDGE   'rm -rf ~/Research/edge-grader/results/local/smoke-*'
```

No commit in this task.

---

### Task 5: the Jetson's runs (~15 h)

**Interfaces:**
- Consumes: Task 4 passed on the Jetson; the clock decision made.
- Produces: 10 run folders in the Jetson's `~/research/edge-grader/results/board/`:
  - `{e2b,e4b}-jetson-test_uq-*`
  - `{e2b,e4b}-jetson-test_ud-*`
  - `gsm-{e2b,e4b}-jetson-gsm8k-*`
  - `gsm-{e2b,e4b}-jetson-symbolic-i10-*`
  - `news-{e2b,e4b}-jetson-test-*`

- [ ] **Step 1: Start the run list, detached**

```bash
ssh MITLAB-JETSON 'cd ~/research/edge-grader && nohup bash -c "
for m in e2b e4b; do
  for args in \"scientsbank test_uq\" \"scientsbank test_ud\" \"gsm gsm8k\" \"gsm symbolic\" \"news test\"; do
    set -- \$args
    echo \"== \$(date) \$m \$1 \$2\"
    BENCH=\$1 INSTANCES=10 scripts/board_grade.sh \$m \$2
    echo \"== rc=\$?\"
    sleep 20   # let the previous llama-server exit before the next run checks the board is free
  done
done" > ~/phase0-jetson.log 2>&1 < /dev/null &'
```

`BENCH` and `INSTANCES` are set on every call (Review Focus 5). `INSTANCES` is ignored by the non-symbolic runs.

- [ ] **Step 2: Watch it**

```bash
ssh MITLAB-JETSON 'grep "^==" ~/phase0-jetson.log; tail -3 ~/phase0-jetson.log'
```

Every `== rc=` should be `0`.

- **A run refused with "board is busy"** (rc 1, no folder): re-run that one line by hand when the board is free.
- **A run failed part-way:** keep its folder (never delete a result), note it, and re-run that line. The board-check is then given the complete folder only.

- [ ] **Step 3: When all 10 are done, list them**

```bash
ssh MITLAB-JETSON 'ls ~/research/edge-grader/results/board | grep -E "test_u[qd]|^gsm-|^news-"'
```

Expected: the 10 folder names above.

---

### Task 6: the Pi's runs (~28–31 h)

Runs at the same time as Task 5: it's a different board.

**Interfaces:**
- Consumes: Task 4 passed on the Pi.
- Produces: 10 run folders in the Pi's `~/Research/edge-grader/results/board/`:
  - `{e2b,e4b}-pi5-test_uq-*`
  - `{e2b,e4b}-pi5-test_ud-*`
  - `gsm-{e2b,e4b}-pi5-gsm8k-*`
  - `gsm-{e2b,e4b}-pi5-symbolic-i1-*`
  - `news-{e2b,e4b}-pi5-test-*`

- [ ] **Step 1: Start the run list, detached** (the same as Task 5, but `INSTANCES=1`)

```bash
ssh MITLAB-EDGE 'cd ~/Research/edge-grader && nohup bash -c "
for m in e2b e4b; do
  for args in \"scientsbank test_uq\" \"scientsbank test_ud\" \"gsm gsm8k\" \"gsm symbolic\" \"news test\"; do
    set -- \$args
    echo \"== \$(date) \$m \$1 \$2\"
    BENCH=\$1 INSTANCES=1 scripts/board_grade.sh \$m \$2
    echo \"== rc=\$?\"
    sleep 20
  done
done" > ~/phase0-pi.log 2>&1 < /dev/null &'
```

- [ ] **Step 2: Watch it, as in Task 5**

```bash
ssh MITLAB-EDGE 'grep "^==" ~/phase0-pi.log; tail -3 ~/phase0-pi.log; systemctl is-active va-llm'
```

`va-llm` is `inactive` during a run and `active` between runs and at the end. If it's inactive after the list finishes, tell the owner. The script's `restore` should have restarted it.

- [ ] **Step 3: When all 10 are done, list them**

```bash
ssh MITLAB-EDGE 'ls ~/Research/edge-grader/results/board | grep -E "test_u[qd]|^gsm-|^news-"'
```

Expected: the 10 folder names above.

---

### Task 7: fetch, judge, check, and write the baseline table

**Files:**
- Add: `results/board/<20 new run folders>/` (fetched, then judged files added for the 4 news runs)
- Add: `results/estimates/board-check-<stamp>.json` (written by the CLI)
- Modify: `docs/results/baselines.md` (new section), `docs/plan.md` (tick phase 0), `docs/research-outline.md` (stage 0 output cell)

**Interfaces:**
- Consumes: Tasks 2, 3, 5 and 6.
- Produces: the phase-0 table, and the gate decision for phase B on the boards.

- [ ] **Step 1: Fetch and commit the raw runs**

```bash
scripts/fetch_results.sh
git add results/board
git commit -m "Phase 0 board runs: SciEntsBank uq/ud, GSM, news on Pi 5 and Jetson

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

Expected: 20 new folders listed by `fetch_results.sh`.

- [ ] **Step 2: Judge the four board news runs on Kaggle (the owner, in the Kaggle UI)**

1. Open `notebooks/kaggle_news_judge.ipynb` on Kaggle (T4, Internet on, `GITHUB_TOKEN` secret), leaving `RUNS = None`, and run all cells.
2. Cell `j11` should list exactly the 4 `results/board/news-*` runs. The Kaggle runs are already judged.
3. Check cell `j13`: `unparsed` should be 0 for each.
4. Download `results-judged.zip`, then:

```bash
unzip -n ~/Downloads/results-judged.zip -d .
git status --short results/board | head
git add results/board
git commit -m "Phase 0: board news runs judged by Qwen3-14B

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Expected: 8 new files (a `.judged-qwen3-14b-q4_k_m.json` and a `raw/….jsonl` per news run). No existing file changes.

- [ ] **Step 3: Run the board-check on all 20 runs**

```bash
uv run edge-grader-board-check \
  results/board/e*-*-test_u[qd]-* results/board/gsm-* results/board/news-*
```

Expected:
- One line per run, none with `pending`.
- Four `Pi vs Jetson` pair lines per model: test_uq, test_ud, gsm8k, symbolic (paired on Pi's 100 ids), news.
- `wrote results/estimates/board-check-<stamp>.json`.
- No line ends in `CPU CLOCK LOCKED` unless the owner chose to run locked (Task 4, Step 3).

**The gate.** Look at every `p` against Kaggle:
- **If any p < 0.05**, stop. The board's answers differ from Kaggle's beyond noise, and that has to be explained before phase B goes on the boards. Compare that run's raw replies with Kaggle's.
- **If none are below 0.05**, the gate passes.

- [ ] **Step 4: Print the table rows from the check file**

```bash
uv run python - "$(ls results/estimates/board-check-*.json | tail -1)" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print("| Benchmark | Set | Model | Board | n | Board acc. | Kaggle acc. (same ids) | Same answer | McNemar p | Median s | J / question | CPU locked |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in sorted(d["runs"], key=lambda r: (r["benchmark"], r["split"], r["model"], r["board"])):
    k = r["vs_kaggle"] or {}
    st = r["board_state"] or {}
    set_ = r["split"] + (f" ({r['instances']} var.)" if r["instances"] else "")
    print(f"| {r['benchmark']} | {set_} | {r['model'].upper()} | {r['board']} | {r['n']} | "
          f"{r['accuracy']:.1%} | {k.get('kaggle_accuracy', 0):.1%} | {k.get('same_grade')}/{k.get('n')} | "
          f"{k.get('mcnemar', {}).get('p')} | {r['median_latency_s']} | {r['j_per_answer']} | "
          f"{'yes' if st.get('cpu_clock_locked') else 'no'} |")
EOF
```

- [ ] **Step 5: Write the docs**

In `docs/results/baselines.md`, add a section after "Cost on the boards (measured, test_ua)":

```markdown
### Phase 0: every baseline on both boards (measured, 2026-10-<dd>)

*Spec: [`../superpowers/specs/2026-10-08-baseline-variations-design.md`](../superpowers/specs/2026-10-08-baseline-variations-design.md) §4. Runs: `results/board/` (20 folders), checked with `uv run edge-grader-board-check` → [`board-check-<stamp>.json`](../../results/estimates/board-check-<stamp>.json). Same GGUFs, flags and builds as test_ua (Pi `661643e` at `-t 3`, Jetson `a894dae`). The Pi ran the first GSM-Symbolic variant set only (100 of 1,000); its comparisons are on those 100 ids. News is judged by Qwen3-14B on Kaggle, as for the Kaggle runs.*

<the table from Step 4>
```

Then add three bullet findings under the table, each from the numbers alone:
1. Does board accuracy equal Kaggle's on every benchmark (the p values)?
2. The Pi/Jetson time and energy ratio per benchmark (from `board_pairs`), against test_ua's 7.7–8.8× and 7.0–7.8×. GSM and news generate long answers, so expect the ratio to move towards MMLU-Pro's 3.4× / 2.3×.
3. J per question per benchmark, which feeds RQ4.

If the owner chose to run with locked Jetson clocks, add that as a caveat line.

**`docs/plan.md`:** tick the phase-0 items, adding the dates and the check file's name.

**`docs/research-outline.md`:** in the stages table, replace the stage-0 output cell with `Done <date>: every benchmark on both boards; board accuracy = Kaggle (all McNemar p ≥ <min p>). See [results/baselines.md](results/baselines.md#phase-0-every-baseline-on-both-boards-measured-2026-10-dd)`. Use the real heading anchor.

- [ ] **Step 6: Commit**

```bash
git add results/estimates docs/results/baselines.md docs/plan.md docs/research-outline.md
git commit -m "Phase 0 done: the baseline on both boards for every benchmark

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
