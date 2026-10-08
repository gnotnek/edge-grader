# Phase B2: little-gemma and prefix reuse on the Jetson, implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure grading through a second engine (little-gemma) and with prefix reuse on the Jetson, on test_ua for E2B and E4B. Include the fair llama.cpp control, and record accuracy, byte-identity and cost.

**Architecture:**
- **`board_grade.sh`** gets a `VARIANT` switch with predefined, tested settings:
  - `base`: llama.cpp as today
  - `swafull`: llama.cpp `--swa-full`
  - `lg`: little-gemma through `agentic-edge`'s `lg_openai_shim.py`
  - `lg-reuse`: the same with the engine's `-reuse`
  - It also gets `RAG=<retrieval file>`, for the conditional RAG variants.
- **The board-check:**
  - computes the energy window from the run's own summary when the server log has no llama.cpp timing lines (the shim writes none)
  - tolerates missing token counts (the shim returns empty `usage`)
  - records the variant
- **Analysis** pairs each variant with the Jetson's llama.cpp baseline.

**Tech Stack:** bash (board side), Python 3 via uv, pytest; on the Jetson, agentic-edge's little-gemma build and shim (read, run, never modified).

**Spec:** [`docs/superpowers/specs/2026-10-08-baseline-variations-design.md`](../specs/2026-10-08-baseline-variations-design.md) §6 B2. **Depends on:** phase 0 merged and its gate passed (board accuracy = Kaggle).

## What is already known (diagnosed 2026-10-09 from the committed stage-1 Jetson run)

`results/board/e2b-jetson-test_ua-20260930-224702/` shows:
- **Answers are already in question order:** 134 question changes over 135 questions. The dataset groups them.
- **llama-server routes them well:** with 4 slots it picks the slot with the most shared prefix (`selected slot by LCP similarity, f_sim_best = 0.496`).
- **But it still re-processes almost the whole prompt:** `prompt eval … 124 tokens` for a 125-token prompt. Mean processed 165.2 against 166.2 sent.

**Likely cause (the first thing Task 2 tests):** Gemma 4's sliding-window attention layers. llama.cpp can't trim a sliding-window cache back to a shared prefix unless the full cache is kept (`--swa-full`).

**Consequences:**
1. **The planned `llama-sorted` control is the existing baseline:** the order is already sorted. The fair llama.cpp control for reuse is **`--swa-full`**, if the pilot shows it brings reuse back.
2. **Reuse can at most skip the shared part** (instructions, question, reference answer): about half of a ~165-token prompt. The label list comes after the student answer, so it isn't shared. Under RAG the shared part also holds three ~250-word passages, which is why the RAG variants are the conditional follow-up.

The prompt text is **never reordered**, so results stay comparable.

## Global Constraints

**Fixed setup:**
- Jetson only. little-gemma on the Pi is out of scope (~40× slower prompt processing).
- test_ua (540), E2B and E4B, thinking off.
- The same GGUFs and the same grading prompt as every other run.

**Serving:**
- llama.cpp flags as the baseline: `-c 8192 --host 127.0.0.1 --port 8080 -ngl 99 -rea off --reasoning-budget -1 --cache-ram 0`. `swafull` adds only `--swa-full`.
- little-gemma through `agentic-edge/benchmark/lg_openai_shim.py`: `--thinking off --think -1 --port 8080`. `lg-reuse` adds only `--engine-flags=-reuse`. Note the `=`: without it argparse rejects a value that starts with a dash (agentic-edge RESULTS.md §7.6).

**Separation:**
- Run through `scripts/board_grade.sh` only, never agentic-edge's queue.
- Nothing written into agentic-edge's folders. Its engine and shim are only executed.

**Results:** every run writes its summary to `results/`; never hand-edit.

**Pass conditions:**
- `lg-reuse` must be **byte-identical** to `lg` (same reply text for every answer), as agentic-edge's S9 gate demanded. If not, it's reported as a failed condition, not as a speed-up.
- **Jetson clocks:** check before running (phase 0 Task 4, Step 3), and compare energy only between runs made in the same clock state.

**Commits:** end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **The shim returns `usage: {}`, so every row's token counts are None.** The board-check must not crash averaging them. Task 1: `test_read_run_without_token_counts_or_timing_lines`.
2. **The shim's log has no `print_timing` lines.** Energy must come from the run's own window (summary `time` minus `wall_s`), not from an empty request list. Task 1: the same test.
3. **A leftover llama-server or shim from an earlier variant** would serve the next run on :8080. `restore` and the start-up must kill both engines' processes. Task 1, Step 3 (`stop_engines`); Task 3 checks `server.log` names the engine.
4. **`VARIANT=lg*` on the Pi**, or with `BENCH` other than SciEntsBank, must be refused, not attempted. Task 1: `test_variants_are_jetson_scientsbank_only`.
5. **`-reuse` may not reuse across different prefixes.** agentic-edge's notes describe "one prefix snapshot", built for MMLU-Pro's shared 5-shot prefix. Task 2 reads how it picks the prefix before any measured run. If it can't follow a prefix that changes every few answers, `lg-reuse` will show no gain, and the write-up must say why rather than report "reuse doesn't help".

---

## File map

| File | Change | Responsibility |
|---|---|---|
| `scripts/board_grade.sh` | Modify | `VARIANT`, `RAG`, `stop_engines`, variant in `board-state.txt` |
| `tests/test_board_grade.py` | Modify | Dry-run tests for the variants |
| `src/edge_grader/board_check.py` | Modify | Window fallback; no-token-count tolerance; `variant` |
| `tests/test_board_check.py` | Modify | Tests |
| `docs/results/variations.md` | Modify | `## B2` section |

---

### Task 1: variants in the board script; board-check handles the shim

**Files:**
- Modify: `scripts/board_grade.sh`, `src/edge_grader/board_check.py`
- Test: `tests/test_board_grade.py`, `tests/test_board_check.py`

**Interfaces:**
- Consumes: phase 0's `server_cmd()`, `runner()`, `NAME`, `DRY_RUN`, `board_state()`; phase 0's `read_run`, `read_board_state`.
- Produces:
  - **Env `VARIANT`** = `base` (default) | `swafull` | `lg` | `lg-reuse`; any value but `base` is Jetson + SciEntsBank only.
  - **Env `RAG`** = a retrieval file path (SciEntsBank only), giving `--rag <file> --k 3`.
  - **Folder names** (SciEntsBank): `[rag-][<variant>-]<model>-<board>-<split>-<stamp>`, e.g. `lg-reuse-e2b-jetson-test_ua-…`, `rag-swafull-e4b-jetson-test_ua-…`.
  - **`board-state.txt`** gains `variant: <VARIANT>` and `engine: <path or llama-server>`.
  - **`read_board_state()`** adds `"variant"` when present.
  - **`read_run()`:**
    - `prefill_fixed_ms`, `prefill_ms_per_token`, `decode_ms_per_token`, `mean_prompt_tokens_processed` are None when the log has no timing lines
    - `mean_prompt_tokens_sent` is None when no row has a count
    - `"window_from": "server.log" | "summary"`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_board_grade.py`, adding `"VARIANT"` and `"RAG"` to the variables `dry()` strips:

```python
def test_swafull_adds_only_that_flag():
    base = fields(dry("e2b"))
    swa = fields(dry("e2b", VARIANT="swafull"))
    assert swa["server"] == base["server"] + " --swa-full"
    assert re.fullmatch(rf"{REPO}/results/board/swafull-e2b-jetson-test_ua-{STAMP}", swa["run_dir"])
    assert swa["runner"] == base["runner"].replace(base["run_dir"], swa["run_dir"])


def test_little_gemma_variants_use_the_shim():
    lg = fields(dry("e4b", VARIANT="lg"))
    assert "lg_openai_shim.py --engine " in lg["server"]
    assert lg["server"].endswith("--thinking off --think -1 --port 8080")
    reuse = fields(dry("e4b", VARIANT="lg-reuse"))
    assert reuse["server"] == lg["server"] + " --engine-flags=-reuse"
    assert re.fullmatch(rf"{REPO}/results/board/lg-reuse-e4b-jetson-test_ua-{STAMP}", reuse["run_dir"])


def test_variants_are_jetson_scientsbank_only():
    for env, board in (({"VARIANT": "lg"}, "pi5"), ({"VARIANT": "lg", "BENCH": "gsm"}, "jetson"),
                       ({"VARIANT": "fast"}, "jetson")):
        p = dry("e2b", board=board, **env)
        assert p.returncode != 0 and "VARIANT" in p.stderr, (env, board, p.stderr)


def test_rag_adds_the_retrieval_file_and_prefix():
    f = fields(dry("e2b", VARIANT="lg-reuse",
                   RAG="results/rag/retrieval-scientsbank-en.bge-small-en-k5.json"))
    assert re.fullmatch(rf"{REPO}/results/board/rag-lg-reuse-e2b-jetson-test_ua-{STAMP}", f["run_dir"])
    assert f["runner"].endswith(
        f"--out {f['run_dir']} --rag results/rag/retrieval-scientsbank-en.bge-small-en-k5.json --k 3")
```

Append to `tests/test_board_check.py` (uses phase 0's `make_run`):

```python
def test_read_run_without_token_counts_or_timing_lines(tmp_path):
    summary = {"_stem": "baseline-e2b-jetson-test_ua-3way-1", "label": "e2b-jetson", "split": "test_ua",
               "time": "2026-10-10T02:00:06+00:00",  # = 10:00:06 +08:00, the window's end
               "metrics": {"n": 2, "accuracy": 0.5, "macro_f1": 0.4, "unparsed": 0, "errors": 0,
                           "median_latency_s": 1.0, "wall_s": 5.0}}
    rows = [{"id": "a", "gold": "correct", "pred": "correct", "prompt_tokens": None},
            {"id": "b", "gold": "correct", "pred": "incorrect", "prompt_tokens": None}]
    run = make_run(tmp_path, "lg-e2b-jetson-test_ua-1", summary, rows,
                   state="variant: lg\nengine: /x/run-cuda-i8\n")
    (run / "server.log").write_text("shim: request 1  500 prompt chars  7 reply chars  0.40s  stop\n")
    r = read_run(run)
    assert r["window_from"] == "summary" and r["grading_window_s"] == 5.0
    assert r["j_per_answer"] == pytest.approx(12.5)  # 5 s x 5 W / 2 answers
    assert r["mean_prompt_tokens_sent"] is None and r["decode_ms_per_token"] is None
    assert r["board_state"]["variant"] == "lg"
```

`make_run` writes telemetry from t=0 to t=2n+2 at 5 W, which covers the summary window [1, 6] s for n=2.

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_board_grade.py tests/test_board_check.py -v`
Expected: the new tests FAIL.

- [ ] **Step 3: Implement in `board_grade.sh`**

After the `THINKING` block (or the `INSTANCES` check if B1 isn't merged yet), add:

```bash
VARIANT="${VARIANT:-base}"
case "$VARIANT" in base|swafull|lg|lg-reuse) ;;
  *) echo "unknown VARIANT: $VARIANT (base|swafull|lg|lg-reuse)" >&2; exit 1 ;;
esac
RAG="${RAG:-}"
if [ "$VARIANT" != base ] || [ -n "$RAG" ]; then
  [ "$BENCH" = scientsbank ] || { echo "VARIANT/RAG are for BENCH=scientsbank only" >&2; exit 1; }
fi
```

After the board `case` (where `BOARD` is known), add:

```bash
case "$VARIANT" in lg|lg-reuse|swafull)
  [ "$BOARD" = jetson ] || { echo "VARIANT=$VARIANT runs on the Jetson only" >&2; exit 1; } ;;
esac
LG_ENGINE="${LG_ENGINE:-$HOME/build/little-gemma/build/run-cuda-i8}"
```

In the `NAME` block's SciEntsBank branch, after the `think-` line, add:

```bash
  [ "$VARIANT" != base ] && NAME="$VARIANT-$NAME"
  [ -n "$RAG" ] && NAME="rag-$NAME"
```

Replace `server_cmd` with:

```bash
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
```

If B1 isn't merged, use the plain `-rea off --reasoning-budget -1` text in place of the `THINKING` expression.

In `runner`, change the SciEntsBank line's end to:

```bash
--out "$RUN_DIR" $( [ "$THINKING" = on ] && echo --thinking ) $( [ -n "$RAG" ] && echo "--rag $RAG --k 3" ) ;;
```

**Check the spacing:** the test compares the whole runner line. When both are empty, the line ends in `--out <dir>`, because `echo` of an unquoted, word-split empty substitution adds no words. Confirm this with the dry-run tests.

Replace the start-up `pkill -f "llama-server -m" …` (Jetson branch) and `restore` with a shared `stop_engines`:

```bash
stop_engines() {   # bracketed patterns never match this script or an ssh command line
  pkill -f "[l]lama-server -m $MODEL" 2>/dev/null
  pkill -f "[l]g_openai_shim.py" 2>/dev/null
  sleep 2
  pkill -f "[r]un-cuda-i8 -m" 2>/dev/null
}
restore() {
  stop_engines
  if [ "$BOARD" = pi5 ]; then
    sudo -n systemctl start va-llm && echo "$(date) va-llm restarted"
  fi
}
```

In the Jetson branch of the start-up, replace `pkill -f "llama-server -m" 2>/dev/null` with `pkill -f "[l]lama-server -m" 2>/dev/null; stop_engines`.

In `board_state()`, add before `return 0`:

```bash
  echo "variant: $VARIANT"
  case "$VARIANT" in lg|lg-reuse) echo "engine: $LG_ENGINE ($(cat "$(dirname "$LG_ENGINE")/BUILD_INFO" 2>/dev/null | head -1))" ;;
                     *) echo "engine: $LLAMA_SERVER" ;; esac
```

In the "fail before touching the board" check, use `$LG_ENGINE` instead of `$LLAMA_SERVER` for the `lg` variants, and also require `$AE/lg_openai_shim.py`:

```bash
ENGINE_BIN=$LLAMA_SERVER
case "$VARIANT" in lg|lg-reuse) ENGINE_BIN=$LG_ENGINE
  [ -f "$AE/lg_openai_shim.py" ] || { echo "missing: $AE/lg_openai_shim.py" >&2; exit 1; } ;; esac
for f in "$ENGINE_BIN" "$MODEL"; do
  [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }
done
```

Pass `VARIANT="$VARIANT" RAG="$RAG" LG_ENGINE="$LG_ENGINE"` in the inner `exec env …` line.

E4B's first little-gemma load can take ~150 s (agentic-edge). The health loop waits up to 300 s, which is enough.

- [ ] **Step 4: Implement in `board_check.py`**

1. Add a line pattern for the extra board-state lines, and extend `read_board_state` so the result also carries `variant`/`engine` when present:

```python
STATE_LINE = re.compile(r"^(variant|engine): (.*)$")
```

```python
def read_board_state(path: Path) -> dict | None:
    """CPU governor and frequency range (and the variant, engine) from board-state.txt;
    None for runs made before it existed."""
    if not path.exists():
        return None
    out: dict = {}
    for line in path.read_text().splitlines():
        m = CPU_STATE.match(line)
        if m and "governor" not in out:
            gov, lo, hi = m.groups()
            out |= {"governor": gov, "cpu_min_khz": int(lo), "cpu_max_khz": int(hi),
                    "cpu_clock_locked": lo == hi}
        s = STATE_LINE.match(line)
        if s:
            out[s.group(1)] = s.group(2)
    return out
```

(Phase 0's `test_read_board_state` still passes: the same keys, and `{}` for a file with neither line.)

2. In `read_run`, replace the window and prefill block (from `reqs = parse_server_log(...)` through the prefill regression) with:

```python
    reqs = parse_server_log((run_dir / "server.log").read_text())
    m = summary["metrics"]
    if reqs:
        start = min(r["end"] - (r["prompt_ms"] + r["gen_ms"]) / 1000 for r in reqs)
        end = max(r["end"] for r in reqs)
        window_from = "server.log"
    else:  # little-gemma's shim logs no per-request timings: the runner's own loop window
        end = datetime.fromisoformat(summary["time"]).timestamp()
        start = end - m["wall_s"]
        window_from = "summary"
```

Keep the telemetry/energy lines after it. Then compute the prefill and decode stats only when `reqs` is not empty:

```python
    if reqs:
        prefill_x = [r["prompt_tokens"] for r in reqs]
        slope, intercept = statistics.linear_regression(prefill_x, [r["prompt_ms"] for r in reqs])
        decode = statistics.median(r["gen_ms"] / r["gen_tokens"] for r in reqs if r["gen_tokens"])
        processed = statistics.fmean(prefill_x)
    else:
        slope = intercept = decode = processed = None
    sent = [a["prompt_tokens"] for a in read_answers(summary_path).values()
            if a.get("prompt_tokens") is not None]
```

3. In the returned dict:
   - `"mean_prompt_tokens_processed": processed and round(processed, 1)`
   - `"mean_prompt_tokens_sent": round(statistics.fmean(sent), 1) if sent else None`
   - `"prefill_fixed_ms": intercept and round(intercept, 1)`
   - `"prefill_ms_per_token": slope and round(slope, 3)`
   - `"decode_ms_per_token": decode and round(decode, 1)`
   - add `"window_from": window_from` and `"requests_in_log": len(reqs)`

   Remove the earlier `m = summary["metrics"]` if it's now defined twice.

4. In `SUMMARY_GLOBS`, add `"rag-*.json"` (RAG grading summaries are named `rag-<label>-…`). In `kaggle_run`'s SciEntsBank branch, a RAG board run (summary `stage == "rag"`) looks up `rag-{model}-kaggle-*-en-k3-{split}-*.json`. Add a `stage` parameter: `kaggle_run(directory, bench, model, split, n, thinking=False, stage="baseline")`, with the prefix `"think"` if thinking, else `stage`. Pass `stage=summary stage` from `main` via a new `read_run` key `"stage": summary.get("stage", "baseline")`.

- [ ] **Step 5: Tests, lint, commit**

Run: `uv run pytest -q && uv run ruff check . && bash -n scripts/board_grade.sh`
Expected: all PASS, including phase 0's regression test on the stage-1 Jetson run.

```bash
git add scripts/board_grade.sh tests/test_board_grade.py src/edge_grader/board_check.py tests/test_board_check.py
git commit -m "Boards: VARIANT (swafull, lg, lg-reuse) and RAG; board-check copes with the lg shim

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
scripts/deploy.sh jetson
```

---

### Task 2: find out what each variant actually does (Jetson, unmeasured)

**Files:** none changed. Findings go into Task 4's write-up, and a decision row goes into `docs/research-outline.md`.

**Interfaces:**
- Consumes: Task 1 deployed.
- Produces three answers:
  - **(a)** does `--swa-full` bring back prompt reuse?
  - **(b)** where is the little-gemma build with `-reuse`, and does `LG_ENGINE` need overriding?
  - **(c)** how does `-reuse` choose its prefix?

- [ ] **Step 1: `--swa-full` pilot** (all of test_ua, unmeasured, about 4 min each)

```bash
ssh MITLAB-JETSON 'cd ~/research/edge-grader &&
  MEASURE=0 EG_RUN_DIR=$PWD/results/local/pilot-base VARIANT=base    scripts/board_grade.sh e2b test_ua 0 &&
  MEASURE=0 EG_RUN_DIR=$PWD/results/local/pilot-swa  VARIANT=swafull scripts/board_grade.sh e2b test_ua 0 &&
  for d in pilot-base pilot-swa; do
    echo "== $d"; grep -c "selected slot by LCP" results/local/$d/server.log
    grep -oE "prompt eval time = +[0-9.]+ ms / +[0-9]+ tokens" results/local/$d/server.log | awk "{s+=\$(NF-1)} END {print \"tokens processed:\", s, \"over\", NR, \"requests\"}"
    grep -m2 -iE "swa|checkpoint|re-processing" results/local/$d/server.log
  done'
```

The whole split (`n` 0) is used on purpose. Any `n` > 0 takes a seeded shuffle (`data.load(..., seed=0)`), which would break the question grouping that reuse depends on.

**Expected if the hypothesis holds:** `pilot-swa` processes far fewer tokens than `pilot-base` (~90k).
- **If so:** `swafull` is the llama.cpp reuse control. Run it in Task 3.
- **If not:** the hypothesis is wrong. Drop `swafull`, write down the server log's own explanation, and treat the stage-1 baseline as the llama.cpp side.

Also compare the grades: `swa-full` must give the same labels as base, or very nearly. Use `uv run python -c` to diff the two raw files' `pred` by id.

- [ ] **Step 2: Find the `-reuse` build and read how it picks the prefix**

```bash
ssh MITLAB-JETSON 'ls -d ~/build/little-gemma* ~/build/*gemma* 2>/dev/null; for b in ~/build/little-gemma*/build/run-cuda-i8; do echo "== $b"; cat "$(dirname $b)/BUILD_INFO" 2>/dev/null | head -3; "$b" -h 2>&1 | grep -iE "reuse|ngram|block" ; done;
  cd ~/research/agentic-edge/benchmark && git status --short | head; grep -nE "VARIANT|ENGINE=|engine-flags" std_mmlupro_lg_jetson.sh'
```

**Record:**
- **Which binary accepts `-reuse`.** If it isn't `~/build/little-gemma/build/run-cuda-i8`, Task 3 sets `LG_ENGINE=<that path>`.
- **How the lab's uncommitted S9 script passes `-reuse`.** It must be `--engine-flags=-reuse`.

Then read the reuse logic in that build's source:

```bash
ssh MITLAB-JETSON 'cd <the build'"'"'s source dir> && grep -rn -iE "reuse|snapshot|prefix" --include=*.c --include=*.cu --include=*.h --include=*.patch . | head -40'
```

**The answer needed:** does `-reuse` keep **one** snapshot of a fixed prefix (e.g. taken from the first request), or follow the longest common prefix of consecutive requests?
- With one fixed snapshot, only the instructions (~15 tokens) are shared across all of test_ua. Expect almost no gain, and say so up front in the write-up.
- If it follows the prefix, the question and reference answer are shared too.

**Don't change little-gemma or agentic-edge's code.** If the answer makes `lg-reuse` pointless, still run it (Task 3) so the result is measured, and note the reason.

- [ ] **Step 3: Clean up and record**

```bash
ssh MITLAB-JETSON 'rm -rf ~/research/edge-grader/results/local/pilot-*'
```

Add an own-decisions row to `docs/research-outline.md` (2026-10-<dd>): "B2 llama.cpp control: `--swa-full` <does / doesn't> restore prompt reuse (pilot: <tokens processed> vs <tokens> for base). `-reuse` <follows / doesn't follow> changing prefixes (<one-line reason from the source>)". Commit:

```bash
git add docs/research-outline.md
git commit -m "B2 pilot: what --swa-full and little-gemma -reuse do with grading prompts

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: measured runs on the Jetson

**Interfaces:**
- Consumes: Task 2's decisions and `LG_ENGINE` path.
- Produces up to six run folders: `{lg,lg-reuse,swafull}-{e2b,e4b}-jetson-test_ua-*`. `swafull` only if Task 2 showed reuse.

- [ ] **Step 1: Run the list** (≈ 15–30 min in all; little-gemma adds E4B's warm-up)

```bash
ssh MITLAB-JETSON 'cd ~/research/edge-grader && nohup bash -c "
for m in e2b e4b; do
  for v in lg lg-reuse swafull; do
    echo == \$(date) \$m \$v
    BENCH=scientsbank VARIANT=\$v LG_ENGINE=<path from Task 2> scripts/board_grade.sh \$m test_ua
    echo == rc=\$?
    sleep 20
  done
done" > ~/b2-jetson.log 2>&1 < /dev/null &'
```

Drop `swafull` from the list if Task 2 ruled it out. Every `== rc=` must be 0. Each run's `board-state.txt` must name the right `variant` and `engine`, and its `server.log` must show the shim (`shim: request …`) or llama-server, as expected.

- [ ] **Step 2: Fetch, commit, board-check**

```bash
scripts/fetch_results.sh jetson
git add results/board
git commit -m "B2 runs: little-gemma, -reuse and the llama.cpp control on the Jetson (test_ua)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
uv run edge-grader-board-check results/board/lg-* results/board/swafull-* results/board/e*-jetson-test_ua-20260930-*
```

The last glob adds the stage-1 Jetson baseline runs to the same check file, so all variants sit side by side.

- [ ] **Step 3: Byte-identity and accuracy against the Jetson baseline**

```bash
uv run python - <<'EOF'
import json
from pathlib import Path
from edge_grader.board_check import compare, find_summary, outcomes, read_answers
B = Path("results/board")
for m in ("e2b", "e4b"):
    base = find_summary(max(B.glob(f"{m}-jetson-test_ua-20260930-*")))
    runs = {v: find_summary(max(B.glob(f"{v}-{m}-jetson-test_ua-*")))
            for v in ("lg", "lg-reuse", "swafull") if list(B.glob(f"{v}-{m}-jetson-test_ua-*"))}
    for v, s in runs.items():
        c = compare(outcomes(s), outcomes(base))
        print(f"{m} {v}: acc {c['first_accuracy']:.1%} vs llama.cpp {c['second_accuracy']:.1%}, "
              f"same label {c['same_grade']}/{c['n']}, McNemar p {c['mcnemar']['p']}")
    if {"lg", "lg-reuse"} <= runs.keys():
        a, b = (read_answers(runs[v]) for v in ("lg", "lg-reuse"))
        diff = [i for i in a if a[i].get("raw") != b.get(i, {}).get("raw")]
        print(f"{m} lg-reuse vs lg: {len(a) - len(diff)}/{len(a)} replies byte-identical")
EOF
```

**Pass conditions:**
- `lg-reuse` is 540/540 byte-identical to `lg`. If not, `lg-reuse` is reported as **failed** (not equal output), with its count.
- `lg` against llama.cpp: McNemar p ≥ 0.05 is expected (agentic-edge found the engines tied on MMLU-Pro).

---

### Task 4 (conditional): the RAG variants

Run this only if Task 3 shows a time saving for `lg-reuse` or `swafull` over its own base on the no-RAG prompt. RAG prompts share ~750 more tokens. **If there's no saving, skip this task** and say so in the write-up: RAG's longer shared prefix doesn't help an engine that can't reuse it.

- [ ] **Step 1: Put the RAG material on the Jetson** (not in git there: `data/` is ignored and `deploy.sh` skips `results/`)

```bash
ssh MITLAB-JETSON 'mkdir -p ~/research/edge-grader/data/rag/chunks ~/research/edge-grader/results/rag'
rsync -a data/rag/chunks/scientsbank-en.jsonl MITLAB-JETSON:research/edge-grader/data/rag/chunks/
rsync -a results/rag/retrieval-scientsbank-en.bge-small-en-k5.json MITLAB-JETSON:research/edge-grader/results/rag/
```

The runner checks the chunks' sha256 against the retrieval file and refuses a mismatch.

- [ ] **Step 2: Run, fetch, check**

```bash
ssh MITLAB-JETSON 'cd ~/research/edge-grader && nohup bash -c "
for m in e2b e4b; do for v in base lg-reuse swafull; do
  BENCH=scientsbank VARIANT=\$v RAG=results/rag/retrieval-scientsbank-en.bge-small-en-k5.json LG_ENGINE=<path> scripts/board_grade.sh \$m test_ua
  echo == rc=\$?; sleep 20
done; done" > ~/b2-rag-jetson.log 2>&1 < /dev/null &'
```

Run only the variants that Task 3 kept. `base` with `RAG` is the plain llama.cpp RAG run on the board. Then fetch, commit and board-check as in Task 3, Step 2, with the globs `results/board/rag-*`. Pair each RAG variant with `rag-e*-jetson-test_ua-*` (the base) in the same way as Task 3, Step 3.

---

### Task 5: results

**Files:** modify `docs/results/variations.md`, `docs/plan.md`, `docs/research-outline.md`.

- [ ] **Step 1: Write `## B2. little-gemma and prefix reuse (Jetson, test_ua)`** with:
- what was found in Task 2: reuse in llama.cpp, and how `-reuse` picks its prefix
- the table, one row per variant and model: accuracy, same label as llama.cpp, McNemar p, byte-identical (for `lg-reuse`), median s/answer, J/answer, mean prompt tokens processed (llama.cpp variants), and the window source
- the RAG table if Task 4 ran
- the Jetson clock state of every run compared; energy is compared only within one clock state
- three plain findings: is the engine a confound for grading, does reuse save time, and how much of that llama.cpp gets with its own flag

- [ ] **Step 2: Tick and commit**

```bash
git add docs/results/variations.md docs/plan.md docs/research-outline.md
git commit -m "B2 done: little-gemma and prefix reuse on the Jetson

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```
