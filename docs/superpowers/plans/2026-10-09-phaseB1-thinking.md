# Phase B1: grading with thinking on, implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Grade SciEntsBank with Gemma 4's thinking on, using `agentic-edge`'s thinking-on flags, on all three splits on Kaggle and on test_ua on both boards. Compare with the thinking-off baseline on accuracy and cost.

**Architecture:**
- **`edge-grader-baseline --thinking`:**
  - raises the answer budget
  - reads the label only from the text after the thought block
  - marks replies cut off at the budget, and thoughts that never close, as unparsed
  - names its files `think-*`
- **Serving:** `llama-server` with `-rea on --reasoning-budget 320 --reasoning-format none`, set by a `THINKING` switch in the Kaggle notebook and in `board_grade.sh`.
- **Board-check:** pairs a `think-*` board run with the Kaggle `think-*` run.

**Tech Stack:** Python 3 via uv, pytest, bash (board script), the Kaggle notebook.

**Spec:** [`docs/superpowers/specs/2026-10-08-baseline-variations-design.md`](../specs/2026-10-08-baseline-variations-design.md) §6 B1. **Depends on:** phase 0's Tasks 1–2 (`BENCH`, `DRY_RUN` and the generalised board-check) being merged.

## Global Constraints

**Serving:**
- Thinking-on flags, verbatim from `agentic-edge/benchmark/baselines.json` `mmlupro-thinking-on`: `-rea on --reasoning-budget 320 --reasoning-format none`.
- Everything else as the baseline: `-c 8192 --host 127.0.0.1 --port 8080 --cache-ram 0`; `-ngl 99` (Kaggle, Jetson); Pi `-ngl 0 -t 3`.
- `--reasoning-format none` is required, or the thoughts go to `reasoning_content` and the reply arrives empty (`agentic-edge` RESULTS.md §7.1).

**Prompt and parsing:**
- The grading prompt is unchanged (`grader.build_prompt`).
- With thinking on, the label is read **only from the text after the thought block** (own decision 2026-10-08).
- Answer budget 384 tokens: the 320-token thought budget plus 64.

**Cost and comparability:**
- MMLU-Pro thinking is **copied** from `agentic-edge`, not re-run.
- The Kaggle build stays `a3a1c47`; the boards stay on `661643e` / `a894dae`.

**Results and separation:**
- Every run writes a summary JSON to `results/`; never hand-edit.
- Board runs go through `scripts/board_grade.sh`, never agentic-edge's queue.

**Metrics:** accuracy and macro-F1; McNemar paired against the baseline per split.

**Commits:** end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **A thought block that names a label** ("…this isn't contradictory…") must not become the grade. Task 1: `test_thinking_reads_the_label_after_the_thought_block`.
2. **A thought that never closes** (agentic-edge saw 1–6 per 300), or a reply cut at the token budget, must count as unparsed, not as a label found inside the thoughts. Task 1: `test_unclosed_thought_or_cut_reply_is_unparsed`.
3. **The server started without thinking while the CLI says `--thinking`** (or the reverse) would give a run labelled wrong. The summary records `median_completion_tokens`, and the pilot checks it is far above 2. Task 1: `test_think_run_is_named_and_records_settings`; Task 2, Step 3.
4. **The real marker may differ from the candidate patterns.** The pilot reads real replies and pins the marker with a test built from one, before any full run. Task 2.
5. **The board-check must pair a board `think-*` run with the Kaggle `think-*` run**, not the thinking-off baseline. Task 3: `test_kaggle_run_pairs_think_with_think`.

---

## File map

| File | Change | Responsibility |
|---|---|---|
| `src/edge_grader/grader.py` | Modify | `THOUGHT_OPEN_RE`/`THOUGHT_CLOSE_RE`, `answer_part()`, `grade(..., thinking=False)` |
| `src/edge_grader/baseline.py` | Modify | `--thinking`; `think-*` names; settings and `median_completion_tokens` |
| `tests/test_grader.py`, `tests/test_baseline_e2e.py` | Modify | Tests |
| `notebooks/kaggle_baseline.ipynb` | Modify (cells `c01`, `c11`) | `THINKING` switch |
| `scripts/board_grade.sh`, `tests/test_board_grade.py` | Modify | `THINKING=on` |
| `src/edge_grader/board_check.py`, `tests/test_board_check.py` | Modify | `think-*` summaries; pair with Kaggle `think-*` |
| `docs/results/variations.md` | Create | B1 results |
| `docs/plan.md`, `docs/research-outline.md` | Modify | Ticks, RQ6 line |

---

### Task 1: the grader and CLI understand thinking

**Files:**
- Modify: `src/edge_grader/grader.py`, `src/edge_grader/baseline.py`
- Test: `tests/test_grader.py`, `tests/test_baseline_e2e.py`

**Interfaces:**
- Produces:
  - `grader.THINK_MAX_TOKENS = 384`
  - `grader.answer_part(text: str) -> str | None`: the text after the last thought-close marker. It returns the whole text when there's no thought block, and **None** when a thought opens and never closes.
  - `grader.grade(client, model, item, labels, max_tokens=16, passages=None, thinking=False) -> dict`. With `thinking=True`:
    - `max_tokens` defaults to `THINK_MAX_TOKENS` when the caller leaves 16
    - `pred` comes from `parse_label(answer_part(text))`
    - `finish == "length"` gives `pred` None
    - the row gains `"finish"`
  - `baseline` CLI `--thinking`: run name `think-<label>-<split>-<scheme>-<stamp>`, `"stage": "think"`, `"thinking": {"server_flags": "-rea on --reasoning-budget 320 --reasoning-format none", "max_tokens": 384, "parse": "label after the thought block"}`, and `metrics.median_completion_tokens`. It refuses `--rag` together with `--thinking`, which isn't a condition in the spec.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_grader.py`:

```python
from edge_grader.grader import THINK_MAX_TOKENS, answer_part, grade


def test_answer_part_without_a_thought_block_is_the_whole_reply():
    assert answer_part("correct") == "correct"


def test_thinking_reads_the_label_after_the_thought_block():
    for opened, closed in (("<|channel>thought\n", "<channel|>"), ("<think>", "</think>")):
        reply = f"{opened}It is not contradictory; maybe incorrect.{closed}correct"
        assert answer_part(reply) == "correct"
        assert parse_label(answer_part(reply), LABELS) == "correct"


def test_unclosed_thought_or_cut_reply_is_unparsed():
    assert answer_part("<|channel>thought\nIt is incorrect because") is None

    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    class R:
                        class usage:
                            prompt_tokens, completion_tokens = 100, 384
                        choices = [type("C", (), {"finish_reason": "length",
                                   "message": type("M", (), {"content": "<think>x</think>incor"})})]
                    assert kw["max_tokens"] == THINK_MAX_TOKENS
                    return R

    item = Item("1", "Q?", "ref", "ans", "correct")
    row = grade(Client, "m", item, LABELS, thinking=True)
    assert row["pred"] is None and row["finish"] == "length"
```

Then, in `tests/test_baseline_e2e.py`, make the fake server answer with a thought block when the request asks for a long answer. In `FakeLlamaServer.do_POST`, replace the `reply = …` line with:

```python
        reply = "correct" if "Student answer: right" in prompt else "incorrect"
        if body.get("max_tokens", 16) > 16:  # a thinking-on request: thoughts first, naming labels
            reply = f"<|channel>thought\nIs it correct or contradictory?<channel|>{reply}"
```

Then append:

```python
def test_think_run_is_named_and_records_settings(server, tmp_path, monkeypatch):
    monkeypatch.setattr(data, "load", lambda *a, **k: ITEMS)
    baseline.main(["--base-url", server, "--label", "fake", "--n", "0", "--out", str(tmp_path),
                   "--thinking"])
    (f,) = tmp_path.glob("think-fake-test_ua-3way-*.json")
    run = json.loads(f.read_text())
    assert run["stage"] == "think"
    assert run["thinking"]["server_flags"] == "-rea on --reasoning-budget 320 --reasoning-format none"
    assert run["metrics"]["accuracy"] == 2 / 3  # "right" -> correct, others -> incorrect
    assert run["metrics"]["unparsed"] == 0
    assert "median_completion_tokens" in run["metrics"]


def test_thinking_and_rag_together_are_refused(tmp_path):
    with pytest.raises(SystemExit):
        baseline.main(["--out", str(tmp_path), "--thinking", "--rag", "x.json"])
```

The existing test `test_baseline_writes_summary_and_raw` patches `data.load` the same way. Copy its `monkeypatch.setattr` line exactly if it differs from the one above.

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_grader.py tests/test_baseline_e2e.py -v`
Expected: `ImportError: cannot import name 'THINK_MAX_TOKENS'`, and the e2e test fails on `--thinking`.

- [ ] **Step 3: Implement in `grader.py`**

Add below `LABEL_HELP`:

```python
THINK_MAX_TOKENS = 384  # the 320-token thought budget (agentic-edge's) plus room for the label
# Thought-block markers as llama-server returns them with --reasoning-format none. Gemma 4's
# channel markers first; <think> as a fallback. The pilot (phase B1 Task 2) pins the real one.
THOUGHT_OPEN_RE = re.compile(r"<\|channel>thought|<think>")
THOUGHT_CLOSE_RE = re.compile(r"<channel\|>|</think>")


def answer_part(text: str) -> str | None:
    """The reply after its thought block: all of it if there's no block, None if a thought
    opened and never closed (the grade would otherwise be read from the thoughts)."""
    closes = list(THOUGHT_CLOSE_RE.finditer(text))
    if closes:
        return text[closes[-1].end():]
    return None if THOUGHT_OPEN_RE.search(text) else text
```

Replace `grade` with:

```python
def grade(client, model: str, item: Item, labels: list[str], max_tokens: int = 16,
          passages: list[str] | None = None, thinking: bool = False) -> dict:
    prompt = build_prompt(item, labels, passages)
    if thinking and max_tokens == 16:
        max_tokens = THINK_MAX_TOKENS
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - start
    choice = resp.choices[0]
    text = choice.message.content or ""
    if thinking:
        part = None if choice.finish_reason == "length" else answer_part(text)
        pred = parse_label(part, labels) if part is not None else None
    else:
        pred = parse_label(text, labels)
    usage = resp.usage
    row = {
        "id": item.id,
        "gold": item.label,
        "pred": pred,
        "raw": text,
        "latency_s": round(latency, 3),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
    }
    if thinking:
        row["finish"] = choice.finish_reason
    return row
```

Thinking-off rows are byte-for-byte what they were (no `finish` key, same `pred`).

- [ ] **Step 4: Implement in `baseline.py`**

1. Add the argument: `ap.add_argument("--thinking", action="store_true", help="the server runs with -rea on --reasoning-budget 320 --reasoning-format none: read the label after the thoughts")`.
2. After `args = ap.parse_args(argv)`, add `if args.thinking and args.rag: sys.exit("--thinking with --rag isn't a condition in this study")`.
3. Change `stage = "rag" if args.rag else "baseline"` to `stage = "think" if args.thinking else ("rag" if args.rag else "baseline")`.
4. In the grading call, pass `thinking=args.thinking`.
5. After `summary["completion_tokens"] = …`, add:

```python
    summary["median_completion_tokens"] = (
        statistics.median(r["completion_tokens"] or 0 for r in ok) if ok else None)
```

6. After `if rag: run["rag"] = rag`, add:

```python
    if args.thinking:
        run["thinking"] = {"server_flags": "-rea on --reasoning-budget 320 --reasoning-format none",
                           "max_tokens": grader.THINK_MAX_TOKENS,
                           "parse": "label after the thought block; unclosed or cut replies unparsed"}
```

- [ ] **Step 5: Run the tests and lint**

Run: `uv run pytest -q && uv run ruff check .`
Expected: all PASS (thinking-off tests unchanged), ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/edge_grader/grader.py src/edge_grader/baseline.py tests/test_grader.py tests/test_baseline_e2e.py
git commit -m "Grading with thinking on: label read after the thought block; think-* runs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Kaggle switch and the pilot that pins the marker

**Files:**
- Modify: `notebooks/kaggle_baseline.ipynb` (cells `c01`, `c11`); `src/edge_grader/grader.py` and `tests/test_grader.py` only if the marker differs
- Add: `tests/fixtures/thinking_reply.json` (one real reply)

**Interfaces:**
- Consumes: Task 1.
- Produces: `THINKING = True` in the notebook starts the server with the thinking flags and passes `--thinking`. A committed real reply pins the marker.

- [ ] **Step 1: Add the switch** (edit script; same pattern as phase A Task 5)

Save as `$SCRATCH/edit_think_nb.py` and run it with `uv run python $SCRATCH/edit_think_nb.py notebooks/kaggle_baseline.ipynb`:

```python
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
nb = json.loads(path.read_text())


def cell_with(text):
    (c,) = [c for c in nb["cells"] if text in "".join(c["source"])]
    return c


def swap(cell, old, new):
    s = "".join(cell["source"])
    assert s.count(old) == 1, old
    cell["source"] = s.replace(old, new).splitlines(keepends=True)


c01 = cell_with("# What to run.")
swap(c01, "RAG = None\n",
     "RAG = None\n"
     "# Phase B1: thinking on (agentic-edge's -rea on --reasoning-budget 320 --reasoning-format none).\n"
     "# scientsbank only, no RAG. Files are named think-*.\n"
     "THINKING = False\n")
c11 = cell_with("def start_server(key):")
swap(c11, '           "-rea", "off", "--reasoning-budget", "-1", "--cache-ram", "0"]',
     '           *(["-rea", "on", "--reasoning-budget", "320", "--reasoning-format", "none"] if THINKING\n'
     '             else ["-rea", "off", "--reasoning-budget", "-1"]),\n'
     '           "--cache-ram", "0"]')
swap(c11, "for key in MODELS:\n",
     'assert not (THINKING and (RAG or TASK != "scientsbank")), "thinking: scientsbank, no RAG"\n'
     'THINK_ARGS = "--thinking" if THINKING else ""\n'
     "for key in MODELS:\n")
swap(c11, "--split {split} --n {N} --out results/kaggle {RAG_ARGS}",
     "--split {split} --n {N} --out results/kaggle {RAG_ARGS} {THINK_ARGS}")
path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
```

Commit and push (the notebook clones the repo from GitHub):

```bash
git add notebooks/kaggle_baseline.ipynb
git commit -m "Kaggle notebook: THINKING switch (phase B1)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

- [ ] **Step 2: Pilot (the owner, Kaggle)**

Set `TASK = "scientsbank"`, `THINKING = True`, `MODELS = ["e2b", "e4b"]`, `SPLITS = ["test_ua"]`, `N = 20`, and run all. Download `results-kaggle.zip` and unzip it into a **scratch directory**, not `results/`. Pilot files aren't committed.

- [ ] **Step 3: Read the pilot**

```bash
uv run python - "$SCRATCH/pilot" <<'EOF'
import json, sys
from pathlib import Path
for f in sorted(Path(sys.argv[1]).glob("raw/think-*.jsonl")):
    rows = [json.loads(x) for x in f.read_text().splitlines()]
    print(f.name, "unparsed", sum(r["pred"] is None for r in rows), "of", len(rows),
          "finish", {r.get("finish") for r in rows})
    for r in rows[:2]:
        print(repr(r["raw"][:120]), "...", repr(r["raw"][-60:]))
EOF
```

Check:
- **(a)** `median_completion_tokens` in the summaries is well above 2 (thoughts are being generated). If it's about 2, the server didn't think. Stop and check the `start_server` command in `/tmp/server-*.log`.
- **(b)** The printed replies show the real open and close markers.
- **(c)** Unparsed is a handful at most.

- [ ] **Step 4: Pin the marker with a real reply**

Copy one pilot reply that has a thought block into `tests/fixtures/thinking_reply.json`, as `{"source": "<pilot file name>", "raw": "<the reply>", "label": "<what the owner reads as its final label>"}`. It's model output for a public dataset item, so it can be committed. Append to `tests/test_grader.py`:

```python
def test_a_real_thinking_reply_parses_to_its_final_label():
    rec = json.loads((Path(__file__).parent / "fixtures" / "thinking_reply.json").read_text())
    assert parse_label(answer_part(rec["raw"]), LABELS) == rec["label"]
```

(add `import json` and `from pathlib import Path` at the top if missing).

Run: `uv run pytest tests/test_grader.py -v`.
- **If it fails**, the real markers differ from `THOUGHT_OPEN_RE`/`THOUGHT_CLOSE_RE`. Add the real ones to the front of both patterns, keep the candidates, and re-run until it passes. Then re-score the pilot rows with the new parser (Step 3's script, re-parsing `raw`) and confirm unparsed stays a handful.
- **If it passes**, the candidates were right.

```bash
git add tests/fixtures/thinking_reply.json tests/test_grader.py src/edge_grader/grader.py
git commit -m "Thinking: pin the thought-block marker with a real reply from the pilot

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

- [ ] **Step 5: Full runs (the owner, Kaggle)**

Set `SPLITS = ["test_ua", "test_uq", "test_ud"]`, `N = 0`, `THINKING = True`, and run.

**Time:** about 384 tokens per answer against 2, so plan for **~6–10 h per model** on a T4 for test_ud's 4,562 answers. Split it across sessions if needed: first `SPLITS = ["test_ua", "test_uq"]`, then `["test_ud"]`.

Unzip into `results/kaggle` and commit:

```bash
unzip -o ~/Downloads/results-kaggle.zip -d results/kaggle
git add results/kaggle
git commit -m "Thinking on: SciEntsBank all splits, E2B + E4B (Kaggle)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 3: boards (test_ua, cost)

**Files:**
- Modify: `scripts/board_grade.sh`, `tests/test_board_grade.py`, `src/edge_grader/board_check.py`, `tests/test_board_check.py`

**Interfaces:**
- Consumes:
  - phase 0's `board_grade.sh`: `server_cmd()`, `runner()`, `NAME`, `DRY_RUN`
  - phase 0's `board_check`: `SUMMARY_GLOBS`, `kaggle_run(directory, bench, model, split, n)`
- Produces:
  - `THINKING=on` env makes `server_cmd` use the thinking flags, makes the SciEntsBank runner pass `--thinking`, and names folders `think-<model>-<board>-<split>-<stamp>`
  - `board_check.SUMMARY_GLOBS` gains `"think-*.json"`
  - `kaggle_run(..., thinking: bool = False)` looks up `think-{model}-kaggle-*-{split}-*.json` when true
  - `read_run` adds `"thinking": bool`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_board_grade.py`:

```python
def test_thinking_on_changes_only_the_reasoning_flags_and_the_name():
    off = fields(dry("e2b", board="pi5"))
    on = fields(dry("e2b", board="pi5", THINKING="on"))
    assert on["server"] == off["server"].replace(
        "-rea off --reasoning-budget -1", "-rea on --reasoning-budget 320 --reasoning-format none")
    assert re.fullmatch(rf"{REPO}/results/board/think-e2b-pi5-test_ua-{STAMP}", on["run_dir"])
    assert on["runner"].endswith(f"--out {on['run_dir']} --thinking")


def test_thinking_is_scientsbank_only():
    p = dry("e2b", BENCH="gsm", THINKING="on")
    assert p.returncode != 0 and "THINKING" in p.stderr
```

Also add `"THINKING"` to the set of variables `dry()` strips from the environment.

Append to `tests/test_board_check.py`:

```python
def test_kaggle_run_pairs_think_with_think(tmp_path):
    for n in ("baseline-e2b-kaggle-t4-test_ua-3way-1.json", "think-e2b-kaggle-t4-test_ua-3way-1.json"):
        (tmp_path / n).write_text(json.dumps({"metrics": {"n": 540}}))
    d = str(tmp_path)
    assert kaggle_run(d, "scientsbank", "e2b", "test_ua", 540).name.startswith("baseline-")
    assert kaggle_run(d, "scientsbank", "e2b", "test_ua", 540, thinking=True).name.startswith("think-")
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_board_grade.py tests/test_board_check.py -v`
Expected: the new tests FAIL.

- [ ] **Step 3: Implement in `board_grade.sh`**

After the `INSTANCES` check, add:

```bash
THINKING="${THINKING:-off}"
case "$THINKING" in
  off) ;;
  on) [ "$BENCH" = scientsbank ] || { echo "THINKING=on is for BENCH=scientsbank only" >&2; exit 1; } ;;
  *) echo "THINKING must be on or off: $THINKING" >&2; exit 1 ;;
esac
```

In the `NAME` block, change the SciEntsBank branch to:

```bash
if [ "$BENCH" = scientsbank ]; then
  NAME="$MODEL_KEY-$BOARD-$SPLIT"
  [ "$THINKING" = on ] && NAME="think-$NAME"
```

In `server_cmd`, replace the second `echo` argument with:

```bash
       "$( [ "$THINKING" = on ] && echo '-rea on --reasoning-budget 320 --reasoning-format none' \
           || echo '-rea off --reasoning-budget -1' ) --cache-ram 0"
```

In `runner`, change the SciEntsBank line's end to `--out "$RUN_DIR" $( [ "$THINKING" = on ] && echo --thinking )`.

Pass `THINKING="$THINKING"` in the inner `exec env …` line, next to `BENCH` and `INSTANCES`.

- [ ] **Step 4: Implement in `board_check.py`**

1. Add `"think-*.json"` to `SUMMARY_GLOBS`.
2. In `kaggle_run`, add the parameter `thinking: bool = False`, and in the SciEntsBank branch use `prefix = "think" if thinking else "baseline"` with the glob `f"{prefix}-{model}-kaggle-*-{split}-*.json"`.
3. In `read_run`, add `"thinking": summary.get("stage") == "think",` to the returned dict.
4. In `main`, pass `thinking=r["thinking"]` to `kaggle_run`.

- [ ] **Step 5: Tests, lint, commit, deploy**

Run: `uv run pytest -q && uv run ruff check . && bash -n scripts/board_grade.sh`
Expected: all PASS.

```bash
git add scripts/board_grade.sh tests/test_board_grade.py src/edge_grader/board_check.py tests/test_board_check.py
git commit -m "Boards: THINKING=on for grading; board-check pairs think runs with Kaggle think runs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
scripts/deploy.sh
```

- [ ] **Step 6: Run on both boards** (after phase 0 has passed its gate; one job per board)

```bash
ssh MITLAB-JETSON 'cd ~/research/edge-grader && nohup bash -c "for m in e2b e4b; do BENCH=scientsbank THINKING=on scripts/board_grade.sh \$m test_ua; echo == rc=\$?; sleep 20; done" > ~/b1-jetson.log 2>&1 < /dev/null &'
ssh MITLAB-EDGE   'cd ~/Research/edge-grader && nohup bash -c "for m in e2b e4b; do BENCH=scientsbank THINKING=on scripts/board_grade.sh \$m test_ua; echo == rc=\$?; sleep 20; done" > ~/b1-pi.log 2>&1 < /dev/null &'
```

**Pi time:** about 540 × 384 tokens at ~16 tok/s for E2B and ~8 for E4B, so ~4 h and ~7 h.

When both are done:

```bash
scripts/fetch_results.sh
git add results/board
git commit -m "Thinking on, test_ua, both boards

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
uv run edge-grader-board-check results/board/think-*
```

Expected: each run is paired with its Kaggle `think-*` run, with McNemar p ≥ 0.05.

---

### Task 4: results

**Files:**
- Create: `docs/results/variations.md`
- Modify: `docs/plan.md`, `docs/research-outline.md`

- [ ] **Step 1: Paired comparison against the baseline, per split**

```bash
uv run python - <<'EOF'
from pathlib import Path
from edge_grader.board_check import compare, outcomes
import json
K = Path("results/kaggle")
for model in ("e2b", "e4b"):
    for split in ("test_ua", "test_uq", "test_ud"):
        base = max(K.glob(f"baseline-{model}-kaggle-t4-{split}-3way-20261005-*.json"))  # build a3a1c47
        think = max(K.glob(f"think-{model}-kaggle-t4-{split}-3way-*.json"))
        c = compare(outcomes(think), outcomes(base))
        tb, tt = (json.loads(p.read_text())["metrics"] for p in (base, think))
        print(f"{model} {split}: {c['second_accuracy']:.1%} -> {c['first_accuracy']:.1%} "
              f"(F1 {tb['macro_f1']:.3f} -> {tt['macro_f1']:.3f}), better/worse "
              f"{c['mcnemar']['only_first_right']}/{c['mcnemar']['only_second_right']}, p {c['mcnemar']['p']}, "
              f"unparsed {tt['unparsed']}, completion tokens {tb['completion_tokens']} -> {tt['completion_tokens']}")
EOF
```

- [ ] **Step 2: Write `docs/results/variations.md`**

Start the page like `rag.md`: a title, one line saying what it holds (phase B, RQ6), and a link to the spec. Then add `## B1. Thinking on`:
- the setup (flags, budget, parser)
- the per-split table from Step 1
- the board cost table from the board-check: J/answer and s/answer, thinking on against off on the same board
- MMLU-Pro's thinking result, copied with its source (`docs/results/baselines.md` §1, agentic-edge `findings/RESULTS.md` §7.2)
- three plain findings, including whether grading behaves differently from MMLU-Pro

Link the page from `docs/README.md` if that file lists the results pages.

- [ ] **Step 3: Tick and commit**

- `docs/plan.md`: tick B1.
- `docs/research-outline.md`: in the stages table's B row, add `B1 done <date>: <one line>`.

```bash
git add docs/results/variations.md docs/plan.md docs/research-outline.md docs/README.md
git commit -m "B1 done: grading with thinking on, Kaggle all splits and board cost

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```
