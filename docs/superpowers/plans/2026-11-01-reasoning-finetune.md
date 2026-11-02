# Reasoning-guided fine-tuning Implementation Plan

> **For agentic workers:** steps use checkbox (`- [ ]`) syntax for tracking. Do the tasks in order; Tasks 1–5 are code on the Mac (no models), Tasks 6–9 are Kaggle runs and analysis.

**Goal:** Qwen3-14B writes a reason for every SciEntsBank train answer; two E2B QLoRA adapters are trained on them (reason-first, label-first); both are graded on Beetle, SAF and SciEntsBank test_uq/test_ud, and compared with the untuned E2B and the Oct 28 label-only adapter.

**Architecture:** The reply format is one idea that runs through three places, so it lives in `grader.py`: `REPLY_FORMATS`, how a reply is written for training (`reply_text`) and how it's read back (`parse_reply`). `finetune.py` and `baseline.py` each get a `--reply` option that uses them. A new module `reasons.py` holds the teacher prompt, the filter and the `edge-grader-reasons` CLI (resumable, like the news judge). A new notebook `kaggle_reasons.ipynb` serves Qwen3-14B and runs it; the existing fine-tune and baseline notebooks get a `REPLY` setting. A small `compare.py` holds the McNemar/Holm comparison and the §2 gate, so the gate is code fixed before any result.

**Tech Stack:** as stage 3: Python 3.12 via uv, the `finetune` group (versions pinned by `uv.lock`: torch 2.14.1, transformers 5.18.0, peft 0.21.2, bitsandbytes 0.50.2), llama.cpp `a3a1c47` on Kaggle.

**Spec:** [`docs/superpowers/specs/2026-11-01-reasoning-finetune-design.md`](../specs/2026-11-01-reasoning-finetune-design.md)

## Global Constraints

- Everything from the [stage 3 plan](2026-10-21-stage3-finetuning.md) Global Constraints still holds (base checkpoints, GGUFs, server flags + `--lora`, 3-way labels, the 90/10 split by question with seed 0, the QLoRA recipe).
- The grading prompt is `grader.build_prompt(item, labels)`, **unchanged** for every format.
- Reply formats (exact text, `<r>` = the reason, `<l>` = the label):
  - `label`: `<l>` (the Oct 28 adapters; the default everywhere, so old runs and callers are unchanged)
  - `reason-first`: `<r>\nLabel: <l>`
  - `label-first`: `<l>\n<r>`
- Teacher: `Qwen/Qwen3-14B-GGUF` / `Qwen3-14B-Q4_K_M.gguf`, the Gemma flags, `-rea off`, temperature 0, `max_tokens` 128.
- Reasons file: `results/finetune/raw/reasons-qwen3-14b-q4_k_m.jsonl`, one `{"id", "gold", "reason", "drop"}` per train answer (`drop` is null or the reason code). Committed, small (about 5k short lines); its sha256 goes in the reasons summary and in each training summary. *(Deviation from spec §5.1's "private Kaggle dataset": the reasons are derived from public data and small, and the code bundle already reaches Kaggle. Spec updated in Task 0.)*
- Run labels: adapters `e2b-ft-rf` (reason-first) and `e2b-ft-lf` (label-first); training summaries `finetune-e2b-<reply>-<stamp>.json`.
- No test split, Beetle or SAF answer is looked at before Task 9, and nothing in Tasks 6–8 is tuned on them.

## Review Focus

1. **Label read from the reason.** Reason-first replies contain words like "correct". `parse_reply` must read only after the last `Label:`; label-first only the first line. Tests (Task 1).
2. **Training text ≠ served text.** The reply is trained inside the same chat template check as Oct 28 (`full.startswith(prompt)`), and the label-first server request stops at `\n`, the same boundary the training reply has. Test (Task 1, Task 3).
3. **The two adapters train on different answers.** Both must drop exactly the training answers whose reason has a `drop` code. Test (Task 3).
4. **Old runs change meaning.** `--reply label` must give byte-identical prompts, `max_tokens` and file names to today's runs. Tests (Tasks 1, 4).
5. **The gate written after the result.** `compare.gate` and its test land in Task 5, before any grading run.

---

### Task 0: Record the decision

**Files:** `docs/research-outline.md`, `docs/plan.md`, the spec.

- [ ] Outline: a row in the own-decisions table (2026-11-01): reasoning-guided QLoRA, Qwen3-14B teacher, two orders, SciEntsBank only, E2B first, alongside stage 4; RQ3. A feedback-table row for the supervisor's request (look for published work that makes Gemma grade better), with the answer: Faza et al. 2025.
- [ ] `plan.md`: under week 8 (Nov 2–8), add the Kaggle pilot beside stage 4, and the Nov 15 end in week 9.
- [ ] Spec §5.1: the reasons file is committed under `results/finetune/raw/` (see Global Constraints).
- [ ] Commit with the spec and this plan.

### Task 1: Reply formats in `grader.py`

**Files:** modify `src/edge_grader/grader.py`; test `tests/test_grader.py` (create if absent; otherwise add to it).

**Produces:**
- `REPLY_FORMATS = ("label", "reason-first", "label-first")`
- `REASON_FIRST_MAX_TOKENS: int` (160 for now; Task 6 sets the measured value)
- `reply_text(label: str, reason: str | None, fmt: str) -> str` (the Global Constraints formats; `reason` required unless `fmt == "label"`)
- `parse_reply(text: str, labels: list[str], fmt: str) -> str | None`:
  - `label`: `parse_label(text, labels)` (unchanged behaviour)
  - `reason-first`: `parse_label` on the text after the **last** `Label:` (case-insensitive); none → `None`
  - `label-first`: `parse_label` on the first line only
- `grade(..., reply: str = "label")`: `label-first` sends `max_tokens=16, stop=["\n"]`; `reason-first` sends `max_tokens=REASON_FIRST_MAX_TOKENS` and a reply with `finish_reason == "length"` is unparsed; the row records `"finish"` for both reason formats. `thinking` and a reply other than `label` together raise `ValueError`.

- [ ] **Tests first:**
  - `parse_reply("The answer names the wrong force, so it is not correct.\nLabel: contradictory", ..., "reason-first") == "contradictory"`
  - reason-first with no `Label:` → `None`; with two `Label:` lines → the last one
  - `parse_reply("incorrect\nIt leaves out ... correct idea ...", ..., "label-first") == "incorrect"`
  - `reply_text("correct", "r", "reason-first") == "r\nLabel: correct"`, `reply_text("correct", "r", "label-first") == "correct\nr"`, `reply_text("correct", None, "label") == "correct"`
  - `parse_reply(reply_text(l, r, f), labels, f) == l` for every format and label, with `r` containing all three label words
  - `grade` with a fake client: `label` sends `max_tokens=16` and no `stop` (today's request, unchanged); `label-first` sends `stop=["\n"]`; `reason-first` cut by length → `pred is None`
- [ ] Run `uv run pytest tests/test_grader.py -v`: fail, then implement, then pass.
- [ ] `uv run pytest -q && uv run ruff check .`; commit.

### Task 2: The teacher (`reasons.py`)

**Files:** create `src/edge_grader/reasons.py`, `src/edge_grader/prompts/reason_teacher.txt`, `tests/test_reasons.py`; modify `pyproject.toml` (`edge-grader-reasons = "edge_grader.reasons:main"`).

**Produces:**
- `teacher_messages(item: Item) -> list[dict]`: one user turn from the prompt file, filled with question, reference, student answer and the gold label with its `grader.LABEL_HELP` text. The prompt asks for 1–3 sentences, at most 60 words, comparing the answer with the reference (what it gets right, what is missing or contradicts it), **without using the words correct, incorrect or contradictory**.
- `check_reason(text: str, finish: str, gold: str, labels: list[str]) -> str | None`, the drop code or `None`: `"empty"` (blank after strip), `"cut"` (`finish == "length"`), `"wrong_label"` (a label word other than `gold` appears; uses the same `\b` matching as `parse_label`, so "incorrect" doesn't count as "correct").
- `main(argv)`: `--base-url`, `--n` (0 = all; a pilot uses 50), `--out` (default `results/finetune`). Loads `data.load(split="train", scheme="3way")`, writes one JSON line per answer to `raw/reasons-<name>.jsonl.part` as it goes, skips ids already in the `.part` file (resume), renames to `.jsonl` when complete, then writes `reasons-<name>-<stamp>.json`: teacher `server_info`, prompt sha256, n, drop counts by code, drop share, the JSONL's sha256, wall time. The name comes from the server's model path (`qwen3-14b-q4_k_m`).

- [ ] **Tests first** (no network: a fake client, as in `test_baseline_e2e.py`):
  - `check_reason` for each code, and `None` for a clean reason; a reason saying "incorrect" for gold `incorrect` is kept; "correct" inside "incorrect" for gold `incorrect` is kept
  - `teacher_messages` includes the gold label's help text and the three answer fields
  - a run of 3 items against a fake server writes 3 lines and a summary with the drop counts; a second run with a `.part` holding 2 lines asks the server only once
- [ ] Implement; `uv run pytest -q && uv run ruff check .`; commit.

### Task 3: `finetune.py` trains a reply format

**Files:** modify `src/edge_grader/finetune.py`, `tests/test_finetune.py`.

**Changes:**
- `to_messages(item, labels, reply="label", reason=None)`: the assistant content is `grader.reply_text(item.label, reason, reply)`.
- `load_reasons(path: Path) -> tuple[dict[str, str], str]`: id → reason for lines with `drop` null, and the file's sha256.
- `main`: `--reply` (choices `grader.REPLY_FORMATS`, default `label`) and `--reasons` (required unless `--reply label`). Training items without a kept reason are dropped after the split (the split is computed on all train items, so the questions match Oct 28). Validation keeps every answer, as on Oct 28: it reads only the gold label, so a dropped reason doesn't matter there, and the epoch choice stays comparable. The count is printed and recorded.
- Validation reads each prediction with `parse_reply(..., args.reply)` and generates up to `REASON_FIRST_MAX_TOKENS` for `reason-first`; for `label-first` it cuts the decoded text at the first `\n` (as the server's stop does).
- Summary `settings` gains `reply`, `reasons` path, `reasons_sha256`, `dropped_no_reason`; the file is `finetune-<model>-<reply>-<stamp>.json` when `reply != "label"` (unchanged name otherwise).
- `--dry-run` prints one example per format it would train.

- [ ] **Tests first:**
  - the split is identical for `label` and both reason formats (same validation questions as `split_by_question` on all items)
  - with a reasons file where 2 ids are dropped, both reason formats train on the same set, and that set excludes those 2
  - `to_messages(..., "label")` equals today's output (regression)
  - `load_reasons` skips dropped lines and returns the sha256 of the file bytes
- [ ] Implement; `uv run pytest -q && uv run ruff check .`; commit.

### Task 4: `baseline.py --reply`

**Files:** modify `src/edge_grader/baseline.py`, `tests/test_baseline_e2e.py`.

- `--reply` (choices `grader.REPLY_FORMATS`, default `label`), passed to `grader.grade`. `--reply` other than `label` with `--thinking` or `--rag` exits.
- The run records `"reply": args.reply`; `summary["unparsed"]` already counts `None` preds; add `summary["cut"]` (rows with `finish == "length"`) for reason formats.
- File names are unchanged; the label carries the adapter (`e2b-kaggle-t4-ft-rf`), set by the notebook.

- [ ] **Tests:** extend `FakeLlamaServer` so a request with `max_tokens > 16` and no thinking returns `"Some reason.\nLabel: correct"` style replies, and one with `stop` returns only the label line:
  - `--reply reason-first` scores like the label run (2/3) and records `reply`
  - `--reply label-first` sends `stop`
  - `--reply reason-first --thinking` exits
- [ ] `uv run pytest -q && uv run ruff check .`; commit.

### Task 5: The comparison and the gate (`compare.py`)

**Files:** create `src/edge_grader/compare.py`, `tests/test_compare.py`.

- Reuse `edge_grader.board_check.mcnemar`, and `board_check.compare` / `read_answers` for the join if they fit (read them first). Holm is a few lines here: `board_check` has none.
- `paired(a_raw: Path, b_raw: Path, two_way: bool) -> dict`: joins two raw JSONL files by id (fails if the id sets differ), gives n, both accuracies, the difference and the McNemar p. 2-way maps every non-`correct` to `incorrect` first.
- `gate(cells: dict[tuple[str, str], dict[str, dict]]) -> tuple[bool, list]`: `cells[(set, split)]["vs_base"]` and `["vs_ft"]` hold `paired` results with Holm-corrected `p_holm`. True if **some** cell has both differences > 0 and both `p_holm < 0.05` (spec §2). Holm runs over all 16 tests (2 adapters × 2 baselines × 4 cells) at once.
- `main`: given the run labels, finds the newest raw files per (dataset, split, label) under `results/kaggle/raw/`, prints the table (Beetle 3-way, SAF 2-way; SciEntsBank and per-class recall described, not tested) and the gate, and writes `results/kaggle/compare-reason-ft-<stamp>.json`.

- [ ] **Tests:** gate true only when one cell wins against both; false when the two wins are in different cells; false when a win is positive but `p_holm ≥ 0.05`; `paired` refuses mismatched ids; 2-way mapping.
- [ ] Commit **before** Task 7 starts.

### Task 6: Notebooks, then the reasons run (Kaggle)

**Files:** create `notebooks/kaggle_reasons.ipynb`; modify `notebooks/kaggle_finetune.ipynb`, `notebooks/kaggle_baseline.ipynb`.

- [x] `kaggle_reasons.ipynb`: copy the news-judge notebook's structure (build `a3a1c47`, download the teacher, get the code, start `llama-server` with the Gemma flags and `-rea off`), then `edge-grader-reasons` with `N` (50 for the pilot, 0 for all), then zip `results/finetune/reasons-*` and `raw/reasons-*` for download. Get the code the way the Oct 30 held-out runs did (the `edge-grader-code` dataset bundle).
- [x] `kaggle_finetune.ipynb`: `REPLY = "label"` setting in cell 1, passed as `--reply {REPLY} --reasons {REPO}/results/finetune/raw/reasons-qwen3-14b-q4_k_m.jsonl` when not `label`; the adapter GGUF is named `{MODEL}-ft-{rf|lf}.gguf`. The Gemma token check from spec §5.3 runs as a cell before training when `REPLY == "reason-first"`: the longest training reply in Gemma's tokenizer, plus 8; if that is above `REASON_FIRST_MAX_TOKENS`, stop and update the constant (commit, rebundle) before training.
- [x] `kaggle_baseline.ipynb`: `REPLY = "label"` beside `LORA`; asserts `REPLY == "label" or LORA`; passes `--reply`; the label tag becomes `-ft-rf` / `-ft-lf`.
- [x] Commit; bundle the code to the `edge-grader-code` dataset (as on Oct 30).
- [x] **Pilot:** run the reasons notebook with `N = 50`. Look at the drop share and read the 50 reasons. If the drop share is above 5%, fix the prompt now (this is still before the full run, so it costs one pilot).
- [x] **Full run** (`N = 0`). If more than 5% are dropped: fix the prompt, rewrite once (spec §5.1); a second failure stops work for the owner's decision.
- [x] **Owner spot check:** draw 30 kept reasons with a fixed seed (`random.Random(0)`) and give them to the owner in a file, each with its question, reference, answer and gold label. Gate: 25/30 faithful. Record the marks in `docs/results/finetuning.md`. Fail → rewrite once; a second failure stops work.
- [x] Commit the reasons JSONL and summary.

### Task 7: Train the two E2B adapters (Kaggle)

- [ ] Run `kaggle_finetune.ipynb` with `MODEL = "e2b"`, `SMOKE = True`, `REPLY = "reason-first"` first (minutes): the template check passes, the loss is finite, validation parses replies. Then `SMOKE = False`.
- [ ] The same for `REPLY = "label-first"`.
- [ ] Check each summary: `versions` equal the Oct 27 ones (Tech Stack); `dropped_no_reason` equal in both; validation unparsed near 0.
- [ ] Upload `e2b-ft-rf.gguf` and `e2b-ft-lf.gguf` to `edge-grader-adapters` (new version, the old files kept). Commit the training summaries.

### Task 8: Grade (Kaggle)

- [ ] `kaggle_baseline.ipynb`, `MODELS = ["e2b"]`, `N = 0`, one kernel per adapter:
  - `DATASETS = ["beetle", "saf"]`, `SPLITS = ["test_ua", "test_uq"]`
  - `DATASETS = ["scientsbank"]`, `SPLITS = ["test_uq", "test_ud"]` (test_ud is the first cut if Nov 15 is at risk, spec §6)
- [ ] Download the results the way Oct 30 did (MCP `download_notebook_output` + curl); unzip into `results/kaggle/`. Check every run: `n` equals the split size, `errors == 0`, and for reason-first the `cut` count. Commit.

### Task 9: Compare and write up

- [ ] `uv run python -m edge_grader.compare ...` with the labels `e2b-kaggle-t4`, `e2b-kaggle-t4-ft`, `e2b-kaggle-t4-ft-rf`, `e2b-kaggle-t4-ft-lf`. Commit its JSON.
- [ ] `docs/results/finetuning.md`: a section "Reasoning-guided fine-tuning (E2B)", covering setup, the reasons check, the table, per-class recall on Beetle, unparsed and completion tokens, the gate result, and reason-first against label-first.
- [ ] Outline and `plan.md`: the result and what happens to step 4 (E4B / Jetson / no-harm) by the spec's rule.

## Not in this plan

E4B, Jetson cost and the MMLU-Pro/GSM checks: they follow only if the gate passes (spec §6) and get their own short plan then.
