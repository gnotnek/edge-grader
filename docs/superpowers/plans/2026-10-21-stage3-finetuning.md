# Stage 3 fine-tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train one QLoRA adapter each for Gemma 4 E2B and E4B on the SciEntsBank train split, serve each on the unchanged QAT GGUF with `--lora`, and grade it on the Jetson against the re-run baselines.

**Architecture:** A new module `edge_grader/finetune.py` holds the data preparation (pure, tested on the Mac) and the training entry point `edge-grader-finetune` (GPU only, run on Kaggle). A new notebook `notebooks/kaggle_finetune.ipynb` builds llama.cpp, checks the chat template, trains, converts the adapter with llama.cpp's `convert_lora_to_gguf.py` and grades a few answers with `--lora`. With `SMOKE = True` this is the Day 1 risk check. `scripts/board_grade.sh` gets a `LORA` setting for every benchmark, so the board runs are ordinary queue jobs.

**Tech Stack:** Python 3.12 via uv; transformers + peft + bitsandbytes + accelerate + torch as an optional `finetune` extra (Kaggle only); llama.cpp pinned to `a3a1c47` on Kaggle, `a894dae` on the Jetson.

**Spec:** [`docs/superpowers/specs/2026-10-21-finetuning-design.md`](../specs/2026-10-21-finetuning-design.md)

## Global Constraints

- Base checkpoints (QAT, unquantized, not gated, pinned by revision):
  - E2B: `google/gemma-4-E2B-it-qat-q4_0-unquantized` @ `6befbaca7398925921802abd1f277b495b78b738`
  - E4B: `google/gemma-4-E4B-it-qat-q4_0-unquantized` @ `476025a01dbf99361c062bbeca3d6a76bb4c4566`
- The GGUFs served are unchanged: `gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf` and `gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf` (unsloth).
- Server flags unchanged except ` --lora <adapter.gguf>` appended: `-c 8192 --host 127.0.0.1 --port 8080 -rea off --reasoning-budget -1 --cache-ram 0` (+ `-ngl`, `-t` as each platform already does).
- Labels: 3-way (`data.labels_for("3way")` = `contradictory`, `correct`, `incorrect`). Prompt: `grader.build_prompt(item, labels)`, no passages.
- Validation: 10% of train **questions** (`data.question_id`), seed 0; it picks the epoch and nothing else. The final adapter is the chosen epoch's checkpoint from the 90% run.
- QLoRA: rank 16, alpha 32, dropout 0.05, language-model linear layers only (`q/k/v/o/gate/up/down_proj`), max 3 epochs, lr 2e-4, effective batch 16, fp16 compute (T4 has no bf16), seed 0.
- Every training run writes a summary JSON to `results/finetune/`; adapters are **not** committed (private Kaggle dataset; sha256 recorded in the summary).
- uv only (`uv add`, `uv run`); the Mac runs no models; board runs only through `board_grade.sh` / `board_queue.py`.
- Grading metrics: accuracy and macro-F1, per split. Holm families: improvement (3 splits × 2 models), no harm (MMLU-Pro, GSM8K, GSM-Symbolic × 2 models).

## Review Focus

1. **Chat template drift**: the training text must be byte-for-byte what `llama-server` renders with `-rea off`. The notebook asserts equality against the server's `/apply-template` before training (Task 3).
2. **Validation leakage**: a question with answers in both train and validation inflates the epoch choice. Test: question sets are disjoint and the split is deterministic (Task 1).
3. **Adapter on the wrong modules**: Gemma 4 is `Gemma4ForConditionalGeneration` with vision and audio towers; a LoRA there can break conversion or do nothing. Test: the target regex matches language-model projections and rejects tower names (Task 1).
4. **A missing adapter file on the board**: the job must fail before it touches the board or starts telemetry. Test (Task 4).
5. **`LORA` with the little-gemma variants**: the shim can't load an adapter, so the run would silently be the baseline. Test: refused (Task 4).

---

### Task 1: Training data and pure helpers

**Files:**
- Create: `src/edge_grader/finetune.py`
- Test: `tests/test_finetune.py`
- Modify: `pyproject.toml` (optional extra and script)

**Interfaces:**
- Consumes: `data.Item`, `data.question_id(answer_id) -> str`, `data.labels_for("3way")`, `grader.build_prompt(item, labels) -> str`
- Produces:
  - `BASES: dict[str, tuple[str, str]]` — `{"e2b": (repo, revision), "e4b": (repo, revision)}` from Global Constraints
  - `TARGET_MODULES: str` — a regex for peft's `target_modules`
  - `split_by_question(items: list[Item], frac: float = 0.1, seed: int = 0) -> tuple[list[Item], list[Item]]` — (train, validation)
  - `to_messages(item: Item, labels: list[str]) -> list[dict]` — `[{"role": "user", "content": build_prompt(...)}, {"role": "assistant", "content": item.label}]`
  - `pick_epoch(history: list[dict]) -> int` — `history` items are `{"epoch": int, "val_accuracy": float, ...}`; highest accuracy, earliest on a tie
  - `summary(model_key: str, args: dict, history: list[dict], chosen: int, train_q: list[str], val_q: list[str], adapter_sha256: str, versions: dict) -> dict`

- [ ] **Step 1: Write the failing tests** in `tests/test_finetune.py`, on hand-made `Item`s (ids like `"EM.45b.299.1"`, so `question_id` gives `"EM.45b"`); no torch import anywhere:
  - `test_split_is_by_question_and_deterministic`: 20 questions × 5 answers → validation holds 2 whole questions (10 items); `{question_id(i.id) for i in train}` and the same for val are disjoint; two calls with seed 0 give the same split; seed 1 gives a different one.
  - `test_split_keeps_every_item`: `len(train) + len(val) == len(items)`.
  - `test_messages_use_the_baseline_prompt`: `to_messages(item, labels)[0]["content"] == grader.build_prompt(item, labels)` and `[1] == {"role": "assistant", "content": item.label}`.
  - `test_pick_epoch_best_then_earliest`: `[{"epoch":1,"val_accuracy":.6},{"epoch":2,"val_accuracy":.7},{"epoch":3,"val_accuracy":.7}]` → `2`.
  - `test_targets_language_model_only`: `re.fullmatch(TARGET_MODULES, n)` is true for `"model.language_model.layers.0.self_attn.q_proj"` and `"model.language_model.layers.3.mlp.down_proj"`, false for `"model.vision_tower.encoder.layers.0.self_attn.q_proj"`, `"model.audio_tower.layers.0.self_attn.q_proj"`, `"model.language_model.embed_tokens"`.
  - `test_summary_records_settings`: `summary(...)` has keys `model`, `base`, `revision`, `settings`, `history`, `chosen_epoch`, `train_questions`, `val_questions`, `adapter_sha256`, `versions`, `created`; `base`/`revision` come from `BASES`.

- [ ] **Step 2: Run** `uv run pytest tests/test_finetune.py -v` — expected: FAIL, `ModuleNotFoundError: edge_grader.finetune`.

- [ ] **Step 3: Implement the helpers** in `src/edge_grader/finetune.py`. `split_by_question`: sort the unique question ids, shuffle with `random.Random(seed)`, take `round(frac * n_questions)` (at least 1) for validation. Torch/transformers/peft imports go inside `main()` only (Task 2), so the boards and the Mac never need them.

- [ ] **Step 4: Add the extra and script**: `uv add --optional finetune torch transformers peft bitsandbytes accelerate` (transformers new enough for `Gemma4ForConditionalGeneration`; note the resolved versions in the commit message), and `edge-grader-finetune = "edge_grader.finetune:main"` under `[project.scripts]`.

- [ ] **Step 5: Run** `uv run pytest -q && uv run ruff check .` — expected: all pass, including the existing suite (the boards' `uv sync --frozen` without the extra must still work: `uv sync --frozen` then `uv run pytest -q`).

- [ ] **Step 6: Commit** `src/edge_grader/finetune.py tests/test_finetune.py pyproject.toml uv.lock`.

### Task 2: The training entry point

**Files:**
- Modify: `src/edge_grader/finetune.py` (add `main`)
- Test: `tests/test_finetune.py`

**Interfaces:**
- Consumes: Task 1's helpers; `data.load(split="train", scheme="3way")`; `grader.parse_label(text, labels)`
- Produces: `edge-grader-finetune --model e2b|e4b --out DIR [--epochs 3] [--max-steps N] [--val-n N] [--seed 0] [--dry-run]`. Writes `DIR/adapter/` (peft `save_pretrained` of the chosen epoch), `DIR/epoch-<k>/` per epoch, and `results/finetune/finetune-<model>-<stamp>.json` (Task 1's `summary`). `--dry-run` prints the split sizes and the first example's messages, then exits before importing torch.

- [ ] **Step 1: Write the failing test** `test_dry_run_needs_no_torch`: `main(["--model", "e2b", "--out", str(tmp_path), "--dry-run"])` with `data.load` monkeypatched to return 30 hand-made items over 10 questions; asserts the printed train/val counts (27/3) and that `"torch" not in sys.modules` afterwards.

- [ ] **Step 2: Run** `uv run pytest tests/test_finetune.py::test_dry_run_needs_no_torch -v` — expected: FAIL (`main` not defined).

- [ ] **Step 3: Implement `main(argv: list[str] | None = None) -> None`.** After the dry-run exit:
  - Load the processor/tokenizer and model from `BASES[model]` at the pinned revision, 4-bit NF4 (bitsandbytes, double quant, compute dtype fp16), `prepare_model_for_kbit_training`, then `LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=TARGET_MODULES, task_type="CAUSAL_LM")`.
  - Text per example: `tokenizer.apply_chat_template(to_messages(...), tokenize=False, enable_thinking=False)`. Loss only on the assistant part: the tokens after `apply_chat_template(messages[:1], add_generation_prompt=True, enable_thinking=False)`, i.e. the label and its end-of-turn token.
  - transformers `Trainer`, batch 4 × gradient accumulation 4, lr 2e-4, cosine schedule, fp16, `save_strategy="epoch"`, `num_train_epochs=args.epochs`, `max_steps=args.max_steps or -1`, seed.
  - After each epoch: greedy generation (max 16 new tokens, prompt rendered with `add_generation_prompt=True, enable_thinking=False`) on the validation items (first `--val-n` if given), `grader.parse_label`, accuracy into `history` with the train loss.
  - `pick_epoch` → copy that checkpoint's adapter to `DIR/adapter`, sha256 of `adapter_model.safetensors`, `summary(...)` with `versions` = torch/transformers/peft/bitsandbytes versions, written to `results/finetune/`.

- [ ] **Step 4: Run** `uv run pytest -q && uv run ruff check .` — expected: PASS.

- [ ] **Step 5: Commit** `src/edge_grader/finetune.py tests/test_finetune.py`.

### Task 3: Kaggle notebook (the Day 1 check with `SMOKE = True`)

**Files:**
- Create: `notebooks/kaggle_finetune.ipynb`

**Interfaces:**
- Consumes: `edge-grader-finetune` (Task 2); `edge-grader-baseline --base-url --label --split --n --out`; the llama.cpp build and repo-clone cells of `notebooks/kaggle_baseline.ipynb` (copy them verbatim, same pin `a3a1c47`, same `GITHUB_TOKEN` secret handling).
- Produces: in `/kaggle/working/`: `edge-grader/results/finetune/*.json`, `edge-grader/results/kaggle/baseline-<model>-ft-kaggle-*.json`, `adapters/<model>-ft.gguf`, and `adapters/<model>/` (the peft adapter).

Cells, in order:
- [ ] **Step 1: Header (markdown)**: what it does, the T4 accelerator, Internet on, the `GITHUB_TOKEN` secret; `SMOKE = True` is the Day 1 check (`--max-steps 20 --val-n 20`, grade 10 test_ua answers), `SMOKE = False` the real run (grade the first 100 test_ua answers).
- [ ] **Step 2: Settings cell**: `MODEL = "e2b"`, `SMOKE = True`.
- [ ] **Step 3: Build llama.cpp** (copied cell), then the converter's own venv with uv: `uv venv /tmp/convert-venv && uv pip install --python /tmp/convert-venv/bin/python -r /tmp/llama.cpp/requirements/requirements-convert_lora_to_gguf.txt`.
- [ ] **Step 4: Clone the repo** (copied cell), `uv sync --extra finetune`.
- [ ] **Step 5: Template check**: download the GGUF (as the baseline notebook does), start `llama-server` with the baseline flags, POST `/apply-template` with `{"messages": [user message of to_messages(item)]}` for the first 5 train items, and assert each equals the tokenizer's `apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)` with a leading `<bos>` stripped. Print both on mismatch and stop. Stop the server.
- [ ] **Step 6: Train**: `uv run edge-grader-finetune --model $MODEL --out /kaggle/working/adapters/$MODEL` (+ `--max-steps 20 --val-n 20` if `SMOKE`).
- [ ] **Step 7: Convert**: `/tmp/convert-venv/bin/python /tmp/llama.cpp/convert_lora_to_gguf.py /kaggle/working/adapters/$MODEL/adapter --base-model-id <BASES repo> --outfile /kaggle/working/adapters/$MODEL-ft.gguf --outtype f16` (use `--base <local snapshot dir>` if the script needs the config offline).
- [ ] **Step 8: Grade with `--lora`**: start `llama-server` with the baseline flags plus `--lora <gguf>`, check the server log contains the adapter load line, then `uv run edge-grader-baseline --label $MODEL-ft-kaggle-t4 --split test_ua --n {10 if SMOKE else 100} --out results/kaggle`. Print accuracy and `unparsed`.
- [ ] **Step 9: Verify locally**: the notebook is valid JSON and its code cells parse: `uv run python -c "import json,ast;nb=json.load(open('notebooks/kaggle_finetune.ipynb'));[ast.parse(''.join(c['source'])) for c in nb['cells'] if c['cell_type']=='code' and not ''.join(c['source']).startswith(('%%','!'))]"` — expected: no output, exit 0.
- [ ] **Step 10: Commit** `notebooks/kaggle_finetune.ipynb`.

### Task 4: `LORA` on the boards and in the Kaggle baseline notebook

**Files:**
- Modify: `scripts/board_grade.sh` (settings block near `VARIANT`, `NAME`, `server_cmd`, the missing-file check, `board_state`)
- Modify: `scripts/board_queue.py:38` (`JOB_ENV`)
- Modify: `notebooks/kaggle_baseline.ipynb` (settings cell, `start_server`)
- Test: `tests/test_board_grade.py`, `tests/test_board_queue.py`

**Interfaces:**
- Produces: `LORA=<path to adapter .gguf>` for every `BENCH`. Folder names get an `ft-` prefix (`ft-e2b-jetson-test_uq-<stamp>`, `mmlupro-ft-e2b-jetson-all-<stamp>`, `gsm-ft-e2b-jetson-symbolic-i1-<stamp>`). Adapters live on the Jetson at `$MODEL_DIR/adapters/<model>-ft.gguf`.

- [ ] **Step 1: Write the failing tests** in `tests/test_board_grade.py` (pass `LORA` through `dry(...)`; add `"LORA"` to the env keys `dry` strips):
  - `test_lora_adds_only_that_flag_for_every_benchmark`: for `BENCH` in scientsbank/gsm/news/mmlupro, `server == base_server + " --lora /m/adapters/e2b-ft.gguf"` and the runner is unchanged apart from the run folder.
  - `test_lora_folder_names`: `ft-e2b-jetson-test_ua-<stamp>`; `gsm-ft-e2b-jetson-symbolic-i1-<stamp>` with `INSTANCES=1`; `mmlupro-ft-e2b-jetson-all-<stamp>`.
  - `test_lora_refused_with_little_gemma`: `VARIANT=lg` or `lg-reuse` with `LORA` → non-zero exit, `"LORA"` in stderr.
  - `test_lora_must_be_a_gguf`: `LORA=/m/a.safetensors` → refused.
  - In `tests/test_board_queue.py`: `test_add_accepts_lora` — `add LORA=/m/adapters/e2b-ft.gguf -- e2b test_uq` is stored with that env.
  - The real-run check for a missing adapter file goes beside the existing `missing:` checks (one line, `[ -z "$LORA" ] || [ -f "$LORA" ]`), so it fails before telemetry; it isn't reachable in dry-run mode, so it's checked by reading in review.

- [ ] **Step 2: Run** `uv run pytest tests/test_board_grade.py tests/test_board_queue.py -v` — expected: the new tests FAIL.

- [ ] **Step 3: Implement** in `board_grade.sh`: `LORA="${LORA:-}"` with the two refusals; `NAME` prefix `ft-` (after `rag-` for SciEntsBank; `$BENCH-ft-` style for the others, matching how `-rag` is inserted); `server_cmd` appends ` --lora $LORA` in the llama.cpp branch; `board_state` prints `lora: $LORA sha256 <sum>` when set. Add `"LORA"` to `JOB_ENV` and so to `STRIP_ENV`.

- [ ] **Step 4: Kaggle baseline notebook**: settings `LORA = None  # or {"e2b": "/kaggle/input/<adapters dataset>/e2b-ft.gguf", ...}`; `start_server` appends `["--lora", LORA[key]]` when set; the label gets `-ft`. Same JSON/ast check as Task 3 Step 9.

- [ ] **Step 5: Run** `uv run pytest -q && uv run ruff check .` — expected: PASS.

- [ ] **Step 6: Commit** the four files.

### Task 5: Day 1 run and the path decision (owner on Kaggle; Claude on the Jetson)

- [ ] **Step 1 (owner):** push `main`; on Kaggle run `kaggle_finetune.ipynb` with `MODEL = "e2b"`, `SMOKE = True`. Download `/kaggle/working/` outputs: commit `results/finetune/*.json` and `results/kaggle/*` (not the adapters).
- [ ] **Step 2: Read the outcome** and record it in `docs/plan.md`:
  - template check passed, training ran on the T4, `convert_lora_to_gguf.py` produced a GGUF, the server log shows the adapter loaded, 10 answers graded with `unparsed` 0 → **main path**.
  - training failed on the T4 → **stop and escalate** to the owner (spec §6, risk 2).
  - conversion or loading failed → **fallback** (spec §5.2): add a task to merge, convert and quantize to Q4_K_M, plus the untuned control on the whole matrix. Write it before going on.
- [ ] **Step 3 (Jetson, after the baseline queue, jobs 36–49, is done):** copy the smoke adapter to `~/research/models/adapters/e2b-ft.gguf`, then `scripts/queue.sh jetson add MEASURE=0 LORA=/home/ari/research/models/adapters/e2b-ft.gguf -- e2b test_ua 10` and start the queue. Expected: the job finishes, `server.log` shows the adapter loaded on `a894dae`. Delete the run folder afterwards (a smoke run, not a result) and record the outcome in `docs/plan.md`.

### Task 6: The real adapters (owner on Kaggle)

- [ ] **Step 1 (owner):** `SMOKE = False`, `MODEL = "e2b"`; then the same for `"e4b"`. Check per model: the summary JSON's `history` (validation accuracy per epoch) and the sanity check on 100 test_ua answers (`unparsed` near 0).
- [ ] **Step 2 (owner):** upload both `<model>-ft.gguf` to a private Kaggle dataset (`edge-grader-adapters`); commit the summary JSONs and Kaggle results.
- [ ] **Step 3:** check that each GGUF's sha256 matches what's recorded (the summary records the safetensors; add the GGUF's sha256 to `results/finetune/adapters.json` with the dataset version), and commit.

### Task 7: Jetson runs

- [ ] **Step 1:** `scripts/deploy.sh jetson`; copy both GGUFs to `~/research/models/adapters/`; check `sha256sum` against `results/finetune/adapters.json`; check `df -h /` and `systemctl --user is-active va-llm` (memories: disk flood, voice-agent).
- [ ] **Step 2:** queue, per model (`e2b`, `e4b`), with `LORA=/home/ari/research/models/adapters/<model>-ft.gguf`: `-- <model> test_ua`, `-- <model> test_uq`, `-- <model> test_ud`, `BENCH=gsm -- <model> gsm8k`, `BENCH=gsm INSTANCES=1 -- <model> symbolic`, `BENCH=mmlupro -- <model> all`. Shortest first, MMLU-Pro last; `start`.
- [ ] **Step 3:** monitor only through `queue.sh jetson status` / `log` (never commands naming the engine; memory: busy-check gotcha). When done, `scripts/fetch_results.sh` and commit `results/board/`.

### Task 8: Analysis and write-up

- [ ] **Step 1:** pair each tuned run with the 2026-10-21 baseline re-run of the same benchmark, split and model, with `board_check.compare` / `board_check.mcnemar` (GSM-Symbolic: the first variant set's 100 ids of the baseline's 1,000, by template as in `results/rag.md`). Holm within each family (Global Constraints).
- [ ] **Step 2:** write `docs/results/finetuning.md`: per split accuracy, macro-F1, per-class recall and confusion matrix; the no-harm table; cost (latency, energy per answer, tokens) against the same-state baselines; the published-numbers table from the spec with the tuned rows added; Kaggle sanity numbers; caveats (one recipe, epochs from validation, test_ua shares train's questions).
- [ ] **Step 3:** update `docs/results/summary.md` (RQ3 section), `docs/research-outline.md` (stage 3 output), `docs/plan.md` (tick steps, change log). Commit.
