# Design: reasoning-guided fine-tuning (stage 3, second recipe)

*2026-11-01. Worked through with the owner. Calls marked **(Claude's call)** were not discussed and can be changed. Once approved, the decisions go into [`research-outline.md`](../../research-outline.md) and [`plan.md`](../../plan.md), which stay the source of truth.*

## 1. Why

The Oct 28 adapters (label-only QLoRA on the SciEntsBank train split, [spec](2026-10-21-finetuning-design.md)) raised SciEntsBank but barely moved the held-out grading sets ([results](../../results/finetuning.md#held-out-grading-sets-beetle-and-saf-2026-10-31)):

- Gains on Beetle and SAF of +2 to +7 points; only E2B Beetle test_uq is significant after Holm. E4B gains nothing that holds up.
- The adapters learned SciEntsBank's label mix: they over-predict "correct" and under-predict "contradictory", which is 10% of SciEntsBank train but 25–30% of Beetle.

The supervisor asked for published work that makes Gemma grade better. The closest match is **Faza, Purnamasari & Ratna (2025)**, IJECBE 3(3), doi 10.62146/ijecbe.v3i3.148: Gemma-3-1B, 4-bit QLoRA, trained to write a reason for the grade before the score, with reasons written by a larger model (DeepSeek-R1). Label-only tuning overfit (QWK 0.728); reason-guided tuning reached 0.779 and stopped tracking answer length. It cost 35× the inference time. Two other papers find that single-dataset fine-tuning transfers poorly, as ours did: Saur & Toth (2026, Frontiers in Computer Science, doi 10.3389/fcomp.2026.1829089) and ASAG2024 (Meyer et al., 2024, doi 10.1145/3649409.3691083).

## 2. Research question

Part of **RQ3**: *Does training on reasons, not just labels, make the tuned small model grade better on grading sets it never saw (Beetle, SAF)? Does the reason have to come before the label, or does it help when it comes after, so the device can stop at the label at no extra cost?*

**What "it worked" means:** in at least one Beetle or SAF cell (set × split), a new adapter is **higher than both** the untuned model and the Oct 28 label-only adapter, and **both** differences stay significant after the Holm correction in §5.5. Both wins must be in the same cell. **(Claude's call)** Gains on SciEntsBank alone don't count; they are what the Oct 28 adapter already showed.

## 3. Goals, non-goals, anti-goals

**Goals:**
- Reasons for every SciEntsBank train answer, written by Qwen3-14B on Kaggle (owner).
- **Two E2B adapters** (owner): *reason-first* (reason, then label) and *label-first* (label, then reason; grading stops at the label).
- Both graded on Beetle and SAF (test_ua, test_uq) and on SciEntsBank test_uq and test_ud, on Kaggle.
- **E4B only if E2B passes** (owner): the same two adapters for E4B only if at least one E2B adapter meets "it worked" in §2. **(Claude's call: the gate is exactly §2's test)**

**Non-goals:**
- **New training data.** SciEntsBank train only, the same 90/10 question split as before (owner). Beetle's train split and class balancing are not used: each would be a second change, and Beetle must stay held out.
- **A hyperparameter search.** The Oct 28 recipe unchanged except for the target text.
- **A different teacher or a paid API.** DeepSeek-R1 was the alternative; Qwen3-14B is free, reproducible, not a Gemma model, and already set up as the news judge (owner).
- **MMLU-Pro / GSM no-harm checks** for this round. They come back only for an adapter that goes forward. **(Claude's call)**
- **Jetson runs during the pilot.** Board cost is measured only for an adapter that passes §2, as part of stage 3's open "cost on the Jetson" item. **(Claude's call)**

**Anti-goals (a failure even if it runs):**
- Anything chosen by looking at a test split or at Beetle/SAF: epoch, prompt, reason format, filters.
- Reasons that state a different label from the gold one, kept in training.
- A comparison that differs from the Oct 28 run in more than the training target: same base checkpoint, recipe, split, seed, grading prompt, GGUF, llama.cpp build and server flags.
- A tuned model that can't run on the Jetson through `llama-server --lora`.

## 4. Constraints

- Fixed setup: Gemma 4 E2B (later E4B), the lab's QAT `UD-Q4_K_XL` GGUF, llama.cpp pinned at `a3a1c47` on Kaggle, `-c 8192 -rea off --reasoning-budget -1 --cache-ram 0 -ngl 99`, temperature 0, thinking off. The adapter is served with `--lora`, as on Oct 28.
- Kaggle T4, about 30 GPU-hours a week. The Mac runs no models.
- Every run writes a summary JSON with its settings to `results/`, including the reason-writing run.
- Metrics: accuracy and macro-F1. 2-way (correct vs not) is the headline for SAF and is reported for Beetle; 3-way for Beetle and SciEntsBank.

## 5. Design

### 5.1 Writing the reasons (teacher)

- **Model:** Qwen3-14B, Qwen's official Q4_K_M GGUF, served in `llama-server` on Kaggle with the Gemma flags, thinking off, temperature 0, as in [`kaggle_news_judge.ipynb`](../../../notebooks/kaggle_news_judge.ipynb).
- **Input:** question, reference answer, student answer and the **gold 3-way label**. The teacher explains why the answer earns that label; it does not grade (owner: every training answer gets a reason).
- **Reason:** 1–3 sentences, at most about 60 words, comparing the student answer with the reference: what it gets right, what is missing or contradicts it. Cap of 128 tokens. **(Claude's call)** The prompt is a file in `src/edge_grader/prompts/`.
- **Automatic filter:** drop a reason that is empty, cut off at the cap, or names a label other than the gold one as the verdict. The share dropped is recorded. If more than 5% are dropped, the prompt is fixed and all reasons rewritten, before any training. **(Claude's call)**
- **Spot check (gate before training):** the owner reads 30 random reasons from the train questions and marks each faithful or not (it explains the gold label and says something true about the answer). Training goes ahead at **25/30 or better**; otherwise the prompt is revised and the reasons rewritten once. Same pattern as the RAG retrieval check. **(Claude's call)**
- **If the rewritten set fails either check again, work stops** and the owner decides: a different teacher, or dropping this recipe. The thresholds are never loosened after seeing the reasons. **(Claude's call)**
- **Output:** a JSONL of answer id → reason, committed as `results/finetune/raw/reasons-qwen3-14b-q4_k_m.jsonl` (about 5k short lines from public data; the code bundle carries it to Kaggle), with its sha256 and the run's settings (model file, build, prompt sha256, drop counts) in a summary JSON under `results/finetune/`. Each answer is saved as it goes, so a stopped run resumes.

### 5.2 Training targets

Same grading prompt as the baseline (`grader.build_prompt`), unchanged for both adapters: only the trained reply differs, so the comparison with Oct 28 stays one change. **(Claude's call)**

| Adapter | Trained reply | How the label is read |
|---|---|---|
| Oct 28 (existing) | `correct` | first label in the reply |
| **reason-first** | `<reason>\nLabel: correct` | the label after the last `Label:`; none → unparsed |
| **label-first** | `correct\n<reason>` | first line only; the server stops at the first newline |

- The reason text must not change between the two adapters: both read the same JSONL.
- Loss on the whole reply (reason and label), as in Faza et al. **(Claude's call)**
- Answers whose reason was filtered out are dropped from training for **both** adapters, so they train on the same answers. The count goes in the summary. **(Claude's call)**
- Recipe as Oct 28: rank 16, alpha 32, dropout 0.05, lr 2e-4, batch 1 × 16, cosine, up to 3 epochs, QAT base checkpoint, seed 0, same 90/10 split by question.
- Training libraries pinned to the Oct 27 training's versions (torch 2.14.1+cu130, transformers 5.18.0, peft 0.21.2, bitsandbytes 0.50.2, from `results/finetune/finetune-e2b-20261027-111124.json`), so the library isn't a second change. **(Claude's call)**
- **Epoch choice:** validation accuracy on the held-out 10% of train questions, read the way each adapter is graded (reason-first generates its reason, up to the cap in §5.3). Nothing else is chosen on validation.

### 5.3 Grading with the adapters

- **label-first:** `max_tokens` 16 with stop at `\n`. The same cost per answer as the baseline, apart from the adapter.
- **reason-first:** `max_tokens` 160 (the 128-token reason cap plus the label line). The cap is counted in Qwen's tokens, so before training the builder measures the training replies in Gemma's tokenizer and raises `max_tokens` to cover the longest one, recording the value. A reply cut before `Label:` counts as unparsed and wrong, as in the thinking runs. **(Claude's call)**
- Runs through `edge-grader-baseline` with a new option for the reply format, recorded in the summary JSON and in the file name, so the dashboard and the analysis can't confuse them with label-only runs.

### 5.4 Evaluation (E2B; E4B the same if it goes ahead)

Kaggle, the whole split each time:

| Set | Splits | Answers |
|---|---|---|
| Beetle | test_ua, test_uq | 439, 819 |
| SAF | test_ua, test_uq | 375, 479 |
| SciEntsBank | test_uq, test_ud | 733, 4,562 |

SciEntsBank test_ua is left out: it shares questions with train, so it can't show transfer. **(Claude's call)** The untuned and Oct 28 runs on these splits already exist on the same build and are reused.

### 5.5 Comparisons

- Each new adapter against the untuned E2B and against the Oct 28 adapter, on each of the four Beetle/SAF cells: paired McNemar, Holm-corrected over the 16 tests (2 adapters × 2 baselines × 4 cells). Beetle is tested on 3-way and SAF on 2-way. **(Claude's call)**
- Reported beside it, not tested: the SciEntsBank cells, per-class recall (whether "contradictory" recovers on Beetle), the share unparsed, and median completion tokens per answer.
- Reason-first against label-first is described, not gated. It answers whether the reason must come first.

### 5.6 Where the code changes

Builder's choice within these lines:
- A reason-writing CLI beside `edge-grader-baseline` (talks to `llama-server`, resumes, writes the JSONL and summary).
- `finetune.py`: an option for the reply format (`label`, `reason-first`, `label-first`) and the reasons file; its sha256 goes in the training summary.
- `grader.py` / `baseline.py`: the matching reply format at grading time (§5.3).
- A Kaggle notebook for the reasons, and the existing fine-tune and baseline notebooks given the new options.
- Tests for the parsing, the training text for each format, and the filter.

## 6. Order and time

1. Write the reasons on Kaggle; owner spot check (gate).
2. Train both E2B adapters; grade both on §5.4.
3. Compare (§5.5) and write it up in `docs/results/finetuning.md`.
4. If §2 passes: E4B, then the Jetson cost and no-harm checks for the adapters that go forward.

**Timing (owner, 2026-11-01):** this runs on Kaggle **alongside stage 4** (Jetson cost, Nov 2–8), which keeps its dates; the two share no hardware. Steps 1–3 finish by the end of the buffer week (Nov 15), before the write-up starts on Nov 16. Step 4 happens only if there is time left in the buffer week; otherwise it goes in the write-up as future work. **(Claude's call: the Nov 15 end and what happens to step 4)**

A rough budget for steps 1–2 is 15–20 Kaggle GPU-hours, mostly reason writing and the reason-first grading of SciEntsBank test_ud. If steps 1–3 won't finish by Nov 15, test_ud is cut first. **(Claude's call)**

## 7. Left to the builder

The exact teacher prompt wording (within §5.1), the reason-writing CLI's name, how training text is tokenised for each format (it must match what `llama-server` serves), and how the notebooks are split.
