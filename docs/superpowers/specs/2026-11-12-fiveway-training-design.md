# Design: train the E4B adapter on SciEntsBank's 5-way labels, report 3-way

*2026-11-12. Method approved by the owner on 2026-11-12 ("5-way first"), chosen from five candidates in the owner's literature notes (`research_notes/2026-11-12-new-methods.md`, not in the repo). Running in parallel with the adapter-soup queue on the DGX Spark, as the owner allowed. Calls marked **(Claude's call)** were not discussed and can be changed until grading starts. The decision goes into [`research-outline.md`](../../research-outline.md).*

## 1. Why

**Where the E4B adapter still goes wrong.** The errors of the E4B label-only SciEntsBank adapter (Kaggle, 2026-10-28, `results/kaggle/raw/baseline-e4b-kaggle-t4-ft-test_*-3way-20261028-*.jsonl`), split by each answer's original 5-way label:

| Split | Errors | Correct called "incorrect" | Partly correct (incomplete) called "correct" | Share of errors | Contradictory called "incorrect" |
|---|---|---|---|---|---|
| test_ua | 113 | 51 | 24 | **66%** | 20 |
| test_uq | 147 | 64 | 38 | **69%** | 17 |
| test_ud | 1,075 | 435 | 348 | **73%** | 95 |

**Two-thirds or more of the errors are about completeness:** fully correct, or only partly? The 3-way label "incorrect" hides that question. In SciEntsBank train it merges three original labels: *partially correct incomplete* (1,324), *irrelevant* (1,115) and *non domain* (23). So the model learns one label that means three different things, and the hardest of them, "partly right", sits in the same bucket as "talks about something else".

**The method.** Train the adapter to reply with the original 5 labels, and map its reply to 3-way before scoring. It then learns "partly correct" as a label of its own.

**Evidence:**
- Chen, Z., Ding, R., Chin, T.-W., & Marculescu, D. (2018). Understanding the impact of label granularity on CNN-based image classification. *IEEE ICDM Workshops*, 895–904. doi:10.1109/ICDMW.2018.00131. Training on finer labels raised accuracy on the coarse classes on every dataset they tried; with 40% of the data, fine-label training beat coarse-label training on all of it. Images, not text.
- Zhang, Y., & Yang, Q. (2022). A survey on multi-task learning. *IEEE Transactions on Knowledge and Data Engineering*, 34(12), 5586–5609. doi:10.1109/TKDE.2021.3070203. Related labels learned together usually help each other.
- The 5-way labels are SciEntsBank's own (Dzikovska et al., 2013), present in the `nkazi/SciEntsBank` copy every run uses; `data.load(..., scheme="5way")` already reads them. Train: correct 2,008, partially correct incomplete 1,324, irrelevant 1,115, contradictory 499, non domain 23.

**Fits the 2026-11-11 decision "no 5-way baseline".** 5-way is used only as the *training target*. Every reported score stays 3-way (headline) and 2-way. Nothing is scored or reported 5-way; the 5-way gold appears only to describe *which* answers are fixed or broken (§5.5).

## 2. Research question

Part of **RQ3**: *Does training the E4B adapter on the 5-way labels, with 3-way scoring unchanged, grade SciEntsBank better than training it on the 3-way labels?*

**What "it helps" means (Claude's call):** the 5-way adapter is significantly better than the 3-way adapter of the same seed (§5.6) on at least one split after Holm, and significantly worse on none. If it helps, more seeds follow before it replaces the 3-way adapter, since one seed can be lucky (§3).

## 3. Goals, non-goals, anti-goals

**Goals:**
- **One E4B label-only adapter, seed 0, trained on the 5-way labels** on the DGX Spark, the Oct 27 recipe otherwise unchanged.
- Graded on the Spark on test_ua, test_uq and test_ud (whole splits), its replies mapped to 3-way.
- Compared with the **soup queue's 3-way seed-0 adapter** (`e4b-ft-s0`, trained on the Spark on 2026-11-11): same seed, so the same answer order, dropout, LoRA starting weights and validation questions (spec 2026-11-10 §5.1). **The labels and the label list in the prompt are the only difference.**

**Non-goals:**
- **More seeds now.** One seed answers "is there an effect worth chasing?" at a third of the cost. The five 3-way seeds give the run-to-run spread to read it against (§5.5). **(Claude's call)**
- **Any 5-way score** (decision 2026-11-11).
- Changes to rank, learning rate, epochs, prompt wording or class weights.
- E2B, Beetle/SAF adapters, reason replies. Beetle and SAF have no 5-way labels.
- Jetson cost. The reply is still one short label; *partially_correct_incomplete* is a few more tokens than *incorrect*. Measured only if the method is adopted.

**Anti-goals (a failure even if it runs):**
- Choosing the epoch, seed or mapping by looking at a test split.
- An epoch picked on a different metric than the one reported: validation picks on **3-way** accuracy (§5.2).
- Grading on different hardware or build from the runs it is compared with.

## 4. Constraints

- Fixed setup: Gemma 4 E4B, the lab's QAT `UD-Q4_K_XL` GGUF, llama.cpp `a3a1c47`, `-c 8192 -ngl 99 -rea off --reasoning-budget -1 --cache-ram 0`, temperature 0, thinking off, adapter served with `--lora`.
- DGX Spark through `docker compose`. **Runs alongside the soup queue** (owner's OK, 2026-11-12): the Spark had ~78 GB of memory free and one E4B training takes ~16 GB. Sharing the GPU slows both queues but changes no result; Spark timings are not cost data.
- Every run writes a summary JSON with its settings to `results/`. Metrics: accuracy and macro-F1, 3-way headline, 2-way beside it.

## 5. Design

### 5.1 The prompt and reply

The prompt is the baseline prompt (`grader.build_prompt`) with the 5 labels in its list instead of 3, using the descriptions already in `grader.LABEL_HELP`:

- correct: a complete and correct answer
- contradictory: contradicts the reference answer
- partially_correct_incomplete: partly correct but missing parts
- irrelevant: on topic but does not answer the question
- non_domain: not about the subject at all

The adapter is trained to reply with one of them. At grading, the reply is read as a 5-way label and mapped to 3-way with SciEntsBank's own map (`data.TO_3WAY`): the last three become *incorrect*. The raw reply and the 5-way label are kept in each answer's row (`pred5`); the scored label (`pred`) is 3-way, so the run file looks like any other 3-way run.

### 5.2 Training

- `edge-grader-finetune --model e4b --scheme 5way --seed 0`, the Oct 27 recipe otherwise: QLoRA nf4, rank 16, alpha 32, dropout 0.05, lr 2e-4, batch 1 × 16, cosine, up to 3 epochs, `split_seed` 0 and `init_seed` 0 as in the soup runs.
- **The epoch is picked on 3-way validation accuracy** (the 5-way reply mapped to 3-way, against the 3-way gold), the metric reported, so the pick is made the same way as for the 3-way seeds. 5-way validation accuracy is recorded beside it. **(Claude's call)**
- Same validation questions as every adapter (the split is by question id and doesn't depend on the labels).
- Files: `data/adapters/e4b-5w-s0/`, `e4b-ft-5w-s0.gguf`; summary `results/finetune/finetune-e4b-5way-*.json`.

### 5.3 Grading

On the Spark, whole splits, label `e4b-spark-gb10-ft-5w-s0`, through `scripts/spark_bench_scientsbank.sh` with `FIVEWAY=1`. Untouched E4B (`e4b-spark-gb10`, graded 2026-11-10) and the 3-way seeds (`e4b-spark-gb10-ft-s0` … `-s4`) come from the soup queue's grading and are not graded again.

### 5.4 Smoke

`SMOKE=1 SCHEME=5way SEED=0 scripts/spark_finetune.sh e4b label`: 20 steps, 20 validation answers, convert, load in `llama-server`, grade 10 answers; output in `data/smoke`. Passes if it trains, converts, loads and replies with parseable labels.

### 5.5 What gets reported

Per split, accuracy and macro-F1, 3-way and 2-way, for the 5-way adapter, the 3-way seed 0, untouched E4B, and the 3-way seeds' mean ± SD and min–max. Also described, not tested:
- Per-label recall (3-way).
- **Completeness errors** for the 5-way and 3-way seed-0 adapters: correct answers called "incorrect", and partly-correct answers called "correct", counted with the 5-way gold.
- The mix of 5-way labels the adapter replies with, against the 5-way gold mix.
- Where the 5-way adapter's accuracy falls in the 3-way seeds' spread: a gain inside the spread is weak evidence however the test comes out.

### 5.6 Comparisons (fixed before grading)

- **The family: 6 tests.** Per split: the 5-way adapter against the 3-way seed-0 adapter (the method's test), and against untouched E4B (the 5-point bar). Paired McNemar on 3-way labels, **Holm over all 6**. **(Claude's call)**
- 2-way beside it: the same 6 pairs, McNemar, its own Holm. Secondary, not part of the decision.
- Code: `edge_grader/compare_fiveway.py`, reusing `compare.latest`, `compare.paired` and `compare.holm`; writes `results/spark/compare-fiveway-scientsbank-*.json`.

## 6. Where the code changes

All defaults stay as they are, so the soup queue's later services read the same behaviour.
- `data.py`: `to_3way(label)`.
- `finetune.py`: `--scheme {3way,5way}` (default 3way; 5way only for SciEntsBank with label replies); 3-way validation accuracy picks the epoch; filename tag `-5way`.
- `baseline.py`: `--prompt-scheme 5way`: the 5-label prompt, the reply mapped to 3-way, `pred5` kept, `prompt_scheme` and the 5-label prompt's sha256 recorded.
- `scripts/spark_finetune.sh`: `SCHEME=5way` trains with `--scheme 5way`, names the run `-5w`, and grades its 100-answer check with the 5-label prompt.
- `scripts/spark_bench_scientsbank.sh`: `FIVEWAY=1` grades `e4b-ft-5w-s0` with the 5-label prompt.
- `compose.yaml`: `finetune-e4b-5w-s0`, `bench-scientsbank-5w`.
- Tests for each, on fixtures; no model.

## 7. Order and time

1. Code and tests on branch `claude/e4b-fiveway`; the owner merges to main; the Spark checkout pulls.
2. Smoke (minutes, plus model loading).
3. `finetune-e4b-5w-s0` now, beside the soup queue; then `bench-scientsbank-5w`. Training was ~3 h alone and ~6.3 h with one other job on the GPU; with the soup and another lab job it may take longer.
4. When the soup queue's grading is also done: `compare_fiveway`, results in `docs/results/`, decision row in `research-outline.md`, one line in `summary.md`.
