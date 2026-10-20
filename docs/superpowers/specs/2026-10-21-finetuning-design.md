# Design: stage 3 fine-tuning revived (QLoRA on SciEntsBank)

*2026-10-21. Worked through with the owner. Once approved, this spec feeds [`research-outline.md`](../../research-outline.md) and [`plan.md`](../../plan.md); those two stay the source of truth for what is decided and when. Calls marked **(Claude's call)** were not discussed with the owner and can be changed.*

## 1. Why now

Fine-tuning (RQ3, stage 3) was dropped on 2026-10-08 so the baseline could be finished first. The owner is bringing it back (2026-10-21) because the baseline and the RAG runs weren't promising for grading:

- Baseline, 3-way accuracy, test_ua / test_uq / test_ud: E2B 55.2 / 59.8 / 60.3%, E4B 67.0 / 67.5 / 70.4% ([baselines](../../results/baselines.md)).
- RAG-en doesn't help grading: E2B −5.1 on test_ud (p < 0.001), E4B +0.3 ([rag](../../results/rag.md)). The Jetson runs agree.

RAG adds knowledge to the prompt; fine-tuning changes how the model judges. The supervisor said on 2026-09-29 that fine-tuning was "necessary, to compare a general model with a tweaked one", so this also answers an original request.

**Where the published numbers sit** (SciEntsBank 3-way accuracy, ua / uq / ud, from Wang & Ormerod 2024, arXiv 2408.03811, Table 2):

| System | test_ua | test_uq | test_ud |
|---|---|---|---|
| Best SemEval-2013 systems | 72.0 | 65.3 | 64.0 |
| Fine-tuned BERT-base (Ghavidel 2020) | 72.6 | 70.8 | 67.2 |
| Claude 3.5 Sonnet + RAG (best published) | 78.7 | 76.9 | 73.4 |

## 2. Research question

**RQ3 (revived):** Does fine-tuning (QLoRA on the SciEntsBank train split) improve grading on the Jetson, especially on unseen questions and unseen domains? What does it cost in general ability (MMLU-Pro, math) and in latency and energy on the device?

**What "it worked" means (owner, 2026-10-21):** for small models, about **65% is the best E2B can be expected to reach and about 75% for E4B**. That's +5 to +10 points from the baseline. The headline is **test_uq and test_ud**, since test_ua shares its questions with train and is expected to rise most.

**"No harm"** means MMLU-Pro and GSM accuracy don't drop significantly (paired McNemar, Holm-corrected over the cells in §5.4).

## 3. Goals, non-goals, anti-goals

**Goals:**
- One QLoRA adapter each for E2B and E4B, trained on the SciEntsBank train split
- The tuned models graded on all three SciEntsBank splits, on the Jetson, with cost
- The same tuned models run on MMLU-Pro and GSM, to check for damage to general ability
- Done in **two weeks (2026-10-21 → 2026-11-03)**; phase C (images) waits until this is done (owner)

**Non-goals:**
- **A hyperparameter search.** One recipe, with epochs picked on a held-out validation set. Saur & Toth's 324-config grid is the kind of search that doesn't fit two weeks.
- **Fine-tuning for MMLU-Pro, math or news.** Only grading is tuned; the other benchmarks are checks.
- **News.** Without RAG both models score 0.5%, so it can't show damage. News with RAG is optional (§5.3).
- **A new grading dataset** (SAS-Bench etc.). test_uq and test_ud already test transfer.
- **Thinking on, RAG + fine-tuning, little-gemma with the adapter.** Each would be a new condition with no question asking for it.
- **The Pi 5** (dropped 2026-10-19). **The supervisor's server**: only a fallback for training (§6).

**Anti-goals (failure even if it runs):**
- Anything chosen by looking at the test splits: epochs, learning rate, checkpoint, prompt.
- A tuned model that only runs on Kaggle. It must run on the Jetson through `llama-server` like every other version.
- A comparison that differs from the baseline in more than the fine-tuning: same prompt, same server flags, same GGUF base (or the confound named and controlled, §5.2).
- Reporting the test_ua gain as the headline.

## 4. Constraints

- Fixed setup from the outline: Gemma 4 E2B / E4B, the lab's QAT `UD-Q4_K_XL` GGUFs, `llama-server -c 8192 -rea off --reasoning-budget -1 --cache-ram 0`, temperature 0, thinking off.
- **QAT caveat** (outline): anything that changes the weights must be matched to the QAT build or the mismatch noted as a confound.
- Kaggle T4 (16 GB, no bf16), about 30 GPU-hours a week. The Mac runs no models.
- Edge first: accuracy and cost on the Jetson are the result; Kaggle is for training and a sanity check.
- Separate from `agentic-edge`: board runs through `scripts/board_grade.sh` only.
- Every run, **training included**, writes a summary JSON with its settings to `results/`.
- Grading metrics: accuracy and macro-F1, per split. Not QWK.

## 5. Design

### 5.1 Training data and recipe

- **Data:** the SciEntsBank train split (4,969 answers), **3-way labels**, the same collapse as the baseline (`data.labels_for("3way")`).
- **Validation:** hold out about 10% of train **by question** (all answers to a held-out question go together), so validation behaves like test_uq. It picks the number of epochs and nothing else. Fixed seed. The final adapter is the checkpoint at the chosen epoch from this 90% run, **not** a retrain on all of train, which would double the GPU time. **(Claude's call)**
- **Prompt:** exactly the baseline's, built with `grader.build_prompt`, wrapped in Gemma 4's chat template with thinking off, as `llama-server` serves it. The target is the label text the parser expects. Loss on the label only. Any difference between the training template and the server's template is a bug.
- **Method:** QLoRA, adapters on all linear layers, rank 16, 1–3 epochs chosen on validation. Library left to the builder (Unsloth or PEFT), as long as it trains Gemma 4 on a T4. **(Claude's call)**
- **Base checkpoint:** Google's unquantized **QAT** checkpoint for each model if one is published, since the GGUF the adapter runs on is the QAT build. Otherwise the regular instruct checkpoint, with the mismatch recorded as a confound. **(Claude's call)**
- **Training summary JSON** in `results/`: base checkpoint, library and versions, hyperparameters, seed, train/validation question split, loss and validation accuracy per epoch, the chosen epoch, the adapter's sha256.
- **Adapters are stored outside git**, as a private Kaggle dataset, with the sha256 recorded in the repo, the same pattern as the RAG chunks.

### 5.2 How the tuned model runs

- **Main path:** convert the adapter to GGUF with llama.cpp's `convert_lora_to_gguf.py` and serve the **unchanged baseline GGUF** with `--lora <adapter>.gguf` added to the fixed flags. Only the adapter differs from the baseline run.
- **Fallback, if conversion or loading fails for Gemma 4:** merge the adapter into the base, convert, and quantize to Q4_K_M. That loses the QAT `UD-Q4_K_XL` weights, so the fallback adds a **control**: the untuned base, converted and quantized the same way, run on the **whole matrix in §5.3** (MMLU-Pro and GSM too, or the no-harm check is mixed up with the quantization change). The fine-tuning effect is then tuned-vs-control, not tuned-vs-baseline. This roughly doubles the Jetson time, so the cut order in §6 applies sooner. **(Claude's call)**
- The llama.cpp build is pinned as before. If `--lora` needs a newer build for Gemma 4, the baseline is re-run on that build on test_ua first, as on 2026-10-05.

### 5.3 Evaluation matrix (per model: E2B and E4B)

| Benchmark | Set | Paired with | Status |
|---|---|---|---|
| SciEntsBank | test_ua 540, test_uq 733, test_ud 4,562 | the Jetson's baseline (stage 1 / phase 0) | must |
| MMLU-Pro | 300 (s1–s3), this project's runner | the Jetson's own-runner MMLU-Pro baseline (queued for stage 4, not run yet) | must |
| GSM | GSM8K 100 + the first GSM-Symbolic variant set 100 | the Jetson's phase 0 runs, same 200 questions | must |
| News + RAG-en | 200, judged by Qwen3-14B on Kaggle | the Jetson's stage 4 RAG-en run | optional |

- **Where:** the Jetson, measured, through `board_grade.sh` with a new adapter option for every `BENCH`. Kaggle runs test_ua (first 100) as the sanity check before anything goes to the board, and is the backup if the board is unavailable.
- **Cost:** latency, prompt and completion tokens, energy per answer, as for every board run. The Jetson's clocks changed between 2026-10-13 and 2026-10-19 (GPU max 612 → 1,020 MHz; plan, stage 4), so **cost is compared with baselines run in the same clock state**. Accuracy isn't affected.
- **Owner, 2026-10-21: every baseline is re-run on the Jetson at the new clocks**, starting the night of 2026-10-21, before any tuned run reaches the board. These re-runs are also the pairs for the stage 4 RAG cost. The voice-agent project deployed on the Jetson on 2026-10-20 is stopped first to free the board (owner's instruction).

### 5.4 Analysis

- Paired McNemar per cell (tuned vs its baseline, same board, same items). For GSM-Symbolic, by template as before.
- **Holm families, fixed before any run (Claude's call):** (1) *improvement*: SciEntsBank test_ua, test_uq, test_ud × E2B, E4B (6 tests); (2) *no harm*: MMLU-Pro, GSM8K, GSM-Symbolic × E2B, E4B (6 tests). The optional news run is reported without correction.
- Per split: accuracy, macro-F1, per-class recall (the baseline's weak spot is recognising *correct*), confusion matrix.
- A row added to the published-numbers table in §1.
- Results page: `docs/results/finetuning.md`; summary and outline updated.

## 6. Timeline and risks

| When | What |
|---|---|
| **Day 1 (Oct 21–22)** | Risk checks: a tiny E2B training run on a Kaggle T4 (a few steps), convert the adapter, load it with `--lora` in the pinned `llama-server` **on Kaggle (`a3a1c47`) and on the Jetson's build (`a894dae`)**, grade 10 answers on each. The result decides main path or fallback |
| Week 1 (to Oct 27) | Training notebook; E2B trained, validated, converted; Kaggle sanity check on test_ua (100); then E4B; adapter option in `board_grade.sh` and the Kaggle notebook, with tests |
| Week 2 (Oct 28 – Nov 3) | Jetson runs (§5.3); analysis; results page; outline and plan updated |
| After | Phase C (images) resumes |

**Risks, worst first:**
1. **`convert_lora_to_gguf.py` or `--lora` doesn't support Gemma 4.** Fallback in §5.2.
2. **Gemma 4 won't train on a T4** (no bf16; new architecture in the training libraries). This is a **stop-and-escalate point**, not a path: the supervisor's server isn't set up and its hardware is unknown, so the owner decides then whether to set it up or change the plan. **(Claude's call)**
3. **The Jetson:** the clock-change question and the disk-full issue from 2026-10-19. Check `df` before long runs.
4. **Overfitting to train questions:** test_ua rises while test_uq/test_ud don't. That is a result, not a failure, and gets reported as such.

**Cut order if the two weeks run short (owner, 2026-10-21), first cut first:** (1) the optional news run; (2) the GSM-Symbolic set (GSM8K stays); (3) test_ua on the Jetson (the Kaggle number stays); (4) phase C slips one more week. Dropping E4B comes last.

## 7. Docs to update with the code

- **Outline:** RQ3 revived (RQ table, stages table); a feedback-table note that this answers the supervisor's 2026-09-29 request; own-decisions rows for the calls in §5 with their reasons.
- **Plan:** stage 3 steps and timeline (weeks 6–7 fine-tuning, phase C after); a change-log row.

## 8. Left to the builder

- The training library and its versions, as long as the prompt and template match `llama-server`'s exactly.
- Learning rate, batch size, LoRA alpha: standard values for QLoRA on a model this size, recorded in the summary JSON.
- The notebook's structure, and the name and shape of the adapter option in `board_grade.sh`.
- How the Jetson jobs are queued (preset or by hand), within the separation rule.
