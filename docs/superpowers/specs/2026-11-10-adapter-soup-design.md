# Design: adapter averaging ("model soup") for E4B on SciEntsBank

*2026-11-10. Approved as a method by the supervisor on 2026-11-10, together with graded examples in the prompt (a separate session). **Spec approved by the owner on 2026-11-10, as written.** Calls marked **(Claude's call)** were not discussed and can be changed. Once approved, the decision goes into [`research-outline.md`](../../research-outline.md), which stays the source of truth.*

## 1. Why

The E4B label-only adapter trained on SciEntsBank (Kaggle T4, 2026-10-27, [results](../../results/finetuning.md)) is the most accurate E4B version so far: 79.1 / 79.9 / 76.4% on test_ua / test_uq / test_ud, against 66.7 / 67.5 / 70.4% untouched. But it is **one training run with one seed**. The one re-run with the same settings (Oct 23 → Oct 27) moved test_ua by about a point, so we don't know how much of the result is that seed's luck.

**The method.** Wortsman et al., ICML 2022, *Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time* (arXiv 2203.05482). Several models fine-tuned from the same starting point are averaged weight by weight into one model. The average usually beats a single run, and it costs the same to run as one model. In plain words: each training run lands at a slightly different spot; averaging them cancels part of each run's noise.

This answers two things at once:
- **Accuracy:** does an average of several runs grade better than one run?
- **Reliability:** the single runs give the run-to-run spread (mean and SD), which every adapter result so far has lacked.

## 2. Research question

Part of **RQ3**: *Does averaging several seeds of the same E4B adapter grade SciEntsBank better than a single seed, at the same serving setup?*

**What "it worked" means (Claude's call):** the soup becomes "the E4B adapter" from here on if, on every test split, (1) its accuracy is at or above the single-seed mean, and (2) no single seed is significantly better than it after Holm (§5.6). A significant win over a single seed is reported but not required: a soup typically gains about a point, and on 540–733 answers McNemar can't detect one point, so requiring it would mostly test the sample size. The project's **5-point bar** (owner, 2026-11-05) is read for the soup against untouched E4B, as for every other version.

## 3. Goals, non-goals, anti-goals

**Goals:**
- **Five E4B label-only adapters** trained on the DGX Spark with seeds 0–4, the Oct 27 recipe otherwise unchanged. Seed 0 is a Spark re-run of the Kaggle adapter's seed. **(Claude's call: 5)**
- **One uniform soup** of the five, served as one adapter with `llama-server --lora`, as the single adapters are.
- All seven versions graded on the Spark on test_ua, test_uq and test_ud (whole splits): untouched E4B, the five seeds, the soup.

**Non-goals:**
- **Greedy soup.** Dropped. **(Claude's call)** Wortsman et al. use it to keep bad ingredients out when the runs differ in hyperparameters. Here only the seed differs, so no ingredient should be bad, and the decision "add this run if validation accuracy doesn't drop" would rest on 499 validation answers from 14 questions. On the Oct 27 E4B run the three epochs differed by **2 answers** (77.6 / 77.2 / 77.2%), so a greedy step would mostly follow noise. It would also need a new way to grade a soup on validation through `llama-server` (the training script validates in transformers, not llama.cpp).
- **Hyperparameter changes** (rank, learning rate, epochs). Only the seed varies, so the soup is the only new thing.
- **E2B, Beetle/SAF-trained adapters, label-first or reason-first.** E4B label-only on SciEntsBank is the version being improved.
- **Jetson cost in this round.** The soup adapter is 5× the rank (§5.3), so it costs the device a little more per token than a single adapter; measuring that is a follow-up, run only if the soup is adopted. **(Claude's call)**

**Anti-goals (a failure even if it runs):**
- Anything chosen by looking at a test split: which seeds go in, how many, the epoch. The number of seeds and the averaging rule are fixed here, before any training.
- A validation split that changes with the seed (§5.1).
- An "average" that isn't the average of the weight updates (§5.2).
- A comparison graded on different hardware: every run in the family is graded on the Spark, with the same build and flags.

## 4. Constraints

- Fixed setup: Gemma 4 E4B, the lab's QAT `UD-Q4_K_XL` GGUF, llama.cpp `a3a1c47`, `-c 8192 -ngl 99 -rea off --reasoning-budget -1 --cache-ram 0`, temperature 0, thinking off. Adapters served with `--lora`.
- DGX Spark through `docker compose` and `scripts/spark_queue.sh`, one GPU job at a time. **The Spark is shared** with the `e4b-examples` session and other lab members: these jobs chain after theirs with `AFTER=<container>`.
- Every run writes a summary JSON with its settings to `results/`. Metrics: accuracy and macro-F1 (3-way headline, 2-way beside it).
- The owner's Mac runs no models.

## 5. Design

### 5.1 Training: only the seed varies

- The Oct 27 recipe unchanged: QLoRA, nf4, rank 16, alpha 32, dropout 0.05, lr 2e-4, batch 1 × 16, cosine, up to 3 epochs, the same base checkpoint. Each run picks its own epoch on validation, as the recipe always has. **(Claude's call: per-run epoch, since that's the recipe being averaged)**
- **The validation split is fixed.** Today `--seed` also seeds `split_by_question`, so seed 1 would hold out different questions and train on a different set. The split gets its own fixed seed (0, the one every adapter so far used), recorded in the summary as `split_seed`. All five runs then train on the same 4,470 answers and validate on the same 499.
- **The LoRA starting weights are the same in every run.** LoRA starts with B = 0, so every run starts from exactly the untouched model; the random A matrices are made before the trainer sets its seed, so today they don't follow `--seed` at all. They are now seeded explicitly with a fixed value, recorded as `init_seed`. The seed then changes the order of the training answers and the dropout, nothing else. This matches the paper's condition that the averaged models share their starting point. **(Claude's call)**
- Files: `data/adapters/e4b-s<seed>/` and `e4b-ft-s<seed>.gguf`; training summaries `results/finetune/finetune-e4b-*.json`, with `seed`, `split_seed` and `init_seed` in their settings. The soup's summary, `results/finetune/e4b-soup-*.json`, lists each ingredient's sha256. The Kaggle adapter `e4b-ft.gguf` is not overwritten.

### 5.2 Averaging LoRA correctly

A LoRA adapter stores two small matrices per layer, A (r × in) and B (out × r), and the change to the model's weight is **ΔW = (α/r)·B·A**. Averaging the As and the Bs separately does **not** give the average ΔW, because the product of averages is not the average of products. The soup must be W₀ + mean(ΔWᵢ).

Options:

| | How | Exact? | Serving |
|---|---|---|---|
| **(a) Stack along the rank** | A_soup = [A₁; …; A₅] (80 × in), B_soup = [B₁ … B₅] (out × 80), α kept at 32 | yes | one adapter, `--lora` as usual |
| (b) Five adapters at once | `--lora-scaled a1.gguf 0.2 … --lora-scaled a5.gguf 0.2` | yes | five files, new server flags |
| (c) Merge into the base | W₀ + mean(ΔWᵢ), then re-quantize | no: re-quantizing changes W₀ | changes the fixed GGUF |
| (d) Stack, then cut back to rank 16 (SVD) | best rank-16 approximation of the soup | no | one rank-16 adapter |

**Choice: (a).** **(Claude's call)** B_soup·A_soup = Σ Bᵢ·Aᵢ exactly. The 1/5 comes for free: llama.cpp scales each adapter by α / rank, and reads the rank from the tensor's shape (`get_scale` in `src/llama-adapter.h` at `a3a1c47`: `alpha * adapter_scale / b->ne[0]`), so with α = 32 and rank 80 the scale is 32/80 = (1/5)·(32/16). That is exactly the mean of the five updates. PEFT reads it the same way when `r` = 80 in the config. `convert_lora_to_gguf.py` at `a3a1c47` writes only α and takes the rank from the tensors, so it accepts the stacked adapter unchanged. (b) is the same arithmetic per token but needs five files and different server flags on every machine; (c) breaks the fixed GGUF; (d) changes the soup, which would then be a second variable.

**Cost:** rank 80 does five times the adapter's own arithmetic per token, the same as (b). The adapter is a small part of the model's work, but it isn't zero; Spark timings are shared and not cost data, so this goes to the Jetson follow-up (§3).

**Check:** a unit test builds random adapters and confirms (α/80)·B_soup·A_soup equals the mean of the (α/16)·Bᵢ·Aᵢ, and that a soup of one adapter is that adapter. The converted soup GGUF is loaded by `llama-server` and grades 100 test_ua answers as a load check, as every adapter does after training (nothing is chosen from it).

### 5.3 Number of seeds

**Five, seeds 0–4.** **(Claude's call)** Enough for a mean and SD of single-run accuracy, and the soup gains in the paper appear with few ingredients. Each run is about 3 hours on the Spark when the GPU isn't shared (estimated from the Beetle + SAF E4B run, 2026-11-06), so 5 runs ≈ 15 GPU-hours. All five go in the soup, whatever their validation accuracy, because seeds don't produce "bad" ingredients and choosing among them would be a greedy soup in disguise.

### 5.4 Grading

All on the Spark, the whole split each time, through `scripts/spark_bench_scientsbank.sh`, which already skips anything graded:

| Version | Label |
|---|---|
| Untouched E4B | `e4b-spark-gb10` |
| Seeds 0–4 | `e4b-spark-gb10-ft-s0` … `-ft-s4` |
| Soup | `e4b-spark-gb10-ft-soup` |

7 versions × 5,835 answers. About 50 minutes per version on Nov 8's shared Spark, so about 6 hours. The Kaggle adapter graded on the Spark on Nov 8 (`e4b-spark-gb10-ft`: 79.1 / 80.2 / 76.5%) is reported beside the seeds as a cross-check of T4 vs GB10 training, outside the spread and the tests. **(Claude's call)**

### 5.5 What gets reported

Per split, accuracy and macro-F1, 3-way and 2-way:
- untouched, each seed, **single-seed mean ± SD** (sample SD, n = 5), min–max, the soup.
- Per-class recall for the soup and the seed mean ("contradictory" is E4B's weak class).

### 5.6 Comparisons (fixed before grading)

- **The family: 18 tests.** Per split: the soup against untouched E4B, and the soup against each of the five seeds. Paired McNemar on 3-way labels, **Holm over all 18 at once**. **(Claude's call)**
- 2-way beside it: the same 18 pairs, McNemar, its own Holm over the 18, as in [`compare_bar.py`](../../../src/edge_grader/compare_bar.py). Secondary, not part of the decision.
- The 5-point bar for the soup against untouched E4B, read from the same tests.
- Not tested, only described: single seeds against untouched, the Kaggle adapter, mean ± SD.
- The code reuses `compare.paired`, `compare.holm` and `compare.latest`, and writes `results/spark/compare-soup-scientsbank-*.json`.

## 6. Where the code changes

- `finetune.py`: `split_seed` and `init_seed` in the recipe; `--seed` no longer reaches the split. Test: the split is the same for `--seed 0` and `--seed 3`.
- A soup module (`edge_grader/soup.py`): stack N PEFT adapters along the rank, check their configs match, write the adapter and its config with `r` = N × 16. Test as in §5.2.
- `scripts/spark_finetune.sh`: `SEED=<n>` trains seed n into the `-s<n>` names; `SOUP="0 1 2 3 4"` builds the soup instead of training, then converts and load-checks it the same way.
- `scripts/spark_bench_scientsbank.sh`: `SOUP=1` grades the seven versions in §5.4.
- `compose.yaml`: services `finetune-e4b-s0` … `-s4`, `soup-e4b`, `bench-scientsbank-soup`.
- A comparison module (`edge_grader/compare_soup.py`) for §5.5–5.6, with a test on fixture runs.

## 7. Order and time

1. Owner approves this spec; code and tests on branch `claude/e4b-adapter-soup-ebb4b2`.
2. The owner agrees how the branch reaches the Spark checkout (the `e4b-examples` session also needs it). Smoke: `SMOKE=1` training with seeds 0 and 1, and a soup of the two smoke adapters, output in `data/smoke`.
3. Queue: `finetune-e4b-s0` … `-s4`, `soup-e4b`, `bench-scientsbank-soup`, chained with `AFTER=` behind the other session's container. About 15 h training + 6 h grading ≈ one day if the GPU isn't shared, longer if it is.
4. Compare (§5.6), write up in `docs/results/finetuning.md`, decision row in `docs/research-outline.md`.

## 8. Left to the builder

File and option names within these lines, and how the soup step reuses the conversion step in `spark_finetune.sh`.
