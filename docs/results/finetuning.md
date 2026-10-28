# Fine-tuning (stage 3)

*Written 2026-10-28. Accuracy on Kaggle; the Jetson runs (cost) are still to do. Design: [`../superpowers/specs/2026-10-21-finetuning-design.md`](../superpowers/specs/2026-10-21-finetuning-design.md). Raw files: [`../../results/kaggle/`](../../results/kaggle/) (`*-ft-*-20261028-*`) and [`../../results/finetune/`](../../results/finetune/).*

## Setup

- **Adapters:** QLoRA on the SciEntsBank train split, one per model, trained on a Kaggle T4 on 2026-10-27 (3 epochs, batch 1 × 16; the same settings as 2026-10-23, re-run as committed Kaggle versions so the adapter files are kept). The epoch is picked on 14 held-out train questions (499 answers):
  - E2B: 76.8 / **78.8** / 78.2% by epoch, so epoch 2 ([`finetune-e2b-20261027-111124.json`](../../results/finetune/finetune-e2b-20261027-111124.json))
  - E4B: **77.6** / 77.2 / 77.2%, so epoch 1 ([`finetune-e4b-20261027-113759.json`](../../results/finetune/finetune-e4b-20261027-113759.json))
- **Serving:** the adapter converted to GGUF and loaded with `--lora` on the unchanged QAT `UD-Q4_K_XL` GGUF, `llama-server` `a3a1c47`, thinking off. These are the same build and prompt as the untuned baselines.
  - GGUF sha256: `e2b-ft.gguf` `6507078a…f108b`, `e4b-ft.gguf` `53e507cb…fc67f`. Both are kept in the private Kaggle dataset `edge-grader-adapters`, not in git.
- **Where:** Kaggle T4, not the Jetson, at the owner's choice on 2026-10-28. The board was off the network that day, and Kaggle gives the same accuracy as the boards (18 of 20 phase 0 comparisons within noise, [summary](summary.md#rq1-how-good-are-the-untouched-models)). The Jetson runs are still needed for latency and energy.
- **Pairs:** each tuned run against the untuned Kaggle run on the same items and build:
  - SciEntsBank: 2026-10-05
  - MMLU-Pro: 2026-10-06
  - GSM: 2026-10-08
- **Test:** paired McNemar, Holm-corrected within the two families fixed in the design:
  - improvement: 3 splits × 2 models
  - no harm: MMLU-Pro, GSM8K, GSM-Symbolic × 2 models

## Smoke test first (supervisor, 2026-10-28)

The first 100 answers of test_uq and test_ud went first. The full splits ran only if accuracy rose:

| | E2B | E4B |
|---|---|---|
| test_uq (100) | 56 → **75%** (24 fixed / 5 broken, p 0.001) | 67 → **77%** (21 / 11, p 0.11) |
| test_ud (100) | 59 → **72%** (18 / 5, p 0.011) | 80 → 83% (11 / 8, p 0.65) |

Accuracy rose for both models, so the full splits went ahead. E4B's first 100 test_ud answers are easier than the whole split (untuned 80% here, 70.4% on all of it), which leaves less room to gain.

## SciEntsBank: the result

Full splits, 3-way, n = 540 / 733 / 4,562:

| | Untuned acc | **Tuned acc** | Untuned macro-F1 | Tuned macro-F1 | Fixed / broken | p (Holm) |
|---|---|---|---|---|---|---|
| E2B test_ua | 55.2% | **79.4%** | 0.518 | 0.790 | 165 / 34 | < 0.001 |
| E2B test_uq | 59.8% | **76.7%** | 0.553 | 0.753 | 168 / 44 | < 0.001 |
| E2B test_ud | 60.3% | **74.9%** | 0.570 | 0.748 | 918 / 252 | < 0.001 |
| E4B test_ua | 66.7% | **79.1%** | 0.661 | 0.769 | 103 / 36 | < 0.001 |
| E4B test_uq | 67.5% | **79.9%** | 0.653 | 0.786 | 137 / 46 | < 0.001 |
| E4B test_ud | 70.4% | **76.4%** | 0.705 | 0.774 | 645 / 371 | < 0.001 |

All 11,670 tuned answers parsed, with 0 errors. test_ua reproduces the 2026-10-27 sanity run (the same predictions on 100% of E2B's answers and 99% of E4B's). The table's E4B test_ua baseline is the 2026-10-05 run (66.7%). The 67.0% quoted elsewhere is the earlier 2026-09-29 run.

- **Both models beat their targets on the headline splits (test_uq and test_ud):**
  - E2B: 76.7 / 74.9% against about 65%
  - E4B: 79.9 / 76.4% against about 75%
- **The gain holds on unseen questions and unseen domains.** test_ua shares its questions with train, so its gain is expected. test_uq and test_ud gain +17 / +15 points for E2B and +12 / +6 for E4B.
- **E2B catches up with E4B.**
  - Tuned, the two are within 3 points on every split.
  - Untuned, E4B was 8–12 points ahead.

### Per class: the baseline's weak spot is fixed

Recall per class, untuned → tuned:

| | correct | contradictory | incorrect |
|---|---|---|---|
| E2B test_ua | 44 → **76%** | 66 → 72% | 63 → 85% |
| E2B test_uq | 48 → **73%** | 78 → 69% | 66 → 81% |
| E2B test_ud | 60 → **74%** | 79 → 78% | 57 → 75% |
| E4B test_ua | 45 → **77%** | 64 → 62% | 88 → 85% |
| E4B test_uq | 40 → **78%** | 67 → 67% | 90 → 83% |
| E4B test_ud | 57 → **77%** | 70 → 72% | 82 → 77% |

- **Recognising *correct* answers was the baseline's weakness, and it gains the most:** +14 to +38 points.
- **E2B no longer overuses "contradictory."**
  - On test_ud it predicted 1,105 contradictory untuned and 449 tuned, against 417 in the gold labels.
  - On test_uq: 203 untuned, 58 tuned, against 64 in the gold labels.
- **E4B trades some *incorrect* recall for it** (−3 to −7 points), because it now says "correct" more often.
- **The errors left are mostly correct ↔ incorrect.** On E4B test_ud, 435 correct answers were called incorrect and 490 incorrect answers were called correct. *Contradictory* is rarely confused with *correct*.

### Against the published numbers

SciEntsBank 3-way accuracy, the same splits and metric (Wang & Ormerod 2024, arXiv 2408.03811, Table 2):

| System | test_ua | test_uq | test_ud |
|---|---|---|---|
| Best SemEval-2013 systems | 72.0 | 65.3 | 64.0 |
| Fine-tuned BERT-base (Ghavidel 2020) | 72.6 | 70.8 | 67.2 |
| Claude 3.5 Sonnet + RAG (best published) | 78.7 | 76.9 | 73.4 |
| **E2B + QLoRA (this project, Q4 on llama.cpp)** | **79.4** | **76.7** | **74.9** |
| **E4B + QLoRA (this project, Q4 on llama.cpp)** | **79.1** | **79.9** | **76.4** |

E4B is above the best published numbers on every split. E2B is above them on test_ua and test_ud and 0.2 points below on test_uq. E4B is ahead of Claude 3.5 Sonnet + RAG by 3.0 points on test_uq and 3.0 on test_ud.

These are single runs with one seed and one prompt. The published systems differ in prompt, training and evaluation code, so read this as "on a par with the best published", not as a ranking.

## No harm: general ability

| | Untuned | Tuned | Untuned right only / tuned right only | p | p (Holm) |
|---|---|---|---|---|---|
| E2B MMLU-Pro (300) | 51.3% | 49.0% | 25 / 18 | 0.36 | 1.0 |
| E2B GSM8K (100) | 83% | 83% | 4 / 4 | 1.0 | 1.0 |
| E2B GSM-Symbolic (100) | 80% | 81% | 4 / 5 | 1.0 | 1.0 |
| E4B MMLU-Pro (300) | 65.3% | 66.0% | 20 / 22 | 0.88 | 1.0 |
| E4B GSM8K (100) | 92% | 90% | 4 / 2 | 0.69 | 1.0 |
| E4B GSM-Symbolic (100) | 91% | 90% | 2 / 1 | 1.0 | 1.0 |

**No significant drop anywhere.** The adapters taught grading without costing general ability or math.

- **GSM-Symbolic** is the first variant of each of the 100 templates, paired with the same items from the untuned 1,000-item run.
- **E2B's MMLU-Pro subset s2 is the one to watch.** It dropped 52 → 42% (12 / 2, p 0.013 on its own, outside the Holm family).
  - More than half of the drop is unanswered questions: 5 untuned, 11 tuned. Most of them hit the token cap.
  - The other two subsets went up: s1 52 → 53%, s3 50 → 52%. Pooled, E2B's change is not significant (p 0.36).
  - Read as: the adapter makes E2B ramble more on some long reasoning questions, not answer them worse.
- **E4B's subsets:** s1 67 → 65%, s2 64 → 65%, s3 65 → 68%.

## Not done yet

- **Cost on the Jetson.** Latency and energy per answer for the tuned models, against the re-run baselines in the same clock state. Kaggle times aren't board cost, so no cost number is quoted here.
  - Also the adapter's load check on the Jetson's build (`a894dae`). The Day 1 check on Kaggle passed; the Jetson half of it is still open.
- **The optional news run** (design §5.3), which was the first cut.
- **One seed.** Training was re-run once (2026-10-23 → 2026-10-27, same settings), and test_ua moved by 1 point (E2B 75 → 76%, E4B 72 → 73% on 100 answers). That gives a rough sense of run-to-run spread, not a measure of it.
