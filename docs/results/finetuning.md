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

## Held-out grading sets: Beetle and SAF (2026-10-31)

**Why (supervisor, 2026-10-30):** the adapters were trained on SciEntsBank, so gains on SciEntsBank's splits can't show they grade better in general. MMLU-Pro and GSM can't show it either: they test knowledge and math, not grading, and only check for damage. The fair test is grading data the adapters never saw. The supervisor accepted this on 2026-10-30.

- **Beetle** (Dzikovska et al. 2013, SemEval-2013 Task 7, \*SEM 2013 pp. 263–274): SciEntsBank's sister corpus, basic electricity and electronics from the BEETLE II tutor. HF `nkazi/Beetle`, the same fields and 3-way labels. test_ua 439, test_uq 819. **Headline: 3-way accuracy and macro-F1.** SciEntsBank's train split has electricity questions (`EM.*`), so the topic isn't wholly new; the questions, students and annotators are.
- **SAF**, English (Filighera et al. 2022, ACL 2022 pp. 8577–8591): communication networks, university level. HF `Short-Answer-Feedback/saf_communication_networks_english`, test_ua 375, test_uq 479. Labels Correct / Partially correct / Incorrect, with no "contradictory", so **the headline is correct vs not (2-way)**: Partially correct counts as incorrect, as in SciEntsBank's 3-way, and the model's "contradictory" counts as incorrect too.
- **Setup:** the same prompt (it still says "a science question"), build `a3a1c47`, GGUFs and Oct 27 adapters as above; Kaggle T4, kernels `edge-grader-heldout-base` and `-ft`, code `a22f2b7`. Each tuned run is paired item by item with the untuned run from the same day. McNemar, Holm over the 8 comparisons.

| | Majority class | Untuned acc | **Tuned acc** | Untuned macro-F1 | Tuned macro-F1 | Fixed / broken | p | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| E2B Beetle test_ua | 40.1% | 53.5% | **59.0%** | 0.528 | 0.531 | 85 / 61 | 0.057 | 0.34 |
| E2B Beetle test_uq | 42.0% | 50.2% | **56.6%** | 0.497 | 0.499 | 157 / 104 | 0.001 | **0.008** |
| E4B Beetle test_ua | 40.1% | 56.7% | 60.1% | 0.535 | 0.525 | 71 / 56 | 0.21 | 0.86 |
| E4B Beetle test_uq | 42.0% | 50.3% | 52.0% | 0.479 | 0.451 | 101 / 87 | 0.34 | 1.0 |
| E2B SAF test_ua (2-way) | 64.0% | 61.9% | **68.8%** | 0.614 | 0.644 | 60 / 34 | 0.010 | 0.07 |
| E2B SAF test_uq (2-way) | 51.6% | 72.9% | 76.0% | 0.729 | 0.760 | 39 / 24 | 0.077 | 0.39 |
| E4B SAF test_ua (2-way) | 64.0% | 70.9% | 72.0% | 0.703 | 0.667 | 57 / 53 | 0.78 | 1.0 |
| E4B SAF test_uq (2-way) | 51.6% | 78.5% | 79.1% | 0.784 | 0.786 | 42 / 39 | 0.82 | 1.0 |

0 errors in all 16 runs; 3 unparsed (untuned E2B SAF test_ua), 0 tuned.

**Most of the SciEntsBank gain does not carry over.**

- On SciEntsBank's unseen questions the adapters added +17 (E2B) and +12 (E4B) points. On the held-out sets they add **+2 to +7, and only one of the 8 comparisons survives Holm** (E2B Beetle test_uq, +6.4).
- **E4B gains nothing that holds up:** +0.6 to +3.4 points, every p ≥ 0.21, and macro-F1 drops on 3 of 4.
- **E2B gains a little**, mostly on Beetle; with its smaller untuned start, it has more to catch up.

**What the adapters learned, on Beetle: SciEntsBank's label habits, not better judging.** Recall per class, untuned → tuned:

| | correct | contradictory | incorrect | Predicted "contradictory" (gold) |
|---|---|---|---|---|
| E2B test_ua | 54 → 83% | 49 → **22%** | 57 → 59% | 136 → 37 (111) |
| E2B test_uq | 52 → 82% | 46 → **16%** | 52 → 62% | 237 → 56 (244) |
| E4B test_ua | 56 → 89% | 30 → **17%** | 77 → 59% | 69 → 24 (111) |
| E4B test_uq | 54 → 69% | 23 → **11%** | 74 → 70% | 108 → 34 (244) |

- Tuned, both models say "correct" far more and "contradictory" rarely. That fits SciEntsBank, where contradictory is 10% of the train split the adapters learned from (499 of 4,969), but Beetle has 25–30%.
- So accuracy rises because "correct" is Beetle's biggest class, while macro-F1 stays flat (E2B) or falls (E4B): the gain on *correct* is paid for by *contradictory*.
- This supports the supervisor's concern in a narrower form. It isn't leakage (no test item was trained on), but **the adapters fit SciEntsBank's label distribution**, and that only partly transfers.

**Read with care:** one seed, one prompt (which says "science question" for SAF's networking questions), and SAF test_uq has only 5 questions. Accuracy on Kaggle, as before; no board runs.

**What it means for the next step:** a better recipe has to teach judging that transfers, for example training on more than one grading set and keeping one out, or balancing the classes. Changing the model alone, with the same SciEntsBank-only recipe, would likely meet the same limit; the held-out sets are now the test any new model or recipe has to pass.

## Reasoning-guided fine-tuning (E2B, stage 3b)

*Design: [`../superpowers/specs/2026-11-01-reasoning-finetune-design.md`](../superpowers/specs/2026-11-01-reasoning-finetune-design.md). The training and the grading are still to come.*

### Teacher reasons (2026-11-01)

- **Teacher:** Qwen3-14B Q4_K_M on a Kaggle T4, `llama-server` `a3a1c47`, thinking off, temperature 0, at most 128 tokens. It is given the gold grade and explains it.
- **Pilot 1** (50 train answers, prompt `cd39ab27`): none dropped, but 28 of the 29 reasons for "correct" answers listed shortcomings, and some argued for a lower grade. Training on them could teach the over-calling of "correct" that the 2026-10-28 adapters showed, so the prompt was tightened: the grade is final, and every sentence must support it ([summary](../../results/finetune/reasons-qwen3-14b-q4_k_m-n50-20261101-045757.json)).
- **Pilot 2** (the same 50, prompt `e73e9afc`): none of the 29 "correct" reasons argues against the grade; 2 dropped (4%) ([summary](../../results/finetune/reasons-qwen3-14b-q4_k_m-n50-e73e9afc-20261101-072837.json)). The prompt was changed after reading training answers only, never test data.
- **Full run** (all 4,969 train answers, prompt `e73e9afc`, 6.9 h): 4,819 kept, 150 dropped (3.0%, under the 5% limit). Every drop is `wrong_label`: a reason for an "incorrect" (116) or "contradictory" (34) answer that uses a grade word, usually "correct". No reason was cut at the token limit. Mean 52 words, longest 84 ([summary](../../results/finetune/reasons-qwen3-14b-q4_k_m-20261101-152101.json)).
  - The drops all fall on non-"correct" answers, so training has slightly more "correct" answers than before (about 5% of the incorrect and contradictory ones are left out).

### Owner spot check (2026-11-02): 27 of 30, passed

30 kept reasons drawn with `random.Random(0)` (15 incorrect, 11 correct, 4 contradictory), each marked faithful if it explains the gold grade and says something true about the answer. Gate: 25 of 30.

The owner marked **27 of 30** faithful. Of the 3 left unmarked:

- **Two are disagreements with the dataset's grade, not with the reason.** In `MX.10.453.1` ("there are bubbles") and `II.13b.576.1` ("loops are facing different direction"), the owner would grade the answer "correct"; the reason explains the gold "incorrect" accurately. Counted as marked; under the strict rule the count would be 29.
- **One is a fair miss.** `ST.54b3.469.1` (student: "A plant.", gold "correct"): the reason stretches the answer to fit the grade and ends with a shortcoming ("lacks specific details").

The owner's two grade disagreements are a small sign that SciEntsBank's "incorrect" can be strict for short answers that name the right evidence without the explanation.

## Not done yet

- **Cost on the Jetson.** Latency and energy per answer for the tuned models, against the re-run baselines in the same clock state. Kaggle times aren't board cost, so no cost number is quoted here.
  - Also the adapter's load check on the Jetson's build (`a894dae`). The Day 1 check on Kaggle passed; the Jetson half of it is still open.
- **The optional news run** (design §5.3), which was the first cut.
- **One seed.** Training was re-run once (2026-10-23 → 2026-10-27, same settings), and test_ua moved by 1 point (E2B 75 → 76%, E4B 72 → 73% on 100 answers). That gives a rough sense of run-to-run spread, not a measure of it.
