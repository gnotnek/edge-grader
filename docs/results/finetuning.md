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

*Design: [`../superpowers/specs/2026-11-01-reasoning-finetune-design.md`](../superpowers/specs/2026-11-01-reasoning-finetune-design.md). E2B only; E4B, the Jetson cost and the no-harm checks come next if time allows (design §6).*

**In short:** the gate passed, in one of the four held-out cells. On Beetle test_ua both new adapters beat the untuned E2B (+11.4 points) and the Oct 28 adapter (+5.9), each significant after Holm. The other three cells show no significant gain over the Oct 28 adapter. "Contradictory" did not recover. The reason doesn't have to come first: label-first matches or beats reason-first on 3 of 4 cells at 2 tokens per answer instead of about 65.

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

### Training (2026-11-02)

The Oct 27 recipe unchanged except for the target text: the same base checkpoint, split, seed, library versions and epoch choice on the 499 validation answers. The 136 train answers whose reason was dropped are left out of both adapters.

| Adapter | Target | Chosen epoch | Validation acc by epoch | Summary |
|---|---|---|---|---|
| Oct 28 (label only, trained Oct 27) | `<label>` | | 78.8% (chosen) | |
| Reason-first | `<reason>\nLabel: <label>` | 3 | 74.3 / 72.7 / 76.4% | [json](../../results/finetune/finetune-e2b-reason-first-20261102-145522.json) |
| Label-first | `<label>\n<reason>` | 3 | 73.5 / 76.2 / 77.2% | [json](../../results/finetune/finetune-e2b-label-first-20261102-115213.json) |

- Converted to GGUF LoRA (`e2b-ft-rf.gguf` sha `190b81da`, `e2b-ft-lf.gguf` sha `282075cb`) and served with `--lora` on `a3a1c47`, as on Oct 28.
- Smoke run for reason-first only (10 answers, 0 unparsed, longest reply 98 of the 160-token cap). Label-first's smoke run was skipped to save GPU time; its first full run served as the check.
- Sanity runs on the first 100 SciEntsBank test_ua answers (not part of the test; test_ua shares questions with train): reason-first 67%, label-first 79%, the Oct 27 adapter 76% ([rf](../../results/kaggle/baseline-e2b-kaggle-t4-ft-rf-test_ua-3way-20261102-145733.json), [lf](../../results/kaggle/baseline-e2b-kaggle-t4-ft-lf-test_ua-3way-20261102-115258.json)).

### Grading and the gate (2026-11-02): passed, in one cell

Kaggle T4, the whole split each time, build `a3a1c47`, kernels `edge-grader-grade-{rf,lf}-{held,sb}`. 0 errors, 0 unparsed and 0 cut at the token cap in all 12 runs. The untuned and Oct 28 runs are the 2026-10-30 (Beetle, SAF) and 2026-10-05 / 10-28 (SciEntsBank) runs above, on the same build. Paired McNemar, Holm over the 16 tests, fixed before grading ([comparison](../../results/kaggle/compare-reason-ft-20261102-200813.json)).

| Cell | Untuned | Oct 28 | **Reason-first** | **Label-first** | RF vs untuned / Oct 28 (Holm p) | LF vs untuned / Oct 28 (Holm p) |
|---|---|---|---|---|---|---|
| Beetle test_ua (3-way) | 53.5% | 59.0% | **64.9%** | **64.9%** | +11.4 (**0.0006**) / +5.9 (**0.031**) | +11.4 (**0.0003**) / +5.9 (**0.009**) |
| Beetle test_uq (3-way) | 50.2% | 56.6% | 55.2% | 56.9% | +5.0 (0.15) / −1.5 (1.0) | +6.7 (**0.005**) / +0.2 (1.0) |
| SAF test_ua (2-way) | 61.9% | 68.8% | 65.1% | 67.2% | +3.2 (1.0) / −3.7 (0.91) | +5.3 (0.15) / −1.6 (1.0) |
| SAF test_uq (2-way) | 72.9% | 76.0% | 77.5% | 71.2% | +4.6 (0.28) / +1.5 (1.0) | −1.7 (1.0) / −4.8 (0.12) |

Macro-F1 (Beetle 3-way, SAF 2-way), untuned / Oct 28 / reason-first / label-first: Beetle test_ua 0.528 / 0.531 / 0.584 / 0.598; Beetle test_uq 0.497 / 0.499 / 0.485 / 0.525; SAF test_ua 0.614 / 0.644 / 0.619 / 0.658; SAF test_uq 0.729 / 0.760 / 0.772 / 0.710.

**The gate passed, but narrowly.**

- Both adapters pass in the same cell, Beetle test_ua, which is what design §2 asks for. There the Oct 28 adapter's own gain (+5.5) had not been significant.
- On the other three cells neither adapter is significantly above the Oct 28 adapter, and two point estimates are below it (label-first SAF test_uq −4.8, reason-first SAF test_ua −3.7; neither significant).
- So training on reasons adds something on Beetle's unseen answers and nothing that holds up elsewhere. One seed: the Oct 23 → Oct 27 re-run moved test_ua by about 1 point on 100 answers, so the +5.9 is beyond that, but a second seed would be the proper check.

**"Contradictory" did not recover.** The design's hope was that reasons would undo the Oct 28 adapter's fit to SciEntsBank's label mix. Beetle recall per class:

| | correct | contradictory | incorrect | Predicted "contradictory" (gold) |
|---|---|---|---|---|
| test_ua: untuned / Oct 28 / RF / LF | 54 / 83 / 89 / 88% | 49 / 22 / 23 / 28% | 57 / 59 / 68 / 66% | 136 / 37 / 31 / 44 (111) |
| test_uq: untuned / Oct 28 / RF / LF | 52 / 82 / 79 / 78% | 46 / 16 / 15 / 25% | 52 / 62 / 63 / 58% | 237 / 56 / 52 / 96 (244) |

- The Beetle test_ua gain comes from "incorrect" (+7 to +9 points of recall over Oct 28) and "correct" (+5), not from "contradictory".
- Label-first predicts "contradictory" somewhat more (25–28% recall against 15–23%), but still far below the untuned model's 46–49%. The label mix the adapters learned is still SciEntsBank's.

**SciEntsBank (described, not tested): slightly below the Oct 28 adapter.**

| | Untuned | Oct 28 | Reason-first | Label-first |
|---|---|---|---|---|
| test_uq (733) | 59.8% | 76.7% | 75.2% | 76.4% |
| test_ud (4,562) | 60.3% | 74.9% | 71.0% | 73.2% |

This matches the validation numbers: reasons don't help on the dataset the adapters were trained on, which the Oct 28 adapter already fits.

**Reason-first against label-first: the reason doesn't have to come first.**

- Accuracy: equal on Beetle test_ua; label-first ahead on Beetle test_uq (+1.7) and SAF test_ua (+2.1), reason-first ahead on SAF test_uq (+6.3); label-first ahead on SciEntsBank (+1.2 test_uq, +2.2 test_ud). Not tested against each other (design §5.5).
- Cost: reason-first writes a median of 63–72 tokens per answer, label-first 2 (the server stops at the label). On the T4 that made reason-first's runs 7.5–9× longer (Beetle test_ua 420 s against 47 s; SciEntsBank test_ud 4,495 s against 601 s). On the Jetson the gap would be larger, since decoding dominates there.
- So for a device, **label-first is the one to carry forward**: it gets the reason's training benefit at the label-only cost. This matches Faza et al. 2025's finding that the reason helps training, without their 35× inference cost.

**Read with care:** one seed; the gate passed in one of four cells; Kaggle accuracy only, no board runs yet. The teacher prompt was changed once, after reading training answers only (pilot 1 → pilot 2), and the epoch was chosen on validation, never on test or held-out data.

## Choosing the next method: is the leniency SciEntsBank's label mix? (2026-11-05)

**Why:** on the held-out sets the Oct 28 E2B adapter's main real error is calling wrong answers "correct" (Beetle test_uq: 45% of contradictory and 34% of incorrect answers; SAF test_ua: 53% of incorrect ones). Missing "contradictory" is mostly contradictory → incorrect, which costs nothing 2-way. Balanced training was the cheapest candidate, so it was checked first, without training.

**How** ([`label_prior.py`](../../src/edge_grader/label_prior.py)): the Oct 28 E2B adapter (Kaggle T4, `a3a1c47`, `--lora e2b-ft.gguf`) grades 500 answers each from **Beetle train and SAF train** (seed 0), which are never trained or tested on, so the test splits stay unseen. It reads the first reply token's top-20 logprobs. *Plain* takes the most likely label. *Corrected* first divides by each label's share in SciEntsBank train (incorrect 49.6%, correct 40.4%, contradictory 10.0%), a fixed rule with nothing tuned. Every reply's first token was a label.

| | Beetle train 3-way | macro-F1 | 2-way | Wrong answers called correct | SAF train 2-way | Wrong answers called correct |
|---|---|---|---|---|---|---|
| Plain | 60.6% | 0.540 | 75.4% | 29.4% | 72.8% | 44.1% |
| Corrected for the label mix | 61.6% | 0.571 | 74.6% | 30.5% | 71.6% | 45.9% |

([Beetle](../../results/assess/beetle-label-prior-e2b-kaggle-t4-ft-train-20261104-163640.json), [SAF](../../results/assess/saf-label-prior-e2b-kaggle-t4-ft-train-20261104-163826.json))

- **The label mix explains the missing "contradictory", not the leniency.** Correcting brings "contradictory" back on Beetle (recall 28 → 43 of 125), but it takes from "incorrect" (95 → 86 of 154). Leniency rises slightly on both sets, and 2-way accuracy falls by about a point.
- **SciEntsBank train has more incorrect than correct answers**, so balancing can't make the adapter stricter. It shifts weight toward "correct" relative to "incorrect".
- **The lenient errors are confident.** On Beetle, wrong answers called "correct" have a median p(correct) of 0.90, against 0.97 for right ones. The model believes them, so no threshold separates them well. On SAF, the gap is wider (0.73 against 0.85).

**Decision:** no balanced-training run. It would buy about a point of 3-way accuracy on Beetle at a small cost in 2-way, and the leniency would stay.

The leniency looks like a judging problem on new subjects, not a label-frequency problem. The candidates that act on it are training on more than one grading set, and a reward on the grade (GRPO). Only the first keeps the 2-token reply, so only it costs the device nothing extra (RQ4). Both are optional under the 2026-11-04 focus on edge cost.

## Trained on Beetle and SAF, tested on SciEntsBank (2026-11-06, grading on Kaggle)

**Why:** SciEntsBank is the main benchmark again, and adapters trained only on other grading sets test whether fine-tuning teaches grading that transfers. No SciEntsBank question appears in Beetle or SAF, and 5 of its 5,835 test answers match a training answer word for word.

**Teacher reasons** (for the label-first and reason-first adapters): Qwen3-14B on the DGX Spark, prompt `e73e9afc`, 5.7 h. 5,575 of 5,641 kept; 66 dropped (1.2%): 57 `wrong_label`, 9 `cut`. Mean 56 words, longest 103 ([summary](../../results/finetune/reasons-qwen3-14b-q4_k_m-beetle-saf-20261105-233653.json)).

**Owner spot check (2026-11-06): 29 of 30, passed** (gate 25; 30 kept reasons by `random.Random(0)`: 22 Beetle, 8 SAF).

- The one left unmarked, `SC.Q12.SBJ26.L1`, is a disagreement with the dataset's grade, not with the reason. The student's "it closes the pathway" is graded contradictory, and the owner would call it incorrect.
- Two ticked items carry the same note: both are graded contradictory, and the owner would grade them differently.
- So all three of the owner's doubts are about Beetle's contradictory label. Where "contradictory" ends and "incorrect" begins is unclear to a person too, which fits the adapters' trouble with that grade.

**Training (DGX Spark, 2026-11-05 → 11-07):** all six adapters trained and converted; validation on 6 held-out questions (491 answers) picked the epoch.

| Adapter | Validation by epoch | Chosen |
|---|---|---|
| E2B label-only | 65.8 / 64.2 / 65.6% | 1 |
| E2B label-first | 62.1 / 61.1 / 61.5% | 1 |
| E2B reason-first | 53.4 / 57.0 / 57.2% | 3 |
| E4B label-only | 64.8 / 68.4 / 67.4% | 2 |
| E4B label-first | 67.2 / 65.6 / 68.2% | 3 |
| E4B reason-first | 61.1 / 61.7 / 60.7% | 2 |

**Grading moved from the Spark to Kaggle (owner, 2026-11-08).** From 22:50 on Nov 6 another lab job shared the Spark's GPU, and steps took 2–2.5× longer. The owner stopped the Spark benchmark after E4B's SciEntsBank adapter and moved the rest to Kaggle T4s, so the comparison runs on one unshared machine type. Kaggle already had untouched E2B and E4B (Oct 5) and E2B `-ft`, `-ft-lf`, `-ft-rf` and E4B `-ft` (Oct 28, Nov 2) on all three splits, same llama.cpp `a3a1c47` and flags: those are reused, not re-run. Kaggle grades the missing eight: E4B `-ft-lf` and `-ft-rf`, and the six Beetle + SAF adapters (labels `…-ft-beetle-saf`, `…-ft-lf-beetle-saf`, `…-ft-rf-beetle-saf`).

The Spark runs in `results/spark/` (E2B's six adapters and E4B `-ft` on all splits, 100-answer checks of the rest) stay as a cross-check, not the comparison:

- Sharing didn't change grades. Two adapters graded the same 100 answers before the sharing (Nov 6) and during it (Nov 7): 99 of 100 identical replies each.
- Spark and Kaggle agree. E2B `-ft`, `-ft-lf`, `-ft-rf` on test_uq and test_ud: same label on 98.3–99.3% of answers (95.4–96.0% for reason-first), accuracy within 0.3 points.
- Spark timings (latency, wall time) are inflated by the sharing and aren't comparable; cost comes from the Jetson anyway.

### Result: transfer helps E2B, not E4B (2026-11-08)

**In short:** E2B's label-only Beetle + SAF adapter passes the 5-point bar on all three SciEntsBank splits, with no SciEntsBank answer in its training. On test_uq it matches the adapter trained on SciEntsBank itself (76.4 vs 76.7%). None of E4B's Beetle + SAF adapters passes anywhere, and the ones that write reasons make it significantly worse on test_ud.

Kaggle T4, the whole split each time, build `a3a1c47`. The 15 new runs (Nov 8: E4B `-ft-lf`, `-ft-rf`, the six Beetle + SAF adapters) had 0 errors, 0 unparsed, 0 cut. The others are the Oct 5, Oct 28 and Nov 2 runs on the same build. Each adapter is paired with its own untouched model on the same answers: McNemar on the 3-way labels, Holm over all 30 tests at once. The test_ua cells of the SciEntsBank-trained adapters are left out, since they trained on its questions. **This family was set after the runs were graded** (2026-11-08), so it's recorded as such ([comparison](../../results/kaggle/compare-bar-scientsbank-20261108-135709.json), `uv run python -m edge_grader.compare_bar`).

Accuracy, change against the untouched model, and Holm p. **Bold** passes the bar (+5 points, Holm p < 0.05).

| Adapter | test_ua | test_uq | test_ud |
|---|---|---|---|
| *E2B untouched* | *55.2%* | *59.8%* | *60.3%* |
| E2B, SciEntsBank, label-only (Oct 28) | trained on its questions | **76.7%, +16.9** (<0.001) | **74.9%, +14.6** (<0.001) |
| E2B, SciEntsBank, label-first | trained on its questions | **76.4%, +16.7** (<0.001) | **73.2%, +13.0** (<0.001) |
| E2B, SciEntsBank, reason-first | trained on its questions | **75.2%, +15.4** (<0.001) | **71.0%, +10.7** (<0.001) |
| E2B, Beetle + SAF, label-only | **69.3%, +14.1** (<0.001) | **76.4%, +16.7** (<0.001) | **68.4%, +8.2** (<0.001) |
| E2B, Beetle + SAF, label-first | 59.6%, +4.4 (0.29) | **68.6%, +8.9** (<0.001) | 57.5%, −2.8 (0.003) |
| E2B, Beetle + SAF, reason-first | 56.1%, +0.9 (1) | **67.4%, +7.6** (0.006) | 59.5%, −0.8 (1) |
| *E4B untouched* | *66.7%* | *67.5%* | *70.4%* |
| E4B, SciEntsBank, label-only (Oct 28) | trained on its questions | **80.0%, +12.4** (<0.001) | **76.4%, +6.0** (<0.001) |
| E4B, SciEntsBank, label-first | trained on its questions | **78.6%, +11.1** (<0.001) | 73.1%, +2.7 (<0.001) |
| E4B, SciEntsBank, reason-first | trained on its questions | **76.4%, +8.9** (<0.001) | 69.6%, −0.8 (1) |
| E4B, Beetle + SAF, label-only | 65.4%, −1.3 (1) | 71.2%, +3.7 (0.23) | 67.5%, −2.9 (<0.001) |
| E4B, Beetle + SAF, label-first | 61.7%, −5.0 (0.14) | 67.8%, +0.3 (1) | 63.4%, −7.0 (<0.001) |
| E4B, Beetle + SAF, reason-first | 58.0%, −8.7 (0.007) | 68.8%, +1.2 (1) | 61.3%, −9.1 (<0.001) |

- **15 of 30 pass.** All six SciEntsBank-trained E2B cells, E4B's Oct 28 adapter on both splits, E4B label-first and reason-first on test_uq only, and five of nine E2B Beetle + SAF cells.
- **Grading learned on other sets carries over for E2B, mostly through the label-only adapter.** The unseen domains (test_ud) are where transfer is weakest: +8.2 from Beetle + SAF against +14.6 from SciEntsBank's own train split.
- **E4B loses from Beetle + SAF training.** It already starts higher, and the adapters pull test_ud down 3–9 points. Macro-F1 falls further than accuracy (test_ud: 0.705 untouched, 0.587 label-only, 0.548 reason-first), so the drop is uneven across the three labels.
- **Writing a reason doesn't help here either.** Label-only scores highest of the three replies in all 15 model × training set × split cells. The same held for the SciEntsBank-trained adapters on Beetle and SAF (stage 3b).
- **E4B's SciEntsBank label-first and reason-first** (graded in full for the first time) pass on test_uq and fall short on test_ud, where label-first gains a significant +2.7, below the bar.
- One seed per adapter, and the validation sets are small (6 questions for Beetle + SAF), so the label-only vs reason gap is a single comparison, not a measured spread.

### Why E4B loses: the contradictory/incorrect boundary (2026-11-09, post hoc)

**Question (owner):** would a stronger teacher for the reasons fix E4B? The reasons already come from Qwen3-14B and passed the spot check (29 of 30), and E4B's label-only adapter, which sees no reasons, loses too. So the check was where E4B's points go: per-label recall, and the same runs scored as correct vs not-correct (contradictory merged into incorrect). No new grading; the 18 Beetle + SAF tests from the table above. **Set after the results were known**, so it explains the result and doesn't count toward the 5-point bar ([comparison](../../results/kaggle/compare-boundary-scientsbank-20261109-054815.json), `uv run python -m edge_grader.compare_boundary`).

**In short:** E4B learned to grade from Beetle + SAF. It also learned Beetle's idea of "contradictory", which doesn't match SciEntsBank's, and that costs more than it gains. A stronger teacher explains the gold grades it's given, so it wouldn't change this.

Change against untouched E4B, 3-way → 2-way accuracy points (2-way Holm p over the 18 tests):

| E4B, Beetle + SAF | test_ua | test_uq | test_ud |
|---|---|---|---|
| label-only | −1.3 → +3.1 (0.20) | +3.7 → **+6.6** (<0.001) | −2.9 → +2.5 (<0.001) |
| label-first | −5.0 → +3.9 (0.03) | +0.3 → **+7.0** (<0.001) | −7.0 → +1.5 (0.03) |
| reason-first | −8.7 → +3.0 (0.39) | +1.2 → **+9.1** (<0.001) | −9.1 → +2.0 (0.006) |

- **All nine E4B cells go up on 2-way,** seven significantly. Untouched E4B's 2-way accuracy is 72.8 / 72.6 / 74.7%.
- **The label-only adapter under-calls "contradictory".** On test_ud it predicts the label as often as before (7%, gold 9%) but finds 31% of the real ones, against 70% untouched.
- **The adapters that write reasons over-call it.** About 20% of answers predicted contradictory against 9–11% gold, and "incorrect" recall falls (test_ud: 0.82 → 0.62 label-first, 0.54 reason-first).
- **E2B shows the same over-calling with reasons** (19–26% predicted contradictory), but it starts from a lower base (its untouched model calls 24–27% contradictory), so it loses less.
- **Matches the owner's spot check,** where all three doubts were about Beetle's contradictory label. SAF is 2-way, so the training set's only contradictory answers are Beetle's.
- **What would act on it:** training with Beetle's contradictory merged into incorrect, or on SciEntsBank directly (E4B's SciEntsBank label-only adapter already passes). Not planned; noted for the owner's choice.

## Not done yet

- **Cost on the Jetson.** Latency and energy per answer for the tuned models, against the re-run baselines in the same clock state. Kaggle times aren't board cost, so no cost number is quoted here.
  - Also the adapter's load check on the Jetson's build (`a894dae`). The Day 1 check on Kaggle passed; the Jetson half of it is still open.
- **The optional news run** (design §5.3), which was the first cut.
- **One seed.** Training was re-run once (2026-10-23 → 2026-10-27, same settings), and test_ua moved by 1 point (E2B 75 → 76%, E4B 72 → 73% on 100 answers). That gives a rough sense of run-to-run spread, not a measure of it.
