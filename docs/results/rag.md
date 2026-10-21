# RAG (stage 2)

*Started 2026-10-02. How the pipeline works: [`../guides/rag.md`](../guides/rag.md). The material: [`../guides/rag-sources.md`](../guides/rag-sources.md). Raw files: [`../../results/rag/`](../../results/rag/).*

## Conditions

| Condition | Status | Material | Embedding model | Why |
|---|---|---|---|---|
| **RAG-en** | **Main condition** (RQ2) | English: Utah OER SEEd grades 3–8 + Simple/English Wikipedia | `bge-small-en-v1.5` (33M) | SciEntsBank is English. The model is small enough to sit next to Gemma on a Pi |
| RAG-id | Optional, run only if time allows | Indonesian: Kurikulum Merdeka IPAS/IPA kelas III–IX + Indonesian Wikipedia | `bge-m3` (568M) | Answers "does material in another language still help?". No research question needs it, so it's the first thing to drop |
| RAG-both | **Dropped** (2026-10-03) | | | Adds a condition without answering a research question |

## Retrieval check (before any grading)

30 train questions (spread over the 12 train units), top 3 passages each. test_ua shares these questions, but test_uq and test_ud questions are never looked at. **"Useful" means the passage would help a teacher grade answers to the question.**

### Owner's marking (re-marked 2026-10-06): the numbers to quote

| Index | Top passage useful | Any of top 3 useful | Useful passages |
|---|---|---|---|
| **RAG-en**, `bge-small-en` | **13 / 30** | **16 / 30** | 18 / 90 (20%) |
| **RAG-id**, `bge-m3` | **13 / 30** | **18 / 30** | 21 / 90 (23%) |

- **Re-marked.** The first marking (2026-10-05) gave 29/30 and 30/30, because every question got at least one tick on the mistaken belief that each needed one. On 2026-10-06 the owner re-marked both sheets, ticking only passages that would actually help. The first marking is in git history; only these numbers should be quoted.
- **About half the questions get a useful passage, in both languages.** That's on a par with MMLU-Pro (15/30). It means the gate ("most questions get a useful passage") was **not** really met, although grading had already run on these frozen settings.
- **Indonesian is no worse than English** (18 vs 16 questions; the same 13 top-1 hits).
- **Files:** [`check-scientsbank-en.bge-small-en-k5.json`](../../results/rag/check-scientsbank-en.bge-small-en-k5.json) and [`check-scientsbank-id.bge-m3-k5.json`](../../results/rag/check-scientsbank-id.bge-m3-k5.json). They hold the ticks per passage, with chunk ids and no text. They're made by `python -m edge_grader.rag.check <retrieval file> --score`.
- **Only these two were marked.** The other models weren't, so the comparison below rests on Claude's first pass.

### Model comparison: Claude's first pass

Read from the first ~20 words of each passage, stricter than the owner's marking above. Use it only to compare models with each other, not as the retrieval quality.

| Round | Chunks | Index | Clear | Partial | Miss |
|---|---|---|---|---|---|
| 1 | first extraction | English, `bge-small-en` | ~19 top-1, ~23 top-3 | | |
| 1 | first extraction | Indonesian, `multilingual-e5-small` | ~10 top-1, ~13 top-3 | | |
| 2 | noise filters | English, `bge-small-en` | ~22 | ~3 | ~5 |
| 2 | noise filters | Indonesian, `multilingual-e5-small` (118M) | ~12 | ~5 | ~13 |
| 2 | noise filters | Indonesian, `multilingual-e5-base` (278M) | ~15 | | |
| 2 | noise filters | Indonesian, **`bge-m3`** (568M) | **~17** | ~8 | ~5 |

**What changed between rounds:**
- **Round 1 → 2, chunks:** the Indonesian top hits were often bibliography entries, author profiles, activity steps and reflection prompts. `corpus.py` now cuts front and back matter and drops references, worksheet prompts and student-directed blocks (rules in [`../guides/rag.md`](../guides/rag.md)). Book chunks: Indonesian 1,075 → 716, English 576 → 514.
- **Indonesian models:** `e5-small` failed on cross-lingual matches (lever ↔ *tuas/pengungkit*, pulley ↔ *katrol*). `e5-base` fixed levers, magnets and circuits. `bge-m3` also found pulleys, the pendulum (*bandul*), the electromagnet and standard units (*satuan*), and its scores spread wider (0.47–0.68 against 0.77–0.84), so no single generic passage won for unrelated questions.
- **Still missing in both languages:** questions the material doesn't cover, such as the calcite acid test, fabric rubbings and the ball-throwing variables question. That's a coverage gap, not a retrieval fault.

**Cost of the embedding model:** building the `bge-m3` index over 1,141 chunks took ~65 min on the owner's 8 GB Mac (swapping), against ~8 min for `e5-base` and ~2 min for the small models. Rebuild it on a Kaggle GPU. Grading never runs it: retrieval is done once per question and cached.

## Frozen settings (2026-10-03)

| Setting | Value |
|---|---|
| Chunks | 250 words, 50 overlap, noise filters as in `corpus.py`; sha256 in `results/rag/chunks-scientsbank.json` |
| Query | Question text only |
| Stored per question | top 5 |
| In the prompt | top 3 (`--k 3`) |
| English model | `BAAI/bge-small-en-v1.5` |
| Indonesian model | `BAAI/bge-m3` (dense vectors) |

Don't change these after grading starts. A change means new retrieval files and new grading runs, and a row in the change log of [`../plan.md`](../plan.md).

## Deployment framing (for stage 4)

A teacher's questions are known before grading, so passages are retrieved **once per question, ahead of time**, and stored. On the boards, grading latency and energy are measured with the passages already in the prompt. The embedding step is reported separately, as a one-off setup cost per question. That keeps the per-answer cost comparable across versions, and it's a realistic design rather than a shortcut.

## Files

| File | What |
|---|---|
| `results/rag/chunks-scientsbank.json` | What was chunked: settings, sha256, chunks per source |
| `results/rag/retrieval-scientsbank-en.bge-small-en-k5.json` | **RAG-en**: top 5 chunk ids and scores per question (200 question ids) |
| `results/rag/retrieval-scientsbank-id.bge-m3-k5.json` | **RAG-id** |
| `results/rag/retrieval-scientsbank-id.e5-{small,base}-multi-k5.json` | The model comparison above; not used for grading |
| Round 1 retrieval files | Git history, commit `efd521e` |

## Grading results: RAG-en (Kaggle T4, 2026-10-05)

Same models, prompt, splits and server flags as the stage-1 baseline, plus the top 3 English passages per question. Files: `results/kaggle/rag-*-en-k3-*.json` and `raw/`. The two 100-answer test runs (`…-071301`, `…-071359`) are kept, like the stage-1 pilots.

The baseline column is the **no-RAG re-run on the same llama.cpp build** (`a3a1c47`, `results/kaggle/baseline-*-20261005-*`), so each row differs only in the passages. See [the build check](#build-check-baseline-re-run-on-a3a1c47) below.

| Model | Split | n | Accuracy, baseline → RAG | Macro-F1 | Paired flips (better / worse) | McNemar p |
|---|---|---|---|---|---|---|
| E2B | test_ua | 540 | 55.2% → 53.3% (−1.9) | 0.518 → 0.510 | 60 / 70 | 0.43 |
| E2B | test_uq | 733 | 59.8% → 56.9% (−2.9) | 0.553 → 0.537 | 69 / 90 | 0.11 |
| E2B | test_ud | 4,562 | 60.3% → 55.1% (**−5.1**) | 0.570 → 0.516 | 373 / 606 | **<0.001** |
| E4B | test_ua | 540 | 66.7% → 65.6% (−1.1) | 0.661 → 0.633 | 31 / 37 | 0.54 |
| E4B | test_uq | 733 | 67.5% → 70.4% (+2.9) | 0.653 → 0.659 | 66 / 45 | 0.057 |
| E4B | test_ud | 4,562 | 70.4% → 70.7% (+0.3) | 0.705 → 0.698 | 260 / 247 | 0.59 |

Recall per class, baseline → RAG:

| Model | Split | correct | contradictory | incorrect |
|---|---|---|---|---|
| E2B | test_ua | 0.44 → **0.58** | 0.66 → 0.76 | 0.63 → 0.44 |
| E2B | test_uq | 0.48 → **0.60** | 0.78 → 0.81 | 0.66 → 0.50 |
| E2B | test_ud | 0.60 → **0.71** | 0.79 → 0.83 | 0.57 → 0.37 |
| E4B | test_ua | 0.45 → **0.53** | 0.64 → 0.66 | 0.88 → 0.78 |
| E4B | test_uq | 0.40 → **0.52** | 0.67 → 0.67 | 0.90 → 0.86 |
| E4B | test_ud | 0.57 → **0.59** | 0.70 → 0.77 | 0.82 → 0.79 |

**What it shows:**
1. **RAG doesn't improve overall grading.** E4B is unchanged (no split significant; test_uq +2.9 is borderline). E2B gets worse, significantly so on unseen domains (−5.1 points).
2. **It does fix the baseline's main weakness, recognising correct answers:** recall on *correct* rises on every split, by up to 14 points (E2B test_ua).
3. **But it buys that by grading more leniently,** not by grading better. Recall on *incorrect* falls on every split, by up to 20 points (E2B test_ud). E2B's "contradictory" overuse also gets worse: 1,105 → 1,522 on test_ud. The passages move the model's threshold rather than sharpening its judgement.
4. **For a model that's already lenient-prone, that's a net loss (E2B); for the strict E4B, it roughly cancels out.**
5. **Cost:** the prompt grows from ~160 to ~1,035 tokens per answer, about 6.3×. On the boards, where grading is almost all prefill, that means roughly 6× the time and energy per answer. Measuring it is stage 4.
6. **0 errors**, and only 3 and 4 unparsed replies (E2B test_uq and test_ud) out of 5,835.

### Build check: baseline re-run on `a3a1c47`

The RAG runs used llama.cpp `a3a1c47`; the original stage-1 baseline used `86ea01d`, because Kaggle cloned the latest llama.cpp each session. The notebook is now pinned to `a3a1c47`, and the no-RAG baseline was re-run on it (Kaggle T4, 2026-10-05, all six full splits). Paired against the original, answer by answer:

| Model | Split | n | Accuracy, `86ea01d` → `a3a1c47` | Same label | Flips (old only / new only right) | McNemar p |
|---|---|---|---|---|---|---|
| E2B | test_ua | 540 | 55.2% → 55.2% | 100% | 0 / 0 | 1.00 |
| E2B | test_uq | 733 | 59.8% → 59.8% | 100% | 0 / 0 | 1.00 |
| E2B | test_ud | 4,562 | 60.3% → 60.3% | 100% | 0 / 0 | 1.00 |
| E4B | test_ua | 540 | 67.0% → 66.7% | 99.3% | 3 / 1 | 0.62 |
| E4B | test_uq | 733 | 67.5% → 67.5% | 98.8% | 4 / 4 | 1.00 |
| E4B | test_ud | 4,562 | 70.4% → 70.4% | 98.7% | 26 / 26 | 1.00 |

- **The build doesn't move accuracy.** E2B gives the identical label on every one of 5,835 answers; E4B changes 0.7–1.3% of its labels, and those changes cancel out.
- **So the RAG results stand as measured.** E2B's −5.1 points on test_ud is the passages, not the build. The table above now uses the re-run as its baseline; against it no E4B row changes by more than 0.4 points, and no conclusion changes.
- **Kaggle latency moved more than accuracy** (E4B median 0.14 → 0.16–0.19 s): T4 sessions vary. That's another reason latency comes from the boards, not Kaggle.

## Grading results: RAG-id (Kaggle T4, 2026-10-05)

The same run with the top 3 **Indonesian** passages (`bge-m3`), on the pinned build `a3a1c47`, against the no-RAG re-run on that build. Files: `results/kaggle/rag-*-id-k3-*.json` and `raw/`. 0 errors and 0 unparsed replies out of 5,835 per model: English instructions with Indonesian passages didn't break the label-only replies.

| Model | Split | n | No RAG | RAG-en | RAG-id | RAG-id flips vs no RAG (better / worse) | McNemar p |
|---|---|---|---|---|---|---|---|
| E2B | test_ua | 540 | 55.2% | 53.3% | 52.6% (−2.6) | 65 / 79 | 0.28 |
| E2B | test_uq | 733 | 59.8% | 56.9% | 55.1% (**−4.6**) | 77 / 111 | **0.016** |
| E2B | test_ud | 4,562 | 60.3% | 55.1% | 52.9% (**−7.4**) | 309 / 646 | **<0.001** |
| E4B | test_ua | 540 | 66.7% | 65.6% | 65.6% (−1.1) | 43 / 49 | 0.60 |
| E4B | test_uq | 733 | 67.5% | 70.4% | 69.9% (+2.4) | 64 / 47 | 0.13 |
| E4B | test_ud | 4,562 | 70.4% | 70.7% | 66.2% (**−4.2**) | 208 / 399 | **<0.001** |

Recall per class (correct / contradictory / incorrect):

| Model | Split | No RAG | RAG-en | RAG-id |
|---|---|---|---|---|
| E2B | test_ua | 0.44 / 0.66 / 0.63 | 0.58 / 0.76 / 0.44 | 0.58 / 0.88 / 0.39 |
| E2B | test_uq | 0.48 / 0.78 / 0.66 | 0.60 / 0.81 / 0.50 | 0.64 / 0.88 / 0.42 |
| E2B | test_ud | 0.60 / 0.79 / 0.57 | 0.71 / 0.83 / 0.37 | 0.69 / 0.83 / 0.34 |
| E4B | test_ua | 0.45 / 0.64 / 0.88 | 0.53 / 0.66 / 0.78 | 0.56 / 0.76 / 0.72 |
| E4B | test_uq | 0.40 / 0.67 / 0.90 | 0.52 / 0.67 / 0.86 | 0.58 / 0.72 / 0.79 |
| E4B | test_ud | 0.57 / 0.70 / 0.82 | 0.59 / 0.77 / 0.79 | 0.60 / 0.81 / 0.69 |

**What it shows:**
1. **Indonesian material hurts more than English.** RAG-id is below RAG-en on 5 of 6 rows, and it also pulls E4B down significantly on unseen domains (−4.2), where RAG-en left E4B unchanged.
2. **The same mechanism, stronger:** recall on *correct* rises (up to +18 points, E4B test_uq), and recall on *incorrect* falls further than with RAG-en (E4B test_ud 0.82 → 0.69). "Contradictory" is used even more.
3. **It isn't explained by worse retrieval.** By the owner's re-marking, Indonesian retrieval was no worse than English (a useful passage in the top 3 for 18/30 sample questions against 16/30). So RAG-id's extra loss isn't about worse passages. Material in another language seems to be harder for these small models to use, or more distracting.
4. **Cost:** prompts grow to ~1,250–1,300 tokens per answer, about 8× the no-RAG prompt (Indonesian takes more tokens per word), against 6.3× for RAG-en.

**For RQ2:** adding course material, in either language, doesn't improve these small graders. It shifts them toward lenient grading, and it costs 6–8× the prompt length. The language result (English material helps more than, or hurts less than, Indonesian) is the secondary finding RAG-id was for.

## MMLU-Pro: retrieval check (owner's marking, 2026-10-06)

`bge-small-en` over the 32,016 English chunks (OpenStax, cut to chapters), the top 3 passages for 30 **validation** questions (the worked-example pool, never scored), spread over the 14 subjects. A tick means the passage holds knowledge that helps answer the question. File: [`check-mmlupro-en.bge-small-en-k5.json`](../../results/rag/check-mmlupro-en.bge-small-en-k5.json); retrieval for all 370 questions: `results/rag/retrieval-mmlupro-en.bge-small-en-k5.json`.

| | SciEntsBank, English | **MMLU-Pro, English** |
|---|---|---|
| A useful passage in the top 3 | 16 / 30 | **15 / 30** |
| Top passage useful | 13 / 30 | 15 / 30 |
| Useful passages | 18 / 90 | 19 / 90 |

| Subjects | Questions with a useful passage |
|---|---|
| biology, chemistry, history | all (7 / 7) |
| physics, math, economics, engineering, philosophy, health, psychology | half (1 of 2 each) |
| business | 1 / 3 |
| **law, other, computer science** | **none (0 / 6)** |

- **Retrieval works where a matching textbook exists and fails where the material is thin.** The plan predicted the thin subjects: OpenStax has no bar-exam law, "other" is a mixed category, and computer science lost Informatika XII.
- **All or nothing:** whenever any of the three passages helps, the top one does. The index either finds the topic or misses it entirely.
- **On a par with SciEntsBank (15/30 against 16/30 English after the re-marking).** Expect RAG-en to help, if at all, only in the well-covered subjects. Report the results per subject, split into well- and poorly-covered groups by this check.

### Does RAG help where a useful passage was found? (test_ua, inconclusive)

The 30 marked questions are train questions, which test_ua shares, so the test_ua answers to them can be split by whether the owner found a useful passage. Accuracy change, no RAG → RAG (same build):

| Condition | Model | Useful passage found | No useful passage |
|---|---|---|---|
| RAG-en | E2B | −8.9 points (56 answers; 2 better, 7 worse) | +10.7 (56; 11 / 5) |
| RAG-en | E4B | 0.0 (56; 4 / 4) | −1.8 (56; 0 / 1) |
| RAG-id | E2B | −2.9 (68; 8 / 10) | −6.8 (44; 2 / 5) |
| RAG-id | E4B | +4.4 (68; 5 / 2) | −11.4 (44; 0 / 5) |

**Too small to conclude anything.** Each cell rests on 44–68 answers and a handful of changed labels, and the direction isn't consistent (E2B does worse with a useful English passage; E4B better with a useful Indonesian one). It gives no support for "RAG helps when the passage is good". Testing that properly would need retrieval marked for many more questions.

## MMLU-Pro: grading results (Kaggle T4, 2026-10-06)

`edge-grader-mmlupro` on the lab's 3 × 100 subset (s1–s3), lm-eval's 5-shot chain-of-thought prompt, 2,048-token budget, llama.cpp `a3a1c47`. No RAG, then RAG-en: the top 3 OpenStax passages in front of the question. Files: `results/kaggle/mmlupro-{baseline,rag}-*` and `raw/`. The two 5-question RAG runs (`…-080057`, `…-080138`) are the quick test.

### The runner reproduces the lab baseline

The no-RAG run, paired question by question with the lab's lm-eval runs on the boards:

| Model | Lab, Pi 5 | Lab, Jetson | **This runner, no RAG** | Same right/wrong as the lab | McNemar p (Pi / Jetson) |
|---|---|---|---|---|---|
| E2B | 51.7% | 52.7% | **51.3%** | 87% | 1.00 / 0.63 |
| E4B | 65.7% | 66.0% | **65.3%** | 91–92% | 1.00 / 0.85 |

No significant difference from either board, so "no RAG" here is the lab baseline. The 8–13% of questions that differ are long chain-of-thought answers taking a different path on a different machine and build; the differences cancel out.

### RAG-en against no RAG (300 questions)

| Model | No RAG | RAG-en | Paired flips (better / worse) | McNemar p | No answer found | Hit the 2,048-token cap |
|---|---|---|---|---|---|---|
| E2B | 51.3% | 52.0% (+0.7) | 28 / 26 | 0.89 | 17 → 29 | 19 → 29 |
| E4B | 65.3% | 63.3% (−2.0) | 18 / 24 | 0.44 | 20 → 25 | 22 → 25 |

By how well the retrieval check found material for the subject:

| Subjects (from the check) | n | E2B, no RAG → RAG | E4B, no RAG → RAG |
|---|---|---|---|
| **Well covered:** biology, chemistry, history | 54 | 55.6% → 59.3% (+3.7; 6 / 4) | 63.0% → 68.5% (+5.6; 5 / 2) |
| **Partly:** physics, math, economics, engineering, philosophy, health, psychology, business | 186 | 54.8% → 54.3% (−0.5; 18 / 19) | 69.9% → 66.7% (−3.2; 9 / 15) |
| **Not covered:** law, "other", computer science | 60 | 36.7% → 38.3% (+1.7; 4 / 3) | 53.3% → 48.3% (−5.0; 4 / 7) |

No group is significant (all p ≥ 0.31).

**What it shows:**
1. **RAG doesn't change MMLU-Pro accuracy.** E2B is +0.7, E4B −2.0, both well within noise. That's the same answer as SciEntsBank: adding course material doesn't make these small models better.
2. **The direction follows the material,** though without significance. Both models gain in the well-covered subjects (+3.7 and +5.6), and E4B loses where the passages don't match (−3.2 and −5.0). Chemistry with E2B is the largest single gain (11 → 16 of 27). With 54–60 questions per group, a 5-point effect can't be told apart from chance.
3. **The passages cost reasoning budget.** Prompts grow from ~1,440 to ~2,430 tokens. Answers that run out of tokens before stating a choice rise (E2B 19 → 29 of 300). For E2B, 7 of the 26 questions it lost with RAG were cut off by the cap, against 3 it gained where the no-RAG answer had been cut off: about 4 questions net. For E4B it's 5 and 5, so no net effect.
4. **For RQ1–RQ2:** the subject-knowledge baseline doesn't move with retrieved knowledge either. Small models seem limited in using the material, not in having access to it, at least with this retrieval.

### Per subject, no RAG: math is the strongest subject (analysis 2026-10-07)

Done for the supervisor's 2026-10-07 question of which topic suits these models (logic, math, news, general knowledge). Same no-RAG runs as above (`results/kaggle/raw/mmlupro-baseline-*.jsonl`), math against the other 13 subjects pooled, two-sided Fisher's exact test, 95% Wilson interval:

| Model | Math | 95% interval | Other subjects | p |
|---|---|---|---|---|
| E2B | 26 / 33 = 78.8% | 62–89% | 128 / 267 = 47.9% | 0.0008 |
| E4B | 29 / 33 = 87.9% | 73–95% | 167 / 267 = 62.5% | 0.003 |

Both remain below a Bonferroni threshold for 14 subjects (0.05 / 14 = 0.0036). Math is the top subject for both models. RAG-en lowered it (E2B 78.8 → 69.7%, E4B 87.9 → 84.8%), in line with point 3 above.

How the supervisor's four topics map onto MMLU-Pro:

| Topic | MMLU-Pro subject | E2B / E4B, no RAG | Can MMLU-Pro answer it? |
|---|---|---|---|
| Math | math (33) | 78.8% / 87.9% | Yes, significant (above) |
| General knowledge | "other" (24), a mixed category | 33.3% / 58.3% | Weakly: small and mixed |
| Logic | none; formal logic sits inside philosophy (12) | 58.3% / 66.7% | No |
| News | none | — | No |

Logic and news need their own benchmarks. What was chosen instead is in the outline's [supervisor feedback](../research-outline.md#supervisor-feedback-so-far) (2026-10-07).

## MMLU-Pro RAG-id: retrieval check failed, not graded (2026-10-07)

`bge-m3` over the 1,670 Indonesian chunks (13 Kurikulum Merdeka SMA books), the same 30 validation questions as the English check. Owner's marking: **0 / 30** questions with a useful passage, 0 of 90 passages. That includes biology and chemistry, where Indonesian books exist and English found useful passages for 5 of 5. File: [`check-mmlupro-id.bge-m3-k5.json`](../../results/rag/check-mmlupro-id.bge-m3-k5.json).

- **Why:** the SMA books are a level below MMLU-Pro's college questions, there are no books for law, philosophy, psychology or engineering, and the material is small (1,670 chunks against 32,016 English).
- **Decision: not graded.** With no useful passage for any question, a grading run would only measure how much irrelevant text distracts the models, not whether Indonesian material helps. RAG-id was optional and the first to drop. The finding stands as it is: Indonesian school material has nothing usable for college-level MMLU-Pro.

## Baseline against RAG-en, all four benchmarks (phase A, 2026-10-10)

*The supervisor (2026-10-08) asked to compare each baseline with its RAG version. Kaggle T4, build `a3a1c47`, frozen RAG settings (`bge-small-en-v1.5`, question as query, top 3). Ordered from knowledge-heavy to skill-heavy.*

| Benchmark | Needs | E2B baseline → RAG-en | E4B baseline → RAG-en | Retrieval check (top 3) |
|---|---|---|---|---|
| News (LiveNewsBench, 200, after the cutoff) | facts the model can't have | 0.5% → **39.0%** | 0.5% → **45.0%** | 21 / 30 |
| MMLU-Pro (300) | broad subject knowledge | 51.3% → 52.0% (p 0.89) | 65.3% → 63.3% (p 0.44) | 15 / 30 |
| SciEntsBank grading, test_ud (4,562) | judging an answer | 60.3% → 55.1% (**p < 0.001**) | 70.4% → 70.7% (p 0.59) | 16 / 30 |
| Math, GSM8K (100) | a skill the model has | 83.0% → 78.0% (p 0.30) | 92.0% → 88.0% (p 0.29) | 21 / 30 |
| Math, GSM-Symbolic (1,000) | the same skill, new numbers | 80.0% → 75.1% (**p 0.04** by template) | 90.5% → 89.1% (p 0.24 by template) | (same) |

**Sources:**
- **News:** [`topics.md`](topics.md#news-results-kaggle-t4-judged-by-qwen3-14b-2026-10-08), judged by Qwen3-14B
- **MMLU-Pro:** [above](#mmlu-pro-grading-results-kaggle-t4-2026-10-06)
- **SciEntsBank:** [above](#grading-results-rag-en-kaggle-t4-2026-10-05). The other splits move the same way: E2B down on all three, E4B with no significant change on any.
- **Math:** [`topics.md`](topics.md#math-rag-en-results-kaggle-t4-2026-10-10)

**The answer to RQ2 across benchmarks:**
- **RAG helps only where the task is missing knowledge the passages supply:**
  - **News:** a large gain.
  - **MMLU-Pro:** a small, non-significant gain on well-covered subjects.
  - **Grading:** nothing for E4B and a loss for E2B.
  - **Math:** a small loss.
- **Retrieval quality doesn't explain the difference on its own:** math and news scored the same retrieval check (21/30) and went opposite ways.
- **The smaller model is hurt more whenever the material isn't needed:** on test_ud grading E2B loses 5.1 points where E4B gains 0.3, and on math E2B loses 4.9–5.0 where E4B loses 1.4–4.0.


## Jetson: baseline against RAG-en (stage 4, 2026-10-20)

*Why: the supervisor (2026-10-19) asked for RAG on every benchmark on the board, and dropped the Pi 5 for new runs. Plan: stage 4 ([`../plan.md`](../plan.md)).*

**Runs:**
- 12 folders in [`results/board/`](../../results/board/) (`rag-*`, `gsm-rag-*`, `news-rag-*`, all `-jetson-`), run by the Jetson's job queue (`scripts/board_queue.py add --preset rag`, jobs 24–35) between 2026-10-19 23:30 and 2026-10-20 11:44, all rc=0.
- Same GGUFs and server flags as phase 0 (`-ngl 99 -rea off --cache-ram 0`); the frozen retrieval files (`bge-small-en-v1.5`, question as query, top 3), the same files the Kaggle RAG runs used.
- **Jetson clocks:** higher limits than in phase 0 (see [Cost](#cost-rag-en-against-the-re-run-baseline)). So the baseline is the **re-run of 2026-10-21** in the same clock state (jobs 36–47, `results/board/*-jetson-*-20261021-*`), not phase 0's runs. Every re-run gives **byte-identical replies** to its phase 0 run (all 12 cells, 14,270 replies), so the accuracy below is unchanged; only cost moved.

**Check file:** [`board-check-20261021-065457.json`](../../results/estimates/board-check-20261021-065457.json), from `uv run edge-grader-board-check` on these 12 runs and the 12 re-run baselines. It replaces [`board-check-20261020-064504.json`](../../results/estimates/board-check-20261020-064504.json), which paired them with phase 0's baselines (kept; its RAG rows are the same). RAG against baseline: `board_check.compare` on the two runs' raw files; GSM-Symbolic also by template (two-sided sign-flip, 20,000 draws, seed 0), as in [`topics.md`](topics.md#math-rag-en-results-kaggle-t4-2026-10-10).

**Not in yet:**
- **News:** RAG-en run, not judged. Judge on Kaggle (Qwen3-14B), as for phase 0. The re-run baseline needs no judging: its replies are phase 0's, which are judged.
- **MMLU-Pro:** not run. Its 4 jobs (baseline + RAG-en, E2B + E4B) weren't in this queue; the two baseline re-runs (jobs 48, 49) failed on a subset-id file `deploy.sh` didn't ship (fixed in `a6c5066`) and were re-queued on 2026-10-21.

"Same answer" means the same label (grading) or the same extracted number (GSM), on the same questions.

### The board against Kaggle (RAG-en on both)

| Benchmark | Set | Model | Board | n | Board acc. | Kaggle acc. (same questions) | Same answer | McNemar p | Median s | J / question |
|---|---|---|---|---|---|---|---|---|---|---|
| SciEntsBank | test_ua | E2B | Jetson | 540 | 53.1% | 53.3% | 515/540 | 1.0 | 0.26 | 6.7 |
| SciEntsBank | test_ua | E4B | Jetson | 540 | 64.8% | 65.6% | 530/540 | 0.344 | 0.36 | 11.9 |
| SciEntsBank | test_uq | E2B | Jetson | 733 | 55.9% | 56.9% | 702/733 | 0.21 | 0.21 | 3.5 |
| SciEntsBank | test_uq | E4B | Jetson | 733 | 69.9% | 70.4% | 720/733 | 0.388 | 0.34 | 6.0 |
| SciEntsBank | test_ud | E2B | Jetson | 4562 | 55.7% | 55.1% | 4383/4562 | 0.05 | 0.21 | 3.3 |
| SciEntsBank | test_ud | E4B | Jetson | 4562 | 70.6% | 70.7% | 4471/4562 | 0.594 | 0.34 | 5.7 |
| GSM | gsm8k | E2B | Jetson | 100 | 78.0% | 78.0% | 85/100 | 1.0 | 5.00 | 125.3 |
| GSM | gsm8k | E4B | Jetson | 100 | 86.0% | 88.0% | 94/100 | 0.5 | 11.47 | 242.7 |
| GSM | symbolic (10 var.) | E2B | Jetson | 1000 | 74.8% | 75.1% | 833/1000 | 0.826 | 5.50 | 134.3 |
| GSM | symbolic (10 var.) | E4B | Jetson | 1000 | 88.4% | 89.1% | 932/1000 | 0.296 | 12.34 | 265.9 |
| News | test | E2B | Jetson | 200 | *to judge* | 39.0% | | | 21.87 | 429.5 |
| News | test | E4B | Jetson | 200 | *to judge* | 45.0% | | | 35.24 | 713.6 |

### RAG-en against the Jetson's own baseline

The baseline is the 2026-10-21 re-run of the same set, whose replies are identical to phase 0's Jetson run ([`baselines.md`](baselines.md#4-phase-0-every-baseline-on-both-boards-measured-2026-10-0910)). "RAG only / baseline only" counts the questions only one of the two got right.

| Benchmark | Set | Model | n | Baseline → RAG-en | Same answer | RAG only / baseline only | McNemar p | Kaggle, same cell |
|---|---|---|---|---|---|---|---|---|
| SciEntsBank | test_ua | E2B | 540 | 55.2% → 53.1% (−2.0) | 385/540 | 55 / 66 | 0.363 | 55.2% → 53.3% |
| SciEntsBank | test_ua | E4B | 540 | 66.7% → 64.8% (−1.9) | 468/540 | 28 / 38 | 0.268 | 67.0% → 65.6% |
| SciEntsBank | test_uq | E2B | 733 | 59.8% → 55.9% (−3.8) | 545/733 | 66 / 94 | **0.032** | 59.8% → 56.9% |
| SciEntsBank | test_uq | E4B | 733 | 67.3% → 69.9% (+2.6) | 620/733 | 63 / 44 | 0.081 | 67.5% → 70.4% |
| SciEntsBank | test_ud | E2B | 4562 | 60.2% → 55.7% (−4.5) | 3317/4562 | 395 / 601 | **< 0.001** | 60.3% → 55.1% |
| SciEntsBank | test_ud | E4B | 4562 | 70.1% → 70.6% (+0.5) | 4008/4562 | 269 / 245 | 0.31 | 70.4% → 70.7% |
| GSM | gsm8k | E2B | 100 | 83.0% → 78.0% (−5.0) | 70/100 | 6 / 11 | 0.332 | 83.0% → 78.0% |
| GSM | gsm8k | E4B | 100 | 91.0% → 86.0% (−5.0) | 84/100 | 3 / 8 | 0.227 | 92.0% → 88.0% |
| GSM | symbolic (10 var.) | E2B | 1000 | 82.1% → 74.8% (−7.3) | 728/1000 | 50 / 123 | **0.002** by template | 80.0% → 75.1% |
| GSM | symbolic (10 var.) | E4B | 1000 | 90.9% → 88.4% (−2.5) | 881/1000 | 26 / 51 | 0.056 by template | 90.5% → 89.1% |

- **GSM-Symbolic by template:** E2B 33 templates worse, 16 better; E4B 18 worse, 6 better. Item-level McNemar, for reference: < 0.001 and 0.006.
- **Macro-F1 (SciEntsBank)**, baseline → RAG, ua / uq / ud: E2B 0.522 → 0.508, 0.548 → 0.527, 0.570 → 0.521; E4B 0.658 → 0.629, 0.650 → 0.654, 0.701 → 0.697. Unparsed with RAG: E2B 5 (test_uq) and 2 (test_ud), 0 elsewhere.
- **Strict (`#### N`), gsm8k / symbolic:** E2B 56% → 45% / 48.5% → 41.3%; E4B 85% → 51% / 81.4% → 42.9%.

Recall per class, baseline → RAG-en:

| Model | Split | correct | contradictory | incorrect | "contradictory" given |
|---|---|---|---|---|---|
| E2B | test_ua | 0.45 → 0.57 | 0.71 → 0.76 | 0.61 → 0.44 | 157 → 195 |
| E2B | test_uq | 0.49 → 0.60 | 0.72 → 0.77 | 0.67 → 0.49 | 193 → 240 |
| E2B | test_ud | 0.61 → 0.71 | 0.79 → 0.83 | 0.56 → 0.38 | 1,100 → 1,507 |
| E4B | test_ua | 0.45 → 0.50 | 0.62 → 0.67 | 0.88 → 0.78 | 49 → 75 |
| E4B | test_uq | 0.40 → 0.50 | 0.67 → 0.67 | 0.90 → 0.86 | 64 → 89 |
| E4B | test_ud | 0.57 → 0.59 | 0.68 → 0.78 | 0.82 → 0.79 | 374 → 530 |

### Cost: RAG-en against the re-run baseline

**The Jetson ran this queue with higher clock limits than phase 0, B1 and B2.** From `board-state.txt`; `nvpmodel` says "15W 0" and the governor is `schedutil` in both. The 2026-10-21 baseline re-runs have the stage 4 limits (all 26 run folders from 2026-10-19 on agree):

| Limit | Phase 0, B1, B2 (to 2026-10-13) | Stage 4 RAG (2026-10-19/20) and the re-run baselines (2026-10-21) |
|---|---|---|
| GPU devfreq max | 612 MHz | 1,020 MHz |
| Memory (`bwmgr`) max | 2,133 MHz | 3,199 MHz |
| CPU max | 1,497.6 MHz | 1,728 MHz |

Effect on the baseline, same replies (phase 0 → re-run):

| | Phase 0 | Re-run, 2026-10-21 |
|---|---|---|
| Decode, grading (E2B / E4B) | 24.8 / 45.9 ms per token | 16.0 / 29.6 ms per token |
| Decode, GSM (E2B / E4B) | 41.5 / 81.7 ms per token | 25.8 / 51.6 ms per token |
| Median s, grading (E2B / E4B) | 0.40–0.42 / 0.70–0.72 | 0.25–0.27 / 0.44–0.45 |
| Mean power while grading | 9.1–9.7 W | 14.8–16.1 W |
| Idle power | 4.5–4.6 W | 5.4–5.5 W |
| J per answer, grading (E2B / E4B) | 3.8–4.0 / 6.8–7.1 | 4.0–4.1 / 7.1–7.4 |

So the new state is ~37% faster per answer for about 5% more energy: the board draws more power for less time.

**Baseline → RAG-en, same clock state** (median seconds and J per question, whole board DC draw):

| Benchmark | Set | Model | Median s | J / question | Change in J |
|---|---|---|---|---|---|
| SciEntsBank | test_ua | E2B | 0.27 → 0.26 | 4.1 → 6.7 | +63% |
| SciEntsBank | test_ua | E4B | 0.45 → 0.36 | 7.4 → 11.9 | +60% |
| SciEntsBank | test_uq | E2B | 0.27 → 0.21 | 4.0 → 3.5 | −13% |
| SciEntsBank | test_uq | E4B | 0.45 → 0.34 | 7.2 → 6.0 | −17% |
| SciEntsBank | test_ud | E2B | 0.25 → 0.21 | 4.0 → 3.3 | −15% |
| SciEntsBank | test_ud | E4B | 0.44 → 0.34 | 7.1 → 5.7 | −20% |
| GSM | gsm8k | E2B | 4.49 → 5.00 | 112.1 → 125.3 | +12% |
| GSM | gsm8k | E4B | 7.24 → 11.47 | 180.0 → 242.7 | +35% |
| GSM | symbolic (10 var.) | E2B | 4.67 → 5.50 | 119.8 → 134.3 | +12% |
| GSM | symbolic (10 var.) | E4B | 7.78 → 12.34 | 189.4 → 265.9 | +40% |
| News | test | E2B | 23.09 → 21.87 | 376.7 → 429.5 | +14% |
| News | test | E4B | 35.92 → 35.24 | 701.3 → 713.6 | +2% |

Embedding and retrieval time isn't included: the retrieval files are precomputed, as on Kaggle.

**Prompt tokens, sent → processed by the server** (per answer, from `server.log`):

| Run | Baseline | RAG-en |
|---|---|---|
| SciEntsBank test_ua | 166 → 165 | 1,030 → 286 |
| SciEntsBank test_uq | 164 → 162 | 1,039 → 83 |
| SciEntsBank test_ud | 156 → 153 | 1,035 → 71 |
| GSM-Symbolic | 1,106 → 70 | 2,074 → 855 |
| News | 219 → 180 | 1,130 → 1,094–1,100 |

### What it shows

- **RAG-en doesn't help grading or math on the Jetson either: the board gives the Kaggle answer again.** E2B is lower on all 5 of its cells, E4B on 3 of 5. After a Holm correction over the 10 tests (GSM-Symbolic by template), two drops stay significant, both E2B: test_ud (−4.5 points, p < 0.01) and GSM-Symbolic (−7.3, p 0.018). E2B test_uq (p 0.032) doesn't survive it (0.26). E4B has no significant change on any cell. Kaggle had the same pattern: E2B down on every grading split, E4B flat, math down for both.
- **The board reproduces Kaggle's RAG runs, as phase 0 did for the baseline.** All 10 scored cells agree with Kaggle's (McNemar p ≥ 0.05; the closest, E2B test_ud at 0.050, is 0.6 points *above* Kaggle). The same answer on 95–98% of grading questions and 83–94% of GSM ones, about phase 0's agreement. So RAG's effect is a property of the models and the material, not of the platform.
- **RAG makes the grader more lenient, not better.** Recall on *correct* rises on all six grading rows (by up to 12 points) and recall on *incorrect* falls on all six (by up to 18): with passages that explain the concept, the model credits answers it rejected without them. E2B also gives "contradictory" more often (1,100 → 1,507 on test_ud; on Kaggle 1,105 → 1,522). The baseline and RAG runs give the same label on only 71–74% (E2B) and 85–88% (E4B) of answers, so RAG moves many grades in both directions, and the gains and losses nearly cancel.
- **The smaller model is hurt more.** test_ud: E2B −4.5 against E4B +0.5. GSM-Symbolic: E2B −7.3 against E4B −2.5. The Jetson's GSM-Symbolic baseline for E2B was 2.1 points above Kaggle's (phase 0), so its drop here is larger than Kaggle's (−4.9), from the same RAG accuracy (74.8% against 75.1%).
- **With the passages, E4B stops writing GSM's answer format but still gets the number.** Strict accuracy falls from 85% to 51% on GSM8K and from 81% to 43% on GSM-Symbolic, while the final number (the headline metric) moves 2.5–5 points. On Kaggle: 85% → 51%, 81% → 42%.
- **RAG doesn't cost 6× on the Jetson; on most grading it costs less.** Against the baseline re-run in the same clock state: test_uq and test_ud grading get 13–20% cheaper and faster with RAG, test_ua 60–63% dearer, math 12–40% dearer (E4B most, 7 → 12 s median), news 2–14%. The grading result follows the prompt cache (next bullet): where many answers share a question, the long RAG prompt is mostly reused, and the baseline's short prompt isn't. Why RAG-en grading prompts process *fewer* tokens than the baseline's isn't checked. Earlier, against phase 0, this couldn't be read: the clock limits rose between 2026-10-13 and 2026-10-19 (cause unknown; the same window as the supervisor's change of its IP).
- **RAG grading prompts are mostly reused by llama.cpp's prompt cache (observed; mechanism not checked).** A RAG grading prompt is ~1,035 tokens, but on test_uq and test_ud the server processed only 71–83 of them per answer (92–93% reused): consecutive answers to one question share the instructions, passages, question and reference answer. test_ua reuses less (286 processed), which fits it having fewer answers per question. Without RAG the grading prompts get almost no reuse (153 of 156), as B2 found for llama.cpp without `--swa-full`. So RAG's 6× longer grading prompt costs far less than 6× the prefill.
- **Reuse follows the length of the shared prefix, not RAG as such.** GSM's baseline already reuses its fixed 5-shot prefix (70 of 1,106 tokens processed); with RAG the passages change per question, so 855 of 2,074 are processed. News shares little in either case (180 of 219; 1,094–1,100 of 1,130). A long shared prefix is reused with the phase 0 flags; the short baseline grading prompt isn't. Why, in llama.cpp's terms, isn't checked.
