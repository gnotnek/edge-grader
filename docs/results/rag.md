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

### Owner's marking (2026-10-05): the numbers to quote

| Index | Top passage useful | Any of top 3 useful | Useful passages |
|---|---|---|---|
| **RAG-en**, `bge-small-en` | **27 / 30** | **29 / 30** | 33 / 90 (37%) |
| **RAG-id**, `bge-m3` | **27 / 30** | **30 / 30** | 43 / 90 (48%) |

- **Both pass the gate:** almost every sample question gets a useful passage in its top 3. The English miss is SE.48 (surface area and temperature change).
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

| Model | Split | n | Accuracy, baseline → RAG | Macro-F1 | Paired flips (better / worse) | McNemar p |
|---|---|---|---|---|---|---|
| E2B | test_ua | 540 | 55.2% → 53.3% (−1.9) | 0.518 → 0.510 | 60 / 70 | 0.43 |
| E2B | test_uq | 733 | 59.8% → 56.9% (−2.9) | 0.553 → 0.537 | 69 / 90 | 0.11 |
| E2B | test_ud | 4,562 | 60.3% → 55.1% (**−5.1**) | 0.570 → 0.516 | 373 / 606 | **<0.001** |
| E4B | test_ua | 540 | 67.0% → 65.6% (−1.5) | 0.664 → 0.633 | 31 / 39 | 0.40 |
| E4B | test_uq | 733 | 67.5% → 70.4% (+2.9) | 0.655 → 0.659 | 64 / 43 | 0.053 |
| E4B | test_ud | 4,562 | 70.4% → 70.7% (+0.3) | 0.705 → 0.698 | 267 / 254 | 0.60 |

Recall per class, baseline → RAG:

| Model | Split | correct | contradictory | incorrect |
|---|---|---|---|---|
| E2B | test_ua | 0.44 → **0.58** | 0.66 → 0.76 | 0.63 → 0.44 |
| E2B | test_uq | 0.48 → **0.60** | 0.78 → 0.81 | 0.66 → 0.50 |
| E2B | test_ud | 0.60 → **0.71** | 0.79 → 0.83 | 0.57 → 0.37 |
| E4B | test_ua | 0.45 → **0.53** | 0.64 → 0.66 | 0.88 → 0.78 |
| E4B | test_uq | 0.41 → **0.52** | 0.67 → 0.67 | 0.90 → 0.86 |
| E4B | test_ud | 0.57 → **0.59** | 0.69 → 0.77 | 0.82 → 0.79 |

**What it shows:**
1. **RAG doesn't improve overall grading.** E4B is unchanged (no split significant; test_uq +2.9 is borderline). E2B gets worse, significantly so on unseen domains (−5.1 points).
2. **It does fix the baseline's main weakness, recognising correct answers:** recall on *correct* rises on every split, by up to 14 points (E2B test_ua).
3. **But it buys that by grading more leniently,** not by grading better. Recall on *incorrect* falls on every split, by up to 20 points (E2B test_ud). E2B's "contradictory" overuse also gets worse: 1,105 → 1,522 on test_ud. The passages move the model's threshold rather than sharpening its judgement.
4. **For a model that's already lenient-prone, that's a net loss (E2B); for the strict E4B, it roughly cancels out.**
5. **Cost:** the prompt grows from ~160 to ~1,035 tokens per answer, about 6.3×. On the boards, where grading is almost all prefill, that means roughly 6× the time and energy per answer. Measuring it is stage 4.
6. **0 errors**, and only 3 and 4 unparsed replies (E2B test_uq and test_ud) out of 5,835.

**Caveat, not yet controlled: a different llama.cpp build.** These runs used llama.cpp `a3a1c47`; the stage-1 baseline used `86ea01d`. Kaggle clones the latest llama.cpp each session. On the Jetson, a different build changed 6% of E2B's labels and 1% of E4B's, with no change in accuracy, so the build probably doesn't explain a 5-point drop. It isn't ruled out, though. The notebook is now pinned to `a3a1c47`, and a no-RAG baseline re-run on it is planned, which makes the comparison exact.
