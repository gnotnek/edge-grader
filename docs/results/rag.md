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

**These counts are a first pass by Claude**, reading the first ~20 words of each passage, not a careful marking. Replace them with the owner's own marking of `data/rag/check/*.md` before quoting them.

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

## Grading results

*Not run yet. Next: `notebooks/kaggle_baseline.ipynb` with `RAG` set, E2B + E4B, all three splits.*
