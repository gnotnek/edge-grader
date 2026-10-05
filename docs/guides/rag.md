# RAG pipeline: how it works and how to build it

*Stage 2. Written 2026-09-30 for someone new to RAG. The checklist is in [`../plan.md`](../plan.md) (stage 2). Why RAG is in the study: [`../research-outline.md`](../research-outline.md).*

## The idea in one paragraph

RAG (retrieval-augmented generation) means: before asking the model, look up a few relevant passages from some reference text and paste them into the prompt. For this project, `build_prompt` in `src/edge_grader/grader.py` already sends the question, the reference answer and the student answer. RAG adds one step before that: find the 3 most relevant textbook passages for the question and add them as "course material". **The model, the server, the label parsing and the metrics stay the same.**

## The pipeline

**Built once, offline:**

| Step | What | Tool |
|---|---|---|
| 1. Extract | Pull the text out of the textbook PDFs; keep the page number for every piece | PyMuPDF (`pymupdf`) |
| 2. Chunk | Split into passages of ~200–300 words, overlapping by ~50 words so no idea is cut in half | plain Python |
| 3. Embed | Turn each chunk into a vector: a list of numbers that captures its meaning | `sentence-transformers`, model `BAAI/bge-small-en-v1.5` (384 numbers per chunk) |
| 4. Save | Vectors in a `.npy` file, chunk text and source page in a `.jsonl` file | `numpy` |

A few thousand chunks don't need a vector database. A numpy array is enough.

**Done for every question:**

| Step | What |
|---|---|
| 5. Retrieve | Embed the question the same way, score it against every chunk (a dot product of normalised vectors, i.e. cosine similarity), take the top k |
| 6. Grade | Add those k passages to the prompt, then grade as before |

```
textbook PDFs ─► extract ─► chunk ─► embed ─► index (.npy + .jsonl)       once
                                                  │
question ─────────────────────────► embed ─► top-k passages ─► prompt ─► llama-server ─► label
```

Steps 5–6 are about 20 lines of code. Most of the real work is in choosing the material and getting steps 1–2 right.

## Things specific to this project

**1. Retrieve once per question, not once per answer.** SciEntsBank has only about 200 distinct questions, each answered by many students. Search with the question text and cache the passages it returns. That makes retrieval nearly free, and it makes the passages easy to check by hand.

**2. The prompt already contains the reference answer, so RAG has to earn its place.** The model already has the "right answer". The textbook can add context: synonyms, related ideas, and the reasoning behind the answer. That context should help with paraphrased correct answers, the baseline's weak spot (E2B overuses "contradictory"). If RAG doesn't help, that's still a result. In the analysis, look at recall on *correct* answers.

**3. Longer prompts cost time on the boards.** Grading is almost all prefill (reading the prompt). With the fitted rates in [`../results/edge-benchmarks.md`](../results/edge-benchmarks.md#grading-cost-estimated), 3 passages of 250 words (~1,000 tokens) add roughly **28 s per answer for E2B and 47 s for E4B on the Pi 5, and 1.8 s / 3.1 s on the Jetson**, on top of today's 5–8 s and 0.4–0.7 s. So k and the chunk size are cost settings as well as accuracy settings. The grader already records prompt token counts.

**4. Keep the baseline prompt unchanged.** With no passages, `build_prompt` must produce exactly the same text as today, so the stage-1 numbers stay comparable.

## The material

**Chosen and downloaded (2026-09-30):** English and Indonesian school science textbooks, plus Wikipedia in both languages for the topics the books miss. Sources, licences, coverage per SciEntsBank unit, and how to download them: [`rag-sources.md`](rag-sources.md).

- SciEntsBank comes from the FOSS elementary and middle-school science curriculum. FOSS is copyrighted, so its text can't be used.
- The material has to cover **every topic, including the test_ud domains** (unseen domains). Otherwise that split can't improve, and it's the most important one.

## Build order

Each step can be tested on its own. Steps 1–5 are built (2026-10-02); the tests use a fake encoder, so none of them runs a model.

| Step | Where | Command / code |
|---|---|---|
| 1. Dependencies | | `rag` group (`pymupdf`, `numpy`) for the Mac; `embed` group (`sentence-transformers`, with torch) for Kaggle |
| 2. Extract + chunk | **Mac** | `uv run --group rag python -m edge_grader.rag.corpus` → `data/rag/chunks/scientsbank-{en,id}.jsonl`, and the committed record `results/rag/chunks-scientsbank.json` (sha256, counts, settings). Code: `src/edge_grader/rag/corpus.py` |
| 3. Embed + retrieve | Mac for the small models (CPU-only models may run locally, 2026-10-02); Kaggle with a GPU for `bge-m3`: [`notebooks/kaggle_rag_index.ipynb`](../../notebooks/kaggle_rag_index.ipynb) | `python -m edge_grader.rag.index build` / `retrieve` → `results/rag/retrieval-<index>-k5.json`: top-5 chunk ids and scores for all 200 question ids. Code: `src/edge_grader/rag/index.py` |
| 4. Retrieval check | **Mac** | `uv run --group rag python -m edge_grader.rag.check results/rag/retrieval-….json` → `data/rag/check/*.md`, 30 train questions × top 3 passages, to mark by hand |
| 5. Grader | Kaggle | `edge-grader-baseline --rag results/rag/retrieval-….json --k 3`. Writes `rag-*.json` with the retrieval file, embedding model, k and chunks sha256. Without `--rag` the prompt is byte-for-byte the stage-1 prompt (a test pins it) |
| 6. Run on Kaggle | Kaggle | `notebooks/kaggle_baseline.ipynb`: set `RAG` to a retrieval file (and `K`). It copies in the chunks from the private dataset and checks their sha256 |

**Why the chunks are built on the Mac and uploaded to Kaggle:** the Wikipedia downloader fetches the *current* article text; the pinned revision ids only report changes. A fresh download on Kaggle could give different chunks and ids. So the chunks are built once, here, uploaded to Kaggle as a **private** dataset (the Indonesian books are all rights reserved), and the notebook checks them against the committed sha256.

**What gets committed:** the chunks record and the retrieval files (ids and scores only). Never the chunk text, the indexes or the check sheets. They stay in the git-ignored `data/`.

**Extraction choices** (`corpus.py`):
- Text comes from PyMuPDF's text blocks.
- **Pages cut:** the books' front matter (preface, contents, "how to use this book") and back matter (glossary, bibliography, index, author profiles).
- **Lines dropped:** image credits, and lines that repeat on at least 10% of a book's pages (running headers and footers).
- **Blocks dropped:**
  - under 8 words (captions, headings, list options, page numbers)
  - reference lists (URLs, "Diakses", ISBN, doi)
  - the books' fixed worksheet prompts ("What are you observing about this phenomenon?", "buku tugas", "Pelajar Pancasila", standards codes like "3.2.2 Analyze…")
  - blocks where at least half the sentences are questions or commands to the student (activity steps, end-of-chapter questions). The commands are a list of specific verbs ("Tulislah", "Diskusikan", "Explain"…), so ordinary sentences such as "Jumlah planet…" or "Work is done…" stay.
- **Fixed:** the broken ligatures (`昀椀`, `ٴ`, `ٶ`, `˪` → fi; `昀氀` → fl).
- **Not fixed:** words where the ligature vanished entirely ("Releksi"). There are only a few dozen, and they can't be told apart safely from real words.
- **Why these filters (2026-10-02):** in the first retrieval round, the Indonesian top hits were often reference lists, activity steps and reflection prompts: about 13 of 30 sample questions had a useful passage in the top 3, against about 23 of 30 for English. The filters cut the Indonesian book chunks from 1,075 to 716 and the English ones from 576 to 514. As a side effect they also drop Wikipedia's reference lists. The first round's retrieval files are kept in git history (commit `efd521e`).
- **Retrieval check uses train questions only.** test_ua shares those questions, but test_uq and test_ud questions are never shown, so the corpus isn't tuned on unseen questions.

## Settings (frozen 2026-10-03)

| Setting | Value | Why |
|---|---|---|
| Chunk size | 250 words | Long enough to hold one idea, short enough to keep the prompt small |
| Overlap | 50 words | An idea split at a boundary still appears whole in one chunk |
| k | 3 | Each passage costs prefill time on the boards |
| Embedding model, English | `bge-small-en-v1.5` (33M) | Retrieved well (~22/30 sample questions) and is small enough for the boards |
| Embedding model, Indonesian | `bge-m3` (568M) | English questions against Indonesian passages need a strong multilingual model. `multilingual-e5-small` and `-base` missed the cross-lingual matches (comparison in [`../results/rag.md`](../results/rag.md)) |
| Query | Question text only | The student answer varies; the question decides the topic |

The settings were tuned only against the retrieval check (train questions). Don't change them after grading starts; see [`../results/rag.md`](../results/rag.md).

## MMLU-Pro (the second baseline)

The same pipeline, pointed at the lab's 300-question MMLU-Pro subset and the OpenStax + SMA books ([`rag-sources.md`](rag-sources.md#part-b-mmlu-pro)). What differs from SciEntsBank:

| | SciEntsBank | MMLU-Pro |
|---|---|---|
| Runner | `edge-grader-baseline` (one short prompt per answer) | `edge-grader-mmlupro`: lm-eval's `mmlu_pro` chat messages, 5 worked examples, chain of thought up to 2,048 tokens |
| Without RAG | the stage-1 prompt, unchanged | **the lab's lm-eval prompt, byte for byte** (`tests/test_mmlupro.py`, from prompts recorded on the lab's runs), so "no RAG" here is the lab baseline |
| Where passages go | before the question, in the one user message | before the question, in the **last** user message only; the worked examples stay as they are |
| Query | the question text | the question text, **never the options** (they'd give the answer away to the retriever) |
| Retrieved for | every question of every split | the 300 test questions, plus the 70 validation questions for the check |
| Retrieval check | 30 train questions | 30 **validation** questions (the worked-example pool, never scored), over the 14 subjects |
| Chunks, English | 755 | 32,016 (25 OpenStax books, cut to their chapters by the PDFs' bookmarks) |

**Commands** (Mac, then Kaggle):

```bash
uv run --group rag python -m edge_grader.rag.corpus --benchmark mmlupro          # Mac: chunks, en + id
# upload data/rag/chunks/mmlupro-{en,id}.jsonl to the private Kaggle chunks dataset (New Version)
# Kaggle, kaggle_rag_index.ipynb: INDEXES = [("mmlupro-en", "bge-small-en")]   -> results-rag.zip
uv run --group rag python -m edge_grader.rag.check results/rag/retrieval-mmlupro-en.bge-small-en-k5.json
# mark data/rag/check/*.md, then --score; then kaggle_baseline.ipynb: TASK = "mmlupro",
# RAG = None, then RAG = "results/rag/retrieval-mmlupro-en.bge-small-en-k5.json"
```

**Both indexes are built on Kaggle.** `bge-small-en` runs on a CPU, but 32,016 chunks is ~40× SciEntsBank's, and on the owner's 8 GB Mac it swapped too hard to finish in reasonable time (2026-10-05). On a T4 it takes a minute or two.

**Settings:** the frozen SciEntsBank ones (250-word chunks, 50 overlap, k = 3, `bge-small-en-v1.5` / `bge-m3`). k = 3 adds roughly 1,000 tokens to a prompt that is already ~1,500–2,400, well inside the 8,192 context with the 2,048-token answer budget.

## When it doesn't work

| Symptom | Likely cause |
|---|---|
| Passages are on the right topic but the wrong grade level | Wrong material; try CK-12 at a lower level |
| Passages are mostly headers, captions or tables of contents | Extraction noise; filter short and non-sentence lines in step 1 |
| The right passage is found but split across two chunks | Chunks too small, or overlap too small |
| Accuracy drops with RAG | The model may be judging the student against the textbook instead of the reference answer; say in the prompt that the reference answer decides |
| Unparsed replies go up | The longer prompt pushes the "reply with the label only" instruction away; keep it at the end of the prompt |

## On the boards (stage 4, later)

The embedding model has to run on the board too. Either use `sentence-transformers` on the CPU (a small model is fine) or `llama-server --embedding` with a GGUF embedding model. Because retrieval is cached per question, it barely affects latency per answer. Report retrieval time separately anyway, as the plan says.
