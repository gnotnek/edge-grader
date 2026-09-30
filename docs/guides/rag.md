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

**3. Longer prompts cost time on the boards.** Pi 5 prompt-reading (prefill) speed is ~70 tok/s with E2B and ~34 tok/s with E4B ([`../results/edge-benchmarks.md`](../results/edge-benchmarks.md)). 3 passages of ~300 tokens each add roughly **13 s per answer on E2B and 26 s on E4B**. So k and the chunk size are cost settings as well as accuracy settings. The grader already records prompt token counts.

**4. Keep the baseline prompt unchanged.** With no passages, `build_prompt` must produce exactly the same text as today, so the stage-1 numbers stay comparable.

## Choosing the material (the first decision)

- SciEntsBank comes from the FOSS elementary and middle-school science curriculum. FOSS is copyrighted, so its text can't be used.
- **CK-12** is at a similar level and free to use (check its current licence, which has changed over the years), so it's probably a better fit than **OpenStax**, which is college-level.
- It has to cover **every topic, including the test_ud domains** (unseen domains). Otherwise that split can't improve, and it's the most important one.
- Note the source, the licence and the download date of every file.

## Build order

Each step can be tested on its own.

1. **Dependencies:** `uv add --group rag sentence-transformers numpy pymupdf` (already pencilled in `pyproject.toml`).
2. **`src/edge_grader/rag/corpus.py`:** PDF → chunks, each with its source file and page. Unit test on a tiny text: chunk sizes, overlap, no empty chunks.
3. **`src/edge_grader/rag/index.py`:** embed the chunks and save them; `retrieve(question, k)` returns the top-k passages with their scores. Test with a handful of hand-written chunks where the right answer is obvious.
4. **Retrieval check, before any grading:** for ~30 questions, print the top 3 passages and mark whether each is relevant. If most aren't, fix the chunking or the material first. Grading with bad passages only adds noise.
5. **Grader:** `build_prompt(item, labels, passages=None)`, and a `--rag` option (with `--k`) on the command line. Write the index path, the embedding model and k into the results JSON, like every other setting.
6. **Run on Kaggle:** E2B and E4B, all splits, the same way as the baseline.

## Starting settings

| Setting | Start with | Why |
|---|---|---|
| Chunk size | 250 words | Long enough to hold one idea, short enough to keep the prompt small |
| Overlap | 50 words | An idea split at a boundary still appears whole in one chunk |
| k | 3 | Test 1 and 5 later if there's time; each passage costs prefill time |
| Embedding model | `bge-small-en-v1.5` | Small, fast on a CPU, good on English retrieval benchmarks |
| Query | Question text only | The student answer varies; the question decides the topic |

Change one setting at a time, and only after the retrieval check (step 4).

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
