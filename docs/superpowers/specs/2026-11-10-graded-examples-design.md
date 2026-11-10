# Design: graded examples in the prompt (E4B, SciEntsBank)

*2026-11-10. The method was approved by the supervisor on 2026-11-10; this design was **approved by the owner on 2026-11-10, with the random-examples control added** (owner). Calls marked **(Claude's call)** were not discussed and can be changed. Once approved, the decision goes into [`research-outline.md`](../../research-outline.md), which stays the source of truth.*

## 1. Why

**In plain words:** before asking E4B to grade an answer, show it a few answers from other students that experts already graded, picked because they look most like the one being graded. The model sees *how the experts draw the line* between correct, contradictory and incorrect, on answers close to this one. Nothing about the model changes; only the prompt gets longer.

This is not the earlier RAG ([results](../../results/rag.md)): that added textbook passages (knowledge) and didn't help grading. Here the added text is graded answers (examples of the judgement itself).

Evidence:
- **Wang & Ormerod 2024** (arXiv 2408.03811): retrieve similar graded train answers with a fine-tuned all-MiniLM-L6-v2, put them in Claude 3.5 Sonnet's prompt. Best published SciEntsBank 3-way: accuracy 78.71 / 76.94 / 73.35 on UA / UQ / UD. They note it helps less on UQ and UD, where train holds no answers to the same question.
- **Liu et al. 2022** (DeeLIO, "What Makes Good In-Context Examples for GPT-3?"): examples picked by similarity (kNN) beat random ones.

What we have to beat on E4B (3-way accuracy, UA / UQ / UD): untouched 66.7 / 67.5 / 70.4% (Kaggle), the label-only adapter 79.1 / 80.2 / 76.5% (Spark, 2026-11-08).

## 2. Research question

Part of **RQ3** (and RQ4 for its cost): *Does putting similar graded train answers in the prompt raise untouched E4B's SciEntsBank grading, per split, without any training? Does it add anything on top of the label-only adapter?*

**What "it worked" means:** the project's 5-point bar (owner, 2026-11-05): at least **+5 accuracy points** over the same model without examples, on a split, and **Holm p < 0.05** (paired McNemar, 3-way), with the Holm family in §5.4.

Expected from Wang & Ormerod: the largest gain on test_ua (same-question examples exist), less on test_uq and test_ud.

## 3. Goals, non-goals, anti-goals

**Goals:**
- **Main condition:** untouched E4B + examples, all three test splits. An inference-only method.
- **Secondary:** the E4B SciEntsBank label-only adapter (`e4b-ft`) + examples, all three splits.
- **A random-examples control** (owner, 2026-11-10): untouched E4B + 4 train answers drawn at random, all three splits. It shows whether *similar* examples matter or any graded examples do (Liu et al.'s comparison).
- **Pairs on the same hardware:** untouched E4B graded on the Spark (it hasn't been yet); the adapter's Spark runs from 2026-11-08 are reused.
- Each prompt's token length recorded, since a longer prompt costs energy on the device.

**Non-goals:**
- **Fine-tuning the retriever** (Wang & Ormerod did). The project's frozen embedder is reused: one change at a time.
- **Tuning k or the prompt on any data.** k is fixed below, before any run. **(Claude's call)**
- **E2B, thinking on, Beetle/SAF, other adapters.** Not asked for.
- **Jetson cost.** Measured later, not in this task; this task records prompt tokens so the cost can be estimated.

**Anti-goals (a failure even if it runs):**
- **Any test answer in the index.** Only SciEntsBank **train** answers are retrievable (Wang & Ormerod's Table 9 ablation indexed test answers; not allowed here). Checked in code, not by eye.
- Any setting chosen by looking at a test split.
- A comparison that differs in more than the examples: same GGUF, adapter, llama.cpp build (`a3a1c47`), server flags, temperature 0, thinking off, same machine.

## 4. Constraints

- Gemma 4 E4B, QAT `UD-Q4_K_XL` GGUF, `llama-server` at `a3a1c47`, `-c 8192 -rea off --reasoning-budget -1 --cache-ram 0 -ngl 99`, temperature 0, thinking off, label-only reply (16 tokens).
- The DGX Spark, through `scripts/spark_queue.sh`, chained with `AFTER=` behind anything else of ours. Shared with the `e4b-soup` session and other people.
- The Mac runs no Gemma. The embedder (bge-small, CPU) may run on the Mac (owner, 2026-10-02).
- Every run writes a summary JSON with its settings to `results/`.
- Metrics: 3-way accuracy and macro-F1 headline; 2-way beside it.

## 5. Design

### 5.1 The index and retrieval (done once, on the Mac)

- **Pool:** the 4,969 SciEntsBank train answers (135 questions, 31–107 answers each), with their gold 3-way labels. Nothing else.
- **Embedder:** `bge-small-en-v1.5`, the project's frozen English embedder (2026-10-03).
- **What is embedded:** `"<question>\n<student answer>"` for every train and test answer, the same way on both sides (a symmetric similarity, so no bge query instruction). Within a question the question text is the same, so the ranking comes from the answer. **(Claude's call)**
- **k = 4**, the middle of the 3–5 range, fixed now. A prompt grows from ~160 to ~600 tokens. **(Claude's call)**
- **Same question first:** if train has answers to the test answer's question (524 of 540 test_ua answers; none in test_uq or test_ud), the k most similar of those; if fewer than k, the rest from the whole pool. Otherwise the k most similar from the whole pool. **(Claude's call: the rule as suggested)**
- **Random control:** 4 answers drawn from the whole train pool (not the same question first), seeded per test answer id (seed 0), so a re-run draws the same ones. **(Claude's call: whole pool, as in Liu et al.)** It therefore differs from the kNN run in both similarity and same-question; on test_uq and test_ud only similarity differs.
- **No label balancing:** the examples are whatever is nearest, even if all share one label. Balancing would be a second change. **(Claude's call)**
- **Output:** one retrieval file in `results/examples/`, test answer id → k train answer ids with scores. Ids only, so it's committed; the grader looks the text up in the train split and stops if an id isn't a train answer or is a test answer.
- Retrieval counts as a one-off setup cost, not per-answer cost, as for RAG (2026-10-03): a teacher's past graded answers exist before grading.

### 5.2 The prompt

The stage-1 prompt with a block before the question; the instructions and labels are unchanged:

```
You are grading a student's answer to a science question.

Graded examples (other students' answers, graded by experts):
[1] Question: ...
Reference answer: ...
Student answer: ...
Label: incorrect
...
[4] ...

Question: ...
Reference answer: ...
Student answer: ...

Choose exactly one label:
...
Reply with the label only.
```

- **Order:** least similar first, most similar last (next to the answer being graded), as in Liu et al.'s best order. **(Claude's call)**
- Each example carries its own question and reference answer, even when it repeats the graded one's: simpler, and one format for all splits. The extra tokens are recorded. **(Claude's call)**
- One user turn, like the RAG prompt; not fake earlier chat turns. **(Claude's call)**

### 5.3 Runs (DGX Spark)

| Run | Label | Splits | Status |
|---|---|---|---|
| E4B untouched | `e4b-spark-gb10` | UA, UQ, UD | to run |
| E4B + examples | `e4b-spark-gb10-ex4` | UA, UQ, UD | to run |
| adapter (`e4b-ft`) | `e4b-spark-gb10-ft` | UA, UQ, UD | done 2026-11-08 |
| adapter + examples | `e4b-spark-gb10-ft-ex4` | UA, UQ, UD | to run |
| E4B + random examples | `e4b-spark-gb10-rand4` | UA, UQ, UD | to run |

Order: a smoke first (`N=5 OUT=data/smoke`, all three runs), then the full runs in one queued container. Rough time: ~5–6 GPU-hours (the adapter's test_ud took 43 min at ~160 tokens a prompt). The untouched Spark run is also checked against Kaggle's (66.7 / 67.5 / 70.4%), described, not tested.

### 5.4 Comparisons (fixed before grading)

Paired McNemar on 3-way labels, **Holm over these 12 tests at once** (`compare_examples`, built on `compare.paired` / `holm`):

| Tests | What it answers | Bar applies |
|---|---|---|
| E4B + examples vs E4B, × 3 splits | **main:** does the method help the untouched model? | yes |
| adapter + examples vs adapter, × 3 splits | does it add to fine-tuning? | yes |
| E4B + examples vs adapter, × 3 splits | can a prompt replace training? | no (different models) |
| E4B + examples vs E4B + random examples, × 3 splits | does similarity matter, or any graded examples? | no |

- 2-way (correct vs not) beside each test, with its own Holm over the same 12, as in `compare_bar`. Secondary; the bar stays 3-way.
- Described, not tested: random examples vs untouched E4B, macro-F1, per-class recall, unparsed, median and max prompt tokens per answer, and the share of examples that came from the same question.
- On test_ua the adapter has trained on the same questions the examples come from; it's reported, not excluded, since both sides of that comparison have seen train. **(Claude's call)**

## 6. Where the code changes

- `edge_grader/examples.py` (new): build the retrieval files (kNN and random, Mac), load one for grading, with the train-only check.
- `grader.build_prompt` / `grade`: an `examples` argument, placed as in §5.2.
- `baseline.py`: `--examples <file> --k 4`; stage `examples` in the file name; the retrieval file and k in the summary; median and max prompt tokens in every summary.
- `scripts/spark_bench_scientsbank.sh` + a `bench-scientsbank-examples` compose service: the four E4B runs in §5.3 (`EXAMPLES=1`), untouched E4B only, so E2B isn't graded for nothing.
- `edge_grader/compare_examples.py`: §5.4.
- Tests: the train-only check, the same-question rule and fill, the order, the prompt text, and that the plain prompt is unchanged byte for byte.

## 7. Deploying to the Spark

The Spark checkout (`~/research/edge-grader`) is on `main`. Code reaches it through `git pull`, so this branch has to be merged into `main` (or checked out there) first. The `e4b-soup` session uses the same checkout, so the owner decides the order before anything is deployed.
