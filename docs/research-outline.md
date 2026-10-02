# Research outline

*Last updated 2026-09-29. The source of truth for what this project is trying to answer. Update it when the supervisor changes direction, and log the reason in the decisions table below.*

## One-line summary

Measure how well a **small LLM on an edge device** grades students' short answers, then test how much **subject knowledge** improves it, first through **RAG** (course material in the prompt) and then through **fine-tuning**. Report the accuracy gained against the time and energy each version costs on the device.

## Why

- **Teachers:** grading short answers takes teachers a lot of time.
- **Why local:** a model running locally avoids per-call API costs, keeps student data on school hardware, works offline, and doesn't change without notice.
- **The problem:** small models lack subject knowledge. The supervisor's hypothesis is that adding it (e.g. for physics or maths) helps the model break down formulas and concepts better.
- **The open question:** how much RAG and fine-tuning help, whether that help carries over to new topics, and what it costs on the device.

## Research questions

| # | Question | Stage |
|---|---|---|
| RQ1 | How well do **untouched** small models (Gemma 4 E2B/E4B) grade student answers? How does that compare with their subject knowledge on the lab's MMLU-Pro baseline? | 1 |
| RQ2 | Does **RAG over course material** improve grading, especially on **unseen questions and unseen domains**? | 2 |
| RQ3 | Does **fine-tuning** (QLoRA) improve grading? Does the gain survive on unseen domains, and how does it compare with RAG, or RAG plus fine-tuning? | 3 |
| RQ4 | What does each version cost on the **Pi 5 / Jetson** (latency, energy per answer), and which gives the best accuracy per unit of energy? | 4 |
| RQ5 | How close does the best local version get to a **cloud model with the same material**, which stands in for NotebookLM? The supervisor wants this tested, not assumed. | 2–4 |
| *opt.* | Does thinking mode help or hurt small graders? | any |

## Stages

| Stage | What | Output |
|---|---|---|
| **1. Baseline** | (a) The lab's MMLU-Pro 100-question subset, for subject knowledge. (b) SciEntsBank grading with the untouched model, served by `llama-server` with thinking off | Done: MMLU-Pro (E2B 51.7–52.7%, E4B 65.7–66.0%), grading accuracy per split, and board cost estimated from `agentic-edge`'s rates. See [`results/baselines.md`](results/baselines.md) |
| **2. RAG** | Chunk open school-science material matched to SciEntsBank's topics (Utah OER textbooks + Wikipedia; Indonesian Kurikulum Merdeka books as an optional cross-lingual test); embed with a small model; retrieve the top 3 chunks per question into the grading prompt. Retrieval is done once per question, ahead of time | Improvement over the baseline, per split. Pipeline and retrieval check done: [`results/rag.md`](results/rag.md) |
| **3. Fine-tuning** | QLoRA with Unsloth on Kaggle/Colab, trained on the SciEntsBank train split; convert to GGUF and serve like the others | Improvement over the baseline; RAG vs fine-tuning vs both |
| **4. Edge cost** | Run every version on the Pi 5 / Jetson with `scripts/board_grade.sh` (agentic-edge's `run_measured.sh` telemetry) | Accuracy vs latency and energy chart |

**Rule:** don't start a stage until the one before it gives a number you trust.

**Steps, checklist and week-by-week timeline:** [`plan.md`](plan.md).

## Datasets

- **SciEntsBank** (SemEval-2013 Task 7), HF mirror `nkazi/SciEntsBank`
  - train 4,969
  - test_ua 540 (unseen answers)
  - test_uq 733 (unseen questions)
  - test_ud 4,562 (unseen domains)
  - Labels: 5-way, collapsible to 3-way or 2-way
- **MMLU-Pro:** the lab's fixed 100-question subset (`agentic-edge/benchmark/mmlupro_subset.py`, seed 20260918). Its physics and maths subsets are an option if a subject-specific benchmark is needed.
- **Course material for RAG:** to be chosen in stage 2. It must match SciEntsBank's science topics.

## Metrics

- **Grading:** accuracy and macro/weighted F1, the SemEval-2013 metrics. SciEntsBank labels are categories, not an ordered scale, so QWK doesn't apply.
- **MMLU-Pro:** accuracy, from the lab harness.
- **Cost:** latency per answer, prompt and completion tokens, and energy per answer (Wh) from the lab telemetry.
- **Always report per split.** Unseen domains (test_ud) is the key one for "does it transfer".

## Fixed setup (keep comparable with `agentic-edge`)

- Models: Gemma 4 E2B / E4B, GGUF
- Serving:

  ```
  llama-server -m <model> -c 8192 --host 127.0.0.1 --port 8080 -rea off --reasoning-budget -1 --cache-ram 0
  ```

- Temperature 0; a fixed seed for sampling answers
- **QAT caveat:** the lab's models are QAT builds. A fine-tuned model is quantized afterwards the ordinary way, so compare base and fine-tuned models under the **same** quantization.

## Supervisor feedback so far

| Date | Feedback | Response |
|---|---|---|
| 2026-09-29 | (1) Compare a baseline with RAG / local embedding with subject knowledge. (2) Compare with NotebookLM, or argue why edge. | RAG stage; cloud model with the same material as the reference |
| 2026-09-29 | Fine-tuning is necessary, to compare a general model with a tweaked one. "Find the benchmark first, create the baseline, then we can see the improvement." MMLU-Pro is fine, or a special subject like maths/physics. | Staged plan (above); MMLU-Pro kept as the second baseline |
| 2026-09-29 | NotebookLM comparison: "that's why we need to check by experimenting on it." | RQ5 |
| 2026-09-29 | Asked for references on why robustness can be dropped and edge measurement made optional | Edge kept (stage 4). Robustness: references sent (arXiv 2606.03090, 2602.00979, 2601.08843, 2412.15275); waiting for reply. |
| 2026-09-29 | For benchmarking, use what the lab already has to cut time | `agentic-edge`'s device benchmarks copied, not re-run: [`results/edge-benchmarks.md`](results/edge-benchmarks.md) |
| 2026-09-30 | "Yeppp" to keeping edge measurement: latency and energy per answer on the Pi 5 / Jetson | Confirmed: stage 4 measures every version on the boards. Stage 1 board cost is estimated from `agentic-edge`, see [`results/baselines.md`](results/baselines.md) |

## Own decisions

Decisions made while doing the work, not from the supervisor. Each has to serve a research question above; anything that doesn't is dropped or marked optional.

| Date | Decision | Why | RQ |
|---|---|---|---|
| 2026-09-30 | Stage-1 board cost **estimated** from `agentic-edge`'s per-request rates, then checked with one measured run per board | Supervisor: reuse the lab's benchmarks. The Jetson check matched Kaggle accuracy, and the estimate was an upper bound (17–25% high) | RQ4 |
| 2026-10-02 | RAG chunks built on the Mac and uploaded as a **private** Kaggle dataset, checked by sha256 | A fresh Wikipedia download could change the chunks; the Indonesian books may not be redistributed | RQ2 |
| 2026-10-02 | Extraction drops front/back matter, references, worksheet prompts and student-directed text | Round-1 Indonesian retrieval returned bibliographies and activity steps | RQ2 |
| 2026-10-02 | Small CPU-only models (embeddings) may run on the Mac; Gemma still runs on Kaggle or the boards | Owner's instruction; it saves a Kaggle round trip per retrieval change | — |
| 2026-10-03 | **RAG-en is the main RAG condition**; RAG-id optional; RAG-both dropped | SciEntsBank is English, so RAG-en answers RQ2. The others add conditions without a research question; time goes to RQ3 and RQ4 | RQ2–4 |
| 2026-10-03 | Embedding models: `bge-small-en-v1.5` (English), `bge-m3` (Indonesian); settings frozen | Best retrieval of the models tested on the train-question check ([`results/rag.md`](results/rag.md)); `bge-small` also fits next to Gemma on a Pi | RQ2, RQ4 |
| 2026-10-03 | On the boards, retrieval counts as a one-off setup cost per question, reported separately from per-answer cost | A teacher's questions are known before grading, so passages can be prepared in advance | RQ4 |

## Open questions

- Robustness section: dropped or not? Waiting on the supervisor.
- Which course material matches SciEntsBank topics well enough for RAG?
- Is SciEntsBank in the models' training data? A newer dataset (e.g. SAS-Bench 2025) could serve as a check.
- How this relates to the lab's existing cascade analysis (`agentic-edge/benchmark/analyze_cascade.py`).

## Related notes

The longer history lives in the `research-ideation` repo:
- `docs/summary/2026-09-29-decisions.md`: the full decisions log
- `docs/literature/`: prior work
- `docs/reports/`: proposals v1 and v2
