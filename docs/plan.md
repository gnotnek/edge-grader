# Plan: steps and timeline

*Last updated 2026-09-29. Tick steps off as they're done, and move dates when reality changes, noting why at the bottom. The research questions and the reasoning for each stage are in [`research-outline.md`](research-outline.md).*

**Weeks are counted from the internship start**, taken as the week of 2026-09-14 (two weeks had passed by 2026-09-26). The internship ends around mid-December.

## Timeline

| Week | Dates | Stage | Goal | Checkpoint |
|---|---|---|---|---|
| 3 | Sep 28 – Oct 4 | 1. Baseline | Grading baseline on Kaggle ✅ and on the boards | Baseline table complete: accuracy + cost |
| 4 | Oct 5 – 11 | 2. RAG | Course material chosen, retrieval pipeline built | Retrieval returns the right passage for most questions |
| 5 | Oct 12 – 18 | 2. RAG | RAG grading runs, all splits, both models | RAG vs baseline table |
| 6 | Oct 19 – 25 | 3. Fine-tuning | QLoRA pipeline on Kaggle; E2B trained and converted to GGUF | Fine-tuned E2B graded on test_ua |
| 7 | Oct 26 – Nov 1 | 3. Fine-tuning | E4B fine-tuned; RAG + fine-tuning combined | Baseline vs RAG vs fine-tuning vs both, per split |
| 8 | Nov 2 – 8 | 4. Edge cost + cloud | All versions on the Pi 5 / Jetson; cloud reference (RQ5) | Accuracy vs energy chart |
| 9 | Nov 9 – 15 | Buffer | Catch up; optional thinking-mode run; analysis and charts | All numbers final |
| 10–11 | Nov 16 – 29 | Write-up | Report draft with figures; supervisor review | Draft sent to supervisor |
| 12 | Nov 30 – Dec 6 | Wrap-up | Revisions, final report, repo cleanup | Final report |

**Rule:** a stage starts only when the one before it has a number you trust. If a stage runs late, cut from the optional list below, not from the core stages.

## Steps

### Stage 1: Baseline (week 3)
- [x] Copy the `agentic-edge` MMLU-Pro baseline, as the subject-knowledge reference: [`results/baselines.md`](results/baselines.md)
- [x] Grading harness: loader, prompt, parser, metrics, CLI, tests
- [x] Kaggle pilot, 100 answers: 0 unparsed, 0 errors
- [x] Kaggle full splits, E2B + E4B: [`results/kaggle/README.md`](../results/kaggle/README.md)
- [ ] Board run on the Pi 5: `scripts/board_grade.sh e2b` and `e4b` (test_ua)
- [ ] Board run on the Jetson: same
- [ ] Fill in the cost table in `results/baselines.md`; check that board accuracy ≈ Kaggle accuracy
- [ ] Supervisor: confirm dropping the robustness section (references already sent)

### Stage 2: RAG (weeks 4–5)
*How RAG works and how to build it: [`guides/rag.md`](guides/rag.md).*

- [ ] Choose course material that matches SciEntsBank's topics (OpenStax / CK-12 open textbooks); note the licence
- [ ] Extract and chunk the text (PyMuPDF); keep the source page for every chunk
- [ ] Embed with a small model (e.g. bge-small or e5-small via sentence-transformers, or `llama-server --embedding` for the edge)
- [ ] Retrieval check: for about 30 questions, is the right passage in the top-k? Fix chunking before grading.
- [ ] Add a `--rag` mode to the grader: the same prompt plus the top-k passages
- [ ] Run on Kaggle: E2B + E4B, all splits
- [ ] Analysis: did recall on *correct* answers improve (the baseline's main weakness)? And E2B's "contradictory" overuse?

### Stage 3: Fine-tuning (weeks 6–7)
- [ ] Unsloth QLoRA notebook on Kaggle; train on the SciEntsBank **train** split only
- [ ] Hold out part of train for validation; never touch the test splits while tuning
- [ ] Convert to GGUF and quantize like the baseline (see the QAT caveat in `research-outline.md`)
- [ ] Grade with the same CLI: E2B, then E4B
- [ ] RAG + fine-tuned combined
- [ ] Analysis: the gain per split, especially **unseen domains** (does fine-tuning transfer?)

### Stage 4: Edge cost + cloud reference (week 8)
*Device rates (tok/s, power, J/token) are reused from `agentic-edge`, not re-measured: [`results/edge-benchmarks.md`](results/edge-benchmarks.md).*

- [ ] Every version (baseline, RAG, fine-tuned, both) × E2B/E4B on the Pi 5 and the Jetson with `board_grade.sh`
- [ ] RAG on the board: include embedding and retrieval time in the latency
- [ ] Cloud reference: a cloud model with the same material, on the same splits (RQ5, the NotebookLM question)
- [ ] Main chart: accuracy vs energy per answer, for every version and board

### Write-up (weeks 10–12)
- [ ] Figures: the main chart, per-split tables, confusion matrices
- [ ] Report: motivation, setup, results per RQ, limitations (dataset age and possible memorisation, one dataset, quantization caveat)
- [ ] Supervisor review, then revise
- [ ] Repo cleanup: README results summary, reproducible commands for every table

## Optional, if time allows
- Thinking on vs off for the best version
- A second dataset (e.g. SAS-Bench 2025, as a check against memorisation)
- The cascade idea: small model first, escalate uncertain answers (check overlap with `agentic-edge`'s `analyze_cascade.py` first)
- Fairness: are non-native English answers scored lower?

## Change log
| Date | Change | Why |
|---|---|---|
| 2026-09-29 | Plan created; stage 1 Kaggle runs done | — |
| 2026-09-29 | `agentic-edge` device benchmarks copied for stage 4 | Supervisor: reuse existing benchmarks to cut time |
