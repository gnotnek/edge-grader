# Plan: steps and timeline

*Last updated 2026-10-05. Tick steps off as they're done, and move dates when reality changes, noting why at the bottom. The research questions and the reasoning for each stage are in [`research-outline.md`](research-outline.md).*

**Weeks are counted from the internship start**, taken as the week of 2026-09-14 (two weeks had passed by 2026-09-26). The internship ends around mid-December.

## Timeline

| Week | Dates | Stage | Goal | Checkpoint |
|---|---|---|---|---|
| 3 | Sep 28 – Oct 4 | 1. Baseline | Grading baseline on Kaggle ✅; board cost estimated ✅, then measured ✅ | Baseline table complete: accuracy + cost ✅ |
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
- [x] Board cost for E2B + E4B on the Pi 5 and the Jetson, **estimated** from `agentic-edge`'s measured per-request rates and the Kaggle token counts (`uv run edge-grader-cost`), with no new board runs: [`results/baselines.md`](results/baselines.md#cost-on-the-boards-estimated-from-agentic-edge-not-measured)
- [x] Measured board check of the estimate and of board accuracy ≈ Kaggle: E2B + E4B, test_ua, on both boards (Jetson 2026-09-30, Pi 5 2026-10-04 at `-t 3`). Board accuracy = Kaggle (McNemar p ≥ 0.25); cost 10–35% under the estimate; Jetson 7.7–8.8× faster, 7.0–7.8× less energy per answer: [`results/baselines.md`](results/baselines.md#cost-on-the-boards-measured-test_ua) (`uv run edge-grader-board-check`)
- [ ] Supervisor: confirm dropping the robustness section (references already sent)

### Stage 2: RAG (weeks 4–5)
*How RAG works and how to build it: [`guides/rag.md`](guides/rag.md).*

- [x] Choose course material that matches SciEntsBank's topics, English and Indonesian; note the licence: [`guides/rag-sources.md`](guides/rag-sources.md)
- [x] Material for MMLU-Pro too (OpenStax + SMA books), so RAG can be tested on both baselines
- [ ] OCR Informatika kelas XII (no text layer), or leave it out
- [x] Extract and chunk the text (PyMuPDF); keep the source page for every chunk; fix the broken ligatures in the Indonesian books. SciEntsBank: 755 English chunks, 1,141 Indonesian after the noise filters (`results/rag/chunks-scientsbank.json`)
- [x] Embed: `bge-small-en-v1.5` for English; for Indonesian, `multilingual-e5-small`, `-base` and `bge-m3` compared, `bge-m3` chosen. Separate indexes per language ([`results/rag.md`](results/rag.md))
- [x] Retrieval check, first pass (Claude): round 1 English ~23/30, Indonesian ~13/30 → noise filters → round 2 English ~22 clear, Indonesian (`bge-m3`) ~17 clear. Settings frozen ([`results/rag.md`](results/rag.md))
- [ ] Owner's own marking of the check sheets (`data/rag/check/`), to replace the first-pass counts
- [x] Add a `--rag` mode to the grader: the same prompt plus the top-k passages (`--rag <retrieval file> --k 3`)
- [ ] Re-upload the round-2 chunks to the private Kaggle dataset (New Version); the notebooks check them against the committed sha256
- [ ] Run on Kaggle (`kaggle_baseline.ipynb` with `RAG` set): **RAG-en**, E2B + E4B, all splits. RAG-id (`bge-m3`) only if time allows. RAG-both dropped
- [ ] Analysis: did recall on *correct* answers improve (the baseline's main weakness)? And E2B's "contradictory" overuse?

**MMLU-Pro with RAG** (same retrieval code, the OpenStax + SMA material):
- [ ] A RAG runner for the 300-question subset: a custom lm-eval task, or this project's own runner. Query with the question only, not the options.
- [ ] Re-run the no-RAG baseline through the same runner, so RAG vs no-RAG compares like with like. The lab's lm-eval numbers stay as the reference.
- [ ] Run E2B + E4B: RAG-en and RAG-id
- [ ] Analysis per subject; expect little help on engineering, law and "other" (thin material)

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
| 2026-09-30 | RAG material gathered in English and Indonesian, for SciEntsBank and MMLU-Pro; MMLU-Pro RAG added to stage 2 | Test RAG on both baselines; IPAS books kept whole for now |
| 2026-09-30 | Stage-1 board runs replaced by a cost estimate from `agentic-edge`'s export; a measured run is now optional | Supervisor: reuse existing benchmarks. The export has per-request prefill and decode rates for both boards at grading-sized prompts |
| 2026-10-02 | Stage 2 started early, while the Pi 5 board check waits in the queue; RAG pipeline built (chunking, index, retrieval, `--rag`) | The Jetson run confirmed the stage-1 numbers, so stage 1 has a number we trust |
| 2026-10-03 | RAG scope: **RAG-en is the main condition**; RAG-id optional, with `bge-m3`; **RAG-both dropped**; retrieval settings frozen | Only RAG-en is needed for RQ2. Time goes to fine-tuning (RQ3) and the boards (RQ4). `bge-m3` was the only model that retrieved Indonesian passages well enough for a fair test |
| 2026-10-05 | Stage 1 done: board cost **measured** on both boards (test_ua), replacing the estimate there; the Pi re-run at `-t 3` | The Pi's first pass used llama.cpp's default 4 threads, unlike every agentic-edge Pi run; grades were identical, speed 5–14% different. The estimate was 10–35% high, so measured numbers are quoted from now on |
