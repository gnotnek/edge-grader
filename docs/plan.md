# Plan: steps and timeline

*Last updated 2026-10-05. Tick steps off as they're done, and move dates when reality changes, noting why at the bottom. The research questions and the reasoning for each stage are in [`research-outline.md`](research-outline.md).*

**Weeks are counted from the internship start**, taken as the week of 2026-09-14 (two weeks had passed by 2026-09-26). The internship ends around mid-December.

## Timeline

| Week | Dates | Stage | Goal | Checkpoint |
|---|---|---|---|---|
| 3 | Sep 28 – Oct 4 | 1. Baseline | Grading baseline on Kaggle ✅; board cost estimated ✅, then measured ✅ | Baseline table complete: accuracy + cost ✅ |
| 4 | Oct 5 – 11 | 2. RAG | SciEntsBank RAG-en graded, all splits, both models ✅; baseline re-run on the pinned build ✅; MMLU-Pro runner and retrieval | RAG vs baseline table (SciEntsBank) ✅ |
| 5 | Oct 12 – 18 | 2. RAG | MMLU-Pro: no-RAG and RAG-en on Kaggle, per-subject analysis; RAG-id if time allows. Start the stage-3 notebook alongside | RAG vs baseline table (MMLU-Pro) |
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

**SciEntsBank** (the main condition for RQ2: done)
- [x] Choose course material that matches SciEntsBank's topics, English and Indonesian; note the licence: [`guides/rag-sources.md`](guides/rag-sources.md)
- [x] Material for MMLU-Pro too (OpenStax + SMA books), so RAG can be tested on both baselines
- [x] Informatika kelas XII: **dropped** (2026-10-05). No text layer, and OCR wasn't feasible
- [x] Extract and chunk the text (PyMuPDF); keep the source page for every chunk; fix the broken ligatures in the Indonesian books. SciEntsBank: 755 English chunks, 1,141 Indonesian after the noise filters (`results/rag/chunks-scientsbank.json`)
- [x] Embed: `bge-small-en-v1.5` for English; for Indonesian, `multilingual-e5-small`, `-base` and `bge-m3` compared, `bge-m3` chosen. Separate indexes per language ([`results/rag.md`](results/rag.md))
- [x] Retrieval check, first pass (Claude): round 1 English ~23/30, Indonesian ~13/30 → noise filters → round 2 English ~22 clear, Indonesian (`bge-m3`) ~17 clear. Settings frozen ([`results/rag.md`](results/rag.md))
- [x] Owner's marking, re-marked 2026-10-06 (the first pass ticked every question): a useful passage in the top 3 for **16/30** (English) and **18/30** (Indonesian) train questions. The gate wasn't really met; grading had already run on the frozen settings
- [x] Add a `--rag` mode to the grader: the same prompt plus the top-k passages (`--rag <retrieval file> --k 3`)
- [x] Re-upload the round-2 chunks to the private Kaggle dataset (New Version); the notebooks check them against the committed sha256
- [x] Run on Kaggle: **RAG-en**, E2B + E4B, all splits (2026-10-05). No overall gain: E4B flat, E2B −2 to −5 points; recall on correct up on every split, recall on incorrect down ([`results/rag.md`](results/rag.md#grading-results-rag-en-kaggle-t4-2026-10-05))
- [x] Pin llama.cpp to `a3a1c47` in `kaggle_baseline.ipynb` (2026-10-05)
- [x] Re-run the no-RAG baseline on the pinned build (E2B + E4B, all splits; Kaggle, 2026-10-05). Same accuracy: E2B identical labels, E4B ≤1.3% of labels changed, no split significant. The RAG table now uses it as the baseline ([build check](results/rag.md#build-check-baseline-re-run-on-a3a1c47)). The boards weren't re-run: each uses one fixed build (Pi `661643e`, Jetson `a894dae`), and stage 4 runs the baseline again next to RAG
- [x] Analysis: recall on *correct* rose on every split (up to +14 points), but recall on *incorrect* fell (up to −20) and E2B's "contradictory" overuse grew (1,105 → 1,522 on test_ud). RAG makes grading more lenient, not better ([`results/rag.md`](results/rag.md#grading-results-rag-en-kaggle-t4-2026-10-05))
- [x] RAG-id (`bge-m3`), E2B + E4B, all splits, pinned build (Kaggle, 2026-10-05). Worse than RAG-en on 5 of 6 rows; E2B −4.6 / −7.4 (test_uq / test_ud) and E4B −4.2 (test_ud), all significant ([`results/rag.md`](results/rag.md#grading-results-rag-id-kaggle-t4-2026-10-05)). RAG-both dropped

**MMLU-Pro** (the subject-knowledge baseline; same retrieval code, OpenStax + SMA material)
- [x] Runner, this project's own (`edge-grader-mmlupro`): without RAG it sends lm-eval's `mmlu_pro` chat messages byte for byte (checked against all 200 prompts recorded from the lab's s1 runs; a test pins one per subject), the same 2,048-token budget and stop string, and reads the answer with lm-eval's own pattern (agrees on all 200 recorded replies). With `--rag`, passages go in front of the question; the five worked examples are unchanged. Query: the question only, not the options
- [x] Chunks: OpenStax cut to its chapters by the PDFs' bookmarks (front matter, answer keys, appendices and indexes out); 32,016 English chunks, 1,670 Indonesian (`results/rag/chunks-mmlupro.json`). SciEntsBank chunks unchanged (same sha256)
- [x] Upload `mmlupro-en.jsonl` and `mmlupro-id.jsonl` to the private Kaggle chunks dataset (New Version)
- [x] English index (`bge-small-en`) and retrieval for the 300 test + 70 validation questions, on Kaggle (2026-10-06)
- [x] Retrieval check on 30 **validation** questions, owner's marking (2026-10-06): a useful passage for **15/30**. Biology, chemistry, history all; law, "other", computer science none ([`results/rag.md`](results/rag.md#mmlu-pro-retrieval-check-owners-marking-2026-10-06))
- [ ] Kaggle: no-RAG through the runner, E2B + E4B, s1–s3 (`TASK = "mmlupro"`). Should land within noise of the lab's lm-eval numbers (E2B 51.7–52.7%, E4B 65.7–66.0%); paired against the lab where its per-question answers exist
- [ ] Kaggle: RAG-en, E2B + E4B, s1–s3
- [ ] RAG-id: `mmlupro-id` index with `bge-m3` on Kaggle (`kaggle_rag_index.ipynb`), then grade. Optional, like SciEntsBank's
- [ ] Analysis per subject, paired McNemar; expect little help on engineering, law and "other" (thin material)

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
| 2026-10-05 | Informatika kelas XII dropped; retrieval check passed on the owner's marking | No text layer and OCR not feasible; 29/30 and 30/30 questions get a useful passage |
| 2026-10-05 | RAG-en run: no overall gain; llama.cpp pinned to `a3a1c47`; baseline re-run on that build added | The RAG and baseline runs used different llama.cpp builds, so the comparison wasn't clean |
| 2026-10-05 | Baseline re-run done (same accuracy, RAG results stand); stage 2 rearranged: SciEntsBank done in week 4, MMLU-Pro RAG moves to week 5, stage-3 notebook can start alongside | SciEntsBank's RAG-en, the condition RQ2 needs, finished a week early; MMLU-Pro only needs Kaggle time once its retrieval passes the check |
| 2026-10-05 | RAG-id run (optional condition) done: worse than RAG-en, significant drops on unseen domains for both models | The run was cheap once RAG-en was set up, and it answers the cross-lingual question |
| 2026-10-06 | MMLU-Pro retrieval check 15/30; go ahead with no-RAG and RAG-en anyway, analysed by subject coverage | Supervisor-requested comparison; the per-subject split turns weak retrieval into a testable contrast |
| 2026-10-06 | SciEntsBank retrieval check re-marked: 29/30 → 16/30 (English), 30/30 → 18/30 (Indonesian) | The first marking ticked something for every question by mistake. RAG results stand; their interpretation now notes that only about half the questions got useful material |
