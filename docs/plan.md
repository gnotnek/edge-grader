# Plan: steps and timeline

*Last updated 2026-11-04. Tick steps off as they're done, and move dates when reality changes, noting why at the bottom. The research questions and the reasoning for each stage are in [`research-outline.md`](research-outline.md).*

**Weeks are counted from the internship start**, taken as the week of 2026-09-14 (two weeks had passed by 2026-09-26). The internship ends around mid-December.

## Timeline

| Week | Dates | Stage | Goal | Checkpoint |
|---|---|---|---|---|
| 3 | Sep 28 – Oct 4 | 1. Baseline | Grading baseline on Kaggle ✅; board cost estimated ✅, then measured ✅ | Baseline table complete: accuracy + cost ✅ |
| 4 | Oct 5 – 11 | 2. RAG | SciEntsBank RAG-en graded, all splits, both models ✅; baseline re-run on the pinned build ✅; MMLU-Pro runner and retrieval | RAG vs baseline table (SciEntsBank) ✅ |
| 5 | Oct 12 – 18 | 0. Edge baseline + A | **Phase 0 on the boards** (Jetson ~15 h, Pi ~30 h). On Kaggle meanwhile: math RAG-en (phase A), thinking-on grading (B1); EmbeddingGemma 2 tooling + check on the Mac (B3). Topic benchmarks done in week 4 ✅ | Baseline on both boards for every benchmark; baseline vs RAG for all four |
| 6 | Oct 19 – 25 | B. Variations + C + 4 | little-gemma + prefix reuse on the Jetson (B2) ✅; thinking-on cost on the boards ✅; **RAG-en for all four benchmarks on the Jetson** (stage 4 brought forward, supervisor 2026-10-19): SciEntsBank, news, GSM ✅, MMLU-Pro to run; answer-as-image rendering, reading check, Kaggle runs (C1); figure-question data search (C2) | Variations vs baseline table ✅; baseline vs RAG on the Jetson |
| 7 | Oct 26 – Nov 1 | 3. Fine-tuning | Week 1 (from Oct 21): adapters for E2B and E4B on Kaggle, sanity check on test_ua; week 2 (Oct 28 – Nov 3): tuned runs on Kaggle ✅ (accuracy), on the Jetson (cost), analysis. Phase C (images) waits until this is done | Tuned vs baseline table ✅, no-harm check ✅; cost on the Jetson |
| 8 | Nov 2 – 8 | 4. Edge cost (main, supervisor 2026-11-04) + 3b | The accuracy–cost trade-off on the Jetson: tuned adapters first, then stage 3b label-first vs reason-first, MMLU-Pro RAG-en; every remaining version on the Jetson (the Pi 5 is dropped, 2026-10-19); cloud reference (RQ5) only if time allows (optional since 2026-10-19). **On Kaggle alongside** (owner, 2026-11-01): reasoning-guided fine-tuning of E2B (stage 3b) | Trade-off table; accuracy vs energy and vs latency charts |
| 9 | Nov 9 – 15 | Buffer | Catch up; analysis and charts; stage 3b done by Nov 15 (E4B / Jetson / no-harm only if its gate passes and time is left) | All numbers final |
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
- [x] Kaggle: no-RAG through the runner, E2B + E4B, s1–s3 (2026-10-06): E2B 51.3%, E4B 65.3%, no significant difference from the lab on either board (87–92% same right/wrong per question)
- [x] Kaggle: RAG-en, E2B + E4B, s1–s3 (2026-10-06): E2B 52.0% (+0.7), E4B 63.3% (−2.0), neither significant
- [x] RAG-id: `mmlupro-id` index with `bge-m3` built on Kaggle (2026-10-07); retrieval check **0/30**, so **not graded** ([`results/rag.md`](results/rag.md#mmlu-pro-rag-id-retrieval-check-failed-not-graded-2026-10-07))
- [x] Analysis per subject, paired McNemar: well-covered subjects trend up (+3.7 / +5.6), uncovered ones trend down for E4B (−5.0); nothing significant. More answers hit the token cap with RAG ([`results/rag.md`](results/rag.md#mmlu-pro-grading-results-kaggle-t4-2026-10-06))

### Topic narrowing: math against news (week 5, pending the supervisor)
*Supervisor, 2026-10-07: which topic suits these models (logic, math, news, general knowledge)? Reasoning in [`research-outline.md`](research-outline.md) (own decisions, 2026-10-07).*

- [x] MMLU-Pro per subject, no RAG: math is the strongest subject for both models, significant; no logic or news category ([`results/rag.md`](results/rag.md#per-subject-no-rag-math-is-the-strongest-subject-analysis-2026-10-07))
- [x] Scope (supervisor, 2026-10-07): beside SciEntsBank grading, not replacing it; stage 3 still fine-tunes on SciEntsBank
- [x] Send the supervisor the proposal
- [x] Gemma 4's training cutoff: **January 2025**, one date for the whole family (model card) ([`results/topics.md`](results/topics.md#gemma-4s-training-cutoff-january-2025))
- [x] Math runner `edge-grader-gsm`: the 100 GSM8K test questions GSM-Symbolic templated, and 10 variants of each (1,000); the lab's lm-eval format with fixed 5 shots, 1,024 tokens; lm-eval's filters reimplemented, agreeing on all 300 recorded lab replies. The lab's tinyGSM8k prompts copied for a runner check ([`results/topics.md`](results/topics.md#math-gsm8k-and-gsm-symbolic))
- [x] News set: **LiveNewsBench** January 2026 release, 200 human-verified questions on Nov–Dec 2025 events, with their source articles, instead of a self-built set. Runner `edge-grader-news` with its own no-search prompt; judge `edge-grader-news-judge` with its SimpleQA prompt ([`results/topics.md`](results/topics.md#news-livenewsbench-january-2026-release))
- [x] News corpus: 1,176 articles → 4,861 chunks, `bge-small-en` index, retrieval (2026-10-07, on the Mac). Automatic check: a passage from the question's own article in the top 3 for **194/200**
- [x] News retrieval check by hand, owner's marking (2026-10-07): a useful passage in the top 3 for **21/30** questions, 16/30 at rank 1 ([`results/topics.md`](results/topics.md#rag-corpus-and-retrieval))
- [x] News judge: **Qwen3-14B on Kaggle**, free, in place of the paper's GPT-4.1 (free tier: 50 requests a day, 800 needed) ([`results/topics.md`](results/topics.md#news-livenewsbench-january-2026-release))
- [x] Kaggle, math (2026-10-08): runner check passes (E2B 77 vs lab 81, p = 0.29; E4B 74 vs 75, p = 1.0). GSM8K vs GSM-Symbolic: E2B 83.0 → 80.0%, E4B 92.0 → 90.5%, no significant gap (permutation p = 0.20, 0.32) ([`results/topics.md`](results/topics.md#math-results-kaggle-t4-2026-10-08))
- [x] Kaggle, news: no RAG and RAG-en, E2B + E4B (2026-10-07/08): 200 answers each, 0 errors, build `a3a1c47`
- [x] Kaggle, judge (2026-10-08): Qwen3-14B, 0 unparsed of 800. No RAG 0.5% (both), RAG-en E2B 39.0%, E4B 45.0% (p < 10⁻²¹ each) ([`results/topics.md`](results/topics.md#news-results-kaggle-t4-judged-by-qwen3-14b-2026-10-08))
- [x] Judge check (GPT-4.1 and the owner's hand grades on 50 answers): **dropped**, owner's choice (2026-10-08). The news numbers are reported as judged by Qwen3-14B, unchecked apart from Claude's 12-grade spot-check
- [x] Analysis: math vs news, no RAG vs RAG; the GSM8K vs GSM-Symbolic gap as the memorisation check ([`results/topics.md`](results/topics.md#math-against-news-what-suits-these-models))

### Stage 3: Fine-tuning (weeks 6–7, revived 2026-10-21)
*Dropped 2026-10-08 so the baseline came first; revived by the owner because RAG didn't improve grading. Design: [`superpowers/specs/2026-10-21-finetuning-design.md`](superpowers/specs/2026-10-21-finetuning-design.md).*

- [x] Re-run every Jetson baseline at the new clocks (14 jobs, queued 2026-10-21; owner): the cost pairs for stage 4 RAG and the tuned runs. All 14 done 2026-10-21: 12 with byte-identical replies to phase 0 (`board-check-20261021-065457.json`, RAG cost table in [`results/rag.md`](results/rag.md#cost-rag-en-against-the-re-run-baseline)); MMLU-Pro (jobs 48, 49, re-queued after `a6c5066`) agrees with Kaggle (`board-check-20261021-141904.json`, [`results/rag.md`](results/rag.md#mmlu-pro-baseline-on-the-jetson-2026-10-21))
- [ ] Day 1 risk check: a tiny E2B QLoRA run on a Kaggle T4, adapter converted to GGUF and loaded with `--lora` on Kaggle's and the Jetson's llama.cpp builds. **Kaggle half done 2026-10-22: main path.** The chat template matches `llama-server`'s; 20 steps at batch 1 × 16 (batch 4 ran out of memory at the loss, `cf54e00`), loss 0.21; `convert_lora_to_gguf.py` worked and `a3a1c47` served the adapter with `--lora`: 10 test_ua answers, 0 unparsed, 2 labels changed from the baseline's (`results/finetune/finetune-e2b-20261022-141705.json`, a smoke run, not a result). Jetson build (`a894dae`) load check still to do, with the real adapter
- [x] Training notebook (`notebooks/kaggle_finetune.ipynb`); E2B and E4B adapters, epochs picked on a held-out 10% of train questions; summary JSON in `results/`. Done 2026-10-23 on a Kaggle T4, 3 epochs at batch 1 × 16. Validation (14 held-out questions, 499 answers) by epoch: E2B 77.2 / 78.8 / **79.2%** (epoch 3 chosen), E4B 76.6 / **77.4** / 76.6% (epoch 2 chosen), 0 unparsed (`results/finetune/finetune-e2b-20261023-043321.json`, `finetune-e4b-20261023-071153.json`). Adapter GGUFs still to bring over from Kaggle, sha256 into `results/finetune/adapters.json`
- [x] Adapter option in `board_grade.sh` and the Kaggle notebook (`80d2ba4`); Kaggle sanity check on test_ua (100), paired with the untuned baseline on the same ids and build (`a3a1c47`): E2B 50 → **75%** (34 fixed, 9 broken, McNemar p < 0.001; `contradictory` predicted 11 times against 30, gold 11), E4B 68 → 72% (15 / 11, p 0.56). test_ua shares its questions with train, so this is a sanity check; the result is test_uq/test_ud on the Jetson
- [x] Re-trained 2026-10-27 as committed Kaggle versions, same settings, so the adapter files are kept (private Kaggle dataset `edge-grader-adapters`): E2B epoch 2 (val 78.8%), E4B epoch 1 (77.6%); test_ua (100) 76% and 73%, within a point of 2026-10-23. These are the adapters every result below uses
- [x] Kaggle (2026-10-28, owner: the Jetson was off the network; Kaggle gives the boards' accuracy): smoke test first, the first 100 of test_uq/test_ud (supervisor: check it rises before the full runs), E2B +19 / +13, E4B +10 / +3 points; then the full splits, MMLU-Pro (300), GSM8K + first GSM-Symbolic set, both models. **SciEntsBank test_uq / test_ud: E2B 76.7 / 74.9%, E4B 79.9 / 76.4%** (targets 65 / 75%), every split p < 0.001 after Holm; no-harm family: no significant drop (Holm p = 1.0 for all six) ([`results/finetuning.md`](results/finetuning.md))
- [ ] Jetson: the same runs for cost (latency, J per answer) against the re-run baselines, and the adapter load check on `a894dae`
- [x] `results/finetuning.md`: paired McNemar, Holm over the improvement and no-harm families; outline and summary updated (2026-10-28, accuracy; the Jetson's cost to add)

### Stage 3b: reasoning-guided fine-tuning (weeks 8–9, Kaggle)
*Held-out check first (2026-10-30), then the supervisor's literature request (2026-10-31). Design: [`superpowers/specs/2026-11-01-reasoning-finetune-design.md`](superpowers/specs/2026-11-01-reasoning-finetune-design.md). Implementation plan: [`superpowers/plans/2026-11-01-reasoning-finetune.md`](superpowers/plans/2026-11-01-reasoning-finetune.md).*

- [x] Held-out grading sets: Beetle and SAF in the baseline runner and Kaggle notebook; the Oct 28 adapters graded on them (2026-10-30): small gains, one significant ([`results/finetuning.md`](results/finetuning.md#held-out-grading-sets-beetle-and-saf-2026-10-31))
- [x] Literature: published work on tuning Gemma for grading (2026-10-31); reasoning-guided QLoRA chosen
- [x] Code: reply formats, `edge-grader-reasons`, `--reply` in training and grading, the comparison and gate (plan Tasks 1–5)
- [x] Reasons by Qwen3-14B on Kaggle: 50-answer pilot, full run, owner's spot check (27/30, passed 2026-11-02)
- [x] Train E2B reason-first and label-first; grade Beetle, SAF, SciEntsBank test_uq/test_ud (2026-11-02)
- [x] Compare against the untuned E2B and the Oct 28 adapter; write up (2026-11-03): **gate passed in one cell** (Beetle test_ua, both adapters), no significant gain elsewhere ([`results/finetuning.md`](results/finetuning.md#reasoning-guided-fine-tuning-e2b-stage-3b))
- [ ] Step 4: E4B reason-first and label-first, trained on the lab's DGX Spark in Docker (owner, 2026-11-05; moved from Kaggle's Nov 7 quota reset), `scripts/spark_finetune.sh`; then the Jetson cost and no-harm checks for label-first (otherwise future work in the write-up)

### Phase 0: the baseline on the boards (week 5, first)
*Implementation plan: [`superpowers/plans/2026-10-09-phase0-edge-baseline.md`](superpowers/plans/2026-10-09-phase0-edge-baseline.md).*
- [x] `board_grade.sh` takes a benchmark (`scientsbank` | `gsm` | `news`); dry-run tests (2026-10-09). Plus the boards' own job queue, `scripts/board_queue.py` / `scripts/queue.sh` (owner's request)
- [x] `edge-grader-board-check` pairs GSM and news board runs with Kaggle (2026-10-09; telemetry folder fix 2026-10-10)
- [x] Jetson: SciEntsBank test_uq + test_ud, GSM8K + GSM-Symbolic (1,000), news; E2B + E4B (2026-10-09, clocks restored by a reboot first)
- [x] Pi 5: SciEntsBank test_uq + test_ud, GSM8K + GSM-Symbolic first variant set (100), news; E2B + E4B (2026-10-09/10)
- [x] Judge the board news runs on Kaggle (Qwen3-14B; 0 unparsed)
- [x] Board-check every run; one baseline table, benchmark × model × board, in `results/baselines.md` §4 (`board-check-20261010-065251.json`)
- [x] Gate: 2 of 20 board-vs-Kaggle comparisons have p = 0.033 (Holm-corrected > 0.6). **Passed with the Holm correction** (owner's decision, 2026-10-10; recorded in the outline's own decisions)

### Phase A: baseline vs RAG for every benchmark (week 5, Kaggle)
*Implementation plan: [`superpowers/plans/2026-10-09-phaseA-math-rag.md`](superpowers/plans/2026-10-09-phaseA-math-rag.md).*
- [x] Math corpus: OpenStax *Prealgebra 2e* (CC BY-NC-SA 4.0), pinned in `sources.json`, chunked with the frozen settings: 641 chunks (2026-10-10)
- [x] Retrieval check: 30 GSM8K **train** questions, owner's marking: **21/30** (17/30 at rank 1)
- [x] Kaggle: math RAG-en, E2B + E4B, GSM8K + GSM-Symbolic (2026-10-10): every cell down (E2B −5.0/−4.9, E4B −4.0/−1.4); E2B GSM-Symbolic significant by template (p 0.04)
- [x] Four-benchmark table: news, MMLU-Pro, SciEntsBank, math, baseline vs RAG ([`results/rag.md`](results/rag.md#baseline-against-rag-en-all-four-benchmarks-phase-a-2026-10-10))

### Phase B: variations on the grading baseline (weeks 5–7)
*Implementation plans: [B1 thinking](superpowers/plans/2026-10-09-phaseB1-thinking.md), [B2 little-gemma + reuse](superpowers/plans/2026-10-09-phaseB2-little-gemma-reuse.md), [B3 EmbeddingGemma 2](superpowers/plans/2026-10-09-phaseB3-embeddinggemma2.md).*
- [x] B1 thinking on: parser reads after the thought block (test); `max_tokens` 336; boards first (2026-10-11): Jetson all splits, Pi test_ua + test_uq, E2B + E4B, against each board's thinking-off baseline; Kaggle optional. Done 2026-10-13: +3.0 to +18.8 points on all 10 runs (p ≤ 0.004), at 11× (Pi) and 35–40× (Jetson) the energy; MMLU-Pro side copied from `agentic-edge` (no change): [`results/variations.md`](results/variations.md#b1-thinking-on)
- [x] B2 little-gemma on the Jetson: why test_ua had no prefix reuse; `lg-base`, `lg-reuse`, `llama-sorted` on test_ua; byte-identical check. Done 2026-10-13, with `--swa-full` as the llama.cpp reuse control in place of `llama-sorted` (pilot, 2026-10-13): accuracy unchanged (p ≥ 0.38); energy −50% / −21% (little-gemma, E2B / E4B), −42% / −44% (`--swa-full`); `lg-reuse` byte-identical to `lg` and reuses nothing on grading prompts: [`results/variations.md`](results/variations.md#b2-engine-little-gemma-and-prefix-reuse)
- [ ] B3 EmbeddingGemma 2: loads on the Mac?; retrieval check on the 30 SciEntsBank train questions; grading run only at ≥ 20/30

### Phase C: text + images (weeks 6–7)
*Implementation plan: [`superpowers/plans/2026-10-09-phaseC-images.md`](superpowers/plans/2026-10-09-phaseC-images.md).*
- [ ] C1: render answers (pinned font), private Kaggle dataset with sha256; matching vision projector; reading check on 100; Kaggle test_ua + test_uq; cost on the Jetson
- [ ] C2: data search (real student answers, figures, labels, English, licence, ~300+); run or drop, recorded in the outline

### Stage 4: Edge cost (week 8)
*Device rates (tok/s, power, J/token) are reused from `agentic-edge`, not re-measured: [`results/edge-benchmarks.md`](results/edge-benchmarks.md).*

*The Pi 5 is dropped (supervisor, 2026-10-19): its accuracy matched the Jetson's in every comparison but one (p = 0.024, 0.096 after Holm), at 3–9× the time. Its phase 0 and B1 runs stay as recorded; no new Pi runs. Every run below records all metrics: accuracy (macro-F1 for grading; flexible and strict for GSM; the judge's grade for news), latency, J per answer, prompt and completion tokens.*

- [ ] Every version (baseline, RAG, and the phase B/C versions that pass) × E2B/E4B on the Jetson with `board_grade.sh`
- [x] RAG-en on the **Jetson first** (supervisor, 2026-10-19): SciEntsBank test_ua/uq/ud, news, GSM8K + GSM-Symbolic (1,000), E2B + E4B, 12 jobs (`scripts/queue.sh jetson add --preset rag`, which since 2026-10-19 also adds the 4 MMLU-Pro jobs below). Same server flags as phase 0; the frozen retrieval files, as on Kaggle. No Pi runs (dropped 2026-10-19). Done 2026-10-20 (jobs 24–35, 2026-10-19 23:30 → 2026-10-20 11:44, all rc=0), in `results/board/{rag,news-rag,gsm-rag}-*-jetson-*`; not analysed yet
- [ ] Judge the two Jetson news RAG-en runs on Kaggle (Qwen3-14B, as for phase 0)
- [ ] MMLU-Pro on the Jetson, so RAG covers all four benchmarks: `BENCH=mmlupro` in `board_grade.sh` (the `edge-grader-mmlupro` runner; set `all` = s1–s3, 300 questions) and in the `rag` preset ✅ (2026-10-19); not in the 2026-10-19 queue (the Jetson had the code from before that commit), so deploy and add the 4 jobs on their own; run the baseline (done 2026-10-21, jobs 48–49) and RAG-en, E2B + E4B (4 jobs, ~8 h without RAG at `agentic-edge`'s Jetson rates, more with the longer RAG prompt). The baseline run is the pairing control: the copied `agentic-edge` numbers come from lm-eval
- [ ] Board-check every RAG run against the Jetson's own baseline (paired McNemar; by template for GSM-Symbolic) and add a Jetson column to the four-benchmark table in `results/rag.md`. SciEntsBank + math done 2026-10-20: same as Kaggle (E2B down on every cell, significant after Holm on test_ud and GSM-Symbolic; E4B no significant change), board matches Kaggle on every cell ([`results/rag.md`](results/rag.md#jetson-baseline-against-rag-en-stage-4-2026-10-20)); news (after the judge) and MMLU-Pro to add
- [x] Jetson clocks changed between 2026-10-13 and 2026-10-19 (GPU max 612 → 1,020 MHz, memory 2,133 → 3,199 MHz, same nvpmodel): RAG cost isn't comparable with phase 0. Ask the supervisor whether it was deliberate; then either re-run the baselines in the new state or restore the old limits before stage 4's cost runs. Resolved 2026-10-21: baselines re-run in the new state (owner)
- [ ] RAG on the board: include embedding and retrieval time in the latency
- [ ] Oct 28 adapters on the Jetson, E2B + E4B, test_uq + test_ud, against the 2026-10-21 baselines (the stage 3 Jetson item; first, as the most accurate versions)
- [ ] Stage 3b E2B label-first and reason-first on the Jetson, test_uq: same accuracy, ~2 against ~65 tokens per answer, so the cost of a written reason on the device
- [ ] Optional: the best adapter with `--swa-full` (the cheapest setting at the best accuracy)
- [ ] Trade-off table, every Jetson version against its same-board baseline: Δ accuracy (McNemar), × latency, × energy, mean/peak W, decode tok/s, peak RSS, max temperature, extra J per extra correct answer (metrics in the outline)
- [ ] Main charts: accuracy vs J per answer and accuracy vs latency, for every version on the Jetson, with the front marked (the Pi 5's baseline and thinking-on points for reference)

### Write-up (weeks 10–12)
- [ ] Figures: the main chart, per-split tables, confusion matrices
- [ ] Report: motivation, setup, results per RQ, limitations (dataset age and possible memorisation, one dataset, quantization caveat)
- [ ] Supervisor review, then revise
- [ ] Repo cleanup: README results summary, reproducible commands for every table

## Optional, if time allows
- Phase C (images, RQ7) and B3 (EmbeddingGemma 2): optional since 2026-11-04, cut first (supervisor: focus on edge cost)
- A second dataset (e.g. SAS-Bench 2025, as a check against memorisation)
- The cascade idea: small model first, escalate uncertain answers (check overlap with `agentic-edge`'s `analyze_cascade.py` first)
- Fairness: are non-native English answers scored lower?
- Cloud reference (RQ5, the NotebookLM question): a cloud model with the same material, on the same splits. Optional since 2026-10-19; if it isn't run, the report lists it as a limitation

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
| 2026-10-06 | MMLU-Pro no-RAG and RAG-en done: runner matches the lab; RAG no significant change; stage 2 core complete (MMLU-Pro RAG-id remains optional) | Both benchmarks give the same answer to RQ2 |
| 2026-10-07 | MMLU-Pro RAG-id dropped after its retrieval check scored 0/30; stage 2 complete | The Indonesian SMA material has nothing usable for college-level questions, so grading would measure only distraction |
| 2026-10-07 | Topic narrowing added to week 5: math (GSM8K + GSM-Symbolic) against post-cutoff news, no RAG and RAG. Pending the supervisor | Supervisor asked which of logic, math, news and general knowledge suits these models. MMLU-Pro answers math only; math and news are the two ends of the knowledge-vs-skill contrast |
| 2026-10-07 | Topic benchmarks built ahead of the supervisor's reply, at the owner's request: math runner, LiveNewsBench in place of a self-built news set, news corpus and retrieval. Kaggle runs and the judge remain | The build is cheap and doesn't touch SciEntsBank or stage 3; a published, human-verified post-cutoff set with source articles already existed |
| 2026-10-08 | News judge: Qwen3-14B on Kaggle instead of GPT-4.1; the planned agreement checks (GPT-4.1, hand grades) dropped, owner's choice | GPT-4.1 needs paid API credit (free tier: 50 requests a day, 800 needed); the open judge is free and reproducible |
| 2026-10-08 | **Fine-tuning dropped**; phases 0 (baseline on the boards, first), A (math RAG), B (thinking, little-gemma + prefix reuse, EmbeddingGemma 2) and C (images) take weeks 5–7 | Supervisor, 2026-10-08: the findings are in the baseline, finish it on the edge first, show baseline vs RAG for every benchmark, narrow multimodal to text + images. Design: `superpowers/specs/2026-10-08-baseline-variations-design.md` |
| 2026-10-10 | Phase 0 runs done: 20 board runs, all complete; board = Kaggle on 18/20, two at p = 0.033 (opposite directions); gate decision pending | Supervisor 2026-10-08: baseline on the edge first. Results: `results/baselines.md` §4 |
| 2026-10-10 | Phase A done: math RAG-en (Prealgebra) lowers GSM accuracy for both models; the four-benchmark baseline-vs-RAG table is in `results/rag.md` | Supervisor 2026-10-08: compare each baseline with RAG. Math and news had the same retrieval check (21/30) and went opposite ways |
| 2026-10-11 | Edge first, cloud optional: B1 runs on the boards (Jetson all splits, Pi test_ua + test_uq), compared within each board; Kaggle thinking runs optional | Owner 2026-10-11. Outline, own decisions |
| 2026-10-19 | Cloud reference (RQ5) moved from stage 4 to the optional list; stage 4 is edge cost only | Owner 2026-10-19: the focus is small models on the edge. Not run → stated as a limitation. The supervisor asked for it (2026-09-29), so confirm with them. Outline, own decisions |
| 2026-10-19 | Stage 4 starts with RAG-en on the Jetson only, all three board benchmarks; `board_grade.sh` takes `RAG` for GSM and news too | Supervisor: the Pi 5 results were too weak and slow to lead with; the Jetson ran the whole phase 0 baseline, so baseline vs RAG is complete on one board |
| 2026-10-19 | Pi 5 dropped for new runs; stage 4 (Jetson only) brought forward to week 6, starting with RAG-en for all four benchmarks, MMLU-Pro included (`BENCH=mmlupro` to add); phase C cost on the Jetson | Supervisor 2026-10-19: the Pi's results match the Jetson's, and RAG should be run for every benchmark on the board. Outline, supervisor feedback |
| 2026-10-21 | **Fine-tuning revived** (stage 3, weeks 6–7, Oct 21 – Nov 3); phase C moves after it. Every Jetson baseline re-run at the new clocks (14 jobs queued); the voice-agent services on the Jetson stopped for it | Owner: RAG didn't improve grading. The clock change made phase 0's cost incomparable with later runs. Outline, own decisions |
| 2026-10-22 | Stage 3 Day 1 check on Kaggle passed: **main path** (adapter converts and loads with `--lora` on the unchanged GGUF). Training batch 1 × 16 accumulation instead of 4 × 4, same effective batch | Batch 4 ran the T4 out of memory at the loss (Gemma 4's 262,144-token vocabulary); the template, conversion and server load all worked |
| 2026-10-23 | Stage 3: E2B and E4B adapters trained on Kaggle (3 epochs; validation picked epoch 3 for E2B, 2 for E4B). Kaggle sanity check on 100 test_ua answers: E2B 50 → 75% (p < 0.001), E4B 68 → 72% (p 0.56) | Week 1 of the two-week stage 3 timeline. Next: adapter GGUFs to the Jetson, then the board runs (test_uq/test_ud are the result) |
| 2026-10-27 | Stage 3 adapters re-trained as committed Kaggle versions, same settings; validation picked epoch 2 for E2B, 1 for E4B | The 2026-10-23 adapters weren't kept as files. test_ua (100) moved by a point (E2B 75 → 76%, E4B 72 → 73%) |
| 2026-10-28 | Stage 3 accuracy runs on **Kaggle instead of the Jetson** (owner), **smoke test first** (supervisor): the first 100 of test_uq/test_ud, the full runs only if accuracy rose | The Jetson was off the network; Kaggle's accuracy stands for the boards' (phase 0). The smoke test cuts development time. Cost still needs the Jetson |
| 2026-10-28 | Stage 3 accuracy done: fine-tuning works. test_uq / test_ud E2B 59.8 / 60.3 → 76.7 / 74.9%, E4B 67.5 / 70.4 → 79.9 / 76.4% (p < 0.001 after Holm); MMLU-Pro and GSM unchanged | Both above their targets, and at or near the best published numbers. Next: the tuned runs on the Jetson for cost |
| 2026-11-01 | Stage 3b, reasoning-guided fine-tuning of E2B, runs on Kaggle alongside stage 4 (weeks 8–9), done by Nov 15; stage 4 keeps its dates | Owner. The Oct 28 adapters barely moved Beetle and SAF; the supervisor asked for published methods, and Faza et al. 2025 is the closest. The two share no hardware (Kaggle vs Jetson) |
| 2026-11-03 | Stage 3b steps 1–3 done: the gate passed in one cell (Beetle test_ua: both adapters +11.4 over untuned, +5.9 over Oct 28, significant after Holm); no significant gain in the other three; "contradictory" not recovered. Step 4 (E4B, Jetson, no-harm) only if the buffer week has time, label-first first | Design §6. Label-first matches reason-first at 2 tokens per answer instead of ~65, so it is the one to cost on the Jetson (Claude's call) |
| 2026-11-03 | Stage 3b step 4 (E4B) starts at the Kaggle quota reset, Nov 7 08:00 local, not now | Owner. 5.2 GPU h were left; E2B training took 3.5 h (label-first) and 6.5 h (reason-first), and E4B is larger, so a run started now would hit the quota before finishing |
| 2026-11-05 | Stage 3b E4B training moves to the lab's DGX Spark, in Docker with one named container per run (`edge-grader-finetune-e4b-lf`, `-rf`); everything it uses stays in the project folder | Owner. Kaggle's GPU quota was spent; the supervisor offered the Spark, and the lab wants work there in Docker where members can see it. Same recipe (QLoRA, nf4, fp16 compute), so only the hardware differs: a grading check on the Spark against an existing Kaggle run comes before any comparison |
| 2026-11-04 | Edge cost becomes the main result; stage 4 runs the tuned adapters first, then stage 3b label-first vs reason-first; phase C and B3 move to the optional list | Supervisor 2026-11-04: time is short, focus on the edge device, power and generation speed: "if the accuracy is high what is the trade off". Outline, supervisor feedback and own decisions |
| 2026-11-05 | ASAP 2.0 becomes the main grading benchmark; SciEntsBank, Beetle and SAF results stay as recorded. It runs first on the DGX Spark: baseline, fine-tuned E2B, fine-tuned E4B (stage 5). Research target set: accuracy, a tweak counts at +5 points and significant | Supervisor, after discussing it with the owner: ASAP 2.0 is the new metric and is preferred over the old sets. Outline, supervisor feedback and own decisions |
| 2026-11-05 | Stage 5 first runs done on the DGX Spark: ASAP 2.0, E2B and E4B untouched plus their Oct 28 and label-first adapters (SciEntsBank-trained, so transfer, not a fine-tune for ASAP). The E4B Oct 28 adapter is the only gain, QWK 0.466 → 0.619 | Owner. [`results/asap2.md`](results/asap2.md). Open: whether to fine-tune on ASAP itself, keeping the 600-essay sample out of training |
