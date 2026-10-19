# Baselines

Stage 1 has two baselines for the same untouched models:
- **Subject knowledge:** MMLU-Pro, carried over from `agentic-edge`
- **Grading:** SciEntsBank, measured in this project

Machine-readable copy: [`../results/reference/agentic-edge-mmlupro-baseline.json`](../../results/reference/agentic-edge-mmlupro-baseline.json).

## 1. MMLU-Pro (from `agentic-edge`)

*Source: `agentic-edge/findings/RESULTS.md` §7.1–7.2, repo commit `12031f0` (2026-09-28). Copied, not re-run.*

**Protocol:**
- lm-eval `mmlu_pro`, 5-shot chain of thought, greedy decoding, answer cap 2048 tokens
- 3 × 100-question stratified subsets (seed 20260918), so **n = 300** per cell
- Gemma 4 **Q4_K_XL QAT** GGUF, `llama-server -c 8192 --cache-ram 0 -rea off --reasoning-budget -1`

### Accuracy

| Model | Pi 5 | Jetson Orin Nano | Pi 5, thinking on | Jetson, thinking on |
|---|---|---|---|---|
| **E2B** | **51.7%** | **52.7%** | 50.0% | 49.7% |
| **E4B** | **65.7%** | **66.0%** | 66.3% | 66.7% |

- **Boards:** Pi 5 and Jetson are not significantly different (McNemar p = 0.71, pooled).
- **Thinking on** (320-token budget): no significant change, and it costs 20–27% more tokens, energy and time.
- **Headline figure:** `agentic-edge` quotes 56% / 64%, which comes from the first 100-question subset only. **Use the pooled n = 300 figures above** as the baseline.

### Device cost (thinking off, totals for 300 questions)

| | Energy | Wall time | J per generated token | Decode speed |
|---|---|---|---|---|
| E2B Pi 5 | 64.2 Wh | 552 min | 1.15 | ~6.8 tok/s |
| E2B Jetson | 25.6 Wh | 155 min | 0.46 | ~22.9 tok/s |
| E4B Pi 5 | 125.5 Wh | 1,111 min | 2.20 | ~3.4 tok/s |
| E4B Jetson | 54.5 Wh | 313 min | 0.96 | ~11.6 tok/s |

Power is board DC draw summed over the rails, not wall power.

**How to compare with grading:** MMLU-Pro answers are long (5-shot CoT, median about 1,900 characters). A grading answer is a single label, so **cost per question is not comparable**. Compare **J per token**, and compute cost per graded answer from the grading runs' own token counts.

## 2. SciEntsBank grading (this project)

**Setup:** 3-way labels, zero-shot prompt with the reference answer, thinking off, the same GGUF files and server flags as MMLU-Pro. Details and error analysis: [`../results/kaggle/README.md`](../../results/kaggle/README.md).

### Accuracy (Kaggle T4, full splits, 2026-09-29)

| Model | test_ua (540) | test_uq (733) | test_ud (4,562) |
|---|---|---|---|
| E2B | 55.2% / F1 0.518 | 59.8% / F1 0.553 | 60.3% / F1 0.570 |
| E4B | **67.0%** / F1 0.664 | **67.5%** / F1 0.655 | **70.4%** / F1 0.705 |

Re-run 2026-10-05 on llama.cpp `a3a1c47` (the RAG build): same accuracy, with E2B giving identical labels and E4B within 0.4 points ([build check](rag.md#build-check-baseline-re-run-on-a3a1c47)).

(Accuracy / macro-F1. 0 unparsed replies, 0 errors.)

### Cost on the boards (measured, test_ua)

*Measured 2026-09-30 (Jetson) and 2026-10-04 (Pi 5) with `scripts/board_grade.sh` through the lab job queue: the full test_ua split (540 answers), the same GGUF files and flags as the Kaggle runs, and the Pi at `-t 3` like every agentic-edge Pi run. Run folders: [`results/board/`](../../results/board/). Check file: [`board-check-20261005-042603.json`](../../results/estimates/board-check-20261005-042603.json), made by `uv run edge-grader-board-check results/board/<run>/ ...`.*

| Model | Board | Accuracy (Kaggle) | Same grade as Kaggle | Median latency | J / answer | Wh / 1,000 answers | Hours / 1,000 answers |
|---|---|---|---|---|---|---|---|
| E2B | Pi 5 | 55.7% (55.2%) | 519 / 540 | 3.22 s | 27.8 | 7.7 | 0.97 |
| E2B | Jetson | 55.2% (55.2%) | 507 / 540 | 0.42 s | 4.0 | 1.1 | 0.12 |
| E4B | Pi 5 | 66.5% (67.0%) | 537 / 540 | 6.33 s | 55.6 | 15.5 | 1.90 |
| E4B | Jetson | 66.7% (67.0%) | 534 / 540 | 0.72 s | 7.1 | 2.0 | 0.20 |

- **Board accuracy equals Kaggle's.** No board run differs significantly from its Kaggle run (exact McNemar p ≥ 0.25), and the Pi and Jetson don't differ from each other (p = 0.65 for E2B, 1.0 for E4B). The differing grades are borderline labels that CPU, CUDA and T4 kernels round differently. **So the Kaggle accuracy per split stands for both boards.**
- **The Jetson is 7.7–8.8× faster and uses 7.0–7.8× less energy per answer** than the Pi 5, at about the same mean power (Jetson 9.1–9.7 W, Pi 7.9–8.2 W). For MMLU-Pro the gap was 3.4× and 2.3×; grading is almost all prefill, where the Jetson's GPU gains the most.
- **E4B on the Jetson beats E2B on the Pi 5 on both counts:** 11 points more accuracy for a quarter of the energy. E4B costs 1.8–2.0× the energy of E2B on either board.
- **Energy** is integrated over the grading window only (first request to last), so server start-up and the dataset download aren't counted. It includes idle draw; above idle it's 21.6 / 41.2 J (Pi E2B / E4B) and 2.0 / 3.7 J (Jetson). Board DC draw, not wall power.
- **Threads:** a first Pi pass with llama.cpp's default 4 threads (runs `*-20261004-075411`, `*-082411`, kept) gave identical grades on all 540 answers and was 5–14% faster. The `-t 3` runs above are the ones to quote, since they match the lab's Pi numbers.
- **Prompt-bound:** about 165 prompt tokens and 2–2.5 output tokens per answer, with no prefix reuse between requests (llama-server processed the full prompt every time).

### The estimate, checked

Before the board runs, cost was estimated from agentic-edge's MMLU-Pro request rates ([`edge-benchmarks.md`](edge-benchmarks.md#grading-cost-estimated), [`board-cost-estimate-20260930-051813.json`](../../results/estimates/board-cost-estimate-20260930-051813.json)). On test_ua it came out **high, by 10–35%**:

| | Median latency, measured / estimate | J per answer, measured / estimate |
|---|---|---|
| E2B Pi 5 | 3.22 / 4.98 s (0.65) | 27.8 / 37.7 (0.74) |
| E2B Jetson | 0.42 / 0.50 s (0.84) | 4.0 / 5.3 (0.75) |
| E4B Pi 5 | 6.33 / 8.41 s (0.75) | 55.6 / 62.0 (0.90) |
| E4B Jetson | 0.72 / 0.87 s (0.83) | 7.1 / 9.7 (0.73) |

- **Why:** prefix reuse, the reason the estimate expected, plays no part: each grading prompt was processed in full. Instead, both prefill and decode are faster than in the MMLU-Pro requests the estimate was fitted on (Pi decode 62 vs 142 ms/token for E2B, 129 vs 291 for E4B). Those requests sat on a ~2,000-token cached 5-shot context, and attention over it slows every token; a grading request has only ~165 tokens of context.
- **The thread count isn't the cause:** the estimate's rates come from `-t 3` runs too, and 4 threads changed latency by only 5–14%.
- **The ranking and the Pi/Jetson gap hold:** the estimate put the Jetson ~10× faster and 6–7× cheaper; measured is 7.7–8.8× and 7.0–7.8×.

**Other splits (estimate, superseded 2026-10-10).** Before phase 0 measured test_uq and test_ud on both boards ([§4](#4-phase-0-every-baseline-on-both-boards-measured-2026-10-0910)), this estimate stood in for them. It was high again: test_ud measured 2.95 / 5.70 s on the Pi and 0.40 / 0.70 s on the Jetson (E2B / E4B). Kept for the record. Unseen domains (test_ud, 4,562 answers), unscaled:

| Model | Board | Accuracy (Kaggle) | Median latency | J / answer | Wh / 1,000 answers | Hours / 1,000 answers |
|---|---|---|---|---|---|---|
| E2B | Pi 5 | 60.3% | 4.7 s | 35.5 | 9.9 | 1.4 |
| E2B | Jetson | 60.3% | 0.48 s | 5.1 | 1.4 | 0.14 |
| E4B | Pi 5 | 70.4% | 7.9 s | 58.5 | 16.2 | 2.3 |
| E4B | Jetson | 70.4% | 0.83 s | 9.3 | 2.6 | 0.24 |

## 3. Side by side

| Model | MMLU-Pro (Pi 5, n=300) | Grading accuracy (Kaggle, test_ud) | Grading cost, measured on test_ua (Pi 5 / Jetson) |
|---|---|---|---|
| E2B | 51.7% | 60.3% | 27.8 / 4.0 J per answer |
| E4B | 65.7% | 70.4% | 55.6 / 7.1 J per answer |

The two tasks aren't directly comparable (10-way multiple choice vs 3-way labels), but the E2B → E4B gain is similar on both, about 10–14 points.

## 4. Phase 0: every baseline on both boards (measured, 2026-10-09/10)

*Why: the supervisor (2026-10-08) asked to finish the baseline on the edge before going further. Spec: [`../superpowers/specs/2026-10-08-baseline-variations-design.md`](../superpowers/specs/2026-10-08-baseline-variations-design.md) §4.*

**Runs:**
- 20 folders in [`results/board/`](../../results/board/), run by the boards' own job queue (`scripts/board_queue.py`, which calls `scripts/board_grade.sh`) between 2026-10-09 00:55 and 2026-10-10 ~02:30.
- Same GGUFs and flags as test_ua: Pi `661643e` at `-t 3`, Jetson `a894dae`.
- **Jetson clocks:** restored by a reboot before its runs, so its energy is comparable with the 2026-09-30 test_ua runs. Every run's `board-state.txt` shows the clocks unlocked.

**Check file:** [`board-check-20261010-065251.json`](../../results/estimates/board-check-20261010-065251.json), from `uv run edge-grader-board-check`.

**Trimmed on the Pi:** it ran GSM-Symbolic's first variant set only (100 of 1,000). Its comparisons are on those 100 questions.

**News:** judged by Qwen3-14B on Kaggle, as for the Kaggle runs, with 0 unparsed.

"Same answer" means the same label (grading), the same extracted number (GSM) or the same judge grade (news), on the same questions.

| Benchmark | Set | Model | Board | n | Board acc. | Kaggle acc. (same questions) | Same answer | McNemar p | Median s | J / question |
|---|---|---|---|---|---|---|---|---|---|---|
| SciEntsBank | test_uq | E2B | Pi 5 | 733 | 59.8% | 59.8% | 712/733 | 1.0 | 3.19 | 26.1 |
| SciEntsBank | test_uq | E2B | Jetson | 733 | 59.8% | 59.8% | 705/733 | 1.0 | 0.41 | 3.8 |
| SciEntsBank | test_uq | E4B | Pi 5 | 733 | 68.2% | 67.5% | 718/733 | 0.267 | 6.36 | 52.4 |
| SciEntsBank | test_uq | E4B | Jetson | 733 | 67.3% | 67.5% | 717/733 | 0.791 | 0.71 | 6.9 |
| SciEntsBank | test_ud | E2B | Pi 5 | 4562 | 60.4% | 60.3% | 4446/4562 | 0.484 | 2.95 | 25.0 |
| SciEntsBank | test_ud | E2B | Jetson | 4562 | 60.2% | 60.3% | 4405/4562 | 0.932 | 0.40 | 3.8 |
| SciEntsBank | test_ud | E4B | Pi 5 | 4562 | 70.2% | 70.4% | 4502/4562 | 0.134 | 5.70 | 49.5 |
| SciEntsBank | test_ud | E4B | Jetson | 4562 | 70.1% | 70.4% | 4498/4562 | **0.033** | 0.70 | 6.8 |
| GSM | gsm8k | E2B | Pi 5 | 100 | 84.0% | 83.0% | 90/100 | 1.0 | 21.18 | 227.0 |
| GSM | gsm8k | E2B | Jetson | 100 | 83.0% | 83.0% | 89/100 | 1.0 | 7.16 | 105.9 |
| GSM | gsm8k | E4B | Pi 5 | 100 | 91.0% | 92.0% | 97/100 | 1.0 | 37.31 | 350.5 |
| GSM | gsm8k | E4B | Jetson | 100 | 91.0% | 92.0% | 97/100 | 1.0 | 11.40 | 167.1 |
| GSM | symbolic (1 var.) | E2B | Pi 5 | 100 | 79.0% | 80.0% | 88/100 | 1.0 | 23.71 | 233.9 |
| GSM | symbolic (10 var.) | E2B | Jetson | 1000 | 82.1% | 80.0% | 833/1000 | **0.033** | 7.41 | 112.6 |
| GSM | symbolic (1 var.) | E4B | Pi 5 | 100 | 89.0% | 91.0% | 96/100 | 0.5 | 39.99 | 401.9 |
| GSM | symbolic (10 var.) | E4B | Jetson | 1000 | 90.9% | 90.5% | 955/1000 | 0.572 | 12.28 | 175.5 |
| News | test | E2B | Pi 5 | 200 | 1.0% | 0.5% | 164/200 | 1.0 | 107.95 | 714.7 |
| News | test | E2B | Jetson | 200 | 0.5% | 0.5% | 170/200 | 1.0 | 36.83 | 353.9 |
| News | test | E4B | Pi 5 | 200 | 0.5% | 0.5% | 172/200 | 1.0 | 169.24 | 1311.5 |
| News | test | E4B | Jetson | 200 | 0.5% | 0.5% | 176/200 | 1.0 | 56.89 | 648.9 |

**MMLU-Pro on both boards, copied from `agentic-edge`** (not re-run; the spec's phase 0 table marks it as copied). Same GGUFs and thinking-off flags, run through lm-eval on the boards in 2026-09. Copied 2026-10-19 into [`agentic-edge-mmlupro-variations.json`](../../results/reference/agentic-edge-mmlupro-variations.json). The "Kaggle" column is this project's runner on the same 300 questions ([`rag.md`](rag.md#the-runner-reproduces-the-lab-baseline)), where "same answer" means the same right/wrong. agentic-edge reports only totals, so the time and energy columns are its totals ÷ 300: a **mean**, not a median.

| Benchmark | Set | Model | Board | n | Board acc. | Kaggle acc. (same questions) | Same right/wrong | McNemar p | Mean s | J / question |
|---|---|---|---|---|---|---|---|---|---|---|
| MMLU-Pro | s1–s3 | E2B | Pi 5 | 300 | 51.7% | 51.3% | 87% | 1.00 | 110 | 770 |
| MMLU-Pro | s1–s3 | E2B | Jetson | 300 | 52.7% | 51.3% | 87% | 0.63 | 31 | 307 |
| MMLU-Pro | s1–s3 | E4B | Pi 5 | 300 | 65.7% | 65.3% | 91–92% | 1.00 | 222 | 1506 |
| MMLU-Pro | s1–s3 | E4B | Jetson | 300 | 66.0% | 65.3% | 91–92% | 0.85 | 63 | 654 |

Pi against Jetson on the same questions (Pi time and energy as a multiple of the Jetson's):

| Benchmark | Pi / Jetson time | Pi / Jetson energy | Same answer, Pi vs Jetson | McNemar p |
|---|---|---|---|---|
| SciEntsBank test_uq | 7.8× (E2B), 9.0× (E4B) | 6.9×, 7.6× | 702/733, 716/733 | 1.0, 0.143 |
| SciEntsBank test_ud | 7.4×, 8.1× | 6.6×, 7.3× | 4437/4562, 4504/4562 | 0.382, 0.597 |
| GSM8K | 3.0×, 3.3× | 2.1×, 2.1× | 87/100, 97/100 | 1.0, 1.0 |
| GSM-Symbolic (Pi's 100) | 3.2×, 3.3× | 2.1×, 2.3× | 84/100, 95/100 | 1.0, 0.625 |
| News | 2.9×, 3.0× | 2.0×, 2.0× | 166/200, 180/200 | 1.0, 1.0 |
| MMLU-Pro (`agentic-edge`, n = 300) | 3.6×, 3.5× | 2.5×, 2.3× | same letter ~72%, ~86% | 0.76, 1.00 |

**What it shows:**
- **Board accuracy equals Kaggle's on 18 of the 20 runs** (McNemar p ≥ 0.13). The two exceptions are both p = 0.033, and they point opposite ways: E4B on the Jetson's test_ud is 0.3 points *below* Kaggle (20 answers right only on the board, 37 only on Kaggle), and E2B on the Jetson's GSM-Symbolic is 2.1 points *above* (55 against 34). Over 20 comparisons about one p below 0.05 is expected by chance; corrected for all 20 (Holm), both are above 0.6. **The gate as planned (any p < 0.05 stops phase B on the boards) is tripped as written; the owner accepted it with the Holm correction (2026-10-10), a choice made after seeing the results and recorded as such in the outline.**
- **The Pi/Jetson gap depends on the task's shape.** Grading (a ~165-token prompt, a 2-token label) runs 7.4–9.0× slower and costs 6.6–7.6× more energy on the Pi, as on test_ua (7.7–8.8×, 7.0–7.8×). GSM and news, whose answers run to hundreds of tokens, are only 2.9–3.3× slower and 2.0–2.3× dearer, close to MMLU-Pro's 3.4× / 2.3×: long answers are decode-bound, where the Jetson's lead is smaller.
- **Cost per question spans three orders of magnitude.** Grading costs 3.8–6.9 J on the Jetson and 25–52 J on the Pi; a GSM question 106–176 J and 227–402 J; a news question 354–649 J and 715–1,312 J, for 0.5–1% accuracy without RAG, because the news prompt asks the model to think step by step before it declines. MMLU-Pro, with its 5-shot chain of thought, is the dearest: 307–654 J on the Jetson and 770–1,506 J on the Pi.
- **With MMLU-Pro, all four benchmarks have a baseline on both boards, and none differs between them.** MMLU-Pro's boards agree on the answer letter less often (~72% for E2B) than grading's on the label (96–99%): a long answer gives CPU and CUDA arithmetic more chances to split at a near-tie. The splits cancel out.

