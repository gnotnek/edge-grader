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

**Other splits (estimate only, an upper bound).** test_uq and test_ud weren't run on the boards, and stage 4 measures every version anyway. Until then, the estimate scaled by the test_ua ratios above is the best figure. Unseen domains (test_ud, 4,562 answers), unscaled:

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
