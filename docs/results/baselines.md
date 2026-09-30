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

(Accuracy / macro-F1. 0 unparsed replies, 0 errors.)

### Cost on the boards (estimated from `agentic-edge`, not measured)

*Per the supervisor (2026-09-29), no new board runs. The cost comes from agentic-edge's measured per-request rates (same GGUF files and flags), applied to each graded answer's own token counts from the Kaggle runs above. Method: [`edge-benchmarks.md`](edge-benchmarks.md#grading-cost-estimated). File: [`board-cost-estimate-20260930-051813.json`](../../results/estimates/board-cost-estimate-20260930-051813.json), made by `uv run edge-grader-cost results/kaggle/baseline-*.json`.*

Unseen domains (test_ud, 4,562 answers; the other splits are within 7%):

| Model | Board | Accuracy (Kaggle) | Median latency | J / answer | Wh / 1,000 answers | Hours / 1,000 answers |
|---|---|---|---|---|---|---|
| E2B | Pi 5 | 60.3% | 4.7 s | 35.5 | 9.9 | 1.4 |
| E2B | Jetson | 60.3% | 0.48 s | 5.1 | 1.4 | 0.14 |
| E4B | Pi 5 | 70.4% | 7.9 s | 58.5 | 16.2 | 2.3 |
| E4B | Jetson | 70.4% | 0.83 s | 9.3 | 2.6 | 0.24 |

- **Prompt-bound:** about 155–166 prompt tokens and 2–2.5 output tokens per answer. Prefill is about 93% of the latency on both boards.
- **The Jetson is ~10× faster and ~6–7× less energy per answer** than the Pi 5. For MMLU-Pro that ratio was 3.4× and 2.3×. The gap grows here because the Jetson's GPU prefill is ~12× faster, and grading is almost all prefill.
- **E4B on the Jetson beats E2B on the Pi 5 on both counts:** +10 points of accuracy for about a quarter of the energy. E4B costs 1.7–1.8× the energy of E2B on either board.
- **Upper bound:** the estimate counts every prompt token, but llama-server reuses the prompt prefix it shares with the previous request (the instructions, and the question and reference answer when consecutive answers are to the same question). Measured cost should come out somewhat lower.
- **Accuracy on the boards is assumed equal to Kaggle's:** same GGUF, same flags, greedy decoding. agentic-edge found no board difference on MMLU-Pro (McNemar p = 0.71), but CPU, CUDA and T4 kernels can still flip a few borderline labels. One short board run (`scripts/board_grade.sh e2b test_ua`) would check both the accuracy and the cost estimate. It's optional.
- Power is board DC draw, not wall power.

## 3. Side by side

| Model | MMLU-Pro (Pi 5, n=300) | Grading accuracy (Kaggle, test_ud) |
|---|---|---|
| E2B | 51.7% | 60.3% |
| E4B | 65.7% | 70.4% |

The two tasks aren't directly comparable (10-way multiple choice vs 3-way labels), but the E2B → E4B gain is similar on both, about 10–14 points.
