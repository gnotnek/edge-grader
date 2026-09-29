# Baselines

Stage 1 has two baselines for the same untouched models:
- **Subject knowledge:** MMLU-Pro, carried over from `agentic-edge`
- **Grading:** SciEntsBank, measured in this project

Machine-readable copy: [`../results/reference/agentic-edge-mmlupro-baseline.json`](../results/reference/agentic-edge-mmlupro-baseline.json).

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

**Setup:** 3-way labels, zero-shot prompt with the reference answer, thinking off, the same GGUF files and server flags as MMLU-Pro. Details and error analysis: [`../results/kaggle/README.md`](../results/kaggle/README.md).

### Accuracy (Kaggle T4, full splits, 2026-09-29)

| Model | test_ua (540) | test_uq (733) | test_ud (4,562) |
|---|---|---|---|
| E2B | 55.2% / F1 0.518 | 59.8% / F1 0.553 | 60.3% / F1 0.570 |
| E4B | **67.0%** / F1 0.664 | **67.5%** / F1 0.655 | **70.4%** / F1 0.705 |

(Accuracy / macro-F1. 0 unparsed replies, 0 errors.)

### Cost on the boards

*Not run yet* (`scripts/board_grade.sh`).

| Model | Board | Median latency | J / answer | Accuracy (check vs Kaggle) |
|---|---|---|---|---|
| E2B | Pi 5 | | | |
| E2B | Jetson | | | |
| E4B | Pi 5 | | | |
| E4B | Jetson | | | |

## 3. Side by side

| Model | MMLU-Pro (Pi 5, n=300) | Grading accuracy (Kaggle, test_ud) |
|---|---|---|
| E2B | 51.7% | 60.3% |
| E4B | 65.7% | 70.4% |

The two tasks aren't directly comparable (10-way multiple choice vs 3-way labels), but the E2B → E4B gain is similar on both, about 10–14 points.
