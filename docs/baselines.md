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

*Not run yet.*

| Model | Board | test_ua | test_uq | test_ud | Median latency | J / answer |
|---|---|---|---|---|---|---|
| E2B | | | | | | |
| E4B | | | | | | |

Metric: accuracy and macro-F1, 3-way labels, thinking off. Fill this in from `results/kaggle/` (accuracy) and `results/board/` (accuracy + cost).
