# Edge benchmarks (from `agentic-edge`)

*Source: `agentic-edge` commit `12031f0` (2026-09-28). Copied, not re-run. Raw files and a map of what came from where: [`../results/reference/agentic-edge/`](../results/reference/agentic-edge/README.md). Full write-up: [`RESULTS.md`](../results/reference/agentic-edge/docs/RESULTS.md).*

**Why these are reused:** the supervisor asked (2026-09-29) to use the device benchmarks the lab already has instead of re-measuring them, to save time. They use the same models, GGUF files and `llama-server` flags as this project, so the device side carries over. The MMLU-Pro accuracy side is already in [`baselines.md`](baselines.md).

## Setup

- **Boards:** Raspberry Pi 5 8 GB (4× Cortex-A76, CPU only) and Jetson Orin Nano (CUDA, `-ngl 99`)
- **Models:** Gemma 4 E2B / E4B, Unsloth Q4_K_XL **QAT** GGUF
- **Server:** `llama-server -c 8192 --cache-ram 0 -rea off --reasoning-budget -1` (thinking off)
- **Power:** board DC draw summed over the rails (Pi PMIC, Jetson INA3221) at 1 Hz, **not wall power**. Say so wherever a number is quoted.
- **Workload:** MMLU-Pro, 3 × 100-question subsets (n = 300), 5-shot CoT, long answers (median ~1,900 characters)

## Device rates (thinking off, n = 300)

| | Decode speed | Mean power | J per generated token | Energy | Wall time |
|---|---|---|---|---|---|
| E2B Pi 5 | ~6.8 tok/s | ~7 W | 1.15 | 64.2 Wh | 552 min |
| E2B Jetson | ~22.9 tok/s | ~10.1–10.6 W | 0.46 | 25.6 Wh | 155 min |
| E4B Pi 5 | ~3.4 tok/s | ~7 W | 2.20 | 125.5 Wh | 1,111 min |
| E4B Jetson | ~11.6 tok/s | ~10.1–10.6 W | 0.96 | 54.5 Wh | 313 min |

- **Jetson vs Pi 5:** ~3.4× faster decode and 2.3–2.5× less energy per token, same accuracy (McNemar p = 0.71, pooled).
- **E4B vs E2B:** about half the speed and twice the energy per token on both boards.
- **Pi 5 idle** is 2.34 W. No thermal throttling on either board. Max SoC temperature 76.5 °C (Pi, E4B) and 60.2 °C (Jetson).

## Other results that matter here

| Topic | Result | Source |
|---|---|---|
| **Thinking on** (320-token budget) | No significant accuracy change; +20–27% tokens, energy and time. Rates (tok/s, J/token) unchanged. | RESULTS.md §7.2 |
| **MTP** (speculative decoding, llama.cpp `draft-mtp`) | 1.5–1.7× faster on the Pi 5 with thinking off, **byte-identical output**, so no accuracy re-run needed | RESULTS.md §3 |
| **Quantization** (E4B, Pi 5) | Q4_K_XL QAT is the smallest (3.93 GiB) and fastest; Q8_0 needs `--cache-ram 0` or it runs out of memory | RESULTS.md §4 |
| **Engine** | llama.cpp ~6× faster decode and ~40× faster prefill than little-gemma on the Pi 5 | gemma4-pi5-benchmarks.md |

## Using these for grading cost

A grading reply is a single label, so the workload is the opposite of MMLU-Pro: **short output, prompt-heavy**. The J per generated token above spreads prefill energy over ~700 generated tokens per question. Multiplying it by a grading reply's few tokens would underestimate the cost.

Estimate cost per graded answer from the grading runs' own token counts instead:

- **Latency per answer ≈ prompt tokens / prefill speed + output tokens / decode speed.** Pi 5 prefill with llama.cpp: E2B ~67–73 tok/s, E4B ~32–36 tok/s (Sep-14 engine comparison, older flags, so treat as approximate). **The Jetson's prefill speed was not measured.**
- **Energy per answer ≈ mean board power × latency per answer.** Mean power comes from the table above.
- Kaggle T4 latency doesn't carry over to the boards.

This gives a Pi 5 estimate without a board run. For the Jetson, or to check the estimate, one short run (e.g. `scripts/board_grade.sh e2b` on test_ua) is still the cheapest way to get a measured figure. Scale the other conditions from it.
