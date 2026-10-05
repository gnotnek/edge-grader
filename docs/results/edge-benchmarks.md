# Edge benchmarks (from `agentic-edge`)

*Source: `agentic-edge` commit `12031f0` (2026-09-28). Copied, not re-run. Raw files and a map of what came from where: [`../results/reference/agentic-edge/`](../../results/reference/agentic-edge/README.md). Full write-up: [`RESULTS.md`](../../results/reference/agentic-edge/docs/RESULTS.md).*

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

## Grading cost (estimated)

> **Superseded by measurement (2026-10-05).** Board runs on test_ua came in 10–35% under this estimate, with the same Kaggle accuracy and a Pi/Jetson gap of 7.7–8.8× in latency and 7.0–7.8× in energy. Measured numbers and the reasons for the gap: [`baselines.md`](baselines.md#the-estimate-checked). The method is kept here because the test_uq and test_ud figures still rest on it.

A grading reply is a single label, so the workload is the opposite of MMLU-Pro: **short output, prompt-heavy**. The J per generated token above spreads prefill energy over ~700 generated tokens per question, so multiplying it by a grading reply's 2 tokens would badly underestimate the cost.

The export ([`experiment-full.json.gz`](../../results/reference/agentic-edge/export/README.md)) records, for every MMLU-Pro request on **both** boards, the prompt tokens processed, the prefill and decode time, and board power. llama-server reuses the cached 5-shot prefix, so the median request processed only ~180 new prompt tokens. That's about the size of a grading prompt (~155–166), so the rates are measured at the right scale. `src/edge_grader/cost.py` fits them per board and model, on the 253 requests with ≤ 600 prompt tokens:

| | Prefill ms = fixed + per token | r² | Prefill speed at 160 tokens | Decode ms / token | Median power |
|---|---|---|---|---|---|
| E2B Pi 5 | 369 + 27.5 × tokens | 0.98 | ~33 tok/s | 142 | 7.1 W |
| E2B Jetson | 111 + 1.78 × tokens | 0.96 | ~400 tok/s | 43 | 10.2 W |
| E4B Pi 5 | 482 + 47.1 × tokens | 0.99 | ~20 tok/s | 291 | 6.9 W |
| E4B Jetson | 207 + 3.07 × tokens | 0.97 | ~230 tok/s | 86 | 10.7 W |

Then, for each graded answer: **latency = prefill(prompt tokens) + output tokens × decode ms**, and **energy = median power × latency**. The results per split are in [`baselines.md`](baselines.md#cost-on-the-boards-estimated-from-agentic-edge-not-measured).

- **This replaces the older Pi 5 prefill figures** (E2B ~67–73, E4B ~32–36 tok/s). Those came from the Sep-14 engine comparison, run with different flags and longer prompts. With the current flags (`-t 3`) and prompts this short, prefill is about half that speed.
- **The Jetson's prefill wasn't in the summary tables, but it is in the export.**
- **Upper bound, for a different reason than expected:** the fit is applied to the full grading prompt, and measurement showed no prefix reuse at all. The real cost is lower because the fitted MMLU-Pro requests attended over a ~2,000-token cached context, which slows prefill and decode; a grading request has ~165 tokens of context.
- Kaggle T4 latency doesn't carry over to the boards.
- Checked by measured runs on both boards (test_ua, E2B + E4B): see the note at the top of this section.
