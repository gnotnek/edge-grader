# ASAP 2.0 on the DGX Spark (stage 5)

*2026-11-05. Accuracy only: the Spark isn't an edge device, so cost on the Jetson comes later. Dataset and metric: [`../research-outline.md`](../research-outline.md).*

**Setup:** the fixed 600-essay sample, rubric prompt, greedy, 8 tokens. llama.cpp `a3a1c47` with the lab's flags, in Docker on the GB10 (`scripts/spark_bench_asap2.sh`, container `edge-grader-bench-asap2`, repo `062ae06`). All six runs: 600 essays, 0 unparsed, 0 errors. Summaries are in [`results/spark/`](../../results/spark/).

The adapters were trained on SciEntsBank's three labels and never saw an essay or a 1–6 score, so this measures how they carry over, not a fine-tune for ASAP. They still reply with a score: the rubric prompt is nothing like the training prompt, so the model follows it.

| Version | QWK | Exact | Within 1 | Predicted 3 (gold 219) |
|---|---|---|---|---|
| E2B untouched | 0.568 | **47.0%** | **93.7%** | 309 |
| E2B Oct 28 adapter | 0.545 | 41.2% | 89.3% | 267 |
| E2B label-first | 0.512 | 43.2% | 92.2% | 369 |
| E4B untouched | 0.466 | 42.2% | 92.2% | 404 |
| **E4B Oct 28 adapter** | **0.619** | 45.8% | 90.3% | 266 |
| E4B label-first | 0.480 | 39.0% | 90.2% | 389 |

Paired against the same model untouched, Holm over the 5 comparisons shown. Exact: McNemar. QWK: paired bootstrap, 5,000 resamples, seed 0.

| Comparison | Exact (better / worse) | Holm p | QWK [95% CI] | Holm p |
|---|---|---|---|---|
| E2B Oct 28 adapter | −5.8 (78 / 113) | 0.068 | −0.023 [−0.060, +0.016] | 0.47 |
| E2B label-first | −3.8 (37 / 60) | 0.093 | −0.056 [−0.100, −0.015] | **0.018** |
| E4B Oct 28 adapter | +3.7 (101 / 79) | 0.15 | **+0.153 [+0.109, +0.198]** | **< 0.001** |
| E4B label-first | −3.2 (41 / 60) | 0.15 | +0.014 [−0.020, +0.047] | 0.47 |
| E4B untouched against E2B untouched | −4.8 (62 / 91) | 0.093 | −0.102 [−0.151, −0.052] | **< 0.001** |

- **One version gains: the E4B Oct 28 adapter, on QWK.** Exact accuracy doesn't move significantly. It uses more of the scale: it gives a 3 to 266 essays where untouched E4B gives 404 (gold 219), and a 2 to 135 where untouched gives 61 (gold 171). QWK rewards that spread.
- **Nothing helps E2B.** Both adapters are below the untouched model, significantly so for label-first on QWK.
- **Untouched E4B is below untouched E2B,** the opposite of SciEntsBank. It piles two thirds of its scores on 3.
- **Label-first gives no gain on either model.** On SciEntsBank's held-out sets it matched the reason-free adapter; here it doesn't. With the adapters' own reply habit, its replies run to the 8-token cap (4,800 completion tokens against 1,200), though the score comes first.
- **Against the 5-point bar:** no version adds 5 points of exact accuracy. Whether the E4B adapter's +0.153 QWK passes depends on how the bar reads on ASAP, which the outline leaves open.

**Read with care:** one training seed per adapter, one prompt, 600 of ~24.7k essays. The E4B adapter's gain is a change in how it spreads scores, and why training on three short-answer labels would do that isn't known. No adapter was trained on ASAP.
