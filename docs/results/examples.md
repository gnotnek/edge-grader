# Graded examples in the prompt (E4B, SciEntsBank, DGX Spark, 2026-11-10)

**In plain words:** before E4B grades an answer, its prompt shows 4 answers from other students that experts already graded. **Result: it helps where the examples answer the same question (test_ua, +7.2 points, passes the 5-point bar), does nothing significant on unseen questions (test_uq, +4.4), and hurts on unseen domains (test_ud, −3.8, significant).** Similar examples are no better than random ones, and examples add nothing to the label-only adapter, which stays well ahead. The prompt grows about 3.5×.

Design: [spec](../superpowers/specs/2026-11-10-graded-examples-design.md). Comparison: `uv run python -m edge_grader.compare_examples` → [`results/spark/compare-examples-scientsbank-20261111-052532.json`](../../results/spark/compare-examples-scientsbank-20261111-052532.json).

## Setup

- **Model and server:** Gemma 4 E4B, QAT `UD-Q4_K_XL`, `llama-server` at `a3a1c47` with the baseline flags, thinking off, temperature 0, label-only reply. Everything ran on the DGX Spark (GB10), and the 2026-11-08 adapter runs were reused.
- **Examples:** 4 SciEntsBank **train** answers per test answer, with their gold 3-way labels. Only train answers can be picked, and the loader stops on any other id.
  - **kNN (`ex4`):** the most similar answers by `bge-small-en-v1.5` over question + answer, same question first. On test_ua, 97% of the examples come from the same question; on test_uq and test_ud there are none to pick from, so all come from other questions.
  - **Random (`rand4`):** 4 answers drawn from the whole pool, seeded per answer.
- **Retrieval files:** built once on the Mac, in `results/examples/`.
- **Tests:** paired McNemar on 3-way labels, Holm over 12 tests fixed before grading. The 5-point bar (+5 points and Holm p < 0.05) applies to the "with vs without examples" rows.

## Accuracy and macro-F1 (3-way)

| Version | test_ua | test_uq | test_ud |
|---|---|---|---|
| E4B untouched | 65.9% / 0.654 | 69.0% / 0.668 | 70.7% / 0.709 |
| E4B + kNN examples | **73.2% / 0.709** | 73.4% / 0.678 | 66.9% / 0.635 |
| E4B + random examples | 71.1% / 0.667 | 74.5% / 0.695 | 67.7% / 0.645 |
| label-only adapter (`e4b-ft`) | 79.1% / 0.769 | **80.2% / 0.788** | **76.5% / 0.774** |
| adapter + kNN examples | **79.8% / 0.790** | 77.2% / 0.758 | 73.4% / 0.740 |

The untouched Spark runs match Kaggle's (66.7 / 67.5 / 70.4%) within 1.5 points. Answers per split: 540 / 733 / 4,562. Nothing was unparsed and no request failed.

## Tests (Holm over 12)

| Second vs first | Split | Δ accuracy | fixed / broken | Holm p | Bar |
|---|---|---|---|---|---|
| E4B + kNN vs E4B | test_ua | **+7.2** | 81 / 42 | 0.004 | **pass** |
| | test_uq | +4.4 | 117 / 85 | 0.17 | |
| | test_ud | **−3.8** | 413 / 587 | < 0.001 | |
| adapter + kNN vs adapter | test_ua | +0.7 | 35 / 31 | 1.0 | |
| | test_uq | −3.0 | 42 / 64 | 0.20 | |
| | test_ud | **−3.1** | 216 / 358 | < 0.001 | |
| E4B + kNN vs adapter | test_ua | −5.9 | 35 / 67 | 0.014 | n/a |
| | test_uq | −6.8 | 57 / 107 | 0.001 | n/a |
| | test_ud | −9.6 | 295 / 733 | < 0.001 | n/a |
| E4B + kNN vs E4B + random | test_ua | +2.0 | 71 / 60 | 1.0 | n/a |
| | test_uq | −1.1 | 50 / 58 | 1.0 | n/a |
| | test_ud | −0.8 | 246 / 284 | 0.43 | n/a |

**Described, not tested:** random examples against untouched E4B changed accuracy by +5.2 / +5.5 / −3.0 points.

**2-way (correct vs not), secondary:** E4B with kNN examples scored 79.3 / 80.0 / 73.6%, against 72.0 / 74.2 / 74.8% untouched. The per-test 2-way scores and their Holm p-values are in the JSON.

## What it means

1. **The method works only when train holds answers to the same question.** That is Wang & Ormerod's finding too, and here it's sharper. test_ua gains +7.2 points. test_uq gains +4.4, which isn't significant. test_ud loses 3.8 points.
2. **The gain comes from leniency, not from similarity.** Untouched E4B is strict: it recognises only 44–57% of correct answers. With examples, recall on "correct" rises to 0.69–0.79, while recall on "incorrect" (0.82 → 0.63 on test_ud) and "contradictory" (0.70 → 0.48) falls. On test_ua and test_uq the trade is a net gain. On test_ud it overshoots: E4B with kNN examples labels 2,196 answers correct against 1,917 gold. Random examples do the same (2,035), and the kNN − random difference is never significant, so seeing *any* expert-graded answers shifts the model. Picking similar ones adds nothing measurable.
3. **A prompt doesn't replace training.** The label-only adapter beats E4B with examples on every split, by 5.9 to 9.6 points (all Holm p ≤ 0.014). Adding examples to the adapter gains nothing on test_ua and costs 3.1 points on test_ud (significant), because the adapter is already lenient and the examples push it further.
4. **Cost:** the median prompt grows from 144–160 tokens to 531–618 (maximum 1,753), about 3.5–4×. That is all prefill, since the reply is still one label. Prefill is the cheap part of a reply on the Jetson, but it isn't free; it's measured only if this version goes to the board, which these results don't support.
5. **Against the published reference:** Claude 3.5 Sonnet with a fine-tuned retriever reached 78.7 / 76.9 / 73.4%. E4B with the same idea reaches 73.2 / 73.4 / 66.9%. The adapter (79.1 / 80.2 / 76.5%) is still the project's best SciEntsBank grader.

## Limitations

- One k (4), one embedder (bge-small, not fine-tuned) and one prompt layout. None of them was tuned, by design, so a tuned retriever as in Wang & Ormerod might do better on test_ua. Nothing here suggests it would rescue test_uq or test_ud, since random examples do as well as similar ones.
- The random draw comes from the whole pool, so on test_ua it also differs from kNN in using other questions' answers.
- E4B only; E2B and thinking on weren't run.

## E2B (2026-11-13, DGX Spark)

**In plain words:** the same four runs with E2B in place of E4B (decision 2026-11-13). **Graded examples lift untouched E2B on every split and pass the 5-point bar on all three (+15.2 / +10.6 / +5.6 points), unlike E4B. They still don't help the E2B adapter (no split differs), and the adapter stays 7–9 points ahead of E2B with examples.** So for the three-grader question, examples are not E2B's best version; the adapter is.

Same setup as above: E2B QAT `UD-Q4_K_XL`, `a3a1c47`, the same retrieval files, the same 12 tests and Holm family. The adapter is the Kaggle-trained E2B label-only adapter (`e2b-ft`), graded on the Spark. All runs on the whole splits, 0 unparsed. Comparison: `MODEL=e2b uv run python -m edge_grader.compare_examples` → [`results/spark/compare-examples-scientsbank-e2b-20261115-041133.json`](../../results/spark/compare-examples-scientsbank-e2b-20261115-041133.json).

| Version | test_ua | test_uq | test_ud |
|---|---|---|---|
| E2B untouched | 55.2% / 0.521 | 59.5% / 0.548 | 60.7% / 0.574 |
| E2B + kNN examples | 70.4% / 0.674 | 70.1% / 0.606 | 66.3% / 0.551 |
| E2B + random examples | 65.0% / 0.561 | 68.1% / 0.568 | 65.0% / 0.546 |
| label-only adapter (`e2b-ft`) | **79.4% / 0.790** | **76.9% / 0.755** | 74.8% / 0.747 |
| adapter + kNN examples | 77.0% / 0.763 | 76.0% / 0.744 | **75.3% / 0.741** |

| Second vs first | test_ua | test_uq | test_ud |
|---|---|---|---|
| E2B + kNN vs E2B (bar) | **+15.2** (135/53, <0.001) **pass** | **+10.6** (147/69, <0.001) **pass** | **+5.6** (691/435, <0.001) **pass** |
| adapter + kNN vs adapter (bar) | −2.4 (31/44, 0.66) | −0.9 (31/38, 0.66) | +0.5 (241/216, 0.66) |
| E2B + kNN vs adapter | **−9.1** (<0.001) | **−6.8** (<0.001) | **−8.4** (<0.001) |
| E2B + kNN vs E2B + random | +5.4 (0.11) | +2.0 (0.66) | **+1.3** (0.04) |

Accuracy points, (fixed / broken, Holm p over the 12); **bold** = significant. Described, not tested: random examples against untouched E2B, +9.8 / +8.6 / +4.3. 2-way, untouched → kNN: 72.0 → 77.4%, 75.6 → 77.5%, 73.5 → 74.3%.

What it means:
1. **For E2B the examples fix a different fault than for E4B.** Untouched E2B overuses "contradictory": its recall on "incorrect" is only 0.61 / 0.66 / 0.58, against 0.69–0.78 on "contradictory". With examples, "incorrect" recall rises to 0.80 / 0.88 / 0.79 and "contradictory" falls to 0.57 / 0.30 / 0.21. "Correct" recall moves less (0.45 → 0.64 on test_ua, 0.61 → 0.62 on test_ud). E4B, by contrast, was too strict on "correct", and examples made it lenient. Seen after grading, not tested.
2. **Similar examples beat random ones only slightly,** and only on test_ud (+1.3, Holm p = 0.04). As with E4B, most of the gain comes from seeing any graded answers.
3. **Training still wins.** The adapter beats E2B with examples by 6.8–9.1 points on every split, without the 3.5–4× longer prompt (median 531–618 tokens against 144–160). Adding examples to the adapter changes nothing significant.
