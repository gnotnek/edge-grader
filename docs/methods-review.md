# Methods besides fine-tuning: a literature shortlist

*Written 2026-11-05 by Claude, at the owner's request, after the outline's 2026-11-04 change (RQ4, the cost of accuracy on the Jetson, is now the main result). These are **proposals, not decisions.** Any method that's adopted gets a row in the outline's "Own decisions" table with the RQ it serves. Sources were found by web search. For most of them only the abstract or a summary was read, not the full paper, so check the figures marked "(abstract)" before you cite them.*

> **2026-11-05, supervisor:** the cascade concept isn't far from Jev (TypeSafe AI's "System One" model, fast decisions with calibrated probabilities). Focus on tweaking the model's settings, one configuration at a time, as the baseline was extended with thinking on/off. So methods 1 and 2 are set aside, and their simulation stays recorded in [variations.md](results/variations.md#cascade-and-defer-to-teacher-simulated-from-b1-2026-11-05). The setting tweaks are listed at the end.
>
> **Later on 2026-11-05, supervisor:** the thinking budget sweep isn't significant, and the lab doesn't really use MTP. Both are dropped (method 5; the MTP and n-gram rows of the tweak table). **Owner:** the research target is grading accuracy, with its edge-device caveats explained ([summary](results/summary.md#accuracy-as-the-target-the-caveats-2026-11-05)).

## What the outline needs

The outline asks which versions sit on the accuracy–energy and accuracy–latency front. It also asks which settings lower cost at the same accuracy. So far the grading ladder on the Jetson (test_ua, [summary](results/summary.md#rq4-what-does-it-cost-on-the-boards)) has three points:

| Version | Accuracy | J / answer |
|---|---|---|
| E2B, reuse-capable engine | ~56% | ~2 |
| E4B, `--swa-full` | ~67% | ~4 |
| E4B, thinking on | ~73% | ~280 |

Between the second and third points the cost jumps 70×, and nothing has been measured in that gap. The tuned adapters add points at about baseline cost. A method helps the outline if it does one of three things:

- **(a)** fills the 4 J to 280 J gap with versions in between
- **(b)** raises accuracy at about the baseline's ~2-token cost
- **(c)** lowers J per answer at the same accuracy

Fine-tuning is excluded here because RQ3 already covers it.

## Shortlist, ranked by fit to RQ4 and board time

| # | Method | Kind | Extra board time | Serves |
|---|---|---|---|---|
| 1 | ~~Confidence cascade: cheap version first, escalate uncertain answers~~ **set aside (supervisor, 2026-11-05)** | (a) | simulated 2026-11-05 | RQ4, RQ6 |
| 2 | ~~Defer to the teacher: grade only confident answers, hand the rest back~~ **set aside (supervisor, 2026-11-05)** | (a) | simulated 2026-11-05 | RQ4 |
| 3 | Read the label from logprobs, then calibrate the label bias | (b) | one re-run per split at baseline cost | RQ4, RQ6 |
| 4 | Retrieve graded **example answers** (not textbook passages) as few-shot examples | (b) | prompt grows; same as RAG-en's cost runs | RQ2, RQ4 |
| 5 | ~~Thinking budget sweep (e.g. 0 / 64 / 128 / 320 tokens)~~ **dropped (supervisor, 2026-11-05)** | (a) | heavy: thinking runs are 14–27 s per answer | RQ6, RQ4 |
| 6 | Jetson power modes (`nvpmodel`) | (c) | one baseline run per mode | RQ4 |
| 7 | Parallel slots (batching a class's answers) | (c) | one baseline run per slot count | RQ4 |
| — | Not recommended: self-consistency voting, MatFormer slicing, speculative decoding (see the end) | | | |

### 1. Confidence cascade

**What it is:** a cheap model grades first. If its confidence is below a threshold, the answer goes up to a dearer version: E2B → E4B → E4B with thinking, or adapter → adapter with thinking.

**Sources:**
- FrugalGPT (Chen, Zaharia, Zou 2023, arXiv 2305.05176) reports matching the best model at 59–98% lower cost (abstract).
- Gupta et al., *Language Model Cascades: Token-level uncertainty and beyond* (ICLR 2024). For generated text, plain sequence confidence has a length bias. With a one-token label that doesn't apply, so classification-style deferral is the right fit here.
- *Cost-Saving LLM Cascades with Early Abstention* (arXiv 2502.09054, 2025) lets the small model abstain.

**Why it fits:** each threshold gives one accuracy/J point, so a threshold sweep draws a curve across the 4 J to 280 J gap. The cost needs no new telemetry. A cascade's J per answer is the share of answers each version handles times that version's measured J per answer.

**How to start without board time:** every grading run already stores per-answer predictions (`results/board/*/raw/*.jsonl`: `id`, `gold`, `pred`). A first signal is whether E2B and E4B agree: accept the label if they do, escalate if they don't. That simulates the cascade from runs you already have, on the same split and clock state. A logprob signal (method 3) is better but needs a re-run.

**Caveats:**
- The threshold is fitted on one split and must be tested on another, so it isn't tuned on test data. Fit on test_ua, report test_uq and test_ud.
- The plan already lists checking `agentic-edge/benchmark/analyze_cascade.py` for overlap first.

### 2. Defer to the teacher (selective grading)

**What it is:** the same machinery as the cascade, but the last stage is the teacher. Report accuracy on the answers the model keeps against the share it keeps (coverage), and the teacher work it saves.

**Sources:**
- Funayama et al., *Preventing Critical Scoring Errors in Short Answer Scoring with Confidence Estimation* (ACL 2020 SRW). It reports error-free scoring on about half of the test data by filtering on confidence (abstract).
- CHiL(L)Grader (arXiv 2603.11957, 2026 preprint): LLM grades that are temperature-scaled and then selectively predicted. It auto-graded 35–65% of answers at expert-level agreement (abstract).
- Korthals et al., *SURE* (MAKE, doi 10.3390/make8030074): disagreement over 20 samples flags wrong grades, which cut manual regrading time by 40–90% (abstract).

**Why it fits:** the outline's motivation is teacher time. "73% at 280 J" and "85% on the 60% of answers it keeps, at 4 J" are different claims, and the second may be the deployable one.

**Caveat:** it changes what "accuracy" means. Report it beside the plain accuracy, never in place of it.

### 3. Logprob labels and calibration

**What it is:** read each label's probability from the first reply token, instead of the generated text. Then correct the model's preference for some labels.

**Sources:**
- Zhao et al., *Calibrate Before Use* (ICML 2021, arXiv 2102.09690), contextual calibration: estimate the bias from a content-free input such as "N/A".
- Zhou et al., *Batch Calibration* (ICLR 2024, arXiv 2309.17249): estimate the bias from the average label probability over the batch being graded. It needs no labels and only a little extra compute (abstract).

**Why it fits:**
- The diagnosed problems are label bias:
  - E2B overuses "contradictory".
  - Both untuned models under-call "correct".
  - RAG shifted the threshold rather than the judgement ([summary](results/summary.md#rq2-does-rag-over-course-material-help)).
- Calibration targets that bias directly at zero decode cost: it's still ~2 tokens.
- It also gives the confidence that methods 1 and 2 need.
- `src/edge_grader/label_prior.py` already reads the top logprobs and applies a fixed prior correction. Batch calibration is the label-free version of the same idea.

**Caveat:** the bias estimate must come from the split being graded without its labels (batch) or from a content-free input (contextual). It must never be fitted on test labels.

### 4. Retrieved graded examples ("RAG over answers")

**What it is:** embed the training answers. For each test answer, put the k most similar graded training answers into the prompt as examples.

**Sources:**
- Wang & Ormerod 2024 (arXiv 2408.03811). This is **the paper behind the project's 78.7 / 76.9 / 73.4% reference numbers**. Their retrieval is over graded student answers, not course material.
- Zhao, Silva & Poulsen, *Language Models are Few-Shot Graders* (arXiv 2502.13337, 2025), on SciEntsBank, Texas and SAF. Retrieved examples beat random ones, and both beat none. They used 5 examples (abstract).

**Why it fits:**
- The project's RAG retrieved textbook passages and didn't help grading. That fits "RAG is for missing facts, not missing skill".
- Graded examples supply what the model is missing: how this grader draws the line between labels.
- It is not fine-tuning, so it's the natural comparison with the adapters: the same training data, used at inference instead of in the weights.
- Board cost is a longer prompt, which prefix caching partly absorbs (RAG-en's prompts grew ~6× and cost −20% to +63%).

**Caveats:**
- test_ua has the same questions in train, so examples there may be near-copies. test_uq and test_ud only get examples from other questions, and those two splits are the fair test.
- Retrieval reuses the frozen `bge-small-en-v1.5`.

### 5. Thinking budget sweep

**What it is:** run thinking on at several token budgets instead of only 320.

**Sources:**
- Muennighoff et al., *s1* (arXiv 2501.19393): budget forcing.
- Ma et al., *Reasoning Models Can Be Effective Without Thinking* (arXiv 2504.09858). An empty thought block often beats thinking under tight budgets, using 2–5× fewer tokens (abstract).
- *Steering LLM Thinking with Budget Guidance* (arXiv 2506.13752). A hard cut can be worse than no thinking.

**Why it fits:**
- Thinking is the largest inference-time gain in the project (+3.0 to +18.8 points).
- Only one budget was measured, and "one thinking budget" is already a listed limitation.
- A sweep gives intermediate front points directly.

**Cost:** it's the heaviest option, at 14–27 s per answer at a 320-token budget. Run test_uq only (733 answers), E2B first.

### 6. Jetson power modes

**What it is:** run the same baseline under each `nvpmodel` mode. The Orin Nano 8 GB has 7 W and 15 W modes, plus 25 W and MAXN SUPER since JetPack 6.2, according to NVIDIA's JetPack 6.2 blog. The 1,020 MHz GPU clock seen since 2026-10-19 is SUPER-mode territory.

**Sources:**
- Arya & Simmhan, *Understanding the Performance and Power of LLM Inferencing on Edge Accelerators* (arXiv 2506.09554, PAISE 2025). It sweeps power modes and batch sizes on a Jetson Orin AGX; lowering the memory clock raised both latency and energy (summary).
- arXiv 2501.08219 (datacentre GPU, not a Jetson) reports ~42% energy saved at the lowest SM clock for a small latency cost (abstract).

**Why it fits:**
- It is the purest "same accuracy, lower cost" setting: temperature 0 means labels should barely change. Check that with McNemar, as in phase 0.
- It also turns the clock change of 10-13 → 10-19 from a confound into a measured factor.

**Caveat:** this changes system settings on the lab's shared board. It needs the owner's and possibly the supervisor's OK, and the board must be put back to its current state afterwards.

### 7. Parallel slots

**What it is:** run `llama-server -np N` with continuous batching. A teacher grades a whole class at once, so the work is a batch job, not interactive.

**Sources:**
- Arya & Simmhan (above): batch sizes 1–128 on Orin.
- arXiv 2605.11999 (an H200 study): batch 1 → 32 cut energy per token by more than 20× (abstract).

**Why it fits:** most grading energy is fixed draw over time. Serving several answers at once spreads that draw across them.

**Caveats:**
- `-np` changes the fixed `llama-server` flags. It's a new variation, and the baseline stays as it is.
- Batched kernels can flip near-tie labels. Board vs Kaggle already flipped 1–5%. Test accuracy with McNemar.

## Not recommended

- **Self-consistency voting:** it multiplies cost by the number of samples. One preprint finds it *lowers* small models' accuracy on most hard science questions (arXiv 2608.11403). Its useful part, disagreement as an uncertainty signal, is covered more cheaply by method 3.
- **MatFormer "mix-n-match" sizes between E2B and E4B:** documented for Gemma 3n (MatFormer Lab, `google/gemma3n-slicing-configs`). No official source confirms that Gemma 4's E2B/E4B are nested the same way, and slicing would need a new GGUF and new tooling.
- **Speculative decoding (E2B drafting for E4B, or n-gram):** the label is 1–2 tokens, which is why phase B dropped it. It could only matter for thinking-on decode, and published Jetson energy numbers are thin (SLED arXiv 2506.09397, HeteroReason arXiv 2609.28717 report speed or assertions, not measured joules).
- **Frozen-embedding classifier (bge-small + logistic regression) as a cost floor:** it would show what ~0 J buys. But it is trained on SciEntsBank's label mix, which repeats the adapters' held-out problem. SemEval-2013's lexical baseline was already hard to beat. Optional at most.

## Model-setting tweaks from agentic-edge (2026-11-05)

Each tweak below is one model with one setting changed, compared with the same board's baseline, as phase B did. "Done here" means it has been measured for grading in this project.

| Tweak | What `agentic-edge` found | Done here for grading? | Worth running for grading? |
|---|---|---|---|
| Thinking on/off (budget 320) | MMLU-Pro unchanged; +20–27% energy | **Done** (B1): +3.0 to +18.8 points at 35–40× energy | No: the budget sweep was dropped (supervisor, 2026-11-05) |
| Engine: little-gemma vs llama.cpp (S3) | Same accuracy, −4% to −17% energy | **Done** (B2): accuracy ±0.4 | No: done |
| Prefix reuse: little-gemma `-reuse` (S9), llama.cpp `--swa-full` | Byte-identical output, less time | **Done** (B2): `--swa-full` halves E2B's energy | Only combined with the best version (adapter + `--swa-full`, already stage 4 item 4) |
| MTP draft (speculative decoding) | **Byte-identical output**, 1.5–1.7× faster; **2.1× faster with thinking on** (Pi 5) | No: dropped 2026-10-08 because a label is 1–2 tokens | No: the lab doesn't really use it (supervisor, 2026-11-05) |
| n-gram speculation (S9 full, S10) | Same accuracy, −6% to −31% energy on MMLU-Pro | No: dropped 2026-10-08 | No: dropped with MTP (2026-11-05) |
| Weight quantization (Q4_K_M / Q4_K_XL QAT / Q5_K_M / Q8_0, Pi 5, E4B) | The QAT Q4 is the smallest, fastest and most accurate | No: Q5_K_M dropped 2026-10-08 | Optional: E2B Q8_0 against the QAT Q4. A non-QAT build changes the weights, which is the outline's QAT caveat. E4B Q8_0 (7.6 GiB) likely doesn't fit the Jetson's 8 GB |
| KV cache quantization (TurboQuant `turbo3`, S8) | E4B −3 to −5 points (MMLU-Pro); n.s. on tool calls | No: dropped 2026-10-08 | No. Grading prompts are ~160 tokens, so the KV cache is tiny and there's little to save |
| Output format (JSON vs Code Mode, tool calls) | Code Mode +6.6 points for E4B | The grading analogue is the reply format: stage 3b label-first vs reason-first | Already covered by stage 3b |
| Clocks locked (`jetson_clocks`) | +5–9% energy at the same speed | Recorded as a confound (10-13 → 10-19 change) | A board setting, not a model tweak (method 6) |

## Suggested order

Since 2026-11-05 (supervisor), the order is one model's settings at a time, toward the accuracy target: at least 5 points over the same model without the tweak, and significant after Holm (owner, 2026-11-05). Stage 4's own order stays first.

1. **The adapters on the Jetson** (stage 4 items 1–2, already planned).
2. **Method 3: logprob labels with calibration**. It changes how the label is read, not the model. Close to the Jev idea, so ask the supervisor before running it.
3. Optional: E2B Q8_0, method 4 (graded examples), method 6 (power modes).

Methods 1, 2 and 5 are set aside.
