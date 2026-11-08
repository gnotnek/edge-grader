# Findings so far, all stages

*Written 2026-10-19. A summary across the results pages. The numbers, tests and caveats are on the linked pages. MMLU-Pro on the boards, and every MMLU-Pro variation, is `agentic-edge`'s (copied, not re-run). Everything else was measured in this project. "Boards" means the Raspberry Pi 5 (CPU, `-t 3`) and the Jetson Orin Nano (CUDA). Energy is board DC draw, not wall power.*

**Done:**
- RQ1, the baseline: four benchmarks, Kaggle and both boards
- RQ2, RAG-en: all four benchmarks, on Kaggle
- RQ3, fine-tuning: accuracy and the no-harm check, on Kaggle (2026-10-28); cost on the Jetson still to do
- RQ6 B1, thinking on: both boards
- RQ6 B2, engine and prefix reuse: Jetson

**Not done yet:**
- B3, the EmbeddingGemma 2 embedder
- Phase C, image answers (RQ7)
- RAG on the boards (stage 4)

**Optional:** the cloud reference (RQ5), since 2026-10-19. The focus is small models on the edge, so it runs only if time allows, and it's listed under [limitations](#limitations) until then.

## RQ1: how good are the untouched models?

| Benchmark | What it needs | E2B | E4B | Page |
|---|---|---|---|---|
| SciEntsBank grading (3-way), test_ua / test_uq / test_ud | judging a student answer | 55.2 / 59.8 / 60.3% | 67.0 / 67.5 / 70.4% | [baselines §2](baselines.md#2-scientsbank-grading-this-project) |
| MMLU-Pro (300) | broad subject knowledge | 51.7–52.7% | 65.7–66.0% | [baselines §1](baselines.md#1-mmlu-pro-from-agentic-edge) |
| GSM8K (100) / GSM-Symbolic (1,000) | a math skill | 83.0 / 80.0% | 92.0 / 90.5% | [topics](topics.md#math-results-kaggle-t4-2026-10-08) |
| News after the cutoff (LiveNewsBench, 200) | facts the model can't have | 0.5% | 0.5% | [topics](topics.md#news-results-kaggle-t4-judged-by-qwen3-14b-2026-10-08) |

1. **E4B is 10–14 points better than E2B on everything the model can know**: grading, MMLU-Pro and math alike. On news, neither can know the facts.
2. **Grading accuracy is moderate.**
   - Both models are weakest at recognising *correct* answers (recall 0.40–0.61).
   - E2B overuses "contradictory".
   - Accuracy doesn't drop on unseen questions or unseen domains. test_ud is the highest split for both models.
3. **Math is the models' strongest subject, and it isn't memorised.**
   - Math is their best MMLU-Pro subject (E2B 79%, E4B 88%).
   - GSM-Symbolic's new names and numbers cost 1.5–3 points, within noise (p 0.20, 0.32).
4. **Without the facts, the models decline rather than invent.** On post-cutoff news, over three-quarters of the answers are "information unavailable".
5. **The boards give Kaggle's accuracy, and each other's.**
   - 18 of 20 board-vs-Kaggle comparisons are within noise. The other two are at p = 0.033, in opposite directions, and above 0.6 after Holm.
   - On MMLU-Pro, Pi vs Jetson gives p = 0.71, pooled.
   - So an accuracy measured on Kaggle stands for both boards. Individual answers do differ at near-ties: grading labels on 1–5% of answers, MMLU-Pro letters on 14–28%.

## RQ2: does RAG over course material help?

Kaggle, `bge-small-en-v1.5`, top 3 passages ([rag.md](rag.md#baseline-against-rag-en-all-four-benchmarks-phase-a-2026-10-10)):

| Benchmark | E2B baseline → RAG-en | E4B baseline → RAG-en | Retrieval check |
|---|---|---|---|
| News | 0.5% → **39.0%** | 0.5% → **45.0%** | 21/30 |
| MMLU-Pro | 51.3% → 52.0% (p 0.89) | 65.3% → 63.3% (p 0.44) | 15/30 |
| SciEntsBank test_ud | 60.3% → 55.1% (**p < 0.001**) | 70.4% → 70.7% (p 0.59) | 16/30 |
| GSM8K / GSM-Symbolic | 83.0 → 78.0% / 80.0 → 75.1% (**p 0.04** by template) | 92.0 → 88.0% / 90.5 → 89.1% | 21/30 |

1. **RAG helps only where the task lacks knowledge the passages supply.** It gives a large gain on news, nothing on MMLU-Pro and grading, and a small loss on math.
2. **Retrieval quality doesn't explain it alone:** math and news scored the same retrieval check (21/30) and went opposite ways.
3. **On grading, RAG shifts the threshold rather than improving judgement.**
   - Recall on *correct* rises by up to 14 points, and recall on *incorrect* falls by up to 20.
   - That is a net loss for the lenient-prone E2B and a wash for E4B.
4. **The smaller model is hurt more when the material isn't needed:**
   - grading test_ud: E2B −5.1, E4B +0.3
   - math: E2B −4.9 to −5.0, E4B −1.4 to −4.0
5. **Passages disturb the answer format.**
   - E4B drops GSM8K's `#### N` (strict 85% → 51%).
   - More MMLU-Pro answers run out of tokens (E2B 19 → 29).
6. **Indonesian material (RAG-id) hurts more than English.**
   - On SciEntsBank it's significantly worse on unseen domains for both models (E2B −7.4, E4B −4.2), although its retrieval check was no worse (18/30 against 16/30) ([rag.md](rag.md#grading-results-rag-id-kaggle-t4-2026-10-05)).
   - For MMLU-Pro it found nothing usable (0/30), so it wasn't graded.

## RQ3: does fine-tuning help?

QLoRA on the SciEntsBank train split, served with `--lora` on the unchanged Q4 GGUF. Paired with the untuned runs on the same build; Holm within each family ([finetuning](finetuning.md)).

| | E2B untuned → tuned | E4B untuned → tuned |
|---|---|---|
| SciEntsBank test_ua / test_uq / test_ud | 55.2 / 59.8 / 60.3 → **79.4 / 76.7 / 74.9%** | 66.7 / 67.5 / 70.4 → **79.1 / 79.9 / 76.4%** |
| MMLU-Pro (300) | 51.3 → 49.0% | 65.3 → 66.0% |
| GSM8K / GSM-Symbolic (100 each) | 83 / 80 → 83 / 81% | 92 / 91 → 90 / 90% |

1. **Fine-tuning is the largest grading gain in the project:** +6 to +24 points, every split p < 0.001 after Holm. Both models beat their targets (65% and 75%) on unseen questions and domains.
2. **It fixes the baseline's weak spot.** Recall on *correct* answers rose from 0.40–0.60 to 0.73–0.78, and E2B stopped overusing "contradictory".
3. **E2B catches up with E4B** (within 3 points on every split). Both are at or near the best published SciEntsBank numbers (Claude 3.5 Sonnet + RAG: 78.7 / 76.9 / 73.4%).
4. **No harm to general ability:** no significant drop on MMLU-Pro or math (Holm p = 1.0 for all six). One E2B MMLU-Pro subset fell 10 points, mostly from more answers hitting the token cap.

## RQ6: do inference-time settings matter?

[variations.md](variations.md):

| Variation | Grading (measured on the boards) | MMLU-Pro (`agentic-edge`) |
|---|---|---|
| **Thinking on** (budget 320) | **+3.0 to +18.8 points**, all 10 runs p ≤ 0.004 | −3.0 to +0.7, none significant |
| little-gemma instead of llama.cpp (Jetson) | ±0.4, p ≥ 0.84 | ±0.7, p ≥ 0.85 |
| Prefix reuse (`--swa-full`, little-gemma `-reuse`) | ±0.9, p ≥ 0.38 | byte-identical |

1. **Thinking is the one setting that changes grading, and it's the biggest gain in the project.**
   - A grading reply is a bare label, so the thought block is the model's only reasoning.
   - MMLU-Pro already asks for chain of thought, so thinking adds little there.
2. **Thinking improves judgement rather than leniency, unlike RAG.**
   - Recall on *correct* rises (E4B test_uq 0.40 → 0.70) and recall on *incorrect* holds.
   - E2B stops overusing "contradictory" (1,100 → 308 on test_ud).
3. **The engine and prefix reuse are accuracy-neutral;** they change only cost.

## RQ4: what does it cost on the boards?

Grading, test_ua (thinking off unless stated):

| Version | Board | Accuracy | J / answer | Median s |
|---|---|---|---|---|
| E2B, little-gemma | Jetson | 55.6% | **1.96** | 0.18 |
| E2B, llama.cpp `--swa-full` | Jetson | 56.1% | 2.30 | 0.22 |
| E2B, llama.cpp | Jetson | 55.2% | 3.95 | 0.42 |
| E4B, llama.cpp `--swa-full` | Jetson | 67.0% | **4.00** | 0.37 |
| E4B, llama.cpp | Jetson | 66.7% | 7.12 | 0.72 |
| E2B, llama.cpp | Pi 5 | 55.7% | 27.8 | 3.22 |
| E4B, llama.cpp | Pi 5 | 66.5% | 55.7 | 6.33 |
| E2B, thinking on | Jetson / Pi 5 | 67.0 / 66.7% | 139 / 303 | 13.8 / 43.6 |
| E4B, thinking on | Jetson / Pi 5 | **72.8** / 72.0% | 280 / 593 | 27.0 / 87.1 |

1. **The Jetson's lead depends on the task's shape.**
   - Grading is a short prompt and a 2-token label, all prefill. There the Jetson is 7–9× faster and 6.6–7.8× cheaper than the Pi.
   - Long answers are decode-bound: GSM, news, MMLU-Pro and thinking-on grading. There the Jetson is only 2.9–3.6× faster and 2.0–2.5× cheaper.
2. **Cost per question spans three orders of magnitude.**
   - Jetson: grading 2–7 J, GSM 106–176 J, MMLU-Pro 307–654 J, news 354–649 J.
   - Pi: everything is 2–8× that.
3. **E4B on the Jetson beats E2B on the Pi on both counts:** +11 points for a quarter of the energy.
4. **The accuracy ladder for grading on the Jetson:**
   - ~56% for ~2 J (E2B, a reuse-capable engine)
   - ~67% for ~4 J (E4B with `--swa-full`)
   - ~73% for ~280 J (E4B with thinking)
   - The last step costs 70× the energy for +6 points on test_ua. Against the plain baseline, thinking adds +12.6 on test_uq.
5. **RAG's cost on the Jetson (stage 4, 2026-10-21), against the baseline re-run in the same clock state:** the grading prompt grows ~6.3× (160 → 1,035 tokens), but llama.cpp's prompt cache reuses most of it, so grading test_uq/test_ud gets 13–20% cheaper and test_ua 60–63% dearer; math costs 12–40% more and news 2–14% ([`rag.md`](rag.md#cost-rag-en-against-the-re-run-baseline)). RAG brought no grading gain to pay for it. The Pi 5 isn't measured (dropped for new runs).
6. **A cascade fills the gap between ~7 J and ~270 J (simulated from existing runs, 2026-11-05; set aside the same day by the supervisor: one model configuration at a time).**
   - E2B and E4B grade every answer, and only their disagreements (31–37%) get thinking on.
   - Escalating to E4B thinking reaches 70.9 / 75.8 / 72.5% (test_ua / test_uq / test_ud) at 95–109 J, about 35–40% of always-on thinking's energy.
   - Handing the disagreements to the teacher instead keeps 63–69% of answers at 72–76% accuracy, for 11 J ([variations.md](variations.md#cascade-and-defer-to-teacher-simulated-from-b1-2026-11-05)).

## What it adds up to

- **Fine-tuning is what makes the small models good graders:** +6 to +24 points on SciEntsBank, at no measured cost to general ability (accuracy on Kaggle; cost on the Jetson to come).
- **For grading on the device, how the model is run matters more than what it's given.**
  - Thinking raised grading accuracy on every split and board.
  - Course material didn't, and it hurt E2B.
  - The engine and prefix reuse change cost only, and halve it on the Jetson.
- **RAG is for missing facts, not missing skill.** It turned 0.5% into 39–45% on post-cutoff news, and slightly lowered math, which the models already do well.
- **The deployable choices on the Jetson are E4B + `--swa-full` (cheap) or E4B + thinking (accurate).** The Pi runs everything at the same accuracy, only slower and dearer.

**Open questions:**
- Does thinking's gain survive RAG?
- Can n-gram drafting cut thinking's decode cost on grading, as it did on MMLU-Pro?
- Does a better embedder (B3) or a cleaner corpus change the RAG picture?

## Accuracy as the target: the caveats (2026-11-05)

The research target is grading accuracy (owner, 2026-11-05). These are the caveats of chasing it on the Jetson, from runs already recorded. The tuned models have no Jetson cost yet, so that part is still open.

### The 5-point bar

A tweak counts if it adds **at least 5 points** of accuracy over the same model without it, per split, **and** the gain is significant: paired McNemar, p < 0.05 after Holm within its family (owner, 2026-11-05). The bar was set after most of these results were in. Gains below are against the untouched model. Bold passes both; "n.s." reaches 5 points but isn't significant.

| Version | SciEntsBank test_ua / uq / ud | Beetle ua / uq, SAF ua / uq (held out) |
|---|---|---|
| E2B, Oct 28 adapter | **+24.2 / +16.9 / +14.6** | +5.5 n.s. / **+6.4** / +6.9 n.s. / +3.1 |
| E4B, Oct 28 adapter | **+12.4 / +12.4 / +6.0** | +3.4 / +1.7 / +1.1 / +0.6 |
| E2B, label-first adapter | not graded / **+16.7 / +13.0** | **+11.4 / +6.7** / +5.3 n.s. / −1.7 |
| E2B, reason-first adapter | not graded / **+15.4 / +10.7** | **+11.4** / +5.0 n.s. / +3.2 / +4.6 |
| E4B, label-first adapter | not graded / **+11.1** / +2.7 | not run |
| E4B, reason-first adapter | not graded / **+8.9** / −0.8 | not run |
| E2B, Beetle + SAF label-only adapter | **+14.1 / +16.7 / +8.2** | trained on them |
| E2B, Beetle + SAF label-first adapter | +4.4 / **+8.9** / −2.8 | trained on them |
| E2B, Beetle + SAF reason-first adapter | +0.9 / **+7.6** / −0.8 | trained on them |
| E4B, Beetle + SAF adapters (all three) | −1.3 to −8.7 / +0.3 to +3.7 / −2.9 to −9.1 | trained on them |
| E2B, thinking on | **+11.9 / +16.0 / +9.9** | not run |
| E4B, thinking on | **+6.1 / +12.6** / +3.0 | not run |
| RAG-en, either model | best +2.9 | not run |

- Thinking's p values are ≤ 0.004 across the 10 B1 runs, which stays below 0.05 under Holm over those 10.
- Stage 3b's SciEntsBank gains were described, not tested, in its own family (16 tests fixed before grading). They were tested on 2026-11-08 in a separate family of 30 (every adapter against its untouched model, SciEntsBank only, set after grading), with the E4B and Beetle + SAF rows ([details](finetuning.md#result-transfer-helps-e2b-not-e4b-2026-11-08)).
- Engine and prefix reuse stay within ±0.9. They're cost settings, so the bar doesn't apply to them.
- Against the Oct 28 adapter rather than the untouched model, stage 3b passes in one held-out cell: Beetle test_ua, +5.9 (Holm p 0.031 reason-first, 0.009 label-first).

### Caveats

**1. The next points cost far more than the last ones.** Grading on the Jetson, 612 MHz runs, each step against its own baseline:

| Step | Points gained (test_ua / test_uq / test_ud) | × energy | × median latency | J per extra correct answer |
|---|---|---|---|---|
| E2B → E4B | +11.5 / +7.5 / +9.9 | 1.8× | 1.7× | 28–41 |
| E2B, thinking on | +11.9 / +16.0 / +9.9 | 35–36× | 33–34× | 830–1,340 |
| E4B, thinking on | +6.1 / +12.6 / +3.0 | 39–40× | 37–38× | 2,100–8,900 |

**2. The price is mostly waiting, not power.** Thinking raises E4B's mean draw only from about 9.7 to 10.7 W, but a reply goes from 0.7 s to 27 s. On test_ud's 4,562 answers that's 53 minutes against 32 hours, so a batch that fits a free period becomes an overnight job or longer. Higher clocks don't change this: 612 → 1,020 MHz made E4B about 1.6× faster at about 1.6× the power, and energy per answer rose 4–5%.

**3. Tokens the model writes are the cost driver, and they don't always buy accuracy.** Stage 3b's reason-first adapter writes 63–72 tokens against label-first's 2, for about the same accuracy. RAG makes the prompt 6.3× longer and gave grading no gain (E2B −5.1 on test_ud).

**4. A high accuracy on one dataset can be fit, not skill.** The adapters reach 75–80% on SciEntsBank, and thinking needs ~270 J for the same on test_uq. But they add only +2 to +7 points on Beetle and SAF, one comparison of 8 survives Holm, and E4B gains nothing that holds up. Thinking hasn't been run on Beetle or SAF, so whether its gain transfers better is unknown.

**5. Accuracy hides which errors remain.** On Beetle the adapters gain on "correct" and lose on "contradictory", so macro-F1 stays flat or falls. E2B's adapter also calls 29% (Beetle) to 44% (SAF) of wrong answers correct on train samples, often confidently (median p 0.90 on Beetle). For a student, that's a wrong answer marked right.

**6. The board limits the ways up.** Eight GB of shared memory rules out much larger models: even E4B at Q8_0 (7.6 GiB) likely won't fit (not tested). That leaves training, which moves the cost off the device, or more tokens per answer, which keeps it on.

**Read with care:** the table's accuracy and energy come from single runs at 612 MHz, so thinking isn't yet measured at the stage 4 clocks. The tuned models' accuracy comes from Kaggle, with one seed.

## Limitations

**Scope: what the project doesn't answer**
- **No cloud comparison (RQ5, optional).**
  - The project measures small models on edge devices against each other, not against a cloud model with the same material (the NotebookLM question).
  - So it can say which edge version grades best per joule. It can't say how much accuracy running locally gives up against the cloud.
  - The supervisor asked for this comparison (2026-09-29). It was made optional on 2026-10-19 to keep the focus on the edge.
- **One grading dataset.**
  - SciEntsBank (2013): English school science, scored on three labels.
  - It may be in the models' training data (an open question in the outline).
  - It isn't real classroom data, and there are no Indonesian student answers.
- **Two models from one family, at one quantization:** Gemma 4 E2B / E4B, Q4_K_XL QAT. Other models or precisions may behave differently.
- **One zero-shot prompt.** Prompt design wasn't explored.
- **Fine-tuning: one seed, one training run per model.** A re-run with the same settings moved test_ua by a point.
- **Labels only.** The model gives no feedback a student could use, and fairness (e.g. non-native English writers) isn't studied.

**RAG**
- **Fixed retrieval:** one small English embedder, the question as the query, the top 3 passages, frozen 2026-10-03. Other settings might change the RAG results; B3 tests one other embedder.
- **Uneven course material.** About a quarter of the math chunks lost their numbers in PDF extraction, and many chunks are exercise lists. "RAG doesn't help math" partly reflects that.
- **Retrieval checked lightly:** 30 questions per benchmark, marked by one person.
- **RAG not run on the boards yet.** Its device cost (a prompt ~6× longer) is an estimate, and on-device embedding and retrieval time isn't included.

**Measurement and statistics**
- **News scores rest on an unchecked judge.** Qwen3-14B graded them, with no human check, so they can't be compared directly with the paper's GPT-4.1 figures.
- **Small samples in places.**
  - GSM8K's 100 questions can't detect small changes.
  - GSM-Symbolic's 1,000 answers are about 100 independent templates.
  - MMLU-Pro is a 300-question subset.
- **One correction chosen after seeing results.** Phase 0's gate was passed with the Holm correction, decided after the results were in; the outline records this.
- **Trimmed runs.**
  - The Pi ran 100 of GSM-Symbolic's 1,000 questions, and no test_ud with thinking on.
  - B2 ran on the Jetson's test_ua only.
- **One thinking budget,** 320 tokens, nearly always used up. A larger budget might do differently.
- **Copied MMLU-Pro numbers.** The board MMLU-Pro figures are `agentic-edge`'s, from lm-eval rather than this project's runner. Its time and energy are means, not medians.
- **Later stages use one board.** The Pi 5 was dropped on 2026-10-19 (supervisor), after phase 0 and B1 showed it gives the Jetson's accuracy at 3–9× the time. RAG on the device, phase C and anything later are measured on the Jetson only, so their cost on a CPU-only board is inferred from the Pi's earlier ratios, not measured.
- **Energy isn't wall power.** It's board DC draw. The Pi runs at 3 threads to match the lab.
