# Phase B: variations on the baseline (RQ6)

*Started 2026-10-11. Spec: [`../superpowers/specs/2026-10-08-baseline-variations-design.md`](../superpowers/specs/2026-10-08-baseline-variations-design.md) §6. Edge first (owner, 2026-10-11): every grading run here is on a board and is compared with **the same board's** thinking-off llama.cpp baseline ([`baselines.md`](baselines.md) §2, §4). The MMLU-Pro side of each variation is **copied from `agentic-edge`**, not re-run, as the spec says: same GGUFs, flags and boards. Copied numbers: [`agentic-edge-mmlupro-variations.json`](../../results/reference/agentic-edge-mmlupro-variations.json) (commit `16eace9`, sections 7.2, 7.4, 7.6, 7.8).*

**Status (2026-10-19):** B1 done on both boards. B2 done on the Jetson, test_ua, thinking off. B3 (EmbeddingGemma 2) not started.

**Check files:** [`board-check-20261016-083521.json`](../../results/estimates/board-check-20261016-083521.json) (Jetson B1 test_ud, B2) and [`board-check-20261019-102403.json`](../../results/estimates/board-check-20261019-102403.json) (B1 test_ua, test_uq, both boards). Paired McNemar against the baseline: `board_check.compare` on the two runs' raw files.

**Jetson clocks:** every B1 and B2 run's `board-state.txt` shows them unlocked (GPU devfreq 306–612 MHz), as do the phase 0 baselines (test_uq, test_ud). The 2026-09-30 test_ua baselines predate `board-state.txt`; they count as unlocked because phase 0 restored the Jetson's clocks with a reboot to make its runs comparable with them ([`baselines.md`](baselines.md) §4). So energy compares across the runs.

## B1. Thinking on

**Setup:**
- Flags: `-rea on --reasoning-budget 320 --reasoning-format none`. Everything else as the baseline.
- `max_tokens` 336. The label is read only after the thought block.
- Runs: Jetson all three splits; Pi 5 test_ua + test_uq (test_ud skipped: ~4 days on the Pi), E2B + E4B, in `results/board/think-*`.

### Grading (SciEntsBank, measured)

| Model | Board | Split | n | Baseline → thinking | Better / worse | McNemar p | Median s | J / answer | Energy × |
|---|---|---|---|---|---|---|---|---|---|
| E2B | Jetson | test_ua | 540 | 55.2% → **67.0%** (+11.9) | 115 / 51 | < 0.001 | 0.42 → 13.8 | 3.95 → 139.1 | 35× |
| E2B | Jetson | test_uq | 733 | 59.8% → **75.7%** (+16.0) | 167 / 50 | < 0.001 | 0.41 → 13.7 | 3.80 → 136.9 | 36× |
| E2B | Jetson | test_ud | 4,562 | 60.2% → **70.1%** (+9.9) | 852 / 403 | < 0.001 | 0.40 → 13.7 | 3.76 → 136.7 | 36× |
| E2B | Pi 5 | test_ua | 540 | 55.7% → **66.7%** (+10.9) | 106 / 47 | < 0.001 | 3.22 → 43.6 | 27.8 → 302.7 | 11× |
| E2B | Pi 5 | test_uq | 733 | 59.8% → **78.6%** (+18.8) | 176 / 38 | < 0.001 | 3.19 → 41.5 | 26.1 → 289.8 | 11× |
| E4B | Jetson | test_ua | 540 | 66.7% → **72.8%** (+6.1) | 67 / 34 | 0.001 | 0.72 → 27.0 | 7.12 → 280.0 | 39× |
| E4B | Jetson | test_uq | 733 | 67.3% → **79.8%** (+12.6) | 128 / 36 | < 0.001 | 0.71 → 26.9 | 6.90 → 269.8 | 39× |
| E4B | Jetson | test_ud | 4,562 | 70.1% → **73.1%** (+3.0) | 416 / 279 | < 0.001 | 0.70 → 26.8 | 6.81 → 273.6 | 40× |
| E4B | Pi 5 | test_ua | 540 | 66.5% → **72.0%** (+5.6) | 65 / 35 | 0.004 | 6.33 → 87.1 | 55.7 → 592.7 | 11× |
| E4B | Pi 5 | test_uq | 733 | 68.2% → **79.1%** (+10.9) | 114 / 34 | < 0.001 | 6.36 → 88.3 | 52.4 → 584.2 | 11× |

- **Unparsed:** 1 answer each on the two test_ud runs, 0 elsewhere.
- **The budget binds:** ~320 generated tokens per answer (median 325 of 336), against 2–2.5 without thinking.

Recall per class on the Jetson, baseline → thinking:

| Model | Split | correct | contradictory | incorrect | "contradictory" given |
|---|---|---|---|---|---|
| E2B | test_ua | 0.45 → 0.56 | 0.71 → 0.52 | 0.61 → 0.81 | 157 → 50 |
| E2B | test_uq | 0.49 → 0.71 | 0.72 → 0.45 | 0.67 → 0.85 | 193 → 48 |
| E2B | test_ud | 0.61 → 0.68 | 0.79 → 0.53 | 0.56 → 0.75 | 1,100 → 308 |
| E4B | test_ua | 0.45 → 0.61 | 0.62 → 0.71 | 0.88 → 0.84 | 49 → 53 |
| E4B | test_uq | 0.40 → 0.70 | 0.67 → 0.66 | 0.90 → 0.90 | 64 → 50 |
| E4B | test_ud | 0.57 → 0.66 | 0.68 → 0.71 | 0.82 → 0.79 | 374 → 332 |

Pi against Jetson, both with thinking on:

| Split | Pi / Jetson time | Pi / Jetson energy | Same label | McNemar p |
|---|---|---|---|---|
| test_ua | 3.2× (E2B), 3.2× (E4B) | 2.2×, 2.1× | 466/540, 477/540 | 0.90, 0.70 |
| test_uq | 3.0×, 3.3× | 2.1×, 2.2× | 648/733, 682/733 | **0.024**, 0.58 |

### MMLU-Pro (copied from `agentic-edge` §7.2)

Same thinking flags and budget, n = 300 per cell:

| Model | Board | Baseline → thinking | Base only / think only | McNemar p | Energy per 300 questions |
|---|---|---|---|---|---|
| E2B | Pi 5 | 51.7% → 50.0% (−1.7) | 34 / 29 | 0.61 | 64.2 → 79.8 Wh (+24%) |
| E2B | Jetson | 52.7% → 49.7% (−3.0) | 32 / 23 | 0.28 | 25.6 → 32.6 Wh (+27%) |
| E4B | Pi 5 | 65.7% → 66.3% (+0.6) | 18 / 20 | 0.87 | 125.5 → 151.2 Wh (+20%) |
| E4B | Jetson | 66.0% → 66.7% (+0.7) | 16 / 18 | 0.86 | 54.5 → 67.5 Wh (+24%) |

On a second engine (little-gemma, Jetson, §7.4), thinking again changes nothing significant: E2B −3.7 (p 0.18), E4B −1.6 (p 0.49). E2B's loss is mostly format: answers with no `answer is (X)` line rise from 22–25 to 39–41.

### What B1 shows

1. **Thinking improves grading, on every split, model and board.** It adds +3.0 to +18.8 points, all ten at p ≤ 0.004. That is the largest accuracy change in the project so far. RAG moved grading by −5.1 to +2.9.
2. **On MMLU-Pro the same flags and budget do nothing.** The change there is −3.0 to +0.7 points, none significant. The difference is in what thinking adds:
   - A grading reply is a bare label, so without thinking the model doesn't reason at all; the thought block is its only chance to.
   - MMLU-Pro's prompt already asks for a 5-shot chain of thought, so thinking puts a capped preamble in front of reasoning the model does anyway.
3. **Thinking makes grading judgement better, not just more lenient.** That is not what RAG did.
   - RAG raised recall on *correct* by giving up recall on *incorrect* ([`rag.md`](rag.md#grading-results-rag-en-kaggle-t4-2026-10-05)).
   - Thinking raises *correct* for both models (E4B test_uq 0.40 → 0.70) and keeps *incorrect* (E4B 0.79–0.90 either way).
   - For E2B it also cures the overuse of "contradictory": 1,100 → 308 on test_ud. E2B's *incorrect* recall rises (0.56 → 0.75) at the cost of *contradictory* recall (0.79 → 0.53).
4. **The gain is smallest on unseen domains** (test_ud: E2B +9.9, E4B +3.0), and largest on unseen questions (test_uq: +12.6 to +18.8).
5. **It costs 11× the energy on the Pi and 35–40× on the Jetson.**
   - Thinking turns a prefill-bound job (a 2-token label) into a decode-bound one (~320 tokens).
   - The Jetson's lead over the Pi shrinks from 6.6–9.0× to 2.1–2.2× in energy and 3.0–3.3× in time, the same as for GSM, news and MMLU-Pro (§4 of [`baselines.md`](baselines.md)).
   - Per answer, thinking takes 14–27 s on the Jetson and 42–88 s on the Pi.
6. **E4B without thinking is the cheap middle ground.**
   - E2B with thinking on the Jetson (67.0 / 75.7 / 70.1% on test_ua / uq / ud) ties E4B without it (66.7 / 67.3 / 70.1%) except on test_uq, at 20× E4B's energy (137 J against 6.9 J).
   - The most accurate version is E4B with thinking (72.8 / 79.8 / 73.1% on the Jetson), at ~275 J per answer.
7. **Thinking loosens board agreement.**
   - The Pi and the Jetson give the same label on 86–93% of answers with thinking, against 96–99% without.
   - One of the four Pi-vs-Jetson pairs has p = 0.024: E2B test_uq, Pi 78.6% against the Jetson's 75.7%. Holm-corrected over the four, it's 0.096.
   - A long generation gives CPU and CUDA arithmetic more chances to split at a near-tie, as on MMLU-Pro.

## B2. Engine (little-gemma) and prefix reuse

**Setup:**
- Jetson, test_ua, E2B + E4B, thinking off, the same GGUFs and prompts.
- Variants:
  - `lg`: little-gemma (`little-gemma-ae` build, through `lg_openai_shim.py`)
  - `lg-reuse`: the same with `-reuse`
  - `swafull`: llama.cpp with `--swa-full`, the llama.cpp reuse control, chosen in the 2026-10-13 pilot (outline, own decisions)
- The spec's `llama-sorted` isn't run separately; the pilot showed `--swa-full` is what restores llama.cpp's reuse.
- **No Pi runs:** little-gemma prefills 41–43× slower than llama.cpp on the Pi's CPU (`agentic-edge` §7.4).

### Grading (SciEntsBank test_ua, Jetson, measured)

| Variant | Model | Accuracy (baseline → variant) | Better / worse | McNemar p | Same label | Median s | J / answer | Energy vs baseline |
|---|---|---|---|---|---|---|---|---|
| llama.cpp (baseline) | E2B | 55.2% | | | | 0.42 | 3.95 | |
| `lg` | E2B | 55.6% (+0.4) | 13 / 11 | 0.84 | 514/540 | 0.18 | 1.96 | **−50%** |
| `lg-reuse` | E2B | 55.6% (+0.4) | 13 / 11 | 0.84 | 514/540 | 0.18 | 1.96 | −50% |
| `swafull` | E2B | 56.1% (+0.9) | 13 / 8 | 0.38 | 517/540 | 0.22 | 2.30 | −42% |
| llama.cpp (baseline) | E4B | 66.7% | | | | 0.72 | 7.12 | |
| `lg` | E4B | 66.5% (−0.2) | 1 / 2 | 1.0 | 536/540 | 0.44 | 5.63 | −21% |
| `lg-reuse` | E4B | 66.5% (−0.2) | 1 / 2 | 1.0 | 536/540 | 0.44 | 5.13 | −28% |
| `swafull` | E4B | 67.0% (+0.4) | 3 / 1 | 0.63 | 536/540 | 0.37 | 4.00 | **−44%** |

- **Byte-identical check:** `lg-reuse`'s replies match `lg`'s on 540/540 for both models.
- **Prompt tokens processed:** `swafull` 82 per answer, against 165 for the baseline.

### MMLU-Pro (copied from `agentic-edge` §7.4, §7.6, §7.8)

Jetson, n = 300, thinking off:

| Variant | E2B accuracy | E2B energy, time vs llama.cpp | E4B accuracy | E4B energy, time vs llama.cpp |
|---|---|---|---|---|
| llama.cpp (S1) | 52.7% | 25.6 Wh, 155 min | 66.0% | 54.5 Wh, 313 min |
| little-gemma (S3) | 53.0% (p 1.00) | −4%, −9% | 65.3% (p 0.85) | −13%, −17% |
| little-gemma `-reuse` (S9) | = S3, byte-identical | −4% time against S3 | = S3, byte-identical | −8% time against S3 |
| little-gemma `-reuse -ngram` (S9 full) | = S3, byte-identical | −21%, −22% | = S3, byte-identical | −31%, −34% |
| llama.cpp n-gram (S10 `n3m3`) | 53.3% (p 0.86) | −6%, −10% | 65.0% (p 0.69) | −7%, −13% |

- **Clocks:** S9 `reuse` and S10 ran with the Jetson's clocks locked, which adds 5–9% energy at the same speed. Read their time column.
- **llama.cpp's reuse:** on MMLU-Pro it already reuses the 5-shot prefix through its own prompt cache (~38k prompt tokens per subset, the same as S9 `-reuse`).

### What B2 shows

1. **Neither the engine nor prefix reuse changes grading accuracy.**
   - All six comparisons are within noise (p ≥ 0.38), with the same label on 95–99% of answers.
   - That matches MMLU-Pro, where little-gemma tied llama.cpp too.
   - So the engine is a cost choice, not an accuracy one.
2. **On grading, both save far more than on MMLU-Pro,** because grading is almost all prefill:
   - little-gemma: −50% energy for E2B, −21% for E4B, against −4% and −13% on MMLU-Pro.
   - The split between the models follows little-gemma's prefill lead on the Jetson (`agentic-edge` §7.4): ~3.5× llama.cpp's speed for E2B (1,690 against 477 tok/s), ~1.3× for E4B (352 against 275).
3. **llama.cpp catches up once its cache works.**
   - `--swa-full` halves the prompt tokens processed and saves 42–44% energy on both models.
   - For E4B that is cheaper than little-gemma (4.0 J against 5.6 J per answer).
   - On MMLU-Pro, reuse was worth only a few percent. With long answers the Jetson's prefill was never the bottleneck.
4. **little-gemma's `-reuse` does nothing on grading prompts.**
   - It snapshots only what comes before the final user turn, and a grading prompt is one user turn (2026-10-13 pilot).
   - The replies are byte-identical to `lg`, and so is E2B's energy.
   - E4B's 0.5 J gap (5.13 against 5.63) comes with identical replies and median latency. It is run-to-run spread, not a saving.
5. **For deployment on the Jetson:**
   - llama.cpp `--swa-full` and little-gemma are the cheapest ways to serve the baseline. Both are accuracy-neutral and cut energy per answer roughly in half for E2B.
   - With thinking on, the job becomes decode-bound. On MMLU-Pro, only n-gram drafting moved the decode cost (−6% to −31%). That hasn't been measured for grading.

## Cascade and defer-to-teacher (simulated from B1, 2026-11-05)

*Set aside the same day by the supervisor: the project tweaks one model configuration at a time, so it doesn't run two models together. This result is kept as recorded, and it isn't developed further or run on the board.*

No new board runs. `uv run python -m edge_grader.cascade` re-reads the Jetson runs of the 612 MHz clock state: the baselines (stage 1 / phase 0) and B1 thinking on, both models, all three splits. Each answer was graded by every version, so a policy only picks which label to keep. Policies:

- **cascade:** grade with E2B and E4B. Where they agree, keep the label. Where they disagree, take the thinking-on label (E2B's or E4B's). The cost is both baselines on every answer, plus thinking on the disagreements.
- **defer:** the same check, but disagreements go back to the teacher.

Agreement has no threshold, so nothing is fitted on the splits reported.

Energy per answer is the run's measured energy, apportioned to answers by latency. Cascade accuracy is paired with plain E4B (McNemar, Holm over the six cascade tests). Raw: [`results/cascade/`](../../results/cascade/).

| Split | E4B | E4B thinking | Cascade → E2B thinking | Cascade → E4B thinking | Defer: coverage, accuracy on kept |
|---|---|---|---|---|---|
| test_ua | 66.7%, 7.1 J | 72.8%, 280 J | 68.9%, 60 J (p 0.21) | 70.9%, 109 J (**p 0.004**) | 65%, 71.9% at 11 J |
| test_uq | 67.3%, 6.9 J | 79.8%, 270 J | 75.0%, 61 J (**p < 0.001**) | 75.8%, 109 J (**p < 0.001**) | 63%, 76.1% at 11 J |
| test_ud | 70.1%, 6.8 J | 73.1%, 274 J | 71.7%, 53 J (**p 0.003**) | 72.5%, 95 J (**p < 0.001**) | 69%, 75.0% at 11 J |

1. **The cascades fill the gap between 7 J and 270 J.**
   - E2B and E4B disagree on 31–37% of answers, so only those are escalated.
   - Escalating to E4B thinking keeps 92–99% of thinking's gain over E4B on test_ua and test_ud, at about 35% of its energy. On test_uq it keeps 68% of the gain.
   - Escalating to E2B thinking costs a fifth of always-on E4B thinking. It loses 1.4–4.8 points against E4B thinking.
   - Every cascade lies on the accuracy–energy front. Always-on E2B thinking lies on it on none of the three splits.
2. **Agreement is a usable confidence signal.**
   - On the 63–69% of answers where the two baselines agree, E4B's accuracy is 72–76%. That is close to E4B thinking's accuracy on all answers, at 11 J instead of 270 J.
   - The other third goes to the teacher. Report this as coverage and accuracy on kept answers, beside plain accuracy.
3. **Caveats:**
   - The cascade's latency is the sum of its stages: a mean of 5–10 s per answer, against 25 s for E4B thinking.
   - Apportioning energy by latency assumes power is flat within a run.
   - test_ua's baselines are from 2026-09-30 (stage 1). Their server flags are the same, but the run has no `board-state.txt`.
   - These are untuned models. The Oct 28 adapters reach 75–80% on Kaggle at about baseline token cost, and if their Jetson cost is near the baseline's they beat every cascade here. That cost is still to be measured (stage 4). The same agreement cascade can run on the adapters once their Jetson runs exist.
