# Every SciEntsBank grader on the Jetson (2026-11-14 to 11-20)

*Written 2026-11-16; [thinking on with every adapter](#thinking-on-with-every-e2b-adapter-2026-11-16-to-11-20) added 2026-11-20. RQ8 (three graders) with RQ4's device cost beside it, plus every other E2B method and the E4B adapter. 32 runs, all rc = 0, in `results/board/*-jetson-test_u{q,d}-2026111{4,5,6}-*`. Table and tests: `uv run python -m edge_grader.compare_jetson` (reads every `edge-grader-board-check` JSON, the latest check of a run winning; this page's numbers come from `results/estimates/compare-jetson-scientsbank-20261116-052730.json` and `board-check-20261116-052532.json`).*

## Setup

- **Board:** Jetson Orin Nano, llama.cpp `a894dae`, the lab's flags (`-c 8192 -ngl 99 -rea off --reasoning-budget -1 --cache-ram 0`), plus `--lora` for adapters.
- **Clocks:** GPU max 612 MHz, memory 2,133 MHz on all 32 runs (`board-state.txt`). This is the clock state of phase 0 and B1, **not** the 1,020 MHz of the Oct 21 baselines. The clocks had gone back to the nvpmodel defaults by Nov 14, probably at a reboot. So the two untouched graders were re-run first in the same queue, and every comparison here is within one clock state.
- **Code:** deployed clean, `702ba6f` (Nov 14), `15deed9` (stage 3b, which adds `REPLY`), `a4ee089` (examples, 5-way and adapter tags in run names).
- **Splits:** test_uq (733 answers, 15 questions) and test_ud (4,562 answers, 46 questions), the headline splits for fine-tuned versions (decision 2026-11-13). test_ua isn't run: its questions are in train.
- **Cost:** `edge-grader-board-check`, energy over the grading window only (first request to last), board DC power. Decode tok/s is llama-server's own timing.
- **Adapters:** sha256 recorded in each run's `board-state.txt`. The Oct 28 adapters (`e2b-ft.gguf` `6507078a…`, `e4b-ft.gguf` `53e507cb…`), the stage 3b adapters (`e2b-ft-lf.gguf` `282075cb…`, `e2b-ft-rf.gguf` `190b81da…`), and the Spark-trained seeds, soup and 5-way adapter (copied from the Spark, checked against its hashes).

## The three graders

| Grader | test_uq accuracy (95% range) | test_ud accuracy (95% range) | Macro-F1 uq / ud | Median latency | J per answer | × E2B untouched energy |
|---|---|---|---|---|---|---|
| E2B untouched | 59.8% (50.1–69.2) | 60.2% (54.9–65.0) | 0.548 / 0.570 | 0.40–0.41 s | 3.7–3.8 | 1.00 |
| **E2B tweaked** (Oct 28 adapter) | **76.7%** (68.3–84.9) | **74.8%** (69.5–79.4) | 0.755 / 0.746 | 0.56–0.58 s | 5.3–5.4 | 1.42 |
| E4B untouched | 67.3% (58.5–77.3) | 70.1% (64.9–74.3) | 0.650 / 0.701 | 0.70–0.71 s | 6.8–6.9 | 1.82 |
| *E4B Oct 28 adapter, for reference* | 80.0% (73.0–86.3) | 76.7% (72.0–81.2) | 0.786 / 0.775 | 0.93–0.97 s | 9.3–9.5 | 2.50 |

Family of 8 tests, chosen 2026-11-16 **after** the runs: paired McNemar on 3-way labels, Holm over the 8. The 2-way score is beside it, with its own Holm.

| Comparison | test_uq | test_ud | 2-way uq / ud |
|---|---|---|---|
| E2B tweaked vs E2B untouched | **+16.9** (163 vs 39, p < 10⁻¹⁷) | **+14.6** (912 vs 248, p < 10⁻⁸⁷) | +4.9 / +5.2 (both p < 0.01) |
| **E2B tweaked vs E4B untouched** | **+9.4** (127 vs 58, p = 8.5 × 10⁻⁷) | **+4.7** (584 vs 369, p = 1.3 × 10⁻¹¹) | +8.1 / +4.2 (both p < 10⁻⁴) |
| E4B untouched vs E2B untouched | **+7.5** (140 vs 85, p = 0.0003) | **+9.8** (820 vs 371, p < 10⁻³⁷) | −3.1 / +1.0 (n.s.) |
| E4B adapter vs E4B untouched | **+12.7** (138 vs 45, p < 10⁻¹⁰) | **+6.6** (663 vs 361, p < 10⁻¹⁹) | +10.6 / +5.0 (both p < 10⁻⁷) |

"163 vs 39" means 163 answers only the first-named grader got right, and 39 only the second.

1. **E2B tweaked replaces E4B untouched on the Jetson, on every count.**
   - Accuracy: 4.7–9.4 points higher on both unseen splits, significant after Holm, in 3-way and in 2-way.
   - Speed: about 20% faster per answer (0.57 s against 0.70 s).
   - Energy: about 22% less per answer (5.3–5.4 J against 6.8–6.9 J).
   - So the answer to RQ8 is yes: tweaking the smaller model passes the bigger untouched one, and it is cheaper to run.
2. **The adapter costs E2B 42% more energy per answer**, at the same prompt and the same 2-token reply. llama.cpp serves the adapter beside the weights (`--lora`), so every token pays for the extra matrices: decode drops from 40.7 to 26.9 tok/s, and prefill slows the same way, from 1.4 to 2.1 ms per prompt token. Merging the adapter into the GGUF would likely remove most of this, but it isn't tested; see [open questions](#open-questions).
3. **Energy per extra correct answer:** 9–11 J for E2B's adapter, against 31–41 J for moving from E2B to E4B. Fine-tuning is the cheapest grading gain measured on the Jetson so far. Thinking on cost 830–1,340 J per extra correct answer for E2B ([summary](summary.md#caveats)).
4. **E4B with its own adapter is the most accurate grader** (80.0 / 76.7%), but it costs 1.8× tweaked E2B's energy for +3.3 / +1.9 points. That's 126–210 J per extra correct answer, which isn't tested against tweaked E2B in this family.
5. **Untouched E4B's lead over untouched E2B is only on the 3-way line.** In 2-way (correct against not correct) the two are within 3 points and not significantly different. E4B's gain is in telling "contradictory" from "incorrect".

## Every E2B version

Described, not tested here. Each method's accuracy was tested on the DGX Spark with families fixed before grading ([examples](examples.md#e2b-2026-11-13-dgx-spark), [soup and 5-way](finetuning.md#adapter-soup-and-5-way-labels-e2b-on-scientsbank-2026-11-14-dgx-spark), [stage 3b](finetuning.md#reasoning-guided-fine-tuning-e2b-stage-3b)). The Jetson gives the same label as the Spark on 97.4–99.9% of answers per run, so those tests hold here.

| E2B version | test_uq | test_ud | Median latency | J per answer | × untouched | Prompt / reply tokens |
|---|---|---|---|---|---|---|
| untouched | 59.8% | 60.2% | 0.41 s | 3.7–3.8 | 1.00 | 156–164 / 2 |
| Oct 28 adapter (Kaggle) | 76.7% | 74.8% | 0.57 s | 5.3–5.4 | 1.42 | same / 2 |
| seeds s0–s4 (Spark), range | 75.4–78.0% (mean 76.8 ± 1.0) | 74.4–76.4% (mean 75.2 ± 0.8) | 0.56–0.58 s | 5.3–5.4 | 1.42–1.43 | same / 2 |
| soup of s0–s4 | 78.3% | 75.3% | 0.57–0.60 s | 5.5–5.6 | 1.48 | same / 2 |
| 5-way s0 | 78.2% | 74.7% | 0.69–0.74 s | 6.8–6.9 | 1.81 | 178–186 / 3 |
| label-first (stage 3b) | 76.1% | 73.1% | 0.56–0.59 s | 5.3–5.4 | 1.43 | same / 2 |
| reason-first (stage 3b) | 75.7% | 70.7% | 4.3–4.4 s | 38.8–39.2 | 10.3–10.4 | same / 63–64 |
| + kNN examples | 69.4% | 66.3% | 0.93 s | 8.6–9.1 | 2.26–2.42 | 547–627 / 2 |
| + random examples | 68.1% | 64.9% | 1.07 s | 9.9–10.1 | 2.66 | 594–602 / 2 |
| adapter + kNN examples | 76.3% | 75.3% | 1.33–1.35 s | 13.3–14.1 | 3.49–3.76 | 547–627 / 2 |

1. **Every plain adapter grades within one band, at one price.** The Oct 28 adapter, the five seeds and the soup all land at 75–78% on test_uq and 74–76% on test_ud, for 5.3–5.6 J. Training randomness alone spans about 2–3 points. So which adapter is "E2B tweaked" changes neither the RQ8 answer nor the cost. The Oct 28 adapter sits mid-range on both splits.
2. **The soup costs only 4% more than one adapter.** That holds although it is the five seeds stacked into one rank-80 adapter (a 242 MB file against 48 MB). Its decode is 25.6 against 26.9 tok/s.
3. **The 5-way adapter costs 28% more than a 3-way one for no accuracy.** The prompt lists five labels (+22 tokens), and the 5-way label names take a third token. Its decode is also slower (19.5 tok/s), presumably from the longer context.
4. **A written reason is the most expensive way to say the same thing.** Reason-first writes 63–64 tokens per answer against label-first's 2. It takes 7.3× the energy and 7.6× the time (4.3 s per answer, so 5.5 hours for test_ud), and it is no more accurate. It's 2.4 points lower on test_ud, though that isn't tested here.
5. **Graded examples are the expensive way to tweak without training.** They lift untouched E2B by 6–10 points at 2.3–2.4× the energy (on the Spark the same runs passed the 5-point bar on every split), about 50–87 J per extra correct answer. The adapter gets 15–17 points at 1.42× the energy. On the adapter, examples add nothing and cost 2.5–2.7× its energy. The prompt grows from ~160 to 550–630 tokens, and prefill pays for every token.

## E2B tweaked: every version in one table

*Added 2026-11-16 (owner's request).* Sources:
- **test_uq / test_ud:** the Jetson, Nov 14–16, wherever the version ran there.
- **test_ua:** the DGX Spark's whole-split runs (`results/spark/`). The SciEntsBank adapters trained on its questions, so for them it reads as "seen questions" (\*).
- **Beetle + SAF adapters:** the Spark on all three splits; they didn't run on the Jetson.
- **Cost:** the Jetson's, over the grading window.

Untouched E4B, the bar to beat, scores 67.3% on test_uq and 70.1% on test_ud at 6.8–6.9 J.

**Fine-tuned on SciEntsBank (what "E2B tweaked" means, owner 2026-11-13)**

| Version | test_ua\* | test_uq | test_ud | Macro-F1 uq / ud | J per answer | Median s | Beats E4B untouched? |
|---|---|---|---|---|---|---|---|
| E2B untouched (reference) | 55.2% | 59.8% | 60.2% | 0.548 / 0.570 | 3.7–3.8 | 0.41 | no |
| **Oct 28 adapter** (Kaggle) | 79.4% | **76.7%** | **74.8%** | 0.755 / 0.746 | 5.3–5.4 | 0.57 | **yes** |
| seed s0 (Spark) | 81.5% | 78.0% | 74.4% | 0.773 / 0.747 | 5.3–5.4 | 0.57 | yes |
| seed s1 | 78.9% | 75.4% | 76.4% | 0.734 / 0.765 | 5.3–5.4 | 0.57 | yes |
| seed s2 | 78.7% | 76.9% | 75.4% | 0.758 / 0.756 | 5.3–5.4 | 0.57 | yes |
| seed s3 | 81.5% | 77.3% | 74.4% | 0.767 / 0.746 | 5.3–5.4 | 0.57 | yes |
| seed s4 | 81.1% | 76.4% | 75.5% | 0.752 / 0.757 | 5.3–5.4 | 0.57 | yes |
| soup of s0–s4 | 82.8% | 78.3% | 75.3% | 0.779 / 0.751 | 5.5–5.6 | 0.59 | yes |
| 5-way labels, s0 | 82.4% | 78.2% | 74.7% | 0.770 / 0.743 | 6.8–6.9 | 0.72 | yes |
| label-first (stage 3b) | 76.3% | 76.1% | 73.1% | 0.725 / 0.716 | 5.3–5.4 | 0.58 | yes |
| reason-first (stage 3b) | 72.6% | 75.7% | 70.7% | 0.730 / 0.680 | 38.8–39.2 | 4.3 | yes, barely on test_ud |
| adapter + kNN examples | 77.0% | 76.3% | 75.3% | 0.744 / 0.740 | 13.3–14.1 | 1.34 | yes |

**Fine-tuned on Beetle + SAF only (never saw SciEntsBank; Spark, not run on the Jetson)**

Each is one more adapter of the same size, so expect about 5.4 J per answer (not measured).

| Version | test_ua | test_uq | test_ud | Macro-F1 uq / ud | Beats E4B untouched? |
|---|---|---|---|---|---|
| label-only | 68.5% | 76.7% | 68.2% | 0.699 / 0.610 | test_uq only |
| label-first | 60.2% | 68.6% | 57.6% | 0.634 / 0.527 | no |
| reason-first | 56.7% | 67.4% | 58.5% | 0.615 / 0.517 | no |

**Not fine-tuning, so not "tweaked" by the owner's definition (context)**

| Version | test_ua | test_uq | test_ud | J per answer | Beats E4B untouched? |
|---|---|---|---|---|---|
| + kNN graded examples in the prompt | 70.4% | 69.4% | 66.3% | 8.6–9.1 | test_uq only |
| + random graded examples | 65.0% | 68.1% | 64.9% | 9.9–10.1 | test_uq only |
| thinking on (B1, Jetson 2026-10-11/12) | 67.0% | 75.7% | 70.1% | 136.7–136.9 | test_uq only (ties on test_ud), at 20× E4B's energy |

1. **Every SciEntsBank adapter beats untouched E4B on both headline splits.** They all land in one band (75–78% on test_uq, 73–76% on test_ud) at 5.3–5.6 J, apart from reason-first and the examples. The five seeds alone span 2–3 points, so differences inside the band are training noise. That's why the soup and the 5-way adapter weren't adopted on the Spark: neither is significantly better than a single seed.
2. **Extras only add cost:** 5-way labels +28% energy, examples on the adapter 2.5×, a written reason 7×, for the same or lower accuracy.
3. **Training on other datasets transfers only partly.** The Beetle + SAF label-only adapter matches the SciEntsBank adapters on test_uq (76.7%) but drops to 68% on test_ud. It remains the leak-free check that the gain isn't only memorised questions.
4. **The Oct 28 adapter is a sensible "E2B tweaked"** (the owner's call). It sits mid-band and is the cheapest version tied for best accuracy. The soup (78.3 / 75.3%) and seed s0 (78.0 / 74.4%) have the highest test_uq, but within noise of it.

The thinking row is from the B1 runs on the Jetson at the same 612 MHz clocks, all three splits ([variations](variations.md)); its median latency is 13.7 s per answer. Unlike the other rows, its test_ua is from the Jetson too.

## Thinking on with every E2B adapter (2026-11-16 to 11-20)

*The supervisor's question (2026-11-16): does thinking on help a fine-tuned grader? E2B only, since untouched E4B with thinking is already measured (B1). Twelve runs, all rc = 0: `results/board/*think*-jetson-test_u{q,d}-2026111[6-9]-*` and `-20261120-*`. Table and tests: `uv run python -m edge_grader.compare_thinking`, from `results/estimates/compare-thinking-scientsbank-20261120-142118.json`.*

**Setup.** Thinking on as in B1 (`-rea on --reasoning-budget 320`, the label read after the thought block), deployed code `4d7e261`. Each version runs with the same adapter file as its thinking-off run above (same sha256), and the same reply format or 5-way prompt. Six versions: untouched E2B, the Oct 28 adapter, the soup, label-first, reason-first and 5-way. The three Beetle + SAF adapters aren't run yet (they're on the Spark's network).

**Clocks differ between the runs, so accuracy compares and cost only roughly.**

| | Thinking off (above) | Thinking on, test_uq | Thinking on, test_ud |
|---|---|---|---|
| Jetson power mode | 15W defaults | 15W, GPU limit raised | **25W** (after a reboot on 2026-11-17, owner's choice to keep it) |
| GPU / CPU max | 612 / 1,498 MHz | 1,020 / 1,728 MHz | 918 / 1,344 MHz (5-way: 1,020 / 1,728) |

The board gives the same grades at any clock: untouched E2B with thinking scores 75.7 / 70.1% here, exactly B1's result at 612 MHz in October.

| Version | test_uq: off → on | test_ud: off → on | Answers with a thought block (uq / ud) | On: median s | On: J per answer | Off: J per answer |
|---|---|---|---|---|---|---|
| E2B untouched | 59.8 → **75.7%** | 60.2 → **70.1%** | 733/733 / 4,562/4,562 | 8.6–9.4 | 144–146 | 3.7–3.8 |
| Oct 28 adapter | 76.7 → 80.1% | 74.8 → 74.6% | 733/733 / 4,539/4,562 | 12.2–13.8 | 169–171 | 5.3–5.4 |
| Soup (s0–s4) | 78.3 → 77.2% | 75.3 → 75.2% | **83/733 / 585/4,562** | 0.42–0.44 | 27–29 | 5.5–5.6 |
| Label-first | 76.1 → 78.4% | 73.1 → 71.4% | 726/733 / 4,511/4,562 | 11.7–12.4 | 161–168 | 5.3–5.4 |
| Reason-first | 75.7 → 74.2% | 70.7 → 70.1% | 733/733 / 4,562/4,562 | 7.1–7.9 | 105–113 | 38.8–39.2 |
| 5-way s0 | 78.2 → 77.6% | 74.7 → 73.9% | **0/733 / 4/4,562** | 0.46–0.50 | 7.2–7.5 | 6.8–6.9 |

Macro-F1 moves the same way as accuracy (largest change +0.14 for untouched E2B; −0.03 to +0.03 for the adapters). No run left an answer unlabelled except one each for untouched E2B and reason-first on test_ud. Label-first hit the length limit on 274 (test_uq) and 1,176 (test_ud) answers, but each kept its label line (the 2026-11-16 fix).

**Tests** (22, McNemar on 3-way labels, Holm over all 22, chosen after the test_uq runs):

| Comparison | test_uq | test_ud |
|---|---|---|
| Untouched E2B: thinking on vs off | **+16.0**, Holm p 10⁻¹⁴ | **+9.8**, Holm p 10⁻³⁶ |
| Oct 28 adapter: on vs off | +3.4, Holm p 0.23 | −0.2, 1 |
| Soup: on vs off | −1.1, 1 | −0.1, 1 |
| Label-first: on vs off | +2.3, 1 | −1.7, 0.11 |
| Reason-first: on vs off | −1.5, 1 | −0.5, 1 |
| 5-way: on vs off | −0.5, 1 | −0.8, 0.057 |
| Oct 28 adapter + thinking vs untouched + thinking | +4.4, 0.11 | **+4.5**, 10⁻¹⁰ |
| Soup + thinking vs untouched + thinking | +1.5, 1 | **+5.1**, 10⁻¹³ |
| Label-first + thinking vs untouched + thinking | +2.7, 1 | +1.4, 0.64 |
| Reason-first + thinking vs untouched + thinking | −1.5, 1 | +0.1, 1 |
| 5-way + thinking vs untouched + thinking | +1.9, 1 | **+3.9**, 10⁻⁷ |

1. **Thinking helps only the untouched model** (+16.0 / +9.8 points). With an adapter it changes nothing significant on either split: −1.7 to +3.4 points. The Oct 28 adapter's +3.4 on test_uq (raw p 0.017) doesn't survive the correction and is gone on test_ud (−0.2), the split with six times the answers.
2. **Thinking and fine-tuning buy the same thing, at very different prices.** Untouched E2B with thinking (75.7 / 70.1%) lands at or below the adapters without it (73–78%), for 144–146 J per answer against 5.3–5.6 J, and 9 s against 0.6 s. On test_ud it ties untouched E4B without thinking (70.1%).
3. **Adapters trained on label-only replies can switch thinking off.** The 5-way adapter never opened a thought block (4 of 5,295 answers) and the soup did on 11–13%. Their "thinking on" rows are their thinking-off grading at the same cost. The Oct 28 adapter, trained on the same kind of replies, still thinks on 99.5%. Why the soup and 5-way differ isn't known.
4. **Reason-first thinks and then writes its reason too,** for no gain: 105–113 J, about 2.8× its own thinking-off cost.

## Caveats

- **The test family was chosen after the runs.** Every test passes with a wide margin (largest Holm p = 0.0003), so a different family wouldn't change the conclusion. It is still recorded as post hoc.
- **The 95% ranges are wide on test_uq** (15 questions, resampled by question) and overlap between graders. The paired tests are the stronger evidence, since they compare the same answers.
- **One board, one clock state, one llama.cpp build.** These costs are at 612 MHz. At 1,020 MHz, earlier runs were about 1.6× faster for about the same energy per answer ([summary, caveat 2](summary.md#caveats)).
- **Thinking-on cost is at other clocks** (1,020 MHz on test_uq, 25W / 918 MHz on test_ud) than the thinking-off runs (612 MHz), so its J and seconds per answer are approximate against them. The ratios (30–40× the energy) are far larger than the clock effect (at 1,020 MHz earlier runs were 1.6× faster for about the same energy).
- **Unmerged adapters only.** The adapter overhead (42% for E2B, 37% for E4B) is llama.cpp's `--lora` path. A merged GGUF should cost the device about what the untouched model does. Not measured.
- **Job 56's run folder** (E4B adapter, test_uq) was first fetched while its telemetry was still being written. It was re-synced from the board's finished files, and on 2026-11-16 all 32 folders were checked file by file against the board (288 files, every sha256 identical).

## Open questions

- Does a merged adapter (`llama-export-lora` into the GGUF) remove the 42% overhead? If so, E2B tweaked would cost about what untouched E2B does, about half of untouched E4B.
- Does `--swa-full` (prefix reuse, −42 to −44% energy on the untouched models in B2) cut the adapters' cost the same way?
