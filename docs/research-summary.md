# Research summary: where the project stands

*Written 2026-11-16, updated 2026-11-20 (thinking on with every adapter), in plain words. The full outline, with every decision and its reason, is [`research-outline.md`](research-outline.md). The numbers and their tests are on the [results pages](results/summary.md). Every E2B version in one table: [jetson-graders](results/jetson-graders.md#e2b-tweaked-every-version-in-one-table).*

## The question

**Can a small AI model grade students' short answers well enough, on a cheap device in the school, and what does it cost to run?**

Teachers spend a lot of time grading short answers. A model running on a small local device (here a Jetson Orin Nano, a credit-card-sized computer with a GPU) costs nothing per answer and keeps student data in the school. But small models know less. So the project tries ways of making them grade better, and measures what each way costs on the device: time per answer, power and energy.

Since 2026-11-13 the main question (RQ8, the supervisor's) is a comparison of three graders:

- **E2B untouched:** Google's small Gemma 4 model, as it comes.
- **E2B tweaked:** the same small model, fine-tuned for grading. An *adapter* is a small add-on file, trained on graded examples, that the model loads beside its own weights.
- **E4B untouched:** the bigger Gemma 4 model, as it comes. It is about twice the size, and slower.

*Can tweaking the small model beat the bigger one?*

## The setup in one glance

| | |
|---|---|
| **Task** | Read a science question, the reference answer and a student's answer, then reply *correct*, *contradictory* or *incorrect* |
| **Dataset** | SciEntsBank (SemEval-2013), the most used set in short-answer grading papers. The headline splits are **unseen questions** (test_uq, 733 answers) and **unseen domains** (test_ud, 4,562 answers): questions or whole topics the adapters never trained on |
| **Scores** | Accuracy and F1. The device cost is reported beside them: median seconds per answer and **joules (J) per answer** (energy) |
| **Hardware** | The Jetson for every result in this table (the Raspberry Pi 5 was dropped on 2026-10-19). Kaggle GPUs and the lab's DGX Spark for training and accuracy checks; both give the same grades as the Jetson |
| **Rule for a gain** | A change counts if it adds at least 5 points of accuracy and the gain is statistically significant (paired test, corrected for running many tests) |

## The headline result (RQ8)

Jetson, 2026-11-14 to 11-16, all at the same clock speed:

| Grader | Unseen questions | Unseen domains | Seconds per answer | Energy per answer |
|---|---|---|---|---|
| E2B untouched | 59.8% | 60.2% | 0.41 | 3.8 J |
| **E2B tweaked** | **76.7%** | **74.8%** | **0.57** | **5.4 J** |
| E4B untouched | 67.3% | 70.1% | 0.70 | 6.9 J |

**Yes: the tweaked small model beats the bigger untouched one.** It is 9.4 and 4.7 points more accurate (both significant), about 20% faster and about 22% cheaper in energy. Fine-tuning costs the small model 42% more energy than untouched, which works out to about 10 J per extra correct answer. That's the cheapest accuracy gain measured in the project.

## Each research question, in short

| # | Question | Answer so far | Status |
|---|---|---|---|
| RQ1 | How well do the untouched models grade? | Moderately. E2B is at 55–60% and E4B at 67–70% on the three splits. They are weakest at recognising *correct* answers. The bigger model is 10–14 points better at everything it can know (grading, MMLU-Pro, math). The Jetson and Pi give the same accuracy as a cloud GPU | Done |
| RQ2 | Does giving the model course material (RAG) help? | Only where the task lacks facts the material supplies: news after the model's training cutoff went from 0.5% to 39–45%. For grading it doesn't help, and it costs the small model 5 points on unseen domains. It also didn't help math or MMLU-Pro | Done (MMLU-Pro with RAG on the Jetson still to run) |
| RQ3 | Does fine-tuning help? | **Yes, it's the biggest gain:** +6 to +24 points, with no loss on general knowledge or math. Gains on other grading datasets (Beetle, SAF) are small (+2 to +7). None of the extras beat a plain adapter: reasons in the reply, examples in the prompt, averaging several adapters ("soup"), finer labels | Done |
| RQ4 | What does each gain cost on the Jetson? | From cheapest to dearest per answer: E2B (≈4 J), E2B tweaked (≈5 J), E4B (≈7 J), E4B tweaked (≈9 J), examples in the prompt (9–14 J), a written reason before the label (≈39 J), thinking on (140–280 J). Thinking adds 3–19 points but makes each answer 35–40× dearer: test_ud's 4,562 answers would take about 32 hours instead of under an hour | Done for every version so far |
| RQ5 | How close is the best local version to a cloud model? | Not measured: optional since 2026-10-19, and stated as a limitation | Optional |
| RQ6 | Do settings (thinking, engine, prefix reuse) matter? | Thinking is the only setting that changes grading (+3 to +19 points) for the untouched models, and it is very expensive. **On a fine-tuned E2B it adds nothing** (−1.7 to +3.4 points, none significant; 2026-11-20). The engine and prefix reuse don't change accuracy, but they cut energy by 21–50% | Done (B3 embedder optional) |
| RQ7 | Can the model grade an answer given as an image? | Not started; optional since 2026-11-04 | Optional |
| **RQ8** | **Can tweaked E2B replace untouched E4B?** | **Yes, on accuracy and on cost** (above). Every plain E2B adapter (the Oct 28 one, five seeds, the soup) lands in the same 2–3-point band at the same cost, so the choice of adapter doesn't change the answer | Done; you pick which adapter is "the" E2B tweaked |

## What it adds up to

1. **Training the small model is the best way to make it grade well, and it stays cheap on the device.** Anything that makes the model read or write more costs far more energy for the same or less accuracy: thinking, written reasons, examples in the prompt, course material.
2. **On the Jetson, the deployable grader is E2B with its adapter:** about 75% on unseen questions and domains, about half a second and 5 J per answer.
3. **Accuracy beyond that is expensive.** E4B with its own adapter gains 2–3 more points for 1.8× the energy. Thinking on top of the E2B adapter gains nothing measurable (below), for 30–40× the energy and hours of waiting.

## Thinking on with the adapters (supervisor's question, 2026-11-16)

Each E2B version graded again with thinking on, on the Jetson, 2026-11-16 to 11-20 ([details](results/jetson-graders.md#thinking-on-with-every-e2b-adapter-2026-11-16-to-11-20)):

| Version | Unseen questions: off → on | Unseen domains: off → on | Energy per answer, on |
|---|---|---|---|
| E2B untouched | 59.8 → **75.7%** | 60.2 → **70.1%** | ≈145 J |
| Oct 28 adapter | 76.7 → 80.1% | 74.8 → 74.6% | ≈170 J |
| Soup | 78.3 → 77.2% | 75.3 → 75.2% | ≈28 J (mostly doesn't think) |
| Label-first | 76.1 → 78.4% | 73.1 → 71.4% | ≈165 J |
| Reason-first | 75.7 → 74.2% | 70.7 → 70.1% | ≈110 J |
| 5-way | 78.2 → 77.6% | 74.7 → 73.9% | ≈7 J (never thinks) |

**Thinking helps the untouched model** (+16 and +10 points, significant) **but not a tweaked one** (no adapter changes significantly). It buys roughly what fine-tuning buys, at 30–40× the energy and 9–14 s instead of half a second per answer. Two adapters (soup and 5-way) mostly or entirely skip the thinking step even when it is switched on. The thinking runs ran at faster clock settings than the thinking-off ones, so their energy is approximate; accuracy isn't affected.

## What's left

- **Your decision:** which adapter to name "E2B tweaked" in the write-up. The Oct 28 Kaggle adapter sits mid-range of the five seeds on both splits.
- **Still to run on the Jetson:** MMLU-Pro with course material (RAG-en).
- **Optional:** the Beetle + SAF adapters, and E4B's adapters, with thinking on.
- **Optional:** a merged adapter, folded into the model file. It might remove most of the adapter's 42% energy overhead. Also the adapter with prefix reuse (`--swa-full`).
- **Weeks 10–12:** figures (accuracy against energy per answer), the report, the supervisor's review.

## Questions still open with the supervisor

- Is SciEntsBank acceptable as the main benchmark, given their point that it is more a dataset than a benchmark, and its age?
- Was SciEntsBank in the models' training data? It can't be tested directly. Adapters trained only on other datasets (Beetle and SAF) still help E2B on SciEntsBank, which is some reassurance.
- The robustness section: dropped or not?
- They asked for the cloud comparison (RQ5) on 2026-09-29. It was made optional on 2026-10-19, and they should confirm that.

## Limitations to state in the report

- One grading dataset (English school science, from 2013), and two models from one family at one compression level.
- One prompt wording, and labels only: the model gives no feedback a student could use.
- Device cost is from one board at one clock speed. Energy is the board's own draw, not wall power.
