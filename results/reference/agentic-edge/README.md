# agentic-edge benchmark results (copied)

*Source: `agentic-edge` repo, commit `12031f0` (2026-09-28), clean working tree. Copied 2026-09-29, not re-run. Don't edit these files; re-copy from a newer commit if they change.*

The supervisor's call (2026-09-29): **reuse these device benchmarks rather than re-measure them**, to save time. The summary for this project is in [`docs/results/edge-benchmarks.md`](../../../docs/results/edge-benchmarks.md).

| Here | From `agentic-edge` | What |
|---|---|---|
| `docs/RESULTS.md` | `findings/RESULTS.md` | Every number measured, with protocol and caveats. **Start here.** |
| `docs/summary.md` | `summary.md` | Project status and serving config |
| `docs/EXPERIMENT_PLAN.md` | `benchmark/EXPERIMENT_PLAN.md` | Protocol and statistical design |
| `docs/gemma4-pi5-benchmarks.md` | `gemma4-pi5-benchmarks.md` | Sep-14 engine comparison, little-gemma vs llama.cpp (predates the current serving flags) |
| `baselines.json` | `benchmark/baselines.json` | The pinned serving configs (`-rea off`, thinking-on, …) |
| `export/experiment-full.json.gz` | `findings/export/` | MMLU-Pro s1/s2/s3 on both boards: every request, 1 Hz telemetry, energy, answer text. Shape in `export/README.md` |
| `lm-eval/` | `findings/stdbench/` | lm-eval `results_*.json` summaries (MMLU-Pro subset 100, tinyGSM8k) and the subset question ids. Per-sample `.jsonl` files left out (they are in the export) |
| `own-suite/` | `benchmark/results/`, `findings/e_classifier_on_bfcl.json` | Own 10-case suite: architectures A–E, MTP × thinking, E4B quantization sweep (Pi 5) |
| `telemetry-sample/` | `findings/measured/` | One `run_measured.sh` output folder, as a format example |
| `early-engine-benchmarks/` | `findings/early-engine-benchmarks/` | Raw Sep-14 scratch output behind `gemma4-pi5-benchmarks.md` |
| `mmlupro-correct-per-question.json` | the dashboard's `/api/compare` (added 2026-10-05, not from commit `12031f0`) | MMLU-Pro baseline, both boards, s1–s3: right or wrong per `question_id`. For pairing `edge-grader-mmlupro` runs with the lab's; its accuracies match the lab's (Pi 51.7 / 65.7%, Jetson 52.7 / 66.0%) |
| `lm-eval/tinygsm8k-lab.json` | `findings/stdbench/{B-e2b,tiny1024-e2b,B-e4b}/samples_tinyGSM8k_*.jsonl` (added 2026-10-07) | tinyGSM8k: the chat messages lm-eval sent per question, and each run's reply and strict/flexible result. For checking `edge-grader-gsm` against the lab (E2B 81/100 at 1,024 tokens, E4B 75/100 at 256) |

Not copied: `findings/viz/` (HTML views, regenerable), the per-sample lm-eval `.jsonl` files, and archived failed runs.
