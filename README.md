# edge-grader

Can a small LLM on an edge board grade students' short answers, and is a fine-tuned small model a better grader than a bigger untouched one?

Three graders are compared on SciEntsBank (SemEval-2013 Task 7, 3-way labels) on a Jetson Orin Nano: **Gemma 4 E2B untouched**, **E2B with a QLoRA adapter**, and **Gemma 4 E4B untouched**. Each is scored for accuracy and F1, and for latency and energy per answer on the board. Short answer so far: the E2B adapter beats untouched E4B on unseen questions and unseen domains, and it uses less energy per answer ([`docs/results/jetson-graders.md`](docs/results/jetson-graders.md)).

Part of an internship at the NTUST MIT Lab. Model files, `llama-server` flags and the energy telemetry match the lab's `agentic-edge` benchmark, so the numbers are comparable with it. Research questions and every decision with its reason: [`docs/research-outline.md`](docs/research-outline.md). All findings by question: [`docs/results/summary.md`](docs/results/summary.md).

## What you need

| For | Needs |
|---|---|
| Re-checking the numbers from committed results | Any machine with [uv](https://docs.astral.sh/uv/). No GPU, no models |
| Grading accuracy | A GPU that runs `llama-server` (we used Kaggle T4s and a DGX Spark) |
| Training an adapter | A CUDA GPU with ~16 GB (Kaggle T4, or the Spark through Docker) |
| Latency and energy | A Jetson Orin Nano (8 GB) with llama.cpp built, plus `agentic-edge`'s `run_measured.sh` for telemetry |

```bash
uv sync
uv run pytest
```

## Replicate

### 0. Re-check the reported numbers (no GPU)

Every run is committed under `results/`, including per-answer replies. These commands rebuild the tables and tests from them:

```bash
uv run python -m edge_grader.compare_jetson      # three graders + every E2B method on the Jetson: accuracy, J/answer, McNemar + Holm
uv run python -m edge_grader.compare_thinking    # thinking on with every E2B adapter
uv run --with matplotlib python scripts/paper_figures.py   # manuscript figures, into docs/results/figures/
```

`compare_jetson` reads the board-check JSONs in `results/estimates/`. To rebuild those from the raw board runs first:

```bash
uv run edge-grader-board-check results/board/*-jetson-test_u[qd]-2026111[456]-*/
```

### 1. Models and server

- **Models:** `unsloth/gemma-4-E2B-it-qat-GGUF` and `unsloth/gemma-4-E4B-it-qat-GGUF`, file `*-UD-Q4_K_XL.gguf`.
- **Server:** llama.cpp's `llama-server`. Kaggle and the Spark use commit `a3a1c47` (pinned in [`docker/Dockerfile`](docker/Dockerfile)). Each run's summary JSON records the build it actually used.
- **Flags** (thinking off, as in `agentic-edge`):

```bash
llama-server -m <model.gguf> -c 8192 --port 8080 -ngl 99 -rea off --reasoning-budget -1 --cache-ram 0 [--lora <adapter.gguf>]
```

### 2. Train the adapter (stage 3)

QLoRA on the SciEntsBank train split; the epoch is picked on 14 held-out train questions. Run either one:

- **Kaggle:** [`notebooks/kaggle_finetune.ipynb`](notebooks/kaggle_finetune.ipynb). Set `MODEL`, then `SMOKE = True` for a quick check and `SMOKE = False` for the real run. It trains, converts the adapter to GGUF and grades with `--lora`. First time on Kaggle: [`docs/guides/kaggle-setup.md`](docs/guides/kaggle-setup.md).
- **Docker (DGX Spark or any CUDA box):** `docker compose run --rm finetune-…` services in [`compose.yaml`](compose.yaml); see [`docs/guides/spark-docker.md`](docs/guides/spark-docker.md).

Both call `uv run --group finetune edge-grader-finetune --model e2b --out <dir>`. Settings and per-epoch scores: [`docs/results/finetuning.md`](docs/results/finetuning.md). Our adapter files are in a private Kaggle dataset, not in git; their sha256 is in that page.

### 3. Grade for accuracy (GPU)

With a server running from step 1:

```bash
uv run edge-grader-baseline --label e2b --split test_uq --n 0   # whole unseen-questions split (733)
uv run edge-grader-baseline --label e2b --split test_ud --n 0   # whole unseen-domains split (4,562)
```

Start an adapter server with `--lora` for the tweaked grader. Each run writes a summary JSON (metrics and settings) and `raw/*.jsonl` (one reply per answer). `--n 100` is a quick check: `unparsed` should be about 0. Kaggle does the same in [`notebooks/kaggle_baseline.ipynb`](notebooks/kaggle_baseline.ipynb).

### 4. Measure cost on the Jetson

Clone this repo next to `agentic-edge` on the board (or push it with `scripts/deploy.sh jetson` from your machine), then on the idle board:

```bash
scripts/board_grade.sh e2b test_uq                                          # E2B untouched
LORA=$HOME/research/models/adapters/e2b-ft.gguf scripts/board_grade.sh e2b test_uq   # E2B tweaked
scripts/board_grade.sh e4b test_uq                                          # E4B untouched
```

Repeat for `test_ud`. The script starts `llama-server` with the flags above, grades inside `run_measured.sh` so power is logged, records the clocks in `board-state.txt`, and restores the board when it exits. Without `agentic-edge`, `MEASURE=0` gives accuracy only. `DRY_RUN=1` prints what it would do. Other options (thinking, RAG, graded examples, 5-way adapters) are listed at the top of [`scripts/board_grade.sh`](scripts/board_grade.sh).

- **Keep the clock state fixed.** Compare energy only between runs at the same GPU clock and power mode. Ours: GPU 612 MHz, memory 2,133 MHz for the Nov 14–16 runs (`board-state.txt` in each run).
- **Run it directly on the board, never through `agentic-edge`'s job queue.** The script refuses if started from the queue, and it refuses to write telemetry outside this repo's `results/`.

Bring the runs back with `scripts/fetch_results.sh`, then go back to step 0.

## Other experiments

These are in the repo with their results. Each page says how its runs were made:

| | Page |
|---|---|
| Baselines on Kaggle, Pi 5 and Jetson; MMLU-Pro | [`baselines.md`](docs/results/baselines.md) |
| RAG over course material (no gain for grading) | [`rag.md`](docs/results/rag.md), how-to in [`guides/rag.md`](docs/guides/rag.md) |
| Math (GSM8K, GSM-Symbolic) and post-cutoff news | [`topics.md`](docs/results/topics.md) |
| Thinking on, engine and prefix-reuse variations | [`variations.md`](docs/results/variations.md) |
| Graded examples in the prompt; adapter soup; 5-way labels | [`examples.md`](docs/results/examples.md), [`finetuning.md`](docs/results/finetuning.md) |
| ASAP 2.0 essays (set aside) | [`asap2.md`](docs/results/asap2.md) |

## Metrics

Accuracy and macro/weighted F1, the SemEval-2013 metrics. SciEntsBank labels are categories, so QWK isn't used for it. Two graders are compared with a paired McNemar test on the same answers, Holm-corrected within a family fixed before looking.

## Layout

```
src/edge_grader/   the grader (grader.py), data loaders, CLIs, comparisons (compare_*.py), rag/
scripts/           board_grade.sh (Jetson/Pi runs), deploy/fetch, Spark Docker jobs, paper figures
notebooks/         Kaggle: baseline grading, fine-tuning, reasons, RAG index, news judge
results/           every run, committed and never hand-edited; layout in results/README.md
docs/              research outline, plan, results pages, guides; index in docs/README.md
manuscript/        the paper (IEEE TLT draft)
dashboard/         local web view of runs and board state (dashboard/README.md)
tests/             includes an end-to-end run against a fake llama-server
```
