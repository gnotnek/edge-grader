# edge-grader

Can a small LLM running on an edge device (Raspberry Pi 5, Jetson Orin Nano) grade students' short answers, and how much do subject knowledge (RAG) and fine-tuning improve it?

Part of an internship at the NTUST MIT Lab. It builds on the lab's `agentic-edge` benchmark: same models (Gemma 4 E2B/E4B), same `llama-server` setup, and the same MMLU-Pro baseline as a comparison point. Background, decisions and literature are in the separate `research-ideation` notes.

**Research questions, stages and supervisor feedback:** [`docs/research-outline.md`](docs/research-outline.md). **Steps and timeline:** [`docs/plan.md`](docs/plan.md).

## Plan

| Stage | What | Status |
|---|---|---|
| 1. Baseline | Untouched model grades SciEntsBank answers; the lab's MMLU-Pro subset as a second baseline | MMLU-Pro copied; grading done on Kaggle (E4B 67–70%, E2B 55–60%); board cost pending ([docs/baselines.md](docs/results/baselines.md)) |
| 2. RAG | Retrieve course material (open textbooks) into the prompt | — |
| 3. Fine-tuning | QLoRA on Kaggle/Colab, converted to GGUF | — |
| 4. Edge cost | Latency and energy per answer on the Pi 5 / Jetson, for each stage | — |

## Setup

Needs [uv](https://docs.astral.sh/uv/).

```bash
uv sync          # creates .venv and installs everything from uv.lock
uv run pytest    # quick check
```

## Run the baseline

There are three ways to run it. All use the lab's model files and `llama-server` flags (thinking off), so the numbers are comparable with `agentic-edge`.

### On Kaggle (accuracy)

First time on Kaggle? Follow [`docs/guides/kaggle-setup.md`](docs/guides/kaggle-setup.md). Otherwise: open [`notebooks/kaggle_baseline.ipynb`](notebooks/kaggle_baseline.ipynb) on Kaggle. Its first cell lists the setup: GPU on, Internet on, and a read-only `GITHUB_TOKEN` secret. It builds `llama-server` with CUDA, downloads the models, and grades. Start with `N = 100`, check that `unparsed` ≈ 0, then run the full splits.

### On a lab board (accuracy + latency + energy)

The code sits next to `agentic-edge` on each board, using the same ssh aliases:

| | Pi 5 (`MITLAB-EDGE`) | Jetson (`MITLAB-JETSON`) |
|---|---|---|
| repo | `~/Research/edge-grader` | `~/research/edge-grader` |

Deploy from the Mac, as agentic-edge does. `deploy.sh` rsyncs the committed files, so the boards need no GitHub access. It also installs uv on first use, runs `uv sync` and the tests on the board, and records the commit in `DEPLOYED.json`:

```bash
scripts/deploy.sh              # both boards (or: pi | jetson)
scripts/fetch_results.sh       # later: pull results/board/ back here to commit
```

Stage 1 board cost is **estimated**, not measured (`uv run edge-grader-cost results/kaggle/baseline-*.json`; see `docs/results/baselines.md`). A measured run is optional, to check the estimate. On the board, with `agentic-edge` deployed and the board idle:

```bash
ssh MITLAB-EDGE                              # or MITLAB-JETSON (~/research/...)
cd ~/Research/edge-grader
scripts/board_grade.sh e4b                  # test_ua, all 540 answers, with telemetry
scripts/board_grade.sh e2b test_ud 200      # split and count are optional
```

- **What the script does:**
  - detects the board
  - stops the lab's server (on the Pi, the `va-llm` service) and starts `llama-server` with the lab's baseline flags
  - runs the grading inside `agentic-edge`'s `run_measured.sh`, so energy is recorded the same way as MMLU-Pro
  - **restores the board on exit**, even if the run fails
- **Outputs:**
  - grading results and telemetry: `results/board/<run>/` on the board; fetch them with `scripts/fetch_results.sh`
- **Run it directly** when the board is idle, rather than through the lab job queue. Queue jobs are measured by the queue's own `run_measured.sh` call, which writes telemetry into `agentic-edge`'s folders.
  - If you must use the queue, submit a `raw` job with label `<label>` and command `env MEASURE=0 <root>/edge-grader/scripts/board_grade.sh e4b && mkdir -p <root>/stdbench/<label> && touch <root>/stdbench/<label>/.done`, where `<root>` is `~/Research` on the Pi and `~/research` on the Jetson, written out in full. Then copy that job's telemetry folder (`<root>/measured/<label>-<stamp>/`) into the run's `results/board/<run>/` folder.
    - **The `.done` part matters.** The queue counts a job as completed only if `stdbench/<label>/.done` exists; the exit code alone isn't enough. Without it, a successful run shows as "failed". The marker is written only if grading exits 0.
    - **Use a new label for every run.** The queue refuses a label whose `stdbench/<label>` folder already exists.

### Anywhere else

Start a `llama-server` yourself, e.g. `llama-server -m <model.gguf> -c 8192 --port 8080 -rea off --reasoning-budget -1 --cache-ram 0`, then:

```bash
uv run edge-grader-baseline --label e4b-q4 --n 100                 # 100 answers, unseen-answers split
uv run edge-grader-baseline --label e4b-q4 --split test_ud --n 0   # whole unseen-domains split
```

The run stops straight away if no server answers. A failed request is recorded as an error and the run continues.

**All outputs stay in this repo and are committed:** nothing is written into `agentic-edge`. See [`results/README.md`](results/README.md) for the layout.

**Metrics:** accuracy and macro/weighted F1, the SemEval-2013 metrics. SciEntsBank labels are categories, not scores, so QWK isn't used here.

## Layout

```
docs/                   index: docs/README.md
  research-outline.md   research questions, stages, supervisor feedback
  plan.md               step checklist and week-by-week timeline
  results/              numbers: baselines, edge benchmarks
  guides/               how-tos: Kaggle setup, RAG pipeline
src/edge_grader/
  data.py       SciEntsBank loader + 5/3/2-way label schemes
  grader.py     prompt, model call, label parsing
  metrics.py    accuracy, F1
  baseline.py   stage-1 CLI
  cost.py       board cost estimate from agentic-edge rates (edge-grader-cost)
  rag/          stage 2: sources.json + sources.py (material, downloader), corpus.py (chunks),
                index.py (embed, retrieve), check.py (retrieval check); see docs/guides/rag.md
notebooks/
  kaggle_baseline.ipynb   stage 1 on a Kaggle GPU
  kaggle_rag_index.ipynb  stage 2: embed the chunks and retrieve, on Kaggle
scripts/
  board_grade.sh          stage 1 on the Pi 5 / Jetson, with telemetry
tests/                    includes an end-to-end run against a fake llama-server
results/
```
