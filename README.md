# edge-grader

Can a small LLM running on an edge device (Raspberry Pi 5, Jetson Orin Nano) grade students' short answers, and how much do subject knowledge (RAG) and fine-tuning improve it?

Part of an internship at the NTUST MIT Lab. It builds on the lab's `agentic-edge` benchmark: same models (Gemma 4 E2B/E4B), same `llama-server` setup, and the same MMLU-Pro baseline as a comparison point. Background, decisions and literature are in the separate `research-ideation` notes.

**Research questions, stages and supervisor feedback:** [`docs/research-outline.md`](docs/research-outline.md). **Steps and timeline:** [`docs/plan.md`](docs/plan.md).

## Plan

| Stage | What | Status |
|---|---|---|
| 1. Baseline | Untouched model grades SciEntsBank answers; the lab's MMLU-Pro subset as a second baseline | Done: MMLU-Pro copied; grading on Kaggle (E4B 67–70%, E2B 55–60%), same accuracy on both boards; cost measured on test_ua (Jetson 4–7 J, Pi 5 28–56 J per answer) ([docs/results/baselines.md](docs/results/baselines.md)) |
| 2. RAG | Retrieve course material (open textbooks) into the prompt | SciEntsBank done: RAG-en doesn't improve grading (E4B flat, E2B −2 to −5 points), it makes it more lenient ([docs/results/rag.md](docs/results/rag.md)). MMLU-Pro runner and retrieval built; Kaggle runs next |
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

Stage 1 board cost is **measured** on test_ua, both boards (`docs/results/baselines.md`); the earlier estimate (`uv run edge-grader-cost`) is kept for the other splits. After fetching runs, `uv run edge-grader-board-check results/board/<run>/ ...` compares them with Kaggle and the estimate. On the board, with `agentic-edge` deployed and the board idle:

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
- **Run it directly on the board, never through the lab's job queue.** edge-grader is a separate project; its runs must not mix with `agentic-edge`'s data. A queue job is wrapped in the queue's own `run_measured.sh`, which writes telemetry into `agentic-edge`'s `measured/` and needs a marker in its `stdbench/`, so the run ends up in the lab's results and on its dashboard.
  - The script refuses to run from the queue (or under any `run_measured.sh` but its own), and refuses to write telemetry outside this repo's `results/`.
  - It refuses to start while an `agentic-edge` run has the board (one job per board), so check the lab's queue first. The lab's queue in turn waits while this script runs.
  - Runs from before 2026-10-05 that went through the queue were moved out of `agentic-edge`'s folders on the boards to `<root>/external/edge-grader/` (`~/Research` on the Pi, `~/research` on the Jetson). Their telemetry is already committed under `results/board/`.

### Anywhere else

Start a `llama-server` yourself, e.g. `llama-server -m <model.gguf> -c 8192 --port 8080 -rea off --reasoning-budget -1 --cache-ram 0`, then:

```bash
uv run edge-grader-baseline --label e4b-q4 --n 100                 # 100 answers, unseen-answers split
uv run edge-grader-baseline --label e4b-q4 --split test_ud --n 0   # whole unseen-domains split
uv run edge-grader-mmlupro --label e4b-q4 --subset s1 --n 5        # MMLU-Pro, lm-eval's prompt
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
  mmlupro.py    MMLU-Pro on the lab's 300 questions, lm-eval's prompt, with or without RAG
  cost.py       board cost estimate from agentic-edge rates (edge-grader-cost)
  board_check.py  board runs vs Kaggle accuracy and the estimate (edge-grader-board-check)
  rag/          stage 2: sources.json + sources.py (material, downloader), corpus.py (chunks),
                index.py (embed, retrieve), check.py (retrieval check); see docs/guides/rag.md
notebooks/
  kaggle_baseline.ipynb   grading and MMLU-Pro on a Kaggle GPU (stages 1-2)
  kaggle_rag_index.ipynb  stage 2: embed the chunks and retrieve, on Kaggle
scripts/
  board_grade.sh          stage 1 on the Pi 5 / Jetson, with telemetry
tests/                    includes an end-to-end run against a fake llama-server
results/
```
