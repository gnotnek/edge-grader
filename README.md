# edge-grader

Can a small LLM running on an edge device (Raspberry Pi 5, Jetson Orin Nano) grade students' short answers, and how much do subject knowledge (RAG) and fine-tuning improve it?

Part of an internship at the NTUST MIT Lab. It builds on the lab's `agentic-edge` benchmark: same models (Gemma 4 E2B/E4B), same `llama-server` setup, and the same MMLU-Pro baseline as a comparison point. Background, decisions and literature are in the separate `research-ideation` notes.

**Research questions, stages and supervisor feedback:** [`docs/research-outline.md`](docs/research-outline.md).

## Plan

| Stage | What | Status |
|---|---|---|
| 1. Baseline | Untouched model grades SciEntsBank answers; the lab's MMLU-Pro subset as a second baseline | in progress |
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

Start a model server first. It's the same command the lab uses; `-rea off` turns thinking off:

```bash
llama-server -m <model.gguf> -c 8192 --host 127.0.0.1 --port 8080 -rea off
```

Then:

```bash
uv run edge-grader-baseline --label e4b-q4 --n 100                 # 100 answers, unseen-answers split
uv run edge-grader-baseline --label e4b-q4 --split test_ud --n 0   # whole unseen-domains split
```

Results:
- `results/<run>.json`: the run summary (committed)
- `results/raw/<run>.jsonl`: one line per answer (not committed)

**Metrics:** accuracy and macro/weighted F1, the SemEval-2013 metrics. SciEntsBank labels are categories, not scores, so QWK isn't used here.

## Layout

```
docs/
  research-outline.md   research questions, stages, supervisor feedback
src/edge_grader/
  data.py       SciEntsBank loader + 5/3/2-way label schemes
  grader.py     prompt, model call, label parsing
  metrics.py    accuracy, F1
  baseline.py   stage-1 CLI
tests/
results/
```
