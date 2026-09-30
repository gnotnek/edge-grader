# Docs

**Read first:** [`research-outline.md`](research-outline.md), then [`plan.md`](plan.md).

| Folder | Holds | Files |
|---|---|---|
| *(top)* | What the project is and where it stands | [`research-outline.md`](research-outline.md): research questions, stages, fixed setup, supervisor feedback<br>[`plan.md`](plan.md): step checklist, week-by-week timeline, change log |
| [`results/`](results/) | Numbers and what they mean, one page per stage | [`baselines.md`](results/baselines.md): stage 1, MMLU-Pro and SciEntsBank grading<br>[`edge-benchmarks.md`](results/edge-benchmarks.md): device rates reused from `agentic-edge` |
| [`guides/`](guides/) | How to do things | [`kaggle-setup.md`](guides/kaggle-setup.md): running the grader on Kaggle<br>[`rag.md`](guides/rag.md): how RAG works and how to build it (stage 2) |

**Where new pages go:**
- A stage's results → `results/<stage>.md` (e.g. `results/rag.md`). It links to the raw files in [`../results/`](../results/README.md), which are never hand-edited.
- A how-to → `guides/`.
- A decision or supervisor feedback → a row in `research-outline.md`, not a new page.
- A change of plan → `plan.md`'s change log.
