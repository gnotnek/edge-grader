# Docs

**Read first:** [`research-outline.md`](research-outline.md), then [`plan.md`](plan.md).

| Folder | Holds | Files |
|---|---|---|
| *(top)* | What the project is and where it stands | [`research-outline.md`](research-outline.md): research questions, stages, fixed setup, supervisor feedback<br>[`plan.md`](plan.md): step checklist, week-by-week timeline, change log<br>[`methods-review.md`](methods-review.md): literature shortlist of methods besides fine-tuning, ranked for RQ4 (proposals, 2026-11-05) |
| [`results/`](results/) | Numbers and what they mean, one page per stage | [`summary.md`](results/summary.md): **every finding so far, by research question; start here**<br>[`baselines.md`](results/baselines.md): stage 1, MMLU-Pro and SciEntsBank grading; phase 0, every baseline on both boards<br>[`variations.md`](results/variations.md): phase B, thinking on, little-gemma and prefix reuse, with `agentic-edge`'s MMLU-Pro side<br>[`edge-benchmarks.md`](results/edge-benchmarks.md): device rates reused from `agentic-edge`<br>[`rag.md`](results/rag.md): stage 2, RAG conditions, retrieval check, frozen settings<br>[`topics.md`](results/topics.md): stage 2 topic benchmarks, math (GSM8K, GSM-Symbolic) against post-cutoff news (LiveNewsBench) |
| [`guides/`](guides/) | How to do things | [`kaggle-setup.md`](guides/kaggle-setup.md): running the grader on Kaggle<br>[`rag.md`](guides/rag.md): how RAG works and how to build it (stage 2)<br>[`rag-sources.md`](guides/rag-sources.md): the RAG material for SciEntsBank and MMLU-Pro, English and Indonesian, and how to download it |

**Where new pages go:**
- A stage's results → `results/<stage>.md` (e.g. `results/rag.md`). It links to the raw files in [`../results/`](../results/README.md), which are never hand-edited.
- A how-to → `guides/`.
- Supervisor feedback → a row in `research-outline.md`'s feedback table. Your own decision → a row in its "Own decisions" table, with the research question it serves.
- A change of plan → `plan.md`'s change log.
