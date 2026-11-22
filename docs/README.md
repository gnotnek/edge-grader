# Docs

**Read first:** [`research-summary.md`](research-summary.md) (plain words, 5 minutes), then [`research-outline.md`](research-outline.md) (research questions, stages, every decision with its reason). To re-run the experiments, see the [top-level README](../README.md).

| Folder | Holds | Files |
|---|---|---|
| *(top)* | What the project is and where it stands | [`research-summary.md`](research-summary.md): the question, the setup and the answer so far, in plain words<br>[`research-outline.md`](research-outline.md): research questions, stages, fixed setup, supervisor feedback, decisions log<br>[`plan.md`](plan.md): step checklist, week-by-week timeline, change log |
| [`results/`](results/) | Numbers and what they mean, one page per stage | [`summary.md`](results/summary.md): **every finding, by research question; start here**<br>[`recap.md`](results/recap.md): every grading run in one table, 3-way and 2-way<br>[`jetson-graders.md`](results/jetson-graders.md): the three graders and every E2B method on the Jetson, accuracy beside cost (RQ8, RQ4)<br>[`baselines.md`](results/baselines.md): stage 1 and phase 0, every baseline on Kaggle and both boards<br>[`finetuning.md`](results/finetuning.md): stage 3, the adapters, reasons, soup and 5-way labels<br>[`examples.md`](results/examples.md): graded examples in the prompt<br>[`rag.md`](results/rag.md): stage 2, RAG conditions, retrieval check, frozen settings<br>[`topics.md`](results/topics.md): math (GSM8K, GSM-Symbolic) and post-cutoff news (LiveNewsBench)<br>[`variations.md`](results/variations.md): thinking on, little-gemma, prefix reuse<br>[`edge-benchmarks.md`](results/edge-benchmarks.md): device rates reused from `agentic-edge`<br>[`asap2.md`](results/asap2.md): ASAP 2.0 essays (set aside) |
| [`design/`](design/) | One design per experiment, written before its runs: what is compared, the test family, the bar | Dated `YYYY-MM-DD-<topic>-design.md`. Result JSONs written before 2026-11-23 cite these as `docs/superpowers/specs/…`, the folder's old name |
| [`literature/`](literature/) | Reading notes behind the method choices | [`2026-11-05-methods-review.md`](literature/2026-11-05-methods-review.md): methods besides fine-tuning, ranked for RQ4<br>[`2026-11-12-new-methods.md`](literature/2026-11-12-new-methods.md): methods to raise E4B's accuracy after graded examples<br>[`2026-11-12-method-references-check.md`](literature/2026-11-12-method-references-check.md): the references behind those methods, checked<br>[`2026-11-13-llm-finetuning-on-scientsbank.md`](literature/2026-11-13-llm-finetuning-on-scientsbank.md): how others fine-tune on SciEntsBank, and the leakage question<br>[`benchmark-survey/`](literature/benchmark-survey/): which datasets 35 recent grading papers use (English, non-English, usage counts) |
| [`guides/`](guides/) | How to do things | [`kaggle-setup.md`](guides/kaggle-setup.md): running the notebooks on Kaggle<br>[`spark-docker.md`](guides/spark-docker.md): the DGX Spark's Docker jobs<br>[`rag.md`](guides/rag.md): how RAG works and how to build it<br>[`rag-sources.md`](guides/rag-sources.md): the RAG material and how to download it |

**Where new pages go:**
- A stage's results → `results/<stage>.md`. It links to the raw files in [`../results/`](../results/README.md), which are never hand-edited.
- An experiment's design, before its runs → `design/YYYY-MM-DD-<topic>-design.md`.
- Reading notes → `literature/YYYY-MM-DD-<topic>.md`.
- A how-to → `guides/`.
- Supervisor feedback → a row in `research-outline.md`'s feedback table. Your own decision → a row in its "Own decisions" table, with the research question it serves.
- A change of plan → `plan.md`'s change log.
