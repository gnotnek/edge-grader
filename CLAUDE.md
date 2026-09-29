# edge-grader

Research code: small LLMs grading student short answers on edge devices (baseline → RAG → fine-tuning → edge cost). Read `docs/research-outline.md` first. It holds the research questions, stages and fixed setup.

- **Python via uv only:** `uv sync`, `uv run pytest`, `uv run ruff check .`, and `uv add <pkg>` for new dependencies. Never pip.
- **The owner's Mac (8 GB) doesn't run models.** Inference runs on Kaggle/Colab or the lab's Pi 5 / Jetson through `llama-server`. This machine is for code, tests and analysis.
- **Keep runs comparable with `../agentic-edge`:** same models, same `llama-server` flags, thinking off unless the run is a thinking condition.
- **Every run writes a summary JSON to `results/`** with its settings. Don't hand-edit results.
- **SciEntsBank metrics are accuracy and F1, not QWK:** the labels are categories.
