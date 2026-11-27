# results/

The experiment record. **Everything here is committed**, so each run can be traced later. Nothing from this project is written into `agentic-edge`. Runs of the experiments the paper doesn't use (ASAP 2.0, GSM, news, MMLU-Pro, little-gemma, `--swa-full`, the Pi 5, the cascade) are on the [`experiments`](https://github.com/gnotnek/edge-grader/tree/experiments/results) branch.

```
results/
  board/       Jetson runs (scripts/board_grade.sh), one folder per run:
    [<version>-]<model>-jetson-<split>-<stamp>/
               version: ft, ft-lf, ft-rf, ft-s0..s4, ft-soup, ft-5w-s0, ex4, ex4-ft, rand4, think, rag,
               and their combinations (ft-think, ...). No prefix = the untouched model.
      baseline-*.json   the runner's summary: metrics, settings, server model path and build, tokens
      raw/*.jsonl       one line per answer, including the model's raw reply
      server.log        llama-server log
      board-state.txt   CPU governor, clocks, devfreq and nvpmodel at the start of the run
      telemetry-*/      agentic-edge's run_measured.sh output: telemetry.csv, requests.csv,
                        meta.json, summary.json (energy, J/token), command.log
  kaggle/      accuracy runs on Kaggle T4s (notebooks/): baseline-*, rag-*, think-* (SciEntsBank),
               beetle-*/saf-* (held-out sets), *-ft-* (with an adapter); compare-* are the tests
  spark/       accuracy runs on the DGX Spark (compose.yaml): seeds, soup, 5-way labels, graded examples;
               compare-* are the tests
  finetune/    training records: finetune-<model>-*.json (settings, per-epoch validation),
               <model>-soup-*.json; raw/ holds the teacher's reasons for stage 3b
  examples/    the graded examples put in the prompt (nearest and random), as answer ids
  rag/         chunks-scientsbank.json (what was chunked, sha256), retrieval-<index>-k5.json (top-k chunk
               ids per question; no text), check-<index>-k5.json (the retrieval check's marks)
  estimates/   board-check-*.json: board runs scored and paired with Kaggle (edge-grader-board-check);
               compare-jetson-* and compare-thinking-*: the paper's Jetson tables and tests;
               board-cost-estimate-*.json: the early estimate from agentic-edge's rates (edge-grader-cost)
  ci/          baseline-ci-*.json: runs re-scored from raw/ with a 95% range (edge-grader-baseline-ci)
  assess/      label-prior check of the adapter on Beetle and SAF train (python -m edge_grader.label_prior)
  reference/   numbers copied from elsewhere, with their source: agentic-edge/ (the lab's device data)
  local/       ad-hoc runs (the CLI's default --out); commit only the ones worth keeping
```

**Rules:**
- **Never hand-edit a result file.** If a run was wrong, keep it, and note why in the commit message or in the results page that reports it.
- **Record the source commit** for anything copied from another repo.
