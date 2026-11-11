# results/

The experiment record. **Everything here is committed**, so each run can be traced later. Nothing from this project is written into `agentic-edge`.

```
results/
  reference/   numbers copied from elsewhere, with their source
               agentic-edge-mmlupro-baseline.json   the lab's MMLU-Pro baseline (not re-run)
               agentic-edge/                        the lab's device benchmarks, copied (see its README)
  kaggle/      accuracy runs on Kaggle GPUs (from notebooks/kaggle_baseline.ipynb): baseline-*/rag-*
               (SciEntsBank), mmlupro-*, gsm-* and news-* (topic benchmarks); news-*.judged-<judge>.json
               are the judge's grades of a news run, written by edge-grader-news-judge
  rag/         stage 2: chunks-<benchmark>.json (what was chunked, sha256),
               retrieval-<index>-k<k>.json (top-k chunk ids per question; no text) and
               check-<index>-k<k>.json (the retrieval check's marks). Benchmarks: scientsbank, mmlupro, news
  estimates/   board-cost-estimate-*.json: cost estimated from agentic-edge's rates + a run's token counts
               board-check-*.json: measured board runs vs Kaggle accuracy and that estimate
               (uv run edge-grader-cost results/kaggle/baseline-*.json)
  board/       runs on the Pi 5 / Jetson (from scripts/board_grade.sh), one folder per run:
    <model>-<board>-<split>-<stamp>/          SciEntsBank grading (stage 1 and phase 0)
    <bench>-<model>-<board>-<set>-<stamp>/    phase 0: gsm (set gsm8k | symbolic-i<variants>), news (set test)
      baseline-*.json | gsm-*.json | news-*.json   the runner's summary: metrics, settings,
                        server model path and build, token totals
      raw/*.jsonl       one line per question, including the model's raw reply
      news-*.judged-qwen3-14b-q4_k_m.json, raw/*.judged-*.jsonl   news only: the Kaggle judge's grades
      server.log        llama-server log
      board-state.txt   CPU governor and clocks (Jetson: devfreq, nvpmodel) at the start of the run
      telemetry-*/      agentic-edge's run_measured.sh output: telemetry.csv, requests.csv,
                        meta.json, summary.json (energy, J/token), command.log
  ci/          baseline-ci-*.json: finished runs re-scored from raw/ with a 95% range (bootstrap over
               questions), the 2-way score and the majority floor (uv run edge-grader-baseline-ci)
  cascade/     cascade-jetson-*.json: cascade and defer-to-teacher simulated from existing Jetson runs
               (uv run python -m edge_grader.cascade); no new board time
  local/       ad-hoc runs (the CLI's default --out); commit only the ones worth keeping
```

**Rules:**
- **Never hand-edit a result file.** If a run was wrong, keep it, and note why in the commit message or in `docs/results/baselines.md`.
- **Record the source commit** for anything copied from another repo.
