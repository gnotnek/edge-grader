# results/

The experiment record. **Everything here is committed**, so each run can be traced later. Nothing from this project is written into `agentic-edge`.

```
results/
  reference/   numbers copied from elsewhere, with their source
               agentic-edge-mmlupro-baseline.json   the lab's MMLU-Pro baseline (not re-run)
               agentic-edge/                        the lab's device benchmarks, copied (see its README)
  kaggle/      accuracy runs on Kaggle GPUs (from notebooks/kaggle_baseline.ipynb)
  estimates/   board cost estimated from agentic-edge's rates + a run's token counts
               (uv run edge-grader-cost results/kaggle/baseline-*.json)
  board/       runs on the Pi 5 / Jetson (from scripts/board_grade.sh), one folder per run:
    <model>-<board>-<split>-<stamp>/
      baseline-*.json   grading summary: metrics, settings, server model path, token totals
      raw/*.jsonl       one line per graded answer, including the model's raw reply
      server.log        llama-server log
      telemetry-*/      agentic-edge's run_measured.sh output: telemetry.csv, requests.csv,
                        meta.json, summary.json (energy, J/token), command.log
  local/       ad-hoc runs (the CLI's default --out); commit only the ones worth keeping
```

**Rules:**
- **Never hand-edit a result file.** If a run was wrong, keep it, and note why in the commit message or in `docs/results/baselines.md`.
- **Record the source commit** for anything copied from another repo.
