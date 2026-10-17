# dashboard/

A local, read-only page that shows where the project stands: the plan's progress, every run, the results so far, and what each board still owes. It reads the repo's own files on every page load, so it never shows anything the files don't say.

```bash
cd dashboard
npm install        # once
npm run dev        # http://127.0.0.1:5173 (this machine only)
```

To refresh, copy from the boards and then press **Reload data**:

```bash
scripts/backup_board.sh pi        # queue + results the board holds (backups/, git-ignored)
scripts/fetch_results.sh pi       # finished run folders into results/board/ (to commit)
```

## Pages

| Page | Shows | Read from |
|---|---|---|
| Overview | This week, runs per board, runs owed, coverage map | everything below |
| Plan | Timeline, every checklist, change log | `docs/plan.md` |
| Runs | Every run, filterable; click one for its summary file | `results/board/`, `results/kaggle/`, `backups/*/results/board/` |
| Results | SciEntsBank by condition, accuracy vs energy, baseline vs RAG, boards vs Kaggle | the run summaries and `results/estimates/board-check-*.json` |
| Boards | Each board's queue jobs and log tails, and what it still owes | `backups/<board>/queue/` |

Numbers are shown as stored, only rounded. News accuracy is always the judge's (`*.judged-*.json`). Energy per answer and the board vs Kaggle comparison come from the newest board-check file that covers the run, never recomputed here.

## Telling it what's still owed

The dashboard can see runs that exist, not runs the plan still calls for. Those go in `expected-runs.json`, one entry per run:

```json
{ "board": "pi5", "model": "e2b", "benchmark": "scientsbank", "split": "test_ua", "condition": "think", "plan": "B1 thinking on" }
```

- `board`: `pi5` | `jetson` | `kaggle`
- `benchmark`: `scientsbank` | `gsm` | `news` | `mmlupro`
- `condition`: `baseline` | `think` | `rag-en` | `rag-id` | `swafull` | `lg` | `lg-reuse`
- `split`: as in the run folder name (`test_ua`, `gsm8k`, `symbolic-i10`, `test`, `s1`)

Each entry is shown as **done** (in `results/`), **on board, not fetched** (only in `backups/`), **running** or **queued** (in the board's `jobs.json`), or **not run yet**.

## Code

- `server/`: Node code that reads the files (`runs.ts`, `plan.ts`, `boards.ts`, `status.ts`) and serves them as `/api/*` from the Vite dev server (`api.ts`). It only reads; any other method gets 405.
- `src/`: the React pages (shadcn/ui, lucide icons, Recharts).
- `npm test` runs the parser tests against the real repo files; `npm run build` type-checks.

This is separate from the Python package in `../src/edge_grader/`: nothing here imports it or writes where it writes.
