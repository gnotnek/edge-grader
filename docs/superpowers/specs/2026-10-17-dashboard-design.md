# Progress dashboard: design

*2026-10-17. Owner's request; layout chosen from mockups (option B, sidebar with sections).*

## Why

Progress and results are spread over `docs/plan.md`, `docs/results/*.md`, the summary JSONs in `results/` and the board backups in `backups/`. The owner wants one local page to check where the project stands instead of reading those files one by one. Right now the main open question is what the Pi 5 still owes. Another student is using the Pi, so its runs are waiting.

## Goals

- One local web app with a sidebar: **Overview, Plan, Runs, Results, Boards**.
- Shows what has been done, what is still expected and what it measured, read from the repo's own files.
- Practical first and good-looking second: shadcn/ui components, lucide icons, light and dark themes.

## Non-goals

- No live connection to the boards: no ssh and no queue control. The owner refreshes the data with `scripts/backup_board.sh` and `scripts/fetch_results.sh`, then reloads the page.
- No editing: plan checkboxes are still ticked in `docs/plan.md`.
- No deployment, no auth, no access from other machines.

## Anti-goals (a failure even if it runs)

- A number on screen that differs from the JSON it came from. Show stored values as they are, rounded only for display, and show the source file for every run.
- Writing anything into `results/`, `backups/`, `docs/` or `../agentic-edge`. The dashboard only reads.
- Mixing agentic-edge's numbers in with this project's. `results/reference/` may be shown only if it is labelled as reference.

## Constraints

- Local only: the server binds to `127.0.0.1`.
- The Mac has 8 GB of RAM and runs no models. The dashboard must be light: one dev process, with nothing indexed ahead of time.
- Node 26, npm, pnpm and bun are installed. Python stays on uv, as CLAUDE.md requires. The dashboard is a separate JS package and never touches the Python environment.

## Decided by the owner

- Layout B: sidebar with five sections (below).
- Local files only, refreshed by the owner's scripts.
- shadcn/ui and lucide icons.
- Built with the available design skills: **frontend-design** for the look, **dataviz** for charts and stat tiles.

## Builder's calls (Claude's, open to change)

- **Stack:** Vite + React + TypeScript, Tailwind + shadcn/ui, lucide-react, Recharts through shadcn's chart component. npm.
- **Location:** `dashboard/` at the repo root, as its own package. `dashboard/node_modules` and build output are git-ignored.
- **Data access:** a small Vite plugin adds read-only `/api/*` routes to the dev server. Each request reads the repo files fresh, so a reload shows new data. There is no build or cache step, and you run it with `npm run dev` in `dashboard/`.
- **Expected runs:** a hand-edited `dashboard/expected-runs.json` lists the runs the plan still calls for (board × model × benchmark × split/set × condition, plus the plan item they belong to). This is the only way the page can know a run is *missing*. It starts with B1 on the Pi: thinking, test_ua + test_uq, E2B + E4B.
- **Tests:** vitest for the parsers, run against real files from `results/` and `backups/`.

## Data sources and how runs are read

| Source | What it gives |
|---|---|
| `results/board/<run>/` | Board runs that have been fetched: the summary JSON, `telemetry-*/summary.json` (energy, J/token, temperature) and `board-state.txt` |
| `results/kaggle/*.json` | Kaggle runs; `*.judged-*.json` holds the news accuracy |
| `results/estimates/board-check-*.json` (all) | Per run: energy per answer, mean power, board vs Kaggle; Pi vs Jetson pairs |
| `backups/<board>/results/board/<run>/` | Runs that exist on a board but haven't been fetched yet |
| `backups/<board>/queue/jobs.json`, `logs/job-<id>.log` | Queue history: state, args, env, start and end times, rc. The file's mtime is used as "last backup" |
| `docs/plan.md` | Timeline table, `###` sections with `- [x]`/`- [ ]` items, change log |
| `dashboard/expected-runs.json` | Runs still owed |

**Identifying a run.** The folder name gives the condition prefix (`think`, `lg`, `lg-reuse`, `swafull`, `gsm`, `news`, or none for SciEntsBank baseline), model, board, split or set, and stamp. The summary JSON gives stage, benchmark and metrics. Use the folder prefix for the condition: the `lg-*` runs store `"stage": "baseline"`. Kaggle file names follow `<stage>[-<bench>]-<model>-kaggle-t4-…`. Benchmarks have different metrics: SciEntsBank uses accuracy and macro F1 (never QWK), GSM uses accuracy and strict accuracy, MMLU-Pro uses accuracy, and news uses judged accuracy from the `judged` file.

**Run status:**
- **done:** in `results/`.
- **on board:** only in `backups/`, so it still needs a fetch.
- **queued/running:** in a backup's `jobs.json` as pending or running.
- **missing:** in `expected-runs.json` with none of the above.

Duplicate runs of the same cell (for example the two Pi test_ua runs) are all listed. Grids show the newest.

**Energy per answer and board vs Kaggle** come from the board-check files (`results/estimates/board-check-*.json`). For each run, use the newest file that includes it: `j_per_answer`, `mean_w`, and `vs_kaggle` (Kaggle accuracy, same-grade count, McNemar p). The dashboard doesn't recompute them. A run that no check covers has a blank there. *(Changed while planning: the checks already store energy over the grading window, which is more accurate than telemetry's whole-run `energy_j` ÷ n. The latest check file has no `board_pairs`, so pairs come from the newest file that has them.)*

## Pages

- **Overview.** Stat tiles: current phase and week (from the timeline dates and today), runs per board, runs still owed per board, last backup per board ("none yet" for the Pi today). Below them: a "waiting on" list (missing and on-board runs, grouped by board) and the 5–8 newest runs.
- **Plan.** Each stage or phase gets a card with a progress bar (ticked/total) and its checklist. The timeline is shown as a table with the current week highlighted, and the change log is collapsible. Markdown links are shown as text with the file path.
- **Runs.** One filterable, sortable table of all runs. Filters: board, model, benchmark, split/set, condition, status. Clicking a run opens a side sheet with its settings, server build, metrics, token totals, telemetry (energy, J/token, temperature, throttling) and the source path.
- **Results.** Comparison tables, each with a chart where a chart helps:
  1. SciEntsBank accuracy and macro F1 by condition (baseline, thinking, RAG, variants) × model × board.
  2. Baseline vs RAG for each benchmark (Kaggle), as in `docs/results/rag.md`'s four-benchmark table.
  3. Board vs Kaggle accuracy for each checked board run, plus the Pi vs Jetson pairs.
  4. Accuracy vs energy per answer for board runs with telemetry: a scatter, the start of stage 4's main chart.
- **Boards.** One card per board: last backup time, queue jobs (state badge, args/env, duration, rc) and a log tail per job on click. The Pi card says plainly when no backup exists.

A reload button in the header re-reads everything. Theme toggle: light, dark, system.

## Left to the builder

The visual direction within shadcn (from frontend-design), chart colours and forms (from dataviz), exact table columns, how much of `plan.md`'s markdown to render, and empty and error states for missing or malformed files. A bad file appears as a row with an error, never a crash.
