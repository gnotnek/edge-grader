# Progress Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local, read-only web dashboard (Overview, Plan, Runs, Results, Boards) built from this repo's own files.

**Architecture:** A Vite + React app in `dashboard/`. A Vite plugin adds read-only `/api/*` routes to the dev server. Each request runs Node parsers in `dashboard/server/` against the repo root (`..`). The parsers return typed JSON, and the React pages render it with shadcn/ui.

**Tech Stack:** Vite, React 19, TypeScript, Tailwind v4, shadcn/ui, lucide-react, Recharts (shadcn chart), vitest. npm.

**Spec:** `docs/superpowers/specs/2026-10-17-dashboard-design.md`

## Global Constraints

- Read-only: nothing under `dashboard/server/` writes a file. Every `fs` call is a read.
- The dev server binds to `127.0.0.1` (`server.host: "127.0.0.1"`).
- SciEntsBank shows accuracy and macro F1, never QWK. News accuracy comes only from `*.judged-*.json`.
- Values are shown as stored and rounded only for display. Every run shows its source path.
- The condition comes from the folder prefix (`think`, `lg`, `lg-reuse`, `swafull`), never from `stage`, because `lg-*` runs store `"stage": "baseline"`.
- Energy per answer and board-vs-Kaggle come from the newest board-check file that includes the run. Nothing is recomputed.
- A malformed or missing file becomes an error entry, never a thrown request.

## Review Focus

- **A run folder with no summary JSON yet** (a board run still in progress in `backups/`) should show as a run with status "on board" and blank metrics, not be dropped. Test in Task 2.
- **Pi with no `backups/pi/`** should show on Boards as "no backup yet" and on Overview as "none yet", not as an error. Test in Task 3.
- **`interrupted-*` folders in `backups/*/results/local/`** are not runs and must not appear in the run list. Test in Task 2.
- **The same cell run twice** (`e2b-pi5-test_ua` ×2): both are listed, and grids and status use the newest stamp. Test in Task 3 (`latestByCell`).
- **Broken JSON** in one result file: it shows as an error row and the other runs still load. Test in Task 2.

---

### Task 1: Scaffold `dashboard/` with shadcn, theme and sidebar shell

**Files:**
- Create: `dashboard/` (Vite react-ts template), `dashboard/components.json`, `dashboard/src/App.tsx`, `dashboard/src/components/app-sidebar.tsx`, `dashboard/src/components/theme.tsx`
- Modify: `.gitignore` (add `dashboard/node_modules/`, `dashboard/dist/`)

- [ ] Create the Vite react-ts app in `dashboard/`, add Tailwind v4 and shadcn (`npx shadcn@latest init`), and add the components: sidebar, card, badge, table, button, select, sheet, progress, tabs, chart, separator, tooltip, scroll-area, collapsible, skeleton.
- [ ] Hash routing in `App.tsx` with `#/overview|plan|runs|results|boards`. No router dependency.
- [ ] Theme provider (light, dark, system; stored in localStorage) and a header with a reload button and the theme toggle.
- [ ] `npm run build` passes, and `npm run dev` serves the shell on 127.0.0.1.
- [ ] Commit.

### Task 2: Run parsers

**Files:**
- Create: `dashboard/server/types.ts`, `dashboard/server/runs.ts`, `dashboard/server/runs.test.ts`

**Interfaces:**
- Produces:
  - `type Board = "pi5" | "jetson" | "kaggle"`
  - `type Benchmark = "scientsbank" | "gsm" | "news" | "mmlupro"`
  - `type Condition = "baseline" | "rag-en" | "rag-id" | "think" | "lg" | "lg-reuse" | "swafull"`
  - `interface Run { id: string; board: Board; model: "e2b" | "e4b"; benchmark: Benchmark; split: string; condition: Condition; stamp: string; time?: string; location: "results" | "backup"; source: string; metrics: Record<string, number | null>; summary?: unknown; error?: string; energy?: { jPerAnswer: number; meanW: number }; vsKaggle?: { kaggleRun: string; kaggleAccuracy: number; boardAccuracy: number; n: number; sameGrade: number; p: number } }`
  - `parseRunFolder(name: string): { condition: Condition; benchmark: Benchmark; model; board; split; stamp } | null`
  - `parseKaggleFile(name: string): same shape | null` (null for `*.judged-*`)
  - `loadRuns(repo: string): Run[]`

- [ ] Write tests:
  - `parseRunFolder("think-e2b-jetson-test_ua-20261011-232209")` gives condition `think`, benchmark `scientsbank`, split `test_ua`.
  - `"lg-reuse-e4b-jetson-test_ua-20261013-154741"` gives `lg-reuse`.
  - `"gsm-e2b-jetson-symbolic-i10-20261009-020450"` gives benchmark `gsm`, split `symbolic-i10`.
  - `"news-e4b-pi5-test-20261010-011719"` gives `news`.
  - `"interrupted-think-e4b-jetson-test_ud-20261013-063811"` gives null.
  - `parseKaggleFile("rag-e4b-kaggle-t4-id-k3-test_ud-3way-20261005-115314.json")` gives `rag-id`, `test_ud`.
  - `"mmlupro-rag-e2b-kaggle-t4-en-k3-s1-…"` gives `mmlupro`, `rag-en`, `s1`.
  - `"gsm-rag-symbolic-e4b-…"` gives `gsm`, `rag-en`, `symbolic`.
  - The `.judged-` file gives null.
  - `loadRuns(REPO)` on the real repo: 38 runs from `results/board`, and the news run's `metrics.accuracy` equals the judged file's accuracy.
  - Fixture tests (a tmp dir): a folder with no JSON gives a run with `metrics: {}` and no error. Broken JSON gives a run with `error` set while the others load. A backup-only folder gives `location: "backup"`. A folder present in both places gives only the `results` one.
- [ ] Implement. Board summary = the folder's top-level `*.json` that isn't `*.judged-*`. Metrics copied: `accuracy, macro_f1, strict_accuracy, n, median_latency_s, wall_s, prompt_tokens, completion_tokens, unparsed, errors`. For news, `accuracy` comes from the judged file next to it, and is null if there is none. Board-check enrichment: read every `results/estimates/board-check-*.json` in name order, and the newest one wins per `run`.
- [ ] Tests pass, then commit.

### Task 3: Plan, boards, expected runs, status

**Files:**
- Create: `dashboard/server/plan.ts`, `dashboard/server/boards.ts`, `dashboard/server/status.ts`, `dashboard/expected-runs.json`, and a test file for each.

**Interfaces:**
- Produces:
  - `parsePlan(md: string): { updated?: string; timeline: { week: string; dates: string; stage: string; goal: string; checkpoint: string; start?: string; end?: string }[]; sections: { title: string; items: { done: boolean; text: string; group?: string }[] }[]; changelog: { date: string; change: string; why: string }[] }`
  - `loadBoards(repo: string): { board: "pi5" | "jetson"; hasBackup: boolean; lastBackup?: string; jobs: Job[] }[]`, where `Job` = the jobs.json fields plus `logTail?: string` (the last 40 lines)
  - `interface Expected { board: Board; model; benchmark: Benchmark; split: string; condition: Condition; plan: string }`
  - `type Status = "done" | "on-board" | "queued" | "running" | "missing"`
  - `cellKey(x): string` = `board|model|benchmark|split|condition`
  - `latestByCell(runs: Run[]): Map<string, Run>`
  - `expectedStatus(expected: Expected[], runs: Run[], boards): (Expected & { status: Status; run?: Run })[]`

- [ ] Write tests:
  - `parsePlan` on the real `docs/plan.md`: the timeline has 9 rows, `Oct 12 – 18` gets start `2026-10-12`, Phase 0 has 7 items all done, Phase B has 3 items none done, the change log has more than 20 rows.
  - `loadBoards` on the real repo: jetson `hasBackup: true` with 23 jobs, pi5 `hasBackup: false`.
  - `latestByCell` keeps the 20261004-225353 Pi test_ua run over 075411.
  - `expectedStatus`: a run in results gives done, backup only gives on-board, a matching pending job gives queued, nothing gives missing.
  - The job↔cell match uses env `BENCH` (default `scientsbank`), `THINKING=1` → `think`, `VARIANT` → condition, `RAG` → `rag-en`, and args `[model, split]`.
- [ ] `expected-runs.json` starts with the four Pi B1 cells (pi5 × e2b/e4b × test_ua/test_uq × think, plan "B1 thinking on").
- [ ] Implement, run the tests, commit.

### Task 4: API plugin

**Files:**
- Create: `dashboard/server/api.ts`
- Modify: `dashboard/vite.config.ts`

**Interfaces:**
- Produces:
  - `GET /api/runs` → `Run[]` (without `summary`)
  - `GET /api/runs/:id` → `Run` with `summary`
  - `GET /api/plan`
  - `GET /api/boards`
  - `GET /api/expected` → the `expectedStatus` output
  - `GET /api/overview` → `{ today; currentWeek; runsByBoard; owedByBoard; lastBackup; waiting; newest }`
  - `GET /api/pairs` → board_pairs from the newest check file that has them

- [ ] Implement `apiPlugin(repo)` as a `configureServer` middleware. Errors become `500 {error}`. `REPO = path.resolve(__dirname, "..")`.
- [ ] Check every route with curl against the dev server. Commit.

### Task 5: Pages

**Files:**
- Create: `dashboard/src/lib/api.ts` (`useApi<T>(path)` hook with a reload key), `dashboard/src/lib/format.ts`, `dashboard/src/pages/{overview,plan,runs,results,boards}.tsx`, and shared `stat-tile.tsx`, `status-badge.tsx`, `run-sheet.tsx`.

Load the **frontend-design** and **dataviz** skills before writing pages.

- [ ] **Overview:** stat tiles (current week and stage, runs per board, owed per board, last backup), a "waiting on" card grouped by board, and the newest runs.
- [ ] **Plan:** progress cards per section, the timeline table with the current week highlighted, and a collapsible change log.
- [ ] **Runs:** filters (board, model, benchmark, split, condition), a sortable table, and a row click that opens `run-sheet` (metrics, energy, vs-Kaggle, settings JSON, source path).
- [ ] **Results:** (1) a SciEntsBank grouped bar chart of accuracy and macro F1 by condition per model/board, (2) a baseline vs RAG table per benchmark (Kaggle, newest per cell) with deltas, (3) a board vs Kaggle table plus the Pi vs Jetson pairs, (4) an accuracy vs J/answer scatter (log x).
- [ ] **Boards:** one card per board with the job table and log tail on click. The Pi gets the "no backup yet" empty state.
- [ ] `npm run build` and `npm test` pass. Commit.

### Task 6: Verify in the browser and write the README

- [ ] Run the dev server, open each page in the browser pane, check light and dark, and compare 3 numbers against their JSONs.
- [ ] `dashboard/README.md`: how to run it, the data sources, and how to edit `expected-runs.json`. Commit.
