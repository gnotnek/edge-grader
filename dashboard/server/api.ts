// Read-only /api routes on the Vite dev server. Every request re-reads the repo's files,
// so a page reload after backup_board.sh / fetch_results.sh shows the new data.
import fs from 'node:fs'
import path from 'node:path'
import type { Plugin } from 'vite'
import { loadBoards } from './boards.ts'
import { loadCascade } from './cascade.ts'
import { docFile, listDocs, readDoc } from './docs.ts'
import { loadFinetune } from './finetune.ts'
import { parsePlan } from './plan.ts'
import { loadReference } from './reference.ts'
import { loadRuns } from './runs.ts'
import { expectedStatus } from './status.ts'
import type { BoardPair, Expected, ExpectedStatus, Overview, Plan, Run } from './types.ts'

const NEWEST = 8
const LOCAL_NAMES = new Set(['127.0.0.1', 'localhost', '[::1]'])

/**
 * The API answers only requests addressed to this machine by name. A page on another site can
 * point its own hostname at 127.0.0.1 (DNS rebinding); its requests then carry that hostname.
 */
export function isLocalHost(host: string | undefined): boolean {
  if (!host) return false
  const name = host.startsWith('[') ? host.slice(0, host.indexOf(']') + 1) : host.split(':')[0]
  return LOCAL_NAMES.has(name.toLowerCase())
}

export function currentWeek(timeline: Plan['timeline'], today: string): Plan['timeline'][number] | undefined {
  return timeline.find((w) => w.start && w.end && w.start <= today && today <= w.end)
}

export function localDate(d = new Date()): string {
  const p = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`
}

const withoutSummary = ({ summary: _s, ...r }: Run): Run => r

function loadPlan(repo: string): Plan {
  return parsePlan(fs.readFileSync(path.join(repo, 'docs/plan.md'), 'utf8'))
}

export function loadExpected(repo: string): Expected[] {
  const file = path.join(repo, 'dashboard/expected-runs.json')
  if (!fs.existsSync(file)) return []
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8')) as Expected[]
  } catch (e) {
    throw new Error(`dashboard/expected-runs.json isn't valid JSON (${(e as Error).message}). Fix it and reload.`)
  }
}

function loadExpectedStatus(repo: string, runs = loadRuns(repo)): ExpectedStatus[] {
  return expectedStatus(loadExpected(repo), runs, loadBoards(repo)).map((e) =>
    e.run ? { ...e, run: withoutSummary(e.run) } : e,
  )
}

/** Pi vs Jetson pairs from the newest board-check file that has any. */
export function loadPairs(repo: string): { file: string | null; pairs: BoardPair[] } {
  const dir = path.join(repo, 'results/estimates')
  const files = fs.existsSync(dir) ? fs.readdirSync(dir).filter((f) => f.startsWith('board-check-')).sort().reverse() : []
  for (const f of files) {
    let d: { board_pairs?: BoardPair[] }
    try {
      d = JSON.parse(fs.readFileSync(path.join(dir, f), 'utf8'))
    } catch {
      continue
    }
    if (d.board_pairs?.length) return { file: `results/estimates/${f}`, pairs: d.board_pairs }
  }
  return { file: null, pairs: [] }
}

export function buildOverview(repo: string, today: string): Overview {
  const runs = loadRuns(repo)
  const plan = loadPlan(repo)
  const boards = loadBoards(repo)
  const waiting = loadExpectedStatus(repo, runs).filter((e) => e.status !== 'done')
  const count = <T>(xs: T[], key: (x: T) => string) =>
    xs.reduce<Record<string, number>>((acc, x) => ({ ...acc, [key(x)]: (acc[key(x)] ?? 0) + 1 }), {})
  const items = plan.sections.flatMap((s) => s.items)
  return {
    today,
    currentWeek: currentWeek(plan.timeline, today),
    runsByBoard: count(runs, (r) => r.board),
    owedByBoard: { pi5: 0, jetson: 0, ...count(waiting, (w) => w.board) },
    lastBackup: Object.fromEntries(boards.map((b) => [b.board, b.lastBackup ?? null])),
    waiting,
    newest: runs.slice(0, NEWEST).map(withoutSummary),
    planProgress: { done: items.filter((i) => i.done).length, total: items.length },
    errors: runs.filter((r) => r.error).length,
  }
}

function route(repo: string, url: string): unknown {
  const p = decodeURIComponent(url.split('?')[0]).replace(/\/+$/, '')
  if (p === '/runs') return loadRuns(repo).map(withoutSummary)
  if (p.startsWith('/runs/')) {
    const id = p.slice('/runs/'.length)
    const run = loadRuns(repo).find((r) => r.id === id)
    if (!run) throw Object.assign(new Error(`no run ${id}`), { status: 404 })
    return run
  }
  if (p === '/plan') return loadPlan(repo)
  if (p === '/boards') return loadBoards(repo)
  if (p === '/expected') return loadExpectedStatus(repo)
  if (p === '/overview') return buildOverview(repo, localDate())
  if (p === '/pairs') return loadPairs(repo)
  if (p === '/reference') return loadReference(repo)
  if (p === '/finetune') return loadFinetune(repo)
  if (p === '/cascade') return loadCascade(repo)
  if (p === '/docs') return listDocs(repo)
  if (p.startsWith('/docs/')) return { name: p.slice('/docs/'.length), markdown: readDoc(repo, p.slice('/docs/'.length)) }
  throw Object.assign(new Error(`no route ${p}`), { status: 404 })
}

export function apiPlugin(repo: string): Plugin {
  return {
    name: 'edge-grader-api',
    configureServer(server) {
      server.middlewares.use('/api', (req, res) => {
        res.setHeader('Content-Type', 'application/json')
        if (!isLocalHost(req.headers.host)) {
          res.statusCode = 403
          res.end(JSON.stringify({ error: 'local requests only' }))
          return
        }
        if (req.method !== 'GET') {
          res.statusCode = 405
          res.end(JSON.stringify({ error: 'read-only' }))
          return
        }
        try {
          // Figures go out as files, not JSON.
          const p = decodeURIComponent((req.url ?? '/').split('?')[0])
          if (p.startsWith('/doc-file/')) {
            const f = docFile(repo, p.slice('/doc-file/'.length))
            res.setHeader('Content-Type', f.type)
            res.end(f.body)
            return
          }
          res.end(JSON.stringify(route(repo, req.url ?? '/')))
        } catch (e) {
          res.statusCode = (e as { status?: number }).status ?? 500
          res.end(JSON.stringify({ error: (e as Error).message }))
        }
      })
    },
  }
}
