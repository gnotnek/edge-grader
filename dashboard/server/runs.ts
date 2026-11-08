// Every run this project has: board run folders (results/board/, and backups/<board>/ for the
// ones not fetched yet) and Kaggle summary files (results/kaggle/). Read-only.
import fs from 'node:fs'
import path from 'node:path'
import type { Benchmark, Board, Condition, Model, Run, RunKey, VsKaggle } from './types.ts'

const STAMP = /^\d{8}-\d{6}$/
const MODELS = new Set(['e2b', 'e4b'])
// Copied as stored; the pages only round them for display.
const METRIC_KEYS = [
  'n', 'accuracy', 'macro_f1', 'weighted_f1', 'strict_accuracy', 'unparsed', 'errors',
  'no_answer', 'no_number', 'hit_token_cap', 'answered', 'median_latency_s', 'wall_s',
  'prompt_tokens', 'completion_tokens', 'median_completion_tokens', 'qwk', 'adjacent',
]

function splitStamp(name: string): { tokens: string[]; stamp: string } | null {
  const parts = name.split('-')
  if (parts.length < 3) return null
  const stamp = parts.slice(-2).join('-')
  if (!STAMP.test(stamp)) return null
  return { tokens: parts.slice(0, -2), stamp }
}

/** A board run folder, as board_grade.sh names it. Null for pilots, interrupted runs and the like. */
export function parseRunFolder(name: string): RunKey | null {
  const s = splitStamp(name)
  if (!s) return null
  const { tokens, stamp } = s
  const m = tokens.findIndex((t) => MODELS.has(t))
  if (m < 0) return null
  const board = tokens[m + 1]
  if (board !== 'pi5' && board !== 'jetson') return null
  const split = tokens.slice(m + 2).join('-')
  if (!split) return null
  const pre = tokens.slice(0, m).join('-')

  let benchmark: Benchmark = 'scientsbank'
  let condition: Condition = 'baseline'
  const [lead, ...rest] = pre.split('-')
  if (lead === 'gsm' || lead === 'news' || lead === 'mmlupro') {
    // The other benchmarks lead with their name, then an optional rag-.
    if (rest.length > 1 || (rest.length === 1 && rest[0] !== 'rag')) return null
    benchmark = lead
    if (rest[0] === 'rag') condition = 'rag-en'
  } else if (pre !== '') {
    // SciEntsBank prefixes stack as rag-<variant>-think-; the outermost one names the condition.
    const first = pre.startsWith('lg-reuse') ? 'lg-reuse' : pre.split('-')[0]
    const known: Record<string, Condition> = {
      rag: 'rag-en', think: 'think', lg: 'lg', 'lg-reuse': 'lg-reuse', swafull: 'swafull',
    }
    if (!(first in known)) return null
    condition = known[first]
  }
  return { condition, benchmark, model: tokens[m] as Model, board, split, stamp }
}

/** A Kaggle or DGX Spark summary file name. Null for judge and comparison files and anything unrecognised. */
export function parseKaggleFile(file: string): RunKey | null {
  if (!file.endsWith('.json') || file.includes('.judged-')) return null
  const s = splitStamp(file.slice(0, -'.json'.length))
  if (!s) return null
  const { tokens, stamp } = s
  const m = tokens.findIndex((t) => MODELS.has(t))
  const device = `${tokens[m + 1]}-${tokens[m + 2]}`
  const board: Board | undefined = ({ 'kaggle-t4': 'kaggle', 'spark-gb10': 'spark' } as const)[device as 'kaggle-t4']
  if (m < 0 || !board) return null
  const pre = tokens.slice(0, m)
  const post = tokens.slice(m + 3) // after <model>-kaggle-t4 or <model>-spark-gb10
  const rag = (): Condition => (post.includes('id') ? 'rag-id' : 'rag-en')
  // A LoRA adapter on the baseline prompt: -ft, and -ft-rf / -ft-lf for the reply formats of stage 3b.
  // A -beetle-saf train tag marks an adapter trained on Beetle+SAF instead of SciEntsBank.
  const base = (): Condition => {
    if (!post.includes('ft')) return 'baseline'
    const reply = post.includes('rf') ? 'ft-rf' : post.includes('lf') ? 'ft-lf' : 'ft'
    return (post.includes('beetle') && post.includes('saf') ? `${reply}-bs` : reply) as Condition
  }
  const key = (benchmark: Benchmark, condition: Condition, split: string): RunKey => ({
    condition, benchmark, model: tokens[m] as Model, board, split, stamp,
  })

  switch (pre[0]) {
    case 'baseline':
    case 'rag': {
      const split = post.find((t) => t.startsWith('test_'))
      if (!split) return null
      return key('scientsbank', pre[0] === 'rag' ? rag() : base(), split)
    }
    case 'beetle':
    case 'saf': {
      const split = post.find((t) => t.startsWith('test_'))
      if (!split || pre[1] !== 'baseline') return null
      return key(pre[0], base(), split)
    }
    case 'asap2':
      if (pre[1] !== 'baseline' && pre[1] !== 'think') return null
      return key('asap2', pre[1] === 'think' ? 'think' : base(), post[post.length - 1] ?? '')
    case 'mmlupro':
      return key('mmlupro', pre[1] === 'rag' ? rag() : base(), post[post.length - 1] ?? '')
    case 'gsm':
      return pre[1] === 'rag' ? key('gsm', rag(), pre[2] ?? '') : key('gsm', base(), pre[1] ?? '')
    case 'news':
      return key('news', pre[1] === 'rag' ? rag() : base(), 'test')
    default:
      return null
  }
}

function readJson(file: string): unknown {
  return JSON.parse(fs.readFileSync(file, 'utf8'))
}

function listDir(dir: string): string[] {
  try {
    return fs.readdirSync(dir).sort()
  } catch {
    return []
  }
}

function pickMetrics(summary: unknown): Record<string, number | null> {
  const metrics = (summary as { metrics?: Record<string, unknown> })?.metrics ?? {}
  const out: Record<string, number | null> = {}
  for (const k of METRIC_KEYS) {
    const v = metrics[k]
    if (typeof v === 'number') out[k] = v
  }
  // The 3-way grading runs also score the same answers 2-way (correct vs not).
  const twoWay = (metrics.two_way as { accuracy?: unknown } | undefined)?.accuracy
  if (typeof twoWay === 'number') out.two_way_accuracy = twoWay
  return out
}

/** The judge's accuracy for a news run, or null when it hasn't been judged. */
function judgedAccuracy(file: string | undefined): number | null {
  if (!file) return null
  const m = (readJson(file) as { metrics?: { accuracy?: unknown } }).metrics
  return typeof m?.accuracy === 'number' ? m.accuracy : null
}

function fillFromSummary(run: Run, summaryFile: string | undefined, judgedFile: string | undefined) {
  if (!summaryFile) return
  try {
    const summary = readJson(summaryFile)
    run.summary = summary
    run.metrics = pickMetrics(summary)
    const time = (summary as { time?: unknown }).time
    if (typeof time === 'string') run.time = time
    // News has no accuracy of its own: the judge grades the answers.
    if (run.benchmark === 'news') run.metrics.accuracy = judgedAccuracy(judgedFile)
  } catch (e) {
    run.error = `${path.basename(summaryFile)}: ${(e as Error).message}`
  }
}

/** One run per folder, except a folder with several summaries (MMLU-Pro s1-s3): one per summary. */
function boardRuns(repo: string, dir: string, key: RunKey, location: Run['location']): Run[] {
  const id = path.basename(dir)
  const files = listDir(dir).filter((f) => f.endsWith('.json'))
  const summaries = files.filter((f) => !f.includes('.judged-'))
  const judged = files.find((f) => f.includes('.judged-'))
  if (summaries.length <= 1) {
    const run: Run = { ...key, id, location, source: path.relative(repo, dir), metrics: {} }
    fillFromSummary(run, summaries[0] && path.join(dir, summaries[0]), judged && path.join(dir, judged))
    return [run]
  }
  return summaries.map((f) => {
    const file = path.join(dir, f)
    const run: Run = { ...key, id: `${id}/${f}`, location, source: path.relative(repo, file), metrics: {} }
    fillFromSummary(run, file, undefined)
    const s = run.summary as { subset?: unknown; split?: unknown } | undefined
    const split = s?.subset ?? s?.split
    if (typeof split === 'string') {
      run.split = split
      run.id = `${id}/${split}`
    }
    return run
  })
}

interface CheckEntry {
  energy?: Run['energy']
  vsKaggle?: VsKaggle
}

/** Per run, what the newest board-check file that covers it says. */
function boardChecks(repo: string): Map<string, CheckEntry> {
  const dir = path.join(repo, 'results/estimates')
  const out = new Map<string, CheckEntry>()
  for (const f of listDir(dir).filter((f) => f.startsWith('board-check-') && f.endsWith('.json'))) {
    let d: { runs?: Record<string, unknown>[] }
    try {
      d = readJson(path.join(dir, f)) as typeof d
    } catch {
      continue
    }
    for (const r of d.runs ?? []) {
      const entry: CheckEntry = {}
      if (typeof r.j_per_answer === 'number')
        entry.energy = { jPerAnswer: r.j_per_answer, meanW: r.mean_w as number }
      const vk = r.vs_kaggle as Record<string, any> | undefined
      if (vk && typeof vk.kaggle_accuracy === 'number')
        entry.vsKaggle = {
          kaggleRun: vk.kaggle_run, kaggleAccuracy: vk.kaggle_accuracy,
          boardAccuracy: vk.board_accuracy_paired, n: vk.n, sameGrade: vk.same_grade,
          p: vk.mcnemar?.p,
        }
      out.set(r.run as string, entry)
    }
  }
  return out
}

export function loadRuns(repo: string): Run[] {
  const runs: Run[] = []
  const seen = new Set<string>()

  const boardDirs: [string, Run['location']][] = [[path.join(repo, 'results/board'), 'results']]
  for (const b of listDir(path.join(repo, 'backups')))
    boardDirs.push([path.join(repo, 'backups', b, 'results/board'), 'backup'])
  for (const [dir, location] of boardDirs) {
    for (const name of listDir(dir)) {
      const key = parseRunFolder(name)
      if (!key || seen.has(name)) continue
      if (!fs.statSync(path.join(dir, name)).isDirectory()) continue
      seen.add(name)
      runs.push(...boardRuns(repo, path.join(dir, name), key, location))
    }
  }

  for (const sub of ['results/kaggle', 'results/spark']) {
    const kdir = path.join(repo, sub)
    const kfiles = listDir(kdir)
    for (const f of kfiles) {
      const key = parseKaggleFile(f)
      if (!key) continue
      const run: Run = { ...key, id: f, location: 'results', source: `${sub}/${f}`, metrics: {} }
      const judged = kfiles.find((g) => g.startsWith(f.slice(0, -'.json'.length) + '.judged-'))
      fillFromSummary(run, path.join(kdir, f), judged && path.join(kdir, judged))
      runs.push(run)
    }
  }

  const checks = boardChecks(repo)
  for (const run of runs) Object.assign(run, checks.get(run.id) ?? {})

  return runs.sort((a, b) => b.stamp.localeCompare(a.stamp))
}

export const BOARDS: Board[] = ['pi5', 'jetson', 'kaggle', 'spark']
