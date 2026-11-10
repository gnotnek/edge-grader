// Grading scores the way edge_grader.metrics.score computes them (sklearn: F1 over the gold
// labels, an unparsed reply counted wrong), so the dashboard's 2-way matches the project's.
import fs from 'node:fs'
import path from 'node:path'

export interface Scores { n: number; accuracy: number; macro_f1: number; weighted_f1: number; unparsed: number }

const r4 = (x: number) => Math.round(x * 1e4) / 1e4

export function score(gold: string[], pred: (string | null)[]): Scores {
  const p = pred.map((x) => x ?? '__unparsed__')
  const labels = [...new Set(gold)].sort()
  const f1 = labels.map((l) => {
    const tp = gold.filter((g, i) => g === l && p[i] === l).length
    const predicted = p.filter((x) => x === l).length
    const support = gold.filter((g) => g === l).length
    const prec = predicted ? tp / predicted : 0
    const rec = support ? tp / support : 0
    return { f1: prec + rec ? (2 * prec * rec) / (prec + rec) : 0, support }
  })
  return {
    n: gold.length,
    accuracy: r4(gold.filter((g, i) => g === p[i]).length / gold.length),
    macro_f1: r4(f1.reduce((a, x) => a + x.f1, 0) / f1.length),
    weighted_f1: r4(f1.reduce((a, x) => a + x.f1 * x.support, 0) / gold.length),
    unparsed: p.filter((x) => x === '__unparsed__').length,
  }
}

/** 2-way asks only "is it correct?": contradictory joins incorrect (edge_grader.compare.two). */
export function twoWay(label: string | null): string | null {
  return label === null ? null : label === 'correct' ? 'correct' : 'incorrect'
}

const cache = new Map<string, Scores | null>()

/**
 * 2-way scores from a run's per-answer predictions, raw/<summary name>.jsonl next to the summary.
 * Null when there is no raw file. Cached per file version: the files are large and never edited.
 */
export function twoWayFromRaw(summaryFile: string): Scores | null {
  const raw = path.join(path.dirname(summaryFile), 'raw', path.basename(summaryFile, '.json') + '.jsonl')
  let stat: fs.Stats
  try {
    stat = fs.statSync(raw)
  } catch {
    return null
  }
  const key = `${raw}:${stat.mtimeMs}`
  if (!cache.has(key)) {
    // One row per answer id, the last one winning, as board_check.read_answers reads them.
    const rows = new Map<string, { gold: string; pred: string | null }>()
    for (const line of fs.readFileSync(raw, 'utf8').split('\n'))
      if (line.trim()) {
        const r = JSON.parse(line)
        rows.set(r.id, r)
      }
    const a = [...rows.values()]
    cache.set(key, a.length && a.every((r) => typeof r.gold === 'string')
      ? score(a.map((r) => twoWay(r.gold)!), a.map((r) => twoWay(r.pred ?? null)))
      : null)
  }
  return cache.get(key)!
}
