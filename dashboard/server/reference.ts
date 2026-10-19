// agentic-edge's MMLU-Pro board runs, as copied into results/reference/. Read-only, and kept
// apart from this project's runs: CLAUDE.md says comparable, not mixed.
import fs from 'node:fs'
import path from 'node:path'
import type { Condition, Model, Reference, ReferenceRow } from './types.ts'

const FILE = 'results/reference/agentic-edge-mmlupro-variations.json'
const MODELS: Model[] = ['e2b', 'e4b']
const BOARDS = ['pi5', 'jetson'] as const

type Obj = Record<string, any>

const num = (v: unknown): number | undefined => (typeof v === 'number' ? v : undefined)

function row(
  section: string, label: string, board: ReferenceRow['board'], model: Model, condition: Condition | undefined,
  src: Obj | undefined, extra: Partial<ReferenceRow> = {},
): ReferenceRow | null {
  if (!src) return null
  const r: ReferenceRow = { section, label, benchmark: 'mmlupro', board, model, ...extra }
  if (condition) r.condition = condition
  const fields: [keyof ReferenceRow, unknown][] = [
    ['accuracy', src.accuracy], ['mcnemarP', src.mcnemar_p ?? src.mcnemar_p_vs_s1], ['energyWh', src.energy_wh],
    ['wallMin', src.wall_min], ['jPerQuestion', src.energy_j_per_question],
  ]
  for (const [k, v] of fields) if (num(v) !== undefined) (r as Obj)[k] = v
  const note = src.vs_s3 ?? src.vs_llama_cpp
  if (typeof note === 'string') r.note = note
  return r
}

export function loadReference(repo: string): Reference {
  const file = path.join(repo, FILE)
  if (!fs.existsSync(file)) return { source: null, rows: [] }
  let d: Obj
  try {
    d = JSON.parse(fs.readFileSync(file, 'utf8'))
  } catch (e) {
    return { source: FILE, rows: [], error: `${FILE}: ${(e as Error).message}` }
  }

  const rows: (ReferenceRow | null)[] = []
  const base = d.baseline_pi_vs_jetson ?? {}
  const baseAcc = (m: Model, b: string) => num(base[m]?.[b])
  for (const m of MODELS)
    for (const b of BOARDS) {
      const acc = baseAcc(m, b)
      if (acc === undefined) continue
      rows.push(row('baseline', 'Baseline (llama.cpp)', b, m, 'baseline', { accuracy: acc, ...(base.cost?.[m]?.[b] ?? {}) }))
    }

  const think = d.thinking_on_budget_320 ?? {}
  for (const m of MODELS)
    for (const b of BOARDS)
      rows.push(row('think', 'Thinking on, 320-token budget', b, m, 'think', think[m]?.[b], { baseline: num(think[m]?.[b]?.baseline) }))

  const lg = d.engine_little_gemma_jetson ?? {}
  for (const m of MODELS) {
    rows.push(row('lg', 'little-gemma', 'jetson', m, 'lg', lg.thinking_off?.[m], { baseline: num(lg.thinking_off?.[m]?.llama_cpp) }))
    rows.push(row('lg-think', 'little-gemma, thinking on', 'jetson', m, undefined, lg.thinking_on?.[m], { baseline: num(lg.thinking_on?.[m]?.llama_cpp) }))
  }

  const pr = d.prefix_reuse_and_ngram_jetson ?? {}
  for (const m of MODELS) {
    rows.push(row('lg-reuse', 'little-gemma + prefix reuse', 'jetson', m, 'lg-reuse', pr[m]?.reuse))
    rows.push(row('lg-ngram', 'little-gemma + n-gram drafts', 'jetson', m, undefined, pr[m]?.ngram))
    rows.push(row('lg-full', 'little-gemma + reuse + n-gram', 'jetson', m, undefined, pr[m]?.full))
    rows.push(row('ngram', 'llama.cpp n-gram drafts', 'jetson', m, undefined, pr.llama_cpp_ngram_s10_n3m3?.[m], { baseline: baseAcc(m, 'jetson') }))
  }

  const out: Reference = { source: FILE, rows: rows.filter((r): r is ReferenceRow => r !== null) }
  if (typeof d.copied === 'string') out.copied = d.copied
  return out
}
