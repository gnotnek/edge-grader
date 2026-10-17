import type { Benchmark, Board, Condition, Model } from '@server/types'

export const BOARD_LABEL: Record<Board, string> = { jetson: 'Jetson', pi5: 'Pi 5', kaggle: 'Kaggle T4' }
export const MODEL_LABEL: Record<Model, string> = { e2b: 'E2B', e4b: 'E4B' }
export const BENCH_LABEL: Record<Benchmark, string> = {
  scientsbank: 'SciEntsBank', gsm: 'GSM', news: 'News', mmlupro: 'MMLU-Pro',
}
export const CONDITION_LABEL: Record<Condition, string> = {
  baseline: 'Baseline', 'rag-en': 'RAG-en', 'rag-id': 'RAG-id', think: 'Thinking on',
  lg: 'little-gemma', 'lg-reuse': 'little-gemma + reuse', swafull: 'SWA full',
}
export const CONDITION_ORDER: Condition[] = ['baseline', 'think', 'rag-en', 'rag-id', 'swafull', 'lg', 'lg-reuse']
export const BENCH_ORDER: Benchmark[] = ['scientsbank', 'gsm', 'news', 'mmlupro']
export const BOARD_ORDER: Board[] = ['jetson', 'pi5', 'kaggle']

export function pct(v: number | null | undefined, digits = 1): string {
  return v == null ? '—' : `${(v * 100).toFixed(digits)}%`
}

export function fixed(v: number | null | undefined, digits = 3): string {
  return v == null ? '—' : v.toFixed(digits)
}

export function count(v: number | null | undefined): string {
  return v == null ? '—' : v.toLocaleString('en-US')
}

/** Percentage points between two accuracies, signed. */
export function deltaPts(a: number | null | undefined, b: number | null | undefined): number | null {
  return a == null || b == null ? null : (b - a) * 100
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

/** "20261011-232209" → "11 Oct, 23:22" (the board's local clock, as stamped). */
export function stampLabel(stamp: string): string {
  const m = stamp.match(/^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})/)
  if (!m) return stamp
  return `${Number(m[3])} ${MONTHS[Number(m[2]) - 1]}, ${m[4]}:${m[5]}`
}

/** Today in the Mac's own time zone (toISOString would give the UTC date). */
export function localToday(d = new Date()): string {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
}

export function dateLabel(iso: string | null | undefined): string {
  if (!iso) return '—'
  const d = new Date(iso)
  return `${d.getDate()} ${MONTHS[d.getMonth()]}, ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`
}

export function ago(iso: string | null | undefined, now = new Date()): string {
  if (!iso) return 'never'
  const mins = Math.round((now.getTime() - new Date(iso).getTime()) / 60000)
  if (mins < 60) return `${mins} min ago`
  const hours = Math.round(mins / 60)
  if (hours < 36) return `${hours} h ago`
  return `${Math.round(hours / 24)} days ago`
}

export function duration(from?: string | null, to?: string | null): string {
  if (!from || !to) return '—'
  const s = Math.round((new Date(to).getTime() - new Date(from).getTime()) / 1000)
  const h = Math.floor(s / 3600)
  const m = Math.round((s % 3600) / 60)
  return h ? `${h} h ${m} min` : `${m} min`
}

/** GSM-Symbolic runs differ in variant count by board (i1, i10); the plate map treats them as one set. */
export function splitGroup(split: string): string {
  return split.startsWith('symbolic') ? 'symbolic' : split
}

/** Plan text is markdown: drop link targets, emphasis markers and the emoji ticks for plain display. */
export function plainMarkdown(s: string): string {
  return s
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/\s*\p{Extended_Pictographic}\uFE0F?/gu, '')
    .trim()
}
