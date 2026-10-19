import fs from 'node:fs'
import path from 'node:path'
import { describe, expect, test } from 'vitest'
import { parsePlan } from './plan.ts'

const md = fs.readFileSync(path.resolve(import.meta.dirname, '../../docs/plan.md'), 'utf8')
const plan = parsePlan(md)

describe('parsePlan on docs/plan.md', () => {
  test('reads the last-updated date', () => {
    expect(plan.updated).toBe('2026-10-08')
  })
  test('timeline has one row per table row, with ISO week bounds', () => {
    expect(plan.timeline).toHaveLength(9)
    const w5 = plan.timeline.find((r) => r.dates === 'Oct 12 – 18')!
    expect(w5.start).toBe('2026-10-12')
    expect(w5.end).toBe('2026-10-18')
    const w3 = plan.timeline.find((r) => r.week === '3')!
    expect([w3.start, w3.end]).toEqual(['2026-09-28', '2026-10-04'])
  })
  test('Phase 0 is fully ticked', () => {
    const p0 = plan.sections.find((s) => s.title.startsWith('Phase 0'))!
    expect(p0.items).toHaveLength(7)
    expect(p0.items.every((i) => i.done)).toBe(true)
  })
  test('Phase B has three items, B1 and B2 done', () => {
    const pb = plan.sections.find((s) => s.title.startsWith('Phase B'))!
    expect(pb.items.map((i) => i.done)).toEqual([true, true, false])
  })
  test('bold sub-headings group the items under them', () => {
    const s2 = plan.sections.find((s) => s.title.startsWith('Stage 2'))!
    expect(s2.items[0].group).toBe('SciEntsBank')
    expect(s2.items.at(-1)!.group).toBe('MMLU-Pro')
  })
  test('change log rows keep their three columns', () => {
    expect(plan.changelog.length).toBeGreaterThan(20)
    expect(plan.changelog[0]).toEqual({ date: '2026-09-29', change: 'Plan created; stage 1 Kaggle runs done', why: '—' })
  })
})

describe('parsePlan on a small file', () => {
  test('a section with no checkboxes still appears, with no items', () => {
    const p = parsePlan('## Steps\n\n### Stage 3: dropped\nReplaced.\n\n### Stage 4\n- [ ] one\n')
    expect(p.sections.map((s) => [s.title, s.items.length])).toEqual([['Stage 3: dropped', 0], ['Stage 4', 1]])
  })
})
