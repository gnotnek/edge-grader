import { describe, expect, test } from 'vitest'
import { deltaPts, localToday, pct, plainMarkdown, splitGroup, stampLabel } from './format'

describe('format', () => {
  test('stampLabel reads the run folder stamp', () => {
    expect(stampLabel('20261011-232209')).toBe('11 Oct, 23:22')
  })
  test('GSM-Symbolic variants share one plate-map well', () => {
    expect(splitGroup('symbolic-i10')).toBe('symbolic')
    expect(splitGroup('symbolic-i1')).toBe('symbolic')
    expect(splitGroup('test_ua')).toBe('test_ua')
  })
  test('plan markdown loses link targets and bold, keeps code', () => {
    expect(plainMarkdown('see [`results/rag.md`](results/rag.md) **dropped**')).toBe('see `results/rag.md` dropped')
  })
  test('plan markdown loses emoji (the plan ticks weeks with them)', () => {
    expect(plainMarkdown('Topic benchmarks done in week 4 ✅')).toBe('Topic benchmarks done in week 4')
    expect(plainMarkdown('Baseline table ✅; cost ✅, then measured')).toBe('Baseline table; cost, then measured')
  })
  test('deltaPts is signed percentage points, null when either side is missing', () => {
    expect(deltaPts(0.92, 0.88)).toBeCloseTo(-4.0)
    expect(deltaPts(null, 0.5)).toBeNull()
  })
  test('pct shows a dash for missing values', () => {
    expect(pct(0.7306)).toBe('73.1%')
    expect(pct(null)).toBe('—')
  })
  test('localToday uses the local calendar date, not UTC', () => {
    expect(localToday(new Date(2026, 9, 17, 1, 30))).toBe('2026-10-17')
  })
})
