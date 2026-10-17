import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { describe, expect, test } from 'vitest'
import { buildOverview, currentWeek, isLocalHost, loadExpected, loadPairs } from './api.ts'
import { parsePlan } from './plan.ts'

const REPO = path.resolve(import.meta.dirname, '../..')

describe('currentWeek', () => {
  const md = '## Timeline\n| Week | Dates | Stage | Goal | Checkpoint |\n|---|---|---|---|---|\n| 5 | Oct 12 – 18 | A | g | c |\n| 6 | Oct 19 – 25 | B | g | c |\n'
  const { timeline } = parsePlan(md)
  test('the row whose dates contain today', () => {
    expect(currentWeek(timeline, '2026-10-17')?.week).toBe('5')
    expect(currentWeek(timeline, '2026-10-19')?.week).toBe('6')
  })
  test('undefined outside the timeline', () => {
    expect(currentWeek(timeline, '2026-12-25')).toBeUndefined()
  })
})

describe('buildOverview on the real repo', () => {
  const o = buildOverview(REPO, '2026-10-17')
  test('counts runs per board', () => {
    expect(o.runsByBoard.pi5).toBeGreaterThanOrEqual(14)
    expect(o.runsByBoard.jetson).toBeGreaterThanOrEqual(24)
  })
  test('the Pi owes its four B1 thinking runs until they arrive', () => {
    const piWaiting = o.waiting.filter((w) => w.board === 'pi5')
    expect(o.owedByBoard.pi5).toBe(piWaiting.length)
    expect(piWaiting.every((w) => w.status !== 'done')).toBe(true)
  })
  test('the Pi has no backup date; the Jetson has one', () => {
    expect(o.lastBackup.pi5).toBeNull()
    expect(o.lastBackup.jetson).toBeTruthy()
  })
  test('week 5 on 2026-10-17', () => {
    expect(o.currentWeek?.week).toBe('5')
  })
})

describe('hand-edited and odd files', () => {
  function repoWith(files: Record<string, string>): string {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'eg-dash-'))
    for (const [f, c] of Object.entries(files)) {
      fs.mkdirSync(path.dirname(path.join(root, f)), { recursive: true })
      fs.writeFileSync(path.join(root, f), c)
    }
    return root
  }
  test('a typo in expected-runs.json names the file in the error', () => {
    const root = repoWith({ 'dashboard/expected-runs.json': '[{"board": "pi5",]' })
    expect(() => loadExpected(root)).toThrow(/expected-runs\.json/)
  })
  test('a broken board-check file is skipped when looking for pairs', () => {
    const pair = { first: 'a', second: 'b', n: 1, same_grade: 1, first_accuracy: 1, second_accuracy: 1, mcnemar: { p: 1 } }
    const root = repoWith({
      'results/estimates/board-check-20261001-000000.json': JSON.stringify({ runs: [], board_pairs: [pair] }),
      'results/estimates/board-check-20261020-000000.json': '{broken',
    })
    expect(loadPairs(root).pairs).toHaveLength(1)
  })
})

describe('isLocalHost (DNS-rebinding guard)', () => {
  test('the dev server\'s own addresses pass', () => {
    expect(isLocalHost('127.0.0.1:5173')).toBe(true)
    expect(isLocalHost('localhost:5173')).toBe(true)
    expect(isLocalHost('[::1]:5173')).toBe(true)
  })
  test('any other name is refused, even if it resolves to 127.0.0.1', () => {
    expect(isLocalHost('evil.example:5173')).toBe(false)
    expect(isLocalHost('127.0.0.1.evil.example:5173')).toBe(false)
    expect(isLocalHost(undefined)).toBe(false)
  })
})
