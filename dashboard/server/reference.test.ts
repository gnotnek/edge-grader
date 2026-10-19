import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { describe, expect, test } from 'vitest'
import { loadReference } from './reference.ts'

const REPO = path.resolve(import.meta.dirname, '../..')
const FILE = 'results/reference/agentic-edge-mmlupro-variations.json'

describe('loadReference on the real repo', () => {
  const ref = loadReference(REPO)
  const row = (section: string, board: string, model: string) =>
    ref.rows.find((r) => r.section === section && r.board === board && r.model === model)

  test('names its source file and the copy date', () => {
    expect(ref.source).toBe(FILE)
    expect(ref.copied).toBe('2026-10-19')
  })
  test('baseline per board, with the per-question cost as copied', () => {
    expect(row('baseline', 'pi5', 'e2b')).toMatchObject({ condition: 'baseline', accuracy: 0.517, jPerQuestion: 770, energyWh: 64.2, wallMin: 552 })
    expect(row('baseline', 'jetson', 'e4b')).toMatchObject({ accuracy: 0.66, jPerQuestion: 654 })
  })
  test('thinking on carries its baseline and McNemar p', () => {
    expect(row('think', 'jetson', 'e4b')).toMatchObject({
      condition: 'think', accuracy: 0.667, baseline: 0.66, mcnemarP: 0.86, jPerQuestion: 810,
    })
  })
  test('little-gemma rows are Jetson only, compared with llama.cpp', () => {
    expect(row('lg', 'jetson', 'e4b')).toMatchObject({ condition: 'lg', accuracy: 0.653, baseline: 0.66, mcnemarP: 0.85 })
    expect(row('lg-think', 'jetson', 'e2b')).toMatchObject({ accuracy: 0.493, baseline: 0.497 })
    expect(ref.rows.some((r) => r.section.startsWith('lg') && r.board === 'pi5')).toBe(false)
  })
  test('prefix reuse has cost but no accuracy of its own', () => {
    const r = row('lg-reuse', 'jetson', 'e2b')!
    expect(r).toMatchObject({ condition: 'lg-reuse', energyWh: 25.5, wallMin: 136 })
    expect(r.accuracy).toBeUndefined()
  })
  test('every row is MMLU-Pro on a board, never Kaggle', () => {
    expect(ref.rows.length).toBeGreaterThan(10)
    expect(ref.rows.every((r) => r.benchmark === 'mmlupro' && (r.board === 'pi5' || r.board === 'jetson'))).toBe(true)
  })
})

describe('loadReference without the file', () => {
  test('no reference file gives no rows and no error', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'eg-dash-'))
    expect(loadReference(root)).toEqual({ source: null, rows: [] })
  })
  test('a broken reference file is an error, not a crash', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'eg-dash-'))
    fs.mkdirSync(path.join(root, 'results/reference'), { recursive: true })
    fs.writeFileSync(path.join(root, FILE), '{nope')
    const ref = loadReference(root)
    expect(ref.rows).toEqual([])
    expect(ref.error).toMatch(/agentic-edge-mmlupro-variations\.json/)
  })
})
