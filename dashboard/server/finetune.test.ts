import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { describe, expect, test } from 'vitest'
import { loadFinetune } from './finetune.ts'

const REPO = path.resolve(import.meta.dirname, '../..')

describe('loadFinetune on the real repo', () => {
  const ft = loadFinetune(REPO)
  test('one entry per training summary, never the teacher-reasons files', () => {
    const files = fs.readdirSync(path.join(REPO, 'results/finetune')).filter((f) => f.startsWith('finetune-'))
    expect(ft.adapters.map((a) => a.file.split('/').pop()).sort()).toEqual(files.sort())
  })
  test('epochs keep train loss and validation accuracy as stored', () => {
    const a = ft.adapters.find((x) => x.file.endsWith('finetune-e2b-20261027-111124.json'))!
    expect(a).toMatchObject({ model: 'e2b', reply: 'label', chosenEpoch: 2, rank: 16, lr: 0.0002, trainQuestions: 121 })
    expect(a.epochs[1]).toEqual({ epoch: 2, trainLoss: 0.08749613761901856, valAccuracy: 0.7875751503006012 })
  })
  test('stage 3b adapters name their reply format', () => {
    expect(ft.adapters.find((x) => x.file.includes('reason-first'))?.reply).toBe('reason-first')
  })
  test('the newest stage 3b comparison and its gate', () => {
    expect(ft.compare?.file).toBe('results/kaggle/compare-reason-ft-20261102-200813.json')
    expect(ft.compare?.passed).toBe(true)
    expect(ft.compare?.wins).toHaveLength(2)
    expect(ft.compare?.tests[0]).toMatchObject({ adapter: 'e2b-kaggle-t4-ft-rf', dataset: 'beetle', diff: 0.1139 })
  })
})

describe('loadFinetune with nothing there', () => {
  test('no folder gives no adapters and no comparison', () => {
    expect(loadFinetune(fs.mkdtempSync(path.join(os.tmpdir(), 'eg-dash-')))).toEqual({ adapters: [], errors: [] })
  })
  test('a broken summary is listed as an error, the rest still load', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'eg-dash-'))
    fs.mkdirSync(path.join(root, 'results/finetune'), { recursive: true })
    fs.writeFileSync(path.join(root, 'results/finetune/finetune-e2b-20261101-000000.json'), '{nope')
    const ft = loadFinetune(root)
    expect(ft.adapters).toEqual([])
    expect(ft.errors[0]).toMatch(/finetune-e2b-20261101-000000\.json/)
  })
})
