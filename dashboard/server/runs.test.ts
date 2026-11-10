import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { describe, expect, test } from 'vitest'
import { loadRuns, parseKaggleFile, parseRunFolder } from './runs.ts'

const REPO = path.resolve(import.meta.dirname, '../..')

describe('parseRunFolder', () => {
  test('thinking run', () => {
    expect(parseRunFolder('think-e2b-jetson-test_ua-20261011-232209')).toEqual({
      condition: 'think', benchmark: 'scientsbank', model: 'e2b', board: 'jetson',
      split: 'test_ua', stamp: '20261011-232209',
    })
  })
  test('plain SciEntsBank baseline', () => {
    expect(parseRunFolder('e4b-pi5-test_uq-20261009-130818')).toMatchObject({
      condition: 'baseline', benchmark: 'scientsbank', board: 'pi5', split: 'test_uq',
    })
  })
  test('lg-reuse is its own condition, not lg', () => {
    expect(parseRunFolder('lg-reuse-e4b-jetson-test_ua-20261013-154741')?.condition).toBe('lg-reuse')
    expect(parseRunFolder('lg-e4b-jetson-test_ua-20261013-153753')?.condition).toBe('lg')
  })
  test('gsm set keeps its variant suffix', () => {
    expect(parseRunFolder('gsm-e2b-jetson-symbolic-i10-20261009-020450')).toMatchObject({
      benchmark: 'gsm', condition: 'baseline', split: 'symbolic-i10',
    })
  })
  test('news', () => {
    expect(parseRunFolder('news-e4b-pi5-test-20261010-011719')).toMatchObject({
      benchmark: 'news', split: 'test', board: 'pi5',
    })
  })
  test('gsm and news RAG board runs', () => {
    expect(parseRunFolder('gsm-rag-e2b-jetson-symbolic-i10-20261020-010000')).toMatchObject({
      benchmark: 'gsm', condition: 'rag-en', split: 'symbolic-i10',
    })
    expect(parseRunFolder('news-rag-e4b-jetson-test-20261020-010000')).toMatchObject({
      benchmark: 'news', condition: 'rag-en', split: 'test',
    })
  })
  test('board RAG and MMLU-Pro folders', () => {
    expect(parseRunFolder('gsm-rag-e2b-jetson-gsm8k-20261020-043935')).toMatchObject({ benchmark: 'gsm', condition: 'rag-en', split: 'gsm8k' })
    expect(parseRunFolder('news-rag-e4b-jetson-test-20261020-022520')).toMatchObject({ benchmark: 'news', condition: 'rag-en', split: 'test' })
    expect(parseRunFolder('mmlupro-e2b-jetson-all-20261021-145638')).toMatchObject({ benchmark: 'mmlupro', condition: 'baseline', split: 'all' })
  })
  test('interrupted and pilot folders are not runs', () => {
    expect(parseRunFolder('interrupted-think-e4b-jetson-test_ud-20261013-063811')).toBeNull()
    expect(parseRunFolder('pilot-base')).toBeNull()
  })
})

describe('parseKaggleFile', () => {
  test('SciEntsBank RAG-id', () => {
    expect(parseKaggleFile('rag-e4b-kaggle-t4-id-k3-test_ud-3way-20261005-115314.json')).toEqual({
      condition: 'rag-id', benchmark: 'scientsbank', model: 'e4b', board: 'kaggle',
      split: 'test_ud', stamp: '20261005-115314',
    })
  })
  test('SciEntsBank baseline', () => {
    expect(parseKaggleFile('baseline-e2b-kaggle-t4-test_uq-3way-20260929-102250.json')).toMatchObject({
      condition: 'baseline', benchmark: 'scientsbank', split: 'test_uq',
    })
  })
  test('MMLU-Pro RAG-en subset', () => {
    expect(parseKaggleFile('mmlupro-rag-e2b-kaggle-t4-en-k3-s1-20261006-134022.json')).toMatchObject({
      benchmark: 'mmlupro', condition: 'rag-en', split: 's1',
    })
  })
  test('GSM RAG on symbolic', () => {
    expect(parseKaggleFile('gsm-rag-symbolic-e4b-kaggle-t4-en-k3-20261010-182457.json')).toMatchObject({
      benchmark: 'gsm', condition: 'rag-en', split: 'symbolic',
    })
  })
  test('GSM baseline set, including the 256-token tiny run', () => {
    expect(parseKaggleFile('gsm-gsm8k-e2b-kaggle-t4-20261008-090049.json')).toMatchObject({
      benchmark: 'gsm', condition: 'baseline', split: 'gsm8k',
    })
    expect(parseKaggleFile('gsm-tiny-e4b-kaggle-t4-256tok-20261008-073642.json')).toMatchObject({
      split: 'tiny',
    })
  })
  test('news', () => {
    expect(parseKaggleFile('news-rag-e4b-kaggle-t4-en-k3-20261008-065226.json')).toMatchObject({
      benchmark: 'news', condition: 'rag-en', split: 'test',
    })
  })
  test('fine-tuned adapters: plain, reason first, label first', () => {
    expect(parseKaggleFile('baseline-e2b-kaggle-t4-ft-test_ua-3way-20261028-075708.json')).toMatchObject({
      benchmark: 'scientsbank', condition: 'ft', split: 'test_ua',
    })
    expect(parseKaggleFile('baseline-e2b-kaggle-t4-ft-rf-test_ua-3way-20261102-145733.json')?.condition).toBe('ft-rf')
    expect(parseKaggleFile('beetle-baseline-e2b-kaggle-t4-ft-lf-test_ua-3way-20261102-161006.json')).toMatchObject({
      benchmark: 'beetle', condition: 'ft-lf', split: 'test_ua',
    })
  })
  test('held-out grading sets and the no-harm checks', () => {
    expect(parseKaggleFile('saf-baseline-e4b-kaggle-t4-test_uq-3way-20261030-182605.json')).toMatchObject({
      benchmark: 'saf', condition: 'baseline', split: 'test_uq',
    })
    expect(parseKaggleFile('mmlupro-baseline-e2b-kaggle-t4-ft-s1-20261028-064436.json')).toMatchObject({
      benchmark: 'mmlupro', condition: 'ft', split: 's1',
    })
    expect(parseKaggleFile('gsm-gsm8k-e2b-kaggle-t4-ft-20261028-052054.json')).toMatchObject({
      benchmark: 'gsm', condition: 'ft', split: 'gsm8k',
    })
  })
  test('DGX Spark files parse like Kaggle ones, on their own board', () => {
    expect(parseKaggleFile('baseline-e4b-spark-gb10-ft-test_ud-3way-20261108-055232.json')).toMatchObject({
      board: 'spark', benchmark: 'scientsbank', condition: 'ft', split: 'test_ud', model: 'e4b',
    })
  })
  test('adapters trained on Beetle+SAF are their own conditions', () => {
    expect(parseKaggleFile('baseline-e2b-kaggle-t4-ft-beetle-saf-test_ua-3way-20261108-122403.json')?.condition).toBe('ft-bs')
    expect(parseKaggleFile('baseline-e2b-kaggle-t4-ft-lf-beetle-saf-test_ud-3way-20261108-124141.json')?.condition).toBe('ft-lf-bs')
    expect(parseKaggleFile('baseline-e2b-spark-gb10-ft-rf-beetle-saf-test_uq-3way-20261107-234952.json')).toMatchObject({
      board: 'spark', condition: 'ft-rf-bs',
    })
  })
  test('ASAP 2.0 essays: baseline, thinking and adapters', () => {
    expect(parseKaggleFile('asap2-baseline-e2b-spark-gb10-sample-20261105-141028.json')).toMatchObject({
      board: 'spark', benchmark: 'asap2', condition: 'baseline', split: 'sample',
    })
    expect(parseKaggleFile('asap2-think-e4b-spark-gb10-sample-20261105-175407.json')?.condition).toBe('think')
    expect(parseKaggleFile('asap2-baseline-e2b-spark-gb10-ft-lf-sample-20261105-142248.json')?.condition).toBe('ft-lf')
  })
  test('the comparison file is not a run', () => {
    expect(parseKaggleFile('compare-reason-ft-20261102-200813.json')).toBeNull()
    expect(parseKaggleFile('compare-bar-scientsbank-20261108-135709.json')).toBeNull()
  })
  test('judge files are not runs', () => {
    expect(parseKaggleFile('news-rag-e4b-kaggle-t4-en-k3-20261008-065226.judged-qwen3-14b-q4_k_m.json')).toBeNull()
  })
})

describe('loadRuns on the real repo', () => {
  const runs = loadRuns(REPO)
  test('every fetched board folder becomes at least one run', () => {
    const ids = new Set(runs.filter((r) => (r.board === 'jetson' || r.board === 'pi5') && r.location === 'results').map((r) => r.id.split('/')[0]))
    expect([...ids].sort()).toEqual(fs.readdirSync(path.join(REPO, 'results/board')).sort())
  })
  test('a board MMLU-Pro folder holds one run per subset', () => {
    const subs = runs.filter((r) => r.id.startsWith('mmlupro-e2b-jetson-all-20261021-145638'))
    expect(subs.map((r) => [r.id, r.split, r.benchmark])).toEqual([
      ['mmlupro-e2b-jetson-all-20261021-145638/s1', 's1', 'mmlupro'],
      ['mmlupro-e2b-jetson-all-20261021-145638/s2', 's2', 'mmlupro'],
      ['mmlupro-e2b-jetson-all-20261021-145638/s3', 's3', 'mmlupro'],
    ])
  })
  test('Spark runs are loaded from results/spark, with QWK for ASAP 2.0', () => {
    const r = runs.find((x) => x.id === 'asap2-baseline-e2b-spark-gb10-sample-20261105-141028.json')!
    expect(r).toMatchObject({ board: 'spark', source: 'results/spark/asap2-baseline-e2b-spark-gb10-sample-20261105-141028.json' })
    expect(r.metrics.qwk).toBe(0.5677)
  })
  test('2-way scores come from the raw predictions when the summary has none, as Python computes them', () => {
    // compare-bar-scientsbank-20261109 holds edge_grader's own 2-way numbers for this base run.
    const r = runs.find((x) => x.id === 'baseline-e2b-kaggle-t4-test_uq-3way-20261005-100230.json')!
    expect(r.metrics.two_way_accuracy).toBe(0.7544)
    expect(r.metrics.two_way_macro_f1).toBe(0.7181)
  })
  test('a board run gets its 2-way scores too', () => {
    const r = runs.find((x) => x.id === 'e2b-jetson-test_ua-20260930-224702')!
    expect(r.metrics.two_way_accuracy).toBeGreaterThan(r.metrics.accuracy!)
  })
  test('tuned Kaggle runs are fine-tuned, not baseline', () => {
    const r = runs.find((x) => x.id === 'baseline-e2b-kaggle-t4-ft-rf-test_ua-3way-20261102-145733.json')!
    expect(r.condition).toBe('ft-rf')
    expect(r.metrics.two_way_accuracy).toBe(0.76)
    expect(r.metrics.two_way_macro_f1).toBe(0.7453)
    expect(r.metrics.two_way_weighted_f1).toBe(0.7576)
    expect(r.metrics.weighted_f1).toBe(0.6679)
  })
  test('news accuracy is the judge\'s, read from the judged file', () => {
    const run = runs.find((r) => r.id === 'news-e2b-pi5-test-20261009-072927')!
    const judged = fs.readdirSync(path.join(REPO, 'results/board', run.id)).find((f) => f.includes('.judged-'))!
    const j = JSON.parse(fs.readFileSync(path.join(REPO, 'results/board', run.id, judged), 'utf8'))
    expect(run.metrics.accuracy).toBe(j.metrics.accuracy)
  })
  test('SciEntsBank keeps accuracy and macro F1 as stored', () => {
    const run = runs.find((r) => r.id === 'think-e2b-jetson-test_ua-20261011-232209')!
    expect(run.metrics.accuracy).toBe(0.6704)
    expect(run.metrics.macro_f1).toBe(0.6377)
  })
  test('lg runs are labelled lg even though their JSON says baseline', () => {
    expect(runs.find((r) => r.id === 'lg-e2b-jetson-test_ua-20261013-152608')?.condition).toBe('lg')
  })
  test('board-check enrichment comes from the newest check that has the run', () => {
    const run = runs.find((r) => r.id === 'e2b-jetson-test_ua-20260930-224702')!
    expect(run.energy?.jPerAnswer).toBe(3.95)
    expect(run.vsKaggle?.kaggleAccuracy).toBe(0.5519)
  })
  test('Kaggle news run takes its judged accuracy', () => {
    const run = runs.find((r) => r.id === 'news-rag-e4b-kaggle-t4-en-k3-20261008-065226.json')!
    expect(run.metrics.accuracy).toBe(0.45)
  })
})

describe('loadRuns on odd folders', () => {
  function fixture(): string {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'eg-dash-'))
    const mk = (p: string, files: Record<string, string> = {}) => {
      fs.mkdirSync(path.join(root, p), { recursive: true })
      for (const [f, c] of Object.entries(files)) fs.writeFileSync(path.join(root, p, f), c)
    }
    const summary = JSON.stringify({ stage: 'baseline', metrics: { n: 10, accuracy: 0.5 } })
    mk('results/board/e2b-pi5-test_ua-20261020-010101', { 'baseline-x.json': summary })
    mk('results/board/e4b-pi5-test_ua-20261020-020202', { 'baseline-y.json': '{not json' })
    mk('results/kaggle')
    mk('backups/pi/results/board/think-e2b-pi5-test_ua-20261021-030303')
    mk('backups/pi/results/board/e2b-pi5-test_ua-20261020-010101', { 'baseline-x.json': summary })
    mk('backups/pi/results/local/interrupted-think-e4b-pi5-test_ua-20261021-040404')
    return root
  }
  const runs = loadRuns(fixture())
  test('a run still in progress on the board is listed with no metrics', () => {
    const r = runs.find((x) => x.id === 'think-e2b-pi5-test_ua-20261021-030303')!
    expect(r.location).toBe('backup')
    expect(r.metrics).toEqual({})
    expect(r.error).toBeUndefined()
  })
  test('broken JSON is an error row, and the other runs still load', () => {
    expect(runs.find((x) => x.id === 'e4b-pi5-test_ua-20261020-020202')?.error).toBeTruthy()
    expect(runs.find((x) => x.id === 'e2b-pi5-test_ua-20261020-010101')?.metrics.accuracy).toBe(0.5)
  })
  test('a run in both results/ and the backup is listed once, from results/', () => {
    const same = runs.filter((x) => x.id === 'e2b-pi5-test_ua-20261020-010101')
    expect(same).toHaveLength(1)
    expect(same[0].location).toBe('results')
  })
  test('interrupted folders in results/local are ignored', () => {
    expect(runs.some((x) => x.id.startsWith('interrupted'))).toBe(false)
    expect(runs).toHaveLength(3)
  })
})
