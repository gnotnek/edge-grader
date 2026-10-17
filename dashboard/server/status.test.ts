import { describe, expect, test } from 'vitest'
import { cellKey, expectedStatus, jobKey, latestByCell } from './status.ts'
import type { BoardState, Expected, Run } from './types.ts'

function run(over: Partial<Run>): Run {
  return {
    id: 'x', condition: 'think', benchmark: 'scientsbank', model: 'e2b', board: 'pi5',
    split: 'test_ua', stamp: '20261020-000000', location: 'results', source: '', metrics: {},
    ...over,
  }
}
const exp: Expected = {
  board: 'pi5', model: 'e2b', benchmark: 'scientsbank', split: 'test_ua', condition: 'think', plan: 'B1',
}
const noBoards: BoardState[] = []
const piWith = (state: string, env: Record<string, string>, args = ['e2b', 'test_ua']): BoardState[] => [
  { board: 'pi5', hasBackup: true, jobs: [{ id: 1, state, args, env }] },
]

describe('latestByCell', () => {
  test('the newer of two runs in the same cell wins', () => {
    const m = latestByCell([
      run({ id: 'old', condition: 'baseline', stamp: '20261004-075411' }),
      run({ id: 'new', condition: 'baseline', stamp: '20261004-225353' }),
    ])
    expect(m.get(cellKey({ ...exp, condition: 'baseline' }))?.id).toBe('new')
  })
})

describe('jobKey', () => {
  test('THINKING=on is the think condition', () => {
    expect(jobKey({ id: 1, state: 'pending', args: ['e4b', 'test_uq'], env: { BENCH: 'scientsbank', THINKING: 'on' } }, 'pi5'))
      .toBe(cellKey({ ...exp, model: 'e4b', split: 'test_uq' }))
  })
  test('VARIANT names the condition; BENCH defaults to scientsbank', () => {
    expect(jobKey({ id: 1, state: 'done', args: ['e2b', 'test_ua'], env: { VARIANT: 'lg-reuse' } }, 'jetson'))
      .toBe('jetson|e2b|scientsbank|test_ua|lg-reuse')
  })
  test('GSM symbolic carries its instance count, as the folder does', () => {
    expect(jobKey({ id: 1, state: 'done', args: ['e2b', 'symbolic'], env: { BENCH: 'gsm', INSTANCES: '10' } }, 'jetson'))
      .toBe('jetson|e2b|gsm|symbolic-i10|baseline')
  })
})

describe('expectedStatus', () => {
  test('a fetched run is done', () => {
    expect(expectedStatus([exp], [run({})], noBoards)[0].status).toBe('done')
  })
  test('a run only in the backup is on-board', () => {
    expect(expectedStatus([exp], [run({ location: 'backup' })], noBoards)[0].status).toBe('on-board')
  })
  test('a pending job is queued', () => {
    expect(expectedStatus([exp], [], piWith('pending', { THINKING: 'on' }))[0].status).toBe('queued')
  })
  test('a running job is running, even with its folder already in the backup', () => {
    expect(expectedStatus([exp], [run({ location: 'backup' })], piWith('running', { THINKING: 'on' }))[0].status).toBe('running')
  })
  test('nothing at all is missing', () => {
    expect(expectedStatus([exp], [], noBoards)[0].status).toBe('missing')
  })
  test('a job for a different condition does not count', () => {
    expect(expectedStatus([exp], [], piWith('pending', {}))[0].status).toBe('missing')
  })
})
