import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { describe, expect, test } from 'vitest'
import { loadCascade } from './cascade.ts'

const REPO = path.resolve(import.meta.dirname, '../..')

describe('loadCascade', () => {
  test('the newest simulation, its policies per split as stored', () => {
    const c = loadCascade(REPO)!
    expect(c.file).toMatch(/^results\/cascade\/cascade-jetson-/)
    expect(c.board).toBe('jetson')
    const p = c.splits.test_ua.find((x) => x.policy === 'cascade e2b+e4b → think-e2b')!
    expect(p).toMatchObject({ accuracy: 0.6889, escalated: 0.3481, j_per_answer: 59.72 })
  })
  test('no simulation gives null', () => {
    expect(loadCascade(fs.mkdtempSync(path.join(os.tmpdir(), 'eg-dash-')))).toBeNull()
  })
})
