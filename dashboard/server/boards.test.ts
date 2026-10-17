import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { describe, expect, test } from 'vitest'
import { loadBoards } from './boards.ts'

const REPO = path.resolve(import.meta.dirname, '../..')

describe('loadBoards on the real repo', () => {
  const boards = loadBoards(REPO)
  test('the Jetson backup has its 23 jobs', () => {
    const j = boards.find((b) => b.board === 'jetson')!
    expect(j.hasBackup).toBe(true)
    expect(j.jobs).toHaveLength(23)
    expect(j.lastBackup).toMatch(/^2026-10-\d\dT/)
  })
  test('a job carries the tail of its own log', () => {
    const j = boards.find((b) => b.board === 'jetson')!
    expect(j.jobs.find((x) => x.id === 1)?.logTail).toBeTruthy()
  })
})

describe('loadBoards with no backups', () => {
  test('a board never backed up is reported as such, not as an error', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'eg-dash-'))
    const boards = loadBoards(root)
    expect(boards.map((b) => [b.board, b.hasBackup, b.jobs.length])).toEqual([
      ['pi5', false, 0],
      ['jetson', false, 0],
    ])
  })
})
