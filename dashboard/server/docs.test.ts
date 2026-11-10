import path from 'node:path'
import { describe, expect, test } from 'vitest'
import { docFile, listDocs, readDoc } from './docs.ts'

const REPO = path.resolve(import.meta.dirname, '../..')

describe('the results write-ups', () => {
  test('listed with their titles, summary first, then the recap', () => {
    const docs = listDocs(REPO)
    expect(docs[0]).toEqual({ name: 'summary.md', title: 'Findings so far, all stages' })
    expect(docs[1].name).toBe('recap.md')
    expect(docs.map((d) => d.name)).toContain('finetuning.md')
  })
  test('a page is read as markdown', () => {
    expect(readDoc(REPO, 'summary.md')).toMatch(/^# Findings so far, all stages/)
  })
  test('only pages in docs/results, by plain name', () => {
    expect(() => readDoc(REPO, '../plan.md')).toThrow()
    expect(() => readDoc(REPO, 'no-such-page.md')).toThrow()
    expect(() => readDoc(REPO, 'summary.txt')).toThrow()
  })
  test('figures are served from under docs/results, never outside it', () => {
    expect(docFile(REPO, 'figures/scientsbank-methods.png')).toMatchObject({ type: 'image/png' })
    expect(() => docFile(REPO, '../../CLAUDE.md')).toThrow()
    expect(() => docFile(REPO, 'figures/../../plan.md')).toThrow()
    expect(() => docFile(REPO, 'summary.md')).toThrow()
  })
})
