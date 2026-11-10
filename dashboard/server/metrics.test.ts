import { describe, expect, test } from 'vitest'
import { score, twoWay } from './metrics.ts'

describe('score, as edge_grader.metrics.score (sklearn) computes it', () => {
  test('accuracy, macro and weighted F1 over the gold labels; an unparsed reply counts as wrong', () => {
    const gold = ['correct', 'correct', 'incorrect', 'contradictory']
    const pred = ['correct', 'incorrect', 'incorrect', null]
    expect(score(gold, pred)).toEqual({ n: 4, accuracy: 0.5, macro_f1: 0.4444, weighted_f1: 0.5, unparsed: 1 })
  })
  test('2-way folds contradictory into incorrect', () => {
    expect(twoWay('contradictory')).toBe('incorrect')
    expect(twoWay('correct')).toBe('correct')
    expect(twoWay(null)).toBeNull()
  })
})
