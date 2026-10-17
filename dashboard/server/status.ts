// Which expected runs are done, waiting on a board, queued or still missing.
import type { BoardState, Condition, Expected, ExpectedStatus, Job, Run } from './types.ts'

type Cell = Pick<Expected, 'board' | 'model' | 'benchmark' | 'split' | 'condition'>

export function cellKey(x: Cell): string {
  return [x.board, x.model, x.benchmark, x.split, x.condition].join('|')
}

/** The newest run per cell; the grids and the status use it, the run list shows them all. */
export function latestByCell(runs: Run[]): Map<string, Run> {
  const out = new Map<string, Run>()
  for (const r of runs) {
    const k = cellKey(r)
    const prev = out.get(k)
    if (!prev || r.stamp > prev.stamp) out.set(k, r)
  }
  return out
}

/** The cell a queue job will fill, named the way board_grade.sh names its folder. */
export function jobKey(job: Job, board: string): string {
  const env = job.env ?? {}
  const bench = env.BENCH || 'scientsbank'
  const [model, split0] = job.args
  let split = split0
  if (bench === 'gsm' && split0 === 'symbolic') split = `symbolic-i${env.INSTANCES ?? ''}`
  let condition: Condition = 'baseline'
  if (bench === 'scientsbank') {
    if (env.RAG) condition = env.RAG === 'id' ? 'rag-id' : 'rag-en'
    else if (env.VARIANT && env.VARIANT !== 'base') condition = env.VARIANT as Condition
    else if (env.THINKING === 'on') condition = 'think'
  }
  return [board, model, bench, split, condition].join('|')
}

export function expectedStatus(expected: Expected[], runs: Run[], boards: BoardState[]): ExpectedStatus[] {
  const latest = latestByCell(runs)
  const jobState = new Map<string, string>()
  for (const b of boards)
    for (const j of b.jobs) {
      const k = jobKey(j, b.board)
      // A running job outranks a pending one for the same cell.
      if (j.state === 'running' || j.state === 'pending') {
        if (jobState.get(k) !== 'running') jobState.set(k, j.state)
      }
    }

  return expected.map((e) => {
    const k = cellKey(e)
    const run = latest.get(k)
    const job = jobState.get(k)
    let status: ExpectedStatus['status'] = 'missing'
    if (run?.location === 'results') status = 'done'
    else if (job === 'running') status = 'running'
    else if (run) status = 'on-board'
    else if (job === 'pending') status = 'queued'
    return run ? { ...e, status, run } : { ...e, status }
  })
}
