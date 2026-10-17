// What scripts/backup_board.sh copied from each board: its queue (jobs.json, per-job logs).
import fs from 'node:fs'
import path from 'node:path'
import type { BoardState, Job } from './types.ts'

const BOARDS: [BoardState['board'], string][] = [['pi5', 'pi'], ['jetson', 'jetson']]
const LOG_TAIL_LINES = 40

function mtime(file: string): number {
  try {
    return fs.statSync(file).mtimeMs
  } catch {
    return 0
  }
}

function tail(file: string): string | undefined {
  try {
    return fs.readFileSync(file, 'utf8').trimEnd().split('\n').slice(-LOG_TAIL_LINES).join('\n')
  } catch {
    return undefined
  }
}

/**
 * rsync -a keeps the board's own mtimes, so the newest copied file says how fresh the copy is:
 * the latest moment on the board that this Mac knows about.
 */
function dataAsOf(dir: string): string | undefined {
  const candidates = [path.join(dir, 'queue/jobs.json'), path.join(dir, 'queue/worker.log')]
  const runs = path.join(dir, 'results/board')
  try {
    for (const r of fs.readdirSync(runs)) candidates.push(path.join(runs, r))
  } catch {
    /* no runs copied */
  }
  const newest = Math.max(0, ...candidates.map(mtime))
  return newest ? new Date(newest).toISOString() : undefined
}

export function loadBoards(repo: string): BoardState[] {
  return BOARDS.map(([board, dirName]) => {
    const dir = path.join(repo, 'backups', dirName)
    if (!fs.existsSync(dir)) return { board, hasBackup: false, jobs: [] }
    const state: BoardState = { board, hasBackup: true, lastBackup: dataAsOf(dir), jobs: [] }
    try {
      const jobs = JSON.parse(fs.readFileSync(path.join(dir, 'queue/jobs.json'), 'utf8')) as Job[]
      state.jobs = jobs.map((j) => ({ ...j, logTail: tail(path.join(dir, `queue/logs/job-${j.id}.log`)) }))
    } catch (e) {
      if ((e as NodeJS.ErrnoException).code !== 'ENOENT') state.error = `jobs.json: ${(e as Error).message}`
    }
    return state
  })
}
