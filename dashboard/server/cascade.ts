// The newest cascade simulation (results/cascade/): policies built from existing board runs.
import fs from 'node:fs'
import path from 'node:path'
import type { Cascade } from './types.ts'

export function loadCascade(repo: string): Cascade | null {
  const dir = path.join(repo, 'results/cascade')
  const f = (fs.existsSync(dir) ? fs.readdirSync(dir) : []).filter((x) => /^cascade-.*\.json$/.test(x)).sort().at(-1)
  if (!f) return null
  const d = JSON.parse(fs.readFileSync(path.join(dir, f), 'utf8'))
  const splits: Cascade['splits'] = {}
  for (const [split, v] of Object.entries<{ points?: Cascade['splits'][string] }>(d.splits ?? {})) splits[split] = v.points ?? []
  return { file: `results/cascade/${f}`, board: d.board, clockState: d.clock_state, energyNote: d.energy_per_answer, splits }
}
