// Coverage as a plate map: one well per (benchmark, condition, split) × (board, model).
// Filled = fetched, half = on the board only, dashed = expected but not here yet.
import type { Benchmark, Board, Condition, ExpectedStatus, Model, Run } from '@server/types'
import { Fragment } from 'react'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import {
  BENCH_LABEL, BENCH_ORDER, BOARD_LABEL, BOARD_ORDER, CONDITION_LABEL, CONDITION_ORDER, MODEL_LABEL, pct,
  splitGroup, stampLabel,
} from '@/lib/format'

const MODELS: Model[] = ['e2b', 'e4b']
const COLUMNS = BOARD_ORDER.flatMap((board) => MODELS.map((model) => ({ board, model })))

type Well =
  | { kind: 'run'; run: Run }
  | { kind: 'expected'; e: ExpectedStatus }
  | { kind: 'empty' }

interface Row {
  benchmark: Benchmark
  condition: Condition
  splits: string[]
}

const key = (b: Benchmark, c: Condition, s: string, board: Board, m: Model) => [b, c, s, board, m].join('|')

function buildRows(runs: Run[], expected: ExpectedStatus[]): Row[] {
  const splits = new Map<string, Set<string>>()
  for (const x of [...runs, ...expected]) {
    const k = `${x.benchmark}|${x.condition}`
    if (!splits.has(k)) splits.set(k, new Set())
    splits.get(k)!.add(splitGroup(x.split))
  }
  const rows: Row[] = []
  for (const benchmark of BENCH_ORDER)
    for (const condition of CONDITION_ORDER) {
      const s = splits.get(`${benchmark}|${condition}`)
      if (s) rows.push({ benchmark, condition, splits: [...s].sort() })
    }
  return rows
}

function WellDot({ well, board }: { well: Well; board: Board }) {
  const color = `var(--${board})`
  if (well.kind === 'run') {
    const half = well.run.location === 'backup'
    return (
      <span
        className="block size-3.5 rounded-full border-2"
        style={{
          borderColor: color,
          background: half ? `linear-gradient(90deg, ${color} 50%, transparent 50%)` : color,
        }}
      />
    )
  }
  if (well.kind === 'expected')
    return <span className="block size-3.5 rounded-full border-2 border-dashed" style={{ borderColor: 'var(--serious)' }} />
  return <span className="block size-3.5 rounded-full border border-border" />
}

function wellText(well: Well, split: string): string {
  if (well.kind === 'run') {
    const r = well.run
    const where = r.location === 'backup' ? ', on the board, not fetched' : ''
    return `${r.split}: ${pct(r.metrics.accuracy)} accuracy, ${stampLabel(r.stamp)}${where}`
  }
  if (well.kind === 'expected') return `${split}: expected for "${well.e.plan}", not here yet`
  return `${split}: not planned`
}

export function PlateMap({ runs, expected, onOpen }: {
  runs: Run[]
  expected: ExpectedStatus[]
  onOpen: (run: Run) => void
}) {
  // Newest run per well; runs arrive newest first.
  const wells = new Map<string, Well>()
  for (const r of runs) {
    const k = key(r.benchmark, r.condition, splitGroup(r.split), r.board, r.model)
    if (!wells.has(k)) wells.set(k, { kind: 'run', run: r })
  }
  for (const e of expected) {
    const k = key(e.benchmark, e.condition, splitGroup(e.split), e.board, e.model)
    if (!wells.has(k) && e.status !== 'done') wells.set(k, { kind: 'expected', e })
  }
  const rows = buildRows(runs, expected)

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-separate border-spacing-0 text-sm">
        <thead>
          <tr>
            <th className="sticky left-0 bg-card" />
            {BOARD_ORDER.map((b) => (
              <th key={b} colSpan={2} className="px-2 pb-1 text-center font-medium">
                <span className="inline-flex items-center gap-1.5">
                  <span className="size-2 rounded-full" style={{ background: `var(--${b})` }} />
                  {BOARD_LABEL[b]}
                </span>
              </th>
            ))}
          </tr>
          <tr className="text-xs text-muted-foreground">
            <th className="sticky left-0 bg-card" />
            {COLUMNS.map((c) => (
              <th key={`${c.board}${c.model}`} className="px-2 pb-2 font-normal">{MODEL_LABEL[c.model]}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => {
            const firstOfBench = i === 0 || rows[i - 1].benchmark !== row.benchmark
            return (
              <Fragment key={`${row.benchmark}${row.condition}`}>
                {firstOfBench && (
                  <tr>
                    <th colSpan={7} className="sticky left-0 border-t bg-card pt-3 pb-1 text-left font-semibold">
                      {BENCH_LABEL[row.benchmark]}
                    </th>
                  </tr>
                )}
                <tr>
                  <th className="sticky left-0 bg-card py-1 pr-4 text-left font-normal whitespace-nowrap text-muted-foreground">
                    {CONDITION_LABEL[row.condition]}
                  </th>
                  {COLUMNS.map((c) => (
                    <td key={`${c.board}${c.model}`} className="px-1 py-1">
                      <div className="flex justify-center gap-0.5">
                        {row.splits.map((split) => {
                          const well = wells.get(key(row.benchmark, row.condition, split, c.board, c.model)) ?? { kind: 'empty' as const }
                          const label = `${BOARD_LABEL[c.board]} ${MODEL_LABEL[c.model]}, ${CONDITION_LABEL[row.condition]}, ${wellText(well, split)}`
                          return (
                            <Tooltip key={split}>
                              <TooltipTrigger asChild>
                                <button
                                  type="button"
                                  aria-label={label}
                                  disabled={well.kind !== 'run'}
                                  onClick={() => well.kind === 'run' && onOpen(well.run)}
                                  className="grid size-5 place-items-center rounded-full focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-default"
                                >
                                  <WellDot well={well} board={c.board} />
                                </button>
                              </TooltipTrigger>
                              <TooltipContent>{label}</TooltipContent>
                            </Tooltip>
                          )
                        })}
                      </div>
                    </td>
                  ))}
                </tr>
              </Fragment>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

export function PlateLegend() {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
      <span className="inline-flex items-center gap-1.5">
        <span className="size-3 rounded-full bg-foreground/70" /> Fetched
      </span>
      <span className="inline-flex items-center gap-1.5">
        <span className="size-3 rounded-full border-2 border-foreground/70" style={{ background: 'linear-gradient(90deg, color-mix(in oklab, var(--foreground) 70%, transparent) 50%, transparent 50%)' }} /> On the board, not fetched
      </span>
      <span className="inline-flex items-center gap-1.5">
        <span className="size-3 rounded-full border-2 border-dashed" style={{ borderColor: 'var(--serious)' }} /> Expected, not here yet
      </span>
      <span className="inline-flex items-center gap-1.5">
        <span className="size-3 rounded-full border border-border" /> Not planned
      </span>
      <span>Each well is one split, in alphabetical order. Hover a well to see which.</span>
    </div>
  )
}
