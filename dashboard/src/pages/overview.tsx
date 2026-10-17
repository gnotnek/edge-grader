import type { Board, ExpectedStatus, Overview, Run } from '@server/types'
import { useState } from 'react'
import { BoardDot, BoardName, ErrorNote, Loading, Section, StatTile, StatusLabel } from '@/components/bits'
import { PlateLegend, PlateMap } from '@/components/plate-map'
import { RunSheet } from '@/components/run-sheet'
import { Card, CardContent } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { useApi } from '@/lib/api'
import { BENCH_LABEL, BOARD_LABEL, CONDITION_LABEL, MODEL_LABEL, ago, pct, stampLabel } from '@/lib/format'

function headline(o: Overview): string {
  const owed = Object.entries(o.owedByBoard).filter(([, n]) => n > 0)
  const owing = owed.length
    ? owed.map(([b, n]) => `the ${BOARD_LABEL[b as Board]} owes ${n} run${n === 1 ? '' : 's'}`).join(' and ')
    : 'no board owes a run'
  // The plan numbers its stages ("0. Edge baseline + A"); the headline reads better without.
  const week = o.currentWeek ? `Week ${o.currentWeek.week}: ${o.currentWeek.stage.replace(/^\w+\.\s+/, '')}` : 'Outside the plan timeline'
  return `${week}. ${owing.charAt(0).toUpperCase()}${owing.slice(1)}.`
}

function WaitingList({ waiting }: { waiting: ExpectedStatus[] }) {
  if (!waiting.length)
    return <p className="text-sm text-muted-foreground">Every expected run is in results/. Add the next ones to dashboard/expected-runs.json.</p>
  return (
    <ul className="divide-y">
      {waiting.map((w) => (
        <li key={`${w.board}${w.model}${w.split}${w.condition}`} className="flex flex-wrap items-center justify-between gap-2 py-2 text-sm">
          <span className="inline-flex items-center gap-2">
            <BoardDot board={w.board} />
            <span>
              {CONDITION_LABEL[w.condition]} {MODEL_LABEL[w.model]} <span className="text-muted-foreground">{w.split}</span>
            </span>
          </span>
          <StatusLabel status={w.status} />
        </li>
      ))}
    </ul>
  )
}

function NewestRuns({ runs, onOpen }: { runs: Run[]; onOpen: (r: Run) => void }) {
  return (
    <ul className="divide-y">
      {runs.map((r) => (
        <li key={r.id}>
          <button type="button" onClick={() => onOpen(r)} className="flex w-full items-center justify-between gap-3 py-2 text-left text-sm hover:bg-muted/50 focus-visible:bg-muted/50 focus-visible:outline-none">
            <span className="min-w-0">
              <span className="block truncate">
                {BENCH_LABEL[r.benchmark]} {CONDITION_LABEL[r.condition]} {MODEL_LABEL[r.model]}{' '}
                <span className="text-muted-foreground">{r.split}</span>
              </span>
              <span className="text-xs text-muted-foreground"><BoardName board={r.board} /> {stampLabel(r.stamp)}</span>
            </span>
            <span className="tabular font-medium">{pct(r.metrics.accuracy)}</span>
          </button>
        </li>
      ))}
    </ul>
  )
}

export function OverviewPage() {
  const overview = useApi<Overview>('/overview')
  const runs = useApi<Run[]>('/runs')
  const expected = useApi<ExpectedStatus[]>('/expected')
  const [open, setOpen] = useState<Run | null>(null)

  const error = overview.error ?? runs.error ?? expected.error
  if (error) return <ErrorNote message={error} />
  const o = overview.data
  if (!o || !runs.data || !expected.data) return <Loading rows={4} />

  const boardRuns = (b: string) => o.runsByBoard[b] ?? 0
  return (
    <div className="space-y-8">
      <div>
        <p className="line-clamp-2 max-w-3xl text-sm text-muted-foreground">
          {o.currentWeek ? `${o.currentWeek.dates}: ${o.currentWeek.goal}` : `Today is ${o.today}`}
        </p>
        <p className="mt-1 max-w-3xl text-2xl font-semibold tracking-tight text-balance md:text-3xl">{headline(o)}</p>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <StatTile label="Plan items ticked" value={`${o.planProgress.done} of ${o.planProgress.total}`}>
          <Progress value={(o.planProgress.done / Math.max(1, o.planProgress.total)) * 100} aria-label="Plan progress" />
        </StatTile>
        <StatTile label="Runs recorded" value={Object.values(o.runsByBoard).reduce((a, b) => a + b, 0)}>
          <span className="flex flex-wrap gap-x-3">
            {(['jetson', 'pi5', 'kaggle'] as Board[]).map((b) => (
              <span key={b} className="inline-flex items-center gap-1"><BoardDot board={b} />{BOARD_LABEL[b]} {boardRuns(b)}</span>
            ))}
          </span>
        </StatTile>
        <StatTile label="Runs still owed" value={o.waiting.length}>
          {o.waiting.length ? `${o.owedByBoard.pi5 ?? 0} on the Pi 5, ${o.owedByBoard.jetson ?? 0} on the Jetson` : 'Nothing expected is missing'}
        </StatTile>
        <StatTile label="Board data as of" value={ago(o.lastBackup.jetson)}>
          <span className="block">Jetson, from its backup</span>
          <span className="block">Pi 5: {o.lastBackup.pi5 ? ago(o.lastBackup.pi5) : 'no backup on this Mac yet'}</span>
        </StatTile>
      </div>

      <Section title="Coverage" description="Every run the project has, by benchmark and condition. Click a filled well to open the run.">
        <Card className="py-4">
          <CardContent className="space-y-3 px-4">
            <PlateMap runs={runs.data} expected={expected.data} onOpen={setOpen} />
            <PlateLegend />
          </CardContent>
        </Card>
      </Section>

      <div className="grid gap-6 lg:grid-cols-2">
        <Section title="Waiting on" description="From dashboard/expected-runs.json.">
          <Card className="py-2"><CardContent className="px-4"><WaitingList waiting={o.waiting} /></CardContent></Card>
        </Section>
        <Section title="Newest runs">
          <Card className="py-2"><CardContent className="px-4"><NewestRuns runs={o.newest} onOpen={setOpen} /></CardContent></Card>
        </Section>
      </div>
      {o.errors > 0 && <ErrorNote message={`${o.errors} result file(s) couldn't be read. They're marked on the Runs page.`} />}
      <RunSheet run={open} onClose={() => setOpen(null)} />
    </div>
  )
}
