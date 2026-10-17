import type { Board, ExpectedStatus, Overview, Run } from '@server/types'
import { useState } from 'react'
import { BoardDot, BoardName, ErrorNote, Loading, Reveal, Section, Stat, StatusLabel } from '@/components/bits'
import { PlateLegend, PlateMap } from '@/components/plate-map'
import { RunSheet } from '@/components/run-sheet'
import { Card, CardContent } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { useApi } from '@/lib/api'
import { BENCH_LABEL, BOARD_LABEL, CONDITION_LABEL, MODEL_LABEL, ago, pct, plainMarkdown, stampLabel } from '@/lib/format'

function weekLine(o: Overview): string {
  // The plan numbers its stages ("0. Edge baseline + A"); the headline reads better without.
  return o.currentWeek ? `Week ${o.currentWeek.week}: ${o.currentWeek.stage.replace(/^\w+\.\s+/, '')}.` : 'Outside the plan timeline.'
}

function owingLine(o: Overview): string {
  const owed = Object.entries(o.owedByBoard).filter(([, n]) => n > 0)
  if (!owed.length) return 'No board owes a run.'
  const parts = owed.map(([b, n]) => `the ${BOARD_LABEL[b as Board]} owes ${n} run${n === 1 ? '' : 's'}`).join(' and ')
  return `${parts.charAt(0).toUpperCase()}${parts.slice(1)}.`
}

function WaitingList({ waiting }: { waiting: ExpectedStatus[] }) {
  if (!waiting.length)
    return (
      <p className="text-sm text-muted-foreground">
        Every expected run is in <code className="font-mono text-xs">results/</code>. Add the next ones to{' '}
        <code className="font-mono text-xs">dashboard/expected-runs.json</code>.
      </p>
    )
  return (
    <ul className="divide-y">
      {waiting.map((w) => (
        <li key={`${w.board}${w.model}${w.split}${w.condition}`} className="flex flex-wrap items-center justify-between gap-2 py-2.5 text-sm first:pt-0 last:pb-0">
          <span className="inline-flex items-center gap-2">
            <BoardDot board={w.board} />
            <span>
              {CONDITION_LABEL[w.condition]} {MODEL_LABEL[w.model]}{' '}
              <span className="font-mono text-xs text-muted-foreground">{w.split}</span>
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
          <button
            type="button"
            onClick={() => onOpen(r)}
            className="-mx-2 flex w-[calc(100%+1rem)] items-center justify-between gap-3 rounded-md px-2 py-2.5 text-left text-sm transition-colors duration-200 hover:bg-muted focus-visible:bg-muted focus-visible:outline-none active:scale-[0.99]"
          >
            <span className="min-w-0">
              <span className="block truncate text-[var(--heading)]">
                {BENCH_LABEL[r.benchmark]} {CONDITION_LABEL[r.condition]} {MODEL_LABEL[r.model]}{' '}
                <span className="font-mono text-xs text-muted-foreground">{r.split}</span>
              </span>
              <span className="text-xs text-muted-foreground">
                <BoardName board={r.board} /> <span className="font-mono">{stampLabel(r.stamp)}</span>
              </span>
            </span>
            <span className="tabular font-medium text-[var(--heading)]">{pct(r.metrics.accuracy)}</span>
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
  if (!o || !runs.data || !expected.data) return <Loading rows={3} />

  const total = Object.values(o.runsByBoard).reduce((a, b) => a + b, 0)
  return (
    <div className="space-y-16">
      <div className="grid gap-6 lg:grid-cols-12">
        <Reveal className="lg:col-span-7">
          <p className="text-sm text-muted-foreground">{o.currentWeek ? o.currentWeek.dates : o.today}</p>
          <h2 className="mt-3 font-serif text-4xl font-medium text-balance md:text-5xl">{weekLine(o)}</h2>
          <p className="mt-3 font-serif text-2xl text-muted-foreground md:text-3xl">{owingLine(o)}</p>
          {o.currentWeek && (
            <p className="mt-6 max-w-prose text-sm text-muted-foreground">{plainMarkdown(o.currentWeek.goal)}</p>
          )}
        </Reveal>
        <Reveal i={1} className="lg:col-span-5">
          <Card className="h-full">
            <CardContent>
              <div className="flex items-baseline justify-between gap-3">
                <div className="text-sm text-muted-foreground">Runs still owed</div>
                <div className="tabular text-3xl font-semibold tracking-tight text-[var(--heading)]">{o.waiting.length}</div>
              </div>
              <div className="mt-4"><WaitingList waiting={o.waiting} /></div>
            </CardContent>
          </Card>
        </Reveal>
      </div>

      <Section
        i={2}
        title="Coverage"
        description="Every run the project has, by benchmark and condition. Click a filled well to open the run."
      >
        <Card>
          <CardContent className="space-y-4">
            <PlateMap runs={runs.data} expected={expected.data} onOpen={setOpen} />
            <PlateLegend />
          </CardContent>
        </Card>
      </Section>

      <div className="grid gap-6 lg:grid-cols-12">
        <Reveal i={3} className="lg:col-span-7">
          <Card className="h-full">
            <CardContent>
              <div className="mb-3 text-sm text-muted-foreground">Newest runs</div>
              <NewestRuns runs={o.newest} onOpen={setOpen} />
            </CardContent>
          </Card>
        </Reveal>
        <Reveal i={4} className="lg:col-span-5">
          <Card className="h-full">
            <CardContent className="divide-y">
              <Stat label="Plan items ticked" value={`${o.planProgress.done} of ${o.planProgress.total}`}>
                <Progress value={(o.planProgress.done / Math.max(1, o.planProgress.total)) * 100} aria-label="Plan progress" className="h-1.5" />
              </Stat>
              <Stat label="Runs recorded" value={total}>
                <span className="flex flex-wrap gap-x-3">
                  {(['jetson', 'pi5', 'kaggle'] as Board[]).map((b) => (
                    <span key={b} className="inline-flex items-center gap-1.5">
                      <BoardDot board={b} />{BOARD_LABEL[b]} <span className="tabular">{o.runsByBoard[b] ?? 0}</span>
                    </span>
                  ))}
                </span>
              </Stat>
              <Stat label="Board data as of" value={ago(o.lastBackup.jetson)}>
                <span className="block">Jetson, from its backup.</span>
                <span className="block">Pi 5: {o.lastBackup.pi5 ? ago(o.lastBackup.pi5) : 'no backup on this Mac yet.'}</span>
              </Stat>
            </CardContent>
          </Card>
        </Reveal>
      </div>

      {o.errors > 0 && <ErrorNote message={`${o.errors} result file(s) couldn't be read. They're marked on the Runs page.`} />}
      <RunSheet run={open} onClose={() => setOpen(null)} />
    </div>
  )
}
