import type { BoardState, ExpectedStatus, Job } from '@server/types'
import { ChevronRight, HardDrive } from 'lucide-react'
import { Fragment, useState } from 'react'
import { BoardDot, ErrorNote, Loading, StatusLabel } from '@/components/bits'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { useApi } from '@/lib/api'
import { BOARD_LABEL, CONDITION_LABEL, MODEL_LABEL, ago, dateLabel, duration } from '@/lib/format'
import { cn } from '@/lib/utils'

const SCRIPT_NAME = { pi5: 'pi', jetson: 'jetson' } as const

function jobLabel(j: Job): string {
  const env = Object.entries(j.env ?? {})
    .filter(([k, v]) => !(k === 'BENCH' && v === 'scientsbank') && k !== 'INSTANCES')
    .map(([k, v]) => `${k}=${v}`)
  const bench = j.env?.BENCH && j.env.BENCH !== 'scientsbank' ? `${j.env.BENCH} ` : ''
  return `${bench}${j.args.join(' ')}${env.length ? `  ${env.join(' ')}` : ''}`
}

function JobState({ state, rc }: { state: string; rc?: number | null }) {
  const variant = state === 'done' && rc === 0 ? 'secondary' : state === 'running' ? 'default' : 'outline'
  return <Badge variant={variant} className={cn(state === 'interrupted' && 'border-[var(--serious)]')}>{state}</Badge>
}

function JobsTable({ jobs }: { jobs: Job[] }) {
  const [openId, setOpenId] = useState<number | null>(null)
  const ordered = [...jobs].reverse()
  return (
    <Table className="tabular">
      <TableHeader>
        <TableRow>
          <TableHead className="w-10 pl-4">#</TableHead>
          <TableHead>Job</TableHead>
          <TableHead>State</TableHead>
          <TableHead className="hidden md:table-cell">Started</TableHead>
          <TableHead className="hidden pr-4 text-right md:table-cell">Took</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {ordered.map((j) => {
          const open = openId === j.id
          return (
            <Fragment key={j.id}>
              <TableRow className="cursor-pointer" onClick={() => setOpenId(open ? null : j.id)}>
                <TableCell className="pl-4 text-muted-foreground">{j.id}</TableCell>
                <TableCell>
                  <button
                    type="button" aria-expanded={open}
                    className="inline-flex items-center gap-1 text-left focus-visible:underline focus-visible:outline-none"
                    onClick={(e) => { e.stopPropagation(); setOpenId(open ? null : j.id) }}
                  >
                    <ChevronRight className={cn('size-3.5 shrink-0 text-muted-foreground transition-transform', open && 'rotate-90')} aria-hidden />
                    <code className="text-xs">{jobLabel(j)}</code>
                  </button>
                </TableCell>
                <TableCell><JobState state={j.state} rc={j.rc} /></TableCell>
                <TableCell className="hidden whitespace-nowrap text-muted-foreground md:table-cell">{dateLabel(j.started)}</TableCell>
                <TableCell className="hidden pr-4 text-right whitespace-nowrap md:table-cell">{duration(j.started, j.ended)}</TableCell>
              </TableRow>
              {open && (
                <TableRow className="hover:bg-transparent">
                  <TableCell colSpan={5} className="px-4">
                    <pre className="max-h-80 overflow-auto rounded-md bg-muted p-3 text-xs leading-relaxed whitespace-pre-wrap">
                      {j.logTail ?? 'No log for this job in the backup.'}
                    </pre>
                  </TableCell>
                </TableRow>
              )}
            </Fragment>
          )
        })}
      </TableBody>
    </Table>
  )
}

function BoardCard({ b, owed }: { b: BoardState; owed: ExpectedStatus[] }) {
  const counts = b.jobs.reduce<Record<string, number>>((acc, j) => ({ ...acc, [j.state]: (acc[j.state] ?? 0) + 1 }), {})
  return (
    <Card className="gap-4 py-4">
      <CardHeader className="px-4">
        <CardTitle className="flex items-center gap-2 text-base">
          <BoardDot board={b.board} className="size-2.5" /> {BOARD_LABEL[b.board]}
        </CardTitle>
        <CardDescription>
          {b.hasBackup
            ? `Board data as of ${dateLabel(b.lastBackup)} (${ago(b.lastBackup)}). ${Object.entries(counts).map(([s, n]) => `${n} ${s}`).join(', ')}.`
            : 'No backup of this board on the Mac yet.'}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4 px-0">
        {owed.length > 0 && (
          <div className="px-4">
            <div className="mb-1 text-sm font-medium">Still owed</div>
            <ul className="divide-y rounded-md border">
              {owed.map((w) => (
                <li key={`${w.model}${w.split}${w.condition}`} className="flex flex-wrap items-center justify-between gap-2 px-3 py-2 text-sm">
                  <span>{CONDITION_LABEL[w.condition]} {MODEL_LABEL[w.model]} <span className="text-muted-foreground">{w.split}</span></span>
                  <StatusLabel status={w.status} />
                </li>
              ))}
            </ul>
          </div>
        )}
        {b.error && <div className="px-4"><ErrorNote message={b.error} /></div>}
        {b.hasBackup ? (
          b.jobs.length ? <JobsTable jobs={b.jobs} /> : <p className="px-4 text-sm text-muted-foreground">The backup has no queue jobs.</p>
        ) : (
          <div className="mx-4 flex items-start gap-3 rounded-md border border-dashed p-4 text-sm">
            <HardDrive className="mt-0.5 size-4 shrink-0 text-muted-foreground" aria-hidden />
            <div>
              <p>When the board is free, copy its queue and results here, then reload:</p>
              <code className="mt-2 inline-block rounded bg-muted px-2 py-1 text-xs">scripts/backup_board.sh {SCRIPT_NAME[b.board]}</code>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

export function BoardsPage() {
  const boards = useApi<BoardState[]>('/boards')
  const expected = useApi<ExpectedStatus[]>('/expected')
  const error = boards.error ?? expected.error
  if (error) return <ErrorNote message={error} />
  if (!boards.data || !expected.data) return <Loading rows={2} />
  return (
    <div className="grid items-start gap-4 xl:grid-cols-2">
      {boards.data.map((b) => (
        <BoardCard key={b.board} b={b} owed={expected.data!.filter((e) => e.board === b.board && e.status !== 'done')} />
      ))}
    </div>
  )
}
