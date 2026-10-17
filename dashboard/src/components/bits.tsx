// Small shared pieces: board identity, run status, stat tiles, loading and error states.
import type { Board, Status } from '@server/types'
import { CircleAlert, CircleCheck, CircleDashed, Clock, HardDriveDownload, LoaderCircle } from 'lucide-react'
import type { ReactNode } from 'react'
import { Card, CardContent } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { BOARD_LABEL } from '@/lib/format'
import { cn } from '@/lib/utils'

export function BoardDot({ board, className }: { board: Board; className?: string }) {
  return (
    <span
      aria-hidden
      className={cn('inline-block size-2 shrink-0 rounded-full', className)}
      style={{ background: `var(--${board})` }}
    />
  )
}

export function BoardName({ board }: { board: Board }) {
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
      <BoardDot board={board} />
      {BOARD_LABEL[board]}
    </span>
  )
}

const STATUS: Record<Status, { label: string; icon: typeof CircleCheck; color: string }> = {
  done: { label: 'Done', icon: CircleCheck, color: 'var(--good)' },
  running: { label: 'Running', icon: LoaderCircle, color: 'var(--foreground)' },
  'on-board': { label: 'On board, not fetched', icon: HardDriveDownload, color: 'var(--warning)' },
  queued: { label: 'Queued', icon: Clock, color: 'var(--muted-foreground)' },
  missing: { label: 'Not run yet', icon: CircleDashed, color: 'var(--serious)' },
}

export function StatusLabel({ status }: { status: Status }) {
  const s = STATUS[status]
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap text-sm">
      <s.icon className="size-4" style={{ color: s.color }} aria-hidden />
      {s.label}
    </span>
  )
}

export function StatTile({ label, value, children }: { label: string; value: ReactNode; children?: ReactNode }) {
  return (
    <Card className="gap-2 py-4">
      <CardContent className="px-4">
        <div className="text-sm text-muted-foreground">{label}</div>
        <div className="mt-1 text-3xl font-semibold tracking-tight">{value}</div>
        {children && <div className="mt-2 text-sm text-muted-foreground">{children}</div>}
      </CardContent>
    </Card>
  )
}

export function Loading({ rows = 3 }: { rows?: number }) {
  return (
    <div className="space-y-3">
      {Array.from({ length: rows }, (_, i) => (
        <Skeleton key={i} className="h-24 w-full" />
      ))}
    </div>
  )
}

export function ErrorNote({ message }: { message: string }) {
  return (
    <div className="flex items-start gap-2 rounded-lg border border-destructive/40 bg-destructive/5 p-4 text-sm">
      <CircleAlert className="mt-0.5 size-4 shrink-0 text-destructive" aria-hidden />
      <div>
        <div className="font-medium">Couldn't read the project files</div>
        <div className="text-muted-foreground">{message}</div>
      </div>
    </div>
  )
}

export function Section({ title, description, actions, children }: {
  title: string
  description?: ReactNode
  actions?: ReactNode
  children: ReactNode
}) {
  return (
    <section className="space-y-3">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold tracking-tight">{title}</h2>
          {description && <p className="max-w-prose text-sm text-muted-foreground">{description}</p>}
        </div>
        {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
      </div>
      {children}
    </section>
  )
}
