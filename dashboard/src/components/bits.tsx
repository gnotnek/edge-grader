// Small shared pieces: board identity, status tags, stats, sections, loading and error states.
import type { Board, Status } from '@server/types'
import { CircleAlert, CircleCheck, CircleDashed, Clock, HardDriveDownload, LoaderCircle } from 'lucide-react'
import { type CSSProperties, type ReactNode, useEffect, useRef } from 'react'
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

export type Tone = 'green' | 'red' | 'blue' | 'yellow' | 'neutral'

/** A small pastel pill. Always carries words; the colour only repeats them. */
export function Tag({ tone, icon: Icon, children, className }: {
  tone: Tone
  icon?: typeof CircleCheck
  children: ReactNode
  className?: string
}) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] leading-4 font-medium tracking-[0.05em] whitespace-nowrap uppercase',
        className,
      )}
      style={{ background: `var(--tag-${tone})`, color: `var(--tag-${tone}-ink)` }}
    >
      {Icon && <Icon className="size-3" aria-hidden />}
      {children}
    </span>
  )
}

const STATUS: Record<Status, { label: string; icon: typeof CircleCheck; tone: Tone }> = {
  done: { label: 'Done', icon: CircleCheck, tone: 'green' },
  running: { label: 'Running', icon: LoaderCircle, tone: 'blue' },
  'on-board': { label: 'Not fetched', icon: HardDriveDownload, tone: 'yellow' },
  queued: { label: 'Queued', icon: Clock, tone: 'neutral' },
  missing: { label: 'Not run yet', icon: CircleDashed, tone: 'red' },
}

export function StatusLabel({ status }: { status: Status }) {
  const s = STATUS[status]
  return <Tag tone={s.tone} icon={s.icon}>{s.label}</Tag>
}

/** A number with its label; no box of its own, so a group of them can share one. */
export function Stat({ label, value, children }: { label: string; value: ReactNode; children?: ReactNode }) {
  return (
    <div className="py-5 first:pt-0 last:pb-0">
      <div className="text-sm text-muted-foreground">{label}</div>
      <div className="tabular mt-1 text-3xl font-semibold tracking-tight text-[var(--heading)]">{value}</div>
      {children && <div className="mt-2 text-sm text-muted-foreground">{children}</div>}
    </div>
  )
}

/** Rises into place once, the first time it scrolls into view. Static under reduced motion. */
export function Reveal({ i = 0, className, children }: { i?: number; className?: string; children: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    if (!('IntersectionObserver' in window)) {
      el.dataset.in = ''
      return
    }
    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          el.dataset.in = ''
          io.disconnect()
        }
      },
      { rootMargin: '0px 0px -5% 0px' },
    )
    io.observe(el)
    return () => io.disconnect()
  }, [])
  return (
    <div ref={ref} className={cn('reveal', className)} style={{ '--i': i } as CSSProperties}>
      {children}
    </div>
  )
}

export function Loading({ rows = 3 }: { rows?: number }) {
  return (
    <div className="space-y-4">
      <Skeleton className="h-10 w-2/3" />
      {Array.from({ length: rows }, (_, i) => (
        <Skeleton key={i} className="h-28 w-full" />
      ))}
    </div>
  )
}

export function ErrorNote({ message }: { message: string }) {
  return (
    <div className="flex items-start gap-2 rounded-lg border p-4 text-sm" style={{ background: 'var(--tag-red)' }}>
      <CircleAlert className="mt-0.5 size-4 shrink-0" style={{ color: 'var(--tag-red-ink)' }} aria-hidden />
      <div>
        <div className="font-medium" style={{ color: 'var(--tag-red-ink)' }}>Couldn't read the project files</div>
        <div className="text-foreground">{message}</div>
      </div>
    </div>
  )
}

export function Section({ title, description, actions, children, i = 0 }: {
  title: string
  description?: ReactNode
  actions?: ReactNode
  children: ReactNode
  i?: number
}) {
  return (
    <Reveal i={i}>
      <section className="space-y-5">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div className="space-y-1.5">
            <h2 className="font-serif text-2xl font-medium">{title}</h2>
            {description && <p className="max-w-prose text-sm text-muted-foreground">{description}</p>}
          </div>
          {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
        </div>
        {children}
      </section>
    </Reveal>
  )
}
