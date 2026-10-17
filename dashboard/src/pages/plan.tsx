import type { Plan, PlanItem } from '@server/types'
import { ChevronDown, Circle, CircleCheck } from 'lucide-react'
import { Fragment, type ReactNode } from 'react'
import { ErrorNote, Loading, Section } from '@/components/bits'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { Progress } from '@/components/ui/progress'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { useApi } from '@/lib/api'
import { localToday, plainMarkdown } from '@/lib/format'
import { cn } from '@/lib/utils'

/** Inline `code` survives as <code>; links and bold become plain text. */
function Md({ text }: { text: string }) {
  const parts = plainMarkdown(text).split(/(`[^`]+`)/)
  return (
    <>
      {parts.map((p, i) =>
        p.startsWith('`') && p.endsWith('`') ? (
          <code key={i} className="rounded bg-muted px-1 py-0.5 text-[0.85em]">{p.slice(1, -1)}</code>
        ) : (
          <Fragment key={i}>{p}</Fragment>
        ),
      )}
    </>
  )
}

function Item({ item }: { item: PlanItem }) {
  const Icon = item.done ? CircleCheck : Circle
  return (
    <li className="flex gap-2 py-1.5 text-sm">
      <Icon
        className={cn('mt-0.5 size-4 shrink-0', item.done ? 'text-[var(--good)]' : 'text-muted-foreground')}
        aria-label={item.done ? 'Done' : 'Open'}
      />
      <span className={cn('max-w-prose', item.done && 'text-muted-foreground')}><Md text={item.text} /></span>
    </li>
  )
}

function Group({ title, children }: { title?: string; children: ReactNode }) {
  return (
    <div>
      {title && <div className="mt-2 text-xs font-medium text-muted-foreground">{title}</div>}
      <ul>{children}</ul>
    </div>
  )
}

function SectionCard({ title, items }: { title: string; items: PlanItem[] }) {
  const done = items.filter((i) => i.done).length
  const groups: { title?: string; items: PlanItem[] }[] = []
  for (const it of items) {
    const last = groups.at(-1)
    if (last && last.title === it.group) last.items.push(it)
    else groups.push({ title: it.group, items: [it] })
  }
  const open = items.some((i) => !i.done)
  return (
    <Card className="gap-3 py-4">
      <Collapsible defaultOpen={open}>
        <CardHeader className="px-4">
          <CollapsibleTrigger className="group flex w-full items-start justify-between gap-3 text-left">
            <div className="min-w-0 flex-1">
              <CardTitle className="text-base">{plainMarkdown(title)}</CardTitle>
              <div className="mt-2 flex items-center gap-3">
                <Progress value={items.length ? (done / items.length) * 100 : 0} className="h-1.5" aria-label={`${title} progress`} />
                <span className="tabular shrink-0 text-sm text-muted-foreground">
                  {items.length ? `${done}/${items.length}` : 'No checklist'}
                </span>
              </div>
            </div>
            <ChevronDown className="mt-1 size-4 shrink-0 text-muted-foreground transition-transform group-data-[state=open]:rotate-180" aria-hidden />
          </CollapsibleTrigger>
        </CardHeader>
        <CollapsibleContent>
          <CardContent className="px-4 pt-2">
            {groups.map((g, i) => (
              <Group key={i} title={g.title}>
                {g.items.map((it, j) => <Item key={j} item={it} />)}
              </Group>
            ))}
          </CardContent>
        </CollapsibleContent>
      </Collapsible>
    </Card>
  )
}

export function PlanPage() {
  const { data: plan, error } = useApi<Plan>('/plan')
  if (error) return <ErrorNote message={error} />
  if (!plan) return <Loading rows={4} />
  const today = localToday()

  return (
    <div className="space-y-8">
      <Section
        title="Timeline"
        description={`From docs/plan.md${plan.updated ? `, last updated ${plan.updated}` : ''}. The highlighted row is this week.`}
      >
        <Card className="py-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Week</TableHead>
                <TableHead>Dates</TableHead>
                <TableHead>Stage</TableHead>
                <TableHead className="hidden md:table-cell">Goal</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {plan.timeline.map((w) => {
                const now = !!w.start && !!w.end && w.start <= today && today <= w.end
                return (
                  <TableRow key={w.week} className={cn(now && 'bg-muted font-medium')} aria-current={now ? 'date' : undefined}>
                    <TableCell className="tabular">{w.week}</TableCell>
                    <TableCell className="whitespace-nowrap">{w.dates}</TableCell>
                    <TableCell>{w.stage}</TableCell>
                    <TableCell className="hidden max-w-md whitespace-normal text-muted-foreground md:table-cell"><Md text={w.goal} /></TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        </Card>
      </Section>

      <Section title="Steps" description="Sections with open items start expanded. Tick items in docs/plan.md.">
        <div className="grid items-start gap-3 lg:grid-cols-2">
          {plan.sections.map((s) => <SectionCard key={s.title} title={s.title} items={s.items} />)}
        </div>
      </Section>

      <Section title="Change log">
        <Card className="py-0">
          <Collapsible>
            <CollapsibleTrigger className="group flex w-full items-center justify-between px-4 py-3 text-sm">
              {plan.changelog.length} changes, newest last
              <ChevronDown className="size-4 text-muted-foreground transition-transform group-data-[state=open]:rotate-180" aria-hidden />
            </CollapsibleTrigger>
            <CollapsibleContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Date</TableHead>
                    <TableHead>Change</TableHead>
                    <TableHead className="hidden lg:table-cell">Why</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {plan.changelog.map((c, i) => (
                    <TableRow key={i}>
                      <TableCell className="align-top whitespace-nowrap">{c.date}</TableCell>
                      <TableCell className="max-w-md align-top whitespace-normal"><Md text={c.change} /></TableCell>
                      <TableCell className="hidden max-w-md align-top whitespace-normal text-muted-foreground lg:table-cell"><Md text={c.why} /></TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CollapsibleContent>
          </Collapsible>
        </Card>
      </Section>
    </div>
  )
}
