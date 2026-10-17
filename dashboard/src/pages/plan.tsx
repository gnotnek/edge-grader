import type { Plan, PlanItem } from '@server/types'
import { Circle, CircleCheck, Minus, Plus } from 'lucide-react'
import { Fragment, type ReactNode } from 'react'
import { ErrorNote, Loading, Section, Tag } from '@/components/bits'
import { Card } from '@/components/ui/card'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { Progress } from '@/components/ui/progress'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { useApi } from '@/lib/api'
import { localToday, plainMarkdown } from '@/lib/format'
import { cn } from '@/lib/utils'

/** Inline `code` survives as <code>; links, bold and emoji become plain text. */
function Md({ text }: { text: string }) {
  const parts = plainMarkdown(text).split(/(`[^`]+`)/)
  return (
    <>
      {parts.map((p, i) =>
        p.startsWith('`') && p.endsWith('`') ? (
          <code key={i} className="rounded bg-muted px-1 py-0.5 font-mono text-[0.8em]">{p.slice(1, -1)}</code>
        ) : (
          <Fragment key={i}>{p}</Fragment>
        ),
      )}
    </>
  )
}

/** A + / − toggle, the plan's accordion marker. */
function PlusMinus() {
  return (
    <span className="grid size-6 shrink-0 place-items-center text-muted-foreground" aria-hidden>
      <Plus className="size-4 group-data-[state=open]:hidden" />
      <Minus className="hidden size-4 group-data-[state=open]:block" />
    </span>
  )
}

function Item({ item }: { item: PlanItem }) {
  const Icon = item.done ? CircleCheck : Circle
  return (
    <li className="flex gap-2.5 py-1.5 text-sm">
      <Icon
        className={cn('mt-1 size-4 shrink-0', item.done ? 'text-[var(--good)]' : 'text-muted-foreground')}
        aria-label={item.done ? 'Done' : 'Open'}
      />
      <span className={cn('max-w-prose', item.done ? 'text-muted-foreground' : 'text-[var(--heading)]')}><Md text={item.text} /></span>
    </li>
  )
}

function Group({ title, children }: { title?: string; children: ReactNode }) {
  return (
    <div className="mt-2 first:mt-0">
      {title && <div className="mt-4 mb-1 text-xs font-medium text-muted-foreground">{title}</div>}
      <ul>{children}</ul>
    </div>
  )
}

function SectionRow({ title, items }: { title: string; items: PlanItem[] }) {
  const done = items.filter((i) => i.done).length
  const groups: { title?: string; items: PlanItem[] }[] = []
  for (const it of items) {
    const last = groups.at(-1)
    if (last && last.title === it.group) last.items.push(it)
    else groups.push({ title: it.group, items: [it] })
  }
  const open = items.some((i) => !i.done)
  const complete = items.length > 0 && done === items.length
  return (
    <Collapsible defaultOpen={open} className="group border-b">
      <CollapsibleTrigger className="flex w-full items-center gap-4 py-4 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
        <span className="min-w-0 flex-1">
          <span className="block font-medium text-[var(--heading)]">{plainMarkdown(title)}</span>
          {items.length > 0 && (
            <span className="mt-2 flex max-w-sm items-center gap-3">
              <Progress value={(done / items.length) * 100} className="h-1" aria-label={`${title} progress`} />
              <span className="tabular shrink-0 text-xs text-muted-foreground">{done}/{items.length}</span>
            </span>
          )}
        </span>
        {complete ? <Tag tone="green">Done</Tag> : items.length === 0 ? <Tag tone="neutral">No checklist</Tag> : <Tag tone="yellow">Open</Tag>}
        <PlusMinus />
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="pb-5 pl-0.5">
          {items.length === 0 && <p className="text-sm text-muted-foreground">This section has no checklist in docs/plan.md.</p>}
          {groups.map((g, i) => (
            <Group key={i} title={g.title}>
              {g.items.map((it, j) => <Item key={j} item={it} />)}
            </Group>
          ))}
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}

export function PlanPage() {
  const { data: plan, error } = useApi<Plan>('/plan')
  if (error) return <ErrorNote message={error} />
  if (!plan) return <Loading rows={4} />
  const today = localToday()

  return (
    <div className="space-y-16">
      <Section
        title="Timeline"
        description={`From docs/plan.md${plan.updated ? `, last updated ${plan.updated}` : ''}. This week is marked.`}
      >
        <Card className="table-inset py-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="pl-6">Week</TableHead>
                <TableHead>Dates</TableHead>
                <TableHead>Stage</TableHead>
                <TableHead className="hidden pr-6 md:table-cell">Goal</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {plan.timeline.map((w) => {
                const now = !!w.start && !!w.end && w.start <= today && today <= w.end
                return (
                  <TableRow key={w.week} className={cn(now && 'bg-muted')} aria-current={now ? 'date' : undefined}>
                    <TableCell className="tabular pl-6">{w.week}</TableCell>
                    <TableCell className="whitespace-nowrap">
                      {w.dates} {now && <Tag tone="blue" className="ml-1">This week</Tag>}
                    </TableCell>
                    <TableCell className="text-[var(--heading)]">{w.stage}</TableCell>
                    <TableCell className="hidden max-w-md pr-6 whitespace-normal text-muted-foreground md:table-cell"><Md text={w.goal} /></TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        </Card>
      </Section>

      <Section i={1} title="Steps" description="Sections with open items start expanded. Tick items in docs/plan.md.">
        <div className="border-t">
          {plan.sections.map((s) => <SectionRow key={s.title} title={s.title} items={s.items} />)}
        </div>
      </Section>

      <Section i={2} title="Change log" description="Why the plan moved, newest last.">
        <Collapsible className="group border-y">
          <CollapsibleTrigger className="flex w-full items-center justify-between gap-4 py-4 text-left text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
            <span className="text-[var(--heading)]">{plan.changelog.length} changes</span>
            <PlusMinus />
          </CollapsibleTrigger>
          <CollapsibleContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="pl-0">Date</TableHead>
                  <TableHead>Change</TableHead>
                  <TableHead className="hidden lg:table-cell">Why</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {plan.changelog.map((c, i) => (
                  <TableRow key={i}>
                    <TableCell className="pl-0 align-top font-mono text-xs whitespace-nowrap">{c.date}</TableCell>
                    <TableCell className="max-w-md align-top whitespace-normal"><Md text={c.change} /></TableCell>
                    <TableCell className="hidden max-w-md align-top whitespace-normal text-muted-foreground lg:table-cell"><Md text={c.why} /></TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CollapsibleContent>
        </Collapsible>
      </Section>
    </div>
  )
}
