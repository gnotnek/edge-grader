import type { Run } from '@server/types'
import { ArrowDown, ArrowUp, CircleAlert, Search, X } from 'lucide-react'
import { useMemo, useState } from 'react'
import { BoardName, ErrorNote, Loading, Tag } from '@/components/bits'
import { RunSheet } from '@/components/run-sheet'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { useApi } from '@/lib/api'
import { cn } from '@/lib/utils'
import {
  BENCH_LABEL, BENCH_ORDER, BOARD_LABEL, BOARD_ORDER, CONDITION_LABEL, CONDITION_ORDER, MODEL_LABEL, count, fixed,
  pct, stampLabel, trainedOn,
} from '@/lib/format'

type Filters = { board: string; model: string; benchmark: string; split: string; condition: string; location: string; q: string }
const ALL = 'all'
const NO_FILTERS: Filters = { board: ALL, model: ALL, benchmark: ALL, split: ALL, condition: ALL, location: ALL, q: '' }

// Every result a run can store, as columns; a run without one shows a dash.
// Grouped under a header row: the 3-way labels (correct / contradictory / incorrect), the same
// answers 2-way (correct / not), then everything else.
const COLUMNS: { key: string; group: string; label: string; get: (r: Run) => number | null | undefined; show: (v: number | null | undefined) => string }[] = [
  { key: 'n', group: '', label: 'n', get: (r) => r.metrics.n, show: count },
  { key: 'accuracy', group: '3-way', label: 'Accuracy', get: (r) => r.metrics.accuracy, show: (v) => pct(v) },
  { key: 'macro_f1', group: '3-way', label: 'Macro F1', get: (r) => r.metrics.macro_f1, show: (v) => fixed(v) },
  { key: 'weighted_f1', group: '3-way', label: 'Weighted F1', get: (r) => r.metrics.weighted_f1, show: (v) => fixed(v) },
  { key: 'two_way_accuracy', group: '2-way', label: 'Accuracy', get: (r) => r.metrics.two_way_accuracy, show: (v) => pct(v) },
  { key: 'two_way_macro_f1', group: '2-way', label: 'Macro F1', get: (r) => r.metrics.two_way_macro_f1, show: (v) => fixed(v) },
  { key: 'two_way_weighted_f1', group: '2-way', label: 'Weighted F1', get: (r) => r.metrics.two_way_weighted_f1, show: (v) => fixed(v) },
  { key: 'qwk', group: 'Other', label: 'QWK', get: (r) => r.metrics.qwk, show: (v) => fixed(v) },
  { key: 'strict_accuracy', group: 'Other', label: 'Strict acc.', get: (r) => r.metrics.strict_accuracy, show: (v) => pct(v) },
  { key: 'unparsed', group: 'Other', label: 'Unparsed', get: (r) => r.metrics.unparsed ?? r.metrics.no_answer ?? r.metrics.no_number, show: count },
  { key: 'energy', group: 'Other', label: 'J / answer', get: (r) => r.energy?.jPerAnswer, show: (v) => fixed(v, 2) },
  { key: 'latency', group: 'Other', label: 'Median s', get: (r) => r.metrics.median_latency_s, show: (v) => fixed(v, 2) },
  { key: 'wall', group: 'Other', label: 'Wall h', get: (r) => (r.metrics.wall_s == null ? null : r.metrics.wall_s / 3600), show: (v) => fixed(v, 2) },
  { key: 'tokens', group: 'Other', label: 'Output tokens', get: (r) => r.metrics.completion_tokens, show: count },
]
type SortKey = string

function FilterSelect({ label, value, onChange, options }: {
  label: string
  value: string
  onChange: (v: string) => void
  options: [string, string][]
}) {
  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger size="sm" aria-label={label} className="w-full">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value={ALL}>All {label.toLowerCase()}s</SelectItem>
        {options.map(([v, l]) => <SelectItem key={v} value={v}>{l}</SelectItem>)}
      </SelectContent>
    </Select>
  )
}

function sortValue(r: Run, k: SortKey): number | string | null {
  if (k === 'stamp') return r.stamp
  return COLUMNS.find((c) => c.key === k)?.get(r) ?? null
}

function SortHead({ label, k, sort, setSort, className }: {
  label: string
  k: SortKey
  sort: { k: SortKey; desc: boolean }
  setSort: (s: { k: SortKey; desc: boolean }) => void
  className?: string
}) {
  const active = sort.k === k
  const Icon = sort.desc ? ArrowDown : ArrowUp
  return (
    <TableHead className={cn('whitespace-nowrap', className)} aria-sort={active ? (sort.desc ? 'descending' : 'ascending') : 'none'}>
      <Button variant="ghost" size="sm" className="-mx-2 h-7 px-2" onClick={() => setSort({ k, desc: active ? !sort.desc : true })}>
        {label}
        {active && <Icon className="size-3.5" />}
      </Button>
    </TableHead>
  )
}

export function RunsPage() {
  const { data: runs, error } = useApi<Run[]>('/runs')
  const [f, setF] = useState<Filters>(NO_FILTERS)
  const [sort, setSort] = useState<{ k: SortKey; desc: boolean }>({ k: 'stamp', desc: true })
  const [open, setOpen] = useState<Run | null>(null)
  const set = (k: keyof Filters) => (v: string) => setF((x) => ({ ...x, [k]: v }))

  const shown = useMemo(() => {
    if (!runs) return []
    const q = f.q.trim().toLowerCase()
    const out = runs.filter((r) =>
      (f.board === ALL || r.board === f.board) &&
      (f.model === ALL || r.model === f.model) &&
      (f.benchmark === ALL || r.benchmark === f.benchmark) &&
      (f.split === ALL || r.split === f.split) &&
      (f.condition === ALL || r.condition === f.condition) &&
      (f.location === ALL || r.location === f.location) &&
      (!q || `${r.id} ${BENCH_LABEL[r.benchmark]} ${CONDITION_LABEL[r.condition]}`.toLowerCase().includes(q)),
    )
    return out.sort((a, b) => {
      const va = sortValue(a, sort.k)
      const vb = sortValue(b, sort.k)
      if (va == null) return 1
      if (vb == null) return -1
      const c = va < vb ? -1 : va > vb ? 1 : 0
      return sort.desc ? -c : c
    })
  }, [runs, f, sort])

  if (error) return <ErrorNote message={error} />
  if (!runs) return <Loading rows={5} />
  const splits = [...new Set(runs.filter((r) => f.benchmark === ALL || r.benchmark === f.benchmark).map((r) => r.split))].sort()
  const filtered = Object.entries(f).some(([k, v]) => (k === 'q' ? v.trim() !== '' : v !== ALL))

  return (
    <div className="space-y-4">
      <div className="space-y-3">
        <div className="flex flex-wrap items-center gap-3">
          <div className="relative min-w-52 flex-1 sm:max-w-sm">
            <Search className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground" aria-hidden />
            <Input value={f.q} onChange={(e) => set('q')(e.target.value)} placeholder="Search runs, e.g. fine-tuned or test_ua" className="h-8 pl-8" aria-label="Search runs" />
          </div>
          {filtered && (
            <Button variant="ghost" size="sm" onClick={() => setF(NO_FILTERS)}>
              <X /> Clear filters
            </Button>
          )}
          <span className="ml-auto text-sm text-muted-foreground">
            <span className="tabular font-medium text-[var(--heading)]">{shown.length}</span> of {runs.length} runs
          </span>
        </div>
        <div className="grid grid-cols-2 gap-2 md:grid-cols-3 2xl:grid-cols-6">
          <FilterSelect label="Board" value={f.board} onChange={set('board')} options={BOARD_ORDER.map((b) => [b, BOARD_LABEL[b]])} />
          <FilterSelect label="Model" value={f.model} onChange={set('model')} options={[['e2b', 'E2B'], ['e4b', 'E4B']]} />
          <FilterSelect label="Benchmark" value={f.benchmark} onChange={(v) => setF((x) => ({ ...x, benchmark: v, split: ALL }))} options={BENCH_ORDER.map((b) => [b, BENCH_LABEL[b]])} />
          <FilterSelect label="Split" value={f.split} onChange={set('split')} options={splits.map((s) => [s, s])} />
          <FilterSelect label="Condition" value={f.condition} onChange={set('condition')} options={CONDITION_ORDER.map((c) => [c, CONDITION_LABEL[c]])} />
          <FilterSelect label="Location" value={f.location} onChange={set('location')} options={[['results', 'Fetched'], ['backup', 'Board only']]} />
        </div>
      </div>

      <Card className="table-inset py-0">
        <Table className="tabular">
          <TableHeader>
            <TableRow className="hover:bg-transparent">
              <TableHead className="sticky left-0 z-[1] bg-card" />
              <TableHead />
              <TableHead />
              {COLUMNS.reduce<{ group: string; span: number }[]>((acc, c) => {
                const last = acc.at(-1)
                if (last && last.group === c.group) last.span++
                else acc.push({ group: c.group, span: 1 })
                return acc
              }, []).map((g, i) => (
                <TableHead key={i} colSpan={g.span} className={cn('h-8 text-center text-xs', g.group && 'border-x')}>
                  {g.group === '3-way' ? '3-way labels' : g.group === '2-way' ? '2-way (correct or not)' : g.group}
                </TableHead>
              ))}
              <TableHead />
            </TableRow>
            <TableRow>
              <TableHead className="sticky left-0 z-[1] bg-card pl-6">Run</TableHead>
              <TableHead>Board</TableHead>
              <TableHead className="whitespace-nowrap">Trained on</TableHead>
              {COLUMNS.map((c) => <SortHead key={c.key} label={c.label} k={c.key} sort={sort} setSort={setSort} className="text-right" />)}
              <SortHead label="Started" k="stamp" sort={sort} setSort={setSort} className="pr-6" />
            </TableRow>
          </TableHeader>
          <TableBody>
            {shown.map((r) => (
              <TableRow key={`${r.location}${r.id}`} className="group/row cursor-pointer transition-colors duration-200" onClick={() => setOpen(r)}>
                <TableCell className="sticky left-0 z-[1] bg-card pl-6 whitespace-nowrap transition-colors duration-200 group-hover/row:bg-[color-mix(in_oklab,var(--muted)_50%,var(--card))]">
                  <button type="button" className="text-left focus-visible:underline focus-visible:outline-none" onClick={() => setOpen(r)}>
                    <span className="font-medium text-[var(--heading)]">{BENCH_LABEL[r.benchmark]} {CONDITION_LABEL[r.condition]} {MODEL_LABEL[r.model]}</span>{' '}
                    <span className="font-mono text-xs text-muted-foreground">{r.split}</span>
                  </button>
                  <div className="mt-1 flex gap-1.5 empty:hidden">
                    {r.location === 'backup' && <Tag tone="yellow">Not fetched</Tag>}
                    {r.error && <Tag tone="red" icon={CircleAlert}>Unreadable</Tag>}
                  </div>
                </TableCell>
                <TableCell><BoardName board={r.board} /></TableCell>
                <TableCell className={cn('whitespace-nowrap', !trainedOn(r.condition) && 'text-muted-foreground')}>
                  {trainedOn(r.condition) ?? '—'}
                </TableCell>
                {COLUMNS.map((c) => {
                  const v = c.get(r)
                  return <TableCell key={c.key} className={cn('text-right', v == null && 'text-muted-foreground')}>{c.show(v)}</TableCell>
                })}
                <TableCell className="font-mono text-xs whitespace-nowrap text-muted-foreground">{stampLabel(r.stamp)}</TableCell>
              </TableRow>
            ))}
            {!shown.length && (
              <TableRow>
                <TableCell colSpan={COLUMNS.length + 4} className="py-8 text-center text-muted-foreground">No run matches these filters.</TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </Card>
      <RunSheet run={open} onClose={() => setOpen(null)} />
    </div>
  )
}
