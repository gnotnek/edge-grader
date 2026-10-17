import type { Run } from '@server/types'
import { ArrowDown, ArrowUp, CircleAlert, Search } from 'lucide-react'
import { useMemo, useState } from 'react'
import { BoardDot, BoardName, ErrorNote, Loading, Tag } from '@/components/bits'
import { RunSheet } from '@/components/run-sheet'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { useApi } from '@/lib/api'
import {
  BENCH_LABEL, BENCH_ORDER, BOARD_LABEL, BOARD_ORDER, CONDITION_LABEL, CONDITION_ORDER, MODEL_LABEL, fixed, pct,
  stampLabel,
} from '@/lib/format'

type Filters = { board: string; model: string; benchmark: string; condition: string; location: string; q: string }
type SortKey = 'stamp' | 'accuracy' | 'macro_f1' | 'energy' | 'latency'
const ALL = 'all'

function FilterSelect({ label, value, onChange, options }: {
  label: string
  value: string
  onChange: (v: string) => void
  options: [string, string][]
}) {
  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger size="sm" aria-label={label} className="min-w-32">
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
  if (k === 'energy') return r.energy?.jPerAnswer ?? null
  if (k === 'latency') return r.metrics.median_latency_s ?? null
  return r.metrics[k] ?? null
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
    <TableHead className={className} aria-sort={active ? (sort.desc ? 'descending' : 'ascending') : 'none'}>
      <Button variant="ghost" size="sm" className="-mx-2 h-7 px-2" onClick={() => setSort({ k, desc: active ? !sort.desc : true })}>
        {label}
        {active && <Icon className="size-3.5" />}
      </Button>
    </TableHead>
  )
}

export function RunsPage() {
  const { data: runs, error } = useApi<Run[]>('/runs')
  const [f, setF] = useState<Filters>({ board: ALL, model: ALL, benchmark: ALL, condition: ALL, location: ALL, q: '' })
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
      (f.condition === ALL || r.condition === f.condition) &&
      (f.location === ALL || r.location === f.location) &&
      (!q || r.id.toLowerCase().includes(q) || r.split.toLowerCase().includes(q)),
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

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <div className="relative">
          <Search className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground" aria-hidden />
          <Input value={f.q} onChange={(e) => set('q')(e.target.value)} placeholder="Search run names" className="h-8 w-52 pl-8" aria-label="Search run names" />
        </div>
        <FilterSelect label="Board" value={f.board} onChange={set('board')} options={BOARD_ORDER.map((b) => [b, BOARD_LABEL[b]])} />
        <FilterSelect label="Model" value={f.model} onChange={set('model')} options={[['e2b', 'E2B'], ['e4b', 'E4B']]} />
        <FilterSelect label="Benchmark" value={f.benchmark} onChange={set('benchmark')} options={BENCH_ORDER.map((b) => [b, BENCH_LABEL[b]])} />
        <FilterSelect label="Condition" value={f.condition} onChange={set('condition')} options={CONDITION_ORDER.map((c) => [c, CONDITION_LABEL[c]])} />
        <FilterSelect label="Location" value={f.location} onChange={set('location')} options={[['results', 'Fetched'], ['backup', 'Board only']]} />
        <span className="ml-auto text-sm text-muted-foreground">{shown.length} of {runs.length} runs</span>
      </div>

      <Card className="table-inset py-0">
        <Table className="tabular">
          <TableHeader>
            <TableRow>
              <TableHead className="pl-6">Run</TableHead>
              <TableHead className="hidden sm:table-cell">Board</TableHead>
              <SortHead label="Accuracy" k="accuracy" sort={sort} setSort={setSort} className="text-right" />
              <SortHead label="Macro F1" k="macro_f1" sort={sort} setSort={setSort} className="hidden text-right md:table-cell" />
              <SortHead label="J / answer" k="energy" sort={sort} setSort={setSort} className="hidden text-right lg:table-cell" />
              <SortHead label="Median s" k="latency" sort={sort} setSort={setSort} className="hidden text-right xl:table-cell" />
              <SortHead label="Started" k="stamp" sort={sort} setSort={setSort} className="pr-6" />
            </TableRow>
          </TableHeader>
          <TableBody>
            {shown.map((r) => (
              <TableRow key={`${r.location}${r.id}`} className="cursor-pointer transition-colors duration-200" onClick={() => setOpen(r)}>
                <TableCell className="pl-6">
                  <button type="button" className="text-left focus-visible:underline focus-visible:outline-none" onClick={() => setOpen(r)}>
                    <BoardDot board={r.board} className="mr-1.5 sm:hidden" />
                    <span className="font-medium text-[var(--heading)]">{BENCH_LABEL[r.benchmark]} {CONDITION_LABEL[r.condition]} {MODEL_LABEL[r.model]}</span>{' '}
                    <span className="font-mono text-xs text-muted-foreground">{r.split}</span>
                  </button>
                  <div className="mt-1 flex gap-1.5 empty:hidden">
                    {r.location === 'backup' && <Tag tone="yellow">Not fetched</Tag>}
                    {r.error && <Tag tone="red" icon={CircleAlert}>Unreadable</Tag>}
                  </div>
                </TableCell>
                <TableCell className="hidden sm:table-cell"><BoardName board={r.board} /></TableCell>
                <TableCell className="text-right">{pct(r.metrics.accuracy)}</TableCell>
                <TableCell className="hidden text-right md:table-cell">{fixed(r.metrics.macro_f1)}</TableCell>
                <TableCell className="hidden text-right lg:table-cell">{fixed(r.energy?.jPerAnswer, 2)}</TableCell>
                <TableCell className="hidden text-right xl:table-cell">{fixed(r.metrics.median_latency_s, 2)}</TableCell>
                <TableCell className="font-mono text-xs whitespace-nowrap text-muted-foreground">{stampLabel(r.stamp)}</TableCell>
              </TableRow>
            ))}
            {!shown.length && (
              <TableRow>
                <TableCell colSpan={7} className="py-8 text-center text-muted-foreground">No run matches these filters.</TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </Card>
      <RunSheet run={open} onClose={() => setOpen(null)} />
    </div>
  )
}
