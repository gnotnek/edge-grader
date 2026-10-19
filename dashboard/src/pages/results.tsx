import type { Board, BoardPair, Condition, Model, Reference, Run } from '@server/types'
import { ArrowDown, ArrowUp, CircleAlert, Minus, TableProperties } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Scatter, ScatterChart, XAxis, YAxis, ZAxis } from 'recharts'
import { BoardName, ErrorNote, Loading, Section } from '@/components/bits'
import { RunSheet } from '@/components/run-sheet'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import {
  type ChartConfig, ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent,
} from '@/components/ui/chart'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Toggle } from '@/components/ui/toggle'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { useApi } from '@/lib/api'
import {
  BENCH_LABEL, BENCH_ORDER, BOARD_LABEL, BOARD_ORDER, CONDITION_LABEL, CONDITION_ORDER, MODEL_LABEL, deltaPts,
  count, fixed, pct,
} from '@/lib/format'

const MODELS: Model[] = ['e2b', 'e4b']
const SPLITS = ['test_ua', 'test_uq', 'test_ud']
const boardConfig = Object.fromEntries(
  BOARD_ORDER.map((b) => [b, { label: BOARD_LABEL[b], color: `var(--${b})` }]),
) satisfies ChartConfig

/** Newest run per key; runs arrive newest first. */
function newestBy(runs: Run[], key: (r: Run) => string): Map<string, Run> {
  const out = new Map<string, Run>()
  for (const r of runs) if (!out.has(key(r))) out.set(key(r), r)
  return out
}

function Delta({ pts }: { pts: number | null }) {
  if (pts == null) return <span className="text-muted-foreground">—</span>
  const Icon = Math.abs(pts) < 0.05 ? Minus : pts > 0 ? ArrowUp : ArrowDown
  const color = Math.abs(pts) < 0.05 ? 'var(--muted-foreground)' : pts > 0 ? 'var(--good)' : 'var(--critical)'
  return (
    <span className="inline-flex items-center justify-end gap-1">
      <Icon className="size-3.5" style={{ color }} aria-hidden />
      {pts > 0 ? '+' : ''}{pts.toFixed(1)} pts
    </span>
  )
}

// 1. SciEntsBank by condition: one chart per model, a bar per board.
function ScientsbankByCondition({ runs }: { runs: Run[] }) {
  const [split, setSplit] = useState('test_ua')
  const [metric, setMetric] = useState<'accuracy' | 'macro_f1'>('accuracy')
  const [asTable, setAsTable] = useState(false)
  const latest = newestBy(
    runs.filter((r) => r.benchmark === 'scientsbank' && r.split === split && r.metrics.accuracy != null),
    (r) => `${r.board}|${r.model}|${r.condition}`,
  )
  const conditions = CONDITION_ORDER.filter((c) => [...latest.values()].some((r) => r.condition === c))
  const rows = (model: Model) =>
    conditions.map((c) => {
      const row: Record<string, string | number | null> = { condition: CONDITION_LABEL[c] }
      for (const b of BOARD_ORDER) row[b] = latest.get(`${b}|${model}|${c}`)?.metrics[metric] ?? null
      return row
    })
  const fmt = metric === 'accuracy' ? (v: number) => pct(v, 0) : (v: number) => v.toFixed(2)

  return (
    <Section
      title="SciEntsBank by condition"
      description="The newest run per board for each condition, on one split. A missing bar means that board has no run of that condition yet."
      actions={
        <>
          <ToggleGroup type="single" size="sm" variant="outline" value={split} onValueChange={(v) => v && setSplit(v)} aria-label="Split">
            {SPLITS.map((s) => <ToggleGroupItem key={s} value={s}>{s}</ToggleGroupItem>)}
          </ToggleGroup>
          <ToggleGroup type="single" size="sm" variant="outline" value={metric} onValueChange={(v) => v && setMetric(v as typeof metric)} aria-label="Metric">
            <ToggleGroupItem value="accuracy">Accuracy</ToggleGroupItem>
            <ToggleGroupItem value="macro_f1">Macro F1</ToggleGroupItem>
          </ToggleGroup>
          <Toggle size="sm" variant="outline" pressed={asTable} onPressedChange={setAsTable} aria-label="Show as table">
            <TableProperties /> Table
          </Toggle>
        </>
      }
    >
      <div className="grid gap-3 lg:grid-cols-2">
        {MODELS.map((model) => (
          <Card key={model} className="gap-2">
            <CardHeader><CardTitle className="font-serif text-xl font-medium">{MODEL_LABEL[model]}</CardTitle></CardHeader>
            <CardContent>
              {asTable ? (
                <Table className="tabular">
                  <TableHeader>
                    <TableRow>
                      <TableHead>Condition</TableHead>
                      {BOARD_ORDER.map((b) => <TableHead key={b} className="text-right">{BOARD_LABEL[b]}</TableHead>)}
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {rows(model).map((r) => (
                      <TableRow key={r.condition as string}>
                        <TableCell>{r.condition}</TableCell>
                        {BOARD_ORDER.map((b) => (
                          <TableCell key={b} className="text-right">
                            {r[b] == null ? '—' : metric === 'accuracy' ? pct(r[b] as number) : fixed(r[b] as number)}
                          </TableCell>
                        ))}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              ) : (
                <ChartContainer config={boardConfig} className="aspect-auto w-full" style={{ height: 56 + conditions.length * 44 }}>
                  <BarChart data={rows(model)} layout="vertical" margin={{ left: 8, right: 16 }} barGap={2} barSize={8}>
                    <CartesianGrid horizontal={false} stroke="var(--grid)" />
                    <XAxis type="number" domain={[0, 1]} tickFormatter={fmt} tickLine={false} axisLine={false} />
                    <YAxis type="category" dataKey="condition" width={132} tickLine={false} axisLine={false} />
                    <ChartTooltip
                      cursor={{ fill: 'var(--muted)', opacity: 0.6 }}
                      content={
                        <ChartTooltipContent
                          formatter={(value, name) => (
                            <div className="flex w-full items-center gap-2">
                              <span className="size-2.5 shrink-0 rounded-[2px]" style={{ background: `var(--color-${name})` }} />
                              <span className="text-muted-foreground">{BOARD_LABEL[name as Board]}</span>
                              <span className="tabular ml-auto font-medium">
                                {metric === 'accuracy' ? pct(Number(value)) : fixed(Number(value))}
                              </span>
                            </div>
                          )}
                        />
                      }
                    />
                    <ChartLegend content={<ChartLegendContent />} />
                    {BOARD_ORDER.map((b) => <Bar key={b} dataKey={b} fill={`var(--color-${b})`} radius={[0, 4, 4, 0]} isAnimationActive={false} />)}
                  </BarChart>
                </ChartContainer>
              )}
            </CardContent>
          </Card>
        ))}
      </div>
    </Section>
  )
}

// 2. Baseline against RAG, Kaggle, every benchmark.
function BaselineVsRag({ runs }: { runs: Run[] }) {
  const kaggle = runs.filter((r) => r.board === 'kaggle' && r.metrics.accuracy != null)
  const latest = newestBy(kaggle, (r) => `${r.benchmark}|${r.split}|${r.model}|${r.condition}`)
  const acc = (b: string, s: string, m: Model, c: Condition) => latest.get(`${b}|${s}|${m}|${c}`)?.metrics.accuracy ?? null
  const rows = BENCH_ORDER.flatMap((bench) => {
    const splits = [...new Set(kaggle.filter((r) => r.benchmark === bench && r.condition !== 'baseline').map((r) => r.split))].sort()
    return splits.flatMap((split) =>
      MODELS.map((model) => ({
        bench, split, model,
        base: acc(bench, split, model, 'baseline'),
        en: acc(bench, split, model, 'rag-en'),
        id: acc(bench, split, model, 'rag-id'),
      })),
    )
  })
  return (
    <Section
      i={2}
      title="Baseline against RAG"
      description="Kaggle T4, the newest run per cell. News accuracy is the Qwen3-14B judge's. Differences are in percentage points; significance is in docs/results/rag.md."
    >
      <Card className="table-inset py-0">
        <Table className="tabular">
          <TableHeader>
            <TableRow>
              <TableHead>Benchmark</TableHead>
              <TableHead>Split</TableHead>
              <TableHead>Model</TableHead>
              <TableHead className="text-right">Baseline</TableHead>
              <TableHead className="text-right">RAG-en</TableHead>
              <TableHead className="text-right">Change</TableHead>
              <TableHead className="hidden text-right md:table-cell">RAG-id</TableHead>
              <TableHead className="hidden text-right md:table-cell">Change</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((r) => (
              <TableRow key={`${r.bench}${r.split}${r.model}`}>
                <TableCell>{BENCH_LABEL[r.bench]}</TableCell>
                <TableCell className="text-muted-foreground">{r.split}</TableCell>
                <TableCell>{MODEL_LABEL[r.model]}</TableCell>
                <TableCell className="text-right">{pct(r.base)}</TableCell>
                <TableCell className="text-right">{pct(r.en)}</TableCell>
                <TableCell className="text-right"><Delta pts={deltaPts(r.base, r.en)} /></TableCell>
                <TableCell className="hidden text-right md:table-cell">{pct(r.id)}</TableCell>
                <TableCell className="hidden text-right md:table-cell"><Delta pts={deltaPts(r.base, r.id)} /></TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Card>
    </Section>
  )
}

function PValue({ p }: { p: number | undefined }) {
  if (p == null) return <>—</>
  if (p >= 0.05) return <>{fixed(p)}</>
  return (
    <span className="inline-flex items-center gap-1">
      <CircleAlert className="size-3.5" style={{ color: 'var(--warning)' }} aria-hidden />
      {fixed(p)} <span className="text-muted-foreground">(p &lt; 0.05)</span>
    </span>
  )
}

// 3. Board against Kaggle, and Pi against Jetson.
function BoardAgreement({ runs, onOpen }: { runs: Run[]; onOpen: (r: Run) => void }) {
  const pairs = useApi<{ file: string | null; pairs: BoardPair[] }>('/pairs')
  const checked = runs.filter((r) => r.vsKaggle)
  return (
    <Section
      i={3}
      title="Boards against Kaggle"
      description="From the board-check files in results/estimates/: the same answers graded on a board and on Kaggle, with an exact McNemar test. Unadjusted p; the phase 0 gate used the Holm correction."
    >
      <Card className="table-inset py-0">
        <Table className="tabular">
          <TableHeader>
            <TableRow>
              <TableHead>Run</TableHead>
              <TableHead>Board</TableHead>
              <TableHead className="text-right">Board</TableHead>
              <TableHead className="text-right">Kaggle</TableHead>
              <TableHead className="hidden text-right md:table-cell">Same grade</TableHead>
              <TableHead className="text-right">McNemar p</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {checked.map((r) => (
              <TableRow key={r.id} className="cursor-pointer" onClick={() => onOpen(r)}>
                <TableCell>
                  {BENCH_LABEL[r.benchmark]} {CONDITION_LABEL[r.condition]} {MODEL_LABEL[r.model]}{' '}
                  <span className="text-muted-foreground">{r.split}</span>
                </TableCell>
                <TableCell><BoardName board={r.board} /></TableCell>
                <TableCell className="text-right">{pct(r.vsKaggle!.boardAccuracy)}</TableCell>
                <TableCell className="text-right">{pct(r.vsKaggle!.kaggleAccuracy)}</TableCell>
                <TableCell className="hidden text-right md:table-cell">
                  {((r.vsKaggle!.sameGrade / r.vsKaggle!.n) * 100).toFixed(1)}%
                </TableCell>
                <TableCell className="text-right"><PValue p={r.vsKaggle!.p} /></TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Card>
      {pairs.data && pairs.data.pairs.length > 0 && (
        <Card className="gap-2">
          <CardHeader>
            <CardTitle className="font-serif text-xl font-medium">Pi 5 against Jetson</CardTitle>
            <p className="text-sm text-muted-foreground">How many times slower and more energy the Pi takes per answer. From {pairs.data.file}.</p>
          </CardHeader>
          <CardContent className="table-inset px-0">
            <Table className="tabular">
              <TableHeader>
                <TableRow>
                  <TableHead className="pl-4">Run on the Pi</TableHead>
                  <TableHead className="text-right">Pi</TableHead>
                  <TableHead className="text-right">Jetson</TableHead>
                  <TableHead className="hidden text-right md:table-cell">Slower</TableHead>
                  <TableHead className="hidden text-right md:table-cell">More energy</TableHead>
                  <TableHead className="pr-4 text-right">McNemar p</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {pairs.data.pairs.map((p) => (
                  <TableRow key={p.first}>
                    <TableCell className="pl-4">{p.first}</TableCell>
                    <TableCell className="text-right">{pct(p.first_accuracy)}</TableCell>
                    <TableCell className="text-right">{pct(p.second_accuracy)}</TableCell>
                    <TableCell className="hidden text-right md:table-cell">{p.latency_ratio != null ? `${p.latency_ratio}×` : '—'}</TableCell>
                    <TableCell className="hidden text-right md:table-cell">{p.energy_ratio != null ? `${p.energy_ratio}×` : '—'}</TableCell>
                    <TableCell className="pr-4 text-right"><PValue p={p.mcnemar.p} /></TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
    </Section>
  )
}

interface Point {
  x: number
  y: number
  run: Run
}

function Dot(props: { cx?: number; cy?: number; payload?: Point; fill?: string }) {
  const { cx, cy, payload, fill } = props
  if (cx == null || cy == null || !payload) return null
  const hollow = payload.run.model === 'e4b'
  return (
    <circle
      cx={cx} cy={cy} r={5.5} strokeWidth={2}
      stroke={hollow ? fill : 'var(--card)'} fill={hollow ? 'var(--card)' : fill}
      style={{ cursor: 'pointer' }}
    />
  )
}

// 4. Accuracy against energy per answer: the start of stage 4's main chart.
function AccuracyVsEnergy({ runs, onOpen }: { runs: Run[]; onOpen: (r: Run) => void }) {
  const byBoard = useMemo(() => {
    const pts = runs
      .filter((r) => r.benchmark === 'scientsbank' && r.energy && r.metrics.accuracy != null)
      .map((r) => ({ x: r.energy!.jPerAnswer, y: r.metrics.accuracy!, run: r }))
    return (['jetson', 'pi5'] as Board[]).map((b) => ({ board: b, pts: pts.filter((p) => p.run.board === b) }))
  }, [runs])
  const n = byBoard.reduce((a, b) => a + b.pts.length, 0)

  return (
    <Section
      i={1}
      title="Accuracy against energy per answer"
      description={`SciEntsBank board runs with a board check (${n} so far). Energy is board DC power over the grading window. The x axis is logarithmic, so each step is three times the energy. Filled dots are E2B, hollow are E4B; click one to open the run.`}
    >
      <Card>
        <CardContent>
          <ChartContainer config={boardConfig} className="aspect-auto h-80 w-full">
            <ScatterChart margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
              <CartesianGrid stroke="var(--grid)" />
              <XAxis
                type="number" dataKey="x" scale="log" domain={[1, 1000]} ticks={[1, 3, 10, 30, 100, 300, 1000]} name="J per answer"
                tickFormatter={(v: number) => `${v} J`} tickLine={false} axisLine={false}
              />
              <YAxis type="number" dataKey="y" domain={[0.4, 0.8]} tickFormatter={(v: number) => pct(v, 0)} tickLine={false} axisLine={false} width={44} />
              <ZAxis range={[80, 80]} />
              <ChartTooltip
                cursor={false}
                content={({ active, payload }) => {
                  const p = active && (payload?.[0]?.payload as Point | undefined)
                  if (!p) return null
                  const r = p.run
                  return (
                    <div className="rounded-md border bg-popover px-3 py-2 text-xs shadow-sm">
                      <div className="font-medium">{CONDITION_LABEL[r.condition]} {MODEL_LABEL[r.model]} {r.split}</div>
                      <div className="text-muted-foreground"><BoardName board={r.board} /></div>
                      <div className="tabular mt-1">{pct(p.y)} accuracy, {fixed(p.x, 2)} J per answer</div>
                    </div>
                  )
                }}
              />
              <ChartLegend content={<ChartLegendContent />} />
              {byBoard.map(({ board, pts }) => (
                <Scatter
                  key={board} name={board} dataKey={board} data={pts} fill={`var(--color-${board})`} isAnimationActive={false}
                  shape={(p: unknown) => <Dot {...(p as Parameters<typeof Dot>[0])} />}
                  onClick={(d: unknown) => {
                    const run = (d as { payload?: Point })?.payload?.run
                    if (run) onOpen(run)
                  }}
                />
              ))}
            </ScatterChart>
          </ChartContainer>
        </CardContent>
      </Card>
    </Section>
  )
}

// 5. agentic-edge's MMLU-Pro board runs: the lab's numbers, copied, shown beside ours and never mixed in.
function MmluProReference() {
  const { data: ref, error } = useApi<Reference>('/reference')
  if (error) return <ErrorNote message={error} />
  if (!ref) return null
  if (ref.error) return <ErrorNote message={ref.error} />
  if (!ref.rows.length) return null
  return (
    <Section
      i={3}
      title="MMLU-Pro on the boards"
      description={
        <>
          agentic-edge's runs, copied {ref.copied ? `on ${ref.copied} ` : ''}into <code className="font-mono text-xs">{ref.source}</code>.
          They are the lab's numbers, not this project's, so they appear only here and as reference wells on the
          coverage map. Same 300 questions (s1 to s3) for every row. The change is against llama.cpp with thinking off on
          the same board.
        </>
      }
    >
      <Card className="table-inset py-0">
        <Table className="tabular">
          <TableHeader>
            <TableRow>
              <TableHead>Variation</TableHead>
              <TableHead>Board</TableHead>
              <TableHead>Model</TableHead>
              <TableHead className="text-right">Accuracy</TableHead>
              <TableHead className="text-right">Change</TableHead>
              <TableHead className="hidden text-right md:table-cell">McNemar p</TableHead>
              <TableHead className="hidden text-right lg:table-cell">J / question</TableHead>
              <TableHead className="hidden text-right lg:table-cell">Energy, 300 q</TableHead>
              <TableHead className="hidden text-right xl:table-cell">Time</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {ref.rows.map((r) => (
              <TableRow key={`${r.section}${r.board}${r.model}`}>
                <TableCell className="whitespace-normal">
                  <span className="text-[var(--heading)]">{r.label}</span>
                  {r.note && <span className="block text-xs text-muted-foreground">{r.note}</span>}
                </TableCell>
                <TableCell><BoardName board={r.board} /></TableCell>
                <TableCell>{MODEL_LABEL[r.model]}</TableCell>
                <TableCell className="text-right">{r.accuracy !== undefined ? pct(r.accuracy) : '—'}</TableCell>
                <TableCell className="text-right"><Delta pts={deltaPts(r.baseline, r.accuracy)} /></TableCell>
                <TableCell className="hidden text-right md:table-cell"><PValue p={r.mcnemarP} /></TableCell>
                <TableCell className="hidden text-right lg:table-cell">{r.jPerQuestion !== undefined ? count(r.jPerQuestion) : '—'}</TableCell>
                <TableCell className="hidden text-right lg:table-cell">{r.energyWh !== undefined ? `${fixed(r.energyWh, 1)} Wh` : '—'}</TableCell>
                <TableCell className="hidden text-right xl:table-cell">{r.wallMin !== undefined ? `${count(r.wallMin)} min` : '—'}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Card>
    </Section>
  )
}

export function ResultsPage() {
  const { data: runs, error } = useApi<Run[]>('/runs')
  const [open, setOpen] = useState<Run | null>(null)
  if (error) return <ErrorNote message={error} />
  if (!runs) return <Loading rows={4} />
  return (
    <div className="space-y-16">
      <ScientsbankByCondition runs={runs} />
      <AccuracyVsEnergy runs={runs} onOpen={setOpen} />
      <BaselineVsRag runs={runs} />
      <MmluProReference />
      <BoardAgreement runs={runs} onOpen={setOpen} />
      <RunSheet run={open} onClose={() => setOpen(null)} />
    </div>
  )
}
