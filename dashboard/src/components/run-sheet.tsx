import type { Run } from '@server/types'
import type { ReactNode } from 'react'
import { BoardName, ErrorNote, Loading } from '@/components/bits'
import { Badge } from '@/components/ui/badge'
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from '@/components/ui/sheet'
import { useApi } from '@/lib/api'
import { BENCH_LABEL, CONDITION_LABEL, MODEL_LABEL, count, fixed, pct, stampLabel } from '@/lib/format'

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex justify-between gap-4 border-b py-1.5 text-sm last:border-0">
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="tabular text-right">{children}</dd>
    </div>
  )
}

function Details({ id }: { id: string }) {
  const { data: run, error } = useApi<Run>(`/runs/${encodeURIComponent(id)}`)
  if (error) return <ErrorNote message={error} />
  if (!run) return <Loading rows={2} />
  const m = run.metrics
  const server = (run.summary as { server?: { build_info?: string; model_path?: string } } | undefined)?.server
  return (
    <div className="space-y-6 px-4 pb-6">
      {run.error && <ErrorNote message={run.error} />}
      <dl>
        <Row label="Board"><BoardName board={run.board} /></Row>
        <Row label="Benchmark">{BENCH_LABEL[run.benchmark]} · {run.split}</Row>
        <Row label="Condition">{CONDITION_LABEL[run.condition]}</Row>
        <Row label="Started">{stampLabel(run.stamp)}</Row>
        <Row label="Where">{run.location === 'results' ? 'Fetched into results/' : 'Only on the board backup'}</Row>
        {server?.build_info && <Row label="llama.cpp build">{server.build_info}</Row>}
      </dl>
      <div>
        <h3 className="mb-1 text-sm font-medium">Metrics</h3>
        {Object.keys(m).length === 0 ? (
          <p className="text-sm text-muted-foreground">No summary yet. The run may still be going on the board.</p>
        ) : (
          <dl>
            <Row label="Answers">{count(m.n)}</Row>
            <Row label={run.benchmark === 'news' ? 'Accuracy (judged)' : 'Accuracy'}>{pct(m.accuracy)}</Row>
            {m.macro_f1 != null && <Row label="Macro F1">{fixed(m.macro_f1)}</Row>}
            {m.strict_accuracy != null && <Row label="Strict accuracy">{pct(m.strict_accuracy)}</Row>}
            {m.unparsed != null && <Row label="Unparsed">{count(m.unparsed)}</Row>}
            {m.errors != null && <Row label="Errors">{count(m.errors)}</Row>}
            {m.median_latency_s != null && <Row label="Median latency">{fixed(m.median_latency_s, 2)} s</Row>}
            {m.wall_s != null && <Row label="Wall time">{fixed(m.wall_s / 3600, 2)} h</Row>}
            {m.prompt_tokens != null && <Row label="Prompt tokens">{count(m.prompt_tokens)}</Row>}
            {m.completion_tokens != null && <Row label="Completion tokens">{count(m.completion_tokens)}</Row>}
          </dl>
        )}
      </div>
      {(run.energy || run.vsKaggle) && (
        <div>
          <h3 className="mb-1 text-sm font-medium">Board check</h3>
          <dl>
            {run.energy && <Row label="Energy per answer">{fixed(run.energy.jPerAnswer, 2)} J</Row>}
            {run.energy && <Row label="Mean power">{fixed(run.energy.meanW, 1)} W</Row>}
            {run.vsKaggle && (
              <>
                <Row label="Kaggle accuracy">{pct(run.vsKaggle.kaggleAccuracy)}</Row>
                <Row label="Same grade as Kaggle">
                  {count(run.vsKaggle.sameGrade)} of {count(run.vsKaggle.n)}
                </Row>
                <Row label="McNemar p">{fixed(run.vsKaggle.p)}</Row>
              </>
            )}
          </dl>
        </div>
      )}
      <div>
        <h3 className="mb-1 text-sm font-medium">Summary file</h3>
        <pre className="max-h-96 overflow-auto rounded-md bg-muted p-3 text-xs leading-relaxed">
          {JSON.stringify(run.summary ?? null, null, 2)}
        </pre>
      </div>
    </div>
  )
}

export function RunSheet({ run, onClose }: { run: Run | null; onClose: () => void }) {
  return (
    <Sheet open={!!run} onOpenChange={(open) => !open && onClose()}>
      <SheetContent className="w-full overflow-y-auto sm:max-w-lg">
        {run && (
          <>
            <SheetHeader>
              <SheetTitle className="flex flex-wrap items-center gap-2">
                {CONDITION_LABEL[run.condition]} · {MODEL_LABEL[run.model]}
                <Badge variant="outline">{run.split}</Badge>
              </SheetTitle>
              <SheetDescription className="break-all">{run.source}</SheetDescription>
            </SheetHeader>
            <Details id={run.id} />
          </>
        )}
      </SheetContent>
    </Sheet>
  )
}
