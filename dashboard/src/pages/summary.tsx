// The results write-ups in docs/results/, rendered: summary.md first, then the recap and the topic pages.
import { useEffect, useState } from 'react'
import Markdown, { type Components } from 'react-markdown'
import rehypeSlug from 'rehype-slug'
import remarkGfm from 'remark-gfm'
import { ErrorNote, Loading, Pick } from '@/components/bits'
import { Card, CardContent } from '@/components/ui/card'
import { useApi } from '@/lib/api'

type Doc = { name: string; title: string }

function components(open: (name: string, anchor?: string) => void): Components {
  return {
    h1: ({ children }) => <h1 className="font-serif text-3xl font-medium text-balance">{children}</h1>,
    h2: ({ children, id }) => <h2 id={id} className="mt-8 scroll-mt-20 border-t pt-5 font-serif text-xl font-medium">{children}</h2>,
    h3: ({ children, id }) => <h3 id={id} className="mt-5 scroll-mt-20 font-semibold text-[var(--heading)]">{children}</h3>,
    h4: ({ children, id }) => <h4 id={id} className="mt-4 scroll-mt-20 font-semibold text-[var(--heading)]">{children}</h4>,
    p: ({ children }) => <p className="mt-2 max-w-4xl">{children}</p>,
    ul: ({ children }) => <ul className="mt-1.5 max-w-4xl list-disc space-y-0.5 pl-5">{children}</ul>,
    ol: ({ children }) => <ol className="mt-1.5 max-w-4xl list-decimal space-y-0.5 pl-5">{children}</ol>,
    strong: ({ children }) => <strong className="font-semibold text-[var(--heading)]">{children}</strong>,
    blockquote: ({ children }) => <blockquote className="mt-2 border-l-2 pl-3 text-muted-foreground">{children}</blockquote>,
    hr: () => <hr className="my-6" />,
    code: ({ children }) => <code className="rounded bg-muted px-1 py-0.5 font-mono text-[0.85em]">{children}</code>,
    pre: ({ children }) => <pre className="mt-2 overflow-x-auto rounded-md border bg-muted p-3 text-xs">{children}</pre>,
    table: ({ children }) => (
      <div className="mt-2 overflow-x-auto rounded-md border">
        <table className="tabular w-full text-xs">{children}</table>
      </div>
    ),
    th: ({ children, style }) => <th style={style} className="border-b bg-muted px-2 py-1.5 text-left font-medium whitespace-nowrap">{children}</th>,
    td: ({ children, style }) => <td style={style} className="border-b px-2 py-1 align-top">{children}</td>,
    img: ({ src, alt }) => (
      <img
        src={typeof src === 'string' && !/^https?:/.test(src) ? `/api/doc-file/${src}` : src}
        alt={alt ?? ''}
        className="mt-2 w-full max-w-2xl rounded-md border bg-white"
      />
    ),
    a: ({ href = '', children }) => {
      const page = href.match(/^([\w-]+\.md)(?:#(.+))?$/)
      if (page)
        return (
          <a href={`#/summary`} onClick={(e) => { e.preventDefault(); open(page[1], page[2]) }} className="underline underline-offset-2">
            {children}
          </a>
        )
      if (href.startsWith('#')) return <a href={href} onClick={(e) => { e.preventDefault(); open('', href.slice(1)) }} className="underline underline-offset-2">{children}</a>
      if (/^https?:/.test(href)) return <a href={href} target="_blank" rel="noreferrer" className="underline underline-offset-2">{children}</a>
      // A file elsewhere in the repo: the dashboard doesn't serve it, so name it instead of linking.
      return <span title={href} className="underline decoration-dotted underline-offset-2">{children}</span>
    },
  }
}

export function SummaryPage() {
  const docs = useApi<Doc[]>('/docs')
  const [name, setName] = useState('')
  const [anchor, setAnchor] = useState<string>()
  const current = name || docs.data?.[0]?.name || ''
  const doc = useApi<{ name: string; markdown: string }>(current ? `/docs/${current}` : '/docs')

  // After a page loads, jump to the section a link asked for.
  useEffect(() => {
    if (!doc.data || !('markdown' in doc.data)) return
    if (anchor) document.getElementById(anchor)?.scrollIntoView({ block: 'start' })
    else window.scrollTo({ top: 0 })
  }, [doc.data, anchor])

  const open = (next: string, at?: string) => {
    if (next) setName(next)
    setAnchor(at)
    if (!next && at) document.getElementById(at)?.scrollIntoView({ block: 'start' })
  }

  const error = docs.error ?? doc.error
  if (error) return <ErrorNote message={error} />
  if (!docs.data || !doc.data || !('markdown' in doc.data)) return <Loading rows={4} />

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <Pick
          label="Page"
          value={current}
          onChange={(v) => open(v)}
          options={docs.data.map((d): [string, string] => [d.name, d.title])}
        />
        <span className="font-mono text-xs text-muted-foreground">docs/results/{current}</span>
      </div>
      <Card>
        <CardContent className="text-sm leading-relaxed">
          <Markdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeSlug]} components={components(open)}>
            {doc.data.markdown}
          </Markdown>
        </CardContent>
      </Card>
    </div>
  )
}
