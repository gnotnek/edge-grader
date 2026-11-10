// The results write-ups (docs/results/*.md) and the figures they embed. Read-only, and nothing
// outside docs/results is ever served.
import fs from 'node:fs'
import path from 'node:path'

const DIR = 'docs/results'
const FIRST = ['summary.md', 'recap.md']
const TYPES: Record<string, string> = { '.png': 'image/png', '.svg': 'image/svg+xml', '.jpg': 'image/jpeg' }

const notFound = (what: string) => Object.assign(new Error(`no ${what}`), { status: 404 })

export function listDocs(repo: string): { name: string; title: string }[] {
  const names = fs.readdirSync(path.join(repo, DIR)).filter((f) => f.endsWith('.md'))
  const rank = (n: string) => (FIRST.includes(n) ? FIRST.indexOf(n) : FIRST.length)
  return names
    .sort((a, b) => rank(a) - rank(b) || a.localeCompare(b))
    .map((name) => {
      const title = fs.readFileSync(path.join(repo, DIR, name), 'utf8').match(/^# (.+)$/m)?.[1] ?? name
      return { name, title }
    })
}

export function readDoc(repo: string, name: string): string {
  if (!/^[\w-]+\.md$/.test(name)) throw notFound(`page ${name}`)
  const file = path.join(repo, DIR, name)
  if (!fs.existsSync(file)) throw notFound(`page ${name}`)
  return fs.readFileSync(file, 'utf8')
}

/** A figure a page embeds, by its path relative to docs/results. */
export function docFile(repo: string, rel: string): { type: string; body: Buffer } {
  const root = path.resolve(repo, DIR)
  const file = path.resolve(root, rel)
  const type = TYPES[path.extname(file).toLowerCase()]
  if (!file.startsWith(root + path.sep) || !type || !fs.existsSync(file)) throw notFound(`file ${rel}`)
  return { type, body: fs.readFileSync(file) }
}
