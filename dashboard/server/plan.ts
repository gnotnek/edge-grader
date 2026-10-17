// docs/plan.md as data: the timeline table, the checklists under "## Steps", the change log.
import type { Plan, PlanItem } from './types.ts'

const YEAR = 2026
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

function cells(line: string): string[] {
  return line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((c) => c.trim())
}

function isTableBody(line: string, header: string): boolean {
  if (!line.trim().startsWith('|')) return false
  const first = cells(line)[0]
  return first !== header && !/^:?-+:?$/.test(first)
}

function iso(month: number, day: number): string {
  return `${YEAR}-${String(month + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`
}

/** "Sep 28 – Oct 4" or "Nov 16 – 29" → ISO start and end. */
function weekBounds(dates: string): { start?: string; end?: string } {
  const m = dates.match(/^([A-Z][a-z]{2}) (\d+) – (?:([A-Z][a-z]{2}) )?(\d+)$/)
  if (!m) return {}
  const m1 = MONTHS.indexOf(m[1])
  const m2 = m[3] ? MONTHS.indexOf(m[3]) : m1
  return { start: iso(m1, Number(m[2])), end: iso(m2, Number(m[4])) }
}

const unbold = (s: string) => s.replace(/\*\*/g, '').trim()

export function parsePlan(md: string): Plan {
  const plan: Plan = { timeline: [], sections: [], changelog: [] }
  plan.updated = md.match(/Last updated (\d{4}-\d{2}-\d{2})/)?.[1]

  let h2 = ''
  let group: string | undefined
  for (const line of md.split('\n')) {
    if (line.startsWith('## ')) {
      h2 = line.slice(3).trim()
      continue
    }
    if (h2 === 'Timeline' && isTableBody(line, 'Week')) {
      const [week, dates, stage, goal, checkpoint] = cells(line).map(unbold)
      plan.timeline.push({ week, dates, stage, goal, checkpoint, ...weekBounds(dates) })
    } else if (h2 === 'Steps') {
      if (line.startsWith('### ')) {
        plan.sections.push({ title: line.slice(4).trim(), items: [] })
        group = undefined
        continue
      }
      const section = plan.sections.at(-1)
      if (!section) continue
      const bold = line.match(/^\*\*([^*]+)\*\*/)
      if (bold) group = bold[1].trim()
      const item = line.match(/^- \[([ xX])\] (.*)$/)
      if (item) {
        const it: PlanItem = { done: item[1] !== ' ', text: item[2].trim() }
        if (group) it.group = group
        section.items.push(it)
      }
    } else if (h2 === 'Change log' && isTableBody(line, 'Date')) {
      const [date, change, why] = cells(line)
      plan.changelog.push({ date, change, why })
    }
  }
  return plan
}
