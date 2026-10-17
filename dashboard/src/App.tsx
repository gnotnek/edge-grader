import type { Overview } from '@server/types'
import { ChartScatter, Cpu, LayoutDashboard, ListChecks, RefreshCw, Rows3 } from 'lucide-react'
import { useEffect, useState } from 'react'
import { ThemeToggle } from '@/components/theme'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import {
  Sidebar, SidebarContent, SidebarFooter, SidebarGroup, SidebarGroupContent, SidebarHeader, SidebarInset,
  SidebarMenu, SidebarMenuBadge, SidebarMenuButton, SidebarMenuItem, SidebarProvider, SidebarRail, SidebarTrigger,
} from '@/components/ui/sidebar'
import { ReloadContext, useApi } from '@/lib/api'
import { BoardsPage } from '@/pages/boards'
import { OverviewPage } from '@/pages/overview'
import { PlanPage } from '@/pages/plan'
import { ResultsPage } from '@/pages/results'
import { RunsPage } from '@/pages/runs'

const PAGES = {
  overview: { title: 'Overview', icon: LayoutDashboard, Page: OverviewPage },
  plan: { title: 'Plan', icon: ListChecks, Page: PlanPage },
  runs: { title: 'Runs', icon: Rows3, Page: RunsPage },
  results: { title: 'Results', icon: ChartScatter, Page: ResultsPage },
  boards: { title: 'Boards', icon: Cpu, Page: BoardsPage },
} as const
type PageKey = keyof typeof PAGES

function pageFromHash(): PageKey {
  const h = window.location.hash.replace(/^#\/?/, '')
  return h in PAGES ? (h as PageKey) : 'overview'
}

function OwedBadge() {
  const { data } = useApi<Overview>('/overview')
  const owed = data ? Object.values(data.owedByBoard).reduce((a, b) => a + b, 0) : 0
  return owed ? <SidebarMenuBadge aria-label={`${owed} runs owed`}>{owed}</SidebarMenuBadge> : null
}

export default function App() {
  const [page, setPage] = useState<PageKey>(pageFromHash)
  const [reload, setReload] = useState(0)
  useEffect(() => {
    const on = () => setPage(pageFromHash())
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])
  const { title, Page } = PAGES[page]

  return (
    <ReloadContext.Provider value={reload}>
      <SidebarProvider>
        <Sidebar collapsible="icon" variant="inset">
          <SidebarHeader>
            <div className="flex items-center gap-2 px-1 py-1.5">
              <div className="grid size-8 shrink-0 place-items-center rounded-md bg-primary text-primary-foreground">
                <Cpu className="size-4" aria-hidden />
              </div>
              <div className="grid leading-tight group-data-[collapsible=icon]:hidden">
                <span className="font-semibold">edge-grader</span>
                <span className="text-xs text-muted-foreground">Small graders on small boards</span>
              </div>
            </div>
          </SidebarHeader>
          <SidebarContent>
            <SidebarGroup>
              <SidebarGroupContent>
                <SidebarMenu>
                  {(Object.keys(PAGES) as PageKey[]).map((key) => {
                    const { title, icon: Icon } = PAGES[key]
                    return (
                      <SidebarMenuItem key={key}>
                        <SidebarMenuButton asChild isActive={page === key} tooltip={title}>
                          <a href={`#/${key}`}>
                            <Icon />
                            <span>{title}</span>
                          </a>
                        </SidebarMenuButton>
                        {key === 'boards' && <OwedBadge />}
                      </SidebarMenuItem>
                    )
                  })}
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>
          </SidebarContent>
          <SidebarFooter className="group-data-[collapsible=icon]:hidden">
            <p className="px-2 text-xs leading-relaxed text-muted-foreground">
              Read from <code>results/</code>, <code>backups/</code> and <code>docs/plan.md</code>. Run{' '}
              <code>backup_board.sh</code> or <code>fetch_results.sh</code>, then reload.
            </p>
          </SidebarFooter>
          <SidebarRail />
        </Sidebar>
        <SidebarInset className="min-w-0">
          <header className="sticky top-0 z-10 flex h-14 items-center gap-2 border-b bg-background/90 px-4 backdrop-blur">
            <SidebarTrigger className="-ml-1" />
            <Separator orientation="vertical" className="mr-1 data-[orientation=vertical]:h-4" />
            <h1 className="font-semibold">{title}</h1>
            <div className="ml-auto flex items-center gap-1">
              <Button variant="ghost" size="sm" onClick={() => setReload((n) => n + 1)}>
                <RefreshCw /> Reload data
              </Button>
              <ThemeToggle />
            </div>
          </header>
          <main className="mx-auto w-full max-w-6xl p-4 md:p-6">
            <Page />
          </main>
        </SidebarInset>
      </SidebarProvider>
    </ReloadContext.Provider>
  )
}
