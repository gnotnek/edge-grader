// Shapes the /api routes return. Shared by the server parsers and the React pages.

export type Board = 'pi5' | 'jetson' | 'kaggle'
export type Model = 'e2b' | 'e4b'
export type Benchmark = 'scientsbank' | 'beetle' | 'saf' | 'gsm' | 'news' | 'mmlupro'
export type Condition = 'baseline' | 'rag-en' | 'rag-id' | 'think' | 'lg' | 'lg-reuse' | 'swafull' | 'ft' | 'ft-rf' | 'ft-lf'

export interface RunKey {
  condition: Condition
  benchmark: Benchmark
  model: Model
  board: Board
  split: string
  stamp: string
}

export interface VsKaggle {
  kaggleRun: string
  kaggleAccuracy: number
  boardAccuracy: number
  n: number
  sameGrade: number
  p: number
}

export interface Run extends RunKey {
  id: string
  time?: string
  // results/ = fetched and committed; backup = only in backups/<board>/, not fetched yet
  location: 'results' | 'backup'
  source: string
  metrics: Record<string, number | null>
  summary?: unknown
  error?: string
  energy?: { jPerAnswer: number; meanW: number }
  vsKaggle?: VsKaggle
}

export interface PlanItem {
  done: boolean
  text: string
  group?: string
}

export interface Plan {
  updated?: string
  timeline: {
    week: string
    dates: string
    stage: string
    goal: string
    checkpoint: string
    start?: string
    end?: string
  }[]
  sections: { title: string; items: PlanItem[] }[]
  changelog: { date: string; change: string; why: string }[]
}

export interface Job {
  id: number
  state: string
  args: string[]
  env: Record<string, string>
  added?: string
  started?: string | null
  ended?: string | null
  rc?: number | null
  run_dir?: string | null
  busy_waits?: number
  logTail?: string
}

export interface BoardState {
  board: 'pi5' | 'jetson'
  hasBackup: boolean
  lastBackup?: string
  jobs: Job[]
  error?: string
}

export interface Expected {
  board: Board
  model: Model
  benchmark: Benchmark
  split: string
  condition: Condition
  plan: string
}

export type Status = 'done' | 'on-board' | 'queued' | 'running' | 'missing'

export interface ExpectedStatus extends Expected {
  status: Status
  run?: Run
}

export interface BoardPair {
  first: string
  second: string
  latency_ratio?: number
  energy_ratio?: number
  n: number
  same_grade: number
  first_accuracy: number
  second_accuracy: number
  mcnemar: { only_first_right: number; only_second_right: number; p: number }
}

export interface Overview {
  today: string
  currentWeek?: Plan['timeline'][number]
  runsByBoard: Record<string, number>
  owedByBoard: Record<string, number>
  lastBackup: Record<string, string | null>
  waiting: ExpectedStatus[]
  newest: Run[]
  planProgress: { done: number; total: number }
  errors: number
}

/** A number copied from agentic-edge (results/reference/): the lab's runs, shown beside ours, never mixed in. */
export interface ReferenceRow {
  section: string
  label: string
  benchmark: 'mmlupro'
  board: 'pi5' | 'jetson'
  model: Model
  condition?: Condition
  accuracy?: number
  baseline?: number
  mcnemarP?: number
  energyWh?: number
  wallMin?: number
  jPerQuestion?: number
  note?: string
}

export interface Reference {
  source: string | null
  copied?: string
  rows: ReferenceRow[]
  error?: string
}

/** One LoRA training run (results/finetune/finetune-*.json). */
export interface Adapter {
  file: string
  model: Model
  reply: string
  created?: string
  chosenEpoch?: number
  epochs: { epoch: number; trainLoss: number; valAccuracy: number }[]
  rank?: number
  alpha?: number
  lr?: number
  trainQuestions?: number
  valQuestions?: number
}

/** A paired test from the stage 3b comparison (results/kaggle/compare-reason-ft-*.json). */
export interface FinetuneTest {
  adapter: string
  baseline: string
  dataset: string
  split: string
  n: number
  first_accuracy: number
  second_accuracy: number
  diff: number
  p: number
  p_holm: number
}

export interface Finetune {
  adapters: Adapter[]
  compare?: {
    file: string
    passed: boolean
    rule: string
    wins: { adapter: string; dataset: string; split: string }[]
    tests: FinetuneTest[]
  }
  errors: string[]
}
