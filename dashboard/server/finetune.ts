// Fine-tuning: the training summaries in results/finetune/ and the newest stage 3b comparison
// in results/kaggle/. The tuned grading runs themselves are ordinary Kaggle runs (runs.ts).
import fs from 'node:fs'
import path from 'node:path'
import type { Adapter, Finetune, Model } from './types.ts'

type Obj = Record<string, any>

function list(dir: string): string[] {
  try {
    return fs.readdirSync(dir).sort()
  } catch {
    return []
  }
}

export function loadFinetune(repo: string): Finetune {
  const out: Finetune = { adapters: [], errors: [] }
  for (const f of list(path.join(repo, 'results/finetune')).filter((f) => /^finetune-.*\.json$/.test(f))) {
    const rel = `results/finetune/${f}`
    try {
      const d: Obj = JSON.parse(fs.readFileSync(path.join(repo, rel), 'utf8'))
      const s: Obj = d.settings ?? {}
      const a: Adapter = {
        file: rel,
        model: d.model as Model,
        reply: typeof s.reply === 'string' ? s.reply : 'label',
        trainSets: Array.isArray(s.train_sets) ? s.train_sets : ['scientsbank'],
        epochs: (d.history ?? []).map((h: Obj) => ({ epoch: h.epoch, trainLoss: h.train_loss, valAccuracy: h.val_accuracy })),
      }
      if (typeof d.created === 'string') a.created = d.created
      if (typeof d.chosen_epoch === 'number') a.chosenEpoch = d.chosen_epoch
      for (const [k, v] of [['rank', s.rank], ['alpha', s.alpha], ['lr', s.lr]] as const) if (typeof v === 'number') a[k] = v
      if (Array.isArray(d.train_questions)) a.trainQuestions = d.train_questions.length
      if (Array.isArray(d.val_questions)) a.valQuestions = d.val_questions.length
      out.adapters.push(a)
    } catch (e) {
      out.errors.push(`${rel}: ${(e as Error).message}`)
    }
  }

  const kaggle = list(path.join(repo, 'results/kaggle'))
  const bar = kaggle.filter((f) => /^compare-bar-.*\.json$/.test(f)).at(-1)
  if (bar) {
    const rel = `results/kaggle/${bar}`
    try {
      const d: Obj = JSON.parse(fs.readFileSync(path.join(repo, rel), 'utf8'))
      out.bar = { file: rel, rule: d.rule ?? '', tests: d.tests ?? [] }
    } catch (e) {
      out.errors.push(`${rel}: ${(e as Error).message}`)
    }
  }

  const cmp = kaggle.filter((f) => /^compare-reason-ft-.*\.json$/.test(f)).at(-1)
  if (cmp) {
    const rel = `results/kaggle/${cmp}`
    try {
      const d: Obj = JSON.parse(fs.readFileSync(path.join(repo, rel), 'utf8'))
      out.compare = { file: rel, passed: !!d.gate?.passed, rule: d.gate?.rule ?? '', wins: d.gate?.wins ?? [], tests: d.tests ?? [] }
    } catch (e) {
      out.errors.push(`${rel}: ${(e as Error).message}`)
    }
  }
  return out
}
