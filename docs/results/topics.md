# Stage 2 topic benchmarks: math against news

*Started 2026-10-07. Why these two topics, and why beside SciEntsBank rather than instead of it: [`research-outline.md`](../research-outline.md) (supervisor feedback and own decisions, 2026-10-07). Steps: [`plan.md`](../plan.md#topic-narrowing-math-against-news-week-5).*

**Status (2026-10-08):** math done: the runner matches the lab, and GSM8K vs GSM-Symbolic shows no significant memorisation gap ([results](#math-results-kaggle-t4-2026-10-08)). News done and judged: near 0% without RAG, 39–45% with it ([results](#news-results-kaggle-t4-judged-by-qwen3-14b-2026-10-08)). Stage 2's topic benchmarks are complete.

The question: which kind of topic suits these small models, and does RAG help there? Math and post-cutoff news are the two ends of that range. Math needs reasoning and no outside facts, and it's already the models' strongest MMLU-Pro subject. News after the training cutoff can't be known without help, so it's the cleanest test of RAG and of the NotebookLM comparison (RQ5).

## Gemma 4's training cutoff: January 2025

The Gemma 4 model card ([ai.google.dev](https://ai.google.dev/gemma/docs/core/model_card_4), and the E2B and E4B Hugging Face cards) says the pre-training data has "a cutoff date of January 2025". One sentence covers the whole family, E2B and E4B included. The technical report is arXiv 2607.02770 (July 2026). Every news question below is about November–December 2025, ten months or more after the cutoff.

## Math: GSM8K and GSM-Symbolic

**Questions** (`uv run edge-grader-gsm`, [`src/edge_grader/gsm.py`](../../src/edge_grader/gsm.py)):

| Set | What | n |
|---|---|---|
| `tiny` | The lab's tinyGSM8k run, replayed: the exact chat messages `agentic-edge`'s lm-eval sent, copied to [`tinygsm8k-lab.json`](../../results/reference/agentic-edge/lm-eval/tinygsm8k-lab.json). Runner check only | 100 |
| `gsm8k` | The 100 GSM8K test questions that GSM-Symbolic (Apple, 2024) turned into templates | 100 |
| `symbolic` | GSM-Symbolic `main`: the first 10 of the 50 variants per template (same reasoning, new names and numbers) | 1,000 |

**Prompt:** the lab's lm-eval format, five worked examples as chat turns (`Question: …\nAnswer:` and the GSM8K solution), then the question. Greedy, up to **1,024 tokens** (the lab's corrected cap; lm-eval's default 256 cut 24% of E2B's answers), stop at `Question:`. For `gsm8k` and `symbolic` the five examples are fixed for every question: GSM8K train picked with seed 1234. They are the same five lm-eval gave the first tinyGSM8k question. So an original and its variants differ only in the question.

**Scoring:** lm-eval's two filters and exact match, reimplemented. **Flexible** (the last number in the reply) is the headline, as in the lab. **Strict** (`#### N`) is recorded too. The reimplementation gives the same extracted string and the same right/wrong as lm-eval on all 300 recorded lab replies (`tests/test_gsm.py`).

**Runner check:** `tiny` with E2B at 1,024 tokens against the lab's **81/100**. The lab ran E4B at 256 tokens only, so the check for E4B is at 256 against **75/100**. Paired McNemar per question, as for MMLU-Pro.

**Analysis:**
- Accuracy on `gsm8k` against the mean over the 10 `symbolic` instance sets, with the spread (min–max of the 10 set accuracies, as the GSM-Symbolic paper reports it).
- Paired by template: the gap is the original's right/wrong minus the share of its 10 variants right, averaged over the 100 templates. 95% bootstrap interval over templates (10,000 resamples).
- **Test: a permutation test, not the sign test first planned** (changed 2026-10-08, during the analysis, after seeing the per-template counts; the sign test's result is reported below too, so the change is visible). The sign test counts templates where the original beats its variants' average, but one original against ten variants isn't symmetric: a correct original "wins" whenever any one of its ten variants slips, so the test favours the original even when the original is just another instance. The permutation test asks exactly the question: under the null, the original is exchangeable with its variants, so each template's original is swapped with a random one of its 11 items (20,000 times) and the gap recomputed. One-sided p: how often a random pick beats the rest by at least the observed gap.
- A clear drop from `gsm8k` to `symbolic` would point to memorised GSM8K answers, or to the new numbers being harder. No drop means the math skill is the model's own.

## News: LiveNewsBench, January 2026 release

**Why LiveNewsBench and not a self-built set** (own decision, 2026-10-07): it already is what the plan described. The [January 2026 release](https://github.com/YunfanZhang42/LiveNewsBench) has 200 **human-verified** test questions about events from **2025-11-01 to 2025-12-31**, and each question ships with the news articles it was written from (1,252 article texts, 1,178 distinct). It's published (arXiv 2602.13543, MIT licence), so the numbers can be compared with others', and it saves writing and checking 200 questions by hand. Its paper reports small-model reference points on the same 200 questions: Qwen3 8B at 3.5% without search and 49.5% with web search.

**What the questions are like:** harder than the supervisor's Pluto example. They are multi-hop, written by GPT-5.1 from Wikipedia's Current Events portal and its cited articles, then accepted or rejected by a person (about 15% rejected). Example: "how many days after the eligibility window closed were the 83rd Golden Globe nominations made public?" Answers are short free text. Categories: armed conflicts 49, politics and elections 40, law and crime 39, international relations 21, disasters 18, business 14, arts 7, sports 7, health 4, science 1.

**Data handling:** the file is downloaded from LiveNewsBench's GitHub at commit `8a6b96e` and checked by sha256 (`edge_grader.news.fetch`). It stays in `data/` (git-ignored), because the articles are third-party news text. Only ids are committed.

**Prompt:** LiveNewsBench's own no-search baseline, byte for byte. That is a system line plus "Today is 2025-12-31. Please answer … step by step … `<answer>…</answer>`", then the question. Greedy, up to 2,048 tokens (as MMLU-Pro; the paper allows 16k to frontier models). The final answer is the text in the last `<answer>` tags. With RAG, the top 3 passages go in front, headed "News articles, for background:".

**Scoring: an LLM judge, as LiveNewsBench does.** The judge gets the question, the gold answer and the model's final answer, and returns CORRECT / INCORRECT / NOT_ATTEMPTED (the SimpleQA grading prompt, copied to [`prompts/simpleqa_grader.txt`](../../src/edge_grader/prompts/simpleqa_grader.txt)). `uv run edge-grader-news-judge <run>.json` writes `<run>.judged-<judge>.json/.jsonl` next to the run and never changes the run's own files. The paper's judge is GPT-4.1, the CLI's default.

**Judge used here: Qwen3-14B (Q4_K_M, Qwen's official GGUF, Apache 2.0) on Kaggle, not GPT-4.1** (own decision, 2026-10-08). GPT-4.1 needs paid API credit; the account's free tier allows 50 requests a day, and the four runs need 800. Qwen3-14B runs free on a Kaggle T4 in `llama-server` with the same flags (thinking off) and the same prompt, temperature 0. It isn't a Gemma model, so the judge doesn't grade its own family. The judge reads the leading letter, so "A: CORRECT" counts like "A"; anything else is `unparsed`. The judged file records the judge server's model path and build. **The cost:** the numbers aren't directly comparable with the paper's GPT-4.1 figures. **The judge is unchecked** (owner's choice, 2026-10-08): the planned checks, GPT-4.1 and the owner's hand grades on a 50-answer sample, were both dropped. The only check is Claude's spot-check of 12 grades (below), all of which read as right calls. Report the news accuracies as "judged by Qwen3-14B", with that caveat.

Before judging, 329 of the 800 answers were plain declines ("Information unavailable", or no `<answer>` tag); they're judged like the rest, not filtered by a rule.

### RAG corpus and retrieval

The RAG corpus is every distinct source article of the 200 questions in one pool, the way a teacher would load all their sources at once. The articles are chunked like the course material (same filters, 250-word windows, 50 overlap): **4,861 chunks from 1,176 articles** (2 articles were too short to keep). The chunks file's sha256 is recorded in [`chunks-news.json`](../../results/rag/chunks-news.json). Embedding is `bge-small-en-v1.5` with the question text as the query, the frozen RAG-en settings. A few articles aren't in English (e.g. Spanish sources on Chile's election); they're kept as the benchmark gives them. Retrieval file: [`retrieval-news-en.bge-small-en-k5.json`](../../results/rag/retrieval-news-en.bge-small-en-k5.json).

**Retrieval check, automatic (all 200 questions):** each question's own source articles are known, so it's possible to count how often the search finds one of them:

| A passage from one of the question's own articles in the | top 1 | top 3 (what the prompt gets) | top 5 |
|---|---|---|---|
| Questions | 181 / 200 | **194 / 200** | 199 / 200 |

507 of the 600 top-3 passages come from the question's own articles. So the search finds the right articles. That doesn't mean the passage holds the answer, which the hand check below measures. Because the pooled search already finds the question's own articles this often, a separate "own articles only" condition would differ little from RAG-en, and it was dropped.

**Retrieval check, by hand (owner's marking, 2026-10-07):** 30 questions over the categories, top 3 passages each shown in full with the gold answer. A passage was ticked if it holds information that helps answer the question. News has no unscored split, so the sheet samples test questions. Settings are frozen, so the check can't tune anything on them. It informs the interpretation and doesn't gate the runs. File: [`check-news-en.bge-small-en-k5.json`](../../results/rag/check-news-en.bge-small-en-k5.json).

| | News (this check) | SciEntsBank, English | MMLU-Pro, English |
|---|---|---|---|
| Questions with a useful passage in the top 3 | **21 / 30** | 16 / 30 | 15 / 30 |
| Useful passage at rank 1 | 16 / 30 | | |
| Useful passages | 22 / 90 | | |

The best retrieval of the three benchmarks, as expected when the corpus is the questions' own sources. But useful passages are rare beyond rank 1 (only 6 of the 22 sit at ranks 2–3), and 9 of the 30 questions get nothing useful. The marking asked whether a passage helps, not whether the top 3 hold every fact a multi-hop question needs, so 21/30 is an upper bound on questions RAG can fully answer.

## How to run

On Kaggle, [`notebooks/kaggle_baseline.ipynb`](../../notebooks/kaggle_baseline.ipynb):
- **Math:** `TASK = "gsm"`, `GSM_SETS = ["tiny", "gsm8k", "symbolic"]`, `RAG = None`. Pilot with `N = 5`, then `N = 0`. About 1,200 questions per model.
- **News, no RAG:** `TASK = "news"`, `RAG = None`.
- **News, RAG-en:** `TASK = "news"`, `RAG = "results/rag/retrieval-news-en.bge-small-en-k5.json"`. The notebook rebuilds the news chunks from the pinned download and checks their sha256, so nothing has to be uploaded.

Then the judge, also on Kaggle, in its own notebook: [`notebooks/kaggle_news_judge.ipynb`](../../notebooks/kaggle_news_judge.ipynb). It grades every committed `results/kaggle/news-*.json` run that has no Qwen3-14B grades yet. (With an API key instead: `OPENAI_API_KEY=… uv run edge-grader-news-judge results/kaggle/news-<…>.json`.)

## Results

### Math results (Kaggle T4, 2026-10-08)

All six runs on llama.cpp `a3a1c47`, 0 errors, every reply has a number. Files: `results/kaggle/gsm-*.json`.

**Runner check: passes.** Same accuracy as the lab within noise, per question:

| | Lab (agentic-edge) | This runner | Same right/wrong | Only lab right / only ours | McNemar p | Identical replies |
|---|---|---|---|---|---|---|
| E2B, 1,024 tokens | 81 / 100 | 77 / 100 | 92 / 100 | 6 / 2 | 0.29 | 34 / 100 |
| E4B, 256 tokens | 75 / 100 | 74 / 100 | 97 / 100 | 2 / 1 | 1.00 | 52 / 100 |

The replies differ in wording more often than in result: the lab served the models on its own build and hardware, and llama.cpp isn't bit-identical across builds (the lab saw 1 of 100 GSM8K answers flip between two of its own restarts). The prompts are byte-identical (replayed), so the runner and its scoring are what's checked here.

**GSM8K against GSM-Symbolic:**

| Model | GSM8K originals (100) | GSM-Symbolic variants (1,000) | 10 instance sets | Gap (pts) | 95% bootstrap | Permutation p |
|---|---|---|---|---|---|---|
| E2B | **83.0%** | **80.0%** | 78–83% | +3.0 | −3.1 to +8.8 | 0.20 |
| E4B | **92.0%** | **90.5%** | 88–93% | +1.5 | −2.8 to +5.7 | 0.32 |

Strict extraction (`#### N`) gives lower numbers but the same pattern: E2B 54.0 → 48.4%, E4B 85.0 → 80.6%. E2B often ends with "Answer: $70.00" instead of "#### 70", as the lab saw.

1. **No significant memorisation gap.** Both models score a little higher on the originals, and the original beats 9 of its 10 instance sets for both, but the gap is within what reshuffling gives (p = 0.20 and 0.32), and the intervals include 0. The upper bounds rule out drops larger than about 9 points (E2B) and 6 points (E4B). (The sign test first planned would have said p ≈ 0.01 for both, from 32 vs 14 and 17 vs 5 templates; but random picks give about 29 and 16 "original wins" by chance, see the analysis note above.)
2. **Math is a strong topic for both models, and it holds on unseen numbers:** E4B solves 90% of new variants, E2B 80%. That agrees with MMLU-Pro, where math was their strongest subject (E2B 79%, E4B 88%).
3. **Running out of tokens is rare and not variant-specific:** E2B hits the 1,024-token cap on 4% of originals and 3.6% of variants (nearly always wrong), E4B on 1% and 1.5%.
4. **For the math-vs-news question:** math is skill the models already have and apply to new numbers, without retrieval.

### News results (Kaggle T4, judged by Qwen3-14B, 2026-10-08)

Four runs on llama.cpp `a3a1c47`, 200 questions each, 0 errors; judged with LiveNewsBench's SimpleQA prompt by Qwen3-14B Q4_K_M on the same build, **0 unparsed** of 800. Files: `results/kaggle/news-*.json` and `news-*.judged-qwen3-14b-q4_k_m.json`.

| | Correct | Incorrect | Not attempted | Accuracy | Right when it answers |
|---|---|---|---|---|---|
| E2B, no RAG | 1 | 44 | 155 | **0.5%** | 2% |
| E4B, no RAG | 1 | 42 | 157 | **0.5%** | 2% |
| E2B, RAG-en | 78 | 69 | 53 | **39.0%** | 53% |
| E4B, RAG-en | 90 | 40 | 70 | **45.0%** | 69% |

Paired per question (exact McNemar): RAG against no RAG, E2B 78 questions gained and 1 lost (p < 10⁻²¹), E4B 89 and 0 (p < 10⁻²⁶). With RAG, E4B against E2B: 15 questions only E4B got right, 3 only E2B (p = 0.008).

1. **Without the articles, the models can't answer and mostly say so.** 0.5% for both, the one "correct" each being a lucky or lenient call (E2B's "CPTPP; strategic materials" for "CPTPP and rare earths" is one the judge let through). Over three-quarters of the answers are declines ("Information unavailable"), not guesses: the same behaviour the lab saw, Gemma 4 refusing rather than inventing. For reference, LiveNewsBench reports Qwen3 8B at 3.5% without search on these questions.
2. **With the articles, RAG works here, unlike on SciEntsBank and MMLU-Pro.** 0.5% → 39% (E2B) and 45% (E4B), on multi-hop questions. For scale, the paper's Qwen3 8B with live web search (up to 5 searches and 5 page visits) scored 49.5%. Here the search is one fixed top-3 retrieval over the right articles. Different judges (GPT-4.1 there), so this is context, not a comparison.
3. **E4B is the more careful reader.** It declines more (70 vs 53) but is right 69% of the times it answers, against 53% for E2B, which gives more wrong answers (69 vs 40).
4. **Retrieval sets the ceiling.** With no passage from the question's own articles in the top 3, 0–1 of 6 right; with three, 51 (E2B) and 61 (E4B) of 135. In a sample of 6 misses (E4B), most were near-misses (658 for 656 days; "Caritas Internationalis" for Caritas Jerusalem) or a two-part answer with one part not in the passages.
5. **Spot-check (Claude, 12 random E4B RAG grades, not the owner's):** all 12 read as right calls, e.g. "Over 1,750,000 hectares" accepted for "Roughly 1.75 million hectares", "658" rejected for "656 days". No person checked the judge's grades (owner's choice); this spot-check is all there is, and it's small and not independent.

### Math against news: what suits these models

| | E2B | E4B | Needs |
|---|---|---|---|
| Math, GSM8K (no RAG) | 83.0% | 92.0% | skill the model has |
| Math, unseen numbers (GSM-Symbolic) | 80.0% | 90.5% | the same skill, not memory |
| News after the cutoff, no RAG | 0.5% | 0.5% | facts the model can't have |
| News after the cutoff, RAG-en | 39.0% | 45.0% | reading the facts it's given |

The two ends of the range behave as expected, and the contrast is the finding: these small models are good at what they can do on their own (math), and RAG is what makes knowledge they lack usable at all. Where RAG didn't help earlier (SciEntsBank grading, MMLU-Pro), two things differed: those tasks lean less on facts the model can't have, and retrieval was weaker (a useful passage for 16/30 and 15/30 checked questions, against 21/30 here). Which of the two matters more isn't separated by these runs. What they do show: RAG pays off when knowledge is the missing piece and retrieval finds it.
