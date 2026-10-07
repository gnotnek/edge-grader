# Stage 2 topic benchmarks: math against news

*Started 2026-10-07. Why these two topics, and why beside SciEntsBank rather than instead of it: [`research-outline.md`](../research-outline.md) (supervisor feedback and own decisions, 2026-10-07). Steps: [`plan.md`](../plan.md#topic-narrowing-math-against-news-week-5).*

**Status (2026-10-07):** runners, question sets, news corpus and retrieval are built and tested; news retrieval checked by hand (21/30). The Kaggle runs and the news judging haven't been done yet, so there are no accuracy numbers on this page yet.

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

**Runner check (planned):** `tiny` with E2B at 1,024 tokens against the lab's **81/100**. The lab ran E4B at 256 tokens only, so the check for E4B is at 256 against **75/100**. Paired McNemar per question, as for MMLU-Pro.

**Analysis (planned):**
- Accuracy on `gsm8k` against the mean over the 10 `symbolic` instance sets, with the spread (min–max of the 10 set accuracies, as the GSM-Symbolic paper reports it).
- Paired by template: original right or wrong against the share of its 10 variants right. Bootstrap interval over templates, sign test.
- A clear drop from `gsm8k` to `symbolic` would point to memorised GSM8K answers, or to the new numbers being harder. No drop means the math skill is the model's own.

## News: LiveNewsBench, January 2026 release

**Why LiveNewsBench and not a self-built set** (own decision, 2026-10-07): it already is what the plan described. The [January 2026 release](https://github.com/YunfanZhang42/LiveNewsBench) has 200 **human-verified** test questions about events from **2025-11-01 to 2025-12-31**, and each question ships with the news articles it was written from (1,252 article texts, 1,178 distinct). It's published (arXiv 2602.13543, MIT licence), so the numbers can be compared with others', and it saves writing and checking 200 questions by hand. Its paper reports small-model reference points on the same 200 questions: Qwen3 8B at 3.5% without search and 49.5% with web search.

**What the questions are like:** harder than the supervisor's Pluto example. They are multi-hop, written by GPT-5.1 from Wikipedia's Current Events portal and its cited articles, then accepted or rejected by a person (about 15% rejected). Example: "how many days after the eligibility window closed were the 83rd Golden Globe nominations made public?" Answers are short free text. Categories: armed conflicts 49, politics and elections 40, law and crime 39, international relations 21, disasters 18, business 14, arts 7, sports 7, health 4, science 1.

**Data handling:** the file is downloaded from LiveNewsBench's GitHub at commit `8a6b96e` and checked by sha256 (`edge_grader.news.fetch`). It stays in `data/` (git-ignored), because the articles are third-party news text. Only ids are committed.

**Prompt:** LiveNewsBench's own no-search baseline, byte for byte. That is a system line plus "Today is 2025-12-31. Please answer … step by step … `<answer>…</answer>`", then the question. Greedy, up to 2,048 tokens (as MMLU-Pro; the paper allows 16k to frontier models). The final answer is the text in the last `<answer>` tags. With RAG, the top 3 passages go in front, headed "News articles, for background:".

**Scoring: an LLM judge, as LiveNewsBench does.** The judge gets the question, the gold answer and the model's final answer, and returns CORRECT / INCORRECT / NOT_ATTEMPTED (the SimpleQA grading prompt, copied to [`prompts/simpleqa_grader.txt`](../../src/edge_grader/prompts/simpleqa_grader.txt)). `uv run edge-grader-news-judge <run>.json` writes `<run>.judged-<judge>.json/.jsonl` next to the run and never changes the run's own files. The paper's judge is GPT-4.1, the default here. **Pending the owner:** which judge (and API key) to use.

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

Then, on this machine: `OPENAI_API_KEY=… uv run edge-grader-news-judge results/kaggle/news-<…>.json` for each news run.

## Results

*Pending the Kaggle runs.*
