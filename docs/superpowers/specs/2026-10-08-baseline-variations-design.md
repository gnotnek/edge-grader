# Design: edge baseline first, then baseline variations (replaces fine-tuning)

*2026-10-08. From the supervisor's notes of 2026-10-08, worked through with the owner. Once approved, this spec feeds [`research-outline.md`](../../research-outline.md) and [`plan.md`](../../plan.md); those two stay the source of truth for what is decided and when.*

## 1. What the supervisor asked (2026-10-08)

1. **Finding in the baseline.** The baseline (MMLU-Pro, SciEntsBank, run on the Pi, the Jetson and the cloud) is where the findings are. Extend it the way `agentic-edge` did: thinking on/off, and an inference engine other than llama.cpp.
2. **Multimodal, narrowed.** Gemma 4 is multimodal; narrow that to **text + images for grading**.
3. **Baseline against RAG for every benchmark.** With the baselines for MMLU-Pro, SciEntsBank, math and news in hand, show how each compares with its RAG version.
4. **Finish the baseline on the edge first.** The project is about edge computing, so the baseline has to be complete on the boards before going further ("so we aren't steps too far").

Owner's decisions in the same session:
- The new work **replaces fine-tuning**. Stage 3 (QLoRA, RQ3) is dropped.
- Engine: **little-gemma**. Embedder: try **EmbeddingGemma 2**. Look through `agentic-edge` for other edge settings worth reusing.
- Images: both **(1) the student's answer as an image** and **(2) a question with a figure**.
- Math RAG retrieves from a **math textbook**, not from similar solved problems.
- Boards: **the Jetson runs everything; the Pi runs a trimmed set**.

## 2. Goals and non-goals

**Goals:**
- Every baseline cell measured on both boards, accuracy and cost, before any new condition reaches the boards
- A baseline-vs-RAG table for all four benchmarks
- Three inference-time variations on the grading baseline: thinking, engine plus prefix reuse, embedder
- Grading from an image of the answer, paired with the text result

**Non-goals:**
- **Fine-tuning:** dropped.
- **n-gram speculation, MTP:** a grading reply is a single label, so there's almost nothing to speed up.
- **TurboQuant KV:** `agentic-edge` S8 cost E4B 3–5 points.
- **Q5_K_M control:** no question here needs it.
- **little-gemma on the Pi:** about 40× slower prompt processing than llama.cpp.
- **New datasets beyond (2) if none fits.**
- Anything written into `agentic-edge`.

## 3. Research questions after the change

| # | Question | Change |
|---|---|---|
| RQ1 | How well do untouched E2B/E4B grade, and how do they do on the subject-knowledge benchmarks (MMLU-Pro, math, news)? **On the cloud and on both boards** | Now covers every benchmark on the boards (Phase 0) |
| RQ2 | Does RAG improve each benchmark: grading, MMLU-Pro, math, news? | Math added (Phase A) |
| ~~RQ3~~ | ~~Fine-tuning~~ | **Dropped** (supervisor, 2026-10-08). The number is kept so older decisions still point to the right thing |
| RQ4 | What does each version cost on the Pi 5 / Jetson? | Unchanged; now includes the new versions |
| RQ5 | How close is the best local version to a cloud model with the same material? | Unchanged |
| **RQ6** | Do inference-time settings (thinking, engine, prefix reuse, embedder) change grading accuracy or cost? | New; replaces the optional thinking question |
| **RQ7** | Can the models grade an answer given as an image, and how much accuracy does reading cost? | New |

## 4. Phase 0: finish the baseline on the boards (first)

### What exists and what's missing

Baseline = no RAG, thinking off, llama.cpp, the fixed flags.

| Baseline | Kaggle | Pi 5 | Jetson |
|---|---|---|---|
| MMLU-Pro (300) | ✅ | ✅ copied from `agentic-edge` | ✅ copied |
| SciEntsBank test_ua (540) | ✅ | ✅ measured | ✅ measured |
| SciEntsBank test_uq (733) + test_ud (4,562) | ✅ | ❌ **run** | ❌ **run** |
| GSM8K (100) + GSM-Symbolic | ✅ (1,000 variants) | ❌ **run, trimmed** | ❌ **run, full** |
| News (200) | ✅ judged | ❌ **run** | ❌ **run** |

### Scope

| Board | SciEntsBank | Math | News | Estimated time (E2B + E4B) |
|---|---|---|---|---|
| Jetson | test_uq, test_ud | GSM8K 100 + GSM-Symbolic 1,000 (`--instances 10`) | 200 | ~15 h |
| Pi 5 (`-t 3`) | test_uq, test_ud | GSM8K 100 + **the first variant set**, 100 (`--instances 1`) | 200 | ~28–31 h |

**Why the Pi is trimmed:**
- GSM-Symbolic's 1,000 variants alone would cost the Pi 27–42 h.
- The board runs measure cost and check that accuracy matches Kaggle. That match has held in every board run so far: grading on both boards, and MMLU-Pro.
- 200 math questions are enough for the cost per question and for a McNemar check against Kaggle's results on the same ids.

**Estimates:**
- **Grading:** the measured test_ua hours per 1,000 answers.
- **GSM and news:** Kaggle's token counts × the boards' measured decode speeds.
- **Accuracy:** ±30%.

### Code

- **`scripts/board_grade.sh`:** add a benchmark argument, `scientsbank` (default, today's behaviour), `gsm` or `news`.
  - The new paths call `edge-grader-gsm` / `edge-grader-news` against the same `llama-server`.
  - Flags, telemetry wrapper (`run_measured.sh`), Pi `-t 3`, the refusal to run from agentic-edge's queue, and the check that no agentic-edge run has the board: all unchanged.
  - New run folders: `results/board/<benchmark>-<model>-<board>-<set>-<stamp>/`. Existing SciEntsBank folders keep their names.
- **`edge-grader-board-check`:**
  - accept GSM and news summaries
  - pair board against Kaggle on question id
  - exact McNemar
  - J and seconds per question
- **News board runs** are judged on Kaggle by the same Qwen3-14B judge, with the same prompt (`kaggle_news_judge.ipynb` reads any `news-*.json`).
- **Tests:**
  - argument parsing and folder naming in the script (dry-run mode)
  - board-check pairing for each benchmark

### Done when

- Every ❌ above has a measured run folder.
- A board-check file compares each run with Kaggle.
- `docs/results/baselines.md` has one table: benchmark × model × board → accuracy, agreement with Kaggle, J/question, s/question.
- The board builds stay as they are (Pi `661643e`, Jetson `a894dae`) and are recorded per run, as for test_ua.

## 5. Phase A: baseline against RAG, math added

**What's already there:**

| Benchmark | E2B baseline → RAG-en | E4B baseline → RAG-en |
|---|---|---|
| News | 0.5% → 39.0% | 0.5% → 45.0% |
| MMLU-Pro | 51.3% → 52.0% (p 0.89) | 65.3% → 63.3% (p 0.44) |
| SciEntsBank test_ud | 60.3% → 55.1% (p < 0.001) | 70.4% → 70.7% (p 0.59) |
| Math | 83.0% / 80.0% → **to run** | 92.0% / 90.5% → **to run** |

**Math RAG-en:**
- **Corpus:** OpenStax *Prealgebra 2e* (CC BY-NC-SA 4.0, as stated on the book's page; first written here as CC BY 4.0, corrected 2026-10-10). The source is checked and pinned in `sources.json` like the other books, and chunked with the frozen settings: 250 words, 50 overlap, the same noise filters, chapters by bookmarks as for the MMLU-Pro OpenStax books.
- **Index:**
  - `bge-small-en-v1.5`
  - the query is the question text only
  - top 5 stored, top 3 in the prompt
- **Prompt:** the fixed 5-shot GSM prompt unchanged; passages go in front of the question, as in MMLU-Pro RAG-en. Same 1,024-token cap and same extraction.
- **Retrieval check:**
  - 30 GSM8K **train** questions (never test)
  - the owner marks whether any of the top 3 is useful
  - Same rule as before: grade whatever the score, unless it's 0/30 (as with MMLU-Pro RAG-id), and report the score next to the result.
- **Runs:** Kaggle, E2B + E4B, GSM8K 100 + GSM-Symbolic 1,000, build `a3a1c47`.
- **Analysis:** paired McNemar against the baseline per set; the GSM8K-vs-Symbolic gap with RAG.
- **Output:** the four-benchmark table in `docs/results/rag.md` (or a new summary page), ordered from knowledge to skill.

## 6. Phase B: variations on the grading baseline (RQ6)

### B1. Thinking on

- **Flags:** `agentic-edge`'s `mmlupro-thinking-on`: `-rea on --reasoning-budget 320 --reasoning-format none`. Everything else as the baseline.
- **Grader changes:**
  - `max_tokens` rises from 16 to 320 + 16.
  - `parse_label` reads **only the text after the thought block**. The thoughts will name labels, and today's parser takes the first label it sees.
  - The thought block's delimiters are taken from a recorded reply, not guessed, and a test pins them.
- **Runs:**
  - Kaggle: E2B + E4B, all three splits
  - Boards: test_ua for cost
- **MMLU-Pro thinking:** copied from `agentic-edge`, not re-run.
- **Check:** paired McNemar against the baseline per split; tokens, J/answer.

### B2. little-gemma, and prefix reuse

**Setup:**
- **Engine:** little-gemma as `agentic-edge` built it on the Jetson (pinned `aee759d` + patch), served through its `lg_openai_shim.py`.
- **Separation:** started by this project's own script, not the lab queue, under the same separation rules as `board_grade.sh`.
- **Weights and prompts:** the same GGUFs and prompts.

**Variants:**

| Variant | Engine | Answers sorted by question | What it measures |
|---|---|---|---|
| `lg-base` | little-gemma | yes | the engine swap (accuracy + cost) |
| `lg-reuse` | little-gemma `-reuse` | yes | prefix reuse; output must be byte-identical to `lg-base` |
| `llama-sorted` | llama.cpp | yes | the fair control: how much llama-server's own prompt cache gets from the same order (as S10 was for S9) |

**Why sorting matters:**
- The grading prompt is shared up to the student answer: instructions, plus passages under RAG, question, reference answer.
- Sorted by question, consecutive requests share that prefix.
- The test_ua runs recorded no reuse: the full prompt was processed every time. Finding out why (request order, `--cache-ram 0`, or something else) is the first step of B2.
- The prompt text itself is **not** reordered, so results stay comparable.

**Scope:**
- Jetson, test_ua, E2B + E4B, thinking off
- RAG-en versions of `lg-reuse` and `llama-sorted` if the no-RAG run shows a gain, since RAG prompts have the longer shared prefix

**Checks:**
- McNemar against the llama.cpp Jetson baseline
- byte-identical replies for `lg-reuse` against `lg-base`
- J/answer and s/answer for each variant

### B3. EmbeddingGemma 2 (gated)

**The model:**
- Released 2026-10-06; Apache 2.0
- Text-only part about 270M parameters
- Same space for text and images

**Steps:**
1. **Tooling check:** does `sentence-transformers` (in the `embed` group) load it on the Mac? If not, B3 waits; nothing else depends on it.
2. **Retrieval check:** the same 30 SciEntsBank train questions and chunks as `bge-small-en`, marked by the owner the same way.
3. **Gate, set now before any result:** **at least 20/30** useful (bge-small: 16/30) → a SciEntsBank RAG-en grading run on Kaggle with the new index. Below that → report the check only.

## 7. Phase C: text + images for grading (RQ7)

### C1. The student's answer as an image

**Images:**
- Every SciEntsBank student answer rendered to a PNG in an openly licensed handwriting-style font, at a fixed size, with black text on white.
- The renderer and font are pinned, so the images are reproducible from the dataset.
- Images are built on the Mac and uploaded as a private Kaggle dataset with a sha256 (as the RAG chunks were).

**Prompt:**
- Unchanged, except that the `Student answer:` line points to the attached image, which is sent as an OpenAI `image_url` part.
- Question and reference answer stay text.

**Serving:** the baseline flags plus `--mmproj <Gemma 4 vision projector>`, matched to each model. Finding the projector file that pairs with the QAT GGUFs is step 1.

**Reading check:**
- 100 images (fixed seed): the model transcribes the answer, and the character error rate is compared with the dataset text.
- This separates reading errors from grading errors.

**Runs:**
- Kaggle: E2B + E4B, test_ua + test_uq; test_ud if time allows
- Boards: test_ua cost on the Jetson; on the Pi a 100-answer sample, since image encoding on its CPU is the unknown

**Analysis:**
- paired McNemar, image against the text result for the same answer ids
- accuracy loss split by whether the transcription was right

### C2. A question with a figure

**Data search:** first, a dataset that has:
- **real student answers** (not multiple choice)
- **questions with figures**
- grade labels
- English
- a licence that allows research use
- roughly 300+ answers

**Deliverable:** a shortlist with those facts checked, and a decision recorded in the outline. If nothing qualifies, C2 is dropped and that goes in the write-up as a gap. No self-made dataset.

**If one qualifies:**
- the same image-message path as C1
- no-image (text-only question) against with-image, the same models, Kaggle first, then Jetson cost

## 8. Timeline

| Week | Dates | Boards | Kaggle / Mac |
|---|---|---|---|
| 5 | Oct 12–18 | **Phase 0** (Jetson ~15 h, Pi ~30 h, over the weekend) | Phase A (math corpus, check, runs); B1 code + Kaggle runs; B3 tooling + check |
| 6 | Oct 19–25 | B2 on the Jetson; B1 test_ua cost | C1 rendering, reading check, Kaggle runs; C2 data search |
| 7 | Oct 26–Nov 1 | C1 board cost | C2 if a dataset qualifies; B3 grading run if it passed |
| 8 | Nov 2–8 | Stage 4: RAG versions on the boards | Cloud reference (RQ5) |
| 9 | Nov 9–15 | Buffer | Analysis, charts |

**Rule (unchanged):** nothing new runs on the boards until Phase 0's board-check shows the board numbers can be trusted.

**If time runs short, cut in this order:**
1. C2
2. B3's grading run
3. C1's test_ud
4. B2's RAG variants

## 9. Results and docs

- **Results files:** every run writes its summary JSON to `results/` with its settings, as now:
  - board runs: `results/board/`
  - Kaggle: `results/kaggle/`
  - new kinds: `think-*`, `lg-*`, `img-*`, `gsm-rag-*`
- **Separation:** nothing goes into `agentic-edge`. Its numbers are copied with the source commit, as before.
- **`research-outline.md`:**
  - RQ table (§3)
  - supervisor feedback table, 2026-10-08
  - own decisions: Pi trimming, math corpus, B3 gate, sorting for B2, thinking parser
- **`plan.md`:** stage 3 replaced by Phases 0/A/B/C; timeline; change log
- **New results pages:** `docs/results/variations.md` (B), `docs/results/images.md` (C). Phase 0 goes in `baselines.md`, Phase A in `rag.md`.

## 10. Risks

| Risk | What happens |
|---|---|
| Pi run is slower than estimated (±30%) | Phase 0 Pi runs back to back; if it runs past the weekend, B2 on the Jetson starts anyway (different board) |
| Thinking replies break the parser, or the 320 budget truncates the label | Caught by the parser test and the unparsed count; the pilot on 100 answers comes before the full splits, as in stage 1 |
| No vision projector matches the QAT GGUFs | Use the projector from the same publisher's non-QAT release, and note it as a confound; if none loads, C1 runs on Kaggle with the matching non-QAT GGUF as its own baseline |
| EmbeddingGemma 2 too new for the tooling | B3 waits; nothing depends on it |
| llama-server's own prompt cache already gives the reuse gain | That's a finding (as S10 was); B2 reports both |
| Board builds differ from Kaggle's pinned build | Already the case for test_ua, where accuracy matched; recorded per run |
