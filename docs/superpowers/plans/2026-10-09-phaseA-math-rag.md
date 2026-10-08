# Phase A: math RAG (GSM8K + GSM-Symbolic), implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A RAG-en condition for the math benchmark. GSM8K's 100 originals and GSM-Symbolic's 1,000 variants are answered with three passages from OpenStax *Prealgebra 2e* in the prompt. This completes the four-benchmark baseline-vs-RAG table.

**Architecture:** The existing RAG pipeline, extended to a fourth benchmark, `gsm`:
- `sources.json` gets the book.
- `corpus.py` chunks it with the frozen settings.
- `index.py` retrieves for the GSM questions plus a 60-question GSM8K **train** sample.
- `check.py` samples the 30-question check from that train sample.
- `gsm.py` gets `--rag/--k`. Passages go in front of the question in the final user turn, as in MMLU-Pro, and run files are named `gsm-rag-<set>-…` so they never mix with baseline runs.
- Chunks and retrieval are built on the Mac (bge-small-en runs on its CPU). Grading runs on Kaggle.

**Tech Stack:** Python 3 via uv, PyMuPDF (`rag` group), sentence-transformers (`embed` group), pytest, the Kaggle notebook `notebooks/kaggle_baseline.ipynb`.

**Spec:** [`docs/superpowers/specs/2026-10-08-baseline-variations-design.md`](../specs/2026-10-08-baseline-variations-design.md) §5.

## Global Constraints

**Tooling:**
- Python via uv only (`uv sync`, `uv run`, `uv add`). Never pip.
- The Mac runs embedding models only, never Gemma.

**Frozen RAG settings (2026-10-03), unchanged:**
- 250-word chunks, 50 overlap, the `corpus.py` noise filters, OpenStax body by its bookmarks
- `BAAI/bge-small-en-v1.5`
- query = question text only
- top 5 stored, top 3 (`--k 3`) in the prompt

**Math prompt:**
- unchanged: lm-eval gsm8k chat format, the same 5 fixed shots (seed 1234), greedy, 1,024 tokens, stop `["Question:", "</s>", "<|im_end|>"]`, flexible extract as the headline
- with RAG, the final user turn is `Course material, for background only:\n[1] …\n[2] …\n[3] …\n\nQuestion: <q>\nAnswer:`

**Kaggle:** build `a3a1c47`, server flags `-c 8192 -ngl 99 -rea off --reasoning-budget -1 --cache-ram 0`, T4, E2B + E4B.

**Retrieval check:**
- 30 GSM8K **train** questions, never test
- the owner marks them
- **grade whatever the score, unless it's 0/30** (as with MMLU-Pro RAG-id); report the score with the result

**Results:**
- every run writes its summary JSON to `results/`; never hand-edit
- the chunks text stays in `data/` (git-ignored); only records with sha256 are committed

**Commits:** end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **The board-check (phase 0) looks Kaggle GSM runs up by `gsm-<set>-<model>-kaggle-*`.** A RAG run named `gsm-gsm8k-e2b-kaggle-t4-en-k3-…` would match that pattern and be taken as the baseline. RAG runs must be named `gsm-rag-…`. Task 3: `test_rag_run_is_named_gsm_rag_and_records_settings`.
2. **The retrieval check must never sample test questions.** GSM8K's test questions are what's graded. Task 2: `test_gsm_check_samples_train_only`.
3. **The 5 fixed shots are GSM8K train questions.** The check sample must exclude them, or the check marks passages for questions the prompt already answers. Task 2: `test_gsm_questions_exclude_the_shots`.
4. **A GSM-Symbolic variant with no retrieved passages** (an id mismatch between retrieval and runner) must stop the run, not grade it silently without RAG. Task 3: `test_rag_run_stops_when_a_question_has_no_passages`.
5. **The PDF's licence or bookmarks may differ from the MMLU-Pro OpenStax books.** `openstax_body` raises if there are no chapter bookmarks, and the licence is recorded from the book's page, not assumed. Task 1, Steps 3 and 5.

---

## File map

| File | Change | Responsibility |
|---|---|---|
| `src/edge_grader/rag/sources.json` | Modify | The Prealgebra 2e entry (`benchmark: "gsm"`) |
| `src/edge_grader/rag/sources.py` | Modify | `BENCHMARKS` gains `"gsm"` |
| `src/edge_grader/rag/corpus.py` | Modify | `--benchmark gsm` |
| `src/edge_grader/gsm.py` | Modify | `questions()` for retrieval; `--rag/--k`; `gsm-rag-*` names |
| `src/edge_grader/rag/index.py` | Modify | `retrieve --benchmark gsm` |
| `src/edge_grader/rag/check.py` | Modify | gsm: train split, "useful" wording, question text |
| `tests/test_rag_sources.py`, `tests/test_rag_index.py`, `tests/test_gsm.py` | Modify | Tests |
| `notebooks/kaggle_baseline.ipynb` | Modify (cells `c01`, the RAG-chunks cell, `c11`) | GSM with `RAG` |
| `docs/guides/rag-sources.md`, `docs/results/topics.md`, `docs/results/rag.md`, `docs/plan.md` | Modify | Records and results |

---

### Task 1: the book in the manifest

**Files:**
- Modify: `src/edge_grader/rag/sources.json`, `src/edge_grader/rag/sources.py:24`
- Test: `tests/test_rag_sources.py`

**Interfaces:**
- Produces: `sources.BENCHMARKS == ("scientsbank", "mmlupro", "gsm")`, and one book with `"benchmark": "gsm"`, `"id": "en-openstax-prealgebra-2e"`, `"lang": "en"`. The `en-openstax-` prefix makes `corpus.build` cut the body by bookmarks.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rag_sources.py`:

```python
def test_gsm_material_is_one_english_openstax_book():
    gsm = [b for b in load_manifest()["books"] if b["benchmark"] == "gsm"]
    assert [b["id"] for b in gsm] == ["en-openstax-prealgebra-2e"]
    assert gsm[0]["lang"] == "en" and gsm[0]["subjects"] == ["math"]
    assert gsm[0]["sha256"]  # pinned after the first download
```

- [ ] **Step 2: Run it to see it fail**

Run: `uv run pytest tests/test_rag_sources.py -v`
Expected:
- `test_gsm_material_is_one_english_openstax_book` FAILS (`[] == [...]`)
- `test_books_name_their_benchmark_…` still passes

- [ ] **Step 3: Find the PDF and its licence**

Open `https://openstax.org/details/books/prealgebra-2e` in the browser.
- Note the **licence line** exactly as the page states it (expected: Creative Commons Attribution 4.0).
- Copy the **PDF download URL**. The other OpenStax entries use `https://assets.openstax.org/oscms-prodcms/media/documents/<name>_-_WEB.pdf`; take the one the page links to, don't construct it.

If the page is unreachable, stop and ask the owner; don't guess a URL.

- [ ] **Step 4: Add the entry and the benchmark**

In `src/edge_grader/rag/sources.json`, add to `"books"`, after the last `mmlupro` OpenStax entry. Use the same keys as the Calculus entries, with `url` and `licence` from Step 3, and leave `sha256` out (`--pin` adds it):

```json
  {
   "id": "en-openstax-prealgebra-2e",
   "lang": "en",
   "benchmark": "gsm",
   "subjects": ["math"],
   "title": "Prealgebra 2e",
   "edition": "OpenStax, current web edition",
   "publisher": "OpenStax, Rice University",
   "licence": "<the licence line from the book's page>",
   "page_url": "https://openstax.org/details/books/prealgebra-2e",
   "url": "<the PDF link from the book's page>"
  },
```

In `src/edge_grader/rag/sources.py`, change `BENCHMARKS = ("scientsbank", "mmlupro")` to `BENCHMARKS = ("scientsbank", "mmlupro", "gsm")`. Also update the module docstring line `Each book serves one benchmark: "scientsbank" (school science) or "mmlupro" (college subjects).` to `Each book serves one benchmark: "scientsbank" (school science), "mmlupro" (college subjects) or "gsm" (school math, for GSM8K).`

- [ ] **Step 5: Download and pin**

Run: `uv run --group rag python -m edge_grader.rag.sources --only gsm --pin`
Expected: `new      en-openstax-prealgebra-2e …  MB`, then the sha256 is written into `sources.json`. If it fails with `not a PDF`, the URL from Step 3 is wrong.

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/test_rag_sources.py -v`
Expected: all PASS.

- [ ] **Step 7: Document the source and commit**

In `docs/guides/rag-sources.md`, add a section `## Math (GSM8K), phase A`: the title, licence line, page URL, why this book (school-level arithmetic and word problems match GSM8K; the supervisor's hypothesis is about course material for formulas), and that it's chunked with the frozen settings.

```bash
git add src/edge_grader/rag/sources.json src/edge_grader/rag/sources.py tests/test_rag_sources.py docs/guides/rag-sources.md
git commit -m "RAG source for math: OpenStax Prealgebra 2e (benchmark gsm)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: chunks, retrieval and the check know about `gsm`

**Files:**
- Modify: `src/edge_grader/rag/corpus.py` (the `--benchmark` choices), `src/edge_grader/rag/index.py` (the `--benchmark` choices and question loading), `src/edge_grader/rag/check.py` (`CHECK_SPLIT`, `USEFUL`, question text), `src/edge_grader/gsm.py` (new `questions()`)
- Test: `tests/test_rag_index.py`, `tests/test_gsm.py`

**Interfaces:**
- Consumes: `gsm.load(name, instances)`, `gsm.fixed_shots`, `gsm.GSM8K_ID`, `gsm.SHOT_SEED`, `gsm.SHOTS`, `check.sample_questions(questions, n, seed, split)`.
- Produces:
  - `gsm.CHECK_POOL = 60`
  - `gsm.questions(instances: int = 10) -> dict[str, dict]`, with keys:
    - gsm8k ids `"<template>"`, `splits ["gsm8k"]`
    - symbolic ids `"<template>/<instance>"`, `splits ["symbolic"]`
    - train ids `"train-<index>"`, `splits ["train"]`
    - each value `{"question": str, "subject": "math", "splits": [...]}`
  - The ids match the runner's row ids, which Task 3's `load_passages` lookup needs.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gsm.py`:

```python
TRAIN = [{"question": f"train q{i}?", "answer": f"#### {i}"} for i in range(200)]


def fake_load_dataset(name, config=None, split=None):
    assert name == gsm.GSM8K_ID and split == "train"
    return TRAIN


def test_gsm_questions_cover_both_sets_and_a_train_pool(monkeypatch):
    monkeypatch.setattr(gsm, "load", lambda name, instances=10: SETS[name])
    import datasets
    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    qs = gsm.questions()
    assert qs["3"] == {"question": "three?", "subject": "math", "splits": ["gsm8k"]}
    assert qs["5/1"]["splits"] == ["symbolic"]
    train = [k for k, v in qs.items() if v["splits"] == ["train"]]
    assert len(train) == gsm.CHECK_POOL and all(k.startswith("train-") for k in train)


def test_gsm_questions_exclude_the_shots(monkeypatch):
    monkeypatch.setattr(gsm, "load", lambda name, instances=10: SETS[name])
    import datasets
    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    shot_texts = {m["content"] for m in gsm.fixed_shots(TRAIN) if m["role"] == "user"}
    pool = {gsm.turn(v["question"]) for v in gsm.questions().values() if v["splits"] == ["train"]}
    assert not shot_texts & pool
```

In each `SETS` doc of `tests/test_gsm.py`, the runner's `load` returns `messages` ending in a user turn `"Question: <q>\nAnswer:"`. `questions()` takes the question text from that last turn, so `_doc` needs no change. Confirm `_doc`'s last message is `{"role": "user", "content": turn(q)}`; if it isn't, adapt `questions()` to read the same field `_doc` sets.

Append to `tests/test_rag_index.py`:

```python
def test_gsm_check_samples_train_only():
    from edge_grader.rag.check import CHECK_SPLIT, sample_questions

    qs = {"3": {"question": "q", "subject": "math", "splits": ["gsm8k"]},
          "3/0": {"question": "q", "subject": "math", "splits": ["symbolic"]},
          **{f"train-{i}": {"question": "q", "subject": "math", "splits": ["train"]}
             for i in range(60)}}
    picked = sample_questions(qs, 30, 0, CHECK_SPLIT["gsm"])
    assert len(picked) == 30 and all(p.startswith("train-") for p in picked)
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_gsm.py tests/test_rag_index.py -v`
Expected: `AttributeError: module 'edge_grader.gsm' has no attribute 'questions'` and `KeyError: 'gsm'`.

- [ ] **Step 3: Implement `gsm.questions()`**

In `src/edge_grader/gsm.py`, below `load()`, add:

```python
CHECK_POOL = 60  # GSM8K train questions retrieved for, so the retrieval check can sample 30


def questions(instances: int = 10) -> dict[str, dict]:
    """For retrieval: the graded questions (gsm8k originals, symbolic variants) under the
    runner's ids, and a fixed pool of GSM8K train questions for the retrieval check. The pool
    leaves out the five worked examples every prompt carries."""
    from datasets import load_dataset

    out = {}
    for name in ("gsm8k", "symbolic"):
        for q in load(name, instances):
            text = q["messages"][-1]["content"].removeprefix("Question: ").removesuffix("\nAnswer:")
            out[str(q["id"])] = {"question": text, "subject": "math", "splits": [name]}
    train = list(load_dataset(GSM8K_ID, "main", split="train"))
    shots = set(random.Random(SHOT_SEED).sample(range(len(train)), SHOTS))
    pool = [i for i in random.Random(0).sample(range(len(train)), CHECK_POOL + SHOTS)
            if i not in shots][:CHECK_POOL]
    for i in pool:
        out[f"train-{i}"] = {"question": train[i]["question"], "subject": "math",
                             "splits": ["train"]}
    return out
```

`fixed_shots` picks with `random.Random(seed).sample(range(len(train)), n)`, so the `shots` set above is exactly the five indices it uses.

- [ ] **Step 4: Wire `gsm` into corpus, index and check**

`src/edge_grader/rag/corpus.py`:
- In `main`, change `choices=["scientsbank", "mmlupro", "news"]` to `choices=["scientsbank", "mmlupro", "news", "gsm"]`.
- In the docstring usage block, add `    uv run --group rag python -m edge_grader.rag.corpus --benchmark gsm --lang en   # math (OpenStax Prealgebra)`.

`build()` already filters books by `benchmark` and uses `openstax_body` for `en-openstax-*`. Nothing else changes.

`src/edge_grader/rag/index.py`:
- In `main`, change the `retrieve` `--benchmark` choices to `["scientsbank", "mmlupro", "news", "gsm"]`.
- Before `else: qs = questions()`, add:

```python
    elif args.benchmark == "gsm":
        from ..gsm import questions as gsm_questions

        qs = gsm_questions()
```

`src/edge_grader/rag/check.py`:
- `CHECK_SPLIT` gains `"gsm": "train"`.
- `USEFUL` gains `"gsm": "holds a concept, formula or method that helps solve the problem"`.
- In `main`, before the final `else:` that loads SciEntsBank's train questions, add:

```python
    elif benchmark == "gsm":
        from ..gsm import questions as gsm_questions

        question_text = {q: v["question"] for q, v in gsm_questions().items()}
```

- In the docstring, after the MMLU-Pro sentence, add: `GSM samples a fixed pool of 60 GSM8K train questions (not the five worked examples), so the graded test questions are never looked at.`

- [ ] **Step 5: Run the tests and lint**

Run: `uv run pytest tests/test_gsm.py tests/test_rag_index.py -v && uv run pytest -q && uv run ruff check .`
Expected: all PASS, ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/edge_grader/gsm.py src/edge_grader/rag/corpus.py src/edge_grader/rag/index.py src/edge_grader/rag/check.py tests/test_gsm.py tests/test_rag_index.py
git commit -m "RAG pipeline: gsm benchmark (questions, corpus, retrieval, train-only check)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `edge-grader-gsm --rag`

**Files:**
- Modify: `src/edge_grader/gsm.py` (`load` signature unchanged; `main` gets `--rag/--k`; the message builder)
- Test: `tests/test_gsm.py`

**Interfaces:**
- Consumes: `baseline.load_passages(retrieval_path, k) -> (dict[qid, list[(chunk_id, text)]], recorded)`, and Task 2's question ids.
- Produces:
  - `gsm.with_passages(messages: list[dict], passages: list[str]) -> list[dict]`: a copy with the material in front of the last user turn.
  - RAG run files: `<out>/gsm-rag-<set>-<label>-<stamp>.json`, with `"stage": "rag"` and a `"rag"` block.
  - Each raw row gets `"passages": [chunk ids]`.
  - Without `--rag`, nothing changes (file `gsm-<set>-…`, `"stage": "topics"`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gsm.py`:

```python
def test_with_passages_prefixes_only_the_last_turn():
    msgs = [{"role": "user", "content": "Question: shot?\nAnswer:"},
            {"role": "assistant", "content": "#### 1"},
            {"role": "user", "content": "Question: three?\nAnswer:"}]
    out = gsm.with_passages(msgs, ["Add the parts.", "Count up."])
    assert out[:2] == msgs[:2]
    assert out[2]["content"] == ("Course material, for background only:\n[1] Add the parts.\n"
                                 "[2] Count up.\n\nQuestion: three?\nAnswer:")
    assert msgs[2]["content"] == "Question: three?\nAnswer:"  # the input isn't changed


@pytest.fixture
def retrieval(tmp_path):
    chunks = tmp_path / "gsm-en.jsonl"
    chunks.write_text("".join(json.dumps({"id": f"en-openstax-prealgebra-2e:{i}", "text": f"p{i}"}) + "\n"
                              for i in range(3)))
    import hashlib
    r = {"settings": {"chunks": str(chunks), "k": 5, "index": "x", "model": "bge-small-en",
                      "chunks_sha256": hashlib.sha256(chunks.read_bytes()).hexdigest()},
         "questions": {q: {"hits": [[f"en-openstax-prealgebra-2e:{i}", 0.5] for i in range(3)]}
                       for q in ("3", "5", "3/0", "3/1", "5/0", "5/1")}}
    path = tmp_path / "retrieval-gsm-en.bge-small-en-k5.json"
    path.write_text(json.dumps(r))
    return path


def test_rag_run_is_named_gsm_rag_and_records_settings(server, tmp_path, retrieval):
    gsm.main(["--base-url", server, "--label", "fake", "--out", str(tmp_path / "out"),
              "--set", "gsm8k", "--rag", str(retrieval), "--k", "2"])
    files = list((tmp_path / "out").glob("gsm-*.json"))
    assert [f.name.startswith("gsm-rag-gsm8k-fake-") for f in files] == [True]
    run = json.loads(files[0].read_text())
    assert run["stage"] == "rag" and run["rag"]["k"] == 2
    assert run["metrics"]["accuracy"] == 1.0
    last = FakeLlamaServer.seen[0]["messages"][-1]["content"]
    assert last.startswith("Course material, for background only:\n[1] p0\n[2] p1\n\n")
    rows = [json.loads(x) for x in next((tmp_path / "out" / "raw").glob("*.jsonl")).read_text().splitlines()]
    assert rows[0]["passages"] == ["en-openstax-prealgebra-2e:0", "en-openstax-prealgebra-2e:1"]


def test_rag_run_stops_when_a_question_has_no_passages(server, tmp_path, retrieval):
    r = json.loads(retrieval.read_text())
    del r["questions"]["5"]
    retrieval.write_text(json.dumps(r))
    with pytest.raises(SystemExit, match="no retrieved passages for question 5"):
        gsm.main(["--base-url", server, "--label", "fake", "--out", str(tmp_path / "out"),
                  "--set", "gsm8k", "--rag", str(retrieval)])
```

Make the fake server ignore the material. In `FakeLlamaServer.do_POST`, replace the line `last = body["messages"][-1]["content"]` with:

```python
        last = body["messages"][-1]["content"].split("\n\n")[-1]  # past any course material
```

The split changes nothing for prompts without material, so the existing tests are unaffected.

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_gsm.py -v`
Expected:
- `AttributeError: … 'with_passages'`
- `unrecognized arguments: --rag`
- the existing tests still PASS

- [ ] **Step 3: Implement**

In `src/edge_grader/gsm.py`, add below `fixed_shots`:

```python
def with_passages(messages: list[dict], passages: list[str]) -> list[dict]:
    """The same chat with course material in front of the final question (as MMLU-Pro's RAG);
    the worked examples are untouched."""
    numbered = "\n".join(f"[{i}] {p}" for i, p in enumerate(passages, 1))
    last = messages[-1]
    material = f"Course material, for background only:\n{numbered}\n\n"
    return [*messages[:-1], {**last, "content": material + last["content"]}]
```

In `main`:
1. Change `from .baseline import server_info` to `from .baseline import load_passages, server_info`.
2. Add the arguments after `--out`:

```python
    ap.add_argument("--rag", default=None, help="retrieval file: add course material to the prompt")
    ap.add_argument("--k", type=int, default=3, help="passages per prompt with --rag")
```

3. After `args = ap.parse_args(argv)`, add `passages, rag = load_passages(args.rag, args.k) if args.rag else ({}, None)`.
4. In the per-question loop, replace `row |= ask(client, args.model, q["messages"], args.max_tokens)` with:

```python
            msgs = q["messages"]
            if rag:
                hits = passages.get(str(q["id"]), [])
                if not hits:
                    sys.exit(f"no retrieved passages for question {q['id']}")
                msgs = with_passages(msgs, [t for _, t in hits])
                row["passages"] = [cid for cid, _ in hits]
            row |= ask(client, args.model, msgs, args.max_tokens)
```

The `sys.exit` sits inside the `try`, but `SystemExit` isn't an `APIError`/`HTTPError`, so it isn't caught. Add `import sys` at the top.

5. In the `run` dict, change `"stage": "topics",` to `"stage": "rag" if rag else "topics",`. After building `run`, add `if rag: run["rag"] = rag`.
6. Change `stem = f"gsm-{name}-{label}-…"` to:

```python
        stem = f"gsm-{'rag-' if rag else ''}{name}-{label}-{now.strftime('%Y%m%d-%H%M%S')}"
```

7. In the module docstring, add a usage line `    uv run edge-grader-gsm --set gsm8k symbolic --rag results/rag/retrieval-gsm-en.bge-small-en-k5.json --k 3`, and a closing paragraph: `With --rag, three passages of course material (OpenStax Prealgebra) go in front of the final question, as in MMLU-Pro's RAG; the worked examples are unchanged. RAG runs are named gsm-rag-<set>-…, so a baseline lookup never picks one up.`

- [ ] **Step 4: Run the tests and lint**

Run: `uv run pytest tests/test_gsm.py -v && uv run pytest -q && uv run ruff check .`
Expected: all PASS, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/edge_grader/gsm.py tests/test_gsm.py
git commit -m "edge-grader-gsm --rag: course material before the question; gsm-rag-* files

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: build the chunks, retrieve and check (on the Mac)

**Files:**
- Add: `results/rag/chunks-gsm.json`, `results/rag/retrieval-gsm-en.bge-small-en-k5.json`, `results/rag/check-gsm-en.bge-small-en-k5.json` (all written by the CLIs)
- Local only (git-ignored): `data/rag/chunks/gsm-en.jsonl`, `data/rag/index/gsm-en.bge-small-en.*`, `data/rag/check/retrieval-gsm-en.bge-small-en-k5.md`

**Interfaces:** Produces the retrieval file Task 5 grades with, and the check score that decides whether Task 5 runs.

- [ ] **Step 1: Chunk**

Run: `uv run --group rag python -m edge_grader.rag.corpus --benchmark gsm --lang en`
Expected: `data/rag/chunks/gsm-en.jsonl: N chunks (N from books, 0 from Wikipedia)`, with N roughly 1,000–2,500 for a ~1,200-page book, and `wrote results/rag/chunks-gsm.json`.

If it raises `no chapter bookmarks`, the PDF's top-level bookmarks don't start with "Chapter"/"Unit". Print `pymupdf.open(path).get_toc()[:15]`. If they read e.g. `1 Whole Numbers`, extend `CHAPTER_RE` to `r"^(chapter|unit)\s+\d|^\d+\s+[A-Z]"` and add a `test_openstax_body_…` case for it in `tests/test_rag_corpus.py` before re-running. Don't cut pages by hand.

Spot-check: `head -c 1500 data/rag/chunks/gsm-en.jsonl`. The first chunk should be chapter 1 text, not the preface or contents.

- [ ] **Step 2: Embed and retrieve** (CPU, a few minutes)

```bash
uv run --group rag --group embed python -m edge_grader.rag.index build --chunks data/rag/chunks/gsm-en.jsonl --model bge-small-en
uv run --group rag --group embed python -m edge_grader.rag.index retrieve --index data/rag/index/gsm-en.bge-small-en --benchmark gsm --k 5
```

Expected: `wrote results/rag/retrieval-gsm-en.bge-small-en-k5.json: 1160 questions` (100 + 1,000 + 60).

- [ ] **Step 3: Make the check sheet**

Run: `uv run --group rag python -m edge_grader.rag.check results/rag/retrieval-gsm-en.bge-small-en-k5.json`
Expected: `wrote data/rag/check/retrieval-gsm-en.bge-small-en-k5.md: 30 questions to mark`.

- [ ] **Step 4: The owner marks the sheet**

Hand the file to the owner. They tick each passage that "holds a concept, formula or method that helps solve the problem". **Claude doesn't mark it:** the 2026-10-06 correction showed why a first pass isn't a marking.

- [ ] **Step 5: Score, record, commit**

```bash
uv run --group rag python -m edge_grader.rag.check results/rag/retrieval-gsm-en.bge-small-en-k5.json --score --marked-by owner
```

Expected: `wrote results/rag/check-gsm-en.bge-small-en-k5.json: {…}` with the number of questions that have a useful passage in the top 3.

**The gate:**
- **If it's 0/30**, stop. Record in `docs/research-outline.md` (own decisions) that math RAG isn't graded, as for MMLU-Pro RAG-id, and skip Task 5.
- **Otherwise, continue.**

Add the score to `docs/results/topics.md` under a new heading `### Math RAG-en: retrieval check`, with a row in the comparison table next to news (21/30), SciEntsBank (16/30) and MMLU-Pro (15/30).

```bash
git add results/rag/chunks-gsm.json results/rag/retrieval-gsm-en.bge-small-en-k5.json results/rag/check-gsm-en.bge-small-en-k5.json docs/results/topics.md
git commit -m "Math RAG: Prealgebra chunks, retrieval and the owner's check (<x>/30)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 5: grade on Kaggle

**Files:**
- Modify: `notebooks/kaggle_baseline.ipynb` (cells `c01`, the "RAG only: copy in the chunks" code cell, `c11`)
- Add: `results/kaggle/gsm-rag-{gsm8k,symbolic}-{e2b,e4b}-kaggle-t4-en-k3-*.json` and their raw files

**Interfaces:**
- Consumes: Task 4's retrieval file; `data/rag/chunks/gsm-en.jsonl` uploaded to the private chunks dataset.
- Produces: the four RAG runs that Task 6 pairs with `results/kaggle/gsm-{gsm8k,symbolic}-{e2b,e4b}-kaggle-t4-2026100*.json`.

- [ ] **Step 1: Let the notebook run GSM with RAG** (an edit script keeps the JSON valid; the notebook round-trips with `json.dumps(nb, indent=1, ensure_ascii=False) + "\n"`)

Save as `$SCRATCH/edit_baseline_nb.py` and run it with `uv run python $SCRATCH/edit_baseline_nb.py notebooks/kaggle_baseline.ipynb`:

```python
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
nb = json.loads(path.read_text())


def cell_with(text: str) -> dict:
    (c,) = [c for c in nb["cells"] if text in "".join(c["source"])]
    return c


def swap(cell: dict, old: str, new: str) -> None:
    s = "".join(cell["source"])
    assert s.count(old) == 1, old
    cell["source"] = s.replace(old, new).splitlines(keepends=True)


c01 = cell_with("# What to run.")
swap(c01, "# GSM has no RAG condition.\n",
     "# GSM (TASK = \"gsm\"), OpenStax Prealgebra 2e (phase A; set GSM_SETS = [\"gsm8k\", \"symbolic\"]):\n"
     "#   RAG = \"results/rag/retrieval-gsm-en.bge-small-en-k5.json\"\n")
rag = cell_with("if RAG:\n    REPO =")
swap(rag, r're.search(r"(scientsbank|mmlupro|news)-(\w+)\.", RAG)',
     r're.search(r"(scientsbank|mmlupro|news|gsm)-(\w+)\.", RAG)')
c11 = cell_with("def start_server(key):")
swap(c11, '            assert not RAG, "GSM has no RAG condition"\n',
     '            assert not (RAG and "tiny" in GSM_SETS), "tiny is the runner check: no RAG"\n')
swap(c11, "--set {gsm_set} --max-tokens {tokens} --n {N} --out results/kaggle",
     "--set {gsm_set} --max-tokens {tokens} --n {N} --out results/kaggle {RAG_ARGS}")
swap(c11, "--label {key}-kaggle-{GPU}{tag} --set {gsm_set}",
     "--label {key}-kaggle-{GPU}{tag}{RAG_TAG} --set {gsm_set}")
path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
```

Check: `git diff --stat notebooks/kaggle_baseline.ipynb` shows one file with a few lines changed.

```bash
git add notebooks/kaggle_baseline.ipynb
git commit -m "Kaggle notebook: GSM with RAG (phase A)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

- [ ] **Step 2: Upload the chunks (the owner, Kaggle UI)**

In the private chunks dataset (the one `kaggle_rag_index.ipynb` reads), choose **New Version** and add `data/rag/chunks/gsm-en.jsonl` next to the existing files. The notebook checks it against `results/rag/chunks-gsm.json`'s sha256.

- [ ] **Step 3: Pilot, then the full runs (the owner, Kaggle UI)**

1. **Settings** in `c01`:
   - `TASK = "gsm"`
   - `GSM_SETS = ["gsm8k", "symbolic"]`
   - `MODELS = ["e2b", "e4b"]`
   - `RAG = "results/rag/retrieval-gsm-en.bge-small-en-k5.json"`
   - `K = 3`
   - `N = 5`
   - Add the chunks dataset as an input. GPU T4, Internet on.
2. **Run all.**
3. **Check the pilot** in cell `c13`:
   - `errors` 0
   - `hit_token_cap` and `no_number` near 0
   - The chunks cell printed `gsm-en.jsonl: … sha256 matches`.
4. **Then set `N = 0`** and run cell `c11` again: about 1.5–2 h per model on a T4. Passages add ~1,000 prompt tokens per question, on top of the 1,100-token 5-shot prompt.
5. **Download** `results-kaggle.zip`, then on the Mac:

```bash
unzip -o ~/Downloads/results-kaggle.zip -d results/kaggle
git add results/kaggle
git commit -m "Math RAG-en on Kaggle: GSM8K + GSM-Symbolic, E2B + E4B

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Delete the 5-question pilot files only if they were saved with the full ones. **Never delete a full run.**

---

### Task 6: analysis and the four-benchmark table

**Files:**
- Modify: `docs/results/topics.md`, `docs/results/rag.md`, `docs/plan.md`, `docs/research-outline.md` (stage A output cell)

- [ ] **Step 1: Paired comparison against the baseline**

```bash
uv run python - <<'EOF'
from pathlib import Path
from edge_grader.board_check import compare, outcomes
K = Path("results/kaggle")
for model in ("e2b", "e4b"):
    for s in ("gsm8k", "symbolic"):
        base = max(p for p in K.glob(f"gsm-{s}-{model}-kaggle-t4-2026*.json"))
        rag = max(K.glob(f"gsm-rag-{s}-{model}-kaggle-t4-en-k3-*.json"))
        c = compare(outcomes(rag), outcomes(base))
        print(f"{model} {s}: n {c['n']}, baseline {c['second_accuracy']:.1%} -> RAG "
              f"{c['first_accuracy']:.1%}, better/worse {c['mcnemar']['only_first_right']}/"
              f"{c['mcnemar']['only_second_right']}, McNemar p {c['mcnemar']['p']}  [{base.name} | {rag.name}]")
EOF
```

This uses phase 0's `board_check.outcomes`/`compare`. If phase 0 Task 2 isn't merged yet, do it first, or compute the same counts from the two raw files' `correct` fields.

- [ ] **Step 2: The memorisation gap with RAG**

The baseline's GSM8K-vs-Symbolic permutation test lives in the analysis that wrote `docs/results/topics.md` (2026-10-08). Run the same test on the RAG pair, and report the gap with and without RAG side by side.

- [ ] **Step 3: Write it up**

**`docs/results/topics.md`:** add `### Math RAG-en (Kaggle T4, <date>)`:
- the table from Step 1
- the gap from Step 2
- the retrieval check score
- three plain findings: did RAG move GSM8K, did it move GSM-Symbolic, and did more answers hit the token cap

**`docs/results/rag.md`:** add `## Baseline against RAG-en, all four benchmarks`, with the table ordered from knowledge to skill:

| Benchmark | Needs | E2B baseline → RAG | E4B baseline → RAG | Retrieval check |
|---|---|---|---|---|
| News | facts after the cutoff | 0.5% → 39.0% | 0.5% → 45.0% | 21/30 |
| MMLU-Pro | broad subject knowledge | 51.3% → 52.0% | 65.3% → 63.3% | 15/30 |
| SciEntsBank (test_ud) | judging an answer | 60.3% → 55.1% | 70.4% → 70.7% | 16/30 |
| Math (GSM8K / GSM-Symbolic) | a skill the model has | <from Step 1> | <from Step 1> | <x>/30 |

Every number comes from a committed result file; cite each file under the table.

**`docs/plan.md`:** tick the phase-A items.

**`docs/research-outline.md`:** replace the stage-A output cell with `Done <date>: <one line>`, linking to the rag.md section.

- [ ] **Step 4: Commit and push**

```bash
git add docs/results/topics.md docs/results/rag.md docs/plan.md docs/research-outline.md
git commit -m "Phase A done: baseline vs RAG-en for all four benchmarks

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```
