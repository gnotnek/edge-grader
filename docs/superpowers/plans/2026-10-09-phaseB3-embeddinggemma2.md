# Phase B3: EmbeddingGemma 2 as the RAG embedder (gated), implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:**
1. Test whether EmbeddingGemma 2, Google's on-device embedder built on Gemma 4 (released 2026-10-06), retrieves better course material for SciEntsBank than `bge-small-en-v1.5`.
2. Grade with it **only** if its retrieval check reaches **20/30** (bge-small: 16/30).

**Architecture:**
- **The embedder:** a fifth entry in `rag/index.py`'s `MODELS`, with its own query/passage prompts from its model card.
- **Index and retrieval:** built on the Mac (CPU) over the **same** `scientsbank-en.jsonl` chunks.
- **The check:** the same 30 train questions, marked by the owner.
- **If it passes:** a Kaggle RAG-en grading run whose file names carry the embedder, so they never mix with the bge-small runs.

**Tech Stack:** Python 3 via uv, sentence-transformers (`embed` group), pytest, the Kaggle notebook.

**Spec:** [`docs/superpowers/specs/2026-10-08-baseline-variations-design.md`](../specs/2026-10-08-baseline-variations-design.md) §6 B3.

## Global Constraints

**Tooling:**
- Python via uv only; `uv add --group embed …` if a newer sentence-transformers is needed. Never pip.
- Embedding models may run on the Mac (owner's instruction, 2026-10-02); Gemma graders don't.

**Only the embedder changes:**
- the chunks file (sha256 in `results/rag/chunks-scientsbank.json`)
- query = question text only
- top 5 stored, `--k 3` in the prompt
- the grading prompt
- the Kaggle build `a3a1c47` and server flags

**The gate, set 2026-10-08 before any result:**
- **≥ 20/30** questions with a useful passage in the top 3 → grade.
- **Below 20/30** → report the check only.
- The owner marks the sheet. Claude doesn't.

**Licence:** if the model is gated behind terms on Hugging Face, the **owner** accepts them in their account. Claude never accepts terms on their behalf.

**Results:**
- Every run writes its summary JSON to `results/`; never hand-edit.
- Passage text stays in `data/` (git-ignored).

**Commits:** end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Wrong prompts for the model.** EmbeddingGemma uses task prompts (the first version used `task: search result | query: ` and `title: none | text: `). Prefixes copied from the wrong version silently weaken retrieval. Task 1 takes them from the v2 model card and pins them in a test.
2. **A different chunk set would make the comparison unfair.** The index must be built from the same `scientsbank-en.jsonl`; `retrieve` refuses a changed file. Task 2 checks the sha256.
3. **The check must sample the same 30 questions as bge-small's.** `check.sample_questions` is seeded and reads the retrieval file's question ids, which are the same set. Task 3: `test_same_check_questions_as_bge_small`.
4. **Kaggle file names must not collide with bge-small's RAG runs.** Today both would be named `rag-<model>-kaggle-t4-en-k3-…`. Task 4 adds the embedder to the tag; test in Task 4, Step 1.
5. **The model may not load in the pinned sentence-transformers.** It's two days old at planning time. Task 1 checks first; if it doesn't load, B3 waits, and nothing else depends on it.

---

### Task 1: the model entry

**Files:**
- Modify: `src/edge_grader/rag/index.py` (`MODELS`), possibly `pyproject.toml`/`uv.lock` (embed group)
- Test: `tests/test_rag_index.py`

**Interfaces:**
- Produces: `MODELS["embeddinggemma2"] = {"hf_id": <id>, "query_prefix": <str>, "passage_prefix": <str>}`. The index key `embeddinggemma2` names files `…scientsbank-en.embeddinggemma2…`.

- [ ] **Step 1: Read the model card**

Find EmbeddingGemma 2 on Hugging Face under the `google` organisation, and record:
- the exact model id
- the licence, and whether it's gated
- the minimum sentence-transformers version
- the **query and document prompts for retrieval**, verbatim
- whether the text-only encoder can be loaded on its own

If the card isn't reachable, stop and tell the owner.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_rag_index.py`:

```python
def test_embeddinggemma2_entry_uses_its_card_prompts():
    from edge_grader.rag.index import MODELS

    m = MODELS["embeddinggemma2"]
    assert m["hf_id"] == "<the id from Step 1>"
    assert m["query_prefix"] == "<the query prompt from Step 1>"
    assert m["passage_prefix"] == "<the document prompt from Step 1>"
```

Fill in the three strings from Step 1 **before running**. They're facts read from the card, not placeholders to leave.

Run: `uv run pytest tests/test_rag_index.py -v`
Expected: FAIL with `KeyError: 'embeddinggemma2'`.

- [ ] **Step 3: Add the entry**

In `MODELS`, after `bge-m3`:

```python
    "embeddinggemma2": {  # Google, 2026-10-06; prompts from its model card (phase B3)
        "hf_id": "<the id from Step 1>",
        "query_prefix": "<the query prompt from Step 1>",
        "passage_prefix": "<the document prompt from Step 1>",
    },
```

If the card asks for a newer sentence-transformers than the lock has, run `uv add --group embed "sentence-transformers>=<version>"` and run the whole suite.

- [ ] **Step 4: Check it loads and encodes** (Mac, CPU)

If the model is gated, the owner first accepts its terms on Hugging Face and logs in with `uv run --group embed huggingface-cli login`. The owner types the token; it isn't pasted into chat.

```bash
uv run --group rag --group embed python -c "
from edge_grader.rag.index import sentence_transformer
enc = sentence_transformer('embeddinggemma2')
v = enc(['How do plants make food?'], 'query'); print(v.shape, float((v**2).sum()))"
```

Expected: a shape like `(1, 768)` and a norm of about 1.0 (`sentence_transformer` normalises).
- **If loading fails:** stop B3, write down the error in `docs/results/variations.md` under `## B3`, and move on.
- **If memory is the problem** on the 8 GB Mac: only if the card offers a text-only checkpoint or loading option, use it, and record that in the entry's comment.

- [ ] **Step 5: Tests, lint, commit**

Run: `uv run pytest -q && uv run ruff check .`

```bash
git add src/edge_grader/rag/index.py tests/test_rag_index.py pyproject.toml uv.lock
git commit -m "RAG: EmbeddingGemma 2 as a model entry (phase B3)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: index and retrieval on the Mac

**Files:** add `results/rag/retrieval-scientsbank-en.embeddinggemma2-k5.json`. The index is local only.

- [ ] **Step 1: Confirm the chunks are the frozen ones**

```bash
shasum -a 256 data/rag/chunks/scientsbank-en.jsonl
uv run python -c "import json; print(json.load(open('results/rag/chunks-scientsbank.json'))['files']['scientsbank-en.jsonl']['sha256'])"
```

The two must match. If they don't, rebuild with `uv run --group rag python -m edge_grader.rag.corpus --lang en` and compare again. Never index a different set.

- [ ] **Step 2: Build and retrieve**

```bash
uv run --group rag --group embed python -m edge_grader.rag.index build --chunks data/rag/chunks/scientsbank-en.jsonl --model embeddinggemma2
uv run --group rag --group embed python -m edge_grader.rag.index retrieve --index data/rag/index/scientsbank-en.embeddinggemma2 --benchmark scientsbank --k 5
```

Expected: `wrote results/rag/retrieval-scientsbank-en.embeddinggemma2-k5.json: N questions`, with the same N as the bge-small file:

```bash
uv run python -c "import json; print(len(json.load(open('results/rag/retrieval-scientsbank-en.bge-small-en-k5.json'))['questions']))"
```

- [ ] **Step 3: Commit**

```bash
git add results/rag/retrieval-scientsbank-en.embeddinggemma2-k5.json
git commit -m "B3: SciEntsBank retrieval with EmbeddingGemma 2

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: the retrieval check and the gate

**Files:**
- Add: `results/rag/check-scientsbank-en.embeddinggemma2-k5.json`
- Test: `tests/test_rag_index.py`

- [ ] **Step 1: A test that the check samples the same 30 questions**

```python
def test_same_check_questions_as_bge_small():
    import json
    from pathlib import Path

    from edge_grader.rag.check import sample_questions

    old = Path("results/rag/retrieval-scientsbank-en.bge-small-en-k5.json")
    new = Path("results/rag/retrieval-scientsbank-en.embeddinggemma2-k5.json")
    if not new.exists():
        pytest.skip("B3 retrieval not built")
    pick = [sample_questions(json.loads(p.read_text())["questions"], 30, 0, "train") for p in (old, new)]
    assert pick[0] == pick[1]
    marked = json.loads(Path("results/rag/check-scientsbank-en.bge-small-en-k5.json").read_text())
    assert sorted(marked["questions"]) == pick[0]
```

The last assert ties it to the questions the owner marked for bge-small. If that check file keys its questions differently, adapt the expression to read its question ids. Don't drop the assert.

Run: `uv run pytest tests/test_rag_index.py -v`
Expected: PASS. Add `import pytest` at the top if missing. The test skips on the boards, where `results/` isn't deployed.

- [ ] **Step 2: Make the sheet, have it marked, score it**

```bash
uv run --group rag python -m edge_grader.rag.check results/rag/retrieval-scientsbank-en.embeddinggemma2-k5.json
```

The owner marks `data/rag/check/retrieval-scientsbank-en.embeddinggemma2-k5.md` with the same rule as before: "would help a teacher grade answers to the question". Then:

```bash
uv run --group rag python -m edge_grader.rag.check results/rag/retrieval-scientsbank-en.embeddinggemma2-k5.json --score --marked-by owner
```

- [ ] **Step 3: Apply the gate and record it**

Read the top-3 count from the printed summary:
- **≥ 20/30:** go to Task 4.
- **Below 20:** B3 ends here.

Either way, write `## B3. EmbeddingGemma 2` in `docs/results/variations.md`:
- the model id and prompts
- a check table next to bge-small (top passage useful, any of top 3, useful passages out of 90)
- the gate decision

Commit:

```bash
git add results/rag/check-scientsbank-en.embeddinggemma2-k5.json tests/test_rag_index.py docs/results/variations.md
git commit -m "B3: EmbeddingGemma 2 retrieval check <x>/30 (gate 20/30: <pass|fail>)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 4 (only if the gate passed): grade on Kaggle

**Files:** modify `notebooks/kaggle_baseline.ipynb` (the RAG-chunks cell); add `results/kaggle/rag-*-en-embeddinggemma2-k3-*`.

- [ ] **Step 1: Put the embedder in the run tag** (edit script, as in phase A Task 5)

In the "RAG only: copy in the chunks" cell, replace

```python
    RAG_ARGS, RAG_TAG = f"--rag {RAG} --k {K}", f"-{lang}-k{K}"
```

with

```python
    # bge-small-en runs keep their names; any other embedder goes into the tag
    embedder = re.search(r"\.([\w-]+)-k\d+\.json$", RAG).group(1)
    RAG_TAG = f"-{lang}-k{K}" if embedder == "bge-small-en" else f"-{lang}-{embedder}-k{K}"
    RAG_ARGS = f"--rag {RAG} --k {K}"
```

Check the regex locally against both file names:

```bash
uv run python -c "
import re
for f in ('results/rag/retrieval-scientsbank-en.bge-small-en-k5.json', 'results/rag/retrieval-scientsbank-en.embeddinggemma2-k5.json'):
    print(re.search(r'\.([\w-]+)-k\d+\.json$', f).group(1))"
```

Expected: `bge-small-en`, then `embeddinggemma2`. Commit and push the notebook.

- [ ] **Step 2: Run (the owner, Kaggle)**

Set:
- `TASK = "scientsbank"`
- `RAG = "results/rag/retrieval-scientsbank-en.embeddinggemma2-k5.json"`
- `K = 3`
- `SPLITS = ["test_ua", "test_uq", "test_ud"]`
- `MODELS = ["e2b", "e4b"]`
- `N = 100` for the pilot, then `N = 0`

The chunks dataset already holds `scientsbank-en.jsonl`.

Unzip into `results/kaggle` and commit.

- [ ] **Step 3: Compare with the bge-small RAG-en run and the baseline, per split**

```bash
uv run python - <<'EOF'
from pathlib import Path
from edge_grader.board_check import compare, outcomes
K = Path("results/kaggle")
for m in ("e2b", "e4b"):
    for s in ("test_ua", "test_uq", "test_ud"):
        base = max(K.glob(f"baseline-{m}-kaggle-t4-{s}-3way-20261005-*.json"))
        bge = max(K.glob(f"rag-{m}-kaggle-t4-en-k3-{s}-3way-*.json"))
        eg = max(K.glob(f"rag-{m}-kaggle-t4-en-embeddinggemma2-k3-{s}-3way-*.json"))
        a, b = compare(outcomes(eg), outcomes(base)), compare(outcomes(eg), outcomes(bge))
        print(f"{m} {s}: baseline {a['second_accuracy']:.1%}, bge {b['second_accuracy']:.1%}, "
              f"EmbeddingGemma 2 {a['first_accuracy']:.1%} (vs baseline p {a['mcnemar']['p']}, vs bge p {b['mcnemar']['p']})")
EOF
```

Add the table to `## B3` in `docs/results/variations.md`, tick `docs/plan.md`, commit and push.
