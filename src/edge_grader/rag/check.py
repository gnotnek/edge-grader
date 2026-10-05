"""Retrieval check (rag.md step 4): print the top passages for a sample of questions, to mark by hand.

    uv run --group rag python -m edge_grader.rag.check results/rag/retrieval-<index>-k5.json
    uv run --group rag python -m edge_grader.rag.check results/rag/retrieval-<index>-k5.json --score

Samples only questions from the train split (test_ua shares them), never test_uq or test_ud
questions, so the check can't tune the corpus on unseen questions. Spreads the sample over the
units. Writes data/rag/check/<retrieval>.md: each passage gets a box to tick if it would help a
teacher grade the question. data/ is git-ignored because the passages quote the books.

--score reads the marked sheet back and writes results/rag/check-<retrieval>.json: the chunk ids
and the ticks per question, and the counts. No passage text, so it can be committed.
"""

import argparse
import json
import random
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from .index import read_chunks, sha256_of

CHECK_DIR = Path("data/rag/check")
RESULTS_DIR = Path("results/rag")
PASSAGE_RE = re.compile(r"^- \[(.)\] \*\*\d+\.\*\* `([^`]+)`")


def sample_questions(questions: dict[str, dict], n: int, seed: int = 0) -> list[str]:
    """Round-robin over units, train questions only."""
    by_unit = defaultdict(list)
    for q, v in sorted(questions.items()):
        if "train" in v["splits"]:
            by_unit[q.split(".")[0]].append(q)
    rng = random.Random(seed)
    for qs in by_unit.values():
        rng.shuffle(qs)
    picked, units = [], sorted(by_unit)
    while len(picked) < n and any(by_unit.values()):
        for u in units:
            if by_unit[u] and len(picked) < n:
                picked.append(by_unit[u].pop())
    return sorted(picked)


def report(retrieval: dict, texts: dict[str, str], question_text: dict[str, str], picked: list[str],
           show: int = 3, words: int = 80) -> str:
    s = retrieval["settings"]
    lines = [
        f"# Retrieval check: {Path(s['index']).name}",
        "",
        f"Model `{s.get('hf_id', s['model'])}`, top {show} of {s['k']}, {len(picked)} train questions.",
        "Tick a box if the passage would help a teacher grade answers to the question.",
        "",
    ]
    for q in picked:
        lines += [f"## {q}: {question_text.get(q, '?')}", ""]
        for rank, (cid, score) in enumerate(retrieval["questions"][q]["hits"][:show], 1):
            excerpt = " ".join(texts[cid].split()[:words])
            lines.append(f"- [ ] **{rank}.** `{cid}` ({score:.3f}) {excerpt} …")
        lines.append("")
    return "\n".join(lines)


def score_sheet(text: str) -> dict:
    """The ticks in a marked check sheet: per question, chunk ids and whether each was useful."""
    questions: dict[str, dict] = {}
    current = None
    for line in text.splitlines():
        if line.startswith("## "):
            current = questions.setdefault(line[3:].split(":")[0], {"passages": [], "useful": []})
        elif current is not None and (m := PASSAGE_RE.match(line)):
            current["passages"].append(m.group(2))
            current["useful"].append(m.group(1).lower() == "x")
    marks = [q["useful"] for q in questions.values()]
    return {
        "questions": questions,
        "summary": {
            "questions": len(marks),
            "top1_useful": sum(bool(m and m[0]) for m in marks),
            "any_useful": sum(any(m) for m in marks),
            "passages_useful": sum(sum(m) for m in marks),
            "passages_total": sum(len(m) for m in marks),
        },
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("retrieval", type=Path)
    ap.add_argument("--n", type=int, default=30, help="questions to check")
    ap.add_argument("--show", type=int, default=3, help="passages per question")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score", action="store_true", help="score the marked sheet instead")
    ap.add_argument("--marked-by", default="owner", help="who ticked the boxes (recorded)")
    args = ap.parse_args(argv)

    if args.score:
        sheet = CHECK_DIR / f"{args.retrieval.stem}.md"
        scored = score_sheet(sheet.read_text())
        out = RESULTS_DIR / f"check-{args.retrieval.stem.removeprefix('retrieval-')}.json"
        out.write_text(json.dumps({
            "stage": "rag",
            "kind": "retrieval-check",
            "time": datetime.now(UTC).isoformat(timespec="seconds"),
            "settings": {"retrieval": str(args.retrieval), "sheet": str(sheet),
                         "marked_by": args.marked_by, "split": "train questions only",
                         "useful": "would help a teacher grade answers to the question"},
            **scored,
        }, indent=1) + "\n")
        print(f"wrote {out}: {scored['summary']}")
        return

    retrieval = json.loads(args.retrieval.read_text())
    chunks_path = Path(retrieval["settings"]["chunks"])
    if sha256_of(chunks_path) != retrieval["settings"]["chunks_sha256"]:
        raise SystemExit(f"local {chunks_path} differs from the one retrieval used; rebuild it with "
                         "python -m edge_grader.rag.corpus from the same sources")
    texts = {c["id"]: c["text"] for c in read_chunks(chunks_path)}
    from .. import data

    question_text = {data.question_id(i.id): i.question for i in data.load("train")}
    picked = sample_questions(retrieval["questions"], args.n, args.seed)
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    out = CHECK_DIR / f"{args.retrieval.stem}.md"
    out.write_text(report(retrieval, texts, question_text, picked, args.show))
    print(f"wrote {out}: {len(picked)} questions to mark")


if __name__ == "__main__":
    main()
