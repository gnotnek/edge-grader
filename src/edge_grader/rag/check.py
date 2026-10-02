"""Retrieval check (rag.md step 4): print the top passages for a sample of questions, to mark by hand.

    uv run --group rag python -m edge_grader.rag.check results/rag/retrieval-<index>-k5.json

Samples only questions from the train split (test_ua shares them), never test_uq or test_ud
questions, so the check can't tune the corpus on unseen questions. Spreads the sample over the
units. Writes data/rag/check/<retrieval>.md: each passage gets a box to tick if it would help a
teacher grade the question. data/ is git-ignored because the passages quote the books.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from .index import read_chunks, sha256_of

CHECK_DIR = Path("data/rag/check")


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


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("retrieval", type=Path)
    ap.add_argument("--n", type=int, default=30, help="questions to check")
    ap.add_argument("--show", type=int, default=3, help="passages per question")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

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
