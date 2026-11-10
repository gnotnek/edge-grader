"""Graded examples in the prompt (spec 2026-11-10): k SciEntsBank train answers, with their gold
3-way labels, before the answer being graded.

    uv run --group embed python -m edge_grader.examples   # on the Mac: CPU embedder only

Writes two files to results/examples/, test answer id -> k train answer ids, in prompt order:
- knn: the most similar train answers by bge-small-en-v1.5 over "<question>\\n<answer>" (the same
  text on both sides, so no query instruction). Answers to the same question first, then the rest
  of the pool; least similar first, most similar last (next to the answer being graded).
- random: k train answers drawn from the whole pool, seeded per test answer id.

Ids only, never text, so the files are committed; the grader looks the text up in the train split
and stops at any id that isn't a train answer. Only train is ever indexed (spec §3).
"""

import argparse
import json
import random
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .data import Item, question_id

OUT_DIR = Path("results/examples")
TEST_SPLITS = ("test_ua", "test_uq", "test_ud")
EMBEDDER = "bge-small-en"  # frozen 2026-10-03
K = 4


def text(item: Item) -> str:
    return f"{item.question}\n{item.student_answer}"


def knn(test_ids: list[str], test_vecs: np.ndarray, train_ids: list[str], train_vecs: np.ndarray,
        k: int) -> dict[str, list[list]]:
    """Per test answer, k [train id, score] in prompt order: same question first, best last."""
    scores = test_vecs @ train_vecs.T
    train_q = [question_id(i) for i in train_ids]
    out = {}
    for row, tid in enumerate(test_ids):
        order = np.argsort(-scores[row], kind="stable")
        same = [j for j in order if train_q[j] == question_id(tid)]
        picks = (same + [j for j in order if train_q[j] != question_id(tid)])[:k]
        out[tid] = [[train_ids[j], round(float(scores[row, j]), 4)] for j in reversed(picks)]
    return out


def random_pick(test_ids: list[str], train_ids: list[str], k: int, seed: int = 0) -> dict[str, list[list]]:
    """k train ids per test answer, the same on every run (a str seed hashes the same everywhere)."""
    pool = sorted(train_ids)
    return {tid: [[j, None] for j in random.Random(f"{seed}-{tid}").sample(pool, k)] for tid in test_ids}


def load(path: str, k: int, train: list[Item]) -> tuple[dict[str, list[Item]], dict]:
    """test answer id -> its k examples (train Items, gold label in .label), and settings to record."""
    f = json.loads(Path(path).read_text())
    if k > f["settings"]["k"]:
        raise SystemExit(f"--k {k} is more than the {f['settings']['k']} examples stored per answer")
    by_id = {i.id: i for i in train}
    out = {}
    for split in f["splits"].values():
        for tid, hits in split.items():
            ids = [h[0] for h in hits][-k:]  # the k nearest, still in prompt order
            if tid in by_id or any(j not in by_id for j in ids):
                raise SystemExit(f"{path}: {tid} or one of its examples isn't train-only; rebuild it")
            out[tid] = [by_id[j] for j in ids]
    return out, {"file": path, "k": k, **{key: f["settings"][key] for key in ("kind", "embedder", "order")}}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--seed", type=int, default=0, help="random file only")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    from . import data
    from .rag.index import MODELS, sentence_transformer

    train = data.load("train")
    tests = {s: data.load(s) for s in TEST_SPLITS}
    train_ids = [i.id for i in train]
    test_ids = [i.id for items in tests.values() for i in items]
    if set(train_ids) & set(test_ids) or len(set(test_ids)) != len(test_ids):
        raise SystemExit("train and test answer ids overlap, or test ids repeat")

    encode = sentence_transformer(EMBEDDER)
    train_vecs = encode([text(i) for i in train], "passage")  # bge-small: no prefix on this side
    files = {"knn": {}, "random": {}}
    for split, items in tests.items():
        ids = [i.id for i in items]
        files["knn"][split] = knn(ids, encode([text(i) for i in items], "passage"), train_ids, train_vecs, args.k)
        files["random"][split] = random_pick(ids, train_ids, args.k, args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    for kind, splits in files.items():
        settings = {
            "kind": kind, "k": args.k, "pool": "scientsbank train", "n_pool": len(train),
            "embedder": MODELS[EMBEDDER]["hf_id"] if kind == "knn" else None,
            "text": "question + newline + student answer, both sides" if kind == "knn" else None,
            "rule": "same question first, then the whole pool" if kind == "knn"
                    else f"whole pool, random.Random(f'{args.seed}-<test id>').sample",
            "order": "least similar first, most similar last" if kind == "knn" else "as drawn",
        }
        if kind == "knn":
            settings["same_question"] = {s: round(sum(
                all(question_id(j) == question_id(t) for j, _ in hits) for t, hits in v.items()) / len(v), 4)
                for s, v in splits.items()}
        name = f"examples-{kind}{'-' + EMBEDDER if kind == 'knn' else ''}-k{args.k}.json"
        (args.out / name).write_text(json.dumps(
            {"kind": "examples", "time": now, "settings": settings, "splits": splits}, indent=1) + "\n")
        print(f"wrote {args.out / name}")


if __name__ == "__main__":
    main()
