"""Embed the chunks and retrieve passages for every benchmark question (rag.md steps 3-5).

Runs on Kaggle: this Mac doesn't run models (CLAUDE.md).

    uv run --group rag --group embed python -m edge_grader.rag.index build \\
        --chunks data/rag/chunks/scientsbank-en.jsonl --model bge-small-en
    uv run --group rag --group embed python -m edge_grader.rag.index retrieve \\
        --index data/rag/index/scientsbank-en.bge-small-en
    uv run --group rag --group embed python -m edge_grader.rag.index retrieve \\
        --index data/rag/index/mmlupro-en.bge-small-en --benchmark mmlupro

build writes <index>.npy (unit vectors, one row per chunk) and <index>.json (settings).
retrieve searches with the question text only and writes the top-k chunk ids and scores per
question to results/rag/retrieval-<index>-k<k>.json. SciEntsBank: every question of every split,
keyed like "EM.45b". MMLU-Pro: the lab's 300 subset questions and the 70 validation questions
(for the retrieval check), keyed by question_id; the options are never part of the query. That file holds ids, never text, so it can
be committed; the grader looks the text up in the chunks file, whose sha256 it checks.
"""

import argparse
import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ..data import question_id

INDEX_DIR = Path("data/rag/index")
RESULTS_DIR = Path("results/rag")
SPLITS = ("train", "test_ua", "test_uq", "test_ud")

# Each model's own instructions for asymmetric search (short query, longer passage).
MODELS = {
    "bge-small-en": {
        "hf_id": "BAAI/bge-small-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
    },
    "e5-small-multi": {
        "hf_id": "intfloat/multilingual-e5-small",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    "e5-base-multi": {
        "hf_id": "intfloat/multilingual-e5-base",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    "bge-m3": {  # dense vectors only; no instruction prefix
        "hf_id": "BAAI/bge-m3",
        "query_prefix": "",
        "passage_prefix": "",
    },
}

Encode = Callable[[list[str], str], np.ndarray]  # (texts, "query" | "passage") -> unit vectors


def sentence_transformer(model: str) -> Encode:
    from sentence_transformers import SentenceTransformer

    spec = MODELS[model]
    st = SentenceTransformer(spec["hf_id"])

    def encode(texts: list[str], kind: str) -> np.ndarray:
        prefix = spec["query_prefix" if kind == "query" else "passage_prefix"]
        return st.encode([prefix + t for t in texts], batch_size=64, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=len(texts) > 500)

    return encode


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_chunks(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def top_k(queries: np.ndarray, vectors: np.ndarray, k: int) -> list[list[tuple[int, float]]]:
    """Cosine similarity (rows are unit vectors), best first."""
    scores = queries @ vectors.T
    k = min(k, vectors.shape[0])
    best = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    return [[(int(j), float(scores[i, j])) for j in row] for i, row in enumerate(best)]


def with_ext(base: Path, ext: str) -> Path:
    """base + ext. Not Path.with_suffix: the model name after the dot ("scientsbank-id.e5-base-multi")
    would be taken as a suffix and replaced, so every model would share one file."""
    return base.with_name(base.name + ext)


def build(chunks_path: Path, model: str, encode: Encode, out_dir: Path = INDEX_DIR) -> Path:
    chunks = read_chunks(chunks_path)
    vectors = encode([c["text"] for c in chunks], "passage").astype(np.float32)
    base = out_dir / f"{chunks_path.stem}.{model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(with_ext(base, ".npy"), vectors)
    meta = {
        "model": model,
        **MODELS[model],
        "chunks": str(chunks_path),
        "chunks_sha256": sha256_of(chunks_path),
        "n_chunks": len(chunks),
        "dim": int(vectors.shape[1]),
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    with_ext(base, ".json").write_text(json.dumps(meta, indent=2) + "\n")
    return base


def questions(splits=SPLITS) -> dict[str, dict]:
    from .. import data

    out: dict[str, dict] = {}
    for split in splits:
        for item in data.load(split):
            q = out.setdefault(question_id(item.id), {"question": item.question, "splits": []})
            if split not in q["splits"]:
                q["splits"].append(split)
    return out


def retrieve(index: Path, qs: dict[str, dict], encode: Encode, k: int,
             benchmark: str = "scientsbank") -> dict:
    meta = json.loads(with_ext(index, ".json").read_text())
    vectors = np.load(with_ext(index, ".npy"))
    chunk_ids = [c["id"] for c in read_chunks(Path(meta["chunks"]))]
    keys = sorted(qs)
    hits = top_k(encode([qs[key]["question"] for key in keys], "query"), vectors, k)
    return {
        "stage": "rag",
        "kind": "retrieval",
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "settings": {"index": str(index), "k": k, "query": "question text only",
                     "benchmark": benchmark, **meta},
        "questions": {
            key: {"splits": qs[key]["splits"],
                  **({"subject": qs[key]["subject"]} if "subject" in qs[key] else {}),
                  "hits": [[chunk_ids[j], round(s, 4)] for j, s in row]}
            for key, row in zip(keys, hits)
        },
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="embed a chunks file")
    b.add_argument("--chunks", type=Path, required=True)
    b.add_argument("--model", choices=list(MODELS), required=True)
    r = sub.add_parser("retrieve", help="top-k chunks for every question of a benchmark")
    r.add_argument("--index", type=Path, required=True, help="index path without extension")
    r.add_argument("--benchmark", default="scientsbank", choices=["scientsbank", "mmlupro"])
    r.add_argument("--k", type=int, default=5, help="stored per question; grade with --k <= this")
    r.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = ap.parse_args(argv)

    if args.cmd == "build":
        base = build(args.chunks, args.model, sentence_transformer(args.model))
        print(f"wrote {base}.npy and {base}.json")
        return
    meta = json.loads(with_ext(args.index, ".json").read_text())
    if sha256_of(Path(meta["chunks"])) != meta["chunks_sha256"]:
        raise SystemExit(f"{meta['chunks']} changed since the index was built; rebuild it")
    if args.benchmark == "mmlupro":
        from ..mmlupro import questions as mmlupro_questions

        qs = mmlupro_questions()
    else:
        qs = questions()
    result = retrieve(args.index, qs, sentence_transformer(meta["model"]), args.k, args.benchmark)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"retrieval-{args.index.name}-k{args.k}.json"
    path.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {path}: {len(result['questions'])} questions")


if __name__ == "__main__":
    main()
