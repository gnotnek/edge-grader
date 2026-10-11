"""Grade SciEntsBank answers: the stage-1 baseline, or with retrieved course material (stage 2).

    uv run edge-grader-baseline --base-url http://127.0.0.1:8080/v1 --n 100
    uv run edge-grader-baseline --rag results/rag/retrieval-<index>-k5.json --k 3 ...

--rag takes a retrieval file from `python -m edge_grader.rag.index retrieve`; the passage text
comes from the chunks file it names, which must be unchanged (sha256 checked).
"""

import argparse
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
from openai import APIError, OpenAI

from . import data, grader, metrics
from .data import question_id


def server_info(base_url: str) -> dict:
    """llama-server's /props, for provenance. Exits early if the server isn't up."""
    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        props = httpx.get(f"{root}/props", timeout=10).json()
    except (httpx.HTTPError, ValueError) as e:
        sys.exit(f"no llama-server at {root} ({e}); start it first")
    return {
        "model_path": props.get("model_path"),
        "n_ctx": props.get("default_generation_settings", {}).get("n_ctx"),
        "build_info": props.get("build_info"),
    }


def load_passages(retrieval_path: str, k: int) -> tuple[dict[str, list[tuple[str, str]]], dict]:
    """question id -> its top-k (chunk id, text), and the settings to record."""
    from .rag.index import read_chunks, sha256_of

    retrieval = json.loads(Path(retrieval_path).read_text())
    settings = retrieval["settings"]
    chunks_path = Path(settings["chunks"])
    if sha256_of(chunks_path) != settings["chunks_sha256"]:
        sys.exit(f"{chunks_path} changed since retrieval; rebuild the chunks, index and retrieval")
    if k > settings.get("k", k):
        sys.exit(f"--k {k} is more than the {settings['k']} passages stored per question")
    text = {c["id"]: c["text"] for c in read_chunks(chunks_path)}
    by_question = {
        q: [(cid, text[cid]) for cid, _ in v["hits"][:k]]
        for q, v in retrieval["questions"].items()
    }
    recorded = {
        "retrieval": retrieval_path,
        "k": k,
        "index": settings.get("index"),
        "embedding_model": settings.get("hf_id", settings.get("model")),
        "chunks": str(chunks_path),
        "chunks_sha256": settings["chunks_sha256"],
    }
    return by_question, recorded


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1", help="llama-server endpoint")
    ap.add_argument("--model", default="local", help="model name sent to the server")
    ap.add_argument("--label", default=None, help="run label for the results file, e.g. e4b-q4")
    ap.add_argument("--split", default="test_ua", choices=["test_ua", "test_uq", "test_ud"])
    ap.add_argument("--scheme", default="3way", choices=list(data.SCHEMES))
    ap.add_argument("--n", type=int, default=100, help="number of answers (0 = whole split)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/local", help="directory for this run's files")
    ap.add_argument("--rag", default=None, help="retrieval file: add course material to the prompt")
    ap.add_argument("--k", type=int, default=3, help="passages per prompt with --rag")
    ap.add_argument("--thinking", action="store_true",
                    help="the server runs with -rea on --reasoning-budget 320 --reasoning-format none: "
                         "read the label after the thoughts")
    args = ap.parse_args(argv)
    if args.thinking and args.rag:
        sys.exit("--thinking with --rag isn't a condition in this study")

    passages, rag = load_passages(args.rag, args.k) if args.rag else ({}, None)
    stage = "think" if args.thinking else ("rag" if args.rag else "baseline")
    server = server_info(args.base_url)
    items = data.load(args.split, args.scheme, n=args.n or None, seed=args.seed)
    labels = data.labels_for(args.scheme)
    client = OpenAI(base_url=args.base_url, api_key="not-needed", timeout=600, max_retries=3)

    rows = []
    start = time.perf_counter()
    for i, item in enumerate(items, 1):
        hits = passages.get(question_id(item.id), []) if rag else []
        if rag and not hits:
            sys.exit(f"no retrieved passages for question {question_id(item.id)}")
        try:
            row = grader.grade(client, args.model, item, labels, passages=[t for _, t in hits],
                               thinking=args.thinking)
            if rag:
                row["passages"] = [cid for cid, _ in hits]
            rows.append(row)
        except (APIError, httpx.HTTPError) as e:  # one failed request must not lose a long run
            rows.append({"id": item.id, "gold": item.label, "pred": None, "error": repr(e)})
        print(f"\r{i}/{len(items)}", end="", flush=True)
    print()
    wall = time.perf_counter() - start

    summary = metrics.score([r["gold"] for r in rows], [r["pred"] for r in rows])
    ok = [r for r in rows if "error" not in r]
    summary["errors"] = len(rows) - len(ok)
    summary["median_latency_s"] = statistics.median(r["latency_s"] for r in ok) if ok else None
    summary["wall_s"] = round(wall, 1)
    summary["prompt_tokens"] = sum(r["prompt_tokens"] or 0 for r in ok)
    summary["completion_tokens"] = sum(r["completion_tokens"] or 0 for r in ok)
    summary["median_completion_tokens"] = (
        statistics.median(r["completion_tokens"] or 0 for r in ok) if ok else None)
    now = datetime.now(UTC)
    run = {
        "stage": stage,
        "label": args.label or args.model,
        "model": args.model,
        "base_url": args.base_url,
        "server": server,
        "split": args.split,
        "scheme": args.scheme,
        "n_requested": args.n,
        "seed": args.seed,
        "time": now.isoformat(timespec="seconds"),
        "metrics": summary,
    }
    if rag:
        run["rag"] = rag
    if args.thinking:
        run["thinking"] = {"server_flags": "-rea on --reasoning-budget 320 --reasoning-format none",
                           "max_tokens": grader.THINK_MAX_TOKENS,
                           "parse": "label after the thought block; unclosed or cut replies unparsed"}

    stamp = now.strftime("%Y%m%d-%H%M%S")
    name = f"{stage}-{run['label']}-{args.split}-{args.scheme}-{stamp}"
    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(run, indent=2))
    with open(out / "raw" / f"{name}.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)

    print(json.dumps(summary, indent=2))
    print(f"saved {out / (name + '.json')}")


if __name__ == "__main__":
    main()
