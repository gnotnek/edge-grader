"""Stage 1 baseline: an untouched model grades SciEntsBank answers.

    uv run edge-grader-baseline --base-url http://127.0.0.1:8080/v1 --n 100
"""

import argparse
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path

from openai import OpenAI

from . import data, grader, metrics


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1", help="llama-server endpoint")
    ap.add_argument("--model", default="local", help="model name sent to the server")
    ap.add_argument("--label", default=None, help="run label for the results file, e.g. e4b-q4")
    ap.add_argument("--split", default="test_ua", choices=["test_ua", "test_uq", "test_ud"])
    ap.add_argument("--scheme", default="3way", choices=list(data.SCHEMES))
    ap.add_argument("--n", type=int, default=100, help="number of answers (0 = whole split)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    items = data.load(args.split, args.scheme, n=args.n or None, seed=args.seed)
    labels = data.labels_for(args.scheme)
    client = OpenAI(base_url=args.base_url, api_key="not-needed")

    rows = []
    for i, item in enumerate(items, 1):
        rows.append(grader.grade(client, args.model, item, labels))
        print(f"\r{i}/{len(items)}", end="", flush=True)
    print()

    summary = metrics.score([r["gold"] for r in rows], [r["pred"] for r in rows])
    summary["median_latency_s"] = statistics.median(r["latency_s"] for r in rows)
    now = datetime.now(UTC)
    run = {
        "stage": "baseline",
        "label": args.label or args.model,
        "model": args.model,
        "base_url": args.base_url,
        "split": args.split,
        "scheme": args.scheme,
        "seed": args.seed,
        "time": now.isoformat(timespec="seconds"),
        "metrics": summary,
    }

    stamp = now.strftime("%Y%m%d-%H%M%S")
    name = f"baseline-{run['label']}-{args.split}-{args.scheme}-{stamp}"
    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(run, indent=2))
    with open(out / "raw" / f"{name}.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)

    print(json.dumps(summary, indent=2))
    print(f"saved {out / (name + '.json')}")


if __name__ == "__main__":
    main()
