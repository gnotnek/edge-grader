"""Is the tuned adapter's leniency SciEntsBank's label mix? A check before balanced training.

    uv run python -m edge_grader.label_prior --label e2b-kaggle-t4-ft --dataset beetle --n 500

Grades train answers of a held-out set (Beetle or SAF train: never trained on, never tested on)
and reads the first reply token's top logprobs instead of the generated label. Each label's
score is its best-scoring top token that starts it. "plain" takes the highest score; "corrected"
subtracts the log of the label's share in SciEntsBank train first, a fixed rule (the target is
equal labels), so nothing is tuned on the set being checked. If corrected beats plain, the label
mix explains part of the gap, and balanced training is worth a run.
"""

import argparse
import json
import math
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from openai import OpenAI

from . import data, grader, metrics
from .baseline import server_info

TOP = 20  # top logprobs per position; every label's first token should be among them


def label_scores(top: list[tuple[str, float]], labels: list[str]) -> dict[str, float]:
    """label -> logprob of its best top token that starts it; a label with none gets the lowest
    logprob seen (an upper bound: it was below the cut)."""
    floor = min(lp for _, lp in top)
    scores = {}
    for name in labels:
        hits = [lp for tok, lp in top if (t := tok.strip().lower()) and name.startswith(t)]
        scores[name] = max(hits, default=floor)
    return scores


def corrected(scores: dict[str, float], prior: dict[str, float]) -> str:
    return max(scores, key=lambda name: scores[name] - math.log(prior[name]))


def two(label: str | None) -> str | None:
    return None if label is None else ("correct" if label == "correct" else "incorrect")


def lenient(rows: list[dict], key: str) -> float:
    """Share of answers that aren't correct but are called correct."""
    wrong = [r for r in rows if r["gold"] != "correct"]
    return round(sum(r[key] == "correct" for r in wrong) / len(wrong), 4) if wrong else 0.0


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="local")
    ap.add_argument("--label", required=True, help="run label, e.g. e2b-kaggle-t4-ft")
    ap.add_argument("--dataset", required=True, choices=["beetle", "saf"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/assess")
    args = ap.parse_args(argv)

    server = server_info(args.base_url)
    labels = data.labels_for("3way")
    train_mix = Counter(i.label for i in data.load("train", "3way", dataset="scientsbank"))
    prior = {k: train_mix[k] / sum(train_mix.values()) for k in labels}
    items = data.load("train", "3way", n=args.n, seed=args.seed, dataset=args.dataset)
    client = OpenAI(base_url=args.base_url, api_key="not-needed", timeout=600, max_retries=3)

    rows = []
    for i, item in enumerate(items, 1):
        resp = client.chat.completions.create(
            model=args.model, temperature=0, max_tokens=1, logprobs=True, top_logprobs=TOP,
            messages=[{"role": "user", "content": grader.build_prompt(item, labels)}])
        content = resp.choices[0].logprobs.content
        if not content:
            sys.exit(f"no logprobs in the reply for {item.id}: does this llama-server return them?")
        top = [(t.token, t.logprob) for t in content[0].top_logprobs]
        scores = label_scores(top, labels)
        rows.append({"id": item.id, "gold": item.label, "top": top, "scores": scores,
                     "plain": max(scores, key=scores.get), "corrected": corrected(scores, prior)})
        print(f"\r{i}/{len(items)}", end="", flush=True)
    print()

    result = {}
    for key in ("plain", "corrected"):
        gold, pred = [r["gold"] for r in rows], [r[key] for r in rows]
        result[key] = {"three_way": metrics.score(gold, pred),
                       "two_way": metrics.score([two(g) for g in gold], [two(p) for p in pred]),
                       "lenient": lenient(rows, key)}
    now = datetime.now(UTC)
    run = {
        "kind": "label-prior-check", "label": args.label, "dataset": args.dataset, "split": "train",
        "n_requested": args.n, "seed": args.seed, "server": server, "top_logprobs": TOP,
        "prior": {k: round(v, 4) for k, v in prior.items()}, "prior_counts": dict(train_mix),
        "not_a_label": sum(all(v == min(r["scores"].values()) for v in r["scores"].values())
                           for r in rows),  # no label among the top tokens
        "time": now.isoformat(timespec="seconds"), "metrics": result,
    }
    name = f"{args.dataset}-label-prior-{args.label}-train-{now:%Y%m%d-%H%M%S}"
    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(run, indent=2) + "\n")
    with open(out / "raw" / f"{name}.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    for key, m in result.items():
        print(f"{key:9} 3-way {m['three_way']['accuracy']:.3f}  2-way {m['two_way']['accuracy']:.3f}  "
              f"lenient {m['lenient']:.3f}")
    print(f"saved {out / (name + '.json')}")


if __name__ == "__main__":
    main()
