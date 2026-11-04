"""ASAP 2.0 essay scoring: the untouched model scores essays 1-6 on the official holistic rubric.

    uv run edge-grader-asap sample                      # on the Mac, once: the fixed 600-essay sample
    uv run edge-grader-asap run --label e2b-jetson      # grade it (board_grade.sh: BENCH=asap2)
    uv run edge-grader-asap run --n 5                   # a quick pilot

ASAP 2.0 (Crossley et al., Assessing Writing 2025; Kaggle "Learning Agency Lab - Automated Essay
Scoring 2.0") is ~24k source-based argumentative essays on 7 prompts, scored 1-6. The full file is
Kaggle dataset lburleigh/asap-2-0 (CC BY 4.0), downloaded to data/ and checked against CSV_SHA256.
`sample` draws 600 essays with a fixed seed, balanced over the 7 prompts (each prompt's own score
mix kept), and writes them with each prompt's assignment and source texts to results/reference/
asap2/, so the boards get the same essays through deploy.sh. Demographic columns are dropped.

The prompt is the rubric's "Holistic Rating for Source-Based Writing" (rubric-source-based.txt,
from the competition's data page), the assignment, the source texts and the essay, in that order,
so all but the essay is shared per prompt. "Facial action coding system" ships no source text
("Copyright Restricted"); its prompt says so. Greedy, 8 tokens; the score is the first whole
number 1-6 in the reply, none counts as unparsed.

Scores are ordinal, so the headline is quadratic weighted kappa (the competition's metric), over
the parsed replies; exact accuracy (unparsed = wrong) and adjacent agreement (within 1) beside it.
"""

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import httpx
from openai import APIError, OpenAI
from sklearn.metrics import cohen_kappa_score

from .baseline import server_info

DATA_CSV = Path("data/asap2/ASAP2_train_sourcetexts.csv")
CSV_SHA256 = "df6b816dfe2c764ecd135fccda62aca9a04483b354a15c16ffdfa6c4234a3dea"  # dataset version 8
REF = Path("results/reference/asap2")
RUBRIC = REF / "rubric-source-based.txt"
PROMPTS = REF / "prompts.json"
SAMPLE_N, SAMPLE_SEED = 600, 0
SAMPLE = REF / f"sample-{SAMPLE_N}-seed{SAMPLE_SEED}.jsonl"
SCORES = range(1, 7)
MAX_TOKENS = 8
SCORE_RE = re.compile(r"(?<!\d)[1-6](?!\d)")


def make_sample(csv: Path = DATA_CSV, n: int = SAMPLE_N, seed: int = SAMPLE_SEED) -> tuple[list, dict]:
    """n essays, round-robin over the prompts after a seeded shuffle; and each prompt's texts."""
    import pandas as pd

    d = pd.read_csv(csv).sample(frac=1, random_state=seed)
    d["turn"] = d.groupby("prompt_name").cumcount()
    picked = d.sort_values(["turn", "prompt_name"]).head(n).sort_values(["prompt_name", "essay_id"])
    essays = [{"id": r.essay_id, "prompt": r.prompt_name, "score": int(r.score), "text": r.full_text}
              for r in picked.itertuples()]
    prompts = {}
    for name, r in d.groupby("prompt_name").first().iterrows():
        sources = [r[f"source_text_{i}"] for i in range(1, 5) if isinstance(r[f"source_text_{i}"], str)]
        prompts[name] = {"assignment": r.assignment,
                         "sources": [s for s in sources if s.strip() != "Copyright Restricted"]}
    return essays, prompts


def build_prompt(essay: dict, prompt: dict, rubric: str) -> str:
    if prompt["sources"]:
        sources = "\n\n".join(f"[{i}] {s.strip()}" for i, s in enumerate(prompt["sources"], 1))
    else:
        sources = "(The source text is not available.)"
    return (
        "You are scoring a student's source-based argumentative essay.\n\n"
        "Scoring rubric (holistic, 1 = lowest, 6 = highest):\n"
        f"{rubric.strip()}\n\n"
        f"Assignment: {prompt['assignment'].strip()}\n\n"
        f"Source text(s) the student was given:\n{sources}\n\n"
        f"Student essay:\n{essay['text'].strip()}\n\n"
        "Reply with the score only: one whole number from 1 to 6."
    )


def parse_score(text: str) -> int | None:
    m = SCORE_RE.search(text)
    return int(m.group()) if m else None


def score(gold: list[int], pred: list[int | None]) -> dict:
    pairs = [(g, p) for g, p in zip(gold, pred, strict=True) if p is not None]
    n = len(gold)
    return {
        "n": n,
        "qwk": round(cohen_kappa_score([g for g, _ in pairs], [p for _, p in pairs],
                                       weights="quadratic", labels=list(SCORES)), 4) if pairs else None,
        "accuracy": round(sum(g == p for g, p in pairs) / n, 4),
        "adjacent": round(sum(abs(g - p) <= 1 for g, p in pairs) / n, 4),
        "unparsed": n - len(pairs),
        "pred_counts": {str(k): v for k, v in sorted(Counter(p for _, p in pairs).items())},
        "gold_counts": {str(k): v for k, v in sorted(Counter(gold).items())},
    }


def cmd_sample(args) -> None:
    csv = Path(args.csv)
    digest = hashlib.sha256(csv.read_bytes()).hexdigest()
    if digest != CSV_SHA256:
        sys.exit(f"{csv}: sha256 {digest}, expected {CSV_SHA256} (a different dataset version?)")
    essays, prompts = make_sample(csv)
    REF.mkdir(parents=True, exist_ok=True)
    SAMPLE.write_text("".join(json.dumps(e) + "\n" for e in essays))
    PROMPTS.write_text(json.dumps(prompts, indent=2, ensure_ascii=False))
    print(f"wrote {SAMPLE} ({len(essays)} essays) and {PROMPTS}")
    print(dict(Counter(e["prompt"] for e in essays)), dict(sorted(Counter(e["score"] for e in essays).items())))


def cmd_run(args) -> None:
    essays = [json.loads(line) for line in SAMPLE.read_text().splitlines() if line.strip()]
    if args.n:
        essays = essays[:: max(1, len(essays) // args.n)][: args.n]  # spread over the prompts
    prompts, rubric = json.loads(PROMPTS.read_text()), RUBRIC.read_text()
    server = server_info(args.base_url)
    client = OpenAI(base_url=args.base_url, api_key="not-needed", timeout=600, max_retries=3)

    rows = []
    start = time.perf_counter()
    for i, e in enumerate(essays, 1):
        try:
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model=args.model, temperature=0, max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": build_prompt(e, prompts[e["prompt"]], rubric)}])
            text = resp.choices[0].message.content or ""
            rows.append({"id": e["id"], "prompt": e["prompt"], "gold": e["score"],
                         "pred": parse_score(text), "raw": text,
                         "latency_s": round(time.perf_counter() - t0, 3),
                         "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                         "completion_tokens": getattr(resp.usage, "completion_tokens", None)})
        except (APIError, httpx.HTTPError) as err:  # one failed request must not lose a long run
            rows.append({"id": e["id"], "prompt": e["prompt"], "gold": e["score"], "pred": None,
                         "error": repr(err)})
        print(f"\r{i}/{len(essays)}", end="", flush=True)
    print()
    wall = time.perf_counter() - start

    summary = score([r["gold"] for r in rows], [r["pred"] for r in rows])
    summary["per_prompt"] = {
        p: score([r["gold"] for r in rows if r["prompt"] == p], [r["pred"] for r in rows if r["prompt"] == p])
        for p in sorted({r["prompt"] for r in rows})}
    ok = [r for r in rows if "error" not in r]
    summary["errors"] = len(rows) - len(ok)
    summary["median_latency_s"] = statistics.median(r["latency_s"] for r in ok) if ok else None
    summary["wall_s"] = round(wall, 1)
    summary["prompt_tokens"] = sum(r["prompt_tokens"] or 0 for r in ok)
    summary["completion_tokens"] = sum(r["completion_tokens"] or 0 for r in ok)
    now = datetime.now(UTC)
    run = {
        "benchmark": "asap2", "stage": "baseline", "label": args.label or args.model,
        "model": args.model, "base_url": args.base_url, "server": server,
        "split": "sample", "sample": str(SAMPLE), "n_requested": args.n,
        "max_tokens": MAX_TOKENS, "time": now.isoformat(timespec="seconds"), "metrics": summary,
    }
    name = f"asap2-baseline-{run['label']}-sample-{now.strftime('%Y%m%d-%H%M%S')}"
    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(run, indent=2))
    with open(out / "raw" / f"{name}.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_prompt"}, indent=2))
    print(f"saved {out / (name + '.json')}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="draw the fixed sample from the downloaded CSV")
    s.add_argument("--csv", default=str(DATA_CSV))
    r = sub.add_parser("run", help="score the sample with a running llama-server")
    r.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    r.add_argument("--model", default="local")
    r.add_argument("--label", default=None, help="run label, e.g. e2b-jetson")
    r.add_argument("--n", type=int, default=0, help="essays to score (0 = the whole sample)")
    r.add_argument("--out", default="results/local")
    args = ap.parse_args(argv)
    (cmd_sample if args.cmd == "sample" else cmd_run)(args)


if __name__ == "__main__":
    main()
