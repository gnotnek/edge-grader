"""Estimate grading cost on the Pi 5 / Jetson from agentic-edge's measured device rates.

The supervisor asked (2026-09-29) to reuse the lab's device benchmarks instead of re-measuring.
agentic-edge's export records every MMLU-Pro request on both boards (same GGUF files, same
llama-server flags), with prompt tokens processed, prefill and decode time, and board power.
Per board and model this fits

    prefill ms = a + b * prompt tokens     (requests with at most --max-prompt tokens)
    decode ms  = median ms per generated token

and applies it to each graded answer's own token counts from a Kaggle run:

    latency = a + b * prompt_tokens + completion_tokens * decode_ms
    energy  = median board power during a request * latency

It uses the full prompt token count, so any prefix llama-server reuses from the previous
request is counted again: an upper bound. Power is board DC draw, not wall power.

    uv run edge-grader-cost results/kaggle/baseline-*.json
"""

import argparse
import gzip
import json
import statistics
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

EXPORT = "results/reference/agentic-edge/export/experiment-full.json.gz"
SOURCE_COMMIT = "12031f0"
# Exported twice under two names; mmlupro100-e4b-s1 is the one to count (export/README.md).
SUPERSEDED = {("pi", "mmlupro100-e4b")}


@dataclass
class Rates:
    board: str
    model: str
    n_requests: int
    prefill_fixed_ms: float
    prefill_ms_per_token: float
    prefill_r2: float
    decode_ms_per_token: float
    watts: float


def fit_rates(board: str, model: str, requests: list[dict], max_prompt: int) -> Rates:
    short = [r for r in requests if r["pt"] <= max_prompt]
    x = [r["pt"] for r in short]
    y = [r["pms"] for r in short]
    slope, intercept = statistics.linear_regression(x, y)
    mean_y = statistics.fmean(y)
    ss_res = sum((yi - (intercept + slope * xi)) ** 2 for xi, yi in zip(x, y))
    ss_tot = sum((yi - mean_y) ** 2 for yi in y)
    return Rates(
        board=board,
        model=model,
        n_requests=len(short),
        prefill_fixed_ms=round(intercept, 1),
        prefill_ms_per_token=round(slope, 3),
        prefill_r2=round(1 - ss_res / ss_tot, 3),
        decode_ms_per_token=round(statistics.median(r["gms"] / r["gt"] for r in requests if r["gt"]), 1),
        watts=round(statistics.median(r["dev"]["w_mean"] for r in requests), 2),
    )


def load_rates(export_path: str | Path, max_prompt: int) -> list[Rates]:
    with gzip.open(export_path, "rt") as f:
        export = json.load(f)
    rates = []
    for board, bd in export["boards"].items():
        by_model: dict[str, list[dict]] = {}
        for name, run in bd["runs"].items():
            if (board, name) not in SUPERSEDED:
                by_model.setdefault(run["model"], []).extend(run["timeline"])
        rates += [fit_rates(board, m, reqs, max_prompt) for m, reqs in sorted(by_model.items())]
    return rates


def estimate(rates: Rates, answers: list[dict]) -> dict:
    lat = [
        (
            rates.prefill_fixed_ms
            + rates.prefill_ms_per_token * a["prompt_tokens"]
            + rates.decode_ms_per_token * a["completion_tokens"]
        )
        / 1000
        for a in answers
    ]
    joules = [rates.watts * s for s in lat]
    return {
        "median_latency_s": round(statistics.median(lat), 2),
        "mean_latency_s": round(statistics.fmean(lat), 2),
        "mean_j_per_answer": round(statistics.fmean(joules), 2),
        "wh_per_1000_answers": round(sum(joules) / len(joules) * 1000 / 3600, 2),
        "hours_per_1000_answers": round(statistics.fmean(lat) * 1000 / 3600, 2),
    }


def read_run(summary_path: Path) -> tuple[dict, list[dict]]:
    summary = json.loads(summary_path.read_text())
    raw = summary_path.parent / "raw" / summary_path.with_suffix(".jsonl").name
    answers = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
    return summary, answers


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="grading run summaries (baseline-*.json)")
    ap.add_argument("--export", default=EXPORT, help="agentic-edge experiment export (.json.gz)")
    ap.add_argument("--max-prompt", type=int, default=600, help="fit prefill on requests up to this")
    ap.add_argument("--min-n", type=int, default=101, help="skip runs smaller than this (pilots)")
    ap.add_argument("--out", default="results/estimates", help="directory for the summary JSON")
    args = ap.parse_args(argv)

    rates = load_rates(args.export, args.max_prompt)
    rows = []
    for path in sorted(args.runs):
        summary, answers = read_run(path)
        if summary["metrics"]["n"] < args.min_n:
            continue
        for r in rates:
            if not summary["label"].startswith(r.model):
                continue
            rows.append({
                "run": path.name,
                "model": r.model,
                "board": r.board,
                "split": summary["split"],
                "n": summary["metrics"]["n"],
                "accuracy_kaggle": summary["metrics"]["accuracy"],
                "macro_f1_kaggle": summary["metrics"]["macro_f1"],
                "mean_prompt_tokens": round(statistics.fmean(a["prompt_tokens"] for a in answers), 1),
                "mean_completion_tokens": round(
                    statistics.fmean(a["completion_tokens"] for a in answers), 2
                ),
                **estimate(r, answers),
            })

    stamp = datetime.now(UTC)
    out = {
        "stage": "baseline",
        "kind": "board-cost-estimate",
        "time": stamp.isoformat(timespec="seconds"),
        "settings": {
            "export": args.export,
            "export_source": f"agentic-edge commit {SOURCE_COMMIT}, MMLU-Pro s1-s3, thinking off",
            "max_prompt_tokens_in_fit": args.max_prompt,
            "min_n": args.min_n,
            "method": "latency = a + b*prompt_tokens + completion_tokens*decode_ms; "
            "energy = median request power * latency. Full prompt tokens (no prefix-cache "
            "credit), so an upper bound. Board DC draw, not wall power.",
        },
        "rates": [asdict(r) for r in rates],
        "estimates": rows,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"board-cost-estimate-{stamp:%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    for r in rows:
        print(
            f"{r['model']} {r['board']:6} {r['split']:7} acc {r['accuracy_kaggle']:.3f} "
            f"median {r['median_latency_s']:5.2f} s  {r['mean_j_per_answer']:6.2f} J/answer"
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
