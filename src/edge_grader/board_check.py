"""Check board grading runs against the Kaggle accuracy and the agentic-edge cost estimate.

Stage 1 estimated board cost from agentic-edge's MMLU-Pro request rates (cost.py) and
assumed board accuracy equals Kaggle's. A measured run on a board (scripts/board_grade.sh)
tests both. Per run folder in results/board/ this reads

    baseline-*.json, raw/*.jsonl   accuracy and each answer's grade
    server.log                     llama-server's own per-request prefill and decode timings
    edge-grader-*/telemetry.csv    1 Hz board power from agentic-edge's run_measured.sh
    edge-grader-*/meta.json        the llama-server command line actually serving the run

Energy is integrated over the grading window only (first request start to last request
end), so server start-up and the dataset download are left out. Power is board DC draw,
not wall power. Accuracy is compared answer by answer: agreement and an exact McNemar test
on which answers each run got right, against the Kaggle run of the same split and model,
and between boards.

    uv run edge-grader-board-check results/board/*/
"""

import argparse
import itertools
import json
import re
import statistics
from datetime import UTC, datetime
from pathlib import Path

from scipy.stats import binomtest

ESTIMATES = "results/estimates"
KAGGLE = "results/kaggle"
BOARDS = {"pi5": "pi", "jetson": "jetson"}  # run label -> board name in the estimate

TIMING = re.compile(
    r"^(\S+) .*print_timing: id\s+\d+ \| task (\d+) \|\s+(prompt eval|eval) time =\s+"
    r"([\d.]+) ms /\s+(\d+) tokens"
)


def parse_server_log(text: str) -> list[dict]:
    """One dict per request: prompt and generated tokens, their ms, and the end time."""
    reqs: dict[str, dict] = {}
    for line in text.splitlines():
        m = TIMING.match(line)
        if not m:
            continue
        ts, task, kind, ms, n = m.groups()
        r = reqs.setdefault(task, {"task": int(task)})
        key = "prompt" if kind == "prompt eval" else "gen"
        r[f"{key}_ms"], r[f"{key}_tokens"] = float(ms), int(n)
        r["end"] = datetime.fromisoformat(ts).timestamp()
    done = [r for r in reqs.values() if "prompt_ms" in r and "gen_ms" in r]
    return sorted(done, key=lambda r: r["task"])


def window_energy(samples: list[tuple[float, float]], start: float, end: float) -> float:
    """Joules between start and end, trapezoidal over (epoch, watts) samples."""
    pts = [(t, w) for t, w in samples if start <= t <= end]
    return sum((t1 - t0) * (w0 + w1) / 2 for (t0, w0), (t1, w1) in itertools.pairwise(pts))


def mcnemar(right_a: list[bool], right_b: list[bool]) -> dict:
    only_a = sum(a and not b for a, b in zip(right_a, right_b))
    only_b = sum(b and not a for a, b in zip(right_a, right_b))
    p = binomtest(only_a, only_a + only_b).pvalue if only_a + only_b else 1.0
    return {"only_first_right": only_a, "only_second_right": only_b, "p": round(p, 3)}


def compare(a: dict[str, dict], b: dict[str, dict]) -> dict:
    ids = sorted(a.keys() & b.keys())
    return {
        "n": len(ids),
        "same_grade": sum(a[i]["pred"] == b[i]["pred"] for i in ids),
        "mcnemar": mcnemar(
            [a[i]["pred"] == a[i]["gold"] for i in ids],
            [b[i]["pred"] == b[i]["gold"] for i in ids],
        ),
    }


def read_answers(summary_path: Path) -> dict[str, dict]:
    raw = summary_path.parent / "raw" / summary_path.with_suffix(".jsonl").name
    rows = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
    return {r["id"]: r for r in rows}


def threads(server_args: str) -> int | None:
    m = re.search(r"(?:^| )-t (\d+)", server_args)
    return int(m.group(1)) if m else None


def read_run(run_dir: Path) -> dict:
    summary_path = next(run_dir.glob("baseline-*.json"))
    summary = json.loads(summary_path.read_text())
    model, board_label = summary["label"].split("-", 1)
    tele = next(run_dir.glob("edge-grader-*/"))
    meta = json.loads((tele / "meta.json").read_text())
    power = json.loads((tele / "summary.json").read_text())["power"]

    reqs = parse_server_log((run_dir / "server.log").read_text())
    start = min(r["end"] - (r["prompt_ms"] + r["gen_ms"]) / 1000 for r in reqs)
    end = max(r["end"] for r in reqs)
    header, *lines = (tele / "telemetry.csv").read_text().splitlines()
    t_col, w_col = (header.split(",").index(c) for c in ("ts_epoch", "power_w"))
    samples = [(float(c[t_col]), float(c[w_col])) for c in (ln.split(",") for ln in lines)]
    joules = window_energy(samples, start, end)
    seconds = end - start
    n = summary["metrics"]["n"]

    prefill_x = [r["prompt_tokens"] for r in reqs]
    prefill_y = [r["prompt_ms"] for r in reqs]
    slope, intercept = statistics.linear_regression(prefill_x, prefill_y)
    m = summary["metrics"]
    return {
        "run": run_dir.name,
        "model": model,
        "board": BOARDS[board_label],
        "split": summary["split"],
        "threads": threads(meta.get("server_args_after", "")),
        "server_args": meta.get("server_args_after", ""),
        "n": n,
        "accuracy": m["accuracy"],
        "macro_f1": m["macro_f1"],
        "unparsed": m["unparsed"],
        "errors": m["errors"],
        "requests_in_log": len(reqs),
        "median_latency_s": round(m["median_latency_s"], 2),
        "grading_window_s": round(seconds, 1),
        "mean_w": round(joules / seconds, 2),
        "idle_w": power["idle_w"],
        "j_per_answer": round(joules / n, 2),
        "j_per_answer_above_idle": round((joules - power["idle_w"] * seconds) / n, 2),
        "wh_per_1000_answers": round(joules / n * 1000 / 3600, 2),
        "mean_prompt_tokens_processed": round(statistics.fmean(prefill_x), 1),
        "mean_prompt_tokens_sent": round(
            statistics.fmean(a["prompt_tokens"] for a in read_answers(summary_path).values()), 1
        ),
        "prefill_fixed_ms": round(intercept, 1),
        "prefill_ms_per_token": round(slope, 3),
        "decode_ms_per_token": round(
            statistics.median(r["gen_ms"] / r["gen_tokens"] for r in reqs if r["gen_tokens"]), 1
        ),
        "_summary": summary_path,
    }


def latest_estimate(directory: str) -> Path:
    return max(Path(directory).glob("board-cost-estimate-*.json"))


def kaggle_run(directory: str, model: str, split: str, n: int) -> Path | None:
    for path in sorted(Path(directory).glob(f"baseline-{model}-kaggle-*-{split}-*.json")):
        if json.loads(path.read_text())["metrics"]["n"] == n:
            return path
    return None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="board run folders (results/board/<run>/)")
    ap.add_argument("--estimate", type=Path, help="board-cost-estimate JSON (default: latest)")
    ap.add_argument("--kaggle", default=KAGGLE, help="directory of the Kaggle runs")
    ap.add_argument("--out", default=ESTIMATES, help="directory for the summary JSON")
    args = ap.parse_args(argv)

    est_path = args.estimate or latest_estimate(ESTIMATES)
    est = json.loads(est_path.read_text())["estimates"]
    runs = [read_run(d) for d in sorted(args.runs)]

    for r in runs:
        e = next(
            (x for x in est
             if (x["model"], x["board"], x["split"]) == (r["model"], r["board"], r["split"])),
            None,
        )
        r["estimate"] = e and {
            "median_latency_s": e["median_latency_s"],
            "j_per_answer": e["mean_j_per_answer"],
            "measured_over_estimate_latency": round(r["median_latency_s"] / e["median_latency_s"], 2),
            "measured_over_estimate_energy": round(r["j_per_answer"] / e["mean_j_per_answer"], 2),
        }
        k = kaggle_run(args.kaggle, r["model"], r["split"], r["n"])
        r["vs_kaggle"] = k and {
            "kaggle_run": k.name,
            "kaggle_accuracy": json.loads(k.read_text())["metrics"]["accuracy"],
            **compare(read_answers(r["_summary"]), read_answers(k)),
        }

    pairs = []
    for a in runs:
        for b in runs:
            if (a["model"], a["split"]) == (b["model"], b["split"]) and (a["board"], b["board"]) == (
                "pi", "jetson"
            ):
                pairs.append({
                    "first": a["run"],
                    "second": b["run"],
                    "latency_ratio": round(a["median_latency_s"] / b["median_latency_s"], 1),
                    "energy_ratio": round(a["j_per_answer"] / b["j_per_answer"], 1),
                    **compare(read_answers(a["_summary"]), read_answers(b["_summary"])),
                })

    stamp = datetime.now(UTC)
    out = {
        "stage": "baseline",
        "kind": "board-check",
        "time": stamp.isoformat(timespec="seconds"),
        "settings": {
            "estimate": str(est_path),
            "method": "energy = trapezoid of telemetry power_w over the grading window (first "
            "request start to last request end, from server.log); board DC draw, not wall "
            "power. Prefill fit: prompt_ms = a + b*prompt tokens processed, every request. "
            "Accuracy: same-grade count and exact McNemar on correctness, same answer ids.",
        },
        "runs": [{k: v for k, v in r.items() if not k.startswith("_")} for r in runs],
        "board_pairs": pairs,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"board-check-{stamp:%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    for r in runs:
        e, k = r["estimate"] or {}, r["vs_kaggle"] or {}
        print(
            f"{r['run']:38} t={r['threads']} acc {r['accuracy']:.3f} (kaggle "
            f"{k.get('kaggle_accuracy', float('nan')):.3f}, same {k.get('same_grade')}/{k.get('n')}"
            f", p {k.get('mcnemar', {}).get('p')})  median {r['median_latency_s']:.2f} s "
            f"(est {e.get('median_latency_s')})  {r['j_per_answer']:.1f} J/answer "
            f"(est {e.get('j_per_answer')})"
        )
    for p in pairs:
        print(
            f"{p['first']} vs {p['second']}: same {p['same_grade']}/{p['n']}, "
            f"McNemar p {p['mcnemar']['p']}, Pi/Jetson latency {p['latency_ratio']}x, "
            f"energy {p['energy_ratio']}x"
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
