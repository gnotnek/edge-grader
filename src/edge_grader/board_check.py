"""Check board runs against their Kaggle runs and the agentic-edge cost estimate.

Stage 1 estimated board cost from agentic-edge's MMLU-Pro request rates (cost.py) and
assumed board accuracy equals Kaggle's. A measured run on a board (scripts/board_grade.sh)
tests both; phase 0 extends that to GSM8K/GSM-Symbolic and news. Per run folder in
results/board/ this reads

    baseline-*.json | gsm-*.json | news-*.json, raw/*.jsonl   accuracy and each answer
    mmlupro-*.json, raw/*.jsonl    MMLU-Pro: one per subset (s1-s3), pooled into one run
    news-*.judged-<judge>.json, raw/*.judged-*.jsonl          news only: the Kaggle judge's grades
    board-state.txt                CPU governor and clock range at the start of the run
    server.log                     llama-server's own per-request prefill and decode timings
    telemetry-*/telemetry.csv      1 Hz board power from agentic-edge's run_measured.sh
    telemetry-*/meta.json           the llama-server command line actually serving the run

Energy is integrated over the grading window only (first request start to last request
end), so server start-up and the dataset download are left out. Power is board DC draw,
not wall power. Accuracy is compared answer by answer: agreement and an exact McNemar test
on which answers each run got right, against the Kaggle run of the same split and model,
and between boards. GSM and news runs are paired with Kaggle on the question ids both
answered, so the Pi's trimmed GSM-Symbolic set compares on its 100 ids.

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
SUMMARY_GLOBS = ("baseline-*.json", "think-*.json", "rag-*.json", "gsm-*.json", "news-*.json",
                 "mmlupro-*.json", "asap2-*.json")
JUDGE = "qwen3-14b-q4_k_m"  # the news judge's file tag (docs/results/topics.md)
CPU_STATE = re.compile(r"^cpu0: governor (\S+) min (\d+) max (\d+)")
STATE_LINE = re.compile(r"^(variant|engine): (.*)$")

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
    """Two runs' outcomes, paired on the question ids both have."""
    ids = sorted(a.keys() & b.keys())

    def acc(x: dict[str, dict]) -> float | None:
        return round(sum(x[i]["right"] for i in ids) / len(ids), 4) if ids else None

    return {
        "n": len(ids),
        "same_grade": sum(a[i]["answer"] == b[i]["answer"] for i in ids),
        "first_accuracy": acc(a),
        "second_accuracy": acc(b),
        "mcnemar": mcnemar([a[i]["right"] for i in ids], [b[i]["right"] for i in ids]),
    }


def read_answers(summary_path: Path) -> dict[str, dict]:
    raw = summary_path.parent / "raw" / summary_path.with_suffix(".jsonl").name
    rows = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
    return {r["id"]: r for r in rows}


def find_summaries(run_dir: Path) -> list[Path]:
    """The runner's summaries in a run folder (a judge's file is not one): exactly one, or
    one per subset for MMLU-Pro."""
    found = sorted(p for g in SUMMARY_GLOBS for p in run_dir.glob(g) if ".judged-" not in p.name)
    if len(found) != 1 and not (found and all(p.name.startswith("mmlupro-") for p in found)):
        raise SystemExit(f"{run_dir}: expected one run summary, found {[p.name for p in found]}")
    return found


def find_summary(run_dir: Path) -> Path:
    found = find_summaries(run_dir)
    if len(found) != 1:
        raise SystemExit(f"{run_dir}: expected one run summary, found {[p.name for p in found]}")
    return found[0]


def benchmark(summary: dict) -> str:
    return summary.get("benchmark", "scientsbank")  # grading summaries have no "benchmark" key


def split_of(summary: dict) -> str:
    return {"gsm": summary.get("set"), "news": "test", "mmlupro": summary.get("subset")}.get(
        benchmark(summary), summary.get("split"))


def pooled_metrics(summaries: list[dict], answers: list[dict]) -> dict:
    """One run's metrics from its per-subset summaries (MMLU-Pro); its unparsed = no answer."""
    ms = [s["metrics"] for s in summaries]
    n = sum(m["n"] for m in ms)
    return {"n": n, "accuracy": round(sum(m["accuracy"] * m["n"] for m in ms) / n, 4),
            "unparsed": sum(m["no_answer"] for m in ms), "errors": sum(m["errors"] for m in ms),
            "median_latency_s": statistics.median(a["latency_s"] for a in answers),
            "wall_s": sum(m["wall_s"] for m in ms)}


def run_outcomes(summary_paths: list[Path], judge: str = JUDGE) -> dict[str, dict] | None:
    """outcomes() over every summary of a run (MMLU-Pro's subsets); None if any is unjudged."""
    parts = [outcomes(p, judge) for p in summary_paths]
    return None if None in parts else {k: v for part in parts for k, v in part.items()}


def outcomes(summary_path: Path, judge: str = JUDGE) -> dict[str, dict] | None:
    """Per question, the answer given and whether it was right; None for an unjudged news run.

    Grading: the label. GSM: the flexible-extract number. News: the judge's grade."""
    bench = benchmark(json.loads(summary_path.read_text()))
    if bench == "news":
        judged = summary_path.parent / "raw" / f"{summary_path.stem}.judged-{judge}.jsonl"
        if not judged.exists():
            return None
        rows = [json.loads(line) for line in judged.read_text().splitlines() if line.strip()]
        return {r["id"]: {"answer": r["grade"], "right": r["grade"] == "correct"} for r in rows}
    rows = read_answers(summary_path).values()
    if bench == "gsm":
        return {r["id"]: {"answer": r["flexible"], "right": bool(r["correct"])} for r in rows}
    return {r["id"]: {"answer": r["pred"], "right": r["pred"] == r["gold"]} for r in rows}


def read_board_state(path: Path) -> dict | None:
    """CPU governor and frequency range (and the variant, engine) from board-state.txt;
    None for runs made before it existed."""
    if not path.exists():
        return None
    out: dict = {}
    for line in path.read_text().splitlines():
        m = CPU_STATE.match(line)
        if m and "governor" not in out:
            gov, lo, hi = m.groups()
            out |= {"governor": gov, "cpu_min_khz": int(lo), "cpu_max_khz": int(hi),
                    "cpu_clock_locked": lo == hi}
        s = STATE_LINE.match(line)
        if s:
            out[s.group(1)] = s.group(2)
    return out


def threads(server_args: str) -> int | None:
    m = re.search(r"(?:^| )-t (\d+)", server_args)
    return int(m.group(1)) if m else None


def read_run(run_dir: Path, judge: str = JUDGE) -> dict:
    summary_paths = find_summaries(run_dir)
    summaries = [json.loads(p.read_text()) for p in summary_paths]
    summary, summary_path = summaries[0], summary_paths[0]
    answers = [a for p in summary_paths for a in read_answers(p).values()]
    bench = benchmark(summary)
    m = pooled_metrics(summaries, answers) if bench == "mmlupro" else summary["metrics"]
    split = "+".join(split_of(s) for s in summaries)
    accuracy = m.get("accuracy")
    if bench == "news":  # the runner doesn't score news; the Kaggle judge does
        judged = summary_path.with_name(f"{summary_path.stem}.judged-{judge}.json")
        accuracy = json.loads(judged.read_text())["metrics"]["accuracy"] if judged.exists() else None
    model, board_label = summary["label"].split("-", 1)
    # run_measured.sh's folder: telemetry-<stamp>/ now, edge-grader-*/ in the stage-1 runs
    tele = next(d for d in sorted(run_dir.iterdir())
                if d.is_dir() and d.name.startswith(("telemetry-", "edge-grader-")))
    meta = json.loads((tele / "meta.json").read_text())
    power = json.loads((tele / "summary.json").read_text())["power"]

    reqs = parse_server_log((run_dir / "server.log").read_text())
    if reqs:
        start = min(r["end"] - (r["prompt_ms"] + r["gen_ms"]) / 1000 for r in reqs)
        end = max(r["end"] for r in reqs)
        window_from = "server.log"
    else:  # little-gemma's shim logs no per-request timings: the runner's own loop window
        ends = [datetime.fromisoformat(s["time"]).timestamp() for s in summaries]
        start = min(e - s["metrics"]["wall_s"] for e, s in zip(ends, summaries))
        end = max(ends)
        window_from = "summary"
    header, *lines = (tele / "telemetry.csv").read_text().splitlines()
    t_col, w_col = (header.split(",").index(c) for c in ("ts_epoch", "power_w"))
    samples = [(float(c[t_col]), float(c[w_col])) for c in (ln.split(",") for ln in lines)]
    joules = window_energy(samples, start, end)
    seconds = end - start
    n = m["n"]

    prefill_x = [r["prompt_tokens"] for r in reqs]
    prefill_y = [r["prompt_ms"] for r in reqs]
    slope = intercept = None
    if len(set(prefill_x)) >= 2:  # a fit needs two different prompt sizes (a 1-question smoke run has one)
        slope, intercept = statistics.linear_regression(prefill_x, prefill_y)
    decode = statistics.median(r["gen_ms"] / r["gen_tokens"] for r in reqs if r["gen_tokens"]) \
        if any(r["gen_tokens"] for r in reqs) else None
    sent = [a["prompt_tokens"] for a in answers if a.get("prompt_tokens") is not None]
    return {
        "run": run_dir.name,
        "model": model,
        "board": BOARDS[board_label],
        "benchmark": bench,
        "split": split,
        "thinking": summary.get("stage") == "think",
        "stage": summary.get("stage", "baseline"),
        "instances": summary.get("generation", {}).get("instances"),
        "threads": threads(meta.get("server_args_after", "")),
        "server_args": meta.get("server_args_after", ""),
        "n": n,
        "accuracy": accuracy,
        "macro_f1": m.get("macro_f1"),
        "unparsed": m.get("unparsed"),
        "errors": m["errors"],
        "requests_in_log": len(reqs),
        "window_from": window_from,
        "median_latency_s": round(m["median_latency_s"], 2),
        "grading_window_s": round(seconds, 1),
        "mean_w": round(joules / seconds, 2),
        "idle_w": power["idle_w"],
        "j_per_answer": round(joules / n, 2),
        "j_per_answer_above_idle": round((joules - power["idle_w"] * seconds) / n, 2),
        "wh_per_1000_answers": round(joules / n * 1000 / 3600, 2),
        "mean_prompt_tokens_processed": round(statistics.fmean(prefill_x), 1) if reqs else None,
        "mean_prompt_tokens_sent": round(statistics.fmean(sent), 1) if sent else None,
        "prefill_fixed_ms": None if intercept is None else round(intercept, 1),
        "prefill_ms_per_token": None if slope is None else round(slope, 3),
        "decode_ms_per_token": None if decode is None else round(decode, 1),
        "board_state": read_board_state(run_dir / "board-state.txt"),
        "_summaries": summary_paths,
    }


def setting(run: dict) -> tuple:
    """What a Pi run and a Jetson run must share to be paired: the task and how it was served."""
    variant = (run["board_state"] or {}).get("variant", "base")
    return (run["benchmark"], run["model"], run["split"], run["stage"], run["thinking"], variant)


def latest_estimate(directory: str) -> Path:
    return max(Path(directory).glob("board-cost-estimate-*.json"))


def kaggle_run(directory: str, bench: str, model: str, split: str, n: int,
               thinking: bool = False, stage: str = "baseline") -> Path | None:
    """The Kaggle run a board run is checked against. Grading: same split and n (as in stage 1),
    a thinking-on board run against the Kaggle thinking-on run, a RAG one against Kaggle's RAG-en.
    GSM and news: the latest Kaggle run of the set with the same stage (baseline or RAG-en);
    pairing is on shared ids."""
    d = Path(directory)
    if bench == "scientsbank":
        prefix = "think" if thinking else stage
        tag = "-en-k3" if prefix == "rag" else ""
        # [^-]+ is the hardware ("t4"): adapter runs (…-kaggle-t4-ft-…) aren't untouched baselines
        same = re.compile(rf"{prefix}-{model}-kaggle-[^-]+{tag}-{split}-")
        for path in sorted(d.glob(f"{prefix}-{model}-kaggle-*{tag}-{split}-*.json")):
            if same.match(path.name) and json.loads(path.read_text())["metrics"]["n"] == n:
                return path
        return None
    rag = stage == "rag"
    pattern = (f"gsm-{'rag-' if rag else ''}{split}-{model}-kaggle-*.json" if bench == "gsm"
               else f"news-{'rag' if rag else 'baseline'}-{model}-kaggle-*.json")
    found = sorted(p for p in d.glob(pattern) if ".judged-" not in p.name)
    return found[-1] if found else None


def kaggle_runs(directory: str, run: dict) -> list[Path]:
    """kaggle_run() as a list; MMLU-Pro: the latest Kaggle run of each of the board run's
    subsets with the same stage, or none if a subset has no Kaggle run."""
    if run["benchmark"] != "mmlupro":
        k = kaggle_run(directory, run["benchmark"], run["model"], run["split"], run["n"],
                       thinking=run["thinking"], stage=run["stage"])
        return [k] if k else []
    kind = "rag" if run["stage"] == "rag" else "baseline"
    found = [sorted(Path(directory).glob(f"mmlupro-{kind}-{run['model']}-kaggle-*-{s}-*.json"))
             for s in run["split"].split("+")]
    return [f[-1] for f in found] if all(found) else []


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="board run folders (results/board/<run>/)")
    ap.add_argument("--estimate", type=Path, help="board-cost-estimate JSON (default: latest)")
    ap.add_argument("--kaggle", default=KAGGLE, help="directory of the Kaggle runs")
    ap.add_argument("--out", default=ESTIMATES, help="directory for the summary JSON")
    ap.add_argument("--judge", default=JUDGE, help="news judge file tag")
    args = ap.parse_args(argv)

    est_path = args.estimate or latest_estimate(ESTIMATES)
    est = json.loads(est_path.read_text())["estimates"]
    runs = [read_run(d, args.judge) for d in sorted(args.runs)]

    for r in runs:
        e = next(
            (x for x in est
             if r["benchmark"] == "scientsbank"
             and (x["model"], x["board"], x["split"]) == (r["model"], r["board"], r["split"])),
            None,
        )
        r["estimate"] = e and {
            "median_latency_s": e["median_latency_s"],
            "j_per_answer": e["mean_j_per_answer"],
            "measured_over_estimate_latency": round(r["median_latency_s"] / e["median_latency_s"], 2),
            "measured_over_estimate_energy": round(r["j_per_answer"] / e["mean_j_per_answer"], 2),
        }
        ks = kaggle_runs(args.kaggle, r)
        k_name = (ks[0].name if len(ks) == 1 else [p.name for p in ks]) if ks else None
        mine, theirs = run_outcomes(r["_summaries"], args.judge), ks and run_outcomes(ks, args.judge)
        if ks and (mine is None or theirs is None):
            r["vs_kaggle"] = {"kaggle_run": k_name, "pending": "judge the news run first"}
        elif ks:
            c = compare(mine, theirs)
            r["vs_kaggle"] = {"kaggle_run": k_name, "kaggle_accuracy": c.pop("second_accuracy"),
                              "board_accuracy_paired": c.pop("first_accuracy"), **c}
        else:
            r["vs_kaggle"] = None

    pairs = []
    for a in runs:
        for b in runs:
            same = setting(a) == setting(b)
            if not (same and (a["board"], b["board"]) == ("pi", "jetson")):
                continue
            oa, ob = run_outcomes(a["_summaries"], args.judge), run_outcomes(b["_summaries"], args.judge)
            pairs.append({
                "first": a["run"],
                "second": b["run"],
                "latency_ratio": round(a["median_latency_s"] / b["median_latency_s"], 1),
                "energy_ratio": round(a["j_per_answer"] / b["j_per_answer"], 1),
                **(compare(oa, ob) if oa is not None and ob is not None
                   else {"pending": "judge the news runs first"}),
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
            "Accuracy: same-grade count and exact McNemar on correctness, same answer ids. "
            "GSM/news: paired with Kaggle and between boards on the question ids both runs "
            "have; news correctness is the judge's grade.",
        },
        "runs": [{k: v for k, v in r.items() if not k.startswith("_")} for r in runs],
        "board_pairs": pairs,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"board-check-{stamp:%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    def num(x, f=".3f"):
        return "n/a" if x is None else format(x, f)

    for r in runs:
        e, k = r["estimate"] or {}, r["vs_kaggle"] or {}
        print(
            f"{r['run']:46} t={r['threads']} acc {num(r['accuracy'])} (kaggle "
            f"{num(k.get('kaggle_accuracy'))} on {k.get('n')}, same {k.get('same_grade')}"
            f", p {k.get('mcnemar', {}).get('p')}{', ' + k['pending'] if 'pending' in k else ''})"
            f"  median {r['median_latency_s']:.2f} s (est {e.get('median_latency_s')})  "
            f"{r['j_per_answer']:.1f} J/question (est {e.get('j_per_answer')})"
            f"{'  CPU CLOCK LOCKED' if (r['board_state'] or {}).get('cpu_clock_locked') else ''}"
        )
    for p in pairs:
        print(
            f"{p['first']} vs {p['second']}: same {p.get('same_grade')}/{p.get('n')}, "
            f"McNemar p {p.get('mcnemar', {}).get('p')}, Pi/Jetson latency {p['latency_ratio']}x, "
            f"energy {p['energy_ratio']}x"
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
