"""Thinking on with every E2B adapter on the Jetson (supervisor, 2026-11-16): accuracy beside device cost.

    uv run edge-grader-board-check results/board/*think*-jetson-test_u[qd]-*/
    uv run python -m edge_grader.compare_thinking

Each version graded with thinking on (budget 320) is paired with the same version thinking off (the
2026-11-14 to 11-16 runs). Thinking-on runs: test_uq 2026-11-16/17 at GPU 1,020 MHz; test_ud 2026-11-17
to 11-20 at nvpmodel 25W (GPU 918 MHz), except 5-way (1,020 MHz). Thinking-off runs: GPU 612 MHz. So
accuracy compares across all of them, cost only roughly; each row records its GPU limit.

Test family, 22 McNemar tests on 3-way labels with Holm, chosen after the test_uq runs: per split,
each version thinking on vs off (6), and each adapter with thinking vs untouched E2B with thinking (5).
"""

import json
import re
import statistics
from datetime import UTC, datetime
from pathlib import Path

from .board_check import find_summary, read_answers
from .compare import holm, paired
from .compare_jetson import board_check_runs, prefix

SPLITS = ("test_uq", "test_ud")
# thinking-off run prefix -> (name, thinking-on run prefix)
VERSIONS = {
    "e2b": ("E2B untouched", "think-e2b"),
    "ft-e2b": ("Oct 28 adapter", "ft-think-e2b"),
    "ft-soup-e2b": ("Soup (s0-s4)", "ft-soup-think-e2b"),
    "ft-lf-e2b": ("Label-first", "ft-lf-think-e2b"),
    "ft-rf-e2b": ("Reason-first", "ft-rf-think-e2b"),
    "ft-5w-s0-e2b": ("5-way s0", "ft-5w-s0-think-e2b"),
}
OFF_SINCE, ON_SINCE = "20261114", "20261116"  # older runs of these prefixes are other clock states
GPU_MAX = re.compile(r"17000000\.gpu: min \d+ max (\d+)")


def compute(board: Path = Path("results/board"), estimates: Path = Path("results/estimates")) -> dict:
    """The rows and paired tests, from the committed runs; writes nothing."""
    cost = {}
    for r in board_check_runs(estimates).values():
        stamp = r["run"].rsplit("-", 2)[-2]
        if r["board"] == "jetson" and r["split"] in SPLITS:
            v = prefix(r["run"], r["split"])
            on = v in {on for _, on in VERSIONS.values()}
            if (v in VERSIONS or on) and stamp >= (ON_SINCE if on else OFF_SINCE):
                cost[v, r["split"]] = r  # sorted by run folder: the latest run of a version wins
    wanted = [(v, s) for off, (_, on) in VERSIONS.items() for v in (off, on) for s in SPLITS]
    if missing := [k for k in wanted if k not in cost]:
        raise SystemExit(f"{estimates}: no board-check entry for {missing}")

    rows, answers = [], {}
    for v, split in wanted:
        r = cost[v, split]
        summary_path = find_summary(board / r["run"])
        m = json.loads(summary_path.read_text())["metrics"]
        answers[v, split] = read_answers(summary_path)
        got = list(answers[v, split].values())
        state = (board / r["run"] / "board-state.txt").read_text()
        thinking = v not in VERSIONS
        rows.append({
            "version": v, "name": VERSIONS[v][0] if not thinking else
            next(n for n, on in VERSIONS.values() if on == v), "thinking": thinking, "split": split,
            "run": r["run"], "n": m["n"], "accuracy": m["accuracy"], "ci95": m["ci95"]["accuracy"],
            "macro_f1": m["macro_f1"], "unparsed": m["unparsed"],
            "thought_blocks": sum("<|channel>thought" in a["raw"] for a in got),
            "cut_by_length": sum(a.get("finish") == "length" for a in got),
            "median_completion_tokens": statistics.median(a["completion_tokens"] for a in got),
            "median_latency_s": r["median_latency_s"], "j_per_answer": r["j_per_answer"],
            "gpu_max_mhz": int(GPU_MAX.search(state).group(1)) // 10**6})

    tests = [(off, on, s) for off, (_, on) in VERSIONS.items() for s in SPLITS] + \
            [("think-e2b", on, s) for off, (_, on) in VERSIONS.items() if off != "e2b" for s in SPLITS]
    tests = [{"first": a, "second": b, "split": s, **paired(answers[a, s], answers[b, s], two_way=False)}
             for a, b, s in tests]
    for t, ph in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = ph
    return {"rows": rows, "tests": tests}


def main(board: Path = Path("results/board"), estimates: Path = Path("results/estimates")) -> None:
    c = compute(board, estimates)
    rows, tests = c["rows"], c["tests"]
    out = estimates / f"compare-thinking-scientsbank-{datetime.now(UTC):%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({
        "kind": "compare-thinking", "board_checks": str(estimates / "board-check-*.json"),
        "family": "22 tests chosen after the test_uq runs: per split, each version thinking on vs off, and "
        "each adapter with thinking vs untouched E2B with thinking; McNemar on 3-way labels, Holm over the 22",
        "rows": rows, "tests": tests, "time": datetime.now(UTC).isoformat(timespec="seconds")}, indent=2) + "\n")

    by = {(r["version"], r["split"]): r for r in rows}
    print("| Version | Split | Off | On | Thought blocks | Cut by length | On: median s | On: J | Off: J |"
          "\n|---|---|---|---|---|---|---|---|---|")
    for off, (name, on) in VERSIONS.items():
        for s in SPLITS:
            a, b = by[off, s], by[on, s]
            print(f"| {name} | {s} | {a['accuracy']:.1%} | {b['accuracy']:.1%} | {b['thought_blocks']}/{b['n']} | "
                  f"{b['cut_by_length']} | {b['median_latency_s']:.2f} | {b['j_per_answer']:.1f} | "
                  f"{a['j_per_answer']:.1f} |")
    print("\n| Comparison | Split | Difference | first-only / second-only right | p | Holm p |\n|---|---|---|---|---|---|")
    for t in tests:
        print(f"| {t['second']} vs {t['first']} | {t['split']} | {t['diff']:+.1%} | "
              f"{t['only_first_right']} / {t['only_second_right']} | {t['p']:.2g} | {t['p_holm']:.2g} |")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
