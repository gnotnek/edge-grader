"""Every SciEntsBank version on the Jetson (2026-11-14 to 11-16): accuracy beside device cost.

    uv run edge-grader-board-check results/board/*-jetson-test_u[qd]-2026111[456]-*/
    uv run python -m edge_grader.compare_jetson

Cost comes from the latest board-check JSON in results/estimates/ (energy over the grading window
only, board DC power), accuracy from each run's own summary. All runs: one board, one clock state
(GPU 612 MHz, memory 2,133 MHz), llama.cpp a894dae, test_uq and test_ud (the headline splits for
fine-tuned versions, decision 2026-11-13).

The family is the three-grader question (RQ8) plus the E4B adapter, 8 tests chosen 2026-11-16
after the runs: per split, E2B tweaked (the Oct 28 adapter) vs E2B untouched and vs E4B untouched,
E4B untouched vs E2B untouched, and the E4B adapter vs E4B untouched. Paired McNemar on 3-way labels,
Holm over the 8; 2-way beside it with its own Holm. Every other version is described, not tested
here: its accuracy was tested on the DGX Spark (docs/results/finetuning.md, examples.md), and the
Jetson grades the same answers the same way.
"""

import json
import statistics
from datetime import UTC, datetime
from pathlib import Path

from .board_check import find_summary, read_answers
from .compare import holm, paired

SPLITS = ("test_uq", "test_ud")
# run-folder prefix -> (name, group); the prefix is the run folder minus "-jetson-<split>-<stamp>"
VERSIONS = {
    "e2b": ("E2B untouched", "three graders"),
    "ft-e2b": ("E2B tweaked (Oct 28 adapter)", "three graders"),
    "e4b": ("E4B untouched", "three graders"),
    "ft-e4b": ("E4B Oct 28 adapter", "E4B adapter"),
    **{f"ft-s{n}-e2b": (f"E2B seed s{n}", "E2B seeds") for n in range(5)},
    "ft-soup-e2b": ("E2B soup (s0-s4)", "E2B soup and 5-way"),
    "ft-5w-s0-e2b": ("E2B 5-way s0", "E2B soup and 5-way"),
    "ft-lf-e2b": ("E2B label-first", "E2B reply format"),
    "ft-rf-e2b": ("E2B reason-first", "E2B reply format"),
    "ex4-e2b": ("E2B + kNN examples", "E2B graded examples"),
    "ex4-ft-e2b": ("E2B adapter + kNN examples", "E2B graded examples"),
    "rand4-e2b": ("E2B + random examples", "E2B graded examples"),
}
TESTS = (("e2b", "ft-e2b"), ("e4b", "ft-e2b"), ("e2b", "e4b"), ("e4b", "ft-e4b"))  # (first, second)


def prefix(run: str, split: str) -> str:
    return run.split(f"-jetson-{split}-")[0]


def main(board: Path = Path("results/board"), estimates: Path = Path("results/estimates")) -> None:
    check_file = max(estimates.glob("board-check-*.json"))
    cost = {}
    for r in json.loads(check_file.read_text())["runs"]:
        if r["board"] == "jetson" and r["split"] in SPLITS and prefix(r["run"], r["split"]) in VERSIONS:
            cost[prefix(r["run"], r["split"]), r["split"]] = r  # later runs of a version replace earlier
    missing = [(v, s) for v in VERSIONS for s in SPLITS if (v, s) not in cost]
    if missing:
        raise SystemExit(f"{check_file}: no board-check entry for {missing}")

    rows, answers = [], {}
    for (v, split), r in cost.items():
        summary_path = find_summary(board / r["run"])
        m = json.loads(summary_path.read_text())["metrics"]
        answers[v, split] = read_answers(summary_path)
        base = cost["e2b", split]
        rows.append({
            "version": v, "name": VERSIONS[v][0], "group": VERSIONS[v][1], "split": split, "run": r["run"],
            "n": m["n"], "accuracy": m["accuracy"], "ci95": m["ci95"]["accuracy"], "macro_f1": m["macro_f1"],
            "two_way_accuracy": m["two_way"]["accuracy"], "unparsed": m["unparsed"],
            "median_latency_s": r["median_latency_s"], "j_per_answer": r["j_per_answer"],
            "energy_vs_e2b_untouched": round(r["j_per_answer"] / base["j_per_answer"], 2),
            "mean_w": r["mean_w"], "decode_tok_s": round(1000 / r["decode_ms_per_token"], 1),
            "mean_prompt_tokens": r["mean_prompt_tokens_sent"],
            "median_completion_tokens": m.get("median_completion_tokens")})
    rows.sort(key=lambda x: (list(VERSIONS).index(x["version"]), SPLITS.index(x["split"])))

    tests = [{"first": a, "second": b, "split": s, **paired(answers[a, s], answers[b, s], two_way=False),
              "two_way": paired(answers[a, s], answers[b, s], two_way=True)} for a, b in TESTS for s in SPLITS]
    for t, ph in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = ph
    for t, ph in zip(tests, holm([t["two_way"]["p"] for t in tests])):
        t["two_way"]["p_holm"] = ph

    seeds = {s: [r["accuracy"] for r in rows if r["group"] == "E2B seeds" and r["split"] == s] for s in SPLITS}
    spread = {s: {"mean": round(statistics.mean(a), 4), "sd": round(statistics.stdev(a), 4),
                  "min": min(a), "max": max(a)} for s, a in seeds.items()}

    out = estimates / f"compare-jetson-scientsbank-{datetime.now(UTC):%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({
        "kind": "compare-jetson", "board_check": str(check_file),
        "family": "8 tests chosen 2026-11-16 after the runs: per split, E2B tweaked vs E2B untouched and vs "
        "E4B untouched, E4B untouched vs E2B untouched, E4B adapter vs E4B untouched; McNemar on 3-way "
        "labels, Holm over the 8; 2-way beside it with its own Holm",
        "rows": rows, "tests": tests, "seed_spread": spread,
        "time": datetime.now(UTC).isoformat(timespec="seconds")}, indent=2) + "\n")

    print("| Version | Split | Accuracy (95% range) | Macro-F1 | 2-way | Median latency | J per answer | x E2B untouched "
          "| Decode tok/s | Prompt tokens |\n|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lo, hi = r["ci95"]
        print(f"| {r['name']} | {r['split']} | {r['accuracy']:.1%} ({lo:.1%}-{hi:.1%}) | {r['macro_f1']:.3f} | "
              f"{r['two_way_accuracy']:.1%} | {r['median_latency_s']:.2f} s | {r['j_per_answer']:.1f} | "
              f"{r['energy_vs_e2b_untouched']:.2f} | {r['decode_tok_s']:.1f} | {r['mean_prompt_tokens']:.0f} |")
    print("\n| Comparison | Split | Difference | first-only / second-only right | Holm p (3-way) | 2-way diff, "
          "Holm p |\n|---|---|---|---|---|---|")
    for t in tests:
        print(f"| {VERSIONS[t['second']][0]} vs {VERSIONS[t['first']][0]} | {t['split']} | {t['diff']:+.1%} | "
              f"{t['only_first_right']} / {t['only_second_right']} | {t['p_holm']:.2g} | "
              f"{t['two_way']['diff']:+.1%}, {t['two_way']['p_holm']:.2g} |")
    for s, v in spread.items():
        print(f"seeds {s}: {v['mean']:.1%} ± {v['sd']:.1%} ({v['min']:.1%}-{v['max']:.1%})")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
