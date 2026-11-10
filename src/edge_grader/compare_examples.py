"""Graded examples in the prompt (spec 2026-11-10 §5.4): E4B on the DGX Spark, all three splits.

    uv run python -m edge_grader.compare_examples

Paired McNemar on 3-way labels, Holm over the 12 tests below at once (fixed before grading). The
5-point bar (owner, 2026-11-05) applies to the two "with vs without examples" rows. 2-way beside
each test with its own Holm over the same 12, secondary as in compare_bar.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from .board_check import read_answers
from .compare import holm, latest, paired, recall, two
from .compare_bar import BAR, SPLITS
from .data import question_id
from .metrics import score

BASE, FT = "e4b-spark-gb10", "e4b-spark-gb10-ft"
EX, FT_EX, RAND = f"{BASE}-ex4", f"{FT}-ex4", f"{BASE}-rand4"
# (first, second, bar applies): second - first is the difference tested.
PAIRS = [(BASE, EX, True), (FT, FT_EX, True), (FT, EX, False), (RAND, EX, False)]
DESCRIBED = [(BASE, RAND)]  # random examples vs none: paired, not in the Holm family


def stage(label: str) -> str:
    return "examples" if label.endswith(("-ex4", "-rand4")) else "baseline"


def plan() -> list[tuple[str, str, bool, str]]:
    return [(a, b, bar, s) for a, b, bar in PAIRS for s in SPLITS]


def main(spark: Path = Path("results/spark")) -> None:
    labels = [BASE, FT, EX, FT_EX, RAND]
    files = {(lab, s): latest(spark, "scientsbank", lab, s, stage(lab)) for lab in labels for s in SPLITS}
    sums = {k: json.loads(f.read_text()) for k, f in files.items()}
    answers = {k: read_answers(f) for k, f in files.items()}
    builds = {s["server"]["build_info"] for s in sums.values()}

    def test(a: str, b: str, split: str) -> dict:
        x, y = answers[(a, split)], answers[(b, split)]
        f2 = [score([two(r["gold"]) for r in v.values()], [two(r["pred"]) for r in v.values()])["macro_f1"]
              for v in (x, y)]
        return {"first": a, "second": b, "split": split,
                "first_macro_f1": sums[(a, split)]["metrics"]["macro_f1"],
                "second_macro_f1": sums[(b, split)]["metrics"]["macro_f1"], **paired(x, y, two_way=False),
                "two_way": {**paired(x, y, two_way=True), "first_macro_f1": f2[0], "second_macro_f1": f2[1]}}

    tests = [{**test(a, b, s), "bar_applies": bar} for a, b, bar, s in plan()]
    for t, ph in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = ph
        t["passes"] = bool(t["bar_applies"] and t["diff"] >= BAR and ph < 0.05)
    for t, ph in zip(tests, holm([t["two_way"]["p"] for t in tests])):
        t["two_way"]["p_holm"] = ph

    runs = []
    for (lab, s), summ in sums.items():
        m, rows = summ["metrics"], answers[(lab, s)].values()
        shots = [r["examples"] for r in rows if "examples" in r]
        runs.append({"label": lab, "split": s, "file": files[(lab, s)].name, "accuracy": m["accuracy"],
                     "macro_f1": m["macro_f1"], "unparsed": m["unparsed"], "errors": m["errors"],
                     "median_prompt_tokens": m.get("median_prompt_tokens"),
                     "max_prompt_tokens": m.get("max_prompt_tokens"),
                     "recall": recall(answers[(lab, s)]),
                     # share of examples from the graded answer's own question
                     "same_question_share": round(sum(question_id(e) == question_id(r["id"])
                                                      for r in rows for e in r.get("examples", []))
                                                  / sum(map(len, shots)), 4) if shots else None})
    described = [test(a, b, s) for a, b in DESCRIBED for s in SPLITS]

    out = spark / f"compare-examples-scientsbank-{datetime.now(UTC):%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({
        "kind": "compare-examples", "spec": "docs/superpowers/specs/2026-11-10-graded-examples-design.md",
        "rule": "bar rows: diff >= +5 accuracy points and Holm p < 0.05, paired McNemar on 3-way labels; "
                "Holm over all 12 tests (fixed before grading, 2026-11-10)",
        "two_way": "secondary: contradictory merged into incorrect, McNemar, own Holm over the same 12",
        "family": len(tests), "builds": sorted(builds), "tests": tests, "described": described, "runs": runs,
        "time": datetime.now(UTC).isoformat(timespec="seconds")}, indent=2) + "\n")

    if len(builds) > 1:
        print(f"WARNING: runs on more than one llama.cpp build: {sorted(builds)}")
    for r in runs:
        print(f"{r['label']:24} {r['split']} acc {r['accuracy']:.1%} mF1 {r['macro_f1']:.3f} "
              f"prompt tokens median {r['median_prompt_tokens']} max {r['max_prompt_tokens']}")
    for t in tests + described:
        print(f"{t['second']:24} vs {t['first']:24} {t['split']} {t['first_accuracy']:.1%} -> "
              f"{t['second_accuracy']:.1%} ({t['diff']:+.1%}) fixed/broken "
              f"{t['only_second_right']}/{t['only_first_right']} "
              f"Holm {t.get('p_holm', float('nan')):.3g} {'PASS' if t.get('passes') else '    '} | 2-way "
              f"{t['two_way']['first_accuracy']:.1%} -> {t['two_way']['second_accuracy']:.1%}")
    print(f"{sum(t['passes'] for t in tests)} of {sum(t['bar_applies'] for t in tests)} bar tests pass -> {out}")


if __name__ == "__main__":
    main()
