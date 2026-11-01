"""Stage 3b: reason-guided adapters against the untuned model and the Oct 28 adapter.

    uv run python -m edge_grader.compare --base e2b-kaggle-t4 --ft e2b-kaggle-t4-ft \\
        --new e2b-kaggle-t4-ft-rf e2b-kaggle-t4-ft-lf

The test and the gate were fixed before any grading run (spec 2026-11-01 §2, §5.5): each new
adapter against both baselines on the four Beetle/SAF cells, paired McNemar (Beetle 3-way, SAF
2-way), Holm over all 16 tests at once. The gate passes if one adapter beats both baselines in
the same cell, both differences positive and significant after Holm. SciEntsBank test_uq/test_ud,
per-class recall, unparsed, cut and completion tokens are described, not tested.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from scipy.stats import binomtest

from .board_check import mcnemar, read_answers

HEADLINE = {"beetle": False, "saf": True}  # two_way? Beetle 3-way, SAF 2-way (no "contradictory")
CELLS = [(d, s) for d in HEADLINE for s in ("test_ua", "test_uq")]
DESCRIBED = [("scientsbank", "test_uq"), ("scientsbank", "test_ud")]


def latest(kaggle: Path, dataset: str, label: str, split: str) -> Path:
    """The newest whole-split 3-way run of exactly this label (not a smoke run of 100)."""
    prefix = "" if dataset == "scientsbank" else f"{dataset}-"
    files = [f for f in sorted(kaggle.glob(f"{prefix}baseline-{label}-{split}-3way-*.json"))
             if json.loads(f.read_text())["n_requested"] == 0]
    if not files:
        raise SystemExit(f"no whole-split run of {label} on {dataset} {split} in {kaggle}")
    return files[-1]


def two(label: str | None) -> str | None:
    return None if label is None else ("correct" if label == "correct" else "incorrect")


def paired(a: dict[str, dict], b: dict[str, dict], two_way: bool) -> dict:
    """Two runs on the same answers: accuracies, the difference (second - first), McNemar."""
    if a.keys() != b.keys():
        raise ValueError("the runs graded different answer ids")
    ids = sorted(a)
    conv = two if two_way else (lambda x: x)
    right = [[conv(r[i]["pred"]) == conv(r[i]["gold"]) for i in ids] for r in (a, b)]
    m = mcnemar(*right)
    k = m["only_first_right"] + m["only_second_right"]
    acc = [round(sum(x) / len(ids), 4) for x in right]
    return {"n": len(ids), "first_accuracy": acc[0], "second_accuracy": acc[1],
            "diff": round(acc[1] - acc[0], 4), "only_first_right": m["only_first_right"],
            "only_second_right": m["only_second_right"],
            # Unrounded (board_check rounds to 3 places), since Holm multiplies it by up to 16.
            "p": binomtest(m["only_first_right"], k).pvalue if k else 1.0}


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = running
    return out


def gate(tests: list[dict]) -> tuple[bool, list[dict]]:
    """Cells where one adapter beats both baselines, each by a positive, Holm-significant margin."""
    wins = {}
    for t in tests:
        if t["diff"] > 0 and t["p_holm"] < 0.05:
            wins.setdefault((t["adapter"], t["dataset"], t["split"]), set()).add(t["baseline"])
    cells = [{"adapter": a, "dataset": d, "split": s}
             for (a, d, s), beaten in sorted(wins.items()) if len(beaten) == 2]
    return bool(cells), cells


def recall(rows: dict[str, dict]) -> dict[str, float]:
    classes = sorted({r["gold"] for r in rows.values()})
    return {c: round(sum(r["pred"] == c for r in rows.values() if r["gold"] == c)
                     / sum(r["gold"] == c for r in rows.values()), 4) for c in classes}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--kaggle", type=Path, default=Path("results/kaggle"))
    ap.add_argument("--base", required=True, help="the untuned model's run label")
    ap.add_argument("--ft", required=True, help="the Oct 28 label-only adapter's run label")
    ap.add_argument("--new", nargs="+", required=True, help="the reason-guided adapters' run labels")
    args = ap.parse_args(argv)
    labels = [args.base, args.ft, *args.new]

    runs, answers = {}, {}
    for cell in [*CELLS, *DESCRIBED]:
        for label in labels:
            f = latest(args.kaggle, cell[0], label, cell[1])
            s = json.loads(f.read_text())
            runs[(*cell, label)] = {"file": f.name, "build": s["server"]["build_info"],
                                    "reply": s.get("reply", "label"), **{
                k: s["metrics"].get(k) for k in ("n", "unparsed", "errors", "cut",
                                                 "median_completion_tokens")}}
            answers[(*cell, label)] = read_answers(f)
    builds = {r["build"] for r in runs.values()}

    tests = []
    for cell in CELLS:
        for new in args.new:
            for name, base in (("base", args.base), ("ft", args.ft)):
                p = paired(answers[(*cell, base)], answers[(*cell, new)], HEADLINE[cell[0]])
                tests.append({"adapter": new, "baseline": name, "dataset": cell[0], "split": cell[1],
                              "scheme": "2way" if HEADLINE[cell[0]] else "3way", **p})
    for t, p in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = p
    passed, wins = gate(tests)
    described = [{"dataset": d, "split": s, "label": label, "accuracy": paired(
                     answers[(d, s, args.base)], answers[(d, s, label)], False)["second_accuracy"]}
                 for d, s in DESCRIBED for label in labels]
    result = {
        "kind": "compare-reason-ft",
        "labels": {"base": args.base, "ft": args.ft, "new": args.new},
        "builds": sorted(builds),
        "runs": [{"dataset": d, "split": s, "label": lab, **r} for (d, s, lab), r in runs.items()],
        "tests": tests,
        "gate": {"passed": passed, "wins": wins,
                 "rule": "one adapter beats both baselines in the same Beetle/SAF cell, "
                         "diff > 0 and Holm p < 0.05 for both (spec 2026-11-01 §2)"},
        "described": described,
        "recall": {"beetle": {s: {lab: recall(answers[("beetle", s, lab)]) for lab in labels}
                              for s in ("test_ua", "test_uq")}},
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    out = args.kaggle / f"compare-reason-ft-{datetime.now(UTC):%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")

    if len(builds) > 1:
        print(f"WARNING: runs on more than one llama.cpp build: {sorted(builds)}")
    for t in tests:
        print(f"{t['dataset']:6} {t['split']} {t['scheme']} {t['adapter']} vs {t['baseline']:4} "
              f"{t['first_accuracy']:.1%} -> {t['second_accuracy']:.1%} ({t['diff']:+.1%}) "
              f"fixed/broken {t['only_second_right']}/{t['only_first_right']} "
              f"p {t['p']:.3g} Holm {t['p_holm']:.3g}")
    print(f"gate: {'PASSED' if passed else 'not passed'} {wins} -> {out}")


if __name__ == "__main__":
    main()
