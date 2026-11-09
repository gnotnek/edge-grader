"""Where the Beetle + SAF adapters lose on SciEntsBank: per-label recall, and the same runs
scored as correct vs not-correct ("contradictory" and "incorrect" merged).

    uv run python -m edge_grader.compare_boundary

Exploratory, set after the 5-point-bar runs were graded (2026-11-09), to answer whether a
stronger reason teacher could fix E4B: if E4B gains once the contradictory/incorrect boundary
is ignored, the loss is about that boundary, not about the reasons. Paired McNemar on the 2-way
labels, Holm over these 18 tests; recorded as post hoc, not part of the 5-point bar.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from .board_check import read_answers
from .compare import holm, latest, paired
from .compare_bar import plan

LABELS = ("correct", "contradictory", "incorrect")


def label_mix(answers: dict[str, dict]) -> dict[str, dict[str, float]]:
    """Per gold label: recall, and the share of all answers predicted as that label."""
    rows = answers.values()
    out = {}
    for g in LABELS:
        gold = [r for r in rows if r["gold"] == g]
        out[g] = {"recall": round(sum(r["pred"] == g for r in gold) / len(gold), 4) if gold else None,
                  "pred_share": round(sum(r["pred"] == g for r in rows) / len(rows), 4),
                  "gold_share": round(len(gold) / len(rows), 4)}
    return out


def main(kaggle: Path = Path("results/kaggle")) -> None:
    tests = []
    for model, trained, label, split in plan():
        if trained != "beetle-saf":
            continue
        files = [latest(kaggle, "scientsbank", lab, split) for lab in (f"{model}-kaggle-t4", label)]
        a, b = (read_answers(f) for f in files)
        three = paired(a, b, two_way=False)
        tests.append({"model": model, "adapter": label, "split": split,
                      "base_file": files[0].name, "adapter_file": files[1].name,
                      "diff_3way": three["diff"], **paired(a, b, two_way=True),
                      "base_labels": label_mix(a), "adapter_labels": label_mix(b)})
    for t, ph in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = ph
    out = kaggle / f"compare-boundary-scientsbank-{datetime.now(UTC):%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({
        "kind": "compare-boundary", "post_hoc": True,
        "rule": "Beetle + SAF adapter vs the same untouched model, paired McNemar on 2-way labels "
        "(contradictory merged into incorrect), Holm over these tests; exploratory, set 2026-11-09",
        "family": len(tests), "tests": tests,
        "time": datetime.now(UTC).isoformat(timespec="seconds")}, indent=2) + "\n")

    for t in tests:
        bl, al = t["base_labels"], t["adapter_labels"]
        print(f"{t['adapter']:32} {t['split']} 3-way {t['diff_3way']:+.1%}  2-way "
              f"{t['first_accuracy']:.1%} -> {t['second_accuracy']:.1%} ({t['diff']:+.1%}, Holm "
              f"{t['p_holm']:.3g})  recall c/x/i " + " ".join(
                  f"{bl[g]['recall']:.2f}->{al[g]['recall']:.2f}" for g in LABELS)
              + f"  pred contra {al['contradictory']['pred_share']:.0%}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
