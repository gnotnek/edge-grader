"""The 5-point bar on SciEntsBank: every adapter against its own untouched model, per split.

    uv run python -m edge_grader.compare_bar

A tweak counts if it adds at least 5 accuracy points over the same model and the gain is
significant: paired McNemar (3-way labels), Holm p < 0.05 (owner, 2026-11-05). The Holm family is
every test below at once: the Beetle + SAF adapters on all three splits, the SciEntsBank-trained
ones on test_uq and test_ud only (test_ua shares its questions with their training set). The
family was set after the runs were graded (2026-11-08), so it's recorded as such.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from .board_check import read_answers
from .compare import holm, latest, paired

SPLITS = ("test_ua", "test_uq", "test_ud")
REPLIES = ("", "-lf", "-rf")  # label-only, label-first, reason-first
TRAINED = {"scientsbank": "", "beetle-saf": "-beetle-saf"}  # label suffix per training set
BAR = 0.05


def plan(models=("e2b", "e4b")) -> list[tuple[str, str, str, str]]:
    """(model, training set, adapter label, split) for every test in the family."""
    return [(m, t, f"{m}-kaggle-t4-ft{r}{suffix}", s)
            for m in models for t, suffix in TRAINED.items() for r in REPLIES for s in SPLITS
            if not (t == "scientsbank" and s == "test_ua")]


def main(kaggle: Path = Path("results/kaggle")) -> None:
    tests, builds = [], set()
    for model, trained, label, split in plan():
        files = [latest(kaggle, "scientsbank", lab, split) for lab in (f"{model}-kaggle-t4", label)]
        sums = [json.loads(f.read_text()) for f in files]
        builds |= {s["server"]["build_info"] for s in sums}
        p = paired(*(read_answers(f) for f in files), two_way=False)
        tests.append({"model": model, "trained_on": trained, "adapter": label, "split": split,
                      "base_file": files[0].name, "adapter_file": files[1].name,
                      "base_macro_f1": sums[0]["metrics"]["macro_f1"],
                      "adapter_macro_f1": sums[1]["metrics"]["macro_f1"], **p})
    for t, ph in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = ph
        t["passes"] = bool(t["diff"] >= BAR and ph < 0.05)  # numpy bool otherwise
    out = kaggle / f"compare-bar-scientsbank-{datetime.now(UTC):%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({
        "kind": "compare-bar", "rule": "diff >= +5 accuracy points and Holm p < 0.05, "
        "adapter vs the same untouched model, paired McNemar on 3-way labels (owner 2026-11-05)",
        "family": len(tests), "builds": sorted(builds), "tests": tests,
        "time": datetime.now(UTC).isoformat(timespec="seconds")}, indent=2) + "\n")

    if len(builds) > 1:
        print(f"WARNING: runs on more than one llama.cpp build: {sorted(builds)}")
    for t in tests:
        print(f"{t['adapter']:32} {t['split']} {t['first_accuracy']:.1%} -> {t['second_accuracy']:.1%} "
              f"({t['diff']:+.1%}) mF1 {t['base_macro_f1']:.3f} -> {t['adapter_macro_f1']:.3f} "
              f"fixed/broken {t['only_second_right']}/{t['only_first_right']} "
              f"Holm {t['p_holm']:.3g} {'PASS' if t['passes'] else ''}")
    print(f"{sum(t['passes'] for t in tests)} of {len(tests)} pass -> {out}")


if __name__ == "__main__":
    main()
