"""5-way labels: the E4B adapter trained on SciEntsBank's 5-way labels against the 3-way one.

    uv run python -m edge_grader.compare_fiveway

Spec 2026-11-12 §5.5-5.6, fixed before grading. Every run is a whole split graded on the DGX Spark,
scored 3-way (the 5-way adapter's replies mapped to 3-way). The family is 6 tests: per split, the
5-way seed-0 adapter against the 3-way seed-0 adapter and against untouched E4B; paired McNemar
on 3-way labels, Holm over all 6. 2-way beside it, its own Holm. Described, not tested: each
version's scores, the 3-way seeds' spread, completeness errors by 5-way gold, the 5-way reply mix.

The method helps (spec §2) if it is significantly better than the 3-way seed 0 on at least one
split after Holm and significantly worse on none.
"""

import json
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from .board_check import read_answers
from .compare import holm, latest, paired
from .compare_soup import BAR, SEED_LABELS, SPLITS, scores, spread

BASE = "e4b-spark-gb10"
THREE = f"{BASE}-ft-s0"  # the soup queue's 3-way seed 0: same seed, only the labels differ
FIVE = f"{BASE}-ft-5w-s0"


def plan() -> list[tuple[str, str]]:
    """(split, the version the 5-way adapter is tested against) for the 6 tests."""
    return [(s, other) for s in SPLITS for other in (THREE, BASE)]


def completeness(rows: dict[str, dict], gold5: dict[str, str]) -> dict[str, int]:
    """The errors the method aims at, counted with the 5-way gold."""
    return {"correct_called_incorrect": sum(gold5[i] == "correct" and r["pred"] == "incorrect"
                                            for i, r in rows.items()),
            "partial_called_correct": sum(gold5[i] == "partially_correct_incomplete" and r["pred"] == "correct"
                                          for i, r in rows.items()),
            "errors": sum(r["pred"] != r["gold"] for r in rows.values())}


def decide(tests: list[dict]) -> bool:
    mine = [t for t in tests if t["against"] == THREE]
    better = any(t["diff"] > 0 and t["p_holm"] < 0.05 for t in mine)
    worse = any(t["diff"] < 0 and t["p_holm"] < 0.05 for t in mine)
    return better and not worse


def gold_5way(split: str) -> dict[str, str]:
    from . import data
    return {i.id: i.label for i in data.load(split, "5way")}


def main(spark: Path = Path("results/spark"), gold5: Callable[[str], dict[str, str]] = gold_5way) -> None:
    splits, builds, answers = {}, set(), {}
    for split in SPLITS:
        versions = {}
        # 3-way seeds with a whole-split run (the 100-answer checks after training don't count)
        seeds = [s for s in SEED_LABELS if any(json.loads(f.read_text())["n_requested"] == 0
                                               for f in spark.glob(f"baseline-{s}-{split}-3way-*.json"))]
        for label in (BASE, THREE, FIVE, *[s for s in seeds if s != THREE]):
            f = latest(spark, "scientsbank", label, split)
            builds.add(json.loads(f.read_text())["server"]["build_info"])
            answers[split, label] = read_answers(f)
            versions[label] = {"file": f.name, **scores(answers[split, label])}
        g5 = gold5(split)
        five = answers[split, FIVE]
        splits[split] = {
            "versions": versions,
            "seeds_3way": {"n": len(seeds), **({k: spread([versions[s][k] for s in seeds])
                                                for k in ("accuracy", "macro_f1")} if len(seeds) > 1 else {})},
            "completeness": {lab: completeness(answers[split, lab], g5) for lab in (THREE, FIVE)},
            "reply_mix_5way": dict(Counter(r.get("pred5") for r in five.values()).most_common()),
            "gold_mix_5way": dict(Counter(g5[i] for i in five).most_common()),
        }

    tests = []
    for split, other in plan():
        a, b = answers[split, other], answers[split, FIVE]
        tests.append({"split": split, "against": other, **paired(a, b, two_way=False),
                      "two_way": paired(a, b, two_way=True)})
    for t, ph in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = ph
        if t["against"] == BASE:
            t["passes_bar"] = bool(t["diff"] >= BAR and ph < 0.05)
    for t, ph in zip(tests, holm([t["two_way"]["p"] for t in tests])):
        t["two_way"]["p_holm"] = ph
    helps = decide(tests)

    out = spark / f"compare-fiveway-scientsbank-{datetime.now(UTC):%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({
        "kind": "compare-fiveway", "spec": "docs/superpowers/specs/2026-11-12-fiveway-training-design.md",
        "family": f"{len(tests)} tests fixed before grading: per split, the 5-way adapter vs the 3-way "
        "seed 0 and vs untouched; McNemar on 3-way labels, Holm over all; 2-way beside it with its own Holm",
        "rule": "helps if significantly better than the 3-way seed 0 (Holm p < 0.05) on at least one split "
        "and significantly worse on none", "helps": helps,
        "builds": sorted(builds), "splits": splits, "tests": tests,
        "time": datetime.now(UTC).isoformat(timespec="seconds")}, indent=2) + "\n")

    if len(builds) > 1:
        print(f"WARNING: runs on more than one llama.cpp build: {sorted(builds)}")
    for split, v in splits.items():
        vs, sd = v["versions"], v["seeds_3way"]
        spread_txt = (f", 3-way seeds {sd['accuracy']['mean']:.1%} ± {sd['accuracy']['sd']:.1%} "
                      f"({sd['accuracy']['min']:.1%}-{sd['accuracy']['max']:.1%})" if "accuracy" in sd else "")
        c3, c5 = v["completeness"][THREE], v["completeness"][FIVE]
        print(f"{split}: untouched {vs[BASE]['accuracy']:.1%}, 3-way s0 {vs[THREE]['accuracy']:.1%}, "
              f"5-way s0 {vs[FIVE]['accuracy']:.1%}{spread_txt} | completeness errors "
              f"{c3['correct_called_incorrect'] + c3['partial_called_correct']} -> "
              f"{c5['correct_called_incorrect'] + c5['partial_called_correct']}")
    for t in tests:
        print(f"  {t['split']} 5-way vs {t['against']:22} {t['diff']:+.1%} fixed/broken "
              f"{t['only_second_right']}/{t['only_first_right']} Holm {t['p_holm']:.3g}"
              f"{' BAR' if t.get('passes_bar') else ''} | 2-way {t['two_way']['diff']:+.1%} "
              f"Holm {t['two_way']['p_holm']:.3g}")
    print(f"5-way labels {'help' if helps else 'do not help'} -> {out}")


if __name__ == "__main__":
    main()
