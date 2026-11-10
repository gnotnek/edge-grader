"""Adapter soup on SciEntsBank: the soup of five E4B seeds against untouched E4B and each seed.

    uv run python -m edge_grader.compare_soup

Spec 2026-11-10 §5.5-5.6, fixed before grading. Every run is a whole split graded on the DGX Spark.
The family is 18 tests: per split, the soup against untouched E4B and against each of the five
seeds; paired McNemar on 3-way labels, Holm over all 18. 2-way (contradictory merged into
incorrect) beside it, its own Holm over the same 18. Described, not tested: each version's scores,
the seeds' mean and SD, and the Kaggle-trained adapter graded on the Spark (e4b-spark-gb10-ft).

The soup is adopted (spec §2) if on every split its accuracy is at least the seeds' mean and no
seed is significantly better than it after Holm.
"""

import json
import statistics
from datetime import UTC, datetime
from pathlib import Path

from .board_check import read_answers
from .compare import holm, latest, paired, recall, two
from .metrics import score

SPLITS = ("test_ua", "test_uq", "test_ud")
SEEDS = (0, 1, 2, 3, 4)
BASE = "e4b-spark-gb10"
SEED_LABELS = tuple(f"{BASE}-ft-s{n}" for n in SEEDS)
SOUP = f"{BASE}-ft-soup"
KAGGLE = f"{BASE}-ft"  # the Oct 27 Kaggle adapter, graded on the Spark on Nov 8
BAR = 0.05


def plan() -> list[tuple[str, str]]:
    """(split, the version the soup is tested against) for the 18 tests."""
    return [(s, other) for s in SPLITS for other in (BASE, *SEED_LABELS)]


def scores(rows: dict[str, dict]) -> dict:
    gold, pred = [r["gold"] for r in rows.values()], [r["pred"] for r in rows.values()]
    s3, s2 = score(gold, pred), score([two(g) for g in gold], [two(p) for p in pred])
    return {"accuracy": s3["accuracy"], "macro_f1": s3["macro_f1"], "unparsed": s3["unparsed"],
            "two_way_accuracy": s2["accuracy"], "two_way_macro_f1": s2["macro_f1"], "recall": recall(rows)}


def spread(xs: list[float]) -> dict:
    return {"mean": round(statistics.mean(xs), 4), "sd": round(statistics.stdev(xs), 4),
            "min": min(xs), "max": max(xs)}


def decide(splits: dict, tests: list[dict]) -> bool:
    above_mean = all(v["versions"][SOUP]["accuracy"] >= v["seeds"]["accuracy"]["mean"] for v in splits.values())
    seed_better = any(t["against"] in SEED_LABELS and t["diff"] < 0 and t["p_holm"] < 0.05 for t in tests)
    return above_mean and not seed_better


def main(spark: Path = Path("results/spark")) -> None:
    splits, builds, answers = {}, set(), {}
    for split in SPLITS:
        versions = {}
        for label in (BASE, *SEED_LABELS, SOUP, KAGGLE):
            f = latest(spark, "scientsbank", label, split)
            builds.add(json.loads(f.read_text())["server"]["build_info"])
            answers[split, label] = read_answers(f)
            versions[label] = {"file": f.name, **scores(answers[split, label])}
        seeds = {k: spread([versions[s][k] for s in SEED_LABELS])
                 for k in ("accuracy", "macro_f1", "two_way_accuracy", "two_way_macro_f1")}
        splits[split] = {"versions": versions, "seeds": seeds}

    tests = []
    for split, other in plan():
        a, b = answers[split, other], answers[split, SOUP]
        tests.append({"split": split, "against": other, **paired(a, b, two_way=False),
                      "two_way": paired(a, b, two_way=True)})
    for t, ph in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = ph
        if t["against"] == BASE:
            t["passes_bar"] = bool(t["diff"] >= BAR and ph < 0.05)
    for t, ph in zip(tests, holm([t["two_way"]["p"] for t in tests])):
        t["two_way"]["p_holm"] = ph
    adopted = decide(splits, tests)

    out = spark / f"compare-soup-scientsbank-{datetime.now(UTC):%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({
        "kind": "compare-soup", "spec": "docs/superpowers/specs/2026-11-10-adapter-soup-design.md",
        "family": f"{len(tests)} tests fixed before grading: per split, soup vs untouched and vs each "
        "seed; McNemar on 3-way labels, Holm over all; 2-way beside it with its own Holm",
        "rule": "adopted if on every split the soup's accuracy >= the seeds' mean and no seed is "
        "significantly better (Holm p < 0.05)", "adopted": adopted,
        "builds": sorted(builds), "splits": splits, "tests": tests,
        "time": datetime.now(UTC).isoformat(timespec="seconds")}, indent=2) + "\n")

    if len(builds) > 1:
        print(f"WARNING: runs on more than one llama.cpp build: {sorted(builds)}")
    for split, v in splits.items():
        sa, sf = v["seeds"]["accuracy"], v["seeds"]["macro_f1"]
        print(f"{split}: untouched {v['versions'][BASE]['accuracy']:.1%}, seeds {sa['mean']:.1%} ± "
              f"{sa['sd']:.1%} ({sa['min']:.1%}-{sa['max']:.1%}), soup {v['versions'][SOUP]['accuracy']:.1%}, "
              f"Kaggle adapter {v['versions'][KAGGLE]['accuracy']:.1%} | macro-F1 seeds {sf['mean']:.3f} ± "
              f"{sf['sd']:.3f}, soup {v['versions'][SOUP]['macro_f1']:.3f}")
    for t in tests:
        print(f"  {t['split']} soup vs {t['against']:24} {t['diff']:+.1%} fixed/broken "
              f"{t['only_second_right']}/{t['only_first_right']} Holm {t['p_holm']:.3g}"
              f"{' BAR' if t.get('passes_bar') else ''} | 2-way {t['two_way']['diff']:+.1%} "
              f"Holm {t['two_way']['p_holm']:.3g}")
    print(f"soup {'adopted' if adopted else 'not adopted'} -> {out}")


if __name__ == "__main__":
    main()
