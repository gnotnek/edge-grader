import json

from edge_grader import compare_soup as cs

GOLD = ["correct", "incorrect", "contradictory"] * 20  # 60 answers


def write(spark, label, split, right):
    """A whole-split run whose first `right` answers are graded correctly."""
    name = f"baseline-{label}-{split}-3way-20261111-000000"
    rows = [{"id": f"EM.1b.{i}.1", "gold": g, "pred": g if i < right else "incorrect" if g != "incorrect"
             else "correct"} for i, g in enumerate(GOLD)]
    (spark / "raw").mkdir(exist_ok=True)
    (spark / "raw" / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (spark / f"{name}.json").write_text(json.dumps({"n_requested": 0, "server": {"build_info": "b-a3a1c47"}}))


def test_family_is_18_fixed_tests():
    tests = cs.plan()
    assert len(tests) == 18 == len(set(tests))
    assert ("test_ud", "e4b-spark-gb10") in tests and ("test_ua", "e4b-spark-gb10-ft-s4") in tests


def test_main_scores_spreads_and_decides(tmp_path, capsys):
    right = {cs.BASE: 20, cs.SOUP: 55, cs.KAGGLE: 50, **dict(zip(cs.SEED_LABELS, (50, 52, 48, 51, 49)))}
    for split in cs.SPLITS:
        for label, r in right.items():
            write(tmp_path, label, split, r)
    cs.main(tmp_path)
    out = json.loads(next(tmp_path.glob("compare-soup-scientsbank-*.json")).read_text())
    ua = out["splits"]["test_ua"]
    assert ua["seeds"]["accuracy"]["mean"] == round(250 / 5 / 60, 4)
    assert ua["seeds"]["accuracy"]["min"] == round(48 / 60, 4)
    assert ua["versions"][cs.SOUP]["accuracy"] == round(55 / 60, 4)
    assert len(out["tests"]) == 18 and all("p_holm" in t and "p_holm" in t["two_way"] for t in out["tests"])
    base = [t for t in out["tests"] if t["against"] == cs.BASE]
    assert all(t["passes_bar"] for t in base)  # +58 points, 35 fixed / 0 broken
    assert out["adopted"] is True
    assert "soup adopted" in capsys.readouterr().out


def test_a_seed_significantly_better_blocks_adoption():
    splits = {s: {"versions": {cs.SOUP: {"accuracy": 0.8}}, "seeds": {"accuracy": {"mean": 0.79}}}
              for s in cs.SPLITS}
    worse = [{"against": cs.SEED_LABELS[2], "diff": -0.02, "p_holm": 0.01}]
    assert cs.decide(splits, []) is True
    assert cs.decide(splits, worse) is False
    splits["test_ud"]["versions"][cs.SOUP]["accuracy"] = 0.78
    assert cs.decide(splits, []) is False
