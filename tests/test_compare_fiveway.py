import json

from edge_grader import compare_fiveway as cf

GOLD5 = ["correct", "partially_correct_incomplete", "contradictory", "irrelevant"] * 15  # 60 answers
GOLD = [{"partially_correct_incomplete": "incorrect", "irrelevant": "incorrect"}.get(g, g) for g in GOLD5]
IDS = [f"EM.1b.{i}.1" for i in range(len(GOLD))]


def write(spark, label, split, right, n_requested=0, five=False):
    """A run whose first `right` answers are graded correctly; the rest wrong in the way the
    method targets (correct -> incorrect, incorrect -> correct)."""
    name = f"baseline-{label}-{split}-3way-20261112-00000{n_requested and 1}"
    rows = []
    for i, g in enumerate(GOLD):
        pred = g if i < right else {"correct": "incorrect", "incorrect": "correct"}.get(g, "incorrect")
        row = {"id": IDS[i], "gold": g, "pred": pred}
        if five:
            row["pred5"] = GOLD5[i] if i < right else "irrelevant"
        rows.append(row)
    (spark / "raw").mkdir(exist_ok=True)
    (spark / "raw" / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (spark / f"{name}.json").write_text(json.dumps({"n_requested": n_requested,
                                                    "server": {"build_info": "b-a3a1c47"}}))


def gold5(_split):
    return dict(zip(IDS, GOLD5))


def test_family_is_6_fixed_tests():
    tests = cf.plan()
    assert len(tests) == 6 == len(set(tests))
    assert ("test_ud", cf.THREE) in tests and ("test_ua", cf.BASE) in tests


def test_main_scores_describes_and_decides(tmp_path, capsys):
    right = {cf.BASE: 20, cf.THREE: 40, cf.FIVE: 58, "e4b-spark-gb10-ft-s1": 42}
    for split in cf.SPLITS:
        for label, r in right.items():
            write(tmp_path, label, split, r, five=label == cf.FIVE)
        write(tmp_path, "e4b-spark-gb10-ft-s2", split, 30, n_requested=100)  # a 100-answer check only
    cf.main(tmp_path, gold5)
    out = json.loads(next(tmp_path.glob("compare-fiveway-scientsbank-*.json")).read_text())
    ua = out["splits"]["test_ua"]
    assert ua["versions"][cf.FIVE]["accuracy"] == round(58 / 60, 4)
    assert ua["seeds_3way"]["n"] == 2 and ua["seeds_3way"]["accuracy"]["max"] == round(42 / 60, 4)
    assert "e4b-spark-gb10-ft-s2" not in ua["versions"]
    # answers 40-59 of the 3-way run are wrong: 5 correct and 5 partly correct among them
    assert ua["completeness"][cf.THREE] == {"correct_called_incorrect": 5, "partial_called_correct": 5,
                                            "errors": 20}
    assert ua["reply_mix_5way"]["irrelevant"] == 14 + 2  # 14 right among the first 58, 2 wrong ones
    assert len(out["tests"]) == 6 and all("p_holm" in t and "p_holm" in t["two_way"] for t in out["tests"])
    assert out["helps"] is True
    assert "5-way labels help" in capsys.readouterr().out


def test_significantly_worse_on_one_split_means_no():
    tests = [{"against": cf.THREE, "diff": 0.05, "p_holm": 0.01},
             {"against": cf.THREE, "diff": -0.04, "p_holm": 0.02},
             {"against": cf.BASE, "diff": 0.10, "p_holm": 0.001}]
    assert cf.decide(tests) is False
    assert cf.decide(tests[:1]) is True
    assert cf.decide(tests[2:]) is False  # beating untouched alone isn't the method's test
