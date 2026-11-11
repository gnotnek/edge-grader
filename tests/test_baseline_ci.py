import json

from edge_grader import baseline_ci, metrics


def test_ci_resamples_whole_questions():
    gold = ["correct"] * 6
    # Question Q1 all right, Q2 all wrong: a range over answers would be narrow, over questions
    # it spans 0 to 1, since a resample can draw Q2 twice.
    pred = ["correct"] * 3 + ["incorrect"] * 3
    groups = ["U.Q1"] * 3 + ["U.Q2"] * 3
    ci = metrics.ci95(gold, pred, groups, resamples=500)
    assert ci["accuracy"] == [0.0, 1.0] and ci["groups"] == 2
    assert metrics.ci95(gold, gold, groups)["accuracy"] == [1.0, 1.0]
    assert metrics.ci95(gold, pred, groups) == metrics.ci95(gold, pred, groups)  # seeded


def test_rescore_adds_two_way_and_majority(tmp_path):
    rows = [("U.Q1.1", "correct", "correct"), ("U.Q1.2", "incorrect", "incorrect"),
            ("U.Q2.1", "contradictory", "incorrect"), ("U.Q2.2", "incorrect", None)]
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "run.jsonl").write_text(
        "".join(json.dumps({"id": i, "gold": g, "pred": p}) + "\n" for i, g, p in rows))
    (tmp_path / "run.json").write_text(json.dumps(
        {"label": "x", "split": "test_uq", "scheme": "3way", "metrics": {"accuracy": 0.5}}))
    baseline_ci.main([str(tmp_path / "run.json"), "--out", str(tmp_path / "ci")])

    [out] = (tmp_path / "ci").glob("baseline-ci-*.json")
    [r] = json.loads(out.read_text())["runs"]
    assert r["three_way"]["accuracy"] == 0.5 and r["three_way"]["unparsed"] == 1
    assert r["two_way"]["accuracy"] == 0.75  # contradictory graded incorrect is right in 2-way
    assert r["majority"]["label"] == "incorrect" and r["majority"]["accuracy"] == 0.5


def test_fast_macro_f1_matches_sklearn():
    from sklearn.metrics import f1_score

    gold = ["correct", "incorrect", "contradictory", "incorrect", "correct", "contradictory"]
    pred = ["correct", "correct", "incorrect", "incorrect", "__unparsed__", "contradictory"]
    labels = sorted(set(gold))
    assert metrics._macro_f1(gold, pred, labels) == f1_score(
        gold, pred, average="macro", labels=labels, zero_division=0)
    assert metrics._macro_f1(["correct"], ["incorrect"], ["correct", "incorrect"]) == 0.0
