"""Stage 3b comparison and its gate (spec 2026-11-01 §2, §5.5), fixed before any grading run."""

import json

import pytest

from edge_grader import compare


def rows(spec: dict[str, tuple[str, str]]) -> dict[str, dict]:
    return {i: {"id": i, "gold": g, "pred": p} for i, (g, p) in spec.items()}


def test_holm_steps_down_and_stays_monotone():
    assert compare.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert compare.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_paired_refuses_different_answers():
    with pytest.raises(ValueError, match="ids"):
        compare.paired(rows({"a": ("correct", "correct")}), rows({"b": ("correct", "correct")}), False)


def test_paired_two_way_counts_contradictory_as_incorrect():
    a = rows({"a": ("contradictory", "incorrect"), "b": ("correct", "correct")})
    b = rows({"a": ("contradictory", "contradictory"), "b": ("correct", "incorrect")})
    three = compare.paired(a, b, two_way=False)
    two = compare.paired(a, b, two_way=True)
    assert (three["first_accuracy"], three["second_accuracy"]) == (0.5, 0.5)
    assert (two["first_accuracy"], two["second_accuracy"]) == (1.0, 0.5)
    assert two["diff"] == -0.5 and two["only_first_right"] == 1 and two["only_second_right"] == 0


def case(adapter, baseline, cell, diff, p_holm):
    return {"adapter": adapter, "baseline": baseline, "dataset": cell[0], "split": cell[1],
            "diff": diff, "p_holm": p_holm}


BEETLE_UQ, SAF_UA = ("beetle", "test_uq"), ("saf", "test_ua")


def test_gate_passes_on_both_wins_in_one_cell():
    passed, wins = compare.gate([case("rf", "base", BEETLE_UQ, 0.05, 0.01),
                                 case("rf", "ft", BEETLE_UQ, 0.03, 0.04),
                                 case("lf", "base", SAF_UA, 0.02, 0.5)])
    assert passed and wins == [{"adapter": "rf", "dataset": "beetle", "split": "test_uq"}]


@pytest.mark.parametrize("tests", [
    # wins in different cells
    [case("rf", "base", BEETLE_UQ, 0.05, 0.01), case("rf", "ft", SAF_UA, 0.05, 0.01)],
    # one win not significant after Holm
    [case("rf", "base", BEETLE_UQ, 0.05, 0.01), case("rf", "ft", BEETLE_UQ, 0.05, 0.05)],
    # significant but worse
    [case("rf", "base", BEETLE_UQ, 0.05, 0.01), case("rf", "ft", BEETLE_UQ, -0.05, 0.01)],
    # each baseline beaten by a different adapter
    [case("rf", "base", BEETLE_UQ, 0.05, 0.01), case("lf", "ft", BEETLE_UQ, 0.05, 0.01)],
])
def test_gate_fails(tests):
    assert compare.gate(tests) == (False, [])


def write_run(kaggle, dataset, label, split, rows_, stamp="20261105-000000", n_requested=0, reply="label"):
    prefix = "" if dataset == "scientsbank" else f"{dataset}-"
    name = f"{prefix}baseline-{label}-{split}-3way-{stamp}"
    (kaggle / "raw").mkdir(parents=True, exist_ok=True)
    (kaggle / f"{name}.json").write_text(json.dumps({
        "label": label, "split": split, "n_requested": n_requested, "reply": reply,
        "server": {"build_info": "b1-a3a1c47"},
        "metrics": {"n": len(rows_), "unparsed": sum(r["pred"] is None for r in rows_),
                    "errors": 0, "median_completion_tokens": 3}}))
    (kaggle / "raw" / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows_))


def test_latest_takes_the_newest_whole_split_run_of_that_label(tmp_path):
    r = [{"id": "a", "gold": "correct", "pred": "correct"}]
    write_run(tmp_path, "beetle", "e2b-kaggle-t4", "test_ua", r, stamp="20261030-000000")
    write_run(tmp_path, "beetle", "e2b-kaggle-t4", "test_ua", r, stamp="20261031-000000", n_requested=100)
    write_run(tmp_path, "beetle", "e2b-kaggle-t4-ft", "test_ua", r, stamp="20261101-000000")
    assert compare.latest(tmp_path, "beetle", "e2b-kaggle-t4", "test_ua").name == \
        "beetle-baseline-e2b-kaggle-t4-test_ua-3way-20261030-000000.json"
    with pytest.raises(SystemExit, match="no whole-split run"):
        compare.latest(tmp_path, "saf", "e2b-kaggle-t4", "test_ua")


def test_main_writes_the_table_and_the_gate(tmp_path):
    labels = {"base": "e2b-kaggle-t4", "ft": "e2b-kaggle-t4-ft",
              "rf": "e2b-kaggle-t4-ft-rf", "lf": "e2b-kaggle-t4-ft-lf"}
    golds = ["correct", "incorrect", "contradictory"] * 20
    for dataset, split in [*compare.CELLS, *compare.DESCRIBED]:
        for who, label in labels.items():
            # rf gets every answer right; the others miss every contradictory one.
            preds = golds if who == "rf" else [g if g != "contradictory" else "incorrect" for g in golds]
            write_run(tmp_path, dataset, label, split,
                      [{"id": str(i), "gold": g, "pred": p} for i, (g, p) in enumerate(zip(golds, preds))])
    compare.main(["--kaggle", str(tmp_path), "--base", labels["base"], "--ft", labels["ft"],
                  "--new", labels["rf"], labels["lf"]])
    [out] = tmp_path.glob("compare-reason-ft-*.json")
    result = json.loads(out.read_text())
    assert len(result["tests"]) == 16
    assert result["gate"]["passed"] is True
    # Beetle is tested 3-way (rf fixes 20); SAF 2-way, where contradictory -> incorrect was right.
    assert {(w["adapter"], w["dataset"]) for w in result["gate"]["wins"]} == {("e2b-kaggle-t4-ft-rf", "beetle")}
    recall = result["recall"]["beetle"]["test_ua"]
    assert recall["e2b-kaggle-t4-ft-rf"]["contradictory"] == 1.0
    assert recall["e2b-kaggle-t4"]["contradictory"] == 0.0
    assert {d["dataset"] for d in result["described"]} == {"scientsbank"}
