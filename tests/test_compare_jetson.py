import json

from edge_grader import compare_jetson as cj

GOLD = ["correct", "incorrect", "contradictory"] * 20  # 60 answers


def make(board, estimates, right):
    """One run folder per version and split, the first `right[v]` answers graded correctly,
    and a board-check JSON with that version's cost (J per answer = its index + 1)."""
    runs = []
    for i, v in enumerate(cj.VERSIONS):
        for split in cj.SPLITS:
            run = f"{v}-jetson-{split}-20261116-000000"
            name = f"baseline-x-{split}-3way-1"
            (board / run / "raw").mkdir(parents=True)
            rows = [{"id": f"a{k}", "gold": g, "pred": g if k < right[v] else
                     ("incorrect" if g != "incorrect" else "correct")} for k, g in enumerate(GOLD)]
            (board / run / "raw" / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            acc = right[v] / 60
            (board / run / f"{name}.json").write_text(json.dumps({"metrics": {
                "n": 60, "accuracy": acc, "ci95": {"accuracy": [acc - 0.1, acc + 0.1]}, "macro_f1": acc,
                "two_way": {"accuracy": acc}, "unparsed": 0}}))
            runs.append({"run": run, "board": "jetson", "split": split, "median_latency_s": 0.5,
                         "j_per_answer": i + 1.0, "mean_w": 9.0, "decode_ms_per_token": 25.0,
                         "mean_prompt_tokens_sent": 160.0})
    (estimates / "board-check-20261116-000000.json").write_text(json.dumps({"runs": runs}))


def test_rows_tests_and_energy_ratio(tmp_path, capsys):
    board, estimates = tmp_path / "board", tmp_path / "estimates"
    estimates.mkdir()
    right = dict.fromkeys(cj.VERSIONS, 45) | {"e2b": 30, "e4b": 40, "ft-e2b": 50}
    make(board, estimates, right)
    cj.main(board, estimates)
    out = json.loads(next(estimates.glob("compare-jetson-scientsbank-*.json")).read_text())
    assert len(out["rows"]) == 2 * len(cj.VERSIONS) and out["rows"][0]["split"] == "test_uq"
    tweaked = next(r for r in out["rows"] if r["version"] == "ft-e2b")
    assert tweaked["energy_vs_e2b_untouched"] == 2.0 and tweaked["decode_tok_s"] == 40.0
    assert len(out["tests"]) == 8 and all("p_holm" in t and "p_holm" in t["two_way"] for t in out["tests"])
    t = next(t for t in out["tests"] if (t["first"], t["second"], t["split"]) == ("e4b", "ft-e2b", "test_ud"))
    assert (t["only_first_right"], t["only_second_right"]) == (0, 10) and abs(t["diff"] - 10 / 60) < 1e-3
    assert out["seed_spread"]["test_uq"]["mean"] == 0.75
    assert "E2B tweaked (Oct 28 adapter) vs E4B untouched" in capsys.readouterr().out


def test_missing_version_is_refused(tmp_path):
    board, estimates = tmp_path / "board", tmp_path / "estimates"
    estimates.mkdir()
    make(board, estimates, dict.fromkeys(cj.VERSIONS, 45))
    f = next(estimates.glob("board-check-*.json"))
    d = json.loads(f.read_text())
    d["runs"] = [r for r in d["runs"] if not r["run"].startswith("ft-soup-")]
    f.write_text(json.dumps(d))
    try:
        cj.main(board, estimates)
    except SystemExit as e:
        assert "ft-soup-e2b" in str(e)
    else:
        raise AssertionError("expected SystemExit")
