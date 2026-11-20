import json

from edge_grader import compare_thinking as ct

GOLD = ["correct", "incorrect", "contradictory"] * 20  # 60 answers


def make(board, estimates, right, thought):
    """A run folder per version and split, the first `right[v]` answers right, the first `thought[v]`
    with a thought block; plus an older off run that must be ignored, and two board-check JSONs."""
    checks = [[], []]
    versions = [(off, "20261114") for off in ct.VERSIONS] + [(on, "20261117") for _, on in ct.VERSIONS.values()]
    for v, day in versions + [("e2b", "20261021")]:
        for split in ct.SPLITS:
            run = f"{v}-jetson-{split}-{day}-000000"
            name = f"baseline-x-{split}-3way-1"
            (board / run / "raw").mkdir(parents=True)
            n_right = right[v] if day != "20261021" else 0
            rows = [{"id": f"a{k}", "gold": g, "pred": g if k < n_right else ("incorrect" if g != "incorrect"
                     else "correct"), "raw": ("<|channel>thought ...<channel|>" if k < thought.get(v, 0) else "")
                     + "x", "finish": "stop", "completion_tokens": 3} for k, g in enumerate(GOLD)]
            (board / run / "raw" / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            acc = n_right / 60
            (board / run / f"{name}.json").write_text(json.dumps({"metrics": {
                "n": 60, "accuracy": acc, "ci95": {"accuracy": [acc - 0.1, acc + 0.1]}, "macro_f1": acc,
                "unparsed": 0}}))
            (board / run / "board-state.txt").write_text(
                "devfreq 17000000.gpu: min 306000000 max 918000000 cur 306000000 (Hz)\n")
            checks[day == "20261117"].append({"run": run, "board": "jetson", "split": split,
                                              "median_latency_s": 0.5, "j_per_answer": 5.0})
    for i, runs in enumerate(checks):
        (estimates / f"board-check-2026112{i}-000000.json").write_text(json.dumps({"runs": runs}))


def test_pairs_tests_and_thought_counts(tmp_path, capsys):
    board, estimates = tmp_path / "board", tmp_path / "estimates"
    estimates.mkdir()
    right = dict.fromkeys(list(ct.VERSIONS) + [on for _, on in ct.VERSIONS.values()], 45) | {"e2b": 30}
    make(board, estimates, right, {"think-e2b": 60, "ft-soup-think-e2b": 7})
    ct.main(board, estimates)
    out = json.loads(next(estimates.glob("compare-thinking-scientsbank-*.json")).read_text())
    assert len(out["rows"]) == 24 and len(out["tests"]) == 22
    row = {(r["version"], r["split"]): r for r in out["rows"]}
    assert row["e2b", "test_uq"]["run"].endswith("20261114-000000")  # the October run is ignored
    assert row["ft-soup-think-e2b", "test_ud"]["thought_blocks"] == 7 and row["think-e2b", "test_uq"]["thinking"]
    assert row["ft-soup-think-e2b", "test_ud"]["name"] == "Soup (s0-s4)" and row["e2b", "test_ud"]["gpu_max_mhz"] == 918
    t = next(t for t in out["tests"] if (t["first"], t["second"], t["split"]) == ("e2b", "think-e2b", "test_ud"))
    assert (t["only_first_right"], t["only_second_right"]) == (0, 15) and "p_holm" in t
    assert "| E2B untouched | test_uq | 50.0% | 75.0% | 60/60 |" in capsys.readouterr().out


def test_missing_run_is_refused(tmp_path):
    board, estimates = tmp_path / "board", tmp_path / "estimates"
    estimates.mkdir()
    make(board, estimates, dict.fromkeys(list(ct.VERSIONS) + [on for _, on in ct.VERSIONS.values()], 45), {})
    f = estimates / "board-check-20261121-000000.json"
    d = json.loads(f.read_text())
    d["runs"] = [r for r in d["runs"] if not r["run"].startswith("ft-rf-think-")]
    f.write_text(json.dumps(d))
    try:
        ct.main(board, estimates)
    except SystemExit as e:
        assert "ft-rf-think-e2b" in str(e)
    else:
        raise AssertionError("expected SystemExit")
