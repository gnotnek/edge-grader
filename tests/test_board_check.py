import json
from datetime import datetime
from pathlib import Path

import pytest

from edge_grader.board_check import (
    compare,
    find_summary,
    kaggle_run,
    main,
    mcnemar,
    outcomes,
    parse_server_log,
    read_board_state,
    read_run,
    threads,
    window_energy,
)

REPO = Path(__file__).resolve().parents[1]
T0 = "2026-10-10T10:00:00+08:00"

LOG = """\
2026-10-04T07:54:38+0800 pi llama-server[0]: 0.25.103.060 I slot print_timing: id  3 | task 0 | prompt eval time =    2200.77 ms /   116 tokens (   18.97 ms per token,    52.71 tokens per second)
2026-10-04T07:54:38+0800 pi llama-server[0]: 0.25.103.067 I slot print_timing: id  3 | task 0 |        eval time =     163.74 ms /     2 tokens (  163.74 ms per token,     6.11 tokens per second)
2026-10-04T07:54:57+0800 pi llama-server[0]: 0.43.821.332 I slot print_timing: id  2 | task 35 | prompt processing, n_tokens =    229, progress = 0.98, t =   3.78 s / 60.65 tokens per second
2026-10-04T07:54:57+0800 pi llama-server[0]: 0.44.137.835 I slot print_timing: id  2 | task 35 | prompt eval time =    3941.66 ms /   233 tokens (   16.92 ms per token,    59.11 tokens per second)
2026-10-04T07:54:57+0800 pi llama-server[0]: 0.44.137.842 I slot print_timing: id  2 | task 35 |        eval time =     150.54 ms /     2 tokens (  150.54 ms per token,     6.64 tokens per second)
2026-10-04T07:54:58+0800 pi llama-server[0]: 0.45.000.000 I slot print_timing: id  2 | task 40 | prompt eval time =    1000.00 ms /   100 tokens (   10.00 ms per token,   100.00 tokens per second)
"""


def test_parse_server_log_pairs_prompt_and_decode_and_skips_progress_and_partial():
    reqs = parse_server_log(LOG)
    assert [r["task"] for r in reqs] == [0, 35]  # task 40 has no decode line yet
    assert reqs[0]["prompt_tokens"] == 116 and reqs[0]["prompt_ms"] == 2200.77
    assert reqs[1]["gen_tokens"] == 2 and reqs[1]["gen_ms"] == 150.54
    assert reqs[1]["end"] - reqs[0]["end"] == 19


def test_window_energy_integrates_only_inside_the_window():
    samples = [(0, 100.0), (1, 2.0), (2, 4.0), (3, 6.0), (4, 100.0)]
    assert window_energy(samples, 1, 3) == pytest.approx(3 + 5)


def test_mcnemar_counts_discordant_pairs():
    a = [True, True, True, False, True]
    b = [True, False, False, False, True]
    r = mcnemar(a, b)
    assert (r["only_first_right"], r["only_second_right"]) == (2, 0)
    assert r["p"] == 0.5
    assert mcnemar([True], [True])["p"] == 1.0


def test_threads_reads_only_the_t_flag():
    assert threads("llama-server -m x.gguf -t 3 -c 8192") == 3
    assert threads("llama-server -m x-t 9.gguf -c 8192 -ngl 0") is None
    assert threads("") is None


# --- phase 0: GSM and news runs ---------------------------------------------------------------

def epoch(offset_s: float) -> float:
    return datetime.fromisoformat(T0).timestamp() + offset_s


def stamp(offset_s: int) -> str:
    return f"2026-10-10T10:00:{offset_s:02d}+0800"


def make_run(root: Path, name: str, summary: dict, rows: list[dict],
             judged: list[dict] | None = None, state: str | None = None,
             tele_name: str = "edge-grader-x") -> Path:
    """A board run folder as board_grade.sh leaves it: one request per row, each 1 s long
    (500 ms prompt + 500 ms decode), ending 2 s apart from t=2; 5 W throughout."""
    run = root / name
    (run / "raw").mkdir(parents=True)
    stem = summary["_stem"]
    body = {k: v for k, v in summary.items() if k != "_stem"}
    (run / f"{stem}.json").write_text(json.dumps(body))
    (run / "raw" / f"{stem}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    log = []
    for i, r in enumerate(rows):
        end = stamp(2 + 2 * i)
        log.append(f"{end} b llama-server[0]: I slot print_timing: id  0 | task {i} | "
                   f"prompt eval time =     500.00 ms / {r.get('prompt_tokens') or 50:5d} tokens")
        log.append(f"{end} b llama-server[0]: I slot print_timing: id  0 | task {i} | "
                   f"       eval time =     500.00 ms /    10 tokens")
    (run / "server.log").write_text("\n".join(log) + "\n")
    tele = run / tele_name
    tele.mkdir()
    (tele / "meta.json").write_text(json.dumps({"server_args_after": "llama-server -m m.gguf -t 3"}))
    (tele / "summary.json").write_text(json.dumps({"power": {"idle_w": 2.0}}))
    lines = ["ts_epoch,power_w"] + [f"{epoch(s)},5.0" for s in range(2 * len(rows) + 3)]
    (tele / "telemetry.csv").write_text("\n".join(lines) + "\n")
    if judged is not None:
        acc = sum(g["grade"] == "correct" for g in judged) / len(judged)
        (run / f"{stem}.judged-qwen3-14b-q4_k_m.json").write_text(
            json.dumps({"metrics": {"n": len(judged), "accuracy": acc}}))
        (run / "raw" / f"{stem}.judged-qwen3-14b-q4_k_m.jsonl").write_text(
            "".join(json.dumps(g) + "\n" for g in judged))
    if state is not None:
        (run / "board-state.txt").write_text(state)
    return run


def gsm_summary(stem: str, label: str, set_: str, n: int, acc: float, instances=None) -> dict:
    gen = {"max_tokens": 1024} | ({"instances": instances} if instances else {})
    return {"_stem": stem, "stage": "topics", "benchmark": "gsm", "set": set_, "label": label,
            "generation": gen,
            "metrics": {"n": n, "accuracy": acc, "errors": 0, "median_latency_s": 1.0}}


def gsm_rows(results: dict[str, bool]) -> list[dict]:
    return [{"id": i, "flexible": "1" if ok else "2", "correct": ok,
             "prompt_tokens": 50 + 10 * k} for k, (i, ok) in enumerate(results.items())]


def write_kaggle(kdir: Path, stem: str, summary: dict, rows: list[dict]) -> None:
    (kdir / "raw").mkdir(parents=True, exist_ok=True)
    (kdir / f"{stem}.json").write_text(json.dumps(summary))
    (kdir / "raw" / f"{stem}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_read_run_gsm(tmp_path):
    rows = gsm_rows({"1/0": True, "2/0": False})
    rows.append({"id": "3/0", "error": "APIError()", "flexible": "[invalid]", "correct": False})
    run = make_run(tmp_path, "gsm-e2b-jetson-symbolic-i10-20261010-100000",
                   gsm_summary("gsm-symbolic-e2b-jetson-20261010-020000", "e2b-jetson",
                               "symbolic", 3, 0.3333, instances=10), rows)
    r = read_run(run)
    assert (r["benchmark"], r["model"], r["board"], r["split"], r["instances"]) == (
        "gsm", "e2b", "jetson", "symbolic", 10)
    assert r["accuracy"] == 0.3333 and r["macro_f1"] is None
    assert r["mean_prompt_tokens_sent"] == 55.0  # the errored row has no prompt_tokens
    # window: first request starts at t=1, last ends at t=6 -> 5 s at 5 W = 25 J over 3 questions
    assert r["grading_window_s"] == 5.0
    assert r["j_per_answer"] == pytest.approx(25 / 3, abs=0.01)
    assert r["board_state"] is None


def test_news_accuracy_waits_for_the_judge(tmp_path):
    summary = {"_stem": "news-baseline-e2b-pi5-20261010-020000", "stage": "baseline",
               "benchmark": "news", "label": "e2b-pi5",
               "metrics": {"n": 2, "answered": 1, "errors": 0, "median_latency_s": 1.0}}
    rows = [{"id": "news-000", "pred": None, "prompt_tokens": 60},
            {"id": "news-001", "pred": "Paris", "prompt_tokens": 70}]
    run = make_run(tmp_path, "news-e2b-pi5-test-20261010-100000", summary, rows)
    r = read_run(run)
    assert r["accuracy"] is None and r["split"] == "test"
    assert outcomes(find_summary(run)) is None

    judged = [{"id": "news-000", "grade": "not_attempted"}, {"id": "news-001", "grade": "correct"}]
    run2 = make_run(tmp_path, "news-e2b-pi5-test-20261010-110000", summary, rows, judged=judged)
    assert read_run(run2)["accuracy"] == 0.5
    assert outcomes(find_summary(run2)) == {
        "news-000": {"answer": "not_attempted", "right": False},
        "news-001": {"answer": "correct", "right": True},
    }


def test_compare_pairs_on_shared_ids():
    a = {"x": {"answer": 1, "right": True}, "y": {"answer": 2, "right": False}}
    b = {"x": {"answer": 1, "right": True}, "y": {"answer": 3, "right": True},
         "z": {"answer": 4, "right": False}}
    c = compare(a, b)
    assert (c["n"], c["same_grade"], c["first_accuracy"], c["second_accuracy"]) == (2, 1, 0.5, 1.0)
    assert (c["mcnemar"]["only_first_right"], c["mcnemar"]["only_second_right"]) == (0, 1)


def test_find_summary_ignores_judged_files_and_wants_exactly_one(tmp_path):
    (tmp_path / "news-baseline-e2b-pi5-1.json").write_text("{}")
    (tmp_path / "news-baseline-e2b-pi5-1.judged-qwen3-14b-q4_k_m.json").write_text("{}")
    assert find_summary(tmp_path).name == "news-baseline-e2b-pi5-1.json"
    (tmp_path / "gsm-gsm8k-e2b-pi5-1.json").write_text("{}")
    with pytest.raises(SystemExit):
        find_summary(tmp_path)


def test_kaggle_run_finds_each_benchmark(tmp_path):
    names = [
        "baseline-e2b-kaggle-t4-test_uq-3way-20261005-1.json",
        "gsm-gsm8k-e2b-kaggle-t4-20261008-1.json",
        "gsm-symbolic-e2b-kaggle-t4-20261008-1.json",
        "gsm-tiny-e2b-kaggle-t4-20261008-1.json",
        "news-baseline-e2b-kaggle-t4-20261007-1.json",
        "news-baseline-e2b-kaggle-t4-20261007-1.judged-qwen3-14b-q4_k_m.json",
        "news-rag-e2b-kaggle-t4-en-k3-20261008-1.json",
    ]
    for n in names:
        (tmp_path / n).write_text(json.dumps({"metrics": {"n": 733}}))
    d = str(tmp_path)
    assert kaggle_run(d, "scientsbank", "e2b", "test_uq", 733).name == names[0]
    assert kaggle_run(d, "scientsbank", "e2b", "test_uq", 100) is None
    assert kaggle_run(d, "gsm", "e2b", "gsm8k", 100).name == names[1]
    assert kaggle_run(d, "gsm", "e2b", "symbolic", 100).name == names[2]  # n may differ (Pi trimmed)
    assert kaggle_run(d, "news", "e2b", "test", 200).name == names[4]
    assert kaggle_run(d, "gsm", "e4b", "gsm8k", 100) is None


def test_read_board_state(tmp_path):
    assert read_board_state(tmp_path / "missing.txt") is None
    p = tmp_path / "board-state.txt"
    p.write_text("time: x\ncpu0: governor schedutil min 729600 max 1728000 cur 1113600 (kHz)\n")
    assert read_board_state(p) == {"governor": "schedutil", "cpu_min_khz": 729600,
                                   "cpu_max_khz": 1728000, "cpu_clock_locked": False}
    p.write_text("cpu0: governor schedutil min 1728000 max 1728000 cur 1728000 (kHz)\n")
    assert read_board_state(p)["cpu_clock_locked"] is True
    p.write_text("kernel: 6.1\n")
    assert read_board_state(p) == {}


def test_main_pairs_trimmed_symbolic_on_shared_ids(tmp_path):
    kdir = tmp_path / "kaggle"
    kaggle_rows = gsm_rows({"1/0": True, "1/1": True, "2/0": False, "2/1": True})
    write_kaggle(kdir, "gsm-symbolic-e2b-kaggle-t4-20261008-1",
                 gsm_summary("x", "e2b-kaggle-t4", "symbolic", 4, 0.75, instances=10), kaggle_rows)
    write_kaggle(kdir, "news-baseline-e2b-kaggle-t4-1",  # exists, so the unjudged board run is "pending"
                 {"benchmark": "news", "label": "e2b-kaggle-t4", "metrics": {"n": 1}},
                 [{"id": "news-000", "pred": None}])
    runs = tmp_path / "board"
    pi = make_run(runs, "gsm-e2b-pi5-symbolic-i1-20261010-100000",
                  gsm_summary("gsm-symbolic-e2b-pi5-1", "e2b-pi5", "symbolic", 2, 0.5, instances=1),
                  gsm_rows({"1/0": True, "2/0": True}),
                  state="cpu0: governor ondemand min 1500000 max 2400000 cur 2400000 (kHz)\n")
    jet = make_run(runs, "gsm-e2b-jetson-symbolic-i10-20261010-100000",
                   gsm_summary("gsm-symbolic-e2b-jetson-1", "e2b-jetson", "symbolic", 4, 0.75,
                               instances=10), kaggle_rows)
    news = make_run(runs, "news-e2b-pi5-test-20261010-100000",
                    {"_stem": "news-baseline-e2b-pi5-1", "benchmark": "news", "label": "e2b-pi5",
                     "metrics": {"n": 1, "errors": 0, "median_latency_s": 1.0}},
                    [{"id": "news-000", "pred": None, "prompt_tokens": 60}])
    est = tmp_path / "est.json"
    est.write_text(json.dumps({"estimates": []}))
    out = tmp_path / "out"
    main([str(pi), str(jet), str(news), "--estimate", str(est), "--kaggle", str(kdir),
          "--out", str(out)])
    data = json.loads(next(out.glob("board-check-*.json")).read_text())
    by = {r["run"]: r for r in data["runs"]}
    pi_vs = by[pi.name]["vs_kaggle"]
    assert (pi_vs["n"], pi_vs["kaggle_accuracy"], pi_vs["board_accuracy_paired"]) == (2, 0.5, 1.0)
    assert by[pi.name]["board_state"]["cpu_clock_locked"] is False
    assert by[news.name]["accuracy"] is None
    assert by[news.name]["vs_kaggle"]["pending"].startswith("judge")
    (pair,) = data["board_pairs"]
    assert (pair["first"], pair["second"], pair["n"]) == (pi.name, jet.name, 2)


STAGE1 = REPO / "results/board/e2b-jetson-test_ua-20260930-224702"


# The boards keep their own partial results/board/ (no telemetry, no results/kaggle/); the
# committed record is complete only in the repo checkout.
@pytest.mark.skipif(not (REPO / "results/kaggle").exists(), reason="the committed results/ isn't on the boards")
def test_stage1_jetson_run_reads_as_before():
    """The committed board-check-20261005-042603.json values for this run, unchanged."""
    r = read_run(STAGE1)
    assert (r["benchmark"], r["split"], r["n"], r["accuracy"]) == ("scientsbank", "test_ua", 540, 0.5519)
    assert (r["j_per_answer"], r["median_latency_s"], r["mean_prompt_tokens_processed"],
            r["decode_ms_per_token"]) == (3.95, 0.42, 165.2, 24.8)
    k = kaggle_run(str(REPO / "results/kaggle"), "scientsbank", "e2b", "test_ua", 540)
    assert k.name == "baseline-e2b-kaggle-t4-test_ua-3way-20260929-102126.json"
    c = compare(outcomes(find_summary(STAGE1)), outcomes(k))
    assert (c["n"], c["same_grade"], c["second_accuracy"]) == (540, 507, 0.5519)
    assert (c["mcnemar"]["only_first_right"], c["mcnemar"]["only_second_right"]) == (15, 15)


def test_reads_the_telemetry_folder_board_grade_writes_today(tmp_path):
    """run_measured.sh names it telemetry-<stamp>/; stage-1 runs have edge-grader-*/."""
    run = make_run(tmp_path, "gsm-e2b-pi5-gsm8k-20261010-100000",
                   gsm_summary("gsm-gsm8k-e2b-pi5-1", "e2b-pi5", "gsm8k", 2, 0.5),
                   gsm_rows({"1": True, "2": False}), tele_name="telemetry-20261010-100000")
    r = read_run(run)
    assert r["threads"] == 3 and r["j_per_answer"] == pytest.approx(7.5)
