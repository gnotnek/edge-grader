"""scripts/board_queue.py, the boards' own job queue, against a fake board_grade.sh: order,
recorded results, busy boards, one worker at a time, stop, cancel/retry and the phase-0 preset."""

import fcntl
import importlib.util
import json
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("board_queue", REPO / "scripts" / "board_queue.py")
bq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bq)

# The fake: records each real call; a dry run fails for the model "bad"; BUSY_TIMES makes the
# first n real calls refuse as busy; FAIL makes a real call exit 3.
FAKE = r"""#!/usr/bin/env bash
if [ "${DRY_RUN:-0}" = 1 ]; then
  [ "$1" = bad ] && { echo "unknown model key: bad" >&2; exit 1; }
  echo "run_dir: /x/dry"; exit 0
fi
echo "$* BENCH=${BENCH:-} INSTANCES=${INSTANCES:-} DRY=${DRY_RUN:-} EGB=${EG_BOARD:-}" >> "$CALLS"
n=$(cat "$COUNT" 2>/dev/null || echo 0); echo $((n + 1)) > "$COUNT"
if [ "$n" -lt "${BUSY_TIMES:-0}" ]; then
  echo "refusing: the board is busy (lm_eval); wait for that run to finish" >&2; exit 1
fi
[ -n "${FAIL:-}" ] && exit 3
echo "Thu $(date) $BENCH finished rc=0; everything for this run is in /results/board/$1-run"
"""


@pytest.fixture
def q(tmp_path, monkeypatch):
    fake = tmp_path / "board_grade.sh"
    fake.write_text(FAKE)
    fake.chmod(0o755)
    monkeypatch.setenv("EG_QUEUE_DIR", str(tmp_path / "queue"))
    monkeypatch.setenv("EG_BOARD_GRADE", str(fake))
    monkeypatch.setenv("EG_BOARD", "jetson")
    monkeypatch.setenv("EG_QUEUE_GAP", "0")
    monkeypatch.setenv("EG_QUEUE_BUSY_WAIT", "0")
    monkeypatch.setenv("CALLS", str(tmp_path / "calls.txt"))
    monkeypatch.setenv("COUNT", str(tmp_path / "count.txt"))
    for k in ("BENCH", "INSTANCES", "DRY_RUN", "BUSY_TIMES", "FAIL"):
        monkeypatch.delenv(k, raising=False)
    return tmp_path


def jobs(q):
    return json.loads((q / "queue" / "jobs.json").read_text())


def calls(q):
    p = q / "calls.txt"
    return p.read_text().splitlines() if p.exists() else []


def test_add_keeps_order_and_parses_env_and_args(q):
    bq.main(["add", "BENCH=gsm", "INSTANCES=1", "--", "e2b", "symbolic"])
    bq.main(["add", "--", "e4b", "test_uq"])
    js = jobs(q)
    assert [j["id"] for j in js] == [1, 2]
    assert js[0]["env"] == {"BENCH": "gsm", "INSTANCES": "1"} and js[0]["args"] == ["e2b", "symbolic"]
    assert js[1]["env"] == {} and all(j["state"] == "pending" for j in js)


def test_add_refuses_a_job_the_script_would_refuse(q):
    with pytest.raises(SystemExit, match="unknown model key"):
        bq.main(["add", "--", "bad"])
    assert not (q / "queue" / "jobs.json").exists() or jobs(q) == []


def test_add_refuses_env_the_queue_does_not_pass(q):
    with pytest.raises(SystemExit, match="EG_RUN_DIR"):
        bq.main(["add", "EG_RUN_DIR=/tmp/x", "--", "e2b"])


def test_phase0_preset_per_board(q, monkeypatch):
    bq.main(["add", "--preset", "phase0"])
    js = jobs(q)
    assert len(js) == 10
    assert [(j["args"], j["env"].get("BENCH")) for j in js[:5]] == [
        (["e2b", "test_uq"], "scientsbank"), (["e2b", "test_ud"], "scientsbank"),
        (["e2b", "gsm8k"], "gsm"), (["e2b", "symbolic"], "gsm"), (["e2b", "test"], "news")]
    assert {j["env"]["INSTANCES"] for j in js} == {"10"}
    assert [j["args"][0] for j in js[5:]] == ["e4b"] * 5

    monkeypatch.setenv("EG_QUEUE_DIR", str(q / "queue-pi"))
    monkeypatch.setenv("EG_BOARD", "pi5")
    bq.main(["add", "--preset", "phase0"])
    pi = json.loads((q / "queue-pi" / "jobs.json").read_text())
    assert {j["env"]["INSTANCES"] for j in pi} == {"1"}


def test_rag_preset(q):
    bq.main(["add", "--preset", "rag"])
    js = jobs(q)
    assert len(js) == 16
    assert [j["args"] for j in js[:2]] == [["e2b", "test_ua"], ["e4b", "test_ua"]]
    assert [(j["env"]["BENCH"], j["args"][1]) for j in js[::2]] == [
        ("scientsbank", "test_ua"), ("scientsbank", "test_uq"), ("scientsbank", "test_ud"),
        ("news", "test"), ("gsm", "gsm8k"), ("gsm", "symbolic"), ("mmlupro", "all"), ("mmlupro", "all")]
    # MMLU-Pro has no board baseline from this project's runner (agentic-edge's is lm-eval),
    # so the preset runs one before its RAG-en run: baseline e2b, e4b, then RAG e2b, e4b.
    mmlu = js[12:]
    assert [(j["args"], "RAG" in j["env"]) for j in mmlu] == [
        (["e2b", "all"], False), (["e4b", "all"], False), (["e2b", "all"], True), (["e4b", "all"], True)]
    for j in js[:12] + mmlu[2:]:
        assert j["env"]["RAG"] == f"results/rag/retrieval-{j['env']['BENCH']}-en.bge-small-en-k5.json"
    assert {j["env"]["INSTANCES"] for j in js[:12]} == {"10"}


def test_run_goes_in_order_and_records_results(q, monkeypatch):
    bq.main(["add", "BENCH=gsm", "--", "e2b", "gsm8k"])
    bq.main(["add", "--", "e4b", "test_uq"])
    bq.main(["run"])
    js = jobs(q)
    assert [j["state"] for j in js] == ["done", "done"]
    assert js[0]["rc"] == 0 and js[0]["run_dir"] == "/results/board/e2b-run"
    assert calls(q)[0].startswith("e2b gsm8k BENCH=gsm")
    assert calls(q)[1].startswith("e4b test_uq BENCH= ")
    assert (q / "queue" / "logs" / "job-1.log").exists()


def test_real_runs_get_no_dry_run_or_pretend_board(q):
    bq.main(["add", "--", "e2b"])
    bq.main(["run"])
    assert "DRY= EGB=" in calls(q)[0]


def test_a_failed_job_is_recorded_and_the_queue_moves_on(q, monkeypatch):
    bq.main(["add", "--", "e2b"])
    bq.main(["add", "--", "e4b"])
    monkeypatch.setenv("FAIL", "1")
    bq.main(["run"])
    assert [(j["state"], j["rc"]) for j in jobs(q)] == [("failed", 3), ("failed", 3)]


def test_a_busy_board_puts_the_job_back_and_tries_again(q, monkeypatch):
    bq.main(["add", "--", "e2b"])
    monkeypatch.setenv("BUSY_TIMES", "2")
    bq.main(["run"])
    (j,) = jobs(q)
    assert j["state"] == "done" and j["busy_waits"] == 2
    assert len(calls(q)) == 3


def test_only_one_worker_at_a_time(q):
    bq.main(["add", "--", "e2b"])
    (q / "queue").mkdir(exist_ok=True)
    with open(q / "queue" / "worker.lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(SystemExit, match="already running"):
            bq.main(["run"])
    assert jobs(q)[0]["state"] == "pending"


def test_stop_finishes_the_current_job_then_exits(q, monkeypatch):
    for m in ("e2b", "e4b"):
        bq.main(["add", "--", m])
    orig = bq.run_job

    def run_then_stop(job, qdir):
        bq.main(["stop"])  # as if typed while job 1 runs
        return orig(job, qdir)

    monkeypatch.setattr(bq, "run_job", run_then_stop)
    bq.main(["run"])
    assert [j["state"] for j in jobs(q)] == ["done", "pending"]
    assert not (q / "queue" / "stop").exists()


def test_cancel_and_retry(q):
    for m in ("e2b", "e4b"):
        bq.main(["add", "--", m])
    bq.main(["cancel", "2"])
    assert jobs(q)[1]["state"] == "cancelled"
    with pytest.raises(SystemExit, match="pending"):
        bq.main(["cancel", "2"])
    bq.main(["retry", "2"])
    assert jobs(q)[1]["state"] == "pending"
    with pytest.raises(SystemExit, match="no job 9"):
        bq.main(["retry", "9"])


def test_a_job_left_running_by_a_dead_worker_is_marked_interrupted(q):
    bq.main(["add", "--", "e2b"])
    js = jobs(q)
    js[0]["state"] = "running"
    (q / "queue" / "jobs.json").write_text(json.dumps(js))
    bq.main(["run"])
    assert jobs(q)[0]["state"] == "interrupted"
    assert calls(q) == []


def test_status_prints_every_job(q, capsys):
    bq.main(["add", "BENCH=news", "--", "e2b", "test"])
    capsys.readouterr()
    bq.main(["status"])
    out = capsys.readouterr().out
    assert "pending" in out and "BENCH=news" in out and "e2b test" in out


def test_env_of_the_worker_does_not_leak_into_jobs(q, monkeypatch):
    monkeypatch.setenv("BENCH", "news")  # left over in the shell that started the worker
    bq.main(["add", "--", "e2b"])
    bq.main(["run"])
    assert "BENCH= " in calls(q)[0]
    assert os.environ["BENCH"] == "news"
