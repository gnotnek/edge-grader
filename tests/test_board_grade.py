"""scripts/board_grade.sh in dry-run mode: argument checks, run-folder names, the runner it
would call, and the llama-server command (which must not change between benchmarks)."""

import os
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "board_grade.sh"
STAMP = r"\d{8}-\d{6}"


def dry(*args: str, board: str = "jetson", **env: str) -> subprocess.CompletedProcess:
    """Run the script with DRY_RUN=1 on a pretend board; BENCH etc. only if given."""
    e = {k: v for k, v in os.environ.items()
         if k not in {"BENCH", "INSTANCES", "EG_RUN_DIR", "EG_INNER", "MEASURE"}}
    e |= {"DRY_RUN": "1", "EG_BOARD": board, **env}
    return subprocess.run(["bash", str(SCRIPT), *args], env=e, capture_output=True, text=True,
                          check=False)


def fields(proc: subprocess.CompletedProcess) -> dict[str, str]:
    assert proc.returncode == 0, proc.stderr
    return dict(line.split(": ", 1) for line in proc.stdout.strip().splitlines())


def test_default_is_scientsbank_with_stage1_folder_name():
    f = fields(dry("e2b"))
    assert re.fullmatch(rf"{REPO}/results/board/e2b-jetson-test_ua-{STAMP}", f["run_dir"])
    assert f["runner"] == (
        "uv run edge-grader-baseline --base-url http://127.0.0.1:8080/v1 --label e2b-jetson "
        f"--split test_ua --n 0 --out {f['run_dir']}"
    )


def test_scientsbank_split_and_n_pass_through():
    f = fields(dry("e4b", "test_ud", "200", board="pi5"))
    assert re.fullmatch(rf"{REPO}/results/board/e4b-pi5-test_ud-{STAMP}", f["run_dir"])
    assert "--split test_ud --n 200" in f["runner"]


def test_gsm_symbolic_trimmed_on_the_pi():
    f = fields(dry("e4b", "symbolic", board="pi5", BENCH="gsm", INSTANCES="1"))
    assert re.fullmatch(rf"{REPO}/results/board/gsm-e4b-pi5-symbolic-i1-{STAMP}", f["run_dir"])
    assert f["runner"] == (
        "uv run edge-grader-gsm --base-url http://127.0.0.1:8080/v1 --label e4b-pi5 "
        f"--set symbolic --instances 1 --n 0 --out {f['run_dir']}"
    )


def test_gsm_defaults_to_gsm8k_and_ten_instances():
    f = fields(dry("e2b", BENCH="gsm"))
    assert re.fullmatch(rf"{REPO}/results/board/gsm-e2b-jetson-gsm8k-{STAMP}", f["run_dir"])
    assert "--set gsm8k --instances 10" in f["runner"]


def test_news():
    f = fields(dry("e2b", BENCH="news"))
    assert re.fullmatch(rf"{REPO}/results/board/news-e2b-jetson-test-{STAMP}", f["run_dir"])
    assert f["runner"] == (
        "uv run edge-grader-news --base-url http://127.0.0.1:8080/v1 --label e2b-jetson "
        f"--n 0 --out {f['run_dir']}"
    )


def test_server_flags_are_the_same_for_every_benchmark():
    for board, extra in (("jetson", "-ngl 99"), ("pi5", "-t 3")):
        servers = {fields(dry("e2b", board=board, BENCH=b))["server"]
                   for b in ("scientsbank", "gsm", "news")}
        assert len(servers) == 1
        (server,) = servers
        assert extra in server
        assert server.endswith("-c 8192 --host 127.0.0.1 --port 8080 -ngl "
                               f"{'99' if board == 'jetson' else '0'} "
                               "-rea off --reasoning-budget -1 --cache-ram 0")


def test_rejects_bad_arguments():
    cases = [
        ({"BENCH": "mmlu"}, ("e2b",), "unknown BENCH"),
        ({"BENCH": "gsm"}, ("e2b", "test_ua"), "unknown split/set"),
        ({}, ("e2b", "symbolic"), "unknown split/set"),
        ({"BENCH": "gsm", "INSTANCES": "0"}, ("e2b", "symbolic"), "INSTANCES"),
        ({"BENCH": "gsm", "INSTANCES": "x"}, ("e2b", "symbolic"), "INSTANCES"),
        ({}, ("e2b", "test_ua", "abc"), "n must be"),
        ({}, ("e9b",), "unknown model key"),
        ({"EG_RUN_DIR": "/tmp/elsewhere"}, ("e2b",), "outside"),
    ]
    for env, args, message in cases:
        p = dry(*args, **env)
        assert p.returncode != 0, (env, args)
        assert message in p.stderr, (env, args, p.stderr)
