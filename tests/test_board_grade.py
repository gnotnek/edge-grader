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
         if k not in {"BENCH", "INSTANCES", "THINKING", "VARIANT", "RAG", "LG_ENGINE", "EG_RUN_DIR", "EG_INNER",
                       "MEASURE", "LORA", "REPLY"}}
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


def test_mmlupro_defaults_to_all_three_subsets():
    f = fields(dry("e2b", BENCH="mmlupro"))
    assert re.fullmatch(rf"{REPO}/results/board/mmlupro-e2b-jetson-all-{STAMP}", f["run_dir"])
    assert f["runner"] == (
        "uv run edge-grader-mmlupro --base-url http://127.0.0.1:8080/v1 --label e2b-jetson "
        f"--subset s1 s2 s3 --n 0 --out {f['run_dir']}"
    )


def test_mmlupro_one_subset_and_n():
    f = fields(dry("e4b", "s2", "5", BENCH="mmlupro"))
    assert re.fullmatch(rf"{REPO}/results/board/mmlupro-e4b-jetson-s2-{STAMP}", f["run_dir"])
    assert "--subset s2 --n 5 " in f["runner"]


def test_asap2():
    f = fields(dry("e4b", BENCH="asap2"))
    assert re.fullmatch(rf"{REPO}/results/board/asap2-e4b-jetson-sample-{STAMP}", f["run_dir"])
    assert f["runner"] == (
        "uv run edge-grader-asap run --base-url http://127.0.0.1:8080/v1 --label e4b-jetson "
        f"--n 0 --out {f['run_dir']}"
    )
    rag = dry("e2b", BENCH="asap2", RAG="results/rag/retrieval-asap2-en.json")
    assert rag.returncode != 0 and "RAG isn't a condition" in rag.stderr


def test_server_flags_are_the_same_for_every_benchmark():
    for board, extra in (("jetson", "-ngl 99"), ("pi5", "-t 3")):
        servers = {fields(dry("e2b", board=board, BENCH=b))["server"]
                   for b in ("scientsbank", "gsm", "news", "mmlupro", "asap2")}
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
        ({"BENCH": "mmlupro"}, ("e2b", "s4"), "unknown split/set"),
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


def test_thinking_on_changes_only_the_reasoning_flags_and_the_name():
    off = fields(dry("e2b", board="pi5"))
    on = fields(dry("e2b", board="pi5", THINKING="on"))
    assert on["server"] == off["server"].replace(
        "-rea off --reasoning-budget -1", "-rea on --reasoning-budget 320 --reasoning-format none")
    assert re.fullmatch(rf"{REPO}/results/board/think-e2b-pi5-test_ua-{STAMP}", on["run_dir"])
    assert on["runner"].endswith(f"--out {on['run_dir']} --thinking")


def test_thinking_is_scientsbank_only():
    p = dry("e2b", BENCH="gsm", THINKING="on")
    assert p.returncode != 0 and "THINKING" in p.stderr


def test_swafull_adds_only_that_flag():
    base = fields(dry("e2b"))
    swa = fields(dry("e2b", VARIANT="swafull"))
    assert swa["server"] == base["server"] + " --swa-full"
    assert re.fullmatch(rf"{REPO}/results/board/swafull-e2b-jetson-test_ua-{STAMP}", swa["run_dir"])
    assert swa["runner"] == base["runner"].replace(base["run_dir"], swa["run_dir"])


def test_little_gemma_variants_use_the_shim():
    lg = fields(dry("e4b", VARIANT="lg"))
    assert "lg_openai_shim.py --engine " in lg["server"]
    assert lg["server"].endswith("--thinking off --think -1 --port 8080")
    reuse = fields(dry("e4b", VARIANT="lg-reuse"))
    assert reuse["server"] == lg["server"] + " --engine-flags=-reuse"
    assert re.fullmatch(rf"{REPO}/results/board/lg-reuse-e4b-jetson-test_ua-{STAMP}", reuse["run_dir"])


def test_variants_are_jetson_scientsbank_only():
    for env, board in (({"VARIANT": "lg"}, "pi5"), ({"VARIANT": "lg", "BENCH": "gsm"}, "jetson"),
                       ({"VARIANT": "fast"}, "jetson")):
        p = dry("e2b", board=board, **env)
        assert p.returncode != 0 and "VARIANT" in p.stderr, (env, board, p.stderr)


def test_rag_adds_the_retrieval_file_and_prefix():
    f = fields(dry("e2b", VARIANT="lg-reuse",
                   RAG="results/rag/retrieval-scientsbank-en.bge-small-en-k5.json"))
    assert re.fullmatch(rf"{REPO}/results/board/rag-lg-reuse-e2b-jetson-test_ua-{STAMP}", f["run_dir"])
    assert f["runner"].endswith(
        f"--out {f['run_dir']} --rag results/rag/retrieval-scientsbank-en.bge-small-en-k5.json --k 3")


def test_rag_for_gsm_and_news_names_the_folder_and_passes_the_file():
    gsm = fields(dry("e2b", "symbolic", BENCH="gsm",
                     RAG="results/rag/retrieval-gsm-en.bge-small-en-k5.json"))
    assert re.fullmatch(rf"{REPO}/results/board/gsm-rag-e2b-jetson-symbolic-i10-{STAMP}", gsm["run_dir"])
    assert gsm["runner"].endswith(
        f"--out {gsm['run_dir']} --rag results/rag/retrieval-gsm-en.bge-small-en-k5.json --k 3")
    news = fields(dry("e4b", BENCH="news", RAG="results/rag/retrieval-news-en.bge-small-en-k5.json"))
    assert re.fullmatch(rf"{REPO}/results/board/news-rag-e4b-jetson-test-{STAMP}", news["run_dir"])
    assert news["runner"].endswith("--rag results/rag/retrieval-news-en.bge-small-en-k5.json --k 3")
    # The server is the baseline's: RAG changes the prompt only.
    assert gsm["server"] == fields(dry("e2b", "symbolic", BENCH="gsm"))["server"]


def test_rag_for_mmlupro_names_the_folder_and_passes_the_file():
    rag = "results/rag/retrieval-mmlupro-en.bge-small-en-k5.json"
    f = fields(dry("e4b", BENCH="mmlupro", RAG=rag))
    assert re.fullmatch(rf"{REPO}/results/board/mmlupro-rag-e4b-jetson-all-{STAMP}", f["run_dir"])
    assert f["runner"].endswith(f"--out {f['run_dir']} --rag {rag} --k 3")
    assert f["server"] == fields(dry("e4b", BENCH="mmlupro"))["server"]


def test_rag_file_must_be_the_benchmarks_own():
    p = dry("e2b", BENCH="news", RAG="results/rag/retrieval-gsm-en.bge-small-en-k5.json")
    assert p.returncode != 0 and "not a news retrieval file" in p.stderr


ADAPTER = "/m/adapters/e2b-ft.gguf"


def test_lora_adds_only_that_flag_for_every_benchmark():
    for bench, split in (("scientsbank", "test_uq"), ("gsm", "gsm8k"), ("news", "test"), ("mmlupro", "all")):
        base = fields(dry("e2b", split, BENCH=bench))
        ft = fields(dry("e2b", split, BENCH=bench, LORA=ADAPTER))
        assert ft["server"] == base["server"] + f" --lora {ADAPTER}", bench
        assert ft["runner"] == base["runner"].replace(base["run_dir"], ft["run_dir"]), bench


def test_lora_folder_names():
    for env, args, name in (({}, ("e2b",), "ft-e2b-jetson-test_ua"),
                            ({"BENCH": "gsm", "INSTANCES": "1"}, ("e2b", "symbolic"),
                             "gsm-ft-e2b-jetson-symbolic-i1"),
                            ({"BENCH": "mmlupro"}, ("e2b",), "mmlupro-ft-e2b-jetson-all")):
        f = fields(dry(*args, LORA=ADAPTER, **env))
        assert re.fullmatch(rf"{REPO}/results/board/{name}-{STAMP}", f["run_dir"]), f["run_dir"]


def test_lora_refused_with_little_gemma():
    for variant in ("lg", "lg-reuse"):
        p = dry("e2b", VARIANT=variant, LORA=ADAPTER)
        assert p.returncode != 0 and "LORA" in p.stderr, (variant, p.stderr)


def test_lora_must_be_a_gguf():
    p = dry("e2b", LORA="/m/a.safetensors")
    assert p.returncode != 0 and "LORA" in p.stderr, p.stderr


def test_reply_passes_through_with_its_adapter():
    base = fields(dry("e2b", "test_uq", LORA=ADAPTER))
    for reply, prefix in (("label-first", "ft-lf"), ("reason-first", "ft-rf")):
        f = fields(dry("e2b", "test_uq", LORA=ADAPTER, REPLY=reply))
        assert re.fullmatch(rf"{REPO}/results/board/{prefix}-e2b-jetson-test_uq-{STAMP}", f["run_dir"]), f["run_dir"]
        assert f["server"] == base["server"]
        assert f["runner"] == base["runner"].replace(base["run_dir"], f["run_dir"]) + f" --reply {reply}"
    assert fields(dry("e2b", "test_uq", LORA=ADAPTER, REPLY="label")).keys() == base.keys()


def test_reply_refused_outside_its_condition():
    for env in ({"REPLY": "label-first"},                                  # no adapter
                {"REPLY": "reason-first", "LORA": ADAPTER, "BENCH": "gsm"},
                {"REPLY": "reason-first", "LORA": ADAPTER, "THINKING": "on"},
                {"REPLY": "reasons", "LORA": ADAPTER}):
        p = dry("e2b", **env)
        assert p.returncode != 0 and "REPLY" in p.stderr, (env, p.stderr)
