"""scripts/board_grade.sh in dry-run mode: argument checks, run-folder names, the runner it
would call, and the llama-server command (which only the condition under test may change)."""

import os
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "board_grade.sh"
STAMP = r"\d{8}-\d{6}"


def dry(*args: str, board: str = "jetson", **env: str) -> subprocess.CompletedProcess:
    """Run the script with DRY_RUN=1 on a pretend board; THINKING etc. only if given."""
    e = {k: v for k, v in os.environ.items()
         if k not in {"THINKING", "VARIANT", "RAG", "LG_ENGINE", "EG_RUN_DIR", "EG_INNER",
                       "MEASURE", "LORA", "REPLY", "EXAMPLES", "PROMPT_SCHEME"}}
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














def test_server_flags_are_agentic_edges_baseline():
    for board, extra in (("jetson", "-ngl 99"), ("pi5", "-t 3")):
        server = fields(dry("e2b", board=board))["server"]
        assert extra in server
        assert server.endswith("-c 8192 --host 127.0.0.1 --port 8080 -ngl "
                               f"{'99' if board == 'jetson' else '0'} "
                               "-rea off --reasoning-budget -1 --cache-ram 0")


def test_rejects_bad_arguments():
    cases = [
        ({}, ("e2b", "symbolic"), "unknown split"),
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


def test_variants_are_jetson_only():
    for env, board in (({"VARIANT": "lg"}, "pi5"), ({"VARIANT": "fast"}, "jetson")):
        p = dry("e2b", board=board, **env)
        assert p.returncode != 0 and "VARIANT" in p.stderr, (env, board, p.stderr)


def test_rag_adds_the_retrieval_file_and_prefix():
    f = fields(dry("e2b", VARIANT="lg-reuse",
                   RAG="results/rag/retrieval-scientsbank-en.bge-small-en-k5.json"))
    assert re.fullmatch(rf"{REPO}/results/board/rag-lg-reuse-e2b-jetson-test_ua-{STAMP}", f["run_dir"])
    assert f["runner"].endswith(
        f"--out {f['run_dir']} --rag results/rag/retrieval-scientsbank-en.bge-small-en-k5.json --k 3")






def test_rag_file_must_be_a_scientsbank_retrieval():
    p = dry("e2b", RAG="results/rag/retrieval-gsm-en.bge-small-en-k5.json")
    assert p.returncode != 0 and "not a SciEntsBank retrieval file" in p.stderr


ADAPTER = "/m/adapters/e2b-ft.gguf"


def test_lora_adds_only_that_flag():
    base = fields(dry("e2b", "test_uq"))
    ft = fields(dry("e2b", "test_uq", LORA=ADAPTER))
    assert ft["server"] == base["server"] + f" --lora {ADAPTER}"
    assert ft["runner"] == base["runner"].replace(base["run_dir"], ft["run_dir"])


def test_lora_folder_name():
    f = fields(dry("e2b", LORA=ADAPTER))
    assert re.fullmatch(rf"{REPO}/results/board/ft-e2b-jetson-test_ua-{STAMP}", f["run_dir"]), f["run_dir"]


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
                {"REPLY": "reasons", "LORA": ADAPTER}):
        p = dry("e2b", **env)
        assert p.returncode != 0 and "REPLY" in p.stderr, (env, p.stderr)


KNN = "results/examples/examples-knn-bge-small-en-k4.json"
RANDOM = "results/examples/examples-random-k4.json"


def test_adapter_tag_names_seeds_soup_and_5way_apart():
    for adapter, prefix in (("/m/adapters/e2b-ft-s3.gguf", "ft-s3"), ("/m/adapters/e2b-ft-soup.gguf", "ft-soup"),
                            ("/m/adapters/e2b-ft-lf.gguf", "ft-lf")):
        f = fields(dry("e2b", "test_uq", LORA=adapter))
        assert re.fullmatch(rf"{REPO}/results/board/{prefix}-e2b-jetson-test_uq-{STAMP}", f["run_dir"]), f["run_dir"]


def test_examples_pass_through_with_or_without_adapter():
    base = fields(dry("e2b", "test_uq"))
    for env, prefix in (({"EXAMPLES": KNN}, "ex4-e2b"), ({"EXAMPLES": RANDOM}, "rand4-e2b"),
                        ({"EXAMPLES": KNN, "LORA": ADAPTER}, "ex4-ft-e2b")):
        f = fields(dry("e2b", "test_uq", **env))
        assert re.fullmatch(rf"{REPO}/results/board/{prefix}-jetson-test_uq-{STAMP}", f["run_dir"]), f["run_dir"]
        assert f["runner"].endswith(f" --examples {env['EXAMPLES']} --k 4"), f["runner"]
    assert "--examples" not in base["runner"]


def test_fiveway_passes_through_with_its_adapter():
    f = fields(dry("e2b", "test_uq", LORA="/m/adapters/e2b-ft-5w-s0.gguf", PROMPT_SCHEME="5way"))
    assert re.fullmatch(rf"{REPO}/results/board/ft-5w-s0-e2b-jetson-test_uq-{STAMP}", f["run_dir"]), f["run_dir"]
    assert f["runner"].endswith(" --prompt-scheme 5way"), f["runner"]


def test_examples_and_fiveway_refused_outside_their_condition():
    for env in ({"EXAMPLES": "results/rag/x.json"},
                {"EXAMPLES": KNN, "THINKING": "on"}, {"EXAMPLES": KNN, "LORA": ADAPTER, "REPLY": "label-first"},
                {"PROMPT_SCHEME": "5way"}, {"PROMPT_SCHEME": "4way", "LORA": ADAPTER},
                {"PROMPT_SCHEME": "5way", "LORA": ADAPTER, "EXAMPLES": KNN}):
        p = dry("e2b", **env)
        assert p.returncode != 0 and ("EXAMPLES" in p.stderr or "PROMPT_SCHEME" in p.stderr), (env, p.stderr)


def test_thinking_runs_with_every_adapter_reply_and_the_5way_prompt():
    for env, prefix, tail in (({"LORA": ADAPTER}, "ft-think", " --thinking"),
                              ({"LORA": ADAPTER, "REPLY": "reason-first"}, "ft-rf-think",
                               " --thinking --reply reason-first"),
                              ({"LORA": "/m/adapters/e2b-ft-lf-beetle-saf.gguf", "REPLY": "label-first"},
                               "ft-lf-beetle-saf-think", " --thinking --reply label-first"),
                              ({"LORA": "/m/adapters/e2b-ft-5w-s0.gguf", "PROMPT_SCHEME": "5way"},
                               "ft-5w-s0-think", " --thinking --prompt-scheme 5way")):
        f = fields(dry("e2b", "test_uq", THINKING="on", **env))
        assert re.fullmatch(rf"{REPO}/results/board/{prefix}-e2b-jetson-test_uq-{STAMP}", f["run_dir"]), f["run_dir"]
        assert " ".join(f["runner"].split()).endswith(tail), f["runner"]
        assert "-rea on --reasoning-budget 320" in f["server"] and "--lora" in f["server"]
