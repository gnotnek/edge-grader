"""The MMLU-Pro runner: lm-eval's prompt byte for byte, its answer pattern, and a fake-server run."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

from edge_grader import mmlupro

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "mmlupro_lab_prompts.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: c["doc"]["category"])
def test_messages_match_what_lm_eval_sent(case):
    # One recorded question per subject, so every subject's worked examples are covered.
    assert mmlupro.messages(case["doc"], case["shots"]) == case["messages"]


@pytest.mark.parametrize("case", FIXTURE["replies"], ids=lambda c: c["lm_eval"])
def test_extract_agrees_with_lm_eval(case):
    expected = None if case["lm_eval"] == "[invalid]" else case["lm_eval"]
    assert mmlupro.extract(case["reply"]) == expected


def test_extract_takes_the_first_answer():
    assert mmlupro.extract("the answer is (B). Wait, the answer is (C)") == "B"
    assert mmlupro.extract("so the answer is D") == "D"
    assert mmlupro.extract("I think B") is None


def test_rag_changes_only_the_last_user_message():
    case = FIXTURE["cases"][0]
    plain = mmlupro.messages(case["doc"], case["shots"])
    rag = mmlupro.messages(case["doc"], case["shots"], ["first passage", "second passage"])
    assert rag[:-1] == plain[:-1]
    assert rag[-1]["content"] == (
        "Course material, for background only:\n[1] first passage\n[2] second passage\n\n"
        + plain[-1]["content"])


def test_subset_ids_are_the_labs_three_disjoint_hundreds():
    ids = {s: mmlupro.subset_ids(s) for s in mmlupro.SUBSETS}
    assert all(len(v) == 100 for v in ids.values())
    assert len(set().union(*ids.values())) == 300


# --- a whole run against a fake llama-server -------------------------------------------

DOCS = [
    {"question_id": 7, "question": "Q seven?", "options": ["x", "y"], "answer": "B",
     "category": "physics", "subset": "s1"},
    {"question_id": 8, "question": "Q eight?", "options": ["x", "y"], "answer": "A",
     "category": "physics", "subset": "s1"},
    {"question_id": 9, "question": "Q nine?", "options": ["x", "y"], "answer": "A",
     "category": "law", "subset": "s2"},
]
SHOTS = {c: [] for c in ("physics", "law")}


class FakeLlamaServer(BaseHTTPRequestHandler):
    seen: ClassVar[list[dict]] = []

    def log_message(self, *args):
        pass

    def _send(self, body: dict):
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        self._send({"model_path": "/models/fake.gguf", "build_info": "b1-fake",
                    "default_generation_settings": {"n_ctx": 8192}})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeLlamaServer.seen.append(body)
        last = body["messages"][-1]["content"]
        # Right on 7, wrong on 8, no answer and out of tokens on 9.
        reply, finish = ("the answer is (B)", "stop") if "seven" in last else \
            ("the answer is (C)", "stop") if "eight" in last else ("hmm", "length")
        self._send({
            "id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": reply}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 50, "total_tokens": 1050},
        })


@pytest.fixture
def server(monkeypatch):
    FakeLlamaServer.seen = []
    monkeypatch.setattr(mmlupro, "load", lambda subsets: (
        [d for d in DOCS if d["subset"] in subsets], SHOTS))
    httpd = HTTPServer(("127.0.0.1", 0), FakeLlamaServer)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1"
    httpd.shutdown()


def test_run_writes_one_file_per_subset_with_lm_eval_settings(server, tmp_path):
    mmlupro.main(["--base-url", server, "--label", "fake", "--out", str(tmp_path),
                  "--subset", "s1", "s2"])

    s1 = json.loads(next(tmp_path.glob("mmlupro-baseline-fake-s1-*.json")).read_text())
    s2 = json.loads(next(tmp_path.glob("mmlupro-baseline-fake-s2-*.json")).read_text())
    assert s1["server"]["build_info"] == "b1-fake"
    assert s1["metrics"]["n"] == 2 and s1["metrics"]["accuracy"] == 0.5
    assert (s2["metrics"]["no_answer"], s2["metrics"]["hit_token_cap"]) == (1, 1)
    assert s2["metrics"]["accuracy"] == 0.0
    assert s1["metrics"]["per_subject"] == {"physics": {"n": 2, "correct": 1}}

    body = FakeLlamaServer.seen[0]
    assert body["max_tokens"] == 2048 and body["temperature"] == 0
    assert body["stop"] == ["Question:"]
    raw = next((tmp_path / "raw").glob("*-s1-*.jsonl")).read_text()
    rows = [json.loads(line) for line in raw.splitlines()]
    assert [r["correct"] for r in rows] == [True, False]


def test_rag_run_records_passages(server, tmp_path):
    from edge_grader.rag.index import sha256_of

    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text("".join(json.dumps({"id": f"b:{i}", "text": f"passage {i}"}) + "\n"
                              for i in range(3)))
    retrieval = tmp_path / "retrieval.json"
    retrieval.write_text(json.dumps({
        "settings": {"chunks": str(chunks), "chunks_sha256": sha256_of(chunks),
                     "model": "bge-small-en", "k": 5},
        "questions": {q: {"splits": ["s1"], "hits": [["b:2", 0.9], ["b:0", 0.8]]}
                      for q in ("7", "8")},
    }))
    mmlupro.main(["--base-url", server, "--label", "fake", "--out", str(tmp_path),
                  "--subset", "s1", "--rag", str(retrieval), "--k", "1"])

    run = json.loads(next(tmp_path.glob("mmlupro-rag-fake-s1-*.json")).read_text())
    assert run["stage"] == "rag" and run["rag"]["k"] == 1
    assert FakeLlamaServer.seen[0]["messages"][-1]["content"].startswith(
        "Course material, for background only:\n[1] passage 2\n\nQuestion:")
    raw = next((tmp_path / "raw").glob("mmlupro-rag-*.jsonl")).read_text()
    row = json.loads(raw.splitlines()[0])
    assert row["passages"] == ["b:2"]
