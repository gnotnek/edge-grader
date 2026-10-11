"""End-to-end run of the baseline CLI against a fake llama-server (no model, no network)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from edge_grader import baseline, data

ITEMS = [
    data.Item("a", "Q1?", "ref", "right answer", "correct"),
    data.Item("b", "Q2?", "ref", "wrong answer", "incorrect"),
    data.Item("c", "Q3?", "ref", "opposite answer", "contradictory"),
]


class FakeLlamaServer(BaseHTTPRequestHandler):
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
        self._send({"model_path": "/models/fake.gguf", "default_generation_settings": {"n_ctx": 8192}})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][0]["content"]
        # Grades "right" answers correct and everything else incorrect.
        reply = "correct" if "Student answer: right" in prompt else "incorrect"
        if body.get("max_tokens", 16) > 16:  # a thinking-on request: thoughts first, naming labels
            reply = f"<|channel>thought\nIs it correct or contradictory?<channel|>{reply}"
        self._send({
            "id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": reply}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 2, "total_tokens": 102},
        })


@pytest.fixture
def server():
    httpd = HTTPServer(("127.0.0.1", 0), FakeLlamaServer)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1"
    httpd.shutdown()


def test_baseline_writes_summary_and_raw(server, tmp_path, monkeypatch):
    monkeypatch.setattr(data, "load", lambda *a, **k: ITEMS)
    baseline.main(["--base-url", server, "--label", "fake", "--out", str(tmp_path)])

    [summary_file] = tmp_path.glob("baseline-fake-*.json")
    run = json.loads(summary_file.read_text())
    assert run["server"]["model_path"] == "/models/fake.gguf"
    m = run["metrics"]
    assert m["n"] == 3 and m["errors"] == 0 and m["unparsed"] == 0
    assert m["accuracy"] == pytest.approx(2 / 3, abs=1e-3)  # misses the contradictory one
    assert m["prompt_tokens"] == 300 and m["completion_tokens"] == 6

    [raw_file] = (tmp_path / "raw").glob("*.jsonl")
    assert len(raw_file.read_text().splitlines()) == 3


def test_baseline_exits_when_no_server(tmp_path):
    with pytest.raises(SystemExit):
        baseline.main(["--base-url", "http://127.0.0.1:9/v1", "--out", str(tmp_path)])


def test_rag_run_adds_passages_and_records_settings(server, tmp_path, monkeypatch):
    monkeypatch.setattr(data, "load", lambda *a, **k: ITEMS)
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text("".join(
        json.dumps({"id": f"b:{i}", "text": f"passage {i}"}) + "\n" for i in range(4)))
    from edge_grader.rag.index import sha256_of

    retrieval = tmp_path / "retrieval.json"
    retrieval.write_text(json.dumps({
        "settings": {"chunks": str(chunks), "chunks_sha256": sha256_of(chunks),
                     "model": "bge-small-en", "k": 5},
        "questions": {q: {"splits": ["test_ua"], "hits": [["b:2", 0.9], ["b:0", 0.8], ["b:1", 0.7]]}
                      for q in "abc"},
    }))
    prompts = []
    monkeypatch.setattr(baseline.grader, "build_prompt", _spy(baseline.grader.build_prompt, prompts))
    baseline.main(["--base-url", server, "--label", "fake", "--out", str(tmp_path),
                   "--rag", str(retrieval), "--k", "2"])

    [summary_file] = tmp_path.glob("rag-fake-*.json")
    run = json.loads(summary_file.read_text())
    assert run["stage"] == "rag"
    assert run["rag"]["k"] == 2 and run["rag"]["retrieval"] == str(retrieval)
    assert run["metrics"]["n"] == 3
    assert "[1] passage 2\n[2] passage 0\n" in prompts[0] and "[3]" not in prompts[0]
    [raw_file] = (tmp_path / "raw").glob("rag-*.jsonl")
    assert json.loads(raw_file.read_text().splitlines()[0])["passages"] == ["b:2", "b:0"]


def test_rag_refuses_a_changed_chunks_file(tmp_path, server, monkeypatch):
    monkeypatch.setattr(data, "load", lambda *a, **k: ITEMS)
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(json.dumps({"id": "b:0", "text": "x"}) + "\n")
    retrieval = tmp_path / "retrieval.json"
    retrieval.write_text(json.dumps({
        "settings": {"chunks": str(chunks), "chunks_sha256": "0" * 64},
        "questions": {},
    }))
    with pytest.raises(SystemExit, match="changed"):
        baseline.main(["--base-url", server, "--out", str(tmp_path), "--rag", str(retrieval)])


def _spy(fn, seen):
    def wrapped(*a, **k):
        out = fn(*a, **k)
        seen.append(out)
        return out
    return wrapped


def test_think_run_is_named_and_records_settings(server, tmp_path, monkeypatch):
    monkeypatch.setattr(data, "load", lambda *a, **k: ITEMS)
    baseline.main(["--base-url", server, "--label", "fake", "--n", "0", "--out", str(tmp_path),
                   "--thinking"])
    (f,) = tmp_path.glob("think-fake-test_ua-3way-*.json")
    run = json.loads(f.read_text())
    assert run["stage"] == "think"
    assert run["thinking"]["server_flags"] == "-rea on --reasoning-budget 320 --reasoning-format none"
    assert run["metrics"]["accuracy"] == pytest.approx(2 / 3, abs=1e-3)  # "right" -> correct, others -> incorrect
    assert run["metrics"]["unparsed"] == 0
    assert "median_completion_tokens" in run["metrics"]


def test_thinking_and_rag_together_are_refused(tmp_path):
    with pytest.raises(SystemExit, match="--thinking with --rag"):
        baseline.main(["--out", str(tmp_path), "--thinking", "--rag", "x.json"])
