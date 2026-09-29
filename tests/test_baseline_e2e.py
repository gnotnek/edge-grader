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
