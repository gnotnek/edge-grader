"""The stage-3b teacher: prompt, filter, and a resumable run against a fake llama-server."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from edge_grader import data, reasons
from edge_grader.data import Item, labels_for

LABELS = labels_for("3way")
ITEMS = [
    Item("EM.1.1", "Q1?", "ref one", "answer one", "correct"),
    Item("EM.1.2", "Q1?", "ref one", "answer two", "incorrect"),
    Item("EM.2.1", "Q2?", "ref two", "answer three", "contradictory"),
]


def test_check_reason_codes():
    assert reasons.check_reason("It names the force and its direction.", "stop", "correct", LABELS) is None
    assert reasons.check_reason("  \n", "stop", "correct", LABELS) == "empty"
    assert reasons.check_reason("It names the force but", "length", "correct", LABELS) == "cut"
    assert reasons.check_reason("It is partially correct.", "stop", "incorrect", LABELS) == "wrong_label"
    assert reasons.check_reason("Missing the cause.\nLabel: incorrect", "stop", "incorrect",
                                LABELS) == "label_line"


def test_check_reason_tolerates_the_gold_label_word():
    assert reasons.check_reason("The answer is incorrect: no cause.", "stop", "incorrect", LABELS) is None
    assert reasons.check_reason("It is correct.", "stop", "correct", LABELS) is None


def test_teacher_prompt_has_the_answer_and_the_gold_label():
    msgs = reasons.teacher_messages(ITEMS[2])
    text = msgs[0]["content"]
    assert msgs[0]["role"] == "user" and len(msgs) == 1
    assert all(s in text for s in ("Q2?", "ref two", "answer three", "contradictory",
                                   "contradicts the reference answer"))
    assert "{" not in text  # every placeholder filled


class FakeTeacher(BaseHTTPRequestHandler):
    calls = 0

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
        self._send({"model_path": "/models/Fake-Teacher.gguf", "build_info": "b1"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).calls += 1
        prompt = body["messages"][0]["content"]
        # The contradictory answer gets a reason that names another label, so it's dropped.
        reply = ("<think>\n\n</think>\n\nIt is incorrect." if "answer three" in prompt
                 else "It covers  the\nmain idea.")
        self._send({"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": reply}}],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 8, "total_tokens": 58}})


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(data, "load", lambda *a, **k: ITEMS)
    FakeTeacher.calls = 0
    httpd = HTTPServer(("127.0.0.1", 0), FakeTeacher)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1"
    httpd.shutdown()


def test_run_writes_reasons_and_summary(server, tmp_path):
    reasons.main(["--base-url", server, "--out", str(tmp_path)])

    lines = [json.loads(x) for x in (tmp_path / "raw" / "reasons-fake-teacher.jsonl").read_text().splitlines()]
    assert [r["id"] for r in lines] == ["EM.1.1", "EM.1.2", "EM.2.1"]
    assert lines[0]["reason"] == "It covers the main idea." and lines[0]["drop"] is None
    assert lines[2]["drop"] == "wrong_label" and lines[2]["reason"] == "It is incorrect."
    [summary_file] = tmp_path.glob("reasons-fake-teacher-*.json")
    s = json.loads(summary_file.read_text())
    assert s["n"] == 3 and s["kept"] == 2 and s["drops"] == {"wrong_label": 1}
    assert s["drop_share"] == pytest.approx(1 / 3, abs=1e-3)
    assert s["teacher"]["model_path"] == "/models/Fake-Teacher.gguf"
    assert len(s["reasons_sha256"]) == 64 and len(s["prompt_sha256"]) == 64
    assert not list((tmp_path / "raw").glob("*.part"))


def test_run_resumes_from_its_part_file(server, tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    sha = reasons.prompt_sha256()
    (raw / "reasons-fake-teacher.jsonl.part").write_text("".join(
        json.dumps({"id": i.id, "gold": i.label, "reason": "Earlier.", "finish": "stop", "drop": None,
                    "prompt_sha256": sha}) + "\n" for i in ITEMS[:2]))
    reasons.main(["--base-url", server, "--out", str(tmp_path)])
    assert FakeTeacher.calls == 1
    lines = (raw / "reasons-fake-teacher.jsonl").read_text().splitlines()
    assert [json.loads(x)["reason"] for x in lines][:2] == ["Earlier.", "Earlier."]


def test_resume_refuses_reasons_from_another_prompt(server, tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "reasons-fake-teacher.jsonl.part").write_text(json.dumps(
        {"id": "EM.1.1", "gold": "correct", "reason": "Old.", "finish": "stop", "drop": None,
         "prompt_sha256": "0" * 64}) + "\n")
    with pytest.raises(SystemExit, match="prompt"):
        reasons.main(["--base-url", server, "--out", str(tmp_path)])


def test_a_pilot_is_named_apart_from_the_full_run(server, tmp_path):
    reasons.main(["--base-url", server, "--out", str(tmp_path), "--n", "2"])
    assert (tmp_path / "raw" / "reasons-fake-teacher-n2.jsonl").exists()
