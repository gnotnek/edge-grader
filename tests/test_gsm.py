"""The GSM8K / GSM-Symbolic runner: lm-eval's filters and exact match, and a fake-server run."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar

import pytest

from edge_grader import gsm

# The lab's recorded replies live in results/, which deploy.sh doesn't copy to the boards.
HAVE_LAB = gsm.LAB_FILE.exists()
LAB = json.loads(gsm.LAB_FILE.read_text()) if HAVE_LAB else {"runs": {}, "docs": {}}
needs_lab = pytest.mark.skipif(not HAVE_LAB, reason="results/ isn't deployed to the boards")
RECORDED = [(run, qid, rec) for run, r in LAB["runs"].items()
            for qid, rec in r["per_question"].items()]


@needs_lab
@pytest.mark.parametrize("filt", ["flexible", "strict"])
def test_extract_and_match_agree_with_lm_eval_on_every_recorded_reply(filt):
    # 3 lab runs x 100 replies: same extracted string and same right/wrong as lm-eval's.
    for run, qid, rec in RECORDED:
        got = gsm.extract(rec["reply"])[filt]
        assert got == rec[filt]["extracted"], (run, qid)
        assert gsm.exact_match(got, LAB["docs"][qid]["target"]) == rec[filt]["correct"], (run, qid)


@needs_lab
def test_lab_counts_are_what_the_lab_reported():
    runs = LAB["runs"]
    assert runs["tiny1024-e2b"]["flexible_correct"] == 81  # agentic-edge RESULTS.md, 1,024 tokens
    assert runs["B-e4b"]["flexible_correct"] == 75


def test_exact_match_normalises_like_lm_eval():
    assert gsm.exact_match("1,250", "so the total is 1250 #### 1,250")
    assert gsm.exact_match("$18.", "#### 18")
    assert not gsm.exact_match("70.00", "#### 70")  # lm-eval doesn't compare numerically
    assert gsm.extract("about 3 apples, so 12")["flexible"] == "12"
    assert gsm.extract("no digits")["flexible"] == gsm.INVALID


@needs_lab
def test_tiny_replays_the_labs_messages():
    docs = gsm.load("tiny")
    assert len(docs) == 100
    assert docs[0]["messages"] == LAB["docs"][docs[0]["id"]]["messages"]
    assert [m["role"] for m in docs[0]["messages"]] == ["user", "assistant"] * 5 + ["user"]


def test_fixed_shots_are_the_same_turns_every_time():
    train = [{"question": f"q{i}", "answer": f"a{i}\n#### {i}"} for i in range(50)]
    shots = gsm.fixed_shots(train)
    assert shots == gsm.fixed_shots(train)
    assert len(shots) == 10 and shots[0]["content"].startswith("Question: q")
    assert shots[1]["role"] == "assistant" and "####" in shots[1]["content"]


# --- a whole run against a fake llama-server -------------------------------------------

def _doc(qid, template, question, answer):
    return {"id": qid, "template": template, "target": f"work\n#### {answer}",
            "messages": [{"role": "user", "content": gsm.turn(question)}]}


SETS = {
    "gsm8k": [_doc("3", 3, "three?", 3), _doc("5", 5, "five?", 5)],
    "symbolic": [_doc("3/0", 3, "three?", 3), _doc("3/1", 3, "six?", 6),
                 _doc("5/0", 5, "five?", 5), _doc("5/1", 5, "ten?", 10)],
}


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
        last = body["messages"][-1]["content"].split("\n\n")[-1]  # past any course material
        # Right except "six?" (says 7) and "ten?" (runs out of tokens with no number).
        reply, finish = {"three?": ("3+0 = 3\n#### 3", "stop"), "five?": ("It is $5.", "stop"),
                         "six?": ("#### 7", "stop"), "ten?": ("let me see", "length")}[
            last.removeprefix("Question: ").removesuffix("\nAnswer:")]
        self._send({
            "id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": reply}}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 40, "total_tokens": 940},
        })


@pytest.fixture
def server(monkeypatch):
    FakeLlamaServer.seen = []
    monkeypatch.setattr(gsm, "load", lambda name, instances=10: SETS[name])
    httpd = HTTPServer(("127.0.0.1", 0), FakeLlamaServer)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1"
    httpd.shutdown()


def test_run_writes_one_file_per_set(server, tmp_path):
    gsm.main(["--base-url", server, "--label", "fake", "--out", str(tmp_path),
              "--set", "gsm8k", "symbolic"])

    orig = json.loads(next(tmp_path.glob("gsm-gsm8k-fake-*.json")).read_text())
    sym = json.loads(next(tmp_path.glob("gsm-symbolic-fake-*.json")).read_text())
    assert orig["stage"] == "topics" and orig["server"]["build_info"] == "b1-fake"
    assert orig["metrics"]["accuracy"] == 1.0 and orig["metrics"]["strict_accuracy"] == 0.5
    assert sym["metrics"]["accuracy"] == 0.5
    assert sym["metrics"]["per_instance_accuracy"] == {"0": 1.0, "1": 0.0}
    assert (sym["metrics"]["no_number"], sym["metrics"]["hit_token_cap"]) == (1, 1)
    assert sym["generation"]["instances"] == 10

    body = FakeLlamaServer.seen[0]
    assert body["max_tokens"] == 1024 and body["temperature"] == 0 and body["seed"] == 1234
    assert body["stop"] == ["Question:", "</s>", "<|im_end|>"]
    rows = [json.loads(line) for line in
            next((tmp_path / "raw").glob("gsm-symbolic-*.jsonl")).read_text().splitlines()]
    assert [r["template"] for r in rows] == [3, 3, 5, 5]
    assert [r["gold"] for r in rows] == ["3", "6", "5", "10"]


# --- retrieval questions (phase A) ---------------------------------------------------------

TRAIN = [{"question": f"train q{i}?", "answer": f"#### {i}"} for i in range(200)]


def fake_load_dataset(name, config=None, split=None):
    assert name == gsm.GSM8K_ID and split == "train"
    return TRAIN


def test_gsm_questions_cover_both_sets_and_a_train_pool(monkeypatch):
    monkeypatch.setattr(gsm, "load", lambda name, instances=10: SETS[name])
    import datasets
    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    qs = gsm.questions()
    assert qs["3"] == {"question": "three?", "subject": "math", "splits": ["gsm8k"]}
    assert qs["5/1"]["splits"] == ["symbolic"]
    train = [k for k, v in qs.items() if v["splits"] == ["train"]]
    assert len(train) == gsm.CHECK_POOL and all(k.startswith("train-") for k in train)


def test_gsm_questions_exclude_the_shots(monkeypatch):
    monkeypatch.setattr(gsm, "load", lambda name, instances=10: SETS[name])
    import datasets
    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    shot_texts = {m["content"] for m in gsm.fixed_shots(TRAIN) if m["role"] == "user"}
    pool = {gsm.turn(v["question"]) for v in gsm.questions().values() if v["splits"] == ["train"]}
    assert not shot_texts & pool


# --- RAG (phase A) -------------------------------------------------------------------------

def test_with_passages_prefixes_only_the_last_turn():
    msgs = [{"role": "user", "content": "Question: shot?\nAnswer:"},
            {"role": "assistant", "content": "#### 1"},
            {"role": "user", "content": "Question: three?\nAnswer:"}]
    out = gsm.with_passages(msgs, ["Add the parts.", "Count up."])
    assert out[:2] == msgs[:2]
    assert out[2]["content"] == ("Course material, for background only:\n[1] Add the parts.\n"
                                 "[2] Count up.\n\nQuestion: three?\nAnswer:")
    assert msgs[2]["content"] == "Question: three?\nAnswer:"  # the input isn't changed


@pytest.fixture
def retrieval(tmp_path):
    import hashlib

    chunks = tmp_path / "gsm-en.jsonl"
    chunks.write_text("".join(json.dumps({"id": f"en-openstax-prealgebra-2e:{i}", "text": f"p{i}"}) + "\n"
                              for i in range(3)))
    r = {"settings": {"chunks": str(chunks), "k": 5, "index": "x", "model": "bge-small-en",
                      "chunks_sha256": hashlib.sha256(chunks.read_bytes()).hexdigest()},
         "questions": {q: {"hits": [[f"en-openstax-prealgebra-2e:{i}", 0.5] for i in range(3)]}
                       for q in ("3", "5", "3/0", "3/1", "5/0", "5/1")}}
    path = tmp_path / "retrieval-gsm-en.bge-small-en-k5.json"
    path.write_text(json.dumps(r))
    return path


def test_rag_run_is_named_gsm_rag_and_records_settings(server, tmp_path, retrieval):
    gsm.main(["--base-url", server, "--label", "fake", "--out", str(tmp_path / "out"),
              "--set", "gsm8k", "--rag", str(retrieval), "--k", "2"])
    files = list((tmp_path / "out").glob("gsm-*.json"))
    assert [f.name.startswith("gsm-rag-gsm8k-fake-") for f in files] == [True]
    run = json.loads(files[0].read_text())
    assert run["stage"] == "rag" and run["rag"]["k"] == 2
    assert run["metrics"]["accuracy"] == 1.0
    last = FakeLlamaServer.seen[0]["messages"][-1]["content"]
    assert last.startswith("Course material, for background only:\n[1] p0\n[2] p1\n\n")
    rows = [json.loads(x) for x in next((tmp_path / "out" / "raw").glob("*.jsonl")).read_text().splitlines()]
    assert rows[0]["passages"] == ["en-openstax-prealgebra-2e:0", "en-openstax-prealgebra-2e:1"]


def test_rag_run_stops_when_a_question_has_no_passages(server, tmp_path, retrieval):
    r = json.loads(retrieval.read_text())
    del r["questions"]["5"]
    retrieval.write_text(json.dumps(r))
    with pytest.raises(SystemExit, match="no retrieved passages for question 5"):
        gsm.main(["--base-url", server, "--label", "fake", "--out", str(tmp_path / "out"),
                  "--set", "gsm8k", "--rag", str(retrieval)])
