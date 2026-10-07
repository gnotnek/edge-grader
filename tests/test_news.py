"""The news runner: LiveNewsBench's prompt and answer rule, a fake-server run, and the judge."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar

import pytest

from edge_grader import news
from edge_grader.rag import check, corpus

ARTICLES = {
    "a": {"title": "A", "article_content": "# Head\n\n" + "The vote was won by Ms Example of party X. " * 4},
    "b": {"title": "B", "article_content": "Turnout reached 55 percent across the whole country. " * 4},
}
QUESTIONS = [
    {"id": "news-000", "date": "2025-11-16", "category": "Politics", "question": "Who won?",
     "answer": "Ms Example", "articles": [ARTICLES["a"], ARTICLES["b"]]},
    {"id": "news-001", "date": "2025-12-01", "category": "Sports", "question": "What turnout?",
     "answer": "55%", "articles": [ARTICLES["b"]]},
]
for q in QUESTIONS:
    q["article_ids"] = sorted({news.article_id(a) for a in q["articles"]})


def test_prompt_is_livenewsbenchs_no_search_baseline():
    msgs = news.messages("Who won?")
    assert msgs[0] == {"role": "system", "content": news.SYSTEM}
    assert "step by step.     You are great" in news.SYSTEM  # their line continuation's spaces
    assert msgs[1]["content"].startswith("Today is 2025-12-31. Please answer")
    assert msgs[1]["content"].endswith("(e.g. <answer>your_answer</answer>).\n\nWho won?")


def test_rag_puts_passages_in_front_of_the_same_prompt():
    plain = news.messages("Who won?")
    rag = news.messages("Who won?", ["p one", "p two"])
    assert rag[0] == plain[0]
    assert rag[1]["content"] == ("News articles, for background:\n[1] p one\n[2] p two\n\n"
                                 + plain[1]["content"])


def test_extract_takes_the_last_answer_tags():
    assert news.extract("x <answer>a</answer> then <answer> b </answer>") == "b"
    assert news.extract("no tags") is None
    assert news.extract("<answer>unclosed") is None


def test_article_ids_dedupe_shared_articles(monkeypatch):
    monkeypatch.setattr(news, "load", lambda path=None: QUESTIONS)
    docs = corpus.news_pages()
    assert len(docs) == 2  # article b serves both questions, chunked once
    chunks = corpus.build("news", "en")
    assert {c["source"] for c in chunks} == {f"news:{news.article_id(a)}" for a in ARTICLES.values()}
    qs = news.questions()
    assert qs["news-001"]["sources"] == [f"news:{news.article_id(ARTICLES['b'])}"]


def test_source_hits_counts_own_articles_in_the_top_k():
    retrieval = {"questions": {"q1": {"hits": [["news:x:0", 0.9], ["news:y:3", 0.8]]},
                               "q2": {"hits": [["news:z:1", 0.9], ["news:x:2", 0.8]]}}}
    hits = check.source_hits(retrieval, {"q1": ["news:y"], "q2": ["news:q"]}, ks=(1, 2))
    assert hits == {"questions": 2, "top1": 0, "top2": 1}


# --- a whole run and its judging against fake servers ----------------------------------

class FakeServer(BaseHTTPRequestHandler):
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
        FakeServer.seen.append(body)
        last = body["messages"][-1]["content"]
        if "Gold target:" in last:  # the judge: right if the prediction names Ms Example
            reply, finish = ("A" if "Predicted answer: Ms Example" in last else
                             "C" if "Predicted answer: \n" in last else "B"), "stop"
        elif "Who won?" in last:
            reply, finish = "Thinking. <answer>Ms Example</answer>", "stop"
        else:
            reply, finish = "Let me think about turnout", "length"
        self._send({
            "id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": reply}}],
            "usage": {"prompt_tokens": 300, "completion_tokens": 30, "total_tokens": 330},
        })


@pytest.fixture
def server(monkeypatch):
    FakeServer.seen = []
    monkeypatch.setattr(news, "load", lambda path=None: QUESTIONS)
    httpd = HTTPServer(("127.0.0.1", 0), FakeServer)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1"
    httpd.shutdown()


def test_run_then_judge(server, tmp_path, monkeypatch):
    news.main(["--base-url", server, "--label", "fake", "--out", str(tmp_path)])
    run_file = next(tmp_path.glob("news-baseline-fake-*.json"))
    run = json.loads(run_file.read_text())
    assert run["benchmark"] == "news" and run["dataset"]["commit"] == news.COMMIT
    assert (run["metrics"]["answered"], run["metrics"]["hit_token_cap"]) == (1, 1)
    body = FakeServer.seen[0]
    assert body["max_tokens"] == 2048 and body["temperature"] == 0 and "stop" not in body

    monkeypatch.setenv("FAKE_KEY", "k")
    news.judge_main([str(run_file), "--base-url", server, "--judge-model", "fake/judge",
                     "--key-env", "FAKE_KEY"])
    judged = json.loads(next(tmp_path.glob("news-baseline-fake-*.judged-fake_judge.json")).read_text())
    assert judged["metrics"]["accuracy"] == 0.5
    assert (judged["metrics"]["correct"], judged["metrics"]["not_attempted"]) == (1, 1)
    assert judged["metrics"]["per_category"]["Politics"] == {"n": 1, "correct": 1}
    assert FakeServer.seen[-1]["messages"][0]["content"] == news.JUDGE_SYSTEM
    assert json.loads(run_file.read_text()) == run  # the run's own file is untouched


def test_rag_run_counts_passages_from_own_articles(server, tmp_path):
    from edge_grader.rag.index import sha256_of

    own = f"news:{news.article_id(ARTICLES['a'])}"
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text("".join(json.dumps({"id": cid, "text": f"text of {cid}"}) + "\n"
                              for cid in (f"{own}:0", "news:other:0")))
    retrieval = tmp_path / "retrieval.json"
    retrieval.write_text(json.dumps({
        "settings": {"chunks": str(chunks), "chunks_sha256": sha256_of(chunks),
                     "model": "bge-small-en", "k": 5},
        "questions": {q["id"]: {"splits": ["test"], "hits": [[f"{own}:0", 0.9],
                                                              ["news:other:0", 0.5]]}
                      for q in QUESTIONS},
    }))
    news.main(["--base-url", server, "--label", "fake", "--out", str(tmp_path),
               "--rag", str(retrieval), "--k", "2"])
    rows = [json.loads(line) for line in
            next((tmp_path / "raw").glob("news-rag-*.jsonl")).read_text().splitlines()]
    assert [r["passages_from_own_articles"] for r in rows] == [1, 0]
    assert FakeServer.seen[0]["messages"][1]["content"].startswith(
        f"News articles, for background:\n[1] text of {own}:0\n[2] text of news:other:0\n\nToday")
