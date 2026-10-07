"""News after the models' training cutoff: LiveNewsBench, no RAG and RAG (stage 2, topic benchmarks).

    uv run edge-grader-news --label e4b-kaggle-t4                                   # no RAG
    uv run edge-grader-news --label e4b-kaggle-t4-en-k3 \\
        --rag results/rag/retrieval-news-en.bge-small-en-k5.json --k 3
    uv run edge-grader-news --n 5                                                   # a quick pilot
    OPENAI_API_KEY=... uv run edge-grader-news-judge results/kaggle/news-baseline-<...>.json

The questions are LiveNewsBench's (arXiv 2602.13543, MIT) January 2026 release, the 200-question
human-verified test set: events of November-December 2025, after Gemma 4's training cutoff
(January 2025, model card). Each question ships with the news articles it was written from; those
articles, chunked like the course material, are the RAG corpus (edge_grader.rag.corpus
--benchmark news). The file is downloaded from LiveNewsBench's GitHub at a pinned commit and
checked against its sha256; it stays in data/ (git-ignored) because the articles are third-party
news text.

The prompt is LiveNewsBench's own no-search baseline (evals/no_search_baseline.py), a system line
and "Today is 2025-12-31. ... <answer>...</answer> ... {question}", byte for byte. With --rag the
passages go in front of it, in the same user message. Greedy, up to 2,048 tokens, like MMLU-Pro.
The final answer is the text inside the last <answer></answer>; none counts as not attempted.

Answers are free text, so they're scored by an LLM judge, as LiveNewsBench does: its SimpleQA
grading prompt, one of CORRECT / INCORRECT / NOT_ATTEMPTED per answer. edge-grader-news-judge
reads a run's files and writes new ones next to them (<name>.judged-<judge>.json/.jsonl); the
run's own files are never changed.
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import httpx
from openai import APIError, OpenAI

COMMIT = "8a6b96e7904893977ac8a3ae3342ae3d39f89c2e"  # LiveNewsBench main, 2026-02-14
URL = (f"https://raw.githubusercontent.com/YunfanZhang42/LiveNewsBench/{COMMIT}/"
       "datasets/jan_2026_release_2/human_verified_test.jsonl")
SHA256 = "e733b10302678c9d57d5e19cf41b735ddb852e81ba418436f2a9da28c96425e6"
DATA_FILE = Path("data/news/raw/livenewsbench-jan2026-human_verified_test.jsonl")
MAX_TOKENS = 2048
SEED = 1234

# LiveNewsBench's DirectAnswerBaseline. The system prompt's odd spacing is theirs: a backslash
# line continuation inside a triple-quoted string keeps the next line's indentation.
SYSTEM = ("You are excellent at following instructions and thinking step by step.     You are great "
          "at memorizing and recalling facts.     You are also good at following output format "
          "instructions.")
PROMPT = ("Today is 2025-12-31. Please answer the following question by thinking step by step. If "
          "you do not know the answer, provide your best guess. The final answer is usually a short "
          "phrase or a few words. Wrap the final answer with <answer> and </answer> tags at the end "
          "of your response (e.g. <answer>your_answer</answer>).\n\n{question}")

# The judge: LiveNewsBench's evals/grade_answers.py with its SimpleQA prompt (from OpenAI's
# simple-evals), GPT-4.1 in the paper.
JUDGE_PROMPT = Path(__file__).with_name("prompts") / "simpleqa_grader.txt"
JUDGE_SYSTEM = "You are a helpful assistant."
GRADES = {"A": "correct", "B": "incorrect", "C": "not_attempted"}


def fetch(path: Path = DATA_FILE) -> Path:
    """Download the question file once, and check it's the pinned one."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        r = httpx.get(URL, follow_redirects=True, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        sys.exit(f"{path} isn't LiveNewsBench {COMMIT[:7]}'s file (sha256 differs); delete it and rerun")
    return path


def article_id(article: dict) -> str:
    """Stable id for an article, by its text: 74 articles serve more than one question."""
    return hashlib.sha256(article["article_content"].encode()).hexdigest()[:12]


def load(path: Path = DATA_FILE) -> list[dict]:
    """The 200 questions in file order, ids news-000 .. news-199, with their articles' ids."""
    rows = [json.loads(line) for line in fetch(path).read_text().splitlines() if line.strip()]
    return [{"id": f"news-{i:03d}", "date": r["date"], "category": r["category"],
             "question": r["question"], "answer": r["answer"], "articles": r["articles"],
             "article_ids": sorted({article_id(a) for a in r["articles"]})}
            for i, r in enumerate(rows)]


def questions() -> dict[str, dict]:
    """For retrieval: question text, category as the subject, and one split, "test"."""
    return {q["id"]: {"question": q["question"], "subject": q["category"], "splits": ["test"],
                      "sources": [f"news:{a}" for a in q["article_ids"]]} for q in load()}


def messages(question: str, passages: list[str] | None = None) -> list[dict]:
    user = PROMPT.format(question=question)
    if passages:
        numbered = "\n".join(f"[{i}] {p}" for i, p in enumerate(passages, 1))
        user = f"News articles, for background:\n{numbered}\n\n{user}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def extract(text: str) -> str | None:
    """LiveNewsBench's rule: the text inside the last <answer>...</answer>, if both tags are there."""
    if "<answer>" in text and "</answer>" in text:
        return text.split("<answer>")[-1].split("</answer>")[0].strip()
    return None


def ask(client, model: str, msgs: list[dict]) -> dict:
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=model, messages=msgs, temperature=0, max_tokens=MAX_TOKENS, seed=SEED,
    )
    latency = time.perf_counter() - start
    choice = resp.choices[0]
    text = choice.message.content or ""
    return {
        "raw": text,
        "pred": extract(text),
        "finish": choice.finish_reason,
        "latency_s": round(latency, 3),
        "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
        "completion_tokens": getattr(resp.usage, "completion_tokens", None),
    }


def summarise(rows: list[dict], wall: float) -> dict:
    ok = [r for r in rows if "error" not in r]
    return {
        "n": len(rows),
        "answered": sum(r.get("pred") is not None for r in ok),
        "hit_token_cap": sum(r.get("finish") == "length" for r in ok),
        "errors": len(rows) - len(ok),
        "median_latency_s": statistics.median(r["latency_s"] for r in ok) if ok else None,
        "wall_s": round(wall, 1),
        "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in ok),
        "completion_tokens": sum(r["completion_tokens"] or 0 for r in ok),
    }


def main(argv: list[str] | None = None) -> None:
    from .baseline import load_passages, server_info

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1", help="llama-server endpoint")
    ap.add_argument("--model", default="local", help="model name sent to the server")
    ap.add_argument("--label", default=None, help="run label for the results file")
    ap.add_argument("--n", type=int, default=0, help="first n questions (0 = all 200)")
    ap.add_argument("--out", default="results/local", help="directory for this run's files")
    ap.add_argument("--rag", default=None, help="retrieval file: add article passages to the prompt")
    ap.add_argument("--k", type=int, default=3, help="passages per prompt with --rag")
    args = ap.parse_args(argv)

    passages, rag = load_passages(args.rag, args.k) if args.rag else ({}, None)
    stage = "rag" if args.rag else "baseline"
    server = server_info(args.base_url)
    todo = load()[: args.n or None]
    client = OpenAI(base_url=args.base_url, api_key="not-needed", timeout=1800, max_retries=3)

    rows, start = [], time.perf_counter()
    for i, q in enumerate(todo, 1):
        hits = passages.get(q["id"], []) if rag else []
        if rag and not hits:
            sys.exit(f"no retrieved passages for question {q['id']}")
        row = {"id": q["id"], "date": q["date"], "category": q["category"], "gold": q["answer"]}
        try:
            row |= ask(client, args.model, messages(q["question"], [t for _, t in hits]))
        except (APIError, httpx.HTTPError) as e:  # one failed request must not lose a run
            row |= {"pred": None, "error": repr(e)}
        if rag:
            row["passages"] = [cid for cid, _ in hits]
            own = {f"news:{a}" for a in q["article_ids"]}
            row["passages_from_own_articles"] = sum(cid.rsplit(":", 1)[0] in own for cid, _ in hits)
        rows.append(row)
        print(f"\r{i}/{len(todo)}", end="", flush=True)
    print()
    now = datetime.now(UTC)
    label = args.label or args.model
    run = {
        "stage": stage,
        "benchmark": "news",
        "label": label,
        "model": args.model,
        "base_url": args.base_url,
        "server": server,
        "dataset": {"name": "LiveNewsBench jan_2026_release_2 human_verified_test",
                    "commit": COMMIT, "sha256": SHA256},
        "n_requested": args.n,
        "generation": {"max_tokens": MAX_TOKENS, "temperature": 0,
                       "prompt": "LiveNewsBench no-search baseline"},
        "time": now.isoformat(timespec="seconds"),
        "metrics": summarise(rows, time.perf_counter() - start),
    }
    if rag:
        run["rag"] = rag
    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    name = f"news-{stage}-{label}-{now.strftime('%Y%m%d-%H%M%S')}"
    (out / f"{name}.json").write_text(json.dumps(run, indent=2))
    with open(out / "raw" / f"{name}.jsonl", "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    m = run["metrics"]
    print(f"answered {m['answered']} of {m['n']}, token cap {m['hit_token_cap']}, "
          f"errors {m['errors']} -> {out / (name + '.json')} (accuracy: run edge-grader-news-judge)")


# --- the judge ------------------------------------------------------------------------------

def judge_one(client, model: str, template: str, question: str, gold: str, pred: str) -> str:
    resp = client.chat.completions.create(
        model=model, temperature=0, top_p=1, max_tokens=1024,
        messages=[{"role": "system", "content": JUDGE_SYSTEM},
                  {"role": "user", "content": template.format(
                      question=question, expected_answer=gold, answer=pred)}],
    )
    return (resp.choices[0].message.content or "").strip()


def judge_main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Grade a news run's answers with an LLM judge "
                                 "(LiveNewsBench's SimpleQA prompt). The API key comes from the "
                                 "environment variable named by --key-env, never the command line.")
    ap.add_argument("run", type=Path, help="a news-*.json summary; its raw/ .jsonl is read too")
    ap.add_argument("--judge-model", default="gpt-4.1", help="LiveNewsBench's judge")
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--key-env", default="OPENAI_API_KEY")
    args = ap.parse_args(argv)

    run = json.loads(args.run.read_text())
    raw_path = args.run.parent / "raw" / f"{args.run.stem}.jsonl"
    rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    key = os.environ.get(args.key_env) or sys.exit(f"set {args.key_env} first")
    client = OpenAI(base_url=args.base_url, api_key=key, max_retries=5)
    template = JUDGE_PROMPT.read_text().strip()
    question = {q["id"]: q["question"] for q in load()}

    graded = []
    for i, r in enumerate(rows, 1):
        # As grade_answers.py: the extracted final answer, "" when there was none.
        reply = judge_one(client, args.judge_model, template, question[r["id"]], r["gold"],
                          r.get("pred") or "")
        graded.append({"id": r["id"], "category": r["category"], "pred": r.get("pred"),
                       "gold": r["gold"], "judge_reply": reply,
                       "grade": GRADES.get(reply, "unparsed")})
        print(f"\r{i}/{len(rows)}", end="", flush=True)
    print()
    counts = Counter(g["grade"] for g in graded)
    n = len(graded)
    per_category = {}
    for g in graded:
        c = per_category.setdefault(g["category"], {"n": 0, "correct": 0})
        c["n"] += 1
        c["correct"] += g["grade"] == "correct"
    tag = args.judge_model.replace("/", "_")
    summary = {
        "kind": "judged",
        "run": str(args.run),
        "run_label": run["label"],
        "run_stage": run["stage"],
        "judge": {"model": args.judge_model, "base_url": args.base_url, "temperature": 0,
                  "prompt": "LiveNewsBench evals/grade_simple_qa_prompt.txt (SimpleQA)",
                  "prompt_sha256": hashlib.sha256(JUDGE_PROMPT.read_bytes()).hexdigest()},
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "metrics": {"n": n, "accuracy": round(counts["correct"] / n, 4) if n else None,
                    **{k: counts[k] for k in ("correct", "incorrect", "not_attempted", "unparsed")},
                    "per_category": dict(sorted(per_category.items()))},
    }
    out = args.run.parent / f"{args.run.stem}.judged-{tag}.json"
    out.write_text(json.dumps(summary, indent=2))
    with open(args.run.parent / "raw" / f"{args.run.stem}.judged-{tag}.jsonl", "w") as f:
        f.writelines(json.dumps(g, ensure_ascii=False) + "\n" for g in graded)
    m = summary["metrics"]
    print(f"{m['accuracy']:.1%} correct of {n} (incorrect {m['incorrect']}, not attempted "
          f"{m['not_attempted']}, unparsed {m['unparsed']}) -> {out}")


if __name__ == "__main__":
    main()
