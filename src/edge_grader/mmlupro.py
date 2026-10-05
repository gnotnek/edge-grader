"""MMLU-Pro on the lab's 300-question subset, with or without retrieved course material (stage 2).

    uv run edge-grader-mmlupro --label e4b-kaggle-t4                    # no RAG, s1-s3
    uv run edge-grader-mmlupro --label e4b-kaggle-t4-en-k3 \\
        --rag results/rag/retrieval-mmlupro-en.bge-small-en-k5.json --k 3
    uv run edge-grader-mmlupro --subset s1 --n 5                         # a quick pilot

Without --rag this sends what agentic-edge's lm-eval run sent: the `mmlu_pro` task's chat
messages (a system line, five worked examples from the validation split, then the question),
greedy, up to 2,048 tokens, stopping at "Question:". tests/test_mmlupro.py pins that against
prompts recorded from the lab's runs, so "no RAG" here is the lab baseline, not a near copy.
The answer is read with lm-eval's own pattern, "answer is (X)"; no match counts as wrong.

With --rag the top-k passages go in front of the question, in the last user message only; the
worked examples stay as they are. Retrieval used the question text alone, never the options.

One file per subset: <out>/mmlupro-<stage>-<label>-<subset>-<stamp>.json (settings and
metrics) and raw/<same>.jsonl (one line per question: the reply, the extracted letter, tokens).
"""

import argparse
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import httpx
from openai import APIError, OpenAI

DATASET_ID = "TIGER-Lab/MMLU-Pro"
SUBSET_DIR = Path("results/reference/agentic-edge/lm-eval")
SUBSETS = ("s1", "s2", "s3")
LETTERS = "ABCDEFGHIJKLMNOP"
SHOTS = 5
MAX_TOKENS = 2048  # AGENTS.md rule 5: lm-eval's default 256 cuts the reasoning short
STOP = ["Question:"]
SEED = 1234  # what lm-eval sends; greedy decoding makes it moot
# lm-eval's mmlu_pro "custom-extract" filter: first match, else no answer.
ANSWER_RE = re.compile(r"answer is \(?([ABCDEFGHIJ])\)?")


def subset_ids(subset: str, ref_dir: Path = SUBSET_DIR) -> list[int]:
    """The lab's question ids for one subset, in its order (by subject, then id)."""
    spec = json.loads((ref_dir / f"mmlupro_subset100_{subset}_ids.json").read_text())
    return [qid for ids in spec["question_ids"].values() for qid in ids]


def load(subsets=SUBSETS) -> tuple[list[dict], dict[str, list[dict]]]:
    """The subset's test questions (each tagged with its subset) and the worked examples per
    subject: the first five validation questions of that subject, in dataset order."""
    from datasets import load_dataset

    test = load_dataset(DATASET_ID, split="test")
    by_id = {row["question_id"]: row for row in test}
    docs = [{**by_id[qid], "subset": s} for s in subsets for qid in subset_ids(s)]
    shots: dict[str, list[dict]] = defaultdict(list)
    for row in load_dataset(DATASET_ID, split="validation"):
        if len(shots[row["category"]]) < SHOTS:
            shots[row["category"]].append(row)
    return docs, dict(shots)


def format_question(doc: dict) -> str:
    options = "".join(f"{LETTERS[i]}. {o.strip()}\n" for i, o in enumerate(doc["options"]))
    return f"Question:\n{doc['question']}\nOptions:\n{options}Answer: Let's think step by step."


def messages(doc: dict, shots: list[dict], passages: list[str] | None = None) -> list[dict]:
    """lm-eval's mmlu_pro chat messages; with passages, course material before the question."""
    subject = doc["category"]
    out = [{"role": "system", "content": (
        f"The following are multiple choice questions (with answers) about {subject}. Think step "
        'by step and then finish your answer with "the answer is (X)" where X is the correct '
        "letter choice.\n")}]
    for shot in shots:
        out.append({"role": "user", "content": format_question(shot)})
        worked = shot["cot_content"].removeprefix("A: Let's think step by step.").strip()
        out.append({"role": "assistant", "content": worked})
    question = format_question(doc)
    if passages:
        numbered = "\n".join(f"[{i}] {p}" for i, p in enumerate(passages, 1))
        question = f"Course material, for background only:\n{numbered}\n\n{question}"
    out.append({"role": "user", "content": question})
    return out


def extract(text: str) -> str | None:
    m = ANSWER_RE.search(text)
    return m.group(1) if m else None


def ask(client, model: str, msgs: list[dict]) -> dict:
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=model, messages=msgs, temperature=0, max_tokens=MAX_TOKENS, stop=STOP, seed=SEED,
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
    by_subject = defaultdict(list)
    for r in rows:
        by_subject[r["subject"]].append(r["correct"])
    return {
        "n": len(rows),
        "accuracy": round(sum(r["correct"] for r in rows) / len(rows), 4) if rows else None,
        "no_answer": sum(r.get("pred") is None for r in ok),
        "hit_token_cap": sum(r.get("finish") == "length" for r in ok),
        "errors": len(rows) - len(ok),
        "median_latency_s": statistics.median(r["latency_s"] for r in ok) if ok else None,
        "wall_s": round(wall, 1),
        "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in ok),
        "completion_tokens": sum(r["completion_tokens"] or 0 for r in ok),
        "per_subject": {s: {"n": len(v), "correct": sum(v)} for s, v in sorted(by_subject.items())},
    }


def main(argv: list[str] | None = None) -> None:
    from .baseline import load_passages, server_info

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1", help="llama-server endpoint")
    ap.add_argument("--model", default="local", help="model name sent to the server")
    ap.add_argument("--label", default=None, help="run label for the results file")
    ap.add_argument("--subset", nargs="+", default=list(SUBSETS), choices=SUBSETS)
    ap.add_argument("--n", type=int, default=0, help="first n questions per subset (0 = all 100)")
    ap.add_argument("--out", default="results/local", help="directory for this run's files")
    ap.add_argument("--rag", default=None, help="retrieval file: add course material to the prompt")
    ap.add_argument("--k", type=int, default=3, help="passages per prompt with --rag")
    args = ap.parse_args(argv)

    passages, rag = load_passages(args.rag, args.k) if args.rag else ({}, None)
    stage = "rag" if args.rag else "baseline"
    server = server_info(args.base_url)
    docs, shots = load(args.subset)
    client = OpenAI(base_url=args.base_url, api_key="not-needed", timeout=1800, max_retries=3)
    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)

    for subset in args.subset:
        todo = [d for d in docs if d["subset"] == subset][: args.n or None]
        rows, start = [], time.perf_counter()
        for i, doc in enumerate(todo, 1):
            qid = str(doc["question_id"])
            hits = passages.get(qid, []) if rag else []
            if rag and not hits:
                sys.exit(f"no retrieved passages for question {qid}")
            row = {"id": doc["question_id"], "subset": subset, "subject": doc["category"],
                   "gold": doc["answer"]}
            try:
                row |= ask(client, args.model, messages(doc, shots[doc["category"]],
                                                        [t for _, t in hits]))
            except (APIError, httpx.HTTPError) as e:  # one failed request must not lose a run
                row |= {"pred": None, "error": repr(e)}
            row["correct"] = row["pred"] == row["gold"]
            if rag:
                row["passages"] = [cid for cid, _ in hits]
            rows.append(row)
            print(f"\r{subset} {i}/{len(todo)}", end="", flush=True)
        print()
        now = datetime.now(UTC)
        label = args.label or args.model
        run = {
            "stage": stage,
            "benchmark": "mmlupro",
            "label": label,
            "model": args.model,
            "base_url": args.base_url,
            "server": server,
            "subset": subset,
            "n_requested": args.n,
            "generation": {"max_tokens": MAX_TOKENS, "temperature": 0, "stop": STOP,
                           "shots": SHOTS, "prompt": "lm-eval mmlu_pro chat messages"},
            "time": now.isoformat(timespec="seconds"),
            "metrics": summarise(rows, time.perf_counter() - start),
        }
        if rag:
            run["rag"] = rag
        name = f"mmlupro-{stage}-{label}-{subset}-{now.strftime('%Y%m%d-%H%M%S')}"
        (out / f"{name}.json").write_text(json.dumps(run, indent=2))
        with open(out / "raw" / f"{name}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        m = run["metrics"]
        print(f"{subset}: {m['accuracy']:.1%} of {m['n']}, no answer {m['no_answer']}, "
              f"token cap {m['hit_token_cap']}, errors {m['errors']} -> {out / (name + '.json')}")


def questions(subsets=SUBSETS, validation: bool = True) -> dict[str, dict]:
    """Questions to retrieve for, keyed by question id as a string: the subset's test questions,
    and the validation questions (the worked-example pool), which the retrieval check samples
    from so the corpus is never tuned on test questions."""
    from datasets import load_dataset

    docs, _ = load(subsets)
    out = {str(d["question_id"]): {"question": d["question"], "subject": d["category"],
                                   "splits": [d["subset"]]} for d in docs}
    if validation:
        for row in load_dataset(DATASET_ID, split="validation"):
            out[str(row["question_id"])] = {"question": row["question"],
                                            "subject": row["category"], "splits": ["validation"]}
    return out


if __name__ == "__main__":
    main()
