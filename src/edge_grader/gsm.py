"""Grade-school math: GSM8K and GSM-Symbolic, no RAG (stage 2, topic benchmarks: math).

    uv run edge-grader-gsm --label e2b-kaggle-t4 --set tiny            # runner check against the lab
    uv run edge-grader-gsm --label e4b-kaggle-t4 --set gsm8k symbolic  # the comparison
    uv run edge-grader-gsm --set gsm8k --n 5                            # a quick pilot

Three question sets:
  tiny      the lab's tinyGSM8k run replayed: the 100 questions with the exact chat messages
            agentic-edge's lm-eval sent (results/reference/agentic-edge/lm-eval/tinygsm8k-lab.json).
            The lab's E2B run at 1,024 tokens got 81/100; E4B was only run at 256 (--max-tokens 256
            to compare, 75/100).
  gsm8k     the 100 GSM8K test questions that GSM-Symbolic turned into templates, keyed by their
            GSM8K test index.
  symbolic  GSM-Symbolic's variants of those 100 (names and numbers changed, same reasoning), the
            first --instances of the 50 per template. Paired with gsm8k by template, the gap is
            the memorisation check: a model that learned GSM8K's answers loses them here.

The prompt is the lab's lm-eval format: five worked examples as chat turns ("Question: ...
\\nAnswer:" and the GSM8K solution), then the question; greedy, stop at "Question:". For gsm8k and
symbolic the five examples are the same for every question (GSM8K train, picked with seed 1234),
so the original and its variants differ only in the question. The answer is read with lm-eval's
two filters: flexible (the last number in the reply, the lab's headline figure) and strict
("#### <number>"); both scored with lm-eval's exact match. "correct" is flexible.

One file per set: <out>/gsm-<set>-<label>-<stamp>.json (settings and metrics) and raw/<same>.jsonl.
"""

import argparse
import json
import random
import re
import statistics
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import httpx
from openai import APIError, OpenAI

GSM8K_ID = "openai/gsm8k"
SYMBOLIC_ID = "apple/GSM-Symbolic"
LAB_FILE = Path("results/reference/agentic-edge/lm-eval/tinygsm8k-lab.json")
SETS = ("tiny", "gsm8k", "symbolic")
SHOTS = 5
SHOT_SEED = 1234  # lm-eval's fewshot_seed
MAX_TOKENS = 1024  # the lab's corrected cap (Meta's GSM8K setting); 256 truncated 24% of E2B
STOP = ["Question:", "</s>", "<|im_end|>"]
SEED = 1234  # what lm-eval sends; greedy decoding makes it moot

# lm-eval's gsm8k filters and exact-match normalisation (tinyGSM8k.yaml, lm-eval 0.4.13).
STRICT_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
FLEXIBLE_RE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
IGNORE = [",", r"\$", r"(?s).*#### ", r"\.$"]
INVALID = "[invalid]"


def _regex_filter(text: str, pattern: re.Pattern, group_select: int) -> str:
    found = pattern.findall(text)
    if not found:
        return INVALID
    match = found[group_select]
    if isinstance(match, tuple):
        match = next((m for m in match if m), INVALID)
    return match.strip()


def extract(text: str) -> dict[str, str]:
    """lm-eval's strict-match (first "#### N") and flexible-extract (last number)."""
    return {"strict": _regex_filter(text, STRICT_RE, 0),
            "flexible": _regex_filter(text, FLEXIBLE_RE, -1)}


def _normalise(s: str) -> str:
    for pattern in IGNORE:
        s = re.sub(pattern, "", s)
    return s.lower()


def exact_match(extracted: str, target: str) -> bool:
    """lm-eval's exact_match with the gsm8k task's regexes_to_ignore and ignore_case."""
    return _normalise(extracted) == _normalise(target)


def turn(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def fixed_shots(train: list[dict], n: int = SHOTS, seed: int = SHOT_SEED) -> list[dict]:
    """The chat turns for n GSM8K train questions, picked once with a fixed seed."""
    out = []
    for i in random.Random(seed).sample(range(len(train)), n):
        out += [{"role": "user", "content": turn(train[i]["question"])},
                {"role": "assistant", "content": train[i]["answer"]}]
    return out


def load(name: str, instances: int = 10) -> list[dict]:
    """One set's questions: {id, template, messages, target}. template pairs gsm8k with symbolic."""
    if name == "tiny":
        lab = json.loads(LAB_FILE.read_text())
        return [{"id": k, "template": None, "messages": d["messages"], "target": d["target"]}
                for k, d in lab["docs"].items()]
    from datasets import load_dataset

    shots = fixed_shots(list(load_dataset(GSM8K_ID, "main", split="train")))
    symbolic = load_dataset(SYMBOLIC_ID, "main", split="test")
    if name == "gsm8k":
        test = load_dataset(GSM8K_ID, "main", split="test")
        templates = sorted({int(r["original_id"]) for r in symbolic})
        rows = [(str(t), t, test[t]["question"], test[t]["answer"]) for t in templates]
    else:
        rows = sorted(((f"{r['original_id']}/{r['instance']}", int(r["original_id"]),
                        r["question"], r["answer"])
                       for r in symbolic if int(r["instance"]) < instances),
                      key=lambda r: (r[1], int(r[0].split("/")[1])))
    return [{"id": i, "template": t, "messages": shots + [{"role": "user", "content": turn(q)}],
             "target": a} for i, t, q, a in rows]


def ask(client, model: str, msgs: list[dict], max_tokens: int) -> dict:
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=model, messages=msgs, temperature=0, max_tokens=max_tokens, stop=STOP, seed=SEED,
    )
    latency = time.perf_counter() - start
    choice = resp.choices[0]
    return {
        "raw": choice.message.content or "",
        "finish": choice.finish_reason,
        "latency_s": round(latency, 3),
        "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
        "completion_tokens": getattr(resp.usage, "completion_tokens", None),
    }


def score(row: dict, target: str) -> dict:
    got = extract(row.get("raw", ""))
    return {"flexible": got["flexible"], "strict": got["strict"],
            "correct": exact_match(got["flexible"], target),
            "strict_correct": exact_match(got["strict"], target)}


def summarise(rows: list[dict], wall: float) -> dict:
    ok = [r for r in rows if "error" not in r]
    n = len(rows)
    out = {
        "n": n,
        "accuracy": round(sum(r["correct"] for r in rows) / n, 4) if n else None,
        "strict_accuracy": round(sum(r["strict_correct"] for r in rows) / n, 4) if n else None,
        "no_number": sum(r["flexible"] == INVALID for r in ok),
        "hit_token_cap": sum(r.get("finish") == "length" for r in ok),
        "errors": n - len(ok),
        "median_latency_s": statistics.median(r["latency_s"] for r in ok) if ok else None,
        "wall_s": round(wall, 1),
        "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in ok),
        "completion_tokens": sum(r["completion_tokens"] or 0 for r in ok),
    }
    by_instance = defaultdict(list)
    for r in rows:
        if "/" in str(r["id"]):
            by_instance[int(r["id"].split("/")[1])].append(r["correct"])
    if by_instance:  # GSM-Symbolic reports the spread over instance sets, not one pooled number
        acc = {i: round(sum(v) / len(v), 4) for i, v in sorted(by_instance.items())}
        out["per_instance_accuracy"] = acc
        out["instance_accuracy_range"] = [min(acc.values()), max(acc.values())]
    return out


def main(argv: list[str] | None = None) -> None:
    from .baseline import server_info

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1", help="llama-server endpoint")
    ap.add_argument("--model", default="local", help="model name sent to the server")
    ap.add_argument("--label", default=None, help="run label for the results file")
    ap.add_argument("--set", nargs="+", default=["gsm8k", "symbolic"], choices=SETS, dest="sets")
    ap.add_argument("--instances", type=int, default=10, help="GSM-Symbolic variants per template")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--n", type=int, default=0, help="first n questions per set (0 = all)")
    ap.add_argument("--out", default="results/local", help="directory for this run's files")
    args = ap.parse_args(argv)

    server = server_info(args.base_url)
    client = OpenAI(base_url=args.base_url, api_key="not-needed", timeout=1800, max_retries=3)
    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)

    for name in args.sets:
        todo = load(name, args.instances)[: args.n or None]
        rows, start = [], time.perf_counter()
        for i, q in enumerate(todo, 1):
            row = {"id": q["id"], "template": q["template"], "gold": q["target"].split("#### ")[-1]}
            try:
                row |= ask(client, args.model, q["messages"], args.max_tokens)
            except (APIError, httpx.HTTPError) as e:  # one failed request must not lose a run
                row |= {"error": repr(e)}
            row |= score(row, q["target"])
            rows.append(row)
            print(f"\r{name} {i}/{len(todo)}", end="", flush=True)
        print()
        now = datetime.now(UTC)
        label = args.label or args.model
        run = {
            "stage": "topics",
            "benchmark": "gsm",
            "set": name,
            "label": label,
            "model": args.model,
            "base_url": args.base_url,
            "server": server,
            "n_requested": args.n,
            "generation": {"max_tokens": args.max_tokens, "temperature": 0, "stop": STOP,
                           "shots": SHOTS,
                           "prompt": ("lab's recorded lm-eval tinyGSM8k messages" if name == "tiny"
                                      else f"lm-eval gsm8k chat format, fixed shots (seed {SHOT_SEED})"),
                           **({"instances": args.instances} if name == "symbolic" else {})},
            "time": now.isoformat(timespec="seconds"),
            "metrics": summarise(rows, time.perf_counter() - start),
        }
        stem = f"gsm-{name}-{label}-{now.strftime('%Y%m%d-%H%M%S')}"
        (out / f"{stem}.json").write_text(json.dumps(run, indent=2))
        with open(out / "raw" / f"{stem}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        m = run["metrics"]
        print(f"{name}: {m['accuracy']:.1%} of {m['n']} (strict {m['strict_accuracy']:.1%}), "
              f"no number {m['no_number']}, token cap {m['hit_token_cap']}, errors {m['errors']} "
              f"-> {out / (stem + '.json')}")


if __name__ == "__main__":
    main()
