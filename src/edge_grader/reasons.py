"""Stage 3b: a teacher model writes a reason for every SciEntsBank train answer.

    uv run edge-grader-reasons --base-url http://127.0.0.1:8080/v1 --n 50   # the pilot
    uv run edge-grader-reasons --base-url http://127.0.0.1:8080/v1          # all 4,969

Design: docs/superpowers/specs/2026-11-01-reasoning-finetune-design.md §5.1. The teacher
(Qwen3-14B in llama-server, thinking off) is given the gold 3-way label and explains it; it
doesn't grade. A reason that is empty, cut at the token cap, holds a "Label:" line or names a
label other than the gold one is kept in the file with a drop code, and isn't trained on.
Each reason is saved as it comes (raw/<name>.jsonl.part), so a stopped run resumes.
"""

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from openai import APIError, OpenAI

from . import data
from .baseline import server_info
from .data import Item
from .grader import LABEL_HELP, LABEL_LINE_RE
from .news import THINK_RE

PROMPT = Path(__file__).parent / "prompts" / "reason_teacher.txt"
MAX_TOKENS = 128  # about 60 words, with room; a reason cut here is dropped


def prompt_sha256() -> str:
    return hashlib.sha256(PROMPT.read_bytes()).hexdigest()


def teacher_messages(item: Item) -> list[dict]:
    return [{"role": "user", "content": PROMPT.read_text().strip().format(
        question=item.question, reference_answer=item.reference_answer,
        student_answer=item.student_answer, label=item.label, help=LABEL_HELP[item.label])}]


def clean(text: str) -> str:
    """One line: Qwen3's empty think block removed, whitespace collapsed."""
    return " ".join(THINK_RE.sub("", text).split())


def check_reason(text: str, finish: str, gold: str, labels: list[str]) -> str | None:
    """The drop code, or None for a reason to train on."""
    if not text.strip():
        return "empty"
    if finish == "length":
        return "cut"
    if LABEL_LINE_RE.search(text):
        return "label_line"  # would be read as the label in a reason-first reply
    lowered = text.lower()
    if any(re.search(rf"\b{re.escape(name)}\b", lowered) for name in labels if name != gold):
        return "wrong_label"
    return None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1", help="the teacher's llama-server")
    ap.add_argument("--model", default="local", help="model name sent to the server")
    ap.add_argument("--n", type=int, default=0, help="answers (0 = the whole train split; a pilot: 50)")
    ap.add_argument("--out", type=Path, default=Path("results/finetune"))
    args = ap.parse_args(argv)

    server = server_info(args.base_url)
    teacher = Path(server["model_path"] or "teacher").stem.lower()
    sha = prompt_sha256()
    # a pilot carries the prompt sha, so trials of different prompts keep their own files
    name = f"reasons-{teacher}" + (f"-n{args.n}-{sha[:8]}" if args.n else "")
    labels = data.labels_for("3way")
    items = data.load(split="train", scheme="3way", n=args.n or None, seed=0)
    raw = args.out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    part = raw / f"{name}.jsonl.part"
    done = {}
    if part.exists():  # an earlier attempt stopped part-way: keep its reasons
        done = {r["id"]: r for r in map(json.loads, part.read_text().splitlines())}
        if any(r["prompt_sha256"] != sha for r in done.values()):
            sys.exit(f"{part} was written with another prompt; move it away to start again")
        print(f"resuming: {len(done)} of {len(items)} already written")

    client = OpenAI(base_url=args.base_url, api_key="not-needed", timeout=600, max_retries=3)
    start = time.perf_counter()
    with open(part, "a") as f:
        for i, item in enumerate(items, 1):
            if item.id in done:
                continue
            try:
                resp = client.chat.completions.create(model=args.model, messages=teacher_messages(item),
                                                      temperature=0, max_tokens=MAX_TOKENS)
            except APIError as e:
                print()
                sys.exit(f"stopped at {len(done)}/{len(items)}, progress saved in {part}; "
                         f"run the same command again to resume.\n{e}")
            choice = resp.choices[0]
            reason = clean(choice.message.content or "")
            r = {"id": item.id, "gold": item.label, "reason": reason, "finish": choice.finish_reason,
                 "drop": check_reason(reason, choice.finish_reason, item.label, labels),
                 "prompt_sha256": sha}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            done[item.id] = r
            print(f"\r{i}/{len(items)}", end="", flush=True)
    print()

    rows = [done[i.id] for i in items]
    out = raw / f"{name}.jsonl"
    with open(out, "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    part.unlink()
    drops = Counter(r["drop"] for r in rows if r["drop"])
    now = datetime.now(UTC)
    summary = {
        "kind": "reasons",
        "teacher": server,
        "model": args.model,
        "settings": {"temperature": 0, "max_tokens": MAX_TOKENS, "split": "train", "scheme": "3way",
                     "n_requested": args.n, "seed": 0},
        "prompt": str(PROMPT.relative_to(Path(__file__).parent)),
        "prompt_sha256": sha,
        "n": len(rows),
        "kept": len(rows) - sum(drops.values()),
        "drops": dict(sorted(drops.items())),
        "drop_share": round(sum(drops.values()) / len(rows), 4) if rows else None,
        "reasons": str(out),
        "reasons_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "wall_s": round(time.perf_counter() - start, 1),
        "time": now.isoformat(timespec="seconds"),
    }
    path = args.out / f"{name}-{now:%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"kept {summary['kept']} of {summary['n']}, dropped {summary['drops']} -> {out}, {path}")


if __name__ == "__main__":
    main()
