"""Grade one answer by prompting a model behind an OpenAI-compatible server (llama-server)."""

import re
import time

from .data import Item

LABEL_HELP = {
    "correct": "a complete and correct answer",
    "contradictory": "contradicts the reference answer",
    "incorrect": "partially correct, incomplete, irrelevant, or off-topic",
    "partially_correct_incomplete": "partly correct but missing parts",
    "irrelevant": "on topic but does not answer the question",
    "non_domain": "not about the subject at all",
}


def build_prompt(item: Item, labels: list[str], passages: list[str] | None = None) -> str:
    """The stage-1 prompt; with passages (RAG), course material goes before the question."""
    options = "\n".join(f"- {name}: {LABEL_HELP[name]}" for name in labels)
    material = ""
    if passages:
        numbered = "\n".join(f"[{i}] {p}" for i, p in enumerate(passages, 1))
        material = (
            "Course material, for background only (the reference answer decides the grade):\n"
            f"{numbered}\n\n"
        )
    return (
        "You are grading a student's answer to a science question.\n\n"
        f"{material}"
        f"Question: {item.question}\n"
        f"Reference answer: {item.reference_answer}\n"
        f"Student answer: {item.student_answer}\n\n"
        f"Choose exactly one label:\n{options}\n\n"
        "Reply with the label only."
    )


def parse_label(text: str, labels: list[str]) -> str | None:
    """Return the first label mentioned in the reply, or None."""
    lowered = text.lower()
    hits = []
    for name in labels:
        m = re.search(rf"\b{re.escape(name)}\b", lowered)
        if m:
            hits.append((m.start(), name))
    # "incorrect" contains "correct" only as a substring, and \b keeps them apart.
    return min(hits)[1] if hits else None


def grade(client, model: str, item: Item, labels: list[str], max_tokens: int = 16,
          passages: list[str] | None = None) -> dict:
    prompt = build_prompt(item, labels, passages)
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - start
    text = resp.choices[0].message.content or ""
    usage = resp.usage
    return {
        "id": item.id,
        "gold": item.label,
        "pred": parse_label(text, labels),
        "raw": text,
        "latency_s": round(latency, 3),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
    }
