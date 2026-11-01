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

THINK_MAX_TOKENS = 336  # the 320-token thought budget (agentic-edge's) plus the usual 16 for the label
# Thought-block markers as llama-server returns them with --reasoning-format none. Gemma 4's
# channel markers first; <think> as a fallback. The pilot (phase B1 Task 2) pins the real one.
THOUGHT_OPEN_RE = re.compile(r"<\|channel>thought|<think>")
THOUGHT_CLOSE_RE = re.compile(r"<channel\|>|</think>")

# Stage 3b: what a tuned adapter was trained to reply (spec 2026-11-01 §5.2). The prompt is the
# same for all three; "label" is every run before stage 3b.
REPLY_FORMATS = ("label", "reason-first", "label-first")
REASON_FIRST_MAX_TOKENS = 160  # the teacher's 128-token reason cap plus the label line
LABEL_LINE_RE = re.compile(r"label:", re.IGNORECASE)


def answer_part(text: str) -> str | None:
    """The reply after its thought block: all of it if there's no block, None if a thought
    opened and never closed (the grade would otherwise be read from the thoughts)."""
    closes = list(THOUGHT_CLOSE_RE.finditer(text))
    if closes:
        return text[closes[-1].end():]
    return None if THOUGHT_OPEN_RE.search(text) else text


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


def reply_text(label: str, reason: str | None, fmt: str) -> str:
    """The reply an adapter is trained to give."""
    if fmt == "label":
        return label
    if not reason:
        raise ValueError(f"{fmt} needs a reason")
    return f"{reason}\nLabel: {label}" if fmt == "reason-first" else f"{label}\n{reason}"


def parse_reply(text: str, labels: list[str], fmt: str) -> str | None:
    """The label in a reply of the given format; reasons are never read, they name labels too."""
    if fmt == "reason-first":
        marks = list(LABEL_LINE_RE.finditer(text))
        return parse_label(text[marks[-1].end():], labels) if marks else None
    if fmt == "label-first":
        return parse_label(text.split("\n", 1)[0], labels)
    return parse_label(text, labels)


def grade(client, model: str, item: Item, labels: list[str], max_tokens: int = 16,
          passages: list[str] | None = None, thinking: bool = False, reply: str = "label") -> dict:
    if thinking and reply != "label":
        raise ValueError("thinking with a reason format isn't a condition in this study")
    prompt = build_prompt(item, labels, passages)
    if thinking and max_tokens == 16:
        max_tokens = THINK_MAX_TOKENS
    if reply == "reason-first" and max_tokens == 16:
        max_tokens = REASON_FIRST_MAX_TOKENS
    # label-first: stop where the trained reply's label line ends, so the reason costs nothing.
    extra = {"stop": ["\n"]} if reply == "label-first" else {}
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
        **extra,
    )
    latency = time.perf_counter() - start
    choice = resp.choices[0]
    text = choice.message.content or ""
    if thinking:
        part = None if choice.finish_reason == "length" else answer_part(text)
        pred = parse_label(part, labels) if part is not None else None
    elif reply == "reason-first" and choice.finish_reason == "length":
        pred = None  # cut before its label line, or inside it
    else:
        pred = parse_reply(text, labels, reply)
    usage = resp.usage
    row = {
        "id": item.id,
        "gold": item.label,
        "pred": pred,
        "raw": text,
        "latency_s": round(latency, 3),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
    }
    if thinking or reply != "label":
        row["finish"] = choice.finish_reason
    return row
