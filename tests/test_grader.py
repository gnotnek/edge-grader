import json
from pathlib import Path

import pytest

from edge_grader.data import TO_2WAY, TO_3WAY, Item, labels_for
from edge_grader.grader import (
    REASON_FIRST_MAX_TOKENS,
    REPLY_FORMATS,
    THINK_MAX_TOKENS,
    answer_part,
    build_prompt,
    grade,
    parse_label,
    parse_reply,
    reply_text,
)
from edge_grader.metrics import score

LABELS = labels_for("3way")


def test_parse_keeps_correct_and_incorrect_apart():
    assert parse_label("incorrect", LABELS) == "incorrect"
    assert parse_label("Correct.", LABELS) == "correct"
    assert parse_label("Label: contradictory", LABELS) == "contradictory"


def test_parse_takes_first_label_mentioned():
    assert parse_label("incorrect, not correct", LABELS) == "incorrect"


def test_parse_returns_none_when_no_label():
    assert parse_label("I am not sure", LABELS) is None


def test_label_schemes_cover_all_five_way_labels():
    assert set(TO_3WAY.values()) == {"correct", "contradictory", "incorrect"}
    assert set(TO_2WAY.values()) == {"correct", "incorrect"}


def test_prompt_contains_answer_and_options():
    item = Item("1", "Why is the sky blue?", "Scattering.", "Because of scattering.", "correct")
    prompt = build_prompt(item, LABELS)
    assert "Because of scattering." in prompt
    assert all(name in prompt for name in LABELS)


def test_unparsed_counts_as_wrong():
    s = score(["correct", "incorrect"], ["correct", None])
    assert s["accuracy"] == 0.5
    assert s["unparsed"] == 1


BASELINE_PROMPT = (
    "You are grading a student's answer to a science question.\n\n"
    "Question: Why is the sky blue?\n"
    "Reference answer: Scattering.\n"
    "Student answer: Because of scattering.\n\n"
    "Choose exactly one label:\n"
    "- contradictory: contradicts the reference answer\n"
    "- correct: a complete and correct answer\n"
    "- incorrect: partially correct, incomplete, irrelevant, or off-topic\n\n"
    "Reply with the label only."
)


def test_prompt_without_passages_is_the_stage_1_prompt():
    # Stage-1 numbers stay comparable only if this text never changes.
    item = Item("1", "Why is the sky blue?", "Scattering.", "Because of scattering.", "correct")
    assert build_prompt(item, LABELS) == BASELINE_PROMPT
    assert build_prompt(item, LABELS, passages=[]) == BASELINE_PROMPT


def test_prompt_with_passages_numbers_them_and_keeps_the_instruction_last():
    item = Item("1", "Why is the sky blue?", "Scattering.", "Because of scattering.", "correct")
    prompt = build_prompt(item, LABELS, passages=["Light scatters off air.", "Blue scatters most."])
    assert "[1] Light scatters off air.\n[2] Blue scatters most." in prompt
    assert "the reference answer decides the grade" in prompt
    assert prompt.index("[2]") < prompt.index("Question:")
    assert prompt.endswith("Reply with the label only.")


def test_answer_part_without_a_thought_block_is_the_whole_reply():
    assert answer_part("correct") == "correct"


def test_thinking_reads_the_label_after_the_thought_block():
    for opened, closed in (("<|channel>thought\n", "<channel|>"), ("<think>", "</think>")):
        reply = f"{opened}It is not contradictory; maybe incorrect.{closed}correct"
        assert answer_part(reply) == "correct"
        assert parse_label(answer_part(reply), LABELS) == "correct"


def test_unclosed_thought_or_cut_reply_is_unparsed():
    assert answer_part("<|channel>thought\nIt is incorrect because") is None

    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    class R:
                        class usage:
                            prompt_tokens, completion_tokens = 100, THINK_MAX_TOKENS
                        choices = (type("C", (), {"finish_reason": "length",
                                   "message": type("M", (), {"content": "<think>x</think>incor"})}),)
                    assert kw["max_tokens"] == THINK_MAX_TOKENS
                    return R

    item = Item("1", "Q?", "ref", "ans", "correct")
    row = grade(Client, "m", item, LABELS, thinking=True)
    assert row["pred"] is None and row["finish"] == "length"


def test_a_real_thinking_reply_parses_to_its_final_label():
    """A Jetson reply (B1, -rea on --reasoning-format none): the markers as Gemma 4 emits them."""
    rec = json.loads((Path(__file__).parent / "fixtures" / "thinking_reply.json").read_text())
    assert parse_label(rec["raw"], LABELS) != rec["label"]  # the thoughts name another label first
    assert parse_label(answer_part(rec["raw"]), LABELS) == rec["label"]


# Stage 3b reply formats (docs/superpowers/specs/2026-11-01-reasoning-finetune-design.md §5.2).
ALL_WORDS = "It is not correct, nor incorrect, nor contradictory."


def fake_client(content: str, finish: str = "stop"):
    seen = {}

    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    seen.update(kw)

                    class R:
                        class usage:
                            prompt_tokens, completion_tokens = 100, 5
                        choices = (type("C", (), {"finish_reason": finish,
                                   "message": type("M", (), {"content": content})}),)
                    return R

    return Client, seen


def test_reason_first_reads_the_label_after_the_last_label_line():
    reply = "The answer names the wrong force, so it is not correct.\nLabel: contradictory"
    assert parse_reply(reply, LABELS, "reason-first") == "contradictory"
    assert parse_reply("Label: correct\nmore\nLABEL: incorrect", LABELS, "reason-first") == "incorrect"
    assert parse_reply("It is correct.", LABELS, "reason-first") is None


def test_label_first_reads_the_first_line_only():
    assert parse_reply("incorrect\nIt leaves out the correct idea.", LABELS, "label-first") == "incorrect"
    assert parse_reply("\nIt is correct.", LABELS, "label-first") is None


def test_label_format_is_parse_label():
    assert parse_reply("incorrect, not correct", LABELS, "label") == "incorrect"


def test_reply_text_formats():
    assert reply_text("correct", "r", "reason-first") == "r\nLabel: correct"
    assert reply_text("correct", "r", "label-first") == "correct\nr"
    assert reply_text("correct", None, "label") == "correct"
    with pytest.raises(ValueError):
        reply_text("correct", None, "reason-first")


@pytest.mark.parametrize("fmt", REPLY_FORMATS)
@pytest.mark.parametrize("label", LABELS)
def test_written_reply_parses_back_to_its_label(fmt, label):
    assert parse_reply(reply_text(label, ALL_WORDS, fmt), LABELS, fmt) == label


def test_label_request_is_unchanged():
    client, seen = fake_client("correct")
    row = grade(client, "m", Item("1", "Q?", "ref", "ans", "correct"), LABELS)
    assert seen["max_tokens"] == 16 and "stop" not in seen
    assert row["pred"] == "correct" and "finish" not in row


def test_label_first_stops_at_the_newline():
    client, seen = fake_client("incorrect")
    row = grade(client, "m", Item("1", "Q?", "ref", "ans", "incorrect"), LABELS, reply="label-first")
    assert seen["max_tokens"] == 16 and seen["stop"] == ["\n"]
    assert row["pred"] == "incorrect" and row["finish"] == "stop"


def test_reason_first_cut_by_length_is_unparsed():
    client, seen = fake_client("It is correct because\nLabel: corr", finish="length")
    row = grade(client, "m", Item("1", "Q?", "ref", "ans", "correct"), LABELS, reply="reason-first")
    assert seen["max_tokens"] == REASON_FIRST_MAX_TOKENS
    assert row["pred"] is None and row["finish"] == "length"


def test_thinking_with_a_reason_format_is_refused():
    client, _ = fake_client("correct")
    with pytest.raises(ValueError):
        grade(client, "m", Item("1", "Q?", "ref", "ans", "correct"), LABELS, thinking=True,
              reply="label-first")
