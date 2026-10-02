from edge_grader.data import TO_2WAY, TO_3WAY, Item, labels_for
from edge_grader.grader import build_prompt, parse_label
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
