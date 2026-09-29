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
