"""data.load on the held-out grading sets (Beetle, SAF), with a fake Hugging Face loader."""

import pytest

from edge_grader import data

FIVE_WAY = ["correct", "contradictory", "partially_correct_incomplete", "irrelevant", "non_domain"]


class FakeLabel:
    names = FIVE_WAY


class FakeDataset(list):
    def __init__(self, rows, features):
        super().__init__(rows)
        self.features = features


@pytest.fixture
def hf(monkeypatch):
    """Records which (id, split) data.load asked for and returns the rows set for it."""
    import datasets

    calls, rows = [], {}

    def fake(name, split):
        calls.append((name, split))
        return rows[name]

    monkeypatch.setattr(datasets, "load_dataset", fake)
    return calls, rows


def test_beetle_has_scientsbanks_fields_and_labels(hf):
    calls, rows = hf
    rows["nkazi/Beetle"] = FakeDataset(
        [{"id": "FF.Q16.SBJ10.L1.SA186", "question": "Q?", "reference_answer": "r",
          "student_answer": "s", "label": 3}], {"label": FakeLabel()})
    [item] = data.load("test_uq", "3way", dataset="beetle")
    assert calls == [("nkazi/Beetle", "test_uq")]
    assert item.label == "incorrect"  # irrelevant collapses as in SciEntsBank


def test_saf_maps_its_fields_splits_and_labels(hf):
    calls, rows = hf
    rows[data.DATASETS["saf"]] = FakeDataset(
        [{"id": str(i), "question": "Q?", "reference_answer": "r", "provided_answer": f"a{i}",
          "verification_feedback": v, "score": "1.0"}
         for i, v in enumerate(["Correct", "Partially correct", "Incorrect"])], {})
    items = data.load("test_ua", "3way", dataset="saf")
    assert calls == [(data.DATASETS["saf"], "test_unseen_answers")]
    assert [i.student_answer for i in items] == ["a0", "a1", "a2"]
    assert [i.label for i in items] == ["correct", "incorrect", "incorrect"]


def test_saf_has_no_5way_labels(hf):
    with pytest.raises(ValueError):
        data.load("test_ua", "5way", dataset="saf")
