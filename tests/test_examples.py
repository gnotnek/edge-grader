import json

import numpy as np
import pytest

from edge_grader.compare_examples import plan
from edge_grader.data import Item
from edge_grader.examples import knn, load, random_pick
from edge_grader.grader import build_prompt

LABELS = ["contradictory", "correct", "incorrect"]


def item(id_, label="correct", answer="ans"):
    return Item(id=id_, question=f"Q {id_[:5]}", reference_answer="ref", student_answer=answer, label=label)


def unit(*rows):
    v = np.array(rows, dtype=float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


TRAIN = ["EM.1a.1.1", "EM.1a.2.1", "EM.2b.1.1", "EM.2b.2.1", "EM.2b.3.1"]
TRAIN_VECS = unit([1, 0], [0.6, 0.8], [0.99, 0.1], [0.98, 0.2], [0, 1])


def test_same_question_first_then_fill_and_most_similar_last():
    got = knn(["EM.1a.9.1"], unit([1, 0]), TRAIN, TRAIN_VECS, k=3)["EM.1a.9.1"]
    ids = [g[0] for g in got]
    # both EM.1a answers come in although EM.2b.1.1 / 2.1 score higher than EM.1a.2.1
    assert set(ids[-2:]) == {"EM.1a.1.1", "EM.1a.2.1"} and ids[-1] == "EM.1a.1.1"
    assert ids[0] == "EM.2b.1.1"  # the fill: best of the rest, placed first


def test_unseen_question_takes_the_nearest_of_the_whole_pool_in_ascending_order():
    got = knn(["XX.9z.1.1"], unit([1, 0]), TRAIN, TRAIN_VECS, k=3)["XX.9z.1.1"]
    assert [g[0] for g in got] == ["EM.2b.2.1", "EM.2b.1.1", "EM.1a.1.1"]
    assert [g[1] for g in got] == sorted(g[1] for g in got)


def test_random_is_reproducible_and_from_train_only():
    a = random_pick(["T.1", "T.2"], TRAIN, 3)
    assert a == random_pick(["T.1", "T.2"], list(reversed(TRAIN)), 3)
    assert all(j in TRAIN for hits in a.values() for j, _ in hits)
    assert a != random_pick(["T.1", "T.2"], TRAIN, 3, seed=1)


def write(tmp_path, splits, k=2):
    p = tmp_path / "ex.json"
    p.write_text(json.dumps({"settings": {"k": k, "kind": "knn", "embedder": "x", "order": "o"},
                             "splits": splits}))
    return str(p)


def test_load_keeps_the_nearest_k_in_prompt_order(tmp_path):
    train = [item(i) for i in TRAIN]
    path = write(tmp_path, {"test_ua": {"T.1": [["EM.2b.3.1", 0.1], ["EM.1a.2.1", 0.5], ["EM.1a.1.1", 0.9]]}}, k=3)
    shots, run = load(path, 2, train)
    assert [e.id for e in shots["T.1"]] == ["EM.1a.2.1", "EM.1a.1.1"]
    assert run["k"] == 2 and run["kind"] == "knn"


@pytest.mark.parametrize("splits", [
    {"test_ua": {"T.1": [["NOT.train.1", 0.9]]}},     # an example that isn't a train answer
    {"test_ua": {"EM.1a.1.1": [["EM.1a.2.1", 0.9]]}},  # a "test" answer that is in train
])
def test_load_refuses_anything_but_train_examples(tmp_path, splits):
    with pytest.raises(SystemExit):
        load(write(tmp_path, splits, k=1), 1, [item(i) for i in TRAIN])


def test_load_refuses_more_examples_than_stored(tmp_path):
    with pytest.raises(SystemExit):
        load(write(tmp_path, {}, k=2), 3, [])


def test_examples_go_before_the_question_with_their_gold_labels():
    graded = item("EM.1a.3.1")
    shots = [item("EM.1a.1.1", "incorrect", "far"), item("EM.1a.2.1", "contradictory", "near")]
    prompt = build_prompt(graded, LABELS, examples=shots)
    assert build_prompt(graded, LABELS, examples=[]) == build_prompt(graded, LABELS)
    assert prompt.index("[1] Question") < prompt.index("[2] Question") < prompt.index("\nQuestion: Q EM.1a")
    assert "Student answer: far\nLabel: incorrect\n[2]" in prompt
    assert "Student answer: near\nLabel: contradictory\n\nQuestion:" in prompt
    assert prompt.endswith("Reply with the label only.")


def test_family_is_the_12_tests_fixed_in_the_spec():
    tests = plan()
    assert len(tests) == 12 == len(set(tests))
    assert sum(t[2] for t in tests) == 6
