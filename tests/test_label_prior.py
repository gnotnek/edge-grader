import math

from edge_grader.label_prior import corrected, label_scores, lenient

LABELS = ["contradictory", "correct", "incorrect"]


def test_scores_take_the_best_token_that_starts_each_label():
    top = [("correct", -0.2), ("in", -1.5), ("Incorrect", -2.0), ("contr", -3.0), ("The", -4.0)]
    assert label_scores(top, LABELS) == {"correct": -0.2, "incorrect": -1.5, "contradictory": -3.0}


def test_a_label_missing_from_the_top_tokens_gets_the_lowest_logprob():
    top = [("correct", -0.1), (" ", -2.0), ("The", -5.0)]  # whitespace starts no label
    assert label_scores(top, LABELS) == {"correct": -0.1, "incorrect": -5.0, "contradictory": -5.0}


def test_correction_divides_by_the_training_mix():
    scores = {"correct": math.log(0.5), "incorrect": math.log(0.4), "contradictory": math.log(0.1)}
    assert max(scores, key=scores.get) == "correct"
    assert corrected(scores, {"correct": 0.6, "incorrect": 0.3, "contradictory": 0.1}) == "incorrect"


def test_lenient_counts_wrong_answers_called_correct():
    rows = [{"gold": "incorrect", "p": "correct"}, {"gold": "contradictory", "p": "incorrect"},
            {"gold": "correct", "p": "correct"}]
    assert lenient(rows, "p") == 0.5
