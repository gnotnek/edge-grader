from edge_grader.compare_boundary import label_mix


def test_label_mix_recall_and_shares():
    rows = [("correct", "correct"), ("correct", "incorrect"), ("contradictory", "incorrect"),
            ("incorrect", "incorrect")]
    m = label_mix({str(i): {"gold": g, "pred": p} for i, (g, p) in enumerate(rows)})
    assert m["correct"] == {"recall": 0.5, "pred_share": 0.25, "gold_share": 0.5}
    assert m["contradictory"]["recall"] == 0.0
    assert m["incorrect"] == {"recall": 1.0, "pred_share": 0.75, "gold_share": 0.25}
