from edge_grader.cascade import front, policies


def row(gold, pred, j, s=1.0):
    return {"gold": gold, "pred": pred, "j": j, "latency_s": s}


def test_cascade_keeps_agreements_and_pays_thinking_only_on_disagreements():
    runs = {  # answer a: the baselines agree (wrongly); answer b: they disagree
        "e2b": {"a": row("correct", "incorrect", 1), "b": row("correct", "incorrect", 1)},
        "e4b": {"a": row("correct", "incorrect", 2), "b": row("correct", "correct", 2)},
        "think-e2b": {"a": row("correct", "correct", 50), "b": row("correct", "incorrect", 50)},
        "think-e4b": {"a": row("correct", "correct", 100), "b": row("correct", "correct", 100)},
    }
    pts = {p["policy"]: p for p in policies(runs)}
    c = pts["cascade e2b+e4b → think-e4b"]
    assert c["accuracy"] == 0.5  # a keeps the agreed wrong label; b takes think-e4b's right one
    assert c["escalated"] == 0.5
    assert c["j_per_answer"] == (3 + 103) / 2
    d = pts["defer e2b+e4b → teacher"]
    assert (d["coverage"], d["accuracy"], d["accuracy_on_kept"], d["j_per_answer"]) == (0.5, None, 0.0, 3)


def test_front_drops_points_beaten_on_both_accuracy_and_energy():
    pts = [{"policy": "cheap", "accuracy": 0.6, "j_per_answer": 2},
           {"policy": "dear", "accuracy": 0.7, "j_per_answer": 100},
           {"policy": "worse", "accuracy": 0.6, "j_per_answer": 50},
           {"policy": "defer", "accuracy": None, "j_per_answer": 1}]
    assert front(pts) == {"cheap", "dear"}
