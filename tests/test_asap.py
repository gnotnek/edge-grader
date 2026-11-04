"""ASAP 2.0: the sample draw, the prompt, the score parser and the QWK summary."""

import pandas as pd
import pytest

from edge_grader import asap


def test_parse_score_takes_the_first_whole_number_1_to_6():
    assert asap.parse_score("4") == 4
    assert asap.parse_score("Score: 3\n") == 3
    assert asap.parse_score("10, I mean 5") == 5  # 10 isn't a 1-6 score
    assert asap.parse_score("4/6") == 4
    assert asap.parse_score("seven") is None
    assert asap.parse_score("0") is None


def test_score_qwk_over_parsed_accuracy_counts_unparsed_as_wrong():
    s = asap.score([1, 2, 3, 4], [1, 2, 3, None])
    assert s["unparsed"] == 1 and s["accuracy"] == 0.75 and s["qwk"] == 1.0
    off = asap.score([1, 2, 3, 4], [2, 3, 4, 5])
    assert off["accuracy"] == 0 and off["adjacent"] == 1.0 and off["qwk"] < 1


def test_prompt_order_and_missing_source():
    rubric = "SCORE OF 6: great"
    essay = {"text": " My essay. "}
    p = asap.build_prompt(essay, {"assignment": "Argue.", "sources": ["Src one", "Src two"]}, rubric)
    assert p.index("SCORE OF 6") < p.index("Argue.") < p.index("[2] Src two") < p.index("My essay.")
    assert p.endswith("Reply with the score only: one whole number from 1 to 6.")
    none = asap.build_prompt(essay, {"assignment": "Argue.", "sources": []}, rubric)
    assert "(The source text is not available.)" in none


def test_sample_is_balanced_over_prompts_and_drops_restricted_sources(tmp_path):
    rows = [{"essay_id": f"{p}{i}", "score": 1 + i % 6, "full_text": "t", "assignment": f"do {p}",
             "prompt_name": p, "source_text_1": "Copyright Restricted" if p == "B" else "src",
             "source_text_2": None, "source_text_3": None, "source_text_4": None}
            for p, k in (("A", 20), ("B", 5), ("C", 20)) for i in range(k)]
    csv = tmp_path / "d.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    essays, prompts = asap.make_sample(csv, n=12, seed=0)
    counts = pd.Series([e["prompt"] for e in essays]).value_counts().to_dict()
    assert counts == {"A": 4, "B": 4, "C": 4}
    assert asap.make_sample(csv, n=12, seed=0)[0] == essays  # fixed seed, same draw
    assert prompts["B"]["sources"] == [] and prompts["A"]["sources"] == ["src"]
    with pytest.raises(KeyError):
        essays[0]["gender"]
