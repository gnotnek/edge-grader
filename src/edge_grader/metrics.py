"""Scores for predicted vs human labels.

SciEntsBank labels are categories, not an ordered scale, so the SemEval-2013 metrics are
accuracy and macro/weighted F1. QWK only makes sense for ordinal scores (e.g. ASAP-SAS).
"""

from sklearn.metrics import accuracy_score, f1_score


def score(gold: list[str], pred: list[str | None]) -> dict:
    # An unparseable reply counts as wrong, never silently dropped.
    pred = [p if p is not None else "__unparsed__" for p in pred]
    return {
        "n": len(gold),
        "accuracy": round(accuracy_score(gold, pred), 4),
        "macro_f1": round(f1_score(gold, pred, average="macro", labels=sorted(set(gold))), 4),
        "weighted_f1": round(
            f1_score(gold, pred, average="weighted", labels=sorted(set(gold))), 4
        ),
        "unparsed": pred.count("__unparsed__"),
    }
