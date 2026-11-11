"""Scores for predicted vs human labels.

SciEntsBank labels are categories, not an ordered scale, so the SemEval-2013 metrics are
accuracy and macro/weighted F1. QWK only makes sense for ordinal scores (e.g. ASAP-SAS).
"""

import random
import statistics
from collections import defaultdict

from sklearn.metrics import accuracy_score, f1_score

UNPARSED = "__unparsed__"


def _fill(pred: list[str | None]) -> list[str]:
    # An unparseable reply counts as wrong, never silently dropped.
    return [p if p is not None else UNPARSED for p in pred]


def score(gold: list[str], pred: list[str | None]) -> dict:
    pred = _fill(pred)
    return {
        "n": len(gold),
        "accuracy": round(accuracy_score(gold, pred), 4),
        "macro_f1": round(f1_score(gold, pred, average="macro", labels=sorted(set(gold))), 4),
        "weighted_f1": round(
            f1_score(gold, pred, average="weighted", labels=sorted(set(gold))), 4
        ),
        "unparsed": pred.count(UNPARSED),
    }


def _macro_f1(gold: list[str], pred: list[str], labels: list[str]) -> float:
    # sklearn's f1_score, without its input checks: 2,000 calls on test_ud took minutes.
    total = 0.0
    for c in labels:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        total += 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return total / len(labels)


def ci95(gold: list[str], pred: list[str | None], groups: list[str],
         resamples: int = 2000, seed: int = 0) -> dict:
    """95% percentile bootstrap for accuracy and macro-F1, resampling whole questions: answers
    to one question are graded alike, so resampling single answers would make the range too
    narrow (the student-block bootstrap of Vasquez Ferrer et al. 2026, by question here)."""
    pred = _fill(pred)
    labels = sorted(set(gold))
    rows = defaultdict(list)
    for i, g in enumerate(groups):
        rows[g].append(i)
    keys, rng = list(rows), random.Random(seed)
    acc, f1 = [], []
    for _ in range(resamples):
        idx = [i for k in rng.choices(keys, k=len(keys)) for i in rows[k]]
        g, p = [gold[i] for i in idx], [pred[i] for i in idx]
        acc.append(sum(a == b for a, b in zip(g, p)) / len(idx))
        f1.append(_macro_f1(g, p, labels))

    def bounds(xs):
        cuts = statistics.quantiles(xs, n=40, method="inclusive")  # every 2.5%
        return [round(cuts[0], 4), round(cuts[-1], 4)]

    return {"accuracy": bounds(acc), "macro_f1": bounds(f1), "groups": len(keys),
            "resamples": resamples, "seed": seed, "by": "question"}
