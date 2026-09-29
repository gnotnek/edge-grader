"""Load SciEntsBank (SemEval-2013 Task 7) student answers.

Test splits: unseen answers (test_ua), unseen questions (test_uq), unseen domains (test_ud).
The Hugging Face id below is a community mirror; check its card and licence before relying on it.
"""

from dataclasses import dataclass

DATASET_ID = "nkazi/SciEntsBank"

# The 5-way labels collapse to the 3-way and 2-way SemEval schemes.
TO_3WAY = {
    "correct": "correct",
    "contradictory": "contradictory",
    "partially_correct_incomplete": "incorrect",
    "irrelevant": "incorrect",
    "non_domain": "incorrect",
}
TO_2WAY = {k: ("correct" if v == "correct" else "incorrect") for k, v in TO_3WAY.items()}
SCHEMES = {
    "5way": {k: k for k in TO_3WAY},
    "3way": TO_3WAY,
    "2way": TO_2WAY,
}


@dataclass
class Item:
    id: str
    question: str
    reference_answer: str
    student_answer: str
    label: str  # in the chosen scheme


def labels_for(scheme: str) -> list[str]:
    return sorted(set(SCHEMES[scheme].values()))


def load(split: str = "test_ua", scheme: str = "3way", n: int | None = None, seed: int = 0):
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split=split)
    if n is not None and n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n))
    label_feature = ds.features["label"]
    names = getattr(label_feature, "names", None)
    mapping = SCHEMES[scheme]
    items = []
    for row in ds:
        raw = names[row["label"]] if names else row["label"]
        items.append(
            Item(
                id=str(row["id"]),
                question=row["question"],
                reference_answer=row["reference_answer"],
                student_answer=row["student_answer"],
                label=mapping[raw],
            )
        )
    return items
