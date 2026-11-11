"""Load SciEntsBank (SemEval-2013 Task 7) student answers.

Test splits: unseen answers (test_ua), unseen questions (test_uq), unseen domains (test_ud).
The Hugging Face id below is a community mirror; check its card and licence before relying on it.
"""

from dataclasses import dataclass

DATASET_ID = "nkazi/SciEntsBank"
# Held-out grading sets, never trained on: Beetle is SemEval-2013 Task 7's other corpus (same
# fields and labels); SAF (Filighera et al., ACL 2022) is communication networks, in English.
DATASETS = {
    "scientsbank": DATASET_ID,
    "beetle": "nkazi/Beetle",
    "saf": "Short-Answer-Feedback/saf_communication_networks_english",
}
SAF_SPLITS = {"test_ua": "test_unseen_answers", "test_uq": "test_unseen_questions"}
# SAF has no "contradictory"; partially correct collapses to incorrect, as in SciEntsBank's 3-way.
SAF_LABELS = {"Correct": "correct", "Partially correct": "incorrect", "Incorrect": "incorrect"}

# The 5-way labels collapse to the 3-way and 2-way SemEval schemes.
TO_3WAY = {
    "correct": "correct",
    "contradictory": "contradictory",
    "partially_correct_incomplete": "incorrect",
    "irrelevant": "incorrect",
    "non_domain": "incorrect",
}
TO_2WAY = {k: ("correct" if v == "correct" else "incorrect") for k, v in TO_3WAY.items()}


def to_3way(label: str | None) -> str | None:
    """A 5-way or 3-way label as 3-way; None (unparsed) stays None."""
    return None if label is None else TO_3WAY.get(label, label)
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


def question_id(answer_id: str) -> str:
    """'EM.45b.299.1' -> 'EM.45b': the unit and question parts of an answer id."""
    return ".".join(answer_id.split(".")[:2])


def labels_for(scheme: str) -> list[str]:
    return sorted(set(SCHEMES[scheme].values()))


def load(split: str = "test_ua", scheme: str = "3way", n: int | None = None, seed: int = 0,
         dataset: str = "scientsbank"):
    from datasets import load_dataset

    saf = dataset == "saf"
    if saf and scheme == "5way":
        raise ValueError("SAF has 3 labels, not SemEval's 5: use 3way or 2way")
    ds = load_dataset(DATASETS[dataset], split=SAF_SPLITS.get(split, split) if saf else split)
    if n is not None and n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n))
    label_feature = ds.features.get("label")
    names = getattr(label_feature, "names", None)
    mapping = SCHEMES[scheme]
    items = []
    for row in ds:
        if saf:
            label, answer = SAF_LABELS[row["verification_feedback"]], row["provided_answer"]
        else:
            label = mapping[names[row["label"]] if names else row["label"]]
            answer = row["student_answer"]
        items.append(
            Item(
                id=str(row["id"]),
                question=row["question"],
                reference_answer=row["reference_answer"],
                student_answer=answer,
                label=label,
            )
        )
    return items
