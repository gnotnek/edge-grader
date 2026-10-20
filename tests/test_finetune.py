import re

from edge_grader.data import Item, labels_for, question_id
from edge_grader.finetune import (
    BASES,
    TARGET_MODULES,
    pick_epoch,
    split_by_question,
    summary,
    to_messages,
)
from edge_grader.grader import build_prompt

LABELS = labels_for("3way")


def items(questions: int = 20, answers: int = 5) -> list[Item]:
    return [Item(id=f"EM.{q}b.{a}.1", question=f"q{q}", reference_answer="ref",
                 student_answer=f"ans {a}", label=LABELS[(q + a) % 3])
            for q in range(questions) for a in range(answers)]


def qids(part: list[Item]) -> set[str]:
    return {question_id(i.id) for i in part}


def test_split_is_by_question_and_deterministic():
    train, val = split_by_question(items(), frac=0.1, seed=0)
    assert len(qids(val)) == 2 and len(val) == 10
    assert not qids(train) & qids(val)
    again = split_by_question(items(), frac=0.1, seed=0)
    assert [i.id for i in again[1]] == [i.id for i in val]
    other = split_by_question(items(), frac=0.1, seed=1)
    assert qids(other[1]) != qids(val)


def test_split_keeps_every_item():
    train, val = split_by_question(items(), frac=0.1, seed=0)
    assert len(train) + len(val) == len(items())


def test_messages_use_the_baseline_prompt():
    item = items()[0]
    user, reply = to_messages(item, LABELS)
    assert user == {"role": "user", "content": build_prompt(item, LABELS)}
    assert reply == {"role": "assistant", "content": item.label}


def test_pick_epoch_best_then_earliest():
    history = [{"epoch": 1, "val_accuracy": 0.6}, {"epoch": 2, "val_accuracy": 0.7},
               {"epoch": 3, "val_accuracy": 0.7}]
    assert pick_epoch(history) == 2


def test_targets_language_model_only():
    for name in ("model.language_model.layers.0.self_attn.q_proj",
                 "model.language_model.layers.3.mlp.down_proj"):
        assert re.fullmatch(TARGET_MODULES, name), name
    for name in ("model.vision_tower.encoder.layers.0.self_attn.q_proj",
                 "model.audio_tower.layers.0.self_attn.q_proj",
                 "model.language_model.embed_tokens"):
        assert not re.fullmatch(TARGET_MODULES, name), name


def test_summary_records_settings():
    s = summary("e2b", {"epochs": 3}, [{"epoch": 1, "val_accuracy": 0.5}], 1, ["EM.1b"], ["EM.2b"],
                "abc", {"peft": "0.1"})
    assert {"model", "base", "revision", "settings", "history", "chosen_epoch", "train_questions",
            "val_questions", "adapter_sha256", "versions", "created"} <= s.keys()
    assert (s["base"], s["revision"]) == BASES["e2b"]
