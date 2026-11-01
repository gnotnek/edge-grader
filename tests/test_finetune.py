import hashlib
import json
import re
import sys

import pytest

from edge_grader.data import Item, labels_for, question_id
from edge_grader.finetune import (
    BASES,
    TARGET_MODULES,
    load_reasons,
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


def test_dry_run_needs_no_torch(tmp_path, monkeypatch, capsys):
    from edge_grader import data, finetune

    monkeypatch.setattr(data, "load", lambda split, scheme: items(questions=10, answers=3))
    monkeypatch.setitem(sys.modules, "torch", None)  # any `import torch` now raises
    finetune.main(["--model", "e2b", "--out", str(tmp_path), "--dry-run"])
    out = capsys.readouterr().out
    assert "train 27 answers / 9 questions, validation 3 answers / 1 questions" in out
    assert "You are grading a student's answer" in out


def test_training_arguments_construct(tmp_path):
    transformers = pytest.importorskip("transformers")
    from edge_grader.finetune import RECIPE, training_kwargs

    kw = training_kwargs(tmp_path, epochs=3, max_steps=0, seed=0)
    assert kw["fp16"] is True and kw["max_steps"] == -1
    args = transformers.TrainingArguments(**{**kw, "fp16": False})  # fp16 needs a GPU here
    assert args.warmup_steps == RECIPE["warmup_ratio"]
    assert args.per_device_train_batch_size * args.gradient_accumulation_steps == 16


def test_prepare_freezes_and_uses_non_reentrant_checkpointing():
    torch = pytest.importorskip("torch")
    from edge_grader.finetune import prepare_for_lora

    class Model(torch.nn.Linear):
        def gradient_checkpointing_enable(self, **kw):
            self.checkpointing = kw

    m = Model(2, 2)
    prepare_for_lora(m)
    assert not any(p.requires_grad for p in m.parameters())
    assert m.checkpointing == {"gradient_checkpointing_kwargs": {"use_reentrant": False}}


# Stage 3b: training on a reply format with teacher reasons.

def reasons_file(tmp_path, part: list[Item], dropped: set[str]):
    path = tmp_path / "reasons.jsonl"
    path.write_text("".join(json.dumps({"id": i.id, "gold": i.label, "reason": f"why {i.id}",
                                        "drop": "wrong_label" if i.id in dropped else None}) + "\n"
                            for i in part))
    return path


def test_messages_for_reason_formats():
    item = items()[0]
    assert to_messages(item, LABELS, "reason-first", "r")[1]["content"] == f"r\nLabel: {item.label}"
    assert to_messages(item, LABELS, "label-first", "r")[1]["content"] == f"{item.label}\nr"
    assert to_messages(item, LABELS, "label-first", "r")[0] == to_messages(item, LABELS)[0]


def test_load_reasons_skips_dropped_and_hashes_the_file(tmp_path):
    part = items(questions=2, answers=2)
    path = reasons_file(tmp_path, part, {part[1].id})
    kept, sha = load_reasons(path)
    assert set(kept) == {part[0].id, part[2].id, part[3].id}
    assert kept[part[0].id] == f"why {part[0].id}"
    assert sha == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("reply", ["reason-first", "label-first"])
def test_reason_formats_drop_the_same_answers_and_keep_the_split(reply, tmp_path, monkeypatch, capsys):
    from edge_grader import data, finetune

    everything = items(questions=10, answers=3)
    train, _ = split_by_question(everything, 0.1, 0)
    dropped = {train[0].id, train[5].id}
    path = reasons_file(tmp_path, everything, dropped)
    monkeypatch.setattr(data, "load", lambda split, scheme: everything)
    monkeypatch.setitem(sys.modules, "torch", None)
    finetune.main(["--model", "e2b", "--out", str(tmp_path), "--dry-run", "--reply", reply,
                   "--reasons", str(path)])
    out = capsys.readouterr().out
    # The same split as the label-only adapter; validation keeps every answer.
    assert "train 27 answers / 9 questions, validation 3 answers / 1 questions" in out
    assert "training on 25 answers: 2 dropped (no kept reason)" in out
    assert f"why {train[1].id}" in out


def test_a_reason_format_needs_reasons(tmp_path):
    from edge_grader import finetune

    with pytest.raises(SystemExit):
        finetune.main(["--model", "e2b", "--out", str(tmp_path), "--dry-run", "--reply", "label-first"])
