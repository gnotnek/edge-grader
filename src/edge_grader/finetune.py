"""Stage 3: a QLoRA adapter for grading, trained on the SciEntsBank train split (RQ3).

Design: docs/superpowers/specs/2026-10-21-finetuning-design.md. The training text is the
baseline prompt (grader.build_prompt) in Gemma 4's chat template with thinking off, the reply
the 3-way label, so the adapter serves through llama-server's --lora with nothing else changed.
Validation is 10% of the train *questions*; it picks the epoch and nothing else.

    uv run --extra finetune edge-grader-finetune --model e2b --out /kaggle/working/adapters/e2b

Torch and friends are imported inside main() only: the Mac and the boards never install them.
"""

import random
from datetime import UTC, datetime

from .data import Item, question_id
from .grader import build_prompt

# The QAT checkpoints the lab's UD-Q4_K_XL GGUFs were made from, pinned (not gated).
BASES = {
    "e2b": ("google/gemma-4-E2B-it-qat-q4_0-unquantized", "6befbaca7398925921802abd1f277b495b78b738"),
    "e4b": ("google/gemma-4-E4B-it-qat-q4_0-unquantized", "476025a01dbf99361c062bbeca3d6a76bb4c4566"),
}
# Language-model projections only: the vision and audio towers stay untouched.
TARGET_MODULES = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def split_by_question(items: list[Item], frac: float = 0.1, seed: int = 0) -> tuple[list[Item], list[Item]]:
    """(train, validation): every answer to a held-out question goes to validation."""
    questions = sorted({question_id(i.id) for i in items})
    random.Random(seed).shuffle(questions)
    held = set(questions[:max(1, round(frac * len(questions)))])
    return ([i for i in items if question_id(i.id) not in held],
            [i for i in items if question_id(i.id) in held])


def to_messages(item: Item, labels: list[str]) -> list[dict]:
    return [{"role": "user", "content": build_prompt(item, labels)},
            {"role": "assistant", "content": item.label}]


def pick_epoch(history: list[dict]) -> int:
    """The epoch with the best validation accuracy, the earliest on a tie."""
    return max(history, key=lambda h: (h["val_accuracy"], -h["epoch"]))["epoch"]


def summary(model_key: str, args: dict, history: list[dict], chosen: int, train_q: list[str],
            val_q: list[str], adapter_sha256: str, versions: dict) -> dict:
    base, revision = BASES[model_key]
    return {"model": model_key, "base": base, "revision": revision, "settings": args,
            "history": history, "chosen_epoch": chosen, "train_questions": train_q,
            "val_questions": val_q, "adapter_sha256": adapter_sha256, "versions": versions,
            "created": datetime.now(UTC).isoformat(timespec="seconds")}
