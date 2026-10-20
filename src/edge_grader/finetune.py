"""Stage 3: a QLoRA adapter for grading, trained on the SciEntsBank train split (RQ3).

Design: docs/superpowers/specs/2026-10-21-finetuning-design.md. The training text is the
baseline prompt (grader.build_prompt) in Gemma 4's chat template with thinking off, the reply
the 3-way label, so the adapter serves through llama-server's --lora with nothing else changed.
Validation is 10% of the train *questions*; it picks the epoch and nothing else.

    uv run --group finetune edge-grader-finetune --model e2b --out /kaggle/working/adapters/e2b

Torch and friends are imported inside main() only: the Mac and the boards never install them.
"""

import argparse
import hashlib
import json
import random
import shutil
from datetime import UTC, datetime
from pathlib import Path

from . import data
from .data import Item, question_id
from .grader import build_prompt, parse_label

# The QAT checkpoints the lab's UD-Q4_K_XL GGUFs were made from, pinned (not gated).
BASES = {
    "e2b": ("google/gemma-4-E2B-it-qat-q4_0-unquantized", "6befbaca7398925921802abd1f277b495b78b738"),
    "e4b": ("google/gemma-4-E4B-it-qat-q4_0-unquantized", "476025a01dbf99361c062bbeca3d6a76bb4c4566"),
}
# Language-model projections only: the vision and audio towers stay untouched.
TARGET_MODULES = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
# The one recipe (spec 5.1); epochs are a ceiling, validation picks one of them.
RECIPE = {"rank": 16, "alpha": 32, "dropout": 0.05, "lr": 2e-4, "batch": 4, "grad_accum": 4,
          "scheduler": "cosine", "warmup_ratio": 0.03, "quant": "nf4, double quant, fp16 compute",
          "val_frac": 0.1, "max_new_tokens": 16}


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


def training_kwargs(out: Path, epochs: int, max_steps: int, seed: int) -> dict:
    """TrainingArguments for the recipe. transformers 5 has no warmup_ratio: a warmup_steps
    below 1 is read as a fraction of the steps."""
    return {"output_dir": str(out / "trainer"), "per_device_train_batch_size": RECIPE["batch"],
            "gradient_accumulation_steps": RECIPE["grad_accum"], "learning_rate": RECIPE["lr"],
            "lr_scheduler_type": RECIPE["scheduler"], "warmup_steps": RECIPE["warmup_ratio"],
            "num_train_epochs": epochs, "max_steps": max_steps or -1, "fp16": True,
            "logging_steps": 10, "save_strategy": "no", "report_to": [], "seed": seed,
            "remove_unused_columns": False}


def prepare_for_lora(model) -> None:
    """Freeze the base and turn on non-reentrant checkpointing, in place of peft's
    prepare_model_for_kbit_training: that casts every non-4-bit weight to fp32 (Gemma 4's
    per-layer embeddings alone are 2.35B parameters, out of memory on a T4), and its reentrant
    checkpointing drops the gradient through the KV that Gemma 4's shared layers reuse."""
    for p in model.parameters():
        p.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(BASES), required=True)
    ap.add_argument("--out", type=Path, required=True, help="adapter checkpoints go here")
    ap.add_argument("--results", type=Path, default=Path("results/finetune"), help="summary JSON")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (the Day 1 smoke run)")
    ap.add_argument("--val-n", type=int, default=0, help="validate on the first N only (smoke)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="show the split and one example, no torch")
    args = ap.parse_args(argv)

    labels = data.labels_for("3way")
    train, val = split_by_question(data.load(split="train", scheme="3way"), RECIPE["val_frac"], args.seed)
    tq, vq = sorted({question_id(i.id) for i in train}), sorted({question_id(i.id) for i in val})
    print(f"train {len(train)} answers / {len(tq)} questions, "
          f"validation {len(val)} answers / {len(vq)} questions")
    if args.dry_run:
        print(json.dumps(to_messages(train[0], labels), indent=2))
        return
    if args.val_n:
        val = val[:args.val_n]

    import bitsandbytes
    import peft
    import torch
    import transformers
    from transformers import (
        AutoTokenizer,
        BitsAndBytesConfig,
        Gemma4ForConditionalGeneration,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    base, revision = BASES[args.model]
    tok = AutoTokenizer.from_pretrained(base, revision=revision)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        base, revision=revision, device_map={"": 0}, torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_use_double_quant=True,
                                               bnb_4bit_compute_dtype=torch.float16))
    prepare_for_lora(model)
    model = peft.get_peft_model(model, peft.LoraConfig(
        r=RECIPE["rank"], lora_alpha=RECIPE["alpha"], lora_dropout=RECIPE["dropout"],
        target_modules=TARGET_MODULES, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    def prompt_text(item: Item) -> str:
        return tok.apply_chat_template(to_messages(item, labels)[:1], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)

    def example(item: Item) -> dict:
        # The prompt is tokenized on its own, as llama-server does; loss only on the reply.
        prompt = prompt_text(item)
        full = tok.apply_chat_template(to_messages(item, labels), tokenize=False, enable_thinking=False)
        if not full.startswith(prompt):
            raise ValueError(f"chat template: the conversation doesn't start with the prompt\n"
                             f"{prompt!r}\n{full!r}")
        p = tok(prompt, add_special_tokens=False)["input_ids"]
        r = tok(full[len(prompt):], add_special_tokens=False)["input_ids"]
        return {"input_ids": p + r, "labels": [-100] * len(p) + r}

    def collate(batch: list[dict]) -> dict:
        width = max(len(b["input_ids"]) for b in batch)
        pad = lambda xs, v: [x + [v] * (width - len(x)) for x in xs]
        ids = [b["input_ids"] for b in batch]
        return {"input_ids": torch.tensor(pad(ids, tok.pad_token_id)),
                "attention_mask": torch.tensor(pad([[1] * len(x) for x in ids], 0)),
                "labels": torch.tensor(pad([b["labels"] for b in batch], -100))}

    history: list[dict] = []

    def validate(epoch: float) -> None:
        model.eval()
        right = unparsed = 0
        with torch.no_grad():
            for item in val:
                enc = tok(prompt_text(item), add_special_tokens=False, return_tensors="pt").to(0)
                out = model.generate(**enc, max_new_tokens=RECIPE["max_new_tokens"], do_sample=False)
                pred = parse_label(tok.decode(out[0][enc["input_ids"].shape[1]:],
                                              skip_special_tokens=True), labels)
                right += pred == item.label
                unparsed += pred is None
        model.train()
        loss = next((h["loss"] for h in reversed(trainer.state.log_history) if "loss" in h), None)
        k = len(history) + 1
        history.append({"epoch": k, "at_epoch": round(epoch, 3), "train_loss": loss,
                        "val_accuracy": right / len(val), "val_unparsed": unparsed})
        model.save_pretrained(args.out / f"epoch-{k}")
        print(f"epoch {k} ({epoch:.2f}): loss {loss}, validation {right}/{len(val)}, unparsed {unparsed}")

    class Validate(TrainerCallback):
        def on_epoch_end(self, _args, state, _control, **_kw):
            validate(state.epoch)

    trainer = Trainer(
        model=model, train_dataset=[example(i) for i in train], data_collator=collate,
        callbacks=[Validate()],
        args=TrainingArguments(**training_kwargs(args.out, args.epochs, args.max_steps, args.seed)))
    trainer.train()
    if not history:  # --max-steps ended inside the first epoch
        validate(trainer.state.epoch)

    chosen = pick_epoch(history)
    shutil.copytree(args.out / f"epoch-{chosen}", args.out / "adapter", dirs_exist_ok=True)
    sha = hashlib.sha256((args.out / "adapter" / "adapter_model.safetensors").read_bytes()).hexdigest()
    versions = {m.__name__: m.__version__ for m in (torch, transformers, peft, bitsandbytes)}
    s = summary(args.model, {**RECIPE, **{k: str(v) if isinstance(v, Path) else v
                                          for k, v in vars(args).items()}},
                history, chosen, tq, vq, sha, versions)
    args.results.mkdir(parents=True, exist_ok=True)
    path = args.results / f"finetune-{args.model}-{datetime.now(UTC):%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(s, indent=2) + "\n")
    print(f"chosen epoch {chosen}; adapter {args.out / 'adapter'} sha256 {sha}; summary {path}")
