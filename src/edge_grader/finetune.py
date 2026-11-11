"""Stage 3: a QLoRA adapter for grading, trained on the SciEntsBank train split (RQ3).

Design: docs/superpowers/specs/2026-10-21-finetuning-design.md. The training text is the
baseline prompt (grader.build_prompt) in Gemma 4's chat template with thinking off, the reply
the 3-way label, so the adapter serves through llama-server's --lora with nothing else changed.
Validation is 10% of the train *questions*; it picks the epoch and nothing else.

Stage 3b (spec 2026-11-01-reasoning-finetune-design.md): --reply reason-first / label-first trains
the same prompt on a reply with a teacher's reason (edge-grader-reasons). Answers whose reason was
dropped are left out of training only; the split and the validation answers stay as above.

--train-sets trains on other grading sets instead, e.g. beetle saf, so SciEntsBank stays unseen.

--scheme 5way (spec 2026-11-12-fiveway-training-design.md) trains on SciEntsBank's 5-way labels
with the 5-label prompt; the epoch is still picked on 3-way validation accuracy, the reported metric.

    uv run --group finetune edge-grader-finetune --model e2b --out /kaggle/working/adapters/e2b
    uv run --group finetune edge-grader-finetune --model e2b --out ... --reply reason-first \\
        --reasons results/finetune/raw/reasons-qwen3-14b-q4_k_m.jsonl

Torch and friends are imported inside main() only: the Mac and the boards never install them.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
from datetime import UTC, datetime
from pathlib import Path

from . import data
from .data import Item, question_id
from .grader import REASON_FIRST_MAX_TOKENS, REPLY_FORMATS, build_prompt, parse_reply, reply_text

# The QAT checkpoints the lab's UD-Q4_K_XL GGUFs were made from, pinned (not gated).
BASES = {
    "e2b": ("google/gemma-4-E2B-it-qat-q4_0-unquantized", "6befbaca7398925921802abd1f277b495b78b738"),
    "e4b": ("google/gemma-4-E4B-it-qat-q4_0-unquantized", "476025a01dbf99361c062bbeca3d6a76bb4c4566"),
}
# Language-model projections only: the vision and audio towers stay untouched.
TARGET_MODULES = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
# The one recipe (spec 5.1); epochs are a ceiling, validation picks one of them.
# Batch 1 x 16: the loss holds every position x Gemma 4's 262,144-token vocabulary in fp32, and
# batch 4 ran a T4 out of memory there (Day 1, 2026-10-22). Same effective batch of 16.
RECIPE = {"rank": 16, "alpha": 32, "dropout": 0.05, "lr": 2e-4, "batch": 1, "grad_accum": 16,
          "scheduler": "cosine", "warmup_ratio": 0.03, "quant": "nf4, double quant, fp16 compute",
          "val_frac": 0.1, "max_new_tokens": 16,
          # Fixed, so --seed varies only the answer order and dropout (spec 2026-11-10 §5.1): every
          # seed validates on the same questions, and LoRA's A starts the same (B starts at 0).
          "split_seed": 0, "init_seed": 0}


def split_by_question(items: list[Item], frac: float = 0.1, seed: int = 0,
                      key=lambda i: question_id(i.id)) -> tuple[list[Item], list[Item]]:
    """(train, validation): every answer to a held-out question goes to validation."""
    questions = sorted({key(i) for i in items})
    random.Random(seed).shuffle(questions)
    held = set(questions[:max(1, round(frac * len(questions)))])
    return [i for i in items if key(i) not in held], [i for i in items if key(i) in held]


def to_messages(item: Item, labels: list[str], reply: str = "label", reason: str | None = None) -> list[dict]:
    return [{"role": "user", "content": build_prompt(item, labels)},
            {"role": "assistant", "content": reply_text(item.label, reason, reply)}]


def load_reasons(path: Path) -> tuple[dict[str, str], str]:
    """Answer id -> reason for the reasons kept by edge-grader-reasons, and the file's sha256."""
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode().splitlines()]
    return {r["id"]: r["reason"] for r in rows if r["drop"] is None}, hashlib.sha256(raw).hexdigest()


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
    ap.add_argument("--reply", choices=REPLY_FORMATS, default="label", help="what the adapter learns to reply")
    ap.add_argument("--reasons", type=Path, default=None, help="edge-grader-reasons JSONL (reason formats)")
    ap.add_argument("--train-sets", nargs="+", choices=list(data.DATASETS), default=["scientsbank"],
                    help="the grading sets whose train splits to learn from")
    ap.add_argument("--scheme", choices=["3way", "5way"], default="3way",
                    help="the labels the adapter learns to reply (5way: SciEntsBank, label replies)")
    args = ap.parse_args(argv)
    if (args.reply == "label") != (args.reasons is None):
        ap.error("--reasons goes with --reply reason-first / label-first, and only with them")
    default_set = args.train_sets == ["scientsbank"]
    if args.scheme == "5way" and (not default_set or args.reply != "label"):
        ap.error("--scheme 5way: SciEntsBank only, label replies only (Beetle and SAF have no 5-way labels)")

    labels = data.labels_for(args.scheme)
    # SAF's ids are random, so another set's questions are told apart by their text.
    key = (lambda i: question_id(i.id)) if default_set else (lambda i: i.question)
    items = (data.load(split="train", scheme=args.scheme) if default_set else
             [i for s in args.train_sets for i in data.load(split="train", scheme="3way", dataset=s)])
    train, val = split_by_question(items, RECIPE["val_frac"], RECIPE["split_seed"], key)
    tq, vq = sorted({key(i) for i in train}), sorted({key(i) for i in val})
    print(f"train {len(train)} answers / {len(tq)} questions, "
          f"validation {len(val)} answers / {len(vq)} questions")
    reasons, reasons_sha = load_reasons(args.reasons) if args.reasons else ({}, None)
    dropped = 0
    if reasons:
        kept = [i for i in train if i.id in reasons]
        dropped, train = len(train) - len(kept), kept
        print(f"training on {len(train)} answers: {dropped} dropped (no kept reason)")
    if args.dry_run:
        print(json.dumps(to_messages(train[0], labels, args.reply, reasons.get(train[0].id)), indent=2))
        return
    if args.val_n:
        val = val[:args.val_n]

    # Before torch loads: reserved-but-fragmented memory (3 GB on Day 1) becomes usable.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
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
    torch.manual_seed(RECIPE["init_seed"])  # the Trainer seeds itself only after this
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
        full = tok.apply_chat_template(to_messages(item, labels, args.reply, reasons.get(item.id)),
                                       tokenize=False, enable_thinking=False)
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

    # Read the way llama-server's runs will be (grader.grade): a reason-first reply cut at the cap
    # is unparsed; label-first is read from its first line.
    max_new = REASON_FIRST_MAX_TOKENS if args.reply == "reason-first" else RECIPE["max_new_tokens"]

    def validate(epoch: float) -> None:
        model.eval()
        right = right3 = unparsed = 0
        with torch.no_grad():
            for item in val:
                enc = tok(prompt_text(item), add_special_tokens=False, return_tensors="pt").to(0)
                out = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
                new = out[0][enc["input_ids"].shape[1]:]
                cut = args.reply == "reason-first" and len(new) >= max_new and new[-1] != tok.eos_token_id
                pred = None if cut else parse_reply(tok.decode(new, skip_special_tokens=True),
                                                    labels, args.reply)
                right += pred == item.label
                right3 += pred is not None and data.to_3way(pred) == data.to_3way(item.label)
                unparsed += pred is None
        model.train()
        loss = next((h["loss"] for h in reversed(trainer.state.log_history) if "loss" in h), None)
        k = len(history) + 1
        # val_accuracy is 3-way, the reported metric, so it picks the epoch under either scheme
        # (for 3-way it equals the exact-match accuracy, as before).
        history.append({"epoch": k, "at_epoch": round(epoch, 3), "train_loss": loss,
                        "val_accuracy": right3 / len(val), "val_unparsed": unparsed,
                        **({"val_accuracy_5way": right / len(val)} if args.scheme == "5way" else {})})
        model.save_pretrained(args.out / f"epoch-{k}")
        print(f"epoch {k} ({epoch:.2f}): loss {loss}, validation {right3}/{len(val)} (3-way), "
              f"{right}/{len(val)} ({args.scheme}), unparsed {unparsed}")

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
                                          for k, v in vars(args).items()},
                             "reasons_sha256": reasons_sha, "dropped_no_reason": dropped,
                             "max_new_tokens": max_new},
                history, chosen, tq, vq, sha, versions)
    args.results.mkdir(parents=True, exist_ok=True)
    tag = (("" if default_set else "-" + "-".join(args.train_sets)) + ("" if args.reply == "label" else f"-{args.reply}")
           + ("-5way" if args.scheme == "5way" else ""))
    path = args.results / f"finetune-{args.model}{tag}-{datetime.now(UTC):%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(s, indent=2) + "\n")
    print(f"chosen epoch {chosen}; adapter {args.out / 'adapter'} sha256 {sha}; summary {path}")
