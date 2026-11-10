"""Adapter averaging, a "model soup" (Wortsman et al., ICML 2022), for LoRA adapters.

Design: docs/superpowers/specs/2026-11-10-adapter-soup-design.md §5.2. A LoRA update is
dW = (alpha/r) B A, so averaging the As and Bs separately is not the average update. Stacking the
As as rows and the Bs as columns gives B A = sum_i B_i A_i exactly; alpha stays, r becomes N r, and
the scale alpha/(N r) supplies the 1/N. llama.cpp (rank from the tensor shape, a3a1c47) and PEFT
(r from the config) both read it that way, so the soup serves as one adapter with --lora.

    uv run --group finetune python -m edge_grader.soup data/adapters/e4b-soup/adapter \\
        data/adapters/e4b-s{0,1,2,3,4}/adapter
"""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

WEIGHTS, CONFIG = "adapter_model.safetensors", "adapter_config.json"


def stack(adapters: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """One adapter whose update is the sum of theirs: A stacked along rows, B along columns."""
    keys = adapters[0].keys()
    if any(a.keys() != keys for a in adapters):
        raise ValueError("the adapters don't have the same tensors")
    out = {}
    for k in keys:
        if ".lora_A." in k:
            out[k] = np.concatenate([a[k] for a in adapters], axis=0)  # (r, in) -> (N r, in)
        elif ".lora_B." in k:
            out[k] = np.concatenate([a[k] for a in adapters], axis=1)  # (out, r) -> (out, N r)
        else:
            raise ValueError(f"not a LoRA A or B matrix: {k}")
    return out


def soup_config(configs: list[dict]) -> dict:
    """The shared config with N times the rank. alpha/r scaling only: rsLoRA, DoRA or per-layer
    ranks would break the 1/N."""
    first = configs[0]
    if any(c != first for c in configs):
        raise ValueError("the adapters were trained with different configs")
    if first.get("use_rslora") or first.get("use_dora") or first.get("rank_pattern") or first.get("alpha_pattern"):
        raise ValueError("only plain LoRA (alpha / r scaling, one rank) averages this way")
    return {**first, "r": first["r"] * len(configs)}


def main(argv: list[str] | None = None) -> None:
    from safetensors.numpy import load_file, save_file  # the finetune group's, as in the container

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path, help="the soup's PEFT adapter folder")
    ap.add_argument("adapters", type=Path, nargs="+", help="PEFT adapter folders to average")
    ap.add_argument("--results", type=Path, default=Path("results/finetune"), help="summary JSON")
    args = ap.parse_args(argv)

    configs = [json.loads((a / CONFIG).read_text()) for a in args.adapters]
    config = soup_config(configs)
    weights = stack([load_file(a / WEIGHTS) for a in args.adapters])
    args.out.mkdir(parents=True, exist_ok=True)
    save_file(weights, args.out / WEIGHTS)
    (args.out / CONFIG).write_text(json.dumps(config, indent=2) + "\n")

    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    args.results.mkdir(parents=True, exist_ok=True)
    path = args.results / f"{args.out.parent.name}-{datetime.now(UTC):%Y%m%d-%H%M%S}.json"  # e4b-soup-…
    path.write_text(json.dumps({
        "kind": "soup", "method": "uniform soup, LoRA stacked along the rank (exact mean of the updates)",
        "ingredients": [{"adapter": str(a), "adapter_sha256": sha(a / WEIGHTS)} for a in args.adapters],
        "rank": config["r"], "lora_alpha": config["lora_alpha"], "out": str(args.out),
        "adapter_sha256": sha(args.out / WEIGHTS),
        "created": datetime.now(UTC).isoformat(timespec="seconds")}, indent=2) + "\n")
    print(f"soup of {len(args.adapters)} adapters, rank {config['r']}: {args.out}; summary {path}")


if __name__ == "__main__":
    main()
