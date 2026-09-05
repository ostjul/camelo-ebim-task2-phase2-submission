#!/usr/bin/env python3
"""Merge a LoRA adapter checkpoint into standalone full weights.

    python scripts/merge_lora_checkpoint.py \
        outputs/runs/pi0_ft_.../checkpoints/020000/pretrained_model \
        --output outputs/runs/pi0_ft_.../checkpoints/020000/merged

A LoRA run saves ~5 MB of `adapter_model.safetensors` and no
`model.safetensors`. Whether a downstream consumer can load that depends
on it having a compatible `peft` installed alongside lerobot — and when it
does not, `from_pretrained` does not fail. It logs

    Could not load state dict from remote files: ... does not appear to
    have a file named model.safetensors.
    Returning model without loading pretrained weights

and hands back a **randomly initialised network** that runs, produces
plausible-looking chunks, and scores garbage (F-81, found on the DGX). The
tell is that loading twice gives different weights.

Merging removes the dependency entirely: the output is an ordinary
checkpoint with full `model.safetensors`, loadable by any lerobot without
peft. Processor files, `config.json` and `train_config.json` are carried
over so `scripts/eval_recipe.py` keeps working against the merged dir.

Needs enough memory to hold the base model (~16 GB for a 4B policy in
fp32) — run it as a short GPU job rather than on a login node.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

CARRY_OVER = ("train_config.json",)


def merge(checkpoint: Path, output: Path, device: str) -> int:
    adapter_config = checkpoint / "adapter_config.json"
    if not adapter_config.is_file():
        print(f"error: {checkpoint} has no adapter_config.json — not a LoRA run", file=sys.stderr)
        return 2
    if (checkpoint / "model.safetensors").is_file():
        print(f"{checkpoint} already has full weights — nothing to merge")
        return 0

    base_id = json.loads(adapter_config.read_text()).get("base_model_name_or_path")
    if not base_id:
        print("error: adapter_config.json has no base_model_name_or_path", file=sys.stderr)
        return 2
    print(f"base: {base_id}")

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class
    from peft import PeftModel

    # Build the BASE policy with the FINE-TUNED config: the adapter was
    # trained against our feature widths, not the base's.
    config = PreTrainedConfig.from_pretrained(str(checkpoint))
    config.pretrained_path = base_id
    config.device = device
    policy = get_policy_class(config.type).from_pretrained(base_id, config=config)

    merged = PeftModel.from_pretrained(policy, str(checkpoint)).merge_and_unload()
    output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output))

    # Processors and provenance travel with the weights, or eval_recipe and
    # the normalization pipeline break against the merged dir.
    for item in checkpoint.iterdir():
        if item.name.startswith("policy_") or item.name in CARRY_OVER:
            shutil.copy2(item, output / item.name)

    weights = output / "model.safetensors"
    size = weights.stat().st_size if weights.is_file() else 0
    print(f"merged -> {output}  (model.safetensors {size / 1e9:.2f} GB)")
    if size == 0:
        print("error: no model.safetensors written", file=sys.stderr)
        return 2
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    return merge(args.checkpoint, args.output, args.device)


if __name__ == "__main__":
    sys.exit(main())
