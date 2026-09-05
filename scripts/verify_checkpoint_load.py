#!/usr/bin/env python3
"""Prove a checkpoint's weights actually load — don't assume it (F-81).

    python scripts/verify_checkpoint_load.py <checkpoint_dir>

`from_pretrained` does not fail when it cannot find weights. lerobot's
`PreTrainedPolicy.from_pretrained` resolves only `model.safetensors`, and
`PI0Policy.from_pretrained` wraps that in a bare `except Exception` which
prints

    Returning model without loading pretrained weights

and hands back a **randomly initialised** policy. It runs. It emits
plausible action chunks. It scores garbage. A LoRA run trips this
directly: it saves a ~5 MB `adapter_model.safetensors` and no
`model.safetensors`, and nothing on lerobot's READ path consumes an
adapter — peft is wired into training (`wrap_with_peft`, saving) only.

The tell is determinism: load twice and diff. Real weights give
max|diff| == 0; a random init does not. That is the whole check, and it is
the one thing a directory listing, a file size and a green training log
all fail to tell you.

Exit 0 = deterministic (weights loaded). Exit 1 = random. Exit 2 = error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def verify(checkpoint: Path, device: str) -> int:
    if not (checkpoint / "config.json").is_file():
        print(f"error: {checkpoint} has no config.json", file=sys.stderr)
        return 2

    weights = [
        name
        for name in ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin")
        if (checkpoint / name).is_file()
    ]
    adapter = (checkpoint / "adapter_config.json").is_file()
    print(f"weight files : {weights or 'NONE'}")
    print(f"adapter      : {adapter}")
    if not weights and adapter:
        print(
            "\nFAIL: adapter-only checkpoint. lerobot's read path cannot apply it, so "
            "from_pretrained returns RANDOM weights (F-81).\n"
            "  Merge first: python scripts/merge_lora_checkpoint.py <ckpt> --output <ckpt>/merged"
        )
        return 1
    if not weights:
        print("\nFAIL: no weight file at all — from_pretrained would return random weights (F-81)")
        return 1

    import torch  # noqa: F401  (import cost is the point of doing this in a job)
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class

    config = PreTrainedConfig.from_pretrained(str(checkpoint))
    config.pretrained_path = str(checkpoint)
    config.device = device
    cls = get_policy_class(config.type)

    first = cls.from_pretrained(str(checkpoint), config=config)
    second = cls.from_pretrained(str(checkpoint), config=config)
    worst = 0.0
    worst_name = ""
    for (name, a), (_, b) in zip(
        first.named_parameters(), second.named_parameters(), strict=True
    ):
        if a.shape != b.shape:
            continue
        diff = (a - b).abs().max().item()
        if diff > worst:
            worst, worst_name = diff, name

    where = f"  ({worst_name})" if worst else ""
    print(f"\nmax|diff| over two independent loads: {worst:.8f}{where}")
    if worst == 0.0:
        print("PASS: deterministic — the weights really loaded")
        return 0
    print("FAIL: two loads differ — the policy is randomly initialised (F-81)")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    return verify(args.checkpoint, args.device)


if __name__ == "__main__":
    sys.exit(main())
