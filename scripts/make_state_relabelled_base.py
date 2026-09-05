#!/usr/bin/env python3
"""Re-declare a base checkpoint's `observation.state` width. Metadata only.

    python scripts/make_state_relabelled_base.py lerobot/smolvla_base 16 \
        --output outputs/checkpoints/smolvla_base_state16

Why this exists (F-76/F-77). `lerobot/smolvla_base` declares a 6-dim
`observation.state` because it was pretrained on SO-100. Its actual
projection is 32-wide (`model.state_proj.weight` is `[960, 32]`) and
`SmolVLAPolicy.prepare_state` PADS to `max_state_dim` — it never slices.
So training on our 16-dim proprio already consumes all 16 dims: a run
against the stock base and one against a relabelled base are bit-identical
in loss and gradient norm.

What the stale `[6]` does break is the *fine-tuned* checkpoint, which
inherits the declaration. Eval's `pack_state` then refuses it outright
(6 < 16, F-45), so the run is unevaluable despite being correctly trained.

This tool fixes the label and nothing else: weights, processors and stats
are SYMLINKED from the source, and only `config.json` is rewritten. It
does not touch the HF cache. Verify with `--check`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def resolve_source(checkpoint: str) -> Path:
    local = Path(checkpoint)
    if (local / "config.json").is_file():
        return local
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(checkpoint))


def build(checkpoint: str, width: int, output: Path) -> int:
    src = resolve_source(checkpoint)
    config = json.loads((src / "config.json").read_text())
    features = config.get("input_features") or {}
    state = features.get("observation.state")
    if state is None:
        print(f"error: {checkpoint} declares no observation.state", file=sys.stderr)
        return 2
    old = list(state.get("shape") or [])
    if old == [width]:
        print(f"{checkpoint} already declares {width}-dim state — nothing to do")
        return 0

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    for item in src.iterdir():
        if item.name == "config.json":
            continue
        (output / item.name).symlink_to(item.resolve())

    state["shape"] = [width]
    (output / "config.json").write_text(json.dumps(config, indent=2))
    print(f"{checkpoint}: observation.state {old} -> [{width}]")
    print(f"  weights symlinked, only config.json rewritten -> {output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="hub repo id or local checkpoint dir")
    parser.add_argument("width", type=int, help="state width to declare")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="report widths and exit")
    args = parser.parse_args()

    if args.check:
        src = resolve_source(args.checkpoint)
        feats = json.loads((src / "config.json").read_text()).get("input_features") or {}
        print(f"{args.checkpoint}: {(feats.get('observation.state') or {}).get('shape')}")
        return 0
    return build(args.checkpoint, args.width, args.output)


if __name__ == "__main__":
    sys.exit(main())
