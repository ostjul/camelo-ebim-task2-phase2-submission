"""Rewrite a dataset's 37-dim recorder state as the 16-dim model proprio.

    python -m camelo.train.convert_model_state \
        --source outputs/datasets/ext_fixpos_train_v1 \
        --output outputs/datasets/ext_fixpos_model16_v1 \
        [--gripper-lead-frames 8] [--gripper-polarity-flip]

Why this exists, twice over:

1. **Parity (F-63).** `LeRobotAdapter` never feeds the recorded 37-dim
   state to a foreign pretrained checkpoint — it feeds
   `model_state_from_state()` zero-padded to whatever width the
   checkpoint declares (``--state-layout model16``). A dataset whose
   ``observation.state`` is already that 16-dim proprio is therefore what
   those checkpoints are *actually* evaluated on. This is the
   parity-correct representation for them, not a workaround.
2. **It unblocks the pi/smolVLA family (F-65).** pi0 projects state
   through a fixed 32-wide linear layer and smolVLA a 6-wide one, so our
   37-dim state cannot enter either. 16 fits both.

The action column is left alone **unless a Track-A gripper transform asks
for it** (`--gripper-lead-frames`, `--gripper-polarity-flip`): 20-dim
canonical is within every candidate's `max_action_dim`, and it is what
``--action-layout canonical`` expects back. With no gripper flag the
column is not read, not rewritten and not re-statted, so a no-flag
conversion is byte-identical to one made before those flags existed.

**The transform is imported, never re-typed.** It is `pack_state(...,
"model16", MODEL_ACTION_DIM)` — the exact function the adapter calls at
inference — so train and eval cannot drift apart silently. Encoding the
same coercion twice is precisely how F-63 happened.

**The gripper transform is imported too**, from
`camelo.train.gripper_transforms`, and applied to the CANONICAL 37-dim
state and 20-dim action *before* the 37 -> 16 projection — the order that
module pins, and the order the eval adapter runs (``encode_state`` on the
37-dim observation, then pack). Applying it after the projection instead
would put train's flip in a different place from eval's: the F-63
double-encode in a new costume. The eval side owes an inverse for the
ENCODING half (`decode_actions`) and owes the RELABELING half nothing at
all; `gripper_transform` in `camelo_provenance.json` says which is which,
and the checkpoint carries the same fact in its sidecar.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from camelo import contracts as C
from camelo.train.gripper_transforms import GripperTransform

os.environ.setdefault("HF_HUB_OFFLINE", "1")

_LEROBOT_HINT = (
    "lerobot with dataset tooling support is required "
    "(pip install 'camelo-ebim[train]'); if an import path moved in "
    "your lerobot version, adjust camelo/train/convert_model_state.py."
)

STATE_KEY = "observation.state"
ACTION_KEY = "action"

#: What every caller that passes no gripper flag gets. Safe as a default
#: argument — the dataclass is frozen, so no caller can mutate the shared
#: instance into someone else's experiment.
IDENTITY = GripperTransform.identity()


def model_state_rows(states: np.ndarray, transform: GripperTransform = IDENTITY) -> np.ndarray:
    """(N, 37) recorded states -> (N, 16) model proprio, via the adapter's
    own packer so train and eval cannot diverge (F-63).

    The gripper transform is applied to the 37-dim state FIRST and the
    projection runs on the result, so the model16 gripper dims (7 and 15)
    inherit it. That is the order `gripper_transforms` pins and the order
    the eval adapter runs; doing it the other way round would encode the
    same fact in two places, which is the shape of F-63.
    """
    from camelo.policy.adapters.lerobot_generic import pack_state

    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != C.STATE_DIM:
        raise ValueError(f"expected (N, {C.STATE_DIM}) recorded states, got {states.shape}")
    states = transform.apply_states(states)  # BEFORE the projection, always
    return np.stack(
        [pack_state(row, "model16", C.MODEL_ACTION_DIM) for row in states],
        axis=0,
    ).astype(np.float32)


def column_edits(
    states: np.ndarray,
    actions: np.ndarray | None,
    features: dict,
    transform: GripperTransform = IDENTITY,
) -> dict[str, tuple[np.ndarray, dict]]:
    """Exactly the columns `convert()` rewrites, shaped for `modify_features`.

    ``observation.state`` is always in here — narrowing it is what this
    tool is for. ``action`` is in here **only when the transform actually
    edits it**: under the identity the key is absent, so the column is
    never removed, never re-added and never re-statted, and the output
    dataset is byte-identical to one produced before the gripper flags
    existed. That no-op is the whole reason this planning step is a
    separate function — it is the one property a caller can check
    without lerobot installed.
    """
    edits: dict[str, tuple[np.ndarray, dict]] = {
        STATE_KEY: (
            model_state_rows(states, transform),
            {"dtype": "float32", "shape": [C.MODEL_ACTION_DIM], "names": None},
        )
    }
    if transform.is_identity:
        return edits
    if actions is None:
        raise ValueError(
            f"gripper transform {transform.to_dict()} edits the action column, "
            "but no actions were read from the dataset"
        )
    edits[ACTION_KEY] = (
        transform.apply_actions(actions),
        {
            "dtype": "float32",
            "shape": [C.ACTION_DIM],
            # The transform edits VALUES, not layout: the recorded joint
            # names still describe the column and must survive the rewrite.
            "names": (features.get(ACTION_KEY) or {}).get("names"),
        },
    )
    return edits


def provenance_payload(source: Path, transform: GripperTransform = IDENTITY) -> dict:
    """What lands in `camelo_provenance.json`. A dataset must be able to
    say what was done to it — including, for the gripper transform, which
    half eval owes an inverse and which half it must leave alone."""
    return {
        "source": str(Path(source).resolve()),
        "transform": "observation.state: 37-dim recorded -> 16-dim model proprio "
        "[L_arm(7), L_grip, R_arm(7), R_grip]",
        "action_column": (
            "untouched (identity gripper transform)"
            if transform.is_identity
            else "rewritten by the gripper transform below"
        ),
        "gripper_transform": transform.provenance(),
        "note": "produced by camelo.train.convert_model_state using the adapter's "
        "pack_state(..., 'model16'). Evaluate checkpoints trained on this with "
        "--state-layout model16 --action-layout canonical (F-63).",
    }


def convert(source: Path, output: Path, *, transform: GripperTransform = IDENTITY) -> None:
    if not (source / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"not a LeRobot dataset: {source}")
    if output.exists():
        raise FileExistsError(
            f"output already exists: {output} — remove it or pick a new --output "
            "(conversions are not resumable)"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        from lerobot.datasets.dataset_tools import (
            modify_features,
            recompute_stats,
            remove_feature,
        )
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(_LEROBOT_HINT) from exc

    dataset = LeRobotDataset(repo_id=source.name, root=source)
    states = np.stack(dataset.hf_dataset[STATE_KEY]).astype(np.float32)
    # Read the action column only when something is going to edit it. Under
    # the identity it must come out of this conversion untouched, and the
    # cheapest way to guarantee that is to never look at it.
    actions = (
        None
        if transform.is_identity
        else np.stack(dataset.hf_dataset[ACTION_KEY]).astype(np.float32)
    )
    edits = column_edits(states, actions, dict(dataset.meta.features), transform)
    proprio = edits[STATE_KEY][0]
    print(f"{source.name}: {states.shape} -> {proprio.shape}")
    if ACTION_KEY in edits:
        print(f"{source.name}: action column rewritten by {transform.to_dict()}")

    # modify_features refuses to add a key that already exists, even when the
    # same call removes it, so a same-key replace has to be two passes.
    scratch = output.parent / f"{output.name}.tmp_nostate"
    if scratch.exists():
        import shutil

        shutil.rmtree(scratch)
    stripped = remove_feature(dataset, list(edits), output_dir=scratch, repo_id=source.name)
    modify_features(
        stripped,
        add_features=edits,
        output_dir=output,
        repo_id=output.name,
    )
    if scratch.exists():
        import shutil

        shutil.rmtree(scratch)

    # STATS ARE NOT CARRIED OVER (F-83). `modify_features` only copies
    # forward per-episode stats it already finds; for a feature it is
    # ADDING it computes none, and the remove pass took the old column's
    # stats with it. Nothing raises. The dataset then trains with
    # `observation.state` UNNORMALIZED, because lerobot returns any key it
    # has no stats for unchanged — survivable for a linear state projection
    # (pi0, smolVLA: arbitrary input scale, train matches eval) and fatal
    # for pi0.5, which discretizes state into 256 fixed bins over [-1, 1]
    # and pins raw radians to the saturation bins.
    #
    # A gripper transform puts the ACTION column in the same boat, and
    # worse: it also CHANGES that column's distribution (`1 - x` moves the
    # gripper dims' mean and quantiles), so carrying the old stats forward
    # would be wrong even if lerobot did it. Recompute + repair + verify,
    # in this order, after any dataset edit — never conditionally.
    recompute_stats(LeRobotDataset(repo_id=output.name, root=output))

    from camelo.train.dataset_stats import repair_degenerate_quantiles, verify

    repair_degenerate_quantiles(output)
    verify(output)

    provenance = provenance_payload(source, transform)
    (output / "camelo_provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"converted -> {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    # Track-A gripper edits (GRASP_EXPERIMENT_PROTOCOL.md §2). Both act on
    # the canonical arrays before the 37 -> 16 projection; both are recorded
    # in camelo_provenance.json; one of them eval must invert and the other
    # eval must not. Default: identity, i.e. today's dataset exactly.
    parser.add_argument(
        "--gripper-lead-frames",
        type=int,
        default=0,
        metavar="K",
        help="A1 (RELABELING): teach the gripper action k frames early, "
        "action[t] <- action[t+k]. Applied here and NEVER undone at eval — "
        "undoing it would cancel the experiment (default: 0)",
    )
    parser.add_argument(
        "--gripper-polarity-flip",
        action="store_true",
        help="A4 (ENCODING): x -> 1 - x on the gripper dims of state and action, "
        "so a pi-family fine-tune reuses its pretrained grasp-direction prior. "
        "Eval MUST invert it (gripper_transforms.decode_actions)",
    )
    args = parser.parse_args()
    transform = GripperTransform(
        lead_frames=args.gripper_lead_frames,
        polarity_flip=args.gripper_polarity_flip,
    )
    convert(args.source, args.output, transform=transform)
    return 0


if __name__ == "__main__":
    sys.exit(main())
