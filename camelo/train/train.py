"""Thin wrapper around ``lerobot-train``: YAML config -> draccus CLI flags.

    python -m camelo.train.train --config configs/act.yaml \
        --data outputs/datasets/task2_merged_v1 [--wandb] [extra draccus args...]

The YAML is flattened to dotted ``--a.b=c`` arguments, so anything
lerobot-train accepts can live in the config; unknown extra CLI args pass
straight through (last one wins in draccus).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from camelo.contracts import TASK2_INSTRUCTION

# Imported as a MODULE, not `from ... import write_mask_sidecar`: this file
# already has one `write_sidecar` (the state route's) and a second one for
# the gripper transform, and all three take different arguments and record
# different facts. Flat imports here would shadow each other silently and
# write the wrong sidecar. `obs_mask` is safe to import at module scope --
# it is numpy + stdlib with no import-time environment side effect, which is
# exactly what `gripper_transforms()` below exists to guarantee about the
# other module (see its docstring).
from camelo.train import obs_mask
from camelo.train.pi05_state_route import ROUTES, write_sidecar

ROUTE_DEFAULT = "digits"


def gripper_transforms():
    """Import the Track-A transform module without inheriting an offline default.

    `camelo.train.gripper_transforms` used to be a dataset-builder CLI that
    `os.environ.setdefault("HF_HUB_OFFLINE", "1")` at IMPORT time; this file
    hands its own environment to the trainer and decides offline mode
    deliberately (a `--policy.path` base must stay fetchable on a cold cache,
    F-25), so that default leaking in on import would have changed what a run
    does. The root cause is fixed — the builder CLI is gone and the module is
    numpy + stdlib — and
    `test_gripper_transforms_import_does_not_touch_offline_env` guards the
    regression. Retained as a belt: the import must leave the variable exactly
    as it found it.

    This is a FUNCTION, not a module-scope `from camelo.train import
    gripper_transforms`. Both cannot coexist: the name binds once, the def
    wins, and every `gripper_transforms.X` attribute access then raises
    AttributeError at runtime with nothing failing at import (P2 convergence
    decision, hazard 1). Reach the module through `gripper_transforms().X`.
    """
    before = os.environ.get("HF_HUB_OFFLINE")
    try:
        from camelo.train import gripper_transforms as module
    finally:
        if before is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = before
    return module


def dataset_captions(root: Path) -> set[str] | None:
    """Task caption strings of a v3 dataset, or None when unreadable
    (no pyarrow on this box, or no meta/tasks.parquet)."""
    tasks_file = root / "meta" / "tasks.parquet"
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None
    if not tasks_file.is_file():
        return None
    rows = pq.read_table(tasks_file).to_pylist()
    return {value for row in rows for value in row.values() if isinstance(value, str)}


def warn_on_caption_drift(root: Path) -> None:
    """F-68: a fine-tune learns whatever caption the dataset carries, and a
    checkpoint must be evaluated with its training caption. Warning only —
    a merged corpus may legitimately mix strings, but never silently."""
    captions = dataset_captions(root)
    if captions is None or captions == {TASK2_INSTRUCTION}:
        return
    print(
        "WARNING (F-68): dataset captions differ from the frozen Task 2 "
        f"instruction.\n  frozen : {TASK2_INSTRUCTION!r}\n  dataset: "
        + ", ".join(repr(c) for c in sorted(captions))
        + "\n  Evaluate the resulting checkpoint with --task set to its "
        "training caption, not the default.",
        file=sys.stderr,
    )


def dataset_state_width(root: Path) -> int | None:
    info = root / "meta" / "info.json"
    if not info.is_file():
        return None
    features = json.loads(info.read_text()).get("features", {})
    shape = (features.get("observation.state") or {}).get("shape")
    return int(shape[0]) if shape else None


def dataset_mask_layout(root: Path) -> str | None:
    """State layout of the A6 gripper-state mask this dataset was built with,
    or None when the dataset is not masked.

    The dataset builder stamps `mask_gripper_state` and `state_layout` into
    `camelo_provenance.json`. A dataset without the key -- every dataset
    built before A6 -- returns None and nothing changes.
    """
    provenance = Path(root) / "camelo_provenance.json"
    if not provenance.is_file():
        return None
    try:
        record = json.loads(provenance.read_text())
    except (OSError, ValueError):
        return None
    if not record.get("mask_gripper_state"):
        return None
    known = obs_mask.GRIP_STATE_DIMS
    layout = record.get("state_layout")
    if layout not in known:
        # Refusing beats guessing: the sidecar is what eval re-applies, and a
        # wrong layout there zeroes the wrong two dims with no error (F-63).
        raise ValueError(
            f"{provenance} says mask_gripper_state=true but carries "
            f"state_layout={layout!r}, which is not one of "
            f"{sorted(known)} — rebuild the dataset with a layout "
            "camelo.train.obs_mask knows rather than train on a mask "
            "eval cannot re-apply"
        )
    return layout


def base_state_width(checkpoint: str) -> int | None:
    """Declared `observation.state` width of a base checkpoint, from its
    config.json alone — no torch, no weights, works offline from the cache."""
    local = Path(checkpoint) / "config.json"
    if not local.is_file():
        try:
            from huggingface_hub import hf_hub_download

            local = Path(hf_hub_download(checkpoint, "config.json", local_files_only=True))
        except Exception:
            return None
    try:
        features = json.loads(local.read_text()).get("input_features") or {}
    except (OSError, ValueError):
        return None
    shape = (features.get("observation.state") or {}).get("shape")
    return int(shape[0]) if shape else None


def check_state_width(config: dict, data: Path) -> str | None:
    """Refuse a fine-tune whose base declares a NARROWER state slot than the
    dataset carries (F-76, corrected by F-77).

    What actually goes wrong is *labelling*, not truncation. Measured:
    smolVLA declares a 6-dim state (SO-100) but `prepare_state` PADS to
    `max_state_dim` and never slices, so all 16 of our proprio dims do
    reach the model — a run with the base's [6] and one with a corrected
    [16] are bit-identical in loss and gradient norm. The damage is that
    the fine-tuned checkpoint INHERITS the [6] declaration, and eval's
    `pack_state` then refuses it (6 < 16, F-45). The weights are fine; the
    checkpoint is unloadable.

    So this guard exists to stop us producing checkpoints that cannot be
    evaluated, not to stop silent data loss. A WIDER slot is fine —
    pi0/pi05 declare 32 and zero-pad our 16 exactly as the adapter does.
    """
    policy = config.get("policy", {})
    checkpoint = (
        policy.get("path") or policy.get("pretrained_path") or policy.get("base_model_path")
    )
    if not checkpoint:
        return None  # from-scratch: the policy adopts the dataset's width
    base = base_state_width(str(checkpoint))
    dataset = dataset_state_width(data)
    if base is None or dataset is None or base >= dataset:
        return None
    return (
        f"state width mismatch: base {checkpoint!r} declares a {base}-dim "
        f"observation.state but dataset {data.name} carries {dataset}-dim. "
        f"The fine-tuned checkpoint would INHERIT the {base}-dim declaration "
        f"and eval would refuse to load it (pack_state, F-45/F-77) — training "
        f"itself pads rather than truncates, so the weights would be fine and "
        f"the loss would look healthy the whole way. Point --policy.path at a "
        f"base whose state slot is >= {dataset} (see "
        f"scripts/make_state_relabelled_base.py), or narrow the dataset "
        f"(camelo.train.convert_model_state)."
    )


def flatten(prefix: str, value) -> list[str]:
    if isinstance(value, dict):
        # Dotted flags address nested CONFIG fields, so a mapping whose own
        # keys contain dots (rename_map: observation.images.head -> ...)
        # cannot be expressed that way — draccus has no way to tell where the
        # field path ends and the key begins. Such mappings go as one JSON
        # value, the form lerobot's own error message prescribes.
        if prefix and any("." in str(key) for key in value):
            return [f"--{prefix}={json.dumps(value)}"]
        args = []
        for key, sub in value.items():
            dotted = f"{prefix}.{key}" if prefix else str(key)
            args.extend(flatten(dotted, sub))
        return args
    if isinstance(value, bool):
        value = str(value).lower()
    return [f"--{prefix}={value}"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="LeRobot dataset root")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the command only")
    # lerobot-train is an accelerate program, so multi-GPU is a launcher
    # prefix rather than a flag. Empty by default: the single-GPU/DGX call
    # stays a direct exec, unchanged.
    parser.add_argument(
        "--launcher",
        default="",
        help='prefix the train command, e.g. "accelerate launch --num_processes=4"',
    )
    # pi0.5 only. Anything but `digits` routes training through
    # camelo.train.pi05_train, which patches the state route before calling
    # lerobot's own main(). The choice is written to a sidecar beside the run
    # because EVAL MUST RE-APPLY IT (F-63) -- see camelo/train/pi05_state_route.py.
    parser.add_argument(
        "--state-route",
        default=ROUTE_DEFAULT,
        choices=list(ROUTES),
        help="how pi0.5 receives proprioception (default: digits = stock lerobot)",
    )
    # The Track-A gripper transform the DATASET was built with
    # (camelo.train.convert_model_state --gripper-*). These flags transform
    # NOTHING here — the edit is already baked into the parquet on disk.
    # They record it beside the run, because the ENCODING half has to be
    # inverted at eval, the RELABELING half must not be, and a checkpoint's
    # weights look identical either way (F-63). Getting them wrong here
    # costs an eval, not a training run: pass what --data was converted with.
    parser.add_argument(
        "--gripper-lead-frames",
        type=int,
        default=0,
        metavar="K",
        help="A1 (RELABELING) the dataset was converted with; recorded, not applied",
    )
    parser.add_argument(
        "--gripper-polarity-flip",
        action="store_true",
        help="A4 (ENCODING) the dataset was converted with; recorded, not applied — "
        "eval inverts it (gripper_transforms.decode_actions)",
    )
    args, passthrough = parser.parse_known_args()
    # Built here, before anything expensive, so an impossible knob (a
    # negative lead) fails at argv rather than at sidecar-writing time,
    # after the guards have passed and the command line has been printed.
    gripper = gripper_transforms().GripperTransform(
        lead_frames=args.gripper_lead_frames,
        polarity_flip=args.gripper_polarity_flip,
    )

    # The scoring camera is never a policy input, and nothing downstream
    # objects: lerobot's feature check passes when the policy's cameras are
    # a subset of the dataset's, so a from-scratch policy just absorbs it as
    # an extra view and trains happily for hours (F-75). Refuse here — this
    # one is fatal, unlike the caption drift below, which is legitimate for
    # a merged corpus.
    info = args.data / "meta" / "info.json"
    if info.is_file():
        features = json.loads(info.read_text()).get("features", {})
        leaked = sorted(k for k in features if "eval_camera" in k)
        if leaked:
            print(
                f"error: {args.data} still contains the SCORING camera {leaked} — "
                "training on it leaks the eval signal (F-75). Strip it first:\n"
                "  python -m camelo.train.convert_model_state  (also does 16-dim state)\n"
                "  or lerobot.datasets.dataset_tools.remove_feature",
                file=sys.stderr,
            )
            return 2

    warn_on_caption_drift(args.data)
    config = yaml.safe_load(args.config.read_text()) or {}
    # A second-resolution timestamp is NOT unique: five arms submitted
    # together start within the same second on different nodes and all resolve
    # to one output_dir, racing into a single checkpoints/ tree with no record
    # of which arm won. Worse for A6 — the mask sidecar is written beside the
    # run dir and `read_mask_sidecar` walks up, so one masked arm sharing a
    # directory with four unmasked ones makes `--expect-mask` pass on an
    # unmasked checkpoint: the F-63 family again, via the guard meant to
    # prevent it. The SLURM job id is the natural discriminator; off-cluster,
    # fall back to the pid. Neither renames any existing run.
    stamp = os.environ.get("SLURM_JOB_ID") or f"p{os.getpid()}"
    run_name = f"{args.config.stem}_{time.strftime('%Y%m%d_%H%M%S')}_{stamp}"
    config.setdefault("output_dir", f"outputs/runs/{run_name}")
    config.setdefault("job_name", run_name)
    config.setdefault("dataset", {})
    config["dataset"]["repo_id"] = args.data.name
    config["dataset"]["root"] = str(args.data)
    config.setdefault("wandb", {})["enable"] = bool(args.wandb)
    # lerobot defaults push_to_hub=True and then dies on the missing repo_id
    # (and our repo ids are local-only anyway) — DGX_FINDINGS.md F-20.
    config.setdefault("policy", {}).setdefault("push_to_hub", False)

    # Refuse BEFORE the first optimizer step: a warning in a 20k-step run
    # scrolls away in seconds, and the cost of missing this is H100 hours
    # spent on a checkpoint that cannot be evaluated at all.
    problem = check_state_width(config, args.data)
    if problem:
        print(f"error: {problem}", file=sys.stderr)
        return 2

    # A6: read the dataset's own provenance rather than adding a flag, so the
    # mask cannot be declared on one side only. Resolved BEFORE the trainer
    # starts, because an unusable declaration is worth failing on at step 0.
    try:
        mask_layout = dataset_mask_layout(args.data)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if mask_layout:
        print(
            f"A6: {args.data.name} carries the gripper-state mask "
            f"(layout {mask_layout}); writing the sidecar so EVAL re-applies it"
        )

    # A launcher has to exec the console script's real path — `accelerate
    # launch lerobot-train` cannot resolve a bare entry-point name.
    # Resolve beside the running interpreter first. Batch jobs invoke us as
    # `.venv/bin/python -m camelo.train.train` without activating the venv,
    # so its bin/ is not on PATH; an activated shell resolves to the same
    # file either way.
    if args.state_route != ROUTE_DEFAULT:
        # Our own entrypoint, which patches the route and then calls lerobot's
        # main(). Same argv, same trainer, one thing different.
        train_prog = [sys.executable, "-m", "camelo.train.pi05_train"]
    else:
        sibling = Path(sys.executable).parent / "lerobot-train"
        resolved = str(sibling) if sibling.exists() else shutil.which("lerobot-train")
        if resolved is None:
            print("error: lerobot-train not found — is the training env installed?",
                  file=sys.stderr)
            return 2
        train_prog = [resolved]
    train_cmd = [*args.launcher.split(), *train_prog] if args.launcher else train_prog

    argv = [*train_cmd, *flatten("", config), *passthrough]
    print(" \\\n  ".join(argv))
    if args.dry_run:
        return 0

    # The sidecar is what makes the run evaluable: eval reads it back and
    # re-applies the same route. Written BEFORE the trainer starts so it
    # exists even if the job dies in the first step.
    write_sidecar(Path(config["output_dir"]), args.state_route)
    # The same contract for a different fact. Written UNCONDITIONALLY, the
    # identity included: an absent sidecar *means* the identity, so a run
    # that writes none is indistinguishable from one whose sidecar was lost
    # in an rsync — and the lost one evaluates upside down without a word.
    # `sidecar_is_explicit` can only tell the two apart if every run leaves
    # one behind.
    gripper_transforms().write_sidecar(Path(config["output_dir"]), gripper)
    # And a THIRD sidecar for the A6 mask, which is deliberately not a knob
    # on the gripper transform: a mask is applied identically on both sides,
    # so it is neither an ENCODING (inverted at eval) nor a RELABELING (train
    # only) and has its own module. Same contract, same moment, same reason
    # (F-63): a checkpoint trained on masked observations is only evaluable
    # if eval knows. Written only when the DATASET declares it, so "absent
    # means unmasked" keeps every pre-A6 checkpoint evaluating unchanged.
    if mask_layout:
        obs_mask.write_mask_sidecar(Path(config["output_dir"]), mask_layout)

    env = dict(os.environ)
    env["CAMELO_STATE_ROUTE"] = args.state_route
    # A --policy.path base model must be fetchable on a cold cache; only
    # force offline for from-scratch runs (DGX_FINDINGS.md F-25). Datasets
    # are local either way.
    if "path" not in config.get("policy", {}):
        env.setdefault("HF_HUB_OFFLINE", "1")
    return subprocess.call(argv, env=env)


if __name__ == "__main__":
    sys.exit(main())
