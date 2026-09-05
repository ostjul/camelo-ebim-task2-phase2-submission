#!/usr/bin/env python3
"""Print the exact `make eval` line for a checkpoint trained on this cluster.

    python scripts/eval_recipe.py outputs/runs/act_20260809_162500
    python scripts/eval_recipe.py outputs/runs/*/checkpoints/020000

A checkpoint from `make train-slurm` carries three training-time
conventions that eval MUST restate, and every one of them fails silently
when omitted — no error, just a wrong number:

- ``--state-layout``  (F-63) 37-dim recorded state vs 16-dim model proprio.
  The widths can even agree while the vectors are unrelated.
- ``--action-layout`` (F-45) a 20-dim output is NOT self-evidently our
  contract; ours is, X-VLA's and RDT2's are not.
- the pi0.5 **state route** — `digits` (stock), `blind`, or `continuous`.
  Not a flag: it is re-applied automatically from the sidecar by
  `LeRobotAdapter`, because all three arms ship a byte-identical
  `policy_preprocessor.json` and nothing else could tell them apart. It is
  PRINTED here so a copied checkpoint that lost its sidecar is visible.
- ``--task``          (F-68) a language-conditioned rung learned the
  DATASET caption. Since the freeze, `DEFAULT_TASK` reads
  `contracts.TASK2_INSTRUCTION` = the corpus caption, so the derived flag
  usually coincides with the default — it stays explicit to catch
  checkpoints trained on any other string.

All three are recoverable from `train_config.json`, which lerobot writes
into every checkpoint, plus the dataset it names — so derive them rather
than remembering them. That is the whole point of this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camelo import contracts as C  # noqa: E402
from camelo.train.pi05_state_route import read_sidecar  # noqa: E402


def find_train_config(target: Path) -> Path:
    """Accept a run dir, a checkpoint dir, or the pretrained_model dir."""
    direct = target / "train_config.json"
    if direct.is_file():
        return direct
    matches = sorted(target.glob("**/train_config.json"))
    if not matches:
        raise FileNotFoundError(f"no train_config.json under {target}")
    # Latest checkpoint wins — paths sort by zero-padded step.
    return matches[-1]


def task_string(dataset_root: Path) -> str | None:
    tasks = dataset_root / "meta" / "tasks.parquet"
    if not tasks.is_file():
        return None
    try:
        import pandas as pd
    except ImportError:
        return None
    frame = pd.read_parquet(tasks).reset_index()
    column = "task" if "task" in frame.columns else frame.columns[0]
    values = [str(v) for v in frame[column].tolist()]
    if len(values) != 1:
        print(f"warning: dataset declares {len(values)} tasks: {values}", file=sys.stderr)
    return values[0] if values else None


def state_layout(dataset_root: Path) -> tuple[str, str]:
    info = dataset_root / "meta" / "info.json"
    if not info.is_file():
        return "model16", "dataset info.json missing — GUESSED, verify before trusting"
    shape = json.loads(info.read_text())["features"]["observation.state"]["shape"]
    dim = int(shape[0])
    if dim == C.STATE_DIM:
        return "canonical", f"dataset state is {dim}-dim, the recorded contract"
    if dim == C.MODEL_ACTION_DIM:
        return "model16", f"dataset state is {dim}-dim model proprio"
    return "model16", f"dataset state is {dim}-dim — matches NEITHER layout, investigate (F-63)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="run dir, checkpoint dir, or pretrained_model dir")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    config_path = find_train_config(args.run)
    config = json.loads(config_path.read_text())
    dataset_root = Path(config["dataset"]["root"])
    checkpoint = config_path.parent

    layout, why = state_layout(dataset_root)
    task = task_string(dataset_root)
    features = json.loads((dataset_root / "meta" / "info.json").read_text())["features"]
    action_dim = int(features["action"]["shape"][0])
    action_layout = "canonical" if action_dim == C.ACTION_DIM else "model16"

    route = read_sidecar(checkpoint)
    print(f"# checkpoint : {checkpoint}")
    print(f"# policy     : {config.get('policy', {}).get('type')}")
    print(f"# state route: {route}" + ("" if route != "digits" else "  (stock lerobot)"))
    if route != "digits":
        print("#   applied automatically at load from the sidecar -- no flag to pass.")
        print("#   Training loss is NOT comparable across routes; rank on the probe/IoU.")
    elif str(config.get("policy", {}).get("type", "")).startswith("pi05"):
        print("#   NOTE: 'digits' is also what an ABSENT sidecar reports. If this")
        print("#   checkpoint was copied out of its run dir, confirm the route --")
        print("#   the sidecar lives beside the run dir and does not travel with it.")
    print(f"# dataset    : {dataset_root}  ({why}; action {action_dim}-dim)")
    if config.get("rename_map"):
        print(f"# rename_map : {config['rename_map']}")
        print("#   cameras were remapped for TRAINING; eval assigns them POSITIONALLY")
        print("#   from obs.image_list() — head, wrist_left, wrist_right (F-64).")
    task_arg = (
        '--task "<UNKNOWN — dataset has no tasks.parquet; do NOT guess (F-68)>"'
        if task is None
        else f'--task "{task}"'
    )
    # Emit the script call, not `make eval`: make cannot forward unknown
    # options, so a `make eval … --state-layout …` line fails with
    # "unrecognized option" (reported from the DGX). The make form is given
    # second, via the EVAL_ARGS passthrough.
    print()
    print("python scripts/eval_batch.py --adapter lerobot --backend local \\")
    print(f"  --checkpoint {checkpoint} \\")
    print(f"  --episodes {args.episodes} \\")
    print(f"  --state-layout {layout} --action-layout {action_layout} \\")
    print(f"  {task_arg}")
    print()
    print("# or through make:")
    print(f"#   make eval ADAPTER=lerobot BACKEND=local CKPT={checkpoint} \\")
    print(f"#     N={args.episodes} TASK={task!r} \\")
    print(f'#     EVAL_ARGS="--state-layout {layout} --action-layout {action_layout}"')
    print()
    print("# Launch the eval scene with the spine pinned (F-50/F-58):")
    print("#   make sim-scene SPINE=0.50")
    return 0


if __name__ == "__main__":
    sys.exit(main())
