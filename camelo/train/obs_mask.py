"""A6: the gripper-state OBSERVATION mask, defined once for both sides.

    from camelo.train.obs_mask import mask_gripper_state, read_mask_sidecar

    states = mask_gripper_state(states)          # dataset build, whole corpus
    packed = mask_gripper_state(packed, layout)  # eval adapter, per observation

**Zero the gripper STATE dims only.** Not F-99's `blind`, which removed ALL
state and produced UNTIMED — this is the targeted form of research-doc fix 3,
which has never been run. A0b measured a deadlock gated on exactly this dim
(pi0.5 commands nothing executable until the measured state leaves 1.000, and
in a rollout the state only moves if a close was already commanded), while
full-corpus ACT times its close with the same dim pinned at 1.000.

**The mask must also be applied at inference**, or the policy meets a state
vector it never trained on and the number is wrong with no error attached
(F-63). It travels in `camelo_obs_mask.json` beside the run, read back by
`read_mask_sidecar`, exactly as the pi0.5 state route travels.

## Why this is its OWN module, next to `gripper_transforms.py`

The P2 convergence decision (GRASP_EXPERIMENT_PROTOCOL.md §1) split the two
deliberately, and the split is the taxonomy, not filing convenience.
`gripper_transforms.GripperTransform` sorts every knob into one of two
`Kind`s and refuses an unclassified one at construction:

- **ENCODING** — applied at train and *exactly inverted* at eval (A4's
  polarity flip).
- **RELABELING** — applied at train only, never inverted (A1's lead).

A6 is neither. The mask is applied **identically on both sides**: the dataset
column is zeroed and the live observation is zeroed, and nothing is ever
undone. Forcing it into `GripperTransform` would mean inventing a third
`Kind` and blurring the one distinction that module exists to keep sharp —
so it lives here instead, on the same reasoning that module already gives for
leaving A3 (a sampler) and A5 (frame selection) to their own experiments.

The two sidecars are therefore independent and both are written by
`camelo/train/train.py`: `camelo_gripper_transform.json` records the
transform (absent means the identity) and `camelo_obs_mask.json` records the
mask (absent means unmasked). Neither can stand in for the other.

## The corpus builder lives here (P5)

The P2 convergence (86c1452) deleted `gripper_transforms.build`/`main`
along with the module split, and with it the only writer of the
`mask_gripper_state` / `state_layout` provenance keys `train.dataset_mask_layout`
reads — existing masked corpora already carry the keys, so only *building a
new one* was blocked. `build_gripmask_corpus` restores that ability, next to
`mask_gripper_state` rather than back in `gripper_transforms.py`, on the same
taxonomy reasoning the split itself rests on. It runs the same F-83 triple
every dataset-rewrite tool in `camelo/train/` runs after touching a column —
`recompute_stats`, then `dataset_stats.repair_degenerate_quantiles`, then
`dataset_stats.verify` — because `modify_features` computes no stats for a
column it is *adding* and the remove pass took the old column's stats with
it; skipping it trains pi0.5 on unnormalized raw radians pinned to its
saturation bins. A1/A4 stacking (a lead- or polarity-flipped corpus that is
*also* gripmasked) is not a flag here: build `convert_model_state` first,
then point `--source` at its output — sequential builds, with the earlier
build's `camelo_provenance.json` nested under this one's `source_provenance`
key, since every builder in this package overwrites the file wholesale and
nesting is the only way lineage survives a second build.

Numpy + stdlib + ``camelo.contracts`` only, on purpose: ``camelo/train/``
must stay importable with no torch, no lerobot and no ROS, and — because the
eval adapter imports this module to re-apply the mask — with no import-time
environment side effects either. A module-scope
``os.environ.setdefault("HF_HUB_OFFLINE", "1")`` here would force an unmasked
GR00T zero-shot offline on a cold cache just by wiring A6 in (F-25);
``tests/test_obs_mask.py`` guards that this module never touches it. The
corpus builder below needs lerobot, but only *inside its own functions* —
imported at call time, the same lazy idiom `convert_model_state.py` and
`trim_ramp.py` use — so this module-scope promise is unaffected by adding it.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from camelo import contracts as C

MASK_SIDECAR = "camelo_obs_mask.json"
STATE_KEY = "observation.state"

# Gripper slots per state layout. Canonical is the 37-dim recorded contract;
# model16 is the packed proprio the pi/smolVLA family actually receives.
GRIP_STATE_DIMS = {
    "canonical": (C.S_LEFT_GRIP, C.S_RIGHT_GRIP),   # 29, 30
    "model16": (C.M_LEFT_GRIP, C.M_RIGHT_GRIP),     # 7, 15
}


def state_layout_of(width: int) -> str:
    if width == C.STATE_DIM:
        return "canonical"
    if width == C.MODEL_ACTION_DIM:
        return "model16"
    raise ValueError(
        f"state width {width} is neither canonical ({C.STATE_DIM}) nor "
        f"model16 ({C.MODEL_ACTION_DIM}); refusing to guess which dims are grippers"
    )


def mask_gripper_state(state: np.ndarray, layout: str | None = None) -> np.ndarray:
    """A6. Zero the gripper state dims, leave arm proprioception intact.

    Accepts a single (D,) vector or an (N, D) batch. This is the ONE
    definition — the dataset builder and the eval adapter both call it.
    """
    arr = np.asarray(state, dtype=np.float32).copy()
    layout = layout or state_layout_of(arr.shape[-1])
    if layout not in GRIP_STATE_DIMS:
        raise ValueError(f"unknown state layout {layout!r}")
    for d in GRIP_STATE_DIMS[layout]:
        arr[..., d] = 0.0
    return arr


def mask_sidecar_path(run_dir: Path) -> Path:
    """A SIBLING file, for the same reason `pi05_state_route.sidecar_path`
    is one: lerobot refuses to start when its output dir already exists, so
    creating the run dir early to hold a marker fails every run before step 0.
    """
    run_dir = Path(run_dir)
    return run_dir.parent / f"{run_dir.name}.{MASK_SIDECAR}"


def write_mask_sidecar(run_dir: Path, layout: str) -> Path:
    path = mask_sidecar_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mask": "gripper_state", "layout": layout}, indent=1) + "\n")
    return path


def read_mask_sidecar(checkpoint: str | Path) -> bool:
    """Walk up from a checkpoint dir looking for the mask sidecar.

    Absent means "no mask", so every checkpoint trained before this module
    existed evaluates exactly as it did before — the same contract
    `pi05_state_route.read_sidecar` keeps for the state route.
    """
    ckpt = Path(checkpoint)
    for parent in [ckpt, *ckpt.parents]:
        for cand in (parent / MASK_SIDECAR, mask_sidecar_path(parent)):
            if cand.is_file():
                return json.loads(cand.read_text()).get("mask") == "gripper_state"
    return False


# --- P5: the corpus builder ------------------------------------------------


def provenance_payload(source: Path, layout: str, source_provenance: dict | None) -> dict:
    """What lands in the output's `camelo_provenance.json`.

    Refuses an unknown layout HERE, at build time (raises `ValueError`) —
    the alternative is `train.dataset_mask_layout` refusing it later, at
    train time, which is the wrong moment to learn a corpus is unusable.

    `source_provenance` carries the source dataset's OWN provenance file
    forward, nested, when it had one — every builder in this package
    overwrites `camelo_provenance.json` wholesale, so nesting is the only
    way an A1/A4-then-A6 stack keeps its lineage.
    """
    if layout not in GRIP_STATE_DIMS:
        raise ValueError(f"unknown state layout {layout!r}")
    return {
        "source": str(Path(source).resolve()),
        "transform": "camelo.train.obs_mask: zero gripper state dims (A6)",
        "mask_gripper_state": True,
        "state_layout": layout,
        "masked_dims": list(GRIP_STATE_DIMS[layout]),
        "source_provenance": source_provenance,
        "note": "the mask MUST also be applied at inference, or the policy meets a "
        "state vector it never trained on and the number is wrong with no error "
        f"attached (F-63). It travels in {MASK_SIDECAR}, written beside the run by "
        "camelo.train.train and read back by read_mask_sidecar.",
        # A literal, not __name__: run as `python -m camelo.train.obs_mask`
        # this module IS __main__, and a provenance record saying so names
        # nothing a reader could import.
        "module": "camelo.train.obs_mask",
    }


def state_feature_spec(source_features: dict, width: int) -> dict:
    """The `modify_features` replacement spec for `observation.state`.

    Carries the source's per-dim `names` through. Hardcoding `names: None`
    here was the bug the deleted H100 test guarded — the mask changes the
    state's VALUES, not its layout, so the recorded names still describe
    the (now partly zeroed) column.
    """
    return {
        "dtype": "float32",
        "shape": [width],
        "names": (source_features.get(STATE_KEY) or {}).get("names"),
    }


def _load_states(source: Path) -> np.ndarray:
    """Lerobot-dependent: load `observation.state` as an (N, D) array.

    Lazy-imported so importing this module — which the eval adapter does,
    to reach `mask_gripper_state` — never requires lerobot. Kept separate
    from `_rewrite_state_column` so a test can fake a states array without
    a dataset on disk.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=source.name, root=source)
    return np.stack(dataset.hf_dataset[STATE_KEY]).astype(np.float32)


def _rewrite_state_column(
    dataset: Path, values: np.ndarray, spec: dict, output: Path, repo_id: str
) -> None:
    """Lerobot-dependent: replace `observation.state` and materialize `output`.

    `dataset` is the SOURCE root (a fresh `LeRobotDataset` is opened here,
    not passed in, so this stays the one function that needs lerobot for
    the rewrite). The remove-then-add idiom is `modify_features`'s own
    requirement: it refuses to add a key that already exists, even when the
    same call removes it, so a same-key replace is two passes through two
    scratch dirs beside `output`, then a `copytree` into `output` itself and
    a cleanup — the same shape `convert_model_state.convert` and the
    deleted `gripper_transforms.build` both use.
    """
    from lerobot.datasets.dataset_tools import modify_features, remove_feature
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source_dataset = LeRobotDataset(repo_id=dataset.name, root=dataset)
    scratch = output.parent / f"{output.name}.tmp_nostate"
    step_out = output.parent / f"{output.name}.tmp_state"
    for d in (scratch, step_out):
        if d.exists():
            shutil.rmtree(d)
    stripped = remove_feature(source_dataset, STATE_KEY, output_dir=scratch, repo_id=dataset.name)
    modify_features(
        stripped,
        add_features={STATE_KEY: (values, spec)},
        output_dir=step_out,
        repo_id=repo_id,
    )
    shutil.copytree(step_out, output)
    for d in (scratch, step_out):
        if d.exists():
            shutil.rmtree(d)


def _recompute_stats(output: Path) -> None:
    """Lerobot-dependent: the F-83 triple, mandatory after every rewrite.

    `modify_features` computes no stats for a column it is ADDING, and the
    remove pass took the old column's stats with it — nothing raises, and
    pi0.5 then bins raw radians into its saturation bins. Recompute, repair,
    verify, in this order, unconditionally.
    """
    from lerobot.datasets.dataset_tools import recompute_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    recompute_stats(LeRobotDataset(repo_id=output.name, root=output))

    from camelo.train.dataset_stats import repair_degenerate_quantiles, verify

    repair_degenerate_quantiles(output)
    verify(output)


def build_gripmask_corpus(source: Path, output: Path) -> Path:
    """A6: rewrite `source`'s `observation.state` with the gripper dims
    zeroed, and write it to `output` as a new LeRobot dataset.

    Refuses a non-dataset source and an existing output BEFORE touching
    lerobot at all, so those mistakes fail fast, offline, without a
    `pip install 'camelo-ebim[train]'` first. Infers the state layout from
    the source's own width (`state_layout_of`) — both canonical (37-dim)
    and model16 (16-dim) sources are accepted, same as the deleted builder.
    """
    source = Path(source)
    output = Path(output)
    if not (source / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"not a LeRobot dataset: {source}")
    if output.exists():
        raise FileExistsError(
            f"output already exists: {output} — remove it or pick a new --output"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    source_info = json.loads((source / "meta" / "info.json").read_text())
    source_features = source_info.get("features", {})

    states = _load_states(source)
    layout = state_layout_of(states.shape[-1])
    masked = mask_gripper_state(states, layout)
    spec = state_feature_spec(source_features, states.shape[-1])

    _rewrite_state_column(source, masked, spec, output, output.name)
    _recompute_stats(output)

    source_provenance_path = source / "camelo_provenance.json"
    source_provenance = (
        json.loads(source_provenance_path.read_text())
        if source_provenance_path.is_file()
        else None
    )
    provenance = provenance_payload(source, layout, source_provenance)
    (output / "camelo_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    dims = GRIP_STATE_DIMS[layout]
    print(f"mask: zeroed state dims {dims} ({layout}) on {states.shape[0]} frames")
    print(f"wrote {output}")
    return output


def main(argv: list[str] | None = None) -> int:
    """Build an A6 gripmask corpus.

        python -m camelo.train.obs_mask \\
            --source outputs/datasets/ext_fixpos200_model16_v1 \\
            --output outputs/datasets/ext_fixpos200_gripmask_v2
    """
    ap = argparse.ArgumentParser(description=main.__doc__.strip().splitlines()[0])
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    # Offline default belongs to the CLI, not to import scope: the eval
    # adapter imports this module for `mask_gripper_state`, and a module-level
    # setdefault would silently force an unmasked GR00T zero-shot offline on a
    # cold cache just by wiring A6 in (F-25). Set it where the build actually
    # needs it, exactly where the deleted `gripper_transforms.main` did.
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    build_gripmask_corpus(args.source, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
