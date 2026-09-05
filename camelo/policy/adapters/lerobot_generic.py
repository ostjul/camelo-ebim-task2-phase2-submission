"""Adapter for any LeRobot policy checkpoint.

One code path serves all of: GR00T N1.7 (lerobot-native since 0.6, incl.
the DROID Franka+Robotiq post-train), X-VLA (soft-prompted
cross-embodiment), pi0, smolVLA, and our own ACT/diffusion/fine-tuned
checkpoints trained on the recorded 20-dim/37-dim contract.

Inference prefers lerobot's processor pipeline (make_pre_post_processors +
a dataset-frame-shaped input, the documented modern API); it falls back to
a manual batch builder for older versions or simple policies. For
pretrained zero-shot checkpoints whose features don't match this
embodiment, inputs are coerced (camera order per ai_data_contract, state
padded/trimmed) and outputs mapped best-effort — measure, don't assume.

**Three layouts, not two.** `model16` / `canonical` are the SIM contract
(37-dim state, 20-dim action, 16-dim model proprio). `s27a15` is the
Munich rig's own contract — 27 state / 15 action / 20 Hz / three cameras
at real-rig resolutions — and it is a different robot, not a projection
of the sim one. It differs in every way that has ever produced a silent
wrong number here, so each is handled explicitly rather than inherited:

  * the state is packed from NAMED rig groups (`adapters/s27a15.py`),
    never sliced out of a raw array whose field order nobody here has
    observed;
  * cameras are mapped from the checkpoint's own `rename_map`, never
    positionally — VLA-JEPA takes two of three (F-64);
  * the only image op is the checkpoint's OWN training Resize, replayed
    with the same torchvision transform on the same uint8 tensor;
  * chunks replay at 20 Hz, and the executed window is the checkpoint's
    declared `n_action_steps`;
  * the emitted chunk stays 15-dim (`action_space == "s27a15"`) and is
    decoded by `rig_command()`, because there is no 20-dim canonical
    form for it and pretending otherwise would feed the sim executor 15
    numbers in 20 slots.

Everything the parity check (`tools/rig_parity_check.py`) exists to
verify lives in that list.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np

from camelo import contracts as C
from camelo.policy.adapters import s27a15
from camelo.policy.base import Obs, PolicyAdapter
from camelo.train.pi05_state_route import apply_for_checkpoint

log = logging.getLogger(__name__)

# What a lerobot checkpoint directory holds when it carries full weights.
# A LoRA fine-tune has NONE of these — only adapter_model.safetensors.
WEIGHT_FILES = ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin")


# Default embodiment head per GR00T base-model repo (F-41). The head is the
# biggest lever on zero-shot quality; every tag ships in the repo's own
# statistics.json / processor_config.json. GrootConfig's "new_embodiment"
# default is a FRESH head for fine-tuning — it has no statistics in any
# NVIDIA repo, so zero-shot must name an existing one:
#   - DROID repo: its only head (single Franka + Robotiq, 2 views).
#   - 3B base:  the XDoF head — dual arm, two parallel grippers, three
#     cameras (top/left/right) — the closest shape to fr3duo_mobile.
DEFAULT_GROOT_EMBODIMENTS = {
    "nvidia/GR00T-N1.7-DROID": "oxe_droid_relative_eef_relative_joint",
    "nvidia/GR00T-N1.7-3B": "xdof_relative_eef_relative_joint",
}


def _apply_temporal_ensemble(config, coeff: float) -> None:
    """Route ACT through lerobot's ACTTemporalEnsembler (plan W4 / protocol B1).

    Mutates the already-constructed config object IN PLACE, before the
    policy is built. Two things force that ordering:
      - ACTConfig.__post_init__ raises unless n_action_steps == 1 whenever
        temporal_ensemble_coeff is set, but __post_init__ only runs at
        dataclass construction — re-instantiating the config would lose
        every value lerobot resolved from the checkpoint's config.json. A
        plain attribute assignment on the existing object does not re-run
        it, so both fields must be forced together, here, on that object.
      - ACTPolicy.__init__ builds the ensembler only when
        config.temporal_ensemble_coeff is not None at construction time
        (modeling_act.py) — setting it after `cls(config, ...)` has already
        run would silently do nothing.
    So this must be called on the config BEFORE cls.from_pretrained /
    ACTPolicy(config) — see the call site in _load_policy.
    """
    if config.type != "act":
        raise ValueError(
            f"temporal_ensemble_coeff is ACT-only (lerobot.ACTTemporalEnsembler "
            f"only exists on ACTPolicy) — checkpoint config.type={config.type!r} "
            "is not 'act'"
        )
    config.temporal_ensemble_coeff = coeff
    config.n_action_steps = 1


def _load_policy(
    checkpoint: str,
    device: str,
    embodiment_tag: str | None = None,
    temporal_ensemble_coeff: float | None = None,
):
    """Returns (policy, is_groot_base)."""
    try:
        from lerobot.policies.factory import get_policy_class
        from lerobot.policies.pretrained import PreTrainedConfig
    except ImportError as exc:
        raise RuntimeError(
            "lerobot is required for this adapter (pip install 'camelo-ebim[policy]'). "
            "If lerobot is installed, its policy API may have moved — adjust "
            "camelo/policy/adapters/lerobot_generic.py:_load_policy for your version."
        ) from exc
    try:
        config = PreTrainedConfig.from_pretrained(checkpoint)
    except Exception as lerobot_exc:
        if temporal_ensemble_coeff is not None:
            # The GR00T base-model fallback is never ACT — refuse here rather
            # than silently ignoring the flag (F-12 class).
            raise ValueError(
                "temporal_ensemble_coeff is ACT-only, but "
                f"{checkpoint!r} did not load as a lerobot checkpoint "
                f"({lerobot_exc}) — refusing to run the GR00T base-model "
                "fallback with it set"
            ) from lerobot_exc
        # Not a lerobot checkpoint (no 'type' in config.json). NVIDIA's GR00T
        # releases are base-model SOURCES, loadable only through
        # GrootConfig.base_model_path (F-41) — try that before giving up.
        return _load_groot_base(checkpoint, device, embodiment_tag, lerobot_exc), True
    cls = get_policy_class(config.type)
    if temporal_ensemble_coeff is not None:
        _apply_temporal_ensemble(config, temporal_ensemble_coeff)
    return _load_weights(cls, checkpoint, config).to(device).eval(), False


def _load_weights(cls, checkpoint: str, config):
    """from_pretrained, but weights are GUARANTEED to have been loaded.

    lerobot's from_pretrained only WARNS when it finds no model.safetensors
    ("Returning model without loading pretrained weights") and hands back a
    randomly initialised network. A LoRA fine-tune ships nothing but
    adapter_model.safetensors, so every pi adapter in the H100 bundle
    evaluated as noise — 100% of executor ticks delta-clamped, IoU 0.0, and
    no error anywhere to say the weights were never there (F-81). Same
    failure family as F-12: a silent fallback is worse than a crash.

    Note this is NOT a missing optional dependency: peft is installed and
    lerobot's _peft_available is True. lerobot applies adapters when
    TRAINING (wrap_with_peft/get_peft_model) and saves them, but no loader
    reads adapter_config.json back — so installing peft changes nothing.
    """
    path = Path(checkpoint).expanduser()
    adapter_config = path / "adapter_config.json"
    if adapter_config.is_file():
        return _load_peft(cls, path, adapter_config, config)
    if path.is_dir() and not any((path / name).is_file() for name in WEIGHT_FILES):
        raise RuntimeError(
            f"{checkpoint!r} holds none of {WEIGHT_FILES} and no adapter_config.json "
            "— lerobot would return a randomly initialised network with only a "
            "warning (F-81). Refusing to run an untrained model."
        )
    return cls.from_pretrained(checkpoint, config=config)


def _load_peft(cls, path: Path, adapter_config: Path, config):
    """Base weights + the LoRA delta on top.

    lerobot trains these via `--peft.method_type=LORA`, but the runtime here
    has no peft support of its own, so the two halves are joined by hand.
    The adapter is MERGED rather than kept wrapped: a PeftModel does not
    expose predict_action_chunk, and merging also drops the LoRA overhead
    from every forward pass.
    """
    base = json.loads(adapter_config.read_text()).get("base_model_name_or_path")
    if not base:
        raise RuntimeError(f"{adapter_config} names no base_model_name_or_path")
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError(
            f"{path} is a LoRA adapter over {base!r} and needs peft installed"
        ) from exc
    log.info("loading LoRA adapter %s over base %s", path, base)
    policy = cls.from_pretrained(base, config=config)
    merged = PeftModel.from_pretrained(policy, str(path)).merge_and_unload()
    log.info("merged LoRA adapter into %s", type(merged).__name__)
    return merged


def _load_groot_base(checkpoint: str, device: str, embodiment_tag: str | None, cause):
    from lerobot.policies.factory import get_policy_class

    try:
        from lerobot.policies.groot.configuration_groot import GrootConfig
    except ImportError as exc:
        raise RuntimeError(
            f"{checkpoint!r} is not a lerobot checkpoint ({cause}) and the GR00T "
            "base-model fallback needs lerobot[groot]"
        ) from exc

    tag = embodiment_tag or DEFAULT_GROOT_EMBODIMENTS.get(checkpoint, "new_embodiment")
    # Hub id → local dir: is_raw_groot_n1_7_checkpoint() only accepts directories.
    local_path = _resolve_groot_local_path(checkpoint)
    try:
        # kwargs only — a pre-built GrootConfig skips placeholder VISUAL features.
        cls = get_policy_class(GrootConfig(base_model_path=local_path).type)  # "groot"
        policy = cls.from_pretrained(local_path, embodiment_tag=tag)
    except Exception as groot_exc:
        raise RuntimeError(
            f"{checkpoint!r} loads neither as a lerobot checkpoint ({cause}) nor "
            f"as a GR00T base model with embodiment_tag={tag!r} ({groot_exc}). "
            "For GR00T zero-shot the tag must name a head that exists in the "
            "repo's statistics.json — override with --gr00t-embodiment."
        ) from groot_exc
    log.info(
        "loaded %s as a GR00T base model (local=%s), embodiment_tag=%r "
        "(the head is the main zero-shot lever; --gr00t-embodiment overrides)",
        checkpoint,
        local_path,
        tag,
    )
    # Stash the resolved path so the processor factory sees a directory.
    policy.config.base_model_path = local_path
    return policy.to(device).eval()


def _resolve_groot_local_path(checkpoint: str) -> str:
    """Hub id or local dir → absolute local snapshot directory."""
    path = Path(checkpoint).expanduser()
    if path.is_dir():
        return str(path.resolve())
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            f"need huggingface_hub to resolve GR00T hub id {checkpoint!r}"
        ) from exc
    return snapshot_download(repo_id=checkpoint)


# What a checkpoint's action vector MEANS — never inferred from its width
# (F-45: C.ACTION_DIM==20 collides with X-VLA's 20-dim absolute EEF and
# RDT2's 20-dim relative EEF). Only checkpoints we trained on the recorded
# canonical contract may claim "canonical", and they must claim it
# explicitly (--action-layout canonical). The droid8 layouts are the
# single-instance arm-splitter mode (F-58, ZERO_SHOT_RUNBOOK.md §1.2):
# 8-dim [joint7, gripper] on the RIGHT arm, left latched at settle —
# "_delta" for joint-delta checkpoints (pi05_droid_jointpos).
ACTION_LAYOUTS = (
    "model16",
    "canonical",
    # The Munich rig's own 15-dim vector (adapters/s27a15.py). NOT a sim
    # layout: it never becomes a 20-dim canonical chunk, so an adapter
    # declared s27a15 sets `action_space = "s27a15"` and the sim executor
    # refuses it rather than executing 15 numbers as if they were 20.
    s27a15.LAYOUT,
    "eef_abs",
    "eef_rel",
    "droid8_right",
    "droid8_right_delta",
)


# What a checkpoint EXPECTS in its state slot — the mirror of the above,
# and needed for the same reason (F-63). "model16" coerces our proprio into
# the 16-dim model contract and zero-pads: right for FOREIGN pretrained
# checkpoints, which is all that existed before M3. "canonical" hands over
# the recorded 37-dim state untouched, which is what a checkpoint we
# fine-tuned on our own dataset was actually trained on. Width cannot tell
# them apart — a 37-dim slot accepts either — so it must be declared.
# "s27a15" is the real-robot 27-dim vector, packed from NAMED rig groups
# rather than sliced out of a raw array (adapters/s27a15.py).
STATE_LAYOUTS = ("model16", "canonical", s27a15.LAYOUT)


def pack_state(state, layout: str, dim: int) -> np.ndarray:
    """Build the vector a checkpoint's `dim`-wide state slot expects, per
    its declared layout. Raises rather than guessing (F-63).

    `state` is the canonical 37-dim observation for the sim layouts and an
    `s27a15.RigObservation` (or an already-packed 27-dim vector) for the
    real-robot one.
    """
    if layout == s27a15.LAYOUT:
        if dim != s27a15.STATE_DIM:
            raise RuntimeError(
                f"state_layout='{s27a15.LAYOUT}' declared but the checkpoint's "
                f"state slot is {dim}-dim (the real corpus is {s27a15.STATE_DIM})"
            )
        if isinstance(state, s27a15.RigObservation):
            return s27a15.pack_state(state)
        vec = np.asarray(state, dtype=np.float32).ravel()
        if vec.size != s27a15.STATE_DIM:
            raise RuntimeError(
                f"state_layout='{s27a15.LAYOUT}' needs an s27a15.RigObservation or "
                f"an already-packed {s27a15.STATE_DIM}-dim vector, got {vec.size} dims"
            )
        return vec
    if layout == "canonical":
        if dim != C.STATE_DIM:
            raise RuntimeError(
                f"state_layout='canonical' declared but the checkpoint's state "
                f"slot is {dim}-dim (recorded contract is {C.STATE_DIM})"
            )
        if state.size != C.STATE_DIM:
            raise RuntimeError(
                f"state_layout='canonical' needs a {C.STATE_DIM}-dim observation, got {state.size}"
            )
        return state.astype(np.float32)
    if layout != "model16":
        raise RuntimeError(f"cannot pack state_layout={layout!r}")
    proprio = C.model_state_from_state(state)
    if dim < proprio.size:
        # A narrow slice of [L_arm7, L_grip, R_arm7, R_grip] silently keeps
        # only the left arm — refuse instead (F-45); per-arm runs need the
        # explicit arm-splitter mode.
        raise RuntimeError(
            f"state feature is {dim}-dim but model proprio is {proprio.size}-dim "
            "[L_arm(7), L_grip, R_arm(7), R_grip] — refusing to truncate silently "
            "(F-45); per-arm checkpoints need --action-layout "
            "droid8_right[_delta] (ZERO_SHOT_RUNBOOK.md)"
        )
    vec = np.zeros(dim, dtype=np.float32)
    vec[: proprio.size] = proprio
    return vec


def _obs_mask():
    """Import the A6 mask module, deferred and environment-neutral.

    Deferred for two reasons, either of which alone would be enough.

    LAYERING: `camelo/policy` -> `camelo/train` is an upward edge (AGENTS.md
    hard rule 3), so the reach happens inside a function, exactly as
    `LeRobotAdapter.__init__` does for the gripper-transform sidecar and as
    `convert_model_state.py:56` does reaching the other way for `pack_state`.

    ENVIRONMENT: the mask helpers used to live in `gripper_transforms`, which
    was also a dataset-builder CLI and `os.environ.setdefault(
    "HF_HUB_OFFLINE", "1")` at IMPORT time — which would have forced this
    process offline just by wiring A6 in, and eval may legitimately fetch
    from the hub (GR00T zero-shot on a cold cache, F-25). The root cause is
    fixed twice over: the builder CLI is gone, and `camelo.train.obs_mask` is
    numpy + stdlib with no environment side effect at all
    (`tests/test_obs_mask.py` guards it). This wrapper is retained as a belt:
    an import must leave the variable exactly as it found it. Nothing here
    pulls in lerobot or torch.
    """
    before = os.environ.get("HF_HUB_OFFLINE")
    try:
        from camelo.train import obs_mask
    finally:
        if before is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = before
    return obs_mask


def read_obs_mask(checkpoint: str | Path) -> bool:
    """True when this checkpoint's run was trained with the A6 mask.

    Absent sidecar means unmasked, exactly as an absent state-route sidecar
    means `digits`, so every checkpoint predating A6 evaluates unchanged.
    """
    return _obs_mask().read_mask_sidecar(checkpoint)


def obs_mask_locations(checkpoint: str | Path) -> list[Path]:
    """Every path `read_mask_sidecar` looks at, in the order it looks.

    Built from that module's OWN name and path helper, so an error message
    cannot describe a search different from the one performed.
    """
    module = _obs_mask()
    ckpt = Path(checkpoint)
    return [
        candidate
        for parent in [ckpt, *ckpt.parents]
        for candidate in (parent / module.MASK_SIDECAR, module.mask_sidecar_path(parent))
    ]


def verify_obs_mask(checkpoint: str | Path, resolved: bool, expected: bool | None) -> None:
    """Fail loudly when the caller's declared A6 expectation is not met.

    `absent sidecar == unmasked` is the right COMPATIBILITY default — it is
    what keeps every checkpoint predating A6 evaluating unchanged — but it is
    exactly the wrong behaviour for the run whose entire result IS the mask:
    a lost sidecar then yields a plausible number, not an error. So an A6
    invocation declares `expect_mask=True` and this refuses to run otherwise.
    `None` (the default) keeps the permissive behaviour.
    """
    if expected is None or bool(expected) == resolved:
        return
    looked = obs_mask_locations(checkpoint)
    shown = "\n  ".join(str(p) for p in looked[:6])
    rest = len(looked) - 6
    more = f"\n  ... and {rest} more, up to the filesystem root" if rest > 0 else ""
    want, got = ("MASKED", "UNMASKED") if expected else ("UNMASKED", "MASKED")
    raise RuntimeError(
        f"A6 mask mismatch for checkpoint {str(checkpoint)!r}: caller expects "
        f"{want}, the sidecar search resolved {got}. Looked in:\n  {shown}{more}\n"
        + (
            "Nothing declared the mask, so the policy would be fed a LIVE "
            "gripper state it never trained on and the number would look fine "
            "(F-63). Re-run training with a masked dataset, or restore the "
            f"run's {_obs_mask().MASK_SIDECAR} sidecar."
            if expected
            else "A mask sidecar was found, so this checkpoint was trained on "
            "zeroed gripper dims and cannot be evaluated as an unmasked "
            "control. Drop --expect-mask 0, or point at the unmasked run."
        )
    )


def pack_state_masked(
    state: np.ndarray, layout: str, dim: int, mask: bool = False
) -> np.ndarray:
    """`pack_state`, then the A6 gripper-state mask if the run carried one.

    ONE call site for the mask on the eval side, and it calls the same
    function the dataset builder called — that is the whole point (F-63).
    The layout is passed EXPLICITLY rather than inferred from the width:
    `pack_state` zero-pads model16 proprio out to the checkpoint's slot (32
    for pi0/pi0.5), and a 32-wide vector matches neither layout, so
    inference would raise on precisely the checkpoints A6 targets. The pad
    is appended at the tail, so dims 7/15 are still the gripper slots.
    """
    packed = pack_state(state, layout, dim)
    if not mask:
        return packed
    return _obs_mask().mask_gripper_state(packed, layout)


def stats_width(pipeline, key: str) -> int | None:
    """Width of `key`'s normalization stats in a processor pipeline, if any.

    A policy that pads internally declares a WIDER state feature than it was
    normalized at: pi0 declares `observation.state [32]` (its max_state_dim)
    while its stats come from a 16-dim dataset column, because lerobot
    normalizes the raw dataset row and only then lets `prepare_state` pad to
    32. Packing to the declared width instead makes the normalizer subtract a
    16-dim mean from a 32-dim vector.

    Until F-83 this could not be seen: those checkpoints shipped no state
    stats at all, so the normalizer passed the state through untouched and
    any width "worked" — unnormalized. Fixing the stats turned a silent
    wrong number into a loud shape error, which is the trade we want.
    """
    for step in getattr(pipeline, "steps", None) or []:
        stats = getattr(step, "_tensor_stats", None)
        if not stats or key not in stats:
            continue
        for tensor in stats[key].values():
            shape = getattr(tensor, "shape", ())
            if len(shape) == 1 and int(shape[0]) > 1:
                return int(shape[0])
    return None


def training_image_size(checkpoint: str) -> tuple[int, int] | None:
    """(H, W) a checkpoint's TRAINING pipeline resized every camera to.

    `config.input_features` records the *dataset's* native camera shapes, not
    what the model consumed: a `dataset.image_transforms` Resize sits between
    them and is applied by the dataloader, which does not exist at inference.
    GR00T makes the gap fatal rather than silent — its input packer does
    `np.stack(cams, axis=2)`, which raises on our 720x1280 head against
    480x848 wrists (F-66). A policy that happened to tolerate ragged shapes
    would instead see a geometry it never trained on, which is worse.

    So restate the transform from `train_config.json`, the same way layouts
    and the caption are restated rather than guessed. Returns None when the
    checkpoint declares no enabled Resize, leaving per-feature shapes in
    charge — which is right for every non-GR00T rung here.
    """
    path = Path(checkpoint).expanduser() / "train_config.json"
    if not path.is_file():
        return None
    try:
        transforms = json.loads(path.read_text())["dataset"]["image_transforms"]
        if not transforms.get("enable"):
            return None
        size = transforms["tfs"]["resize"]["kwargs"]["size"]
    except (KeyError, TypeError, ValueError):
        return None
    if isinstance(size, (list, tuple)) and len(size) == 2:
        return int(size[0]), int(size[1])
    if isinstance(size, int):
        return size, size
    return None


def training_image_transforms(checkpoint: str) -> dict | None:
    """The `dataset.image_transforms` blob a checkpoint TRAINED under.

    Same restatement pattern as `training_image_size`, but it hands back the
    whole blob rather than a size, because the s27a15 path has to reproduce
    the transform with the SAME torchvision class the dataloader used, not
    an approximation of it. `training_image_size` + PIL's `Image.resize`
    (bicubic, on the HWC array) is close enough for a sim rollout and NOT
    close enough for a parity check against the training pipeline: lerobot
    applies `torchvision.transforms.v2.Resize` (bilinear + antialias) to the
    **uint8 CHW** tensor, before the /255 cast (`dataset_reader.get_item` →
    `lerobot_train.py:603-606`). Different filter, different rounding point,
    different pixels.

    Returns None when the checkpoint declares no enabled transforms.
    """
    path = Path(checkpoint).expanduser() / "train_config.json"
    if not path.is_file():
        return None
    try:
        blob = json.loads(path.read_text())["dataset"]["image_transforms"]
    except (KeyError, TypeError, ValueError, OSError):
        return None
    if not isinstance(blob, dict) or not blob.get("enable"):
        return None
    return blob


def training_rename_map(checkpoint: str) -> dict[str, str]:
    """`train_config.json`'s `rename_map` (dataset key -> model key).

    VLA-JEPA's is `{head: exterior_1_left, wrist_right: exterior_2_left}` —
    two cameras out of three, and the one that decides whether the model's
    second view is the right wrist it trained on or the left wrist it never
    saw. Empty for every rung whose model keys are the dataset's own.
    """
    path = Path(checkpoint).expanduser() / "train_config.json"
    if not path.is_file():
        return {}
    try:
        return dict(json.loads(path.read_text()).get("rename_map") or {})
    except (TypeError, ValueError, OSError):
        return {}


def training_dataset_fps(checkpoint: str) -> float | None:
    """FPS of the dataset a checkpoint was trained on, from `train_config.json`'s
    `dataset.root` -> `<root>/meta/info.json`'s `fps` (W5: rate-matched
    checkpoints train at something other than the recorder's 30 fps, and the
    executor must replay their chunks at THAT rate, not silently fall
    through to the 30 fps default — see `chunk_dt` in `LeRobotAdapter.__init__`).

    Same restatement pattern as `training_image_size` above: the fact lives
    beside the checkpoint, not in it, so it is read rather than guessed.
    Returns None when anything along the chain is missing or unparseable
    (no train_config.json, no dataset.root, root's meta/info.json gone or
    unreadable) — callers fall back to the pre-existing default in that case.
    """
    path = Path(checkpoint).expanduser() / "train_config.json"
    if not path.is_file():
        return None
    try:
        root = json.loads(path.read_text())["dataset"]["root"]
    except (KeyError, TypeError, ValueError, OSError):
        return None
    if not root:
        return None
    info_path = Path(root).expanduser() / "meta" / "info.json"
    if not info_path.is_file():
        return None
    try:
        fps = json.loads(info_path.read_text())["fps"]
        return float(fps)
    except (KeyError, TypeError, ValueError, OSError):
        return None


def coerce_chunk(actions: np.ndarray, layout: str) -> np.ndarray:
    """Map a (T, A) chunk onto the canonical 20-dim contract per its
    declared layout. Raises rather than guessing (F-45)."""
    if layout == s27a15.LAYOUT:
        # No coercion exists: the rig's 15-dim vector is not a projection of
        # the sim's 20-dim contract, it is a different robot's action space.
        # The chunk leaves here in the layout the checkpoint emitted and is
        # decoded by `s27a15.unpack_action` on the rig side.
        if actions.shape[-1] != s27a15.ACTION_DIM:
            raise RuntimeError(
                f"action_layout='{s27a15.LAYOUT}' declared but the checkpoint emits "
                f"{actions.shape[-1]}-dim actions (the real corpus is "
                f"{s27a15.ACTION_DIM})"
            )
        return actions.astype(np.float32)
    if layout == "canonical":
        if actions.shape[-1] != C.ACTION_DIM:
            raise RuntimeError(
                f"action_layout='canonical' declared but the checkpoint emits "
                f"{actions.shape[-1]}-dim actions (contract is {C.ACTION_DIM})"
            )
        return actions.astype(np.float32)
    if layout != "model16":
        raise RuntimeError(f"cannot execute action_layout={layout!r}")
    model_actions = actions[:, : C.MODEL_ACTION_DIM]
    if model_actions.shape[-1] < C.MODEL_ACTION_DIM:
        pad = np.zeros(
            (model_actions.shape[0], C.MODEL_ACTION_DIM - model_actions.shape[-1]),
            dtype=np.float32,
        )
        model_actions = np.concatenate([model_actions, pad], axis=-1)
    return C.model_to_canonical_actions(model_actions)


class LeRobotAdapter(PolicyAdapter):
    # Class-level defaults for the s27a15 fields, so an adapter built
    # field-by-field (tests/test_obs_mask_parity.py builds one with
    # object.__new__ to exercise the state path in isolation) takes the sim
    # branch rather than raising AttributeError deep inside `_frame`.
    real_layout: bool = False
    camera_map: dict[str, str] | None = None
    train_resize = None
    gripper_command: str = s27a15.DEFAULT_GRIPPER_COMMAND

    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda",
        name: str = "lerobot",
        embodiment_tag: str | None = None,
        action_layout: str = "model16",
        camera_map: dict[str, str] | None = None,
        chunk_dt: float | None = None,
        state_layout: str = "model16",
        num_inference_steps: int | None = None,
        expect_mask: bool | None = None,
        temporal_ensemble_coeff: float | None = None,
        gripper_command: str = s27a15.DEFAULT_GRIPPER_COMMAND,
        allow_caption_drift: bool = False,
    ):
        import torch

        if action_layout not in ACTION_LAYOUTS:
            raise ValueError(f"action_layout must be one of {ACTION_LAYOUTS}")
        if state_layout not in STATE_LAYOUTS:
            raise ValueError(f"state_layout must be one of {STATE_LAYOUTS}")
        self.state_layout = state_layout

        # The real-robot layout is a MATCHED PAIR: the 27-dim state and the
        # 15-dim action are one robot's contract, and half of it is never
        # right. Declaring only one is the F-63 shape of mistake (widths
        # agree, meanings do not), so refuse instead of interpreting it.
        self.real_layout = s27a15.LAYOUT in (action_layout, state_layout)
        if self.real_layout and action_layout != state_layout:
            raise ValueError(
                f"the {s27a15.LAYOUT!r} layout is the Munich rig's whole contract "
                f"(27 state / 15 action); got action_layout={action_layout!r} "
                f"state_layout={state_layout!r} — pass {s27a15.LAYOUT!r} for both "
                "or neither"
            )
        if self.real_layout:
            self.action_space = s27a15.LAYOUT
        self.gripper_command = gripper_command
        # Validate the convention name NOW, at load, not on the first chunk:
        # a typo discovered mid-rollout is a stopped robot at best.
        s27a15.gripper_command(1.0, gripper_command)
        self.allow_caption_drift = allow_caption_drift
        if action_layout.startswith("eef"):
            # Fail at load, before the sim is touched: EEF output cannot be
            # executed on our joint-space contract until the differential-IK
            # work item lands (F-45; ZERO_SHOT_POLICIES.md).
            raise RuntimeError(
                f"checkpoint {checkpoint!r} is declared action_layout="
                f"{action_layout!r}: EEF-space actions cannot drive the "
                "16-dim joint contract without the differential-IK work item "
                "(see docs/runbooks/ZERO_SHOT_POLICIES.md) — refusing to run zero-shot"
            )
        self.action_layout = action_layout
        self.splitter_layout = action_layout.startswith("droid8")
        self.name = name
        self.device = device
        self.torch = torch

        # A6 (protocol §2): a run trained on gripper-state-masked observations
        # must be EVALUATED on them too. Nothing in the checkpoint records the
        # mask -- the weights and every processor config are byte-identical to
        # an unmasked run -- so without this the policy meets a live gripper
        # state of 1.000 where it only ever saw 0.0, with no error and no
        # shape mismatch. That is F-63 again, which is why the mask travels in
        # a sidecar beside the run dir exactly as the state route does.
        self.obs_mask_gripper = read_obs_mask(checkpoint)
        if self.real_layout and self.obs_mask_gripper:
            # `mask_gripper_state` is defined on the canonical/model16 gripper
            # slots; s27a15's is dim 14 of a different vector. Rather than
            # inventing a third masked convention here, refuse — no real-robot
            # rung was trained masked, so this can only be a stray sidecar.
            raise RuntimeError(
                f"checkpoint {checkpoint!r} carries the A6 gripper-state mask "
                f"sidecar but is declared {s27a15.LAYOUT!r}, whose gripper slot is "
                "not one of the masked layouts (canonical/model16) — refusing to "
                "evaluate a masked checkpoint on an unmasked state (F-63)"
            )
        # An A6 run declares what it expects and dies on disagreement: the
        # mask IS the independent variable there, so "absent means unmasked"
        # -- correct for every checkpoint predating A6 -- must not be allowed
        # to quietly decide the experiment's answer.
        verify_obs_mask(checkpoint, self.obs_mask_gripper, expect_mask)
        # Logged ALWAYS, masked or not, so every probe log records what the
        # policy was actually fed rather than only the unusual case.
        log.info(
            "A6 gripper-state mask: %s for %s (state_layout=%r, expect_mask=%r)",
            "ON — gripper state dims zeroed" if self.obs_mask_gripper else "off (no sidecar)",
            checkpoint,
            state_layout,
            expect_mask,
        )
        if self.obs_mask_gripper:
            if action_layout.startswith("droid8"):
                # The splitter packs [R_arm(7), R_grip_droid], which is neither
                # of the two layouts the mask is defined over; masking it would
                # mean inventing a third gripper-slot convention here. Refuse
                # loudly rather than evaluate a masked checkpoint unmasked.
                raise RuntimeError(
                    f"checkpoint {checkpoint!r} carries the A6 gripper-state "
                    f"mask sidecar, but action_layout={action_layout!r} packs "
                    "the 8-dim DROID state, whose gripper slot is not one of "
                    "the masked layouts (canonical/model16) — refusing to "
                    "evaluate a masked checkpoint on an unmasked state (F-63)"
                )

        # Re-apply the pi0.5 state route this checkpoint was TRAINED under,
        # before the model is built (`continuous` patches PI05Pytorch.__init__).
        #
        # Nothing in a checkpoint records the route: all three arms of the
        # 3-way ship a byte-identical policy_preprocessor.json, because the
        # route is a monkeypatch over prompt construction, not saved config.
        # So a `blind` checkpoint loaded normally is fed a prompt full of
        # state digits it was never trained on -- no error, no shape
        # mismatch, just a wrong number. That is F-63 exactly. The route
        # travels in a sidecar beside the run dir; an absent sidecar means
        # `digits`, so every checkpoint predating this evaluates unchanged.
        self.state_route = apply_for_checkpoint(checkpoint)
        if self.state_route != "digits":
            log.info("pi0.5 state route %r re-applied for %s", self.state_route, checkpoint)

        # The gripper transform this checkpoint was TRAINED under (P2 of
        # GRASP_EXPERIMENT_PROTOCOL.md §1), so its ENCODING half can be
        # inverted before anything reaches the wire.
        #
        # Same shape of fact as the route above, and unreadable from the
        # weights for the same reason: a polarity-flipped fine-tune (A4,
        # `x -> 1 - x` on the gripper dims) ships a config and a state dict
        # byte-identical to an unflipped one. Load it without its transform
        # and every gripper command comes out upside down -- no error, no
        # shape mismatch, just a policy that opens to grasp. That is F-63
        # again. So the fact travels in a sidecar beside the run dir, and an
        # ABSENT sidecar means the IDENTITY -- every checkpoint predating
        # this module evaluates exactly as it did before.
        #
        # Imported inside the body rather than at module scope because
        # `camelo/policy` -> `camelo/train` is an upward edge in the layering
        # (AGENTS.md hard rule 3); this mirrors `convert_model_state.py:56`,
        # which reaches the other way for `pack_state` from inside a
        # function for the same reason.
        from camelo.train.gripper_transforms import read_sidecar as read_gripper_sidecar

        self.gripper_transform = read_gripper_sidecar(checkpoint)
        if not self.gripper_transform.is_identity:
            log.info(
                "gripper transform %s re-applied for %s (encoding inverted at eval; "
                "relabeling is train-only and deliberately NOT inverted)",
                self.gripper_transform.to_dict(),
                checkpoint,
            )

        # A sidecar transform and the droid8 splitter are MUTUALLY EXCLUSIVE.
        # The splitter is ITSELF a gripper encoding -- canonical open=1 to
        # DROID open=0 packing in, and back again decoding out
        # (droid8.py:11-13) -- so honouring both would flip the same channel
        # twice and land silently back where it started, since
        # `1 - (1 - x) == x` raises nothing. The exclusion is sound rather
        # than merely convenient: a droid8 checkpoint is foreign, so it was
        # never trained through one of our dataset transforms in the first
        # place. Refused here, before the weights load, for the same reason
        # the eef layouts are: a double flip is a wrong number with a right
        # shape, which is the whole F-63 family.
        if self.real_layout and not self.gripper_transform.is_identity:
            # GripperTransform's dims are the canonical 37-dim state (29/30)
            # and 20-dim action (17/18). Under s27a15 those indices name a
            # wrist-force component and a right-arm joint — applying the
            # transform there is a wrong number with a right shape.
            raise RuntimeError(
                f"checkpoint {checkpoint!r} carries a gripper transform sidecar "
                f"({self.gripper_transform.to_dict()}) but is declared "
                f"{s27a15.LAYOUT!r}: that transform is defined on the canonical "
                "37/20 dims, which name different quantities in the rig's 27/15 "
                "vectors — refusing to apply it to the wrong channel"
            )
        if self.splitter_layout and not self.gripper_transform.is_identity:
            raise RuntimeError(
                f"checkpoint {checkpoint!r} carries a gripper transform sidecar "
                f"({self.gripper_transform.to_dict()}) but is declared "
                f"action_layout={action_layout!r}: the droid8 splitter already "
                "flips the gripper in both directions, so honouring both would "
                "double-encode the same channel — drop one"
            )

        # OFF by default (plan W4 / protocol B1): reroutes ACT through
        # lerobot's ACTTemporalEnsembler instead of the raw predict_action_
        # chunk() output — see _apply_temporal_ensemble and infer() below.
        # Applied INSIDE _load_policy, before ACTPolicy is constructed
        # (config.temporal_ensemble_coeff must be set before __init__ runs
        # for the ensembler to be built at all).
        self.temporal_ensemble_coeff = temporal_ensemble_coeff
        self.policy, self.is_groot_base = _load_policy(
            checkpoint, device, embodiment_tag, temporal_ensemble_coeff
        )
        self.task = ""
        cfg = self.policy.config
        self.image_features = [k for k in cfg.input_features if "image" in k]
        self.state_features = [k for k in cfg.input_features if "state" in k or "environment" in k]
        self.action_dim = cfg.output_features["action"].shape[0]

        # Single-instance arm-splitter mode (F-58, ZERO_SHOT_RUNBOOK.md §1.2):
        # explicit camera map (never positional), 8-dim state, per-layout
        # decode, and the checkpoint's native 15 Hz chunk timing.
        self.splitter = self.camera_map = None
        if self.splitter_layout:
            from camelo.policy.adapters.droid8 import (
                DROID_CHUNK_HZ,
                Droid8RightSplitter,
                resolve_camera_map,
            )

            self.splitter = Droid8RightSplitter(delta_arm=action_layout.endswith("_delta"))
            self.camera_map = resolve_camera_map(self.image_features, checkpoint, camera_map)
            self.chunk_dt = chunk_dt if chunk_dt is not None else 1.0 / DROID_CHUNK_HZ
        else:
            if camera_map and not self.real_layout:
                raise ValueError(
                    "--camera-map is a droid8-splitter option; layout "
                    f"{action_layout!r} assigns cameras in contract order"
                )
            if self.real_layout:
                # EXPLICIT, from this checkpoint's own train_config rename_map.
                # The sim path assigns cameras positionally in contract order,
                # which is right when the model takes all three; VLA-JEPA takes
                # TWO (head + wrist_right), and positionally its second slot
                # gets wrist_left — the "wrong everywhere" row of the parity
                # failure table (§1.2), and F-64 again.
                self.camera_map = s27a15.resolve_camera_map(
                    self.image_features, training_rename_map(checkpoint), camera_map
                )
                log.info(
                    "s27a15 camera map (checkpoint key <- our camera): %s", self.camera_map
                )
            if chunk_dt is not None:
                self.chunk_dt = chunk_dt
            else:
                # W5: derive from the training dataset's own fps rather than
                # assuming the recorder's 30 — a rate-matched checkpoint (7.5
                # Hz / 2 Hz corpora) must replay chunks at ITS rate, or the
                # executor stretches/compresses every chunk silently (F-63
                # family: a wrong number with a right shape). A checkpoint
                # trained at 30 fps resolves to 1/30 either way, so nothing
                # about existing runs changes.
                fps = training_dataset_fps(checkpoint)
                if fps:
                    self.chunk_dt = 1.0 / fps
                    log.info(
                        "chunk_dt=%.6f derived from training dataset fps=%s for %s",
                        self.chunk_dt, fps, checkpoint,
                    )
                elif self.real_layout:
                    # The rig corpus is 20 Hz. `train_config.json` points at a
                    # cluster path that does not exist on the rig box, so the
                    # fps lookup legitimately fails there — and falling through
                    # to the sim recorder's 30 would replay every chunk 1.5x
                    # fast, straight into the slew clamp.
                    self.chunk_dt = 1.0 / s27a15.FPS
                    log.info(
                        "chunk_dt=%.6f (s27a15 corpus rate %.0f Hz — the training "
                        "dataset's meta/info.json was not resolvable from %s)",
                        self.chunk_dt, s27a15.FPS, checkpoint,
                    )
                else:
                    self.chunk_dt = 1.0 / 30.0
                    log.info(
                        "chunk_dt=%.6f (default 30 fps — training dataset fps not "
                        "resolvable for %s)",
                        self.chunk_dt, checkpoint,
                    )
            if self.real_layout and chunk_dt is None:
                fps_seen = training_dataset_fps(checkpoint)
                if fps_seen and abs(fps_seen - s27a15.FPS) > 1e-6:
                    raise RuntimeError(
                        f"checkpoint {checkpoint!r} is declared {s27a15.LAYOUT!r} but "
                        f"its training dataset runs at {fps_seen} fps, not "
                        f"{s27a15.FPS}. Either this is not a Munich-corpus "
                        "checkpoint, or its train_config.json points at the wrong "
                        "dataset — pass --chunk-dt explicitly if the rate is "
                        "deliberate"
                    )

        # Denoising budget of a flow-matching / diffusion head (pi0, pi0.5:
        # `num_inference_steps`, default 10). Purely an inference-time knob —
        # the same weights, integrated more finely — so it is a legitimate
        # eval-side lever, unlike anything that changes the checkpoint. It
        # costs latency linearly, which the executor sees as a later replan.
        # A policy whose config has no such field would ignore the flag
        # silently, which is exactly how a "sensitivity sweep" ends up
        # comparing a value against itself — refuse instead.
        if num_inference_steps is not None:
            if not hasattr(cfg, "num_inference_steps"):
                raise RuntimeError(
                    f"--num-inference-steps given but {cfg.type} has no such "
                    "config field (it is a flow-matching/diffusion knob; pi0 "
                    "and pi0.5 have it, ACT and GR00T do not) — drop the flag "
                    "rather than run a sweep that changes nothing"
                )
            log.info(
                "num_inference_steps %s -> %d", cfg.num_inference_steps, num_inference_steps
            )
            cfg.num_inference_steps = int(num_inference_steps)
        self.num_inference_steps = getattr(cfg, "num_inference_steps", None)

        # Modern lerobot ships per-policy pre/post processors (tokenization,
        # normalization, device placement) — use them when available. For a
        # GR00T base model this call routes to the groot branch, which reads
        # the repo's own sidecars (processor_config.json + the embodiment
        # tag's statistics.json entry).
        self.preprocess = self.postprocess = None
        # GR00T base: processors must see a local directory (see
        # _resolve_groot_local_path / is_raw_groot_n1_7_checkpoint).
        if self.is_groot_base:
            processor_path = getattr(cfg, "base_model_path", None) or checkpoint
        else:
            processor_path = checkpoint
        try:
            from lerobot.policies.factory import make_pre_post_processors

            self.preprocess, self.postprocess = make_pre_post_processors(
                cfg,
                processor_path,
                preprocessor_overrides={"device_processor": {"device": str(device)}},
            )
        except Exception as exc:  # older lerobot or checkpoint without processor config
            if self.is_groot_base:
                # Without the pipeline there is no head selection and no
                # relative->absolute conversion — the output would be
                # plausible-looking garbage (F-12/F-41). Never fall through.
                raise RuntimeError(
                    "GR00T base models require the processor pipeline "
                    f"(embodiment head + relative->absolute): {exc}"
                ) from exc
            if self.real_layout:
                # The pipeline IS the normalization (and, for VLA-JEPA, the
                # camera rename and the action clip). Falling back to the
                # manual batch builder would hand the policy unnormalized
                # 27-dim state and emit unnormalized actions — the "every dim
                # inflated by a factor" row of the parity table, on a real
                # robot. Never fall through on the rig path.
                raise RuntimeError(
                    f"checkpoint {checkpoint!r} is declared {s27a15.LAYOUT!r} but its "
                    f"processor pipeline did not load ({exc}). That pipeline carries "
                    "the normalizer built from this corpus; running without it is a "
                    "wrongly-scaled command stream with no error (F-12)"
                ) from exc
            log.warning("no processor pipeline (%s); using manual batch builder", exc)

        # -- images -------------------------------------------------------
        # s27a15 applies NO image op of its own. The only geometric step is
        # the checkpoint's OWN training Resize, replayed with the same
        # torchvision v2 transform on the same uint8 tensor the dataloader
        # resized (see training_image_transforms) — everything else
        # (normalization, tokenization, device placement) belongs to the
        # policy's processor pipeline and is left there.
        self.train_resize = None
        self.image_size = None if self.real_layout else training_image_size(checkpoint)
        if self.real_layout:
            self.train_resize = self._build_train_resize(checkpoint)
        if self.image_size is not None:
            log.info(
                "resizing every camera to %dx%d — the Resize this checkpoint "
                "trained under, which config.input_features does not record",
                *self.image_size,
            )

        # F-84: prefer the width the state was NORMALIZED at over the width
        # the checkpoint declares. Falls back to the declared width when the
        # pipeline carries no stats for the key, so nothing changes for
        # checkpoints that never had the mismatch.
        self.state_dims = {}
        for key in self.state_features:
            declared = cfg.input_features[key].shape[0]
            width = stats_width(self.preprocess, key) or declared
            if width != declared:
                log.info(
                    "%s: packing %d-dim state (the normalizer's stat width) "
                    "rather than the declared %d — the policy pads the rest "
                    "internally",
                    key,
                    width,
                    declared,
                )
            self.state_dims[key] = width

        log.info(
            "loaded %s from %s: image features %s, action dim %d, processors=%s",
            cfg.type,
            checkpoint,
            self.image_features,
            self.action_dim,
            self.preprocess is not None,
        )
        if self.temporal_ensemble_coeff is not None:
            log.info(
                "ACT temporal ensembling ON: coeff=%s, n_action_steps forced to 1 "
                "(infer() now calls policy.select_action() instead of "
                "predict_action_chunk() and returns a 1-step chunk — plan W4 / "
                "protocol B1)",
                self.temporal_ensemble_coeff,
            )
        if self.action_layout == "canonical" and self.action_dim != C.ACTION_DIM:
            raise RuntimeError(
                f"action_layout='canonical' declared but checkpoint emits "
                f"{self.action_dim}-dim actions (contract is {C.ACTION_DIM})"
            )
        if self.splitter is not None:
            if self.action_dim < 8:
                raise RuntimeError(
                    f"action_layout={self.action_layout!r} declared but the "
                    f"checkpoint emits {self.action_dim}-dim actions (< 8) — "
                    "not a DROID-format checkpoint"
                )
            if self.action_dim > 8:
                log.info(
                    "droid8: %d-dim padded action, front-slicing to 8 "
                    "(pi05-family pads its 8 real dims — runbook §3)",
                    self.action_dim,
                )
            log.info("droid8 camera map (checkpoint key <- our camera): %s", self.camera_map)
        # The executed window, per policy, from the checkpoint's own config —
        # 10 for ACT (chunk 21), 7 for VLA-JEPA (chunk 7). The rig executor
        # replans after this many steps at chunk_dt; §3.1 is free to override
        # it, but the DEFAULT must be what the checkpoint declares rather than
        # a number typed into a runbook.
        self.n_action_steps = int(getattr(cfg, "n_action_steps", 0) or 0)
        if self.real_layout:
            if self.n_action_steps <= 0:
                raise RuntimeError(
                    f"checkpoint {checkpoint!r} declares no n_action_steps; the rig "
                    "executor will not guess an execution window"
                )
            if self.action_dim != s27a15.ACTION_DIM:
                raise RuntimeError(
                    f"action_layout={s27a15.LAYOUT!r} declared but the checkpoint "
                    f"emits {self.action_dim}-dim actions (the real corpus is "
                    f"{s27a15.ACTION_DIM})"
                )
            for key, width in self.state_dims.items():
                if width != s27a15.STATE_DIM:
                    raise RuntimeError(
                        f"state_layout={s27a15.LAYOUT!r} declared but {key} is "
                        f"{width}-dim (the real corpus is {s27a15.STATE_DIM})"
                    )
            log.info(
                "s27a15: %d-dim state in, %d-dim action out; chunk %d, "
                "n_action_steps %d, chunk_dt %.4f s (%.1f Hz); gripper command "
                "convention %r",
                s27a15.STATE_DIM,
                s27a15.ACTION_DIM,
                getattr(cfg, "chunk_size", None) or getattr(cfg, "horizon", -1),
                self.n_action_steps,
                self.chunk_dt,
                1.0 / self.chunk_dt,
                self.gripper_command,
            )
        if self.action_layout == "model16" and self.action_dim != C.MODEL_ACTION_DIM:
            log.warning(
                "checkpoint action dim %d != model dim %d — first-%d slice/pad "
                "best effort under declared layout 'model16' (width alone never "
                "implies the canonical contract — F-45)",
                self.action_dim,
                C.MODEL_ACTION_DIM,
                C.MODEL_ACTION_DIM,
            )

    def reset(self, task: str) -> None:
        if self.real_layout:
            # F-68, on the rig side. VLA-JEPA reads batch["task"] and
            # conditions on it (modeling_vla_jepa.py:395), so a caption typed
            # by hand — an added period, "blue thermal pad" from the sim
            # lineage — is a different prompt from the one 30,000 steps were
            # trained under, with no error to say so.
            if not task:
                raise RuntimeError(
                    f"the {s27a15.LAYOUT!r} rungs were trained with a caption and "
                    "at least one of them conditions on it — refusing an empty "
                    f"--task. Use: {s27a15.TASK_CAPTION!r}"
                )
            if task != s27a15.TASK_CAPTION and not self.allow_caption_drift:
                raise RuntimeError(
                    f"--task {task!r} is not the corpus caption "
                    f"{s27a15.TASK_CAPTION!r} (byte-exact from meta/tasks.parquet, "
                    "no trailing period — F-68). Pass allow_caption_drift=True / "
                    "--allow-caption-drift only for a deliberate prompt experiment, "
                    "and say so in the log."
                )
        self.task = task
        self.policy.reset()
        if self.splitter is not None:
            self.splitter.reset()  # re-latch the park pose on the first infer()

    # -- s27a15 -----------------------------------------------------------
    def _build_train_resize(self, checkpoint: str):
        """The checkpoint's own training Resize, or None. Refuses augmentations.

        The transform set is replayed, not approximated: same class, same
        kwargs, same input dtype. Anything OTHER than a resize in an enabled
        set is a random augmentation — `ImageTransforms` samples a subset per
        call — and a stochastic preprocessing step at inference makes every
        rollout unreproducible and the parity check meaningless.
        """
        blob = training_image_transforms(checkpoint)
        if blob is None:
            return None
        tfs = blob.get("tfs") or {}
        enabled = {k: v for k, v in tfs.items() if float(v.get("weight", 1.0) or 0.0) > 0.0}
        extra = sorted(k for k in enabled if k != "resize")
        if extra:
            raise RuntimeError(
                f"checkpoint {checkpoint!r} trained with image transforms {extra} "
                "enabled alongside the resize. Those are random augmentations "
                "(lerobot samples a subset per call), so replaying them at "
                "inference would make every observation nondeterministic — "
                "refusing rather than silently dropping them"
            )
        spec = enabled.get("resize")
        if spec is None:
            return None
        if spec.get("type") != "Resize":
            raise RuntimeError(
                f"checkpoint {checkpoint!r} declares an enabled 'resize' transform "
                f"of type {spec.get('type')!r}, not 'Resize'"
            )
        from torchvision.transforms import v2

        kwargs = dict(spec.get("kwargs") or {})
        log.info("s27a15: replaying the training Resize(%s) on every camera", kwargs)
        return v2.Resize(**kwargs)

    def _real_image_tensor(self, image: np.ndarray) -> object:
        """HxWx3 uint8 RGB -> the exact tensor the training dataloader produced.

        uint8 CHW -> (training Resize, if any) -> float32 / 255. The order is
        load-bearing: lerobot resizes the uint8 tensor
        (`dataset_reader.get_item`) and only then casts and scales
        (`lerobot_train.py:603-606`). Resizing after the cast rounds at a
        different point and changes pixels.
        """
        torch = self.torch
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
        if self.train_resize is not None:
            tensor = self.train_resize(tensor)
        return tensor.to(dtype=torch.float32) / 255.0

    def _rig_state(self, obs: Obs):
        """The s27a15 state source: named rig groups, or an already-packed 27."""
        if obs.rig is not None:
            return s27a15.RigObservation.from_mapping(obs.rig)
        state = np.asarray(obs.state, dtype=np.float32).ravel()
        if state.size == s27a15.STATE_DIM:
            return state
        raise RuntimeError(
            f"state_layout={s27a15.LAYOUT!r} needs either Obs.rig (named groups "
            "from the real-robot bridge: left_arm, right_arm, right_gripper_rad, "
            f"left_wrench, right_wrench) or an already-packed {s27a15.STATE_DIM}-dim "
            f"Obs.state; got a {state.size}-dim state and no rig groups"
        )

    def rig_command(self, chunk: np.ndarray) -> object:
        """(T, 15) chunk -> `s27a15.RigCommand` in the driver's units.

        The seam between this adapter and the rig's own controllers. Kept
        here so the gripper convention the adapter was loaded with is the one
        the command is built with — a second call site would be a second
        chance to disagree about polarity.
        """
        if not self.real_layout:
            raise RuntimeError(
                f"rig_command() is the {s27a15.LAYOUT!r} seam; this adapter is "
                f"action_layout={self.action_layout!r}"
            )
        return s27a15.unpack_action(chunk, self.gripper_command, self.chunk_dt)

    # -- input assembly ----------------------------------------------------
    def _image_tensor(self, image: np.ndarray, shape) -> object:
        torch = self.torch
        # A training-time Resize overrides the declared per-feature shape —
        # see training_image_size(). Without it GR00T's packer raises on our
        # ragged camera shapes, and any policy would see the wrong geometry.
        target = self.image_size or (None if shape is None else (shape[1], shape[2]))
        if target is not None and image.shape[:2] != tuple(target):
            from PIL import Image

            image = np.asarray(Image.fromarray(image).resize((target[1], target[0])))
        return torch.from_numpy(image.copy()).float().permute(2, 0, 1) / 255.0

    # -- gripper transform (P2): the two CANONICAL seams -------------------
    #
    # The transform is defined on canonical arrays only -- the 37-dim state
    # (S_LEFT_GRIP=29 / S_RIGHT_GRIP=30) and the 20-dim action
    # (A_LEFT_GRIP=17 / A_RIGHT_GRIP=18) -- so these two wrappers are the
    # only places it may be called, and each sits on the canonical side of
    # its layout coercion: `_encode_state` runs BEFORE `pack_state` projects
    # to the 16-dim model proprio, `_decode_chunk` runs AFTER `coerce_chunk`
    # has mapped the chunk back to 20 dims. On the other side of either
    # coercion the same indices name different quantities (37-dim 29/30 are
    # the grippers; model16 29/30 do not exist, and 17/18 of a model16 chunk
    # are right-arm joints), so the flip would land on the wrong dims and
    # produce a wrong number with a right shape.
    #
    # Both short-circuit on the identity, so a checkpoint with no sidecar --
    # i.e. every checkpoint on disk today -- runs the same code over the same
    # arrays it ran before this existed, with nothing copied.

    def _encode_state(self, state: np.ndarray) -> np.ndarray:
        """Canonical 37-dim observation -> what the dataset column held.

        Same call the converter made (`apply_states`), so train and eval
        cannot disagree about the state side.
        """
        if self.gripper_transform.is_identity:
            return state
        return self.gripper_transform.encode_state(state)

    def _decode_chunk(self, actions: np.ndarray) -> np.ndarray:
        """Canonical 20-dim chunk -> canonical units, the ENCODING undone.

        Only the encoding half is inverted, and that asymmetry is the point.
        A1's lead is a RELABELING with no eval-side inverse: undoing it would
        shift the policy's early close back to where the untransformed data
        would have put it and cancel the experiment outright, so a lead-only
        transform decodes as the identity on purpose
        (camelo/train/gripper_transforms.py).
        """
        if self.gripper_transform.is_identity:
            return actions
        return self.gripper_transform.decode_actions(actions)

    def _frame(self, obs: Obs) -> dict:
        """Dataset-frame-shaped input for the processor pipeline."""
        torch = self.torch
        frame: dict = {"task": self.task}
        if self.real_layout:
            for key, camera in self.camera_map.items():
                if camera not in obs.images:
                    raise RuntimeError(
                        f"s27a15 needs camera {camera!r} for feature {key!r} but the "
                        f"observation carries {sorted(obs.images)} — check --cameras"
                    )
                frame[key] = self._real_image_tensor(obs.images[camera])
            rig_state = self._rig_state(obs)
            for key in self.state_features:
                frame[key] = torch.from_numpy(
                    pack_state(rig_state, self.state_layout, self.state_dims[key])
                )
            return frame
        if self.splitter is not None:
            # Explicit key->camera map, every declared feature present: a
            # missing key is FILLED by MolmoAct2's allow_image_key_fallback
            # instead of erroring, and GR00T falls back to alphabetical
            # order — never let either happen (ZERO_SHOT_RUNBOOK.md §1.2a).
            for key, camera in self.camera_map.items():
                if camera not in obs.images:
                    raise RuntimeError(
                        f"droid8 camera map needs {camera!r} for {key!r} but the "
                        f"observation carries {sorted(obs.images)} — check --cameras"
                    )
                shape = self.policy.config.input_features[key].shape
                frame[key] = self._image_tensor(obs.images[camera], shape)
            droid_state = self.splitter.pack_state(obs.state)
            for key in self.state_features:
                dim = self.policy.config.input_features[key].shape[0]
                if dim != droid_state.size:
                    raise RuntimeError(
                        f"state feature {key!r} is {dim}-dim but droid8 packs "
                        f"{droid_state.size}-dim [R_arm(7), R_grip_droid] — this "
                        "checkpoint is not 8-dim-DROID-shaped"
                    )
                frame[key] = torch.from_numpy(droid_state.copy())
            return frame
        images = obs.image_list()  # contract order: head, wrist_left, wrist_right
        if self.image_features and not images:
            raise RuntimeError(
                f"policy expects image features {self.image_features} but the "
                "observation carries no images — check the --cameras subset"
            )
        for i, key in enumerate(self.image_features):
            shape = self.policy.config.input_features[key].shape
            frame[key] = self._image_tensor(images[min(i, len(images) - 1)], shape)
        # Encode on the CANONICAL 37-dim state, before any layout coercion
        # (see the seam note above `_encode_state`).
        state = self._encode_state(obs.state)
        for key in self.state_features:
            # Two state paths compose here: the droid8 layouts pack their
            # 8-dim state upstream and never reach this branch; everything
            # else goes through pack_state's declared layouts (F-63). If a
            # third path ever appears, fold them into one layout registry
            # instead of stacking another branch.
            dim = self.state_dims[key]
            # A4 decode FIRST (already done, on the canonical 37-dim
            # `state` above), then the A6 mask inside `pack_state_masked`.
            # The two COMPOSE — they are not alternatives: a run can be both
            # polarity-flipped and gripper-masked, and dropping either half
            # is a wrong number with a right shape (F-63). Order matters
            # because the mask is defined on the PACKED layout's gripper
            # slots while the encoding is defined on canonical dims 29/30.
            frame[key] = torch.from_numpy(
                pack_state_masked(state, self.state_layout, dim, self.obs_mask_gripper)
            )
        return frame

    def _batch_manual(self, obs: Obs) -> dict:
        frame = self._frame(obs)
        batch = {"task": [frame.pop("task")]}
        for key, value in frame.items():
            batch[key] = value.unsqueeze(0).to(self.device)
        return batch

    def _postprocess_chunk(self, chunk):
        """Unnormalize a (B, T, A) chunk; NEVER silently return raw output.

        The postprocessor is what unnormalizes (and, for GR00T, converts
        relative -> absolute) — raw normalized output fed to the executor is
        a plausible-looking but wrongly-scaled command stream (F-12). Some
        postprocessors only accept (B, A), so fall back to a per-timestep
        loop (lerobot's own async server does the same); if both fail, raise.
        """
        if self.postprocess is None:
            return chunk
        try:
            return self.postprocess(chunk)
        except Exception as full_exc:
            try:
                steps = [self.postprocess(chunk[:, t]) for t in range(chunk.shape[1])]
                return self.torch.stack(steps, dim=1)
            except Exception as step_exc:
                raise RuntimeError(
                    "action postprocessing failed for both full-chunk "
                    f"({full_exc}) and per-timestep ({step_exc}) shapes — "
                    "refusing to emit un-unnormalized actions"
                ) from step_exc

    # -- inference ---------------------------------------------------------
    def infer(self, obs: Obs) -> np.ndarray:
        torch = self.torch
        if self.preprocess is not None:
            batch = self.preprocess(self._frame(obs))
        else:
            batch = self._batch_manual(obs)

        with torch.inference_mode():
            if self.temporal_ensemble_coeff is not None:
                # select_action() is the ONLY call that touches the
                # ensembler: it runs predict_action_chunk() internally and
                # then folds the chunk into ACTTemporalEnsembler.update()
                # (modeling_act.py) — predict_action_chunk() alone would
                # skip the ensembler entirely and return the raw chunk, as
                # if this flag were never set.
                chunk = self.policy.select_action(batch).unsqueeze(1)  # (B, 1, A)
            elif hasattr(self.policy, "predict_action_chunk"):
                chunk = self.policy.predict_action_chunk(batch)
            else:  # single-step policies (queue-based select_action)
                chunk = self.policy.select_action(batch).unsqueeze(1)
        # Some policies return (chunk_size, action_dim) without a batch axis;
        # normalize to (B, T, A) like lerobot's own server does (F-13).
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)
        chunk = self._postprocess_chunk(chunk)
        actions = chunk[0].float().cpu().numpy()
        if self.splitter is not None:
            # The splitter decodes to canonical AND owns the gripper flip for
            # DROID checkpoints; __init__ refuses to pair it with a sidecar
            # transform, so there is nothing left to undo here.
            return self.splitter.decode(actions, obs.state)
        # coerce_chunk FIRST, then decode: the transform's action dims are
        # canonical (17/18) and do not exist until the chunk is 20-dim. This
        # is the last point before the chunk reaches the executor.
        return self._decode_chunk(coerce_chunk(actions, self.action_layout))
