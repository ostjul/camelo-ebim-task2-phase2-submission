"""Single-instance arm-splitter for 8-dim DROID checkpoints (F-58).

Task 2's reference demos are right-arm-only (base identically zero, left
arm parked after settle — F-58), so a DROID-format policy
`[joint7, gripper]` can drive the robot through a single instance: the
right arm takes the policy's output, the left arm is latched at its
measured settle pose for the whole episode. Spec: ZERO_SHOT_RUNBOOK.md
§1.2; candidate glue: ZERO_SHOT_POLICIES.md (per-arm rungs).

Conventions handled here, in both directions:
- DROID gripper is 0.0=open / 1.0=closed; our canonical channel is
  1.0=open. The flip is ``x -> 1 - x`` on the state going in AND the
  action coming out.
- ``droid8_right`` decodes ABSOLUTE joint radians (MolmoAct2-DROID).
- ``droid8_right_delta`` decodes joint DELTAS with an ABSOLUTE gripper
  (pi05_droid_jointpos): ``cumsum`` anchored on the MEASURED right-arm
  joints at each chunk start — re-anchoring every chunk keeps
  integration drift from accumulating across an episode.

Numpy-only on purpose: unit-testable without torch, lerobot, ROS or a
checkpoint on disk.
"""

from __future__ import annotations

import logging

import numpy as np

from camelo import contracts as C

log = logging.getLogger(__name__)

DROID_STATE_DIM = 8  # [joint_position(7), gripper_position(1)], 15 Hz
DROID_CHUNK_HZ = 15.0  # both ladder checkpoints emit chunk_size 15 at 15 Hz

# Camera maps per checkpoint, OUR camera name per checkpoint image key.
# Explicit and exact-name by design: MolmoAct2's preprocessor sets
# allow_image_key_fallback=true, so a missing key is silently filled, and
# GR00T falls back to ALPHABETICAL order on zero matches — both are
# confident-wrong-answer traps (ZERO_SHOT_RUNBOOK.md §1.2a). The "left" in
# DROID's exterior/wrist key names is the ZED stereo-left frame, NOT a
# robot side — the wrist view must come from the ACTIVE (right) arm.
DEFAULT_CAMERA_MAPS = {
    "lerobot/MolmoAct2-DROID-LeRobot": {
        "observation.images.exterior_1_left": "head",
        # DROID rigs carry two exteriors and training sampled one; we have
        # a single head camera, so both slots see it (runbook fallback
        # knob 3 swaps this for the parked arm's wrist view).
        "observation.images.exterior_2_left": "head",
        "observation.images.wrist_left": "wrist_right",
    },
    "DAVIAN-Robotics/pi05_droid_jointpos": {
        # openpi naming; here left/right ARE robot sides.
        "observation.images.base_0_rgb": "head",
        "observation.images.left_wrist_0_rgb": "wrist_left",
        "observation.images.right_wrist_0_rgb": "wrist_right",
    },
}


def resolve_camera_map(
    image_features: list[str], checkpoint: str, override: dict[str, str] | None = None
) -> dict[str, str]:
    """Checkpoint image key -> our camera name, for every declared feature.

    Never positional, never partial: an unmapped feature raises instead of
    letting a preprocessor fallback pick a wrong camera silently.
    ``override`` entries may use the full feature key or its last
    dot-segment (``exterior_1_left=head``).
    """
    base = dict(DEFAULT_CAMERA_MAPS.get(checkpoint, {}))
    for key, camera in (override or {}).items():
        matches = [f for f in image_features if f == key or f.rsplit(".", 1)[-1] == key]
        if not matches:
            raise ValueError(
                f"--camera-map entry {key!r} matches no image feature of this "
                f"checkpoint (features: {image_features})"
            )
        for f in matches:
            base[f] = camera
    mapped, unmapped = {}, []
    for feature in image_features:
        camera = base.get(feature)
        if camera is None:
            unmapped.append(feature)
        elif camera not in C.CAMERA_KEYS:
            raise ValueError(f"camera {camera!r} for {feature!r} not in {C.CAMERA_KEYS}")
        else:
            mapped[feature] = camera
    if unmapped:
        raise ValueError(
            f"no camera mapping for image features {unmapped} of {checkpoint!r} — "
            "the droid8 splitter refuses positional/fallback camera assignment "
            "(ZERO_SHOT_RUNBOOK.md §1.2a); pass --camera-map <feature>=<camera>"
        )
    return mapped


class Droid8RightSplitter:
    """State packing + action decode for one DROID policy on the right arm."""

    def __init__(self, delta_arm: bool = False):
        self.delta_arm = delta_arm
        self.left_park: np.ndarray | None = None  # (7,) canonical radians
        self.left_grip_park: float = 1.0  # canonical 1=open

    def reset(self) -> None:
        """New episode: drop the park latch; re-latch on the first infer()
        so the park is the MEASURED settle pose, not a stale one (§1.2d)."""
        self.left_park = None
        self.left_grip_park = 1.0

    def pack_state(self, state: np.ndarray) -> np.ndarray:
        """37-dim recorder state -> 8-dim DROID state, gripper flipped."""
        out = np.empty(DROID_STATE_DIM, dtype=np.float32)
        out[:7] = state[C.S_RIGHT_ARM]
        out[7] = 1.0 - float(state[C.S_RIGHT_GRIP])  # canonical open=1 -> DROID open=0
        return out

    def decode(self, actions: np.ndarray, state: np.ndarray) -> np.ndarray:
        """(T, >=8) checkpoint chunk -> (T, 20) canonical.

        Latching happens here (the adapter's reset(task) gets no
        observation). Wider chunks are front-sliced: pi05_droid_jointpos
        pads its 8 real dims to 32 (verified from its config.json).
        """
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] < DROID_STATE_DIM:
            raise RuntimeError(
                f"droid8 layout needs a (T, >=8) chunk, got {actions.shape} — "
                "this checkpoint does not speak the DROID action format"
            )
        if actions.shape[1] > DROID_STATE_DIM:
            log.debug("front-slicing %d-dim padded chunk to 8", actions.shape[1])
            actions = actions[:, :DROID_STATE_DIM]

        if self.left_park is None:
            self.left_park = np.asarray(state[C.S_LEFT_ARM], dtype=np.float32).copy()
            self.left_grip_park = float(state[C.S_LEFT_GRIP])
            log.info(
                "droid8: latched left-arm park pose (grip %.2f) from measured state",
                self.left_grip_park,
            )

        if self.delta_arm:
            # Anchor on the MEASURED joints at chunk start, not the previous
            # chunk's end — re-anchoring bounds integration drift (§3).
            anchor = np.asarray(state[C.S_RIGHT_ARM], dtype=np.float32)
            arm = anchor[None, :] + np.cumsum(actions[:, :7], axis=0)
        else:
            arm = actions[:, :7]

        out = np.zeros((actions.shape[0], C.ACTION_DIM), dtype=np.float32)
        out[:, C.A_RIGHT_ARM] = arm
        out[:, C.A_RIGHT_GRIP] = 1.0 - actions[:, 7]  # DROID open=0 -> canonical open=1
        out[:, C.A_LEFT_ARM] = self.left_park
        out[:, C.A_LEFT_GRIP] = self.left_grip_park
        # A_BASE stays 0.0 — the base is identically zero in every reference
        # episode (F-58); A_SPINE NaN = executor holds (no ROS command path).
        out[:, C.A_SPINE] = np.nan
        return out
