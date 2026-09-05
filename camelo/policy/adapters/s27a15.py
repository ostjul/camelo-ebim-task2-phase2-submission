"""The `s27a15` real-robot layout: 27-dim state in, 15-dim action out.

The Munich rig's contract, and the third layout the LeRobot adapter speaks
alongside the sim's `model16` / `canonical` (37/20/16). It is NOT a
re-slicing of the sim contract — different robot, different recorder,
different vector — so no *layout* here is derived from `camelo.contracts`.
The one place the two vectors meet is the chunk seam at the bottom of this
module (`chunk_for_executor`), which exists precisely so that meeting
happens ONCE, in the open, with both sets of index constants visible
side by side (`A_LEFT_ARM` here is the rig's, `C.A_LEFT_ARM` the sim's).
That seam used to live in `camelo/control/rig_chunk.py`, which made
`control` import `policy` and broke the one-way layering in
`camelo/__init__.py` (AGENTS.md hard rule 3, now machine-checked by
`tests/test_layering.py`). The two other things borrowed from contracts are
gripper scalars, not layout: `GRIPPER_CLOSED_RAD`, which the corpus builder
deliberately shares so the real and sim gripper channels stay mergeable
(`camelo/train/build_real_corpus.py`, D7), and `gripper_width_percent`,
which is also what `camelo/ros/command_publisher.py` publishes on the real
gripper topic — one definition, so the decode and the publish cannot
disagree about how open "open" is. (That topic's units were MEASURED on
2026-09-02: a 0..1 open fraction, despite the name. docs/CONTRACTS.md.)

Provenance for every number below, in order of authority:
  * `outputs/datasets/.../task2_munich_s27a15_hires/meta/info.json`
    — the 27 state names and 15 action names, in order, as the policies
    were actually trained;
  * `camelo/train/build_real_corpus.py` (`STATE_KEEP_S27`,
    `gripper_open_fraction`) — the code that built them;
  * `docs/realdata/00_PROTOCOL.md` §0.2 and 15_RIG_WINDOW_RUNBOOK.md §1.3.

`tests/test_s27a15_layout.py` checks the frozen names here against the
corpus's own `meta/info.json` when the corpus is on disk, so this file
cannot drift from what the checkpoints saw.

**Why a named observation and not a raw vector.** The sim side can pack
from a 37-dim array because `camelo/contracts.py` mirrors the recorder
that built it and a drift alarm keeps the mirror honest. There is no such
mirror for the Munich rig: this repo has never seen its ROS graph, and
the field order of its joint-state messages is UNCONFIRMED
(12_INFERENCE_PATH_NOTE.md). So the bridge hands over **named** groups
and this module does the ordering — a wrong name raises, where a wrong
index would have produced a confident wrong answer.

Numpy + stdlib only: unit-testable with no torch, no lerobot, no ROS, no
checkpoint on disk.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from camelo import contracts as C
from camelo.contracts import GRIPPER_CLOSED_RAD, gripper_width_percent

log = logging.getLogger(__name__)

LAYOUT = "s27a15"
#: `PolicyAdapter.action_space` for the SIM contract — the other value the
#: chunk seam at the bottom of this module dispatches on. `LAYOUT` is this
#: one's counterpart, and callers say `s27a15.LAYOUT` for the rig.
CANONICAL_SPACE = "canonical20"

#: Widths of the two vectors. Both are checked against the checkpoint's own
#: declared feature shapes at adapter load — never inferred from them.
STATE_DIM = 27
ACTION_DIM = 15

#: The corpus is 20 Hz where the whole sim lineage was 30 fps. Every chunk
#: horizon in 15_RIG_WINDOW_RUNBOOK.md §1.6 was re-derived at this rate;
#: replaying a chunk at 1/30 would run it 1.5x fast into the slew clamp.
FPS = 20.0

# ---------------------------------------------------------------------------
# observation.state — 27 dims
# ---------------------------------------------------------------------------
S_LEFT_ARM = slice(0, 7)
S_RIGHT_ARM = slice(7, 14)
S_RIGHT_GRIP = 14
S_LEFT_WRENCH = slice(15, 21)
S_RIGHT_WRENCH = slice(21, 27)

# ---------------------------------------------------------------------------
# action — 15 dims. Absolute joint targets in radians; dim i of the state and
# dim i of the action name the same joint for i < 14.
# ---------------------------------------------------------------------------
A_LEFT_ARM = slice(0, 7)
A_RIGHT_ARM = slice(7, 14)
A_RIGHT_GRIP = 14

#: Source indices into the Munich recorder's own 42-dim state column, in
#: output order — kept so a bridge that reproduces the RECORDER's vector can
#: be mapped mechanically instead of by hand. Mirrors
#: `build_real_corpus.STATE_KEEP_S27`.
RECORDER_STATE_INDICES = (
    *range(0, 7),  # left arm
    *range(21, 28),  # right arm
    28,  # right gripper knuckle, RADIANS at the source
    *range(15, 21),  # left external wrench
    *range(36, 42),  # right external wrench
)

STATE_NAMES = (
    *(f"franka_robot_left_measured_joint_states_left_fr3_joint{i}" for i in range(1, 8)),
    *(f"franka_robot_right_measured_joint_states_right_fr3_joint{i}" for i in range(1, 8)),
    "franka_robot_right_gripper_open_fraction",
    *(
        f"franka_robot_left_external_wrench_in_stiffness_frame_{c}"
        for c in ("force_x", "force_y", "force_z", "torque_x", "torque_y", "torque_z")
    ),
    *(
        f"franka_robot_right_external_wrench_in_stiffness_frame_{c}"
        for c in ("force_x", "force_y", "force_z", "torque_x", "torque_y", "torque_z")
    ),
)

ACTION_NAMES = (
    *(f"left_follower_gello_joint_states_fr3_joint{i}" for i in range(1, 8)),
    *(f"right_follower_gello_joint_states_fr3_joint{i}" for i in range(1, 8)),
    "right_follower_gripper_gripper_client_target_gripper_width_percent_value",
)

#: Cameras, in the corpus's own key order. Passed to the checkpoint by an
#: EXPLICIT map built from its `rename_map` (see `resolve_camera_map`), never
#: positionally — VLA-JEPA takes only two of these three and a positional
#: assignment hands it `wrist_left` where it trained on `wrist_right` (F-64).
CAMERA_KEYS = ("head", "wrist_left", "wrist_right")
#: Real-rig native resolutions (H, W). The sim wrists are 480x848; these are
#: 480x640. Checked against the checkpoint's declared feature shapes only
#: where the checkpoint declares native shapes (i.e. no training Resize).
CAMERA_SHAPES = {"head": (720, 1280), "wrist_left": (480, 640), "wrist_right": (480, 640)}

#: The caption, byte-exact from `meta/tasks.parquet` — **no trailing period**
#: (F-68). Typing it is how train/eval skew gets in, so it is typed exactly
#: once, here, and `tests/test_s27a15_layout.py` reads the corpus back
#: against it.
TASK_CAPTION = "Pick up the thermal pad and place it on the target RAM board"


# ---------------------------------------------------------------------------
# Gripper: the state side (rig -> policy) and the command side (policy -> rig)
# ---------------------------------------------------------------------------
def open_fraction_from_knuckle_rad(radians) -> np.ndarray:
    """Raw right-knuckle angle -> the open fraction training built dim 14 from.

    `1 - clip(rad, 0, 0.8) / 0.8`, the exact expression of
    `build_real_corpus.gripper_open_fraction` — 0.0 rad = open = 1.0,
    0.79 rad = closed = ~0.01. The measured knuckle maximum is 0.7929 rad,
    so the closed end lands at 0.0089 rather than 0; that ~1 % dead band is
    part of the trained contract and must NOT be "corrected" here.
    """
    rad = np.asarray(radians, dtype=np.float64)
    return 1.0 - np.clip(rad, 0.0, GRIPPER_CLOSED_RAD) / GRIPPER_CLOSED_RAD


def knuckle_rad_from_open_fraction(open_fraction) -> np.ndarray:
    """Open fraction -> knuckle radians: `rad = 0.8 * (1 - open)`.

    The exact inverse of the above, and the same expression as
    `contracts.opening_to_driver_rad` on the sim side.
    """
    value = np.clip(np.asarray(open_fraction, dtype=np.float64), 0.0, 1.0)
    return (1.0 - value) * GRIPPER_CLOSED_RAD


#: Plausible-units band for the raw knuckle feedback, in radians. A driver
#: reporting millimetres, percent or a normalized 0-1 width lands far outside
#: it, and `clip(rad, 0, 0.8)` would silently turn every one of those into
#: "fully closed" — a perfectly-timed close that never grips (§1.4). So the
#: band raises instead. Slack on both ends for sensor noise around the stops.
KNUCKLE_RAD_MIN = -0.05
KNUCKLE_RAD_MAX = 1.20


#: Command conventions the real gripper driver might want, policy-side value
#: (open fraction, 1.0 = OPEN) -> wire value. Pluggable because the driver's
#: units are the one thing on the rig this repo has never observed; the
#: default is the sim-side convention, which is also the inverse of the
#: transform the corpus was built with.
#:
#: `close_fraction` is what the sim's `/bridge/*` republisher wants with the
#: task2 default `REPUBLISHER_GRIPPER_INVERT=true` (F-18); it is here so the
#: rig can select it without new code if its driver turns out to match.
GRIPPER_COMMAND_CONVENTIONS = {
    "knuckle_rad": knuckle_rad_from_open_fraction,
    "open_fraction": lambda x: np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0),
    "close_fraction": lambda x: 1.0 - np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0),
    # The same function camelo/ros/command_publisher.py publishes with.
    #
    # MEASURED 2026-09-02 (T1(c), docs/CONTRACTS.md "The gripper command
    # unit"): `target_gripper_width_percent` takes a 0..1 OPEN FRACTION, so
    # `width_percent` is now NUMERICALLY IDENTICAL to `open_fraction` above —
    # same clip, same polarity, no ×100. The two entries stay separate on
    # purpose: `width_percent` names the real rig's topic (misnomer and all)
    # and is pinned to the publisher's own callable, so if the driver is ever
    # rescaled only this one moves. Do not collapse them.
    "width_percent": gripper_width_percent,
}
DEFAULT_GRIPPER_COMMAND = "knuckle_rad"


def gripper_command(open_fraction, convention: str = DEFAULT_GRIPPER_COMMAND) -> np.ndarray:
    """Policy gripper output -> the driver's units, per the declared convention."""
    fn = GRIPPER_COMMAND_CONVENTIONS.get(convention)
    if fn is None:
        raise ValueError(
            f"unknown gripper command convention {convention!r}; known: "
            f"{sorted(GRIPPER_COMMAND_CONVENTIONS)}. The default "
            f"{DEFAULT_GRIPPER_COMMAND!r} is the sim-side inverse "
            "(rad = 0.8 * (1 - open)) — verify it live against the real driver "
            "before any rollout (15_RIG_WINDOW_RUNBOOK.md §1.4)"
        )
    return np.asarray(fn(open_fraction), dtype=np.float64)


# ---------------------------------------------------------------------------
# The rig's observation
# ---------------------------------------------------------------------------
def _vector(value, width: int, what: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).ravel()
    if array.size != width:
        raise RuntimeError(
            f"s27a15: {what} must be {width}-dim, got {array.size} "
            f"({np.asarray(value).shape}) — check the rig bridge's field order"
        )
    if not np.all(np.isfinite(array)):
        raise RuntimeError(
            f"s27a15: {what} carries non-finite values {array.tolist()} — a NaN "
            "here reaches the normalizer and the policy emits NaN joint targets"
        )
    return array


@dataclass(frozen=True)
class RigObservation:
    """One live observation from the real-robot bridge, by NAME not index.

    Every field is required. In particular there is **no zero default for
    the wrenches**: 12 of the 27 dims the policies trained on are external
    wrench, they were normalized with the rest of the vector, and handing
    the policy silent zeros where it saw real forces is F-63 in its purest
    form — a wrong number with a right shape. A bridge that genuinely has
    no wrench must say so by constructing `zero_wrench=True` explicitly, at
    which point it is logged on every load.

    Measured, so the guard is structural rather than value-based: the right
    wrench is **exactly** all-zero in 4,359 of 121,828 corpus frames, in 27
    episodes, for runs of up to 395 consecutive frames (~20 s). An all-zero
    reading is therefore real data, not evidence of an unwired topic, and
    nothing here may reject it.
    """

    left_arm: np.ndarray  # (7,) measured joint positions, radians
    right_arm: np.ndarray  # (7,) measured joint positions, radians
    right_gripper_open: float  # open fraction, 1.0 = OPEN (already converted)
    left_wrench: np.ndarray  # (6,) force xyz + torque xyz, stiffness frame
    right_wrench: np.ndarray  # (6,) same, right arm
    wrench_declared_absent: bool = False

    @classmethod
    def from_mapping(cls, obs: Mapping[str, object]) -> RigObservation:
        """Build from the bridge's dict. Raises on anything missing or unclear.

        Required keys: `left_arm`, `right_arm`, `left_wrench`, `right_wrench`,
        and exactly one of `right_gripper_rad` (raw knuckle feedback, the
        expected case) or `right_gripper_open` (a bridge that already
        converted). Supplying both raises rather than picking one.
        """
        missing = [
            k for k in ("left_arm", "right_arm", "left_wrench", "right_wrench") if k not in obs
        ]
        if missing:
            raise RuntimeError(
                f"s27a15 rig observation is missing {missing}. These are policy "
                "state dims that were normalized during training — zeros are NOT "
                "an acceptable silent substitute (F-63). Wire the topic, or "
                "construct RigObservation(..., wrench_declared_absent=True) "
                "deliberately and accept that the run is off-contract."
            )
        has_rad = "right_gripper_rad" in obs
        has_open = "right_gripper_open" in obs
        if has_rad == has_open:
            raise RuntimeError(
                "s27a15 rig observation needs exactly one of 'right_gripper_rad' "
                "(raw knuckle radians, the expected real-robot feedback) or "
                "'right_gripper_open' (already an open fraction, 1.0 = open); "
                f"got rad={has_rad} open={has_open}. The two point OPPOSITE ways "
                "and guessing is how a policy learns to open in order to grasp "
                "(15_RIG_WINDOW_RUNBOOK.md §1.4)."
            )
        if has_rad:
            rad = float(np.asarray(obs["right_gripper_rad"], dtype=np.float64).ravel()[0])
            if not np.isfinite(rad):
                raise RuntimeError("s27a15: right_gripper_rad is non-finite")
            if not (KNUCKLE_RAD_MIN <= rad <= KNUCKLE_RAD_MAX):
                raise RuntimeError(
                    f"s27a15: right_gripper_rad={rad:.4f} is outside the plausible "
                    f"knuckle band [{KNUCKLE_RAD_MIN}, {KNUCKLE_RAD_MAX}] rad. That is "
                    "almost certainly the wrong UNIT (mm / percent / normalized "
                    "width), and clip(rad, 0, 0.8) would silently turn it into "
                    "'fully closed'. Confirm the driver's units on site."
                )
            open_fraction = float(open_fraction_from_knuckle_rad(rad))
        else:
            open_fraction = float(
                np.asarray(obs["right_gripper_open"], dtype=np.float64).ravel()[0]
            )
            if not np.isfinite(open_fraction):
                raise RuntimeError("s27a15: right_gripper_open is non-finite")
            if not (-0.01 <= open_fraction <= 1.01):
                raise RuntimeError(
                    f"s27a15: right_gripper_open={open_fraction:.4f} is not a "
                    "fraction in [0, 1] — pass raw feedback as "
                    "'right_gripper_rad' instead"
                )
            open_fraction = float(np.clip(open_fraction, 0.0, 1.0))
        return cls(
            left_arm=_vector(obs["left_arm"], 7, "left_arm"),
            right_arm=_vector(obs["right_arm"], 7, "right_arm"),
            right_gripper_open=open_fraction,
            left_wrench=_vector(obs["left_wrench"], 6, "left_wrench"),
            right_wrench=_vector(obs["right_wrench"], 6, "right_wrench"),
            wrench_declared_absent=bool(obs.get("wrench_declared_absent", False)),
        )

    @classmethod
    def from_state(cls, state) -> RigObservation:
        """Inverse of `pack_state`: a packed 27-dim vector -> the rig groups.

        For replaying RECORDED observations through the adapter — the
        parity harness (`tools/rig_parity_check.py`) uses it so the check
        exercises the real packing path rather than shortcutting it. The
        gripper round-trips through radians on purpose: `open -> rad ->
        open` re-runs the exact clip the training builder ran, so a parity
        residual would expose a broken inverse instead of hiding it.
        """
        vec = _vector(state, STATE_DIM, "observation.state")
        return cls(
            left_arm=vec[S_LEFT_ARM].copy(),
            right_arm=vec[S_RIGHT_ARM].copy(),
            right_gripper_open=float(
                open_fraction_from_knuckle_rad(
                    knuckle_rad_from_open_fraction(vec[S_RIGHT_GRIP])
                )
            ),
            left_wrench=vec[S_LEFT_WRENCH].copy(),
            right_wrench=vec[S_RIGHT_WRENCH].copy(),
        )

    def pack(self) -> np.ndarray:
        """-> (27,) float32 `observation.state`, in the trained order."""
        out = np.empty(STATE_DIM, dtype=np.float32)
        out[S_LEFT_ARM] = self.left_arm
        out[S_RIGHT_ARM] = self.right_arm
        out[S_RIGHT_GRIP] = self.right_gripper_open
        out[S_LEFT_WRENCH] = self.left_wrench
        out[S_RIGHT_WRENCH] = self.right_wrench
        return out


def pack_state(rig: RigObservation) -> np.ndarray:
    """Rig observation -> the policy's 27-dim `observation.state`."""
    if not isinstance(rig, RigObservation):
        raise TypeError(
            "s27a15 packs from a RigObservation (named groups), not from a raw "
            "vector: the real rig's joint-state field order is unconfirmed and "
            "an index-based pack would fail silently. Build one with "
            "RigObservation.from_mapping({...})."
        )
    return rig.pack()


# ---------------------------------------------------------------------------
# The rig's command
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RigCommand:
    """One decoded action chunk, ready for the rig's own controllers.

    `right_gripper` is already in `gripper_convention`'s units. The arm
    entries are ABSOLUTE joint targets in radians, exactly as recorded —
    there is no delta mode in this corpus.
    """

    left_arm: np.ndarray  # (T, 7) absolute joint targets, radians
    right_arm: np.ndarray  # (T, 7)
    right_gripper: np.ndarray  # (T,) in `gripper_convention` units
    gripper_convention: str
    dt: float = 1.0 / FPS

    @property
    def horizon(self) -> int:
        return int(self.left_arm.shape[0])


def unpack_action(
    chunk, gripper_convention: str = DEFAULT_GRIPPER_COMMAND, dt: float = 1.0 / FPS
) -> RigCommand:
    """(T, 15) policy chunk -> the rig's command groups.

    The gripper channel passes through the model UNCHANGED on the way in
    (D8: `action[14]` is already an open fraction, 1.0 = open, and the
    corpus builder only clamped it to [0, 1]); the only transform is the
    driver-units conversion on the way out, which is the pluggable half.
    """
    actions = np.asarray(chunk, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None]
    if actions.ndim != 2 or actions.shape[-1] != ACTION_DIM:
        raise RuntimeError(
            f"s27a15 expects a (T, {ACTION_DIM}) chunk, got {actions.shape}. Width "
            "alone never implies a layout (F-45) — if this checkpoint emits "
            "something else it is not an s27a15 checkpoint."
        )
    if not np.all(np.isfinite(actions)):
        raise RuntimeError(
            "s27a15: the policy emitted non-finite actions — refusing to hand "
            "NaN joint targets to a real robot"
        )
    return RigCommand(
        left_arm=actions[:, A_LEFT_ARM].astype(np.float32),
        right_arm=actions[:, A_RIGHT_ARM].astype(np.float32),
        right_gripper=gripper_command(actions[:, A_RIGHT_GRIP], gripper_convention).astype(
            np.float32
        ),
        gripper_convention=gripper_convention,
        dt=float(dt),
    )


# ---------------------------------------------------------------------------
# Cameras: explicit, from the checkpoint's own rename_map
# ---------------------------------------------------------------------------
def resolve_camera_map(
    image_features: list[str],
    rename_map: Mapping[str, str] | None = None,
    override: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Checkpoint image feature -> our camera name, for EVERY declared feature.

    Built from the checkpoint's own `train_config.json` `rename_map`
    (dataset key -> model key), inverted. Never positional: VLA-JEPA takes
    two of the three cameras (`head -> exterior_1_left`,
    `wrist_right -> exterior_2_left`), and a positional walk of
    `[head, wrist_left, wrist_right]` hands its second slot `wrist_left` —
    the same class of silent mis-assignment as F-64, and the "wrong
    everywhere" row of the parity failure table.

    An empty rename_map (ACT, Diffusion, GR00T) means the model keys ARE
    the dataset keys, so the map is the identity over `observation.images.*`.
    """
    forward = dict(rename_map or {})
    inverse: dict[str, str] = {}
    for dataset_key, model_key in forward.items():
        if model_key in inverse:
            raise ValueError(
                f"rename_map sends two dataset keys to {model_key!r}: "
                f"{inverse[model_key]!r} and {dataset_key!r}"
            )
        inverse[model_key] = dataset_key

    mapped: dict[str, str] = {}
    unmapped: list[str] = []
    for feature in image_features:
        dataset_key = inverse.get(feature, feature)
        camera = dataset_key.rsplit(".", 1)[-1]
        if camera in CAMERA_KEYS:
            mapped[feature] = camera
        else:
            unmapped.append(feature)

    for key, camera in (override or {}).items():
        matches = [f for f in image_features if f == key or f.rsplit(".", 1)[-1] == key]
        if not matches:
            raise ValueError(
                f"--camera-map entry {key!r} matches no image feature of this "
                f"checkpoint (features: {image_features})"
            )
        if camera not in CAMERA_KEYS:
            raise ValueError(f"camera {camera!r} for {key!r} not in {list(CAMERA_KEYS)}")
        for f in matches:
            mapped[f] = camera
            if f in unmapped:
                unmapped.remove(f)

    if unmapped:
        raise RuntimeError(
            f"s27a15: no camera for image features {unmapped}. The map is built "
            "from the checkpoint's train_config.json rename_map and refuses "
            "positional assignment (F-64) — pass --camera-map <feature>=<camera> "
            f"with cameras from {list(CAMERA_KEYS)}"
        )
    return mapped


# ---------------------------------------------------------------------------
# The chunk seam: an s27a15 chunk -> the canonical rows the executor eats
# ---------------------------------------------------------------------------
# The Munich checkpoints emit the rig's own 15-dim vector (above) and declare
# `action_space == LAYOUT`. `camelo.control.chunk_executor` consumes the sim
# recorder's 20-dim rows (`C.A_BASE`, `C.A_LEFT_ARM`, `C.A_RIGHT_ARM`,
# `C.A_LEFT_GRIP`, `C.A_RIGHT_GRIP`, `C.A_SPINE`), because that is where the
# per-tick slew clamp lives — the only rate limiter in the whole command
# path. Rather than teach the executor a second layout, the 15-dim chunk is
# widened here, once, on the way in.
#
# Two things this is NOT:
#
#   * **Not a re-slicing of the sim contract.** The rig is a different robot
#     with a different recorder; the widening only works because dims 0-13 of
#     both vectors happen to name the same fourteen arm joints, in the same
#     order, as absolute radians. The base and spine slots get filled with
#     "do nothing" values (0 twist, NaN = hold) because the rig's policy has
#     no base or spine channel at all — not because those channels were
#     dropped from a 20-dim prediction.
#   * **Not a gripper unit conversion.** Dim 14 stays the canonical open
#     fraction (1.0 = OPEN) all the way to
#     `CommandPublisher.publish_grippers`, which applies the real driver's
#     units (`contracts.gripper_width_percent` — the same callable as the
#     `width_percent` convention above). Converting here would apply the
#     transform twice. `unpack_action` / `LeRobotAdapter.rig_command` are the
#     OTHER seam, for a caller that drives the rig's controllers directly
#     (the parity harness); they own `--gripper-command` and this path does
#     not go through them.
#
# Width alone never implies a layout (F-45), so both directions refuse: a
# 15-wide chunk outside `LAYOUT`, and a 20-wide chunk under it. That pair of
# refusals is the point — a 15-dim chunk poured into 20 slots would put the
# right arm's joint 1 into the left gripper and command the base with joint
# targets, all with a valid shape and no error.


def rig_chunk_to_canonical(chunk15, left_gripper_hold: float) -> np.ndarray:
    """(T, 15) `s27a15` chunk -> (T, 20) canonical rows for the executor.

    ``left_gripper_hold`` is the canonical open fraction (1.0 = open) to
    command on the LEFT gripper for the whole chunk: the rig's action vector
    has no left-gripper channel, and the executor emits a value on every
    tick, so somebody has to say what it is. The runner passes the measured
    opening captured before the rollout, i.e. "leave it where it is" — a
    hard-coded 1.0 here would command an open on a left hand that may be
    holding something.

    Base is zeroed (the policy does not drive the base) and the spine is NaN,
    which is the executor's hold value — the same convention
    `contracts.model_to_canonical_actions` uses for the 16-dim path.
    """
    actions = np.asarray(chunk15, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None]
    if actions.ndim != 2 or actions.shape[-1] != ACTION_DIM:
        raise RuntimeError(
            f"rig_chunk_to_canonical expects a (T, {ACTION_DIM}) "
            f"{LAYOUT!r} chunk, got {actions.shape}"
        )
    hold = float(left_gripper_hold)
    if not np.isfinite(hold):
        raise RuntimeError(
            "left_gripper_hold must be a finite open fraction (1.0 = open); "
            f"got {left_gripper_hold!r}. It is commanded on every tick of the "
            "chunk, so a NaN would reach the gripper driver"
        )
    out = np.zeros((actions.shape[0], C.ACTION_DIM), dtype=np.float32)
    out[:, C.A_LEFT_ARM] = actions[:, A_LEFT_ARM]
    out[:, C.A_RIGHT_ARM] = actions[:, A_RIGHT_ARM]
    out[:, C.A_RIGHT_GRIP] = actions[:, A_RIGHT_GRIP]
    out[:, C.A_LEFT_GRIP] = np.float32(hold)
    out[:, C.A_SPINE] = np.nan  # executor holds the spine (no command path)
    return out


def chunk_for_executor(
    actions, action_space: str, left_gripper_hold: float = 1.0
) -> np.ndarray:
    """Whatever the backend returned -> (T, 20), or raise saying why not.

    The single gate every chunk passes through on its way to
    `ChunkExecutor.set_chunk`. ``action_space`` is DECLARED by the caller
    (the adapter's attribute locally, `--action-layout` across the remote
    split) and is never inferred from the width — inferring it is exactly
    the mistake the refusals below exist to catch.
    """
    array = np.asarray(actions, dtype=np.float32)
    if array.ndim == 1:
        array = array[None]
    width = array.shape[-1] if array.ndim == 2 else None
    if action_space == LAYOUT:
        if width == C.ACTION_DIM:
            raise RuntimeError(
                f"action_space={LAYOUT!r} but the chunk is {C.ACTION_DIM}-wide: "
                "that is the SIM contract's vector, and its dims do not line up "
                f"with the rig's ({ACTION_DIM}). Either the server is "
                "serving a sim checkpoint or --action-layout disagrees with it"
            )
        return rig_chunk_to_canonical(array, left_gripper_hold)
    if width == ACTION_DIM:
        raise RuntimeError(
            f"a {ACTION_DIM}-wide chunk arrived under "
            f"action_space={action_space!r}: that is the Munich rig's vector "
            f"({LAYOUT!r}) and its 14 arm joints would land in the base, arm "
            "and gripper slots of the canonical 20 with a perfectly valid "
            "shape. Declare --action-layout s27a15 (and --state-layout s27a15) "
            "or serve a canonical checkpoint"
        )
    if width != C.ACTION_DIM:
        raise RuntimeError(
            f"chunk must be (T, {C.ACTION_DIM}) under "
            f"action_space={action_space!r}, got {array.shape}"
        )
    return array
