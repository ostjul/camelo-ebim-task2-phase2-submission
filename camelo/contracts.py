"""The fr3duo_mobile contract: topics, layouts, and pure assembly helpers.

Constants and helpers are MIRRORED from the benchmark rather than imported,
because the authoritative implementation
(task2_isaacsim/services/recording/record_task2.py) imports rclpy and
lerobot at module scope — unimportable on the x86 policy server and in CI.

Drift protection, in order of authority:
  * ``verify_topics()`` — compares every mirrored topic name against the
    benchmark's live config/topics.yaml; ROS entry points call it at
    startup and fail loudly.
  * ``tests/test_contract_drift.py`` — AST-parses the benchmark source and
    asserts the mirrored constants and layouts still match.

Provenance (ebim-benchmark checkout):
  task2_isaacsim/config/topics.yaml                       (topic names)
  task2_isaacsim/services/recording/record_task2.py       (layouts, helpers)
  task1_isaacsim/scripts/adapters/gello_to_bridge.py      (/bridge command names)
  task2_isaacsim/scripts/isaacsim_fr3duo_teleop_bridge_core.py  (pedal tokens)
  task1_isaacsim/assets/embodiments/fr3duo_mobile/ai_data_contract.yaml
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Joints
# ---------------------------------------------------------------------------
LEFT_JOINTS = [f"left_fr3v2_joint{i}" for i in range(1, 8)]
RIGHT_JOINTS = [f"right_fr3v2_joint{i}" for i in range(1, 8)]
SPINE_JOINT = "franka_spine_vertical_joint"
LEFT_GRIPPER_DRIVER = "left_right_finger_joint"
RIGHT_GRIPPER_DRIVER = "right_right_finger_joint"
#: SIM driver stroke, and the frozen D7 corpus denominator. MIRRORED from the
#: benchmark recorder (`record_task2.py`), drift-alarmed in
#: tests/test_contract_drift.py — do NOT retune it to a real-robot number.
#: `camelo/train/build_real_corpus.py` divides the Munich rig's knuckle angle
#: by this same 0.8 (D7), so it is also part of the TRAINED contract: the
#: checkpoints saw `1 - rad/0.8`, and `s27a15.open_fraction_from_knuckle_rad`
#: must keep decoding live feedback the same way.
GRIPPER_CLOSED_RAD = 0.8

#: The REAL Robotiq 2F-85's closed knuckle angle, in radians — a different
#: robot's constant, deliberately NOT the same symbol as the sim/corpus one
#: above. MEASURED 2026-09-02 (T1(c)) by reading the rig's own driver:
#: `robotiq_description/urdf/2f_85.ros2_control.xacro:18` and
#: `ros2_robotiq_gripper/robotiq_driver/src/hardware_interface.cpp:276,297-299`
#: (`gripper_closed_pos_ = 0.7929`, register = pos/0.7929*227+3 clamped to
#: [0, 255]). 0.0 rad = fully open. This is the scale of the `/…/gripper/
#: joint_states` FEEDBACK; it is NOT what the command topic takes — see
#: `gripper_width_percent` for that.
ROBOTIQ_CLOSED_RAD = 0.7929

# The pose a scene reset teleports the arms to. MIRRORED from the benchmark
# (isaacsim_fr3duo_teleop_bridge_core.py ARM_READY_POSE), drift-alarmed in
# tests/test_contract_drift.py like every other mirrored constant.
#
# We need it because a reset moves the JOINTS but not the position
# controller: whatever target was last published is re-applied within ~5 s
# and the arms leave the ready pose again (F-86). The runner therefore has
# to command this pose explicitly after every reset, exactly as it already
# commands the grippers open.
ARM_READY_POSE = (0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854)

# JointState names on the /bridge gripper command topics (0..1 open fraction).
LEFT_GRIPPER_OPENING = "left_robotiq_opening"
RIGHT_GRIPPER_OPENING = "right_robotiq_opening"

# ---------------------------------------------------------------------------
# Topics (mirror of config/topics.yaml + gello_to_bridge.py)
# ---------------------------------------------------------------------------
CLOCK_TOPIC = "/isaac/clock"
PEDAL_STATE_TOPIC = "/pedal/state"

BRIDGE_LEFT_ARM_CMD = "/bridge/left_joint_commands"
BRIDGE_RIGHT_ARM_CMD = "/bridge/right_joint_commands"
BRIDGE_LEFT_GRIPPER_CMD = "/bridge/left_robotiq_joint_commands"
BRIDGE_RIGHT_GRIPPER_CMD = "/bridge/right_robotiq_joint_commands"

# Browser-controller command topics: a live publisher here during a policy
# run means the browser UI is fighting the policy (preflight check).
BROWSER_CMD_TOPICS = [
    "/isaac/browser/left_joint_commands",
    "/isaac/browser/right_joint_commands",
    "/isaac/browser/left_robotiq_joint_commands",
    "/isaac/browser/right_robotiq_joint_commands",
]

FULL_STATES_TOPIC = "/isaac/joint_states_full"
APPLIED_COMMANDS_TOPIC = "/isaac/applied_joint_commands"
ODOM_TOPIC = "/isaac/odom"
CMD_VEL_APPLIED_TOPIC = "/isaac/cmd_vel_applied"
EE_POSE_TOPICS = {"left": "/isaac/left_ee_pose", "right": "/isaac/right_ee_pose"}
SCENE_RESET_TOPIC = "/isaac/task2/scene_reset"
SCENE_RESET_REQUEST_TOPIC = "/isaac/task2/scene_reset_request"
# Ground-truth object poses, JSON in a std_msgs/String. The grasp gate reads
# the pad LIVE from here rather than a constant: the assembly gets nudged
# mid-episode, and a gate measured against a stale pose drifts with it.
OBJECT_POSES_TOPIC = "/isaac/task2/object_poses"
# The thermal pad is a DEFORMABLE body, and its prim transform on
# OBJECT_POSES_TOPIC never updates -- measured 2026-08-25: frozen at
# (1.750, 1.950, 0.850) across 7183 ticks of a rollout that scored IoU
# 0.87, i.e. one that demonstrably moved the pad. Object poses are still
# correct for the RIGID prims and for the pad's reset jitter (which is
# what the grasp gate needs pre-grasp), but pad MOTION can only be read
# from the mesh vertices here: [sim_time, n_points, x0, y0, z0, x1, ...]
# in world frame, std_msgs/Float32MultiArray.
PAD_POINTS_TOPIC = "/isaac/task2/pad_points"

# FROZEN (2026-08-09, F-68): ONE Task 2 instruction string everywhere —
# dataset captions, fine-tune eval, zero-shot headline runs. Fine-tuned
# checkpoints are locked to their training caption while pretrained VLAs
# can adapt, so holding both at this string is what lets the comparison
# vary weights alone.
# docs/CONTRACTS.md is the doc of record; change it there or not at all.
#
# RE-FROZEN 2026-08-10 (user decision, F-77): colour-grounded phrasing,
# adopted to sit closer to the zero-shot headline runs.
#
# This DELIBERATELY breaks the tie to the recorder default below, so the
# two concepts are now separate constants rather than one:
#   RECORDER_DEFAULT_CAPTION - a mirrored BENCHMARK fact: the caption any
#       recording carries unless --single_task overrides it. Drift-alarmed
#       against the benchmark source like every other mirrored constant.
#   TASK2_INSTRUCTION        - OUR training/eval caption. A choice.
# While they differ, every corpus arrives captioned with the recorder
# string and MUST be re-captioned before training
# (`dataset_tools.modify_tasks`), or the fine-tune learns one string while
# eval sends the other — the F-68 skew, from the recording side.
# `camelo.train.train` warns on exactly that mismatch.
#
# Checkpoints trained before this change are locked to the old caption
# (== RECORDER_DEFAULT_CAPTION) and must be evaluated with --task set to
# it explicitly.
TASK2_INSTRUCTION = "Pick up the blue thermal pad and place it on the red target RAM board."

# Mirrored from the benchmark recorder's --single_task default
# (record_task2.py). Not our choice to make — drift alarm in
# tests/test_contract_drift.py.
RECORDER_DEFAULT_CAPTION = "Pick up the thermal pad and place it on the target RAM board."

EVAL_SERVICE = "/isaac/eval_camera/evaluate"

# One Trigger on EVAL_SERVICE writes a full artifact set, every file named
# eval_camera_<kind>_<ts>.<ext> with ONE shared %Y%m%d_%H%M%S_%f timestamp
# (scripts/evaluation/task2/node.py `_artifact_path`) — fixed-width, so
# lexicographic sort == chronological sort.
EVAL_IOU_PREFIX = "eval_camera_iou_"


def eval_artifact_paths(iou_json) -> list:
    """All files of the evaluation that produced `iou_json`: the JSON plus
    every sibling artifact sharing its timestamp (rgb/depth/semantic/bbox)."""
    ts = iou_json.stem.removeprefix(EVAL_IOU_PREFIX)
    return sorted(iou_json.parent.glob(f"eval_camera_*_{ts}.*"))

# Camera keys double as dataset video feature suffixes
# (observation.images.<key>) — order matters for VLA adapters.
CAMERAS = {
    "head": {"image_topic": "/isaac/head_camera/image_raw", "shape": (720, 1280, 3)},
    "wrist_left": {"image_topic": "/isaac/left_wrist_camera/image_raw", "shape": (480, 848, 3)},
    "wrist_right": {"image_topic": "/isaac/right_wrist_camera/image_raw", "shape": (480, 848, 3)},
}
CAMERA_KEYS = list(CAMERAS)
# Published by ROS2CameraInfoHelper. Live K overrides the yaml 90°×60° which
# the USD Camera prim does not actually use (Isaac default film → ~60° HFOV).
HEAD_CAMERA_INFO_TOPIC = "/isaac/head_camera/camera_info"

# ---------------------------------------------------------------------------
# Topic maps: sim (benchmark /isaac/* + /bridge/*) vs real (TMR station)
# ---------------------------------------------------------------------------
# The module-level names above stay the sim contract — verify_topics() and
# the AST drift tests pin them against the benchmark checkout. Real-robot
# names are the hardware-verified list in record_bag.bash (LABS manifest +
# 2026-08-23/30 topic-name corrections). Select with topics_for("sim"|"real")
# or CAMELO_WORLD; never mutate the sim constants.
WORLD_SIM = "sim"
WORLD_REAL = "real"
WORLDS = (WORLD_SIM, WORLD_REAL)

# What the real FR3 duo's MEASURED JointState messages carry. The topic
# namespace is the side (/left/..., /right/...), so those names arrive
# unprefixed and `canonicalize_joint_name` rewrites them to the canonical
# `left_fr3v2_jointN` / `right_fr3v2_jointN`.
# MEASURED names only. The COMMAND side is NOT symmetric: the companion's
# joint_impedance_controller rejects anything but the prefixed names, so
# `_real_topics` publishes LEFT_JOINTS / RIGHT_JOINTS (2026-09-01, below).
REAL_ARM_JOINTS = tuple(f"fr3_joint{i}" for i in range(1, 8))


@dataclass(frozen=True)
class TopicMap:
    """What the ROS collector / publisher subscribe and publish.

    ``joint_state_topics`` is ``((topic, side), ...)`` where side is
    ``'left'`` / ``'right'`` / ``'spine'`` / ``''`` (already-canonical names).
    """

    world: str
    clock: str | None
    joint_state_topics: tuple[tuple[str, str], ...]
    applied_command_topics: tuple[tuple[str, str], ...]
    odom: str | None
    cmd_vel_applied: str | None
    ee_poses: tuple[tuple[str, str], ...]
    cameras: dict
    scene_reset: str | None
    scene_reset_request: str | None
    object_poses: str | None
    pad_points: str | None
    # External wrench in the stiffness frame, per arm. REAL ONLY: 12 of the
    # 27 dims the Munich policies trained on are wrench, and s27a15 raises on
    # a missing group rather than substituting zeros (F-63), so these are
    # subscribed on the real map and are None in sim.
    left_wrench: str | None
    right_wrench: str | None
    left_arm_cmd: str
    right_arm_cmd: str
    left_gripper_cmd: str
    right_gripper_cmd: str
    gripper_cmd_kind: str  # "joint_state" | "width_percent"
    left_arm_joint_names: tuple[str, ...]
    right_arm_joint_names: tuple[str, ...]
    base_cmd: str | None
    base_cmd_kind: str  # "pedal_token" | "twist"
    spine_cmd: str | None
    browser_cmd_topics: tuple[str, ...]
    verify_against_benchmark: bool
    expected_obs_streams: tuple[str, ...]
    # Real base only: the SwerveDriveController takes geometry_msgs/TwistStamped
    # and ages every message against ITS clock (REAL_BASE_CMD_TIMEOUT_S), so an
    # unstamped Twist reads as ~1.8e9 s old and is silently replaced by zeros.
    base_cmd_stamped: bool = False


def _sim_topics() -> TopicMap:
    return TopicMap(
        world=WORLD_SIM,
        clock=CLOCK_TOPIC,
        joint_state_topics=((FULL_STATES_TOPIC, ""),),
        applied_command_topics=((APPLIED_COMMANDS_TOPIC, ""),),
        odom=ODOM_TOPIC,
        cmd_vel_applied=CMD_VEL_APPLIED_TOPIC,
        ee_poses=tuple(EE_POSE_TOPICS.items()),
        cameras={key: dict(cam) for key, cam in CAMERAS.items()},
        scene_reset=SCENE_RESET_TOPIC,
        scene_reset_request=SCENE_RESET_REQUEST_TOPIC,
        object_poses=OBJECT_POSES_TOPIC,
        pad_points=PAD_POINTS_TOPIC,
        left_wrench=None,  # the sim publishes no external wrench
        right_wrench=None,
        left_arm_cmd=BRIDGE_LEFT_ARM_CMD,
        right_arm_cmd=BRIDGE_RIGHT_ARM_CMD,
        left_gripper_cmd=BRIDGE_LEFT_GRIPPER_CMD,
        right_gripper_cmd=BRIDGE_RIGHT_GRIPPER_CMD,
        gripper_cmd_kind="joint_state",
        left_arm_joint_names=tuple(LEFT_JOINTS),
        right_arm_joint_names=tuple(RIGHT_JOINTS),
        base_cmd=PEDAL_STATE_TOPIC,
        base_cmd_kind="pedal_token",
        spine_cmd=None,
        browser_cmd_topics=tuple(BROWSER_CMD_TOPICS),
        verify_against_benchmark=True,
        expected_obs_streams=("clock", "joint_states_full", "odom", "ee_left", "ee_right"),
    )


# Real base (TMR swerve, Munich rig) — from the site's own driver,
# docs/realdata/site_scripts/companion/base_nudge.py (2026-08-23 → 09-03):
#   * commands are geometry_msgs/TwistStamped, stamped from the node clock;
#     SwerveDriveController substitutes zeros for anything older than
#     cmd_vel_timeout (0.5 s) — `ros2 topic pub` (stamp 0) never moves it;
#   * the controller clamps to 0.1 m/s / 0.1 rad/s and ramps at 0.1 m/s²;
#   * publish at 20 Hz and send zeros on the way out, or the base coasts
#     until the watchdog fires (~5 cm).
# The message type follows the site bridge; confirm on site with
# `ros2 topic type /swerve_drive_controller/cmd_vel` before the first run.
REAL_BASE_MAX_LINEAR_MPS = 0.1
REAL_BASE_MAX_ANGULAR_RADPS = 0.1
REAL_BASE_ACCEL_MPS2 = 0.1
REAL_BASE_CMD_TIMEOUT_S = 0.5
REAL_BASE_CMD_HZ = 20.0


def _real_topics() -> TopicMap:
    # Names from record_bag.bash. Cameras keep the sim keys (head / wrist_*)
    # so adapters and --cameras do not change; only the ROS names differ.
    return TopicMap(
        world=WORLD_REAL,
        clock=None,  # no /isaac/clock; collector drives t_sim from ROS time
        joint_state_topics=(
            ("/left/franka_robot_state_broadcaster/measured_joint_states", "left"),
            ("/left/gripper/joint_states", "left"),
            ("/right/franka_robot_state_broadcaster/measured_joint_states", "right"),
            ("/right/gripper/joint_states", "right"),
            ("/spine/joint_states", "spine"),
        ),
        applied_command_topics=(),  # build_action falls back to measured
        odom="/swerve_drive_controller/odom",
        cmd_vel_applied="/swerve_drive_controller/cmd_vel_out",
        ee_poses=(),
        # MEASURED 2026-09-01 on the rig. The wrists are RealSense D405 at
        # 640x480, NOT the sim's 848x480 — the two worlds share camera KEYS,
        # not resolutions. `shape` is documentation plus the sim-only drift
        # check in verify_topics(); nothing on the ingest path reads it
        # (camera_workers decodes from the message's own height/width via
        # image_msg_to_array), so a 480x640 frame flows through unchanged.
        cameras={
            "head": {  # ZED, 1280x720
                "image_topic": "/head_camera/zed_node/rgb/color/rect/image",
                "shape": (720, 1280, 3),
            },
            "wrist_left": {  # RealSense D405, 640x480
                "image_topic": "/wrist_camera_left/camera/color/image_rect_raw",
                "shape": (480, 640, 3),
            },
            "wrist_right": {  # RealSense D405, 640x480
                "image_topic": "/wrist_camera_right/camera/color/image_rect_raw",
                "shape": (480, 640, 3),
            },
        },
        scene_reset=None,
        scene_reset_request=None,
        object_poses=None,
        pad_points=None,
        # MEASURED 2026-09-01: geometry_msgs/WrenchStamped, BEST_EFFORT with
        # the rest of the broadcaster's output.
        left_wrench="/left/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame",
        right_wrench="/right/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame",
        left_arm_cmd="/left/gello/joint_states",
        right_arm_cmd="/right/gello/joint_states",
        left_gripper_cmd="/left/gripper/gripper_client/target_gripper_width_percent",
        right_gripper_cmd="/right/gripper/gripper_client/target_gripper_width_percent",
        # The topic's name is a misnomer: the wire value is a 0..1 OPEN
        # FRACTION, not a percent (MEASURED 2026-09-02, see
        # gripper_width_percent). The kind keeps the topic's own word so the
        # two cannot be confused with each other.
        gripper_cmd_kind="width_percent",
        # MEASURED 2026-09-01 on the rig: the companion's
        # joint_impedance_controller validates every incoming GELLO
        # JointState against its own joint list
        # (`validateGelloJointState_`), which is the PREFIXED
        # `left_fr3v2_jointN` / `right_fr3v2_jointN` — the same names the
        # sim contract uses. Sending the unprefixed measured names
        # (REAL_ARM_JOINTS) makes the controller reject the message, and
        # `switch_controller` then answers ok=False — with no error at all
        # on the publishing side.
        left_arm_joint_names=tuple(LEFT_JOINTS),
        right_arm_joint_names=tuple(RIGHT_JOINTS),
        base_cmd="/swerve_drive_controller/cmd_vel",
        base_cmd_kind="twist",
        base_cmd_stamped=True,
        spine_cmd="/spine/target_height",
        browser_cmd_topics=(),
        verify_against_benchmark=False,
        expected_obs_streams=("clock", "joint_states_full", "odom"),
    )


def default_world() -> str:
    return os.environ.get("CAMELO_WORLD", WORLD_SIM).strip().lower() or WORLD_SIM


def topics_for(world: str | None = None) -> TopicMap:
    """Return the topic map for ``world`` (``sim`` / ``real``).

    ``None`` reads ``CAMELO_WORLD`` (default ``sim``).
    """
    resolved = (world or default_world()).strip().lower()
    if resolved == WORLD_REAL:
        return _real_topics()
    if resolved == WORLD_SIM:
        return _sim_topics()
    raise ValueError(f"world must be one of {WORLDS}, got {world!r}")

# ---------------------------------------------------------------------------
# Canonical recorder layouts (LeRobot dataset action/state columns)
# ---------------------------------------------------------------------------
ACTION_DIM = 20
STATE_DIM = 37

A_BASE = slice(0, 3)  # vx, vy, wz (body frame)
A_LEFT_ARM = slice(3, 10)
A_RIGHT_ARM = slice(10, 17)
A_LEFT_GRIP = 17
A_RIGHT_GRIP = 18
A_SPINE = 19

S_LEFT_EE = slice(0, 7)  # x y z qx qy qz qw, world
S_RIGHT_EE = slice(7, 14)
S_LEFT_ARM = slice(14, 21)
S_RIGHT_ARM = slice(21, 28)
S_SPINE = 28
S_LEFT_GRIP = 29
S_RIGHT_GRIP = 30
S_BASE_ODOM = slice(31, 34)  # x, y, yaw (world)
S_BASE_VEL = slice(34, 37)  # vx, vy, wz (body frame)

# ---------------------------------------------------------------------------
# Model-facing contract (ai_data_contract.yaml, embodiment franka_fr3_duo):
# 16-dim [left_arm(7), left_gripper, right_arm(7), right_gripper],
# ABSOLUTE joint radians, 16-step action horizon. Base and spine are NOT
# part of the model contract.
# ---------------------------------------------------------------------------
MODEL_ACTION_DIM = 16
MODEL_HORIZON = 16
M_LEFT_ARM = slice(0, 7)
M_LEFT_GRIP = 7
M_RIGHT_ARM = slice(8, 15)
M_RIGHT_GRIP = 15


def model_state_from_state(state: np.ndarray) -> np.ndarray:
    """37-dim recorder state -> 16-dim model proprio (contract key order)."""
    out = np.empty(MODEL_ACTION_DIM, dtype=np.float32)
    out[M_LEFT_ARM] = state[S_LEFT_ARM]
    out[M_LEFT_GRIP] = state[S_LEFT_GRIP]
    out[M_RIGHT_ARM] = state[S_RIGHT_ARM]
    out[M_RIGHT_GRIP] = state[S_RIGHT_GRIP]
    return out


def model_to_canonical_actions(model_actions: np.ndarray) -> np.ndarray:
    """[H, 16] model actions -> [H, 20] canonical (base 0, spine NaN=hold)."""
    model_actions = np.asarray(model_actions, dtype=np.float32)
    if model_actions.ndim == 1:
        model_actions = model_actions[None]
    out = np.zeros((model_actions.shape[0], ACTION_DIM), dtype=np.float32)
    out[:, A_LEFT_ARM] = model_actions[:, M_LEFT_ARM]
    out[:, A_LEFT_GRIP] = model_actions[:, M_LEFT_GRIP]
    out[:, A_RIGHT_ARM] = model_actions[:, M_RIGHT_ARM]
    out[:, A_RIGHT_GRIP] = model_actions[:, M_RIGHT_GRIP]
    out[:, A_SPINE] = np.nan  # executor holds the spine (no ROS command path)
    return out


def canonical_to_model_actions(actions: np.ndarray) -> np.ndarray:
    """[H, 20] canonical -> [H, 16] model order (drops base + spine)."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None]
    out = np.empty((actions.shape[0], MODEL_ACTION_DIM), dtype=np.float32)
    out[:, M_LEFT_ARM] = actions[:, A_LEFT_ARM]
    out[:, M_LEFT_GRIP] = actions[:, A_LEFT_GRIP]
    out[:, M_RIGHT_ARM] = actions[:, A_RIGHT_ARM]
    out[:, M_RIGHT_GRIP] = actions[:, A_RIGHT_GRIP]
    return out


# ---------------------------------------------------------------------------
# Task 2 approach zone (world frame) — FROZEN 2026-08-10
# ---------------------------------------------------------------------------
# Spawn: TASK_ROBOT_POSES["task2"]. Goal: desk-front seat (live odom target).
TASK2_SPAWN_XY_YAW = (4.4, 2.6, math.radians(-90.0))
TASK2_APPROACH_XY_YAW = (2.10, 3.05, math.radians(-90.0))

# Spine SOP (m): pin at launch; approach waits for measured ≥ MIN (F-57 droop).
SPINE_SOP_M = 0.50
SPINE_SOP_MEASURED_MIN_M = 0.45

# Manipulation-ready joint poses (rad) — MEASURED 2026-08-10 from the
# reference demos, medians over ext_hermanprawiro_task2_fixpos_v1 (F-58);
# re-derive with tools/ready_pose_from_dataset.py. Both put the flange ~0.19 m
# over the table, where the demos hand over to manipulation. Do NOT substitute
# the benchmark's ARM_READY_POSE: that is the spine-DOWN start pose and sits
# ~0.5 m too high once the spine is at the SOP.
TASK2_ARM_READY_RIGHT = (
    -0.4209,
    -0.2092,
    -0.6482,
    -2.6307,
    -0.1910,
    2.4484,
    -0.0781,
)
TASK2_ARM_READY_LEFT = (
    -0.2066,
    -0.3323,
    1.3072,
    -2.5817,
    0.4977,
    2.4231,
    1.4583,
)
TASK2_GRIPPER_OPEN = 1.0

# ---------------------------------------------------------------------------
# Pedal tokens (mirror of pedal_base_twist in the task2 bridge core)
# ---------------------------------------------------------------------------
PEDAL_TOKENS = ("FWD", "BACK", "A", "B", "A+C", "B+C", "NONE")

# Helper-stack defaults (task2 launcher): --pedal-linear-speed / --pedal-angular-speed.
PEDAL_LINEAR_SPEED = 0.5  # m/s
PEDAL_ANGULAR_SPEED = 1.2  # rad/s


def token_to_twist(
    token: str,
    linear_speed: float = PEDAL_LINEAR_SPEED,
    angular_speed: float = PEDAL_ANGULAR_SPEED,
) -> tuple[float, float, float]:
    token = token.strip().upper().replace(" ", "")
    if token == "FWD":
        return linear_speed, 0.0, 0.0
    if token == "BACK":
        return -linear_speed, 0.0, 0.0
    if token == "A":
        return 0.0, linear_speed, 0.0
    if token == "B":
        return 0.0, -linear_speed, 0.0
    if token in ("A+C", "C+A"):
        return 0.0, 0.0, angular_speed
    if token in ("B+C", "C+B"):
        return 0.0, 0.0, -angular_speed
    return 0.0, 0.0, 0.0


# ---------------------------------------------------------------------------
# Pure helpers (mirror of record_task2.py lines ~186-495)
# ---------------------------------------------------------------------------
def canonicalize_joint_name(name: str, side: str = "") -> str:
    """Map a hardware JointState name onto the canonical sim contract name.

    ``side`` is the topic's arm (``left`` / ``right``), ``spine``, or ``''``
    when the name is already globally unique (sim ``/isaac/joint_states_full``).
    Real FR3 topics live under ``/left`` and ``/right`` namespaces and carry
    unprefixed ``fr3_jointN`` / ``finger_joint`` names; without this rewrite
    they collide when merged and never match ``LEFT_JOINTS``.
    """
    raw = (name or "").strip()
    if not raw:
        return raw
    side = (side or "").strip().lower()

    if side == "spine" or raw == SPINE_JOINT or "spine" in raw.lower():
        return SPINE_JOINT

    lowered = raw.lower()
    if any(token in lowered for token in ("finger", "gripper", "robotiq", "knuckle")):
        if side == "right" or raw.startswith("right"):
            return RIGHT_GRIPPER_DRIVER
        if side == "left" or raw.startswith("left"):
            return LEFT_GRIPPER_DRIVER
        return raw

    body = raw
    inferred = side
    for prefix in ("left_", "right_"):
        if body.startswith(prefix):
            inferred = prefix[:-1]
            body = body[len(prefix) :]
            break
    if not inferred:
        return raw

    number = None
    for needle in ("fr3v2_1_joint", "fr3v2_joint", "fr3_joint", "panda_joint", "joint"):
        rest = body[len(needle) :] if body.startswith(needle) else None
        if rest is not None and rest.isdigit():
            number = int(rest)
            break
    if number is None or not 1 <= number <= 7:
        if not raw.startswith(inferred + "_"):
            return f"{inferred}_{raw}"
        return raw
    return f"{inferred}_fr3v2_joint{number}"


def _candidate_joint_names(name):
    yield name
    if "fr3v2_joint" in name:
        yield name.replace("fr3v2_joint", "fr3v2_1_joint")
        yield name.replace("fr3v2_joint", "fr3_joint")
    if name == "left_right_finger_joint":
        yield "left_fr3v2_finger_joint1"
        yield "left_finger_joint"
        yield "finger_joint"
    if name == "right_right_finger_joint":
        yield "right_fr3v2_finger_joint1"
        yield "right_finger_joint"
        yield "finger_joint"


def resolve_joint(joint_map: dict, name: str, default=math.nan) -> float:
    for candidate in _candidate_joint_names(name):
        value = joint_map.get(candidate)
        if value is not None and math.isfinite(value):
            return float(value)
    return default


#: Suffix that distinguishes a GRIPPER `joint_states` topic from an arm one
#: inside `TopicMap.joint_state_topics` (both are `JointState`, both carry a
#: side). `ObsCollector` counts the two separately so a probe can tell
#: "gripper feedback is arriving" from "the arm broadcaster is arriving" —
#: sharing one counter hid exactly that during T1(c).
GRIPPER_JOINT_STATE_SUFFIX = "/gripper/joint_states"


def is_gripper_joint_state_topic(topic: str) -> bool:
    return str(topic).endswith(GRIPPER_JOINT_STATE_SUFFIX)


def gripper_open_fraction(driver_position_rad: float) -> float:
    if not math.isfinite(driver_position_rad):
        return math.nan
    return float(np.clip(1.0 - driver_position_rad / GRIPPER_CLOSED_RAD, 0.0, 1.0))


def opening_to_driver_rad(open_fraction: float) -> float:
    """Open fraction -> driver radians (0.0 = open, 0.8 = closed).

    This is the DIRECT `/isaac/*_robotiq_joint_commands` convention (the
    inverse of gripper_open_fraction). It is NOT the `/bridge` wire value —
    see gripper_wire_value for that.
    """
    return float(np.clip(1.0 - open_fraction, 0.0, 1.0) * GRIPPER_CLOSED_RAD)


def gripper_wire_value(open_fraction: float, invert: bool) -> float:
    """Canonical open fraction (1 = open) -> `/bridge/*` gripper wire value.

    The republisher maps the wire value to the driver joint; with the task2
    default `REPUBLISHER_GRIPPER_INVERT=true` its mapping is
    `driver = wire * stroke`, so the wire value is a CLOSE fraction and the
    canonical open fraction must be inverted before publishing. Measured on
    the DGX (docs/DGX_FINDINGS.md F-18): publishing 1.0 raw drove the driver
    to 0.80 rad = closed.
    """
    value = float(np.clip(open_fraction, 0.0, 1.0))
    return 1.0 - value if invert else value


def gripper_width_percent(open_fraction, invert: bool = False):
    """Canonical open fraction (1 = open) -> real gripper wire value (0..1).

    **The topic name lies, and this function keeps its name to mirror the
    lie.** `/left|right/gripper/gripper_client/target_gripper_width_percent`
    (`std_msgs/Float32`) takes a **0..1 OPEN FRACTION**, not a 0..100
    percent: 1.0 = open, 0.0 = closed. MEASURED on the rig 2026-09-02
    (T1(c)); the evidence chain, all read-only off the companion Jetson:

      * `franka_gripper_manager/robotiq_gripper_client.py:27-35` — the only
        subscriber. It dedups (`if abs(target - last_width) < 0.02: return`),
        then sends `goal.command.position = 1 - gripper_position` with
        `max_effort = 1.0`. So the wire value is read as an open fraction and
        converted to a CLOSE position.
      * `GripperCommand.position` is knuckle radians, 0 open to 0.7929 closed
        (`ROBOTIQ_CLOSED_RAD`; `2f_85.ros2_control.xacro:18`,
        `robotiq_driver/src/hardware_interface.cpp:276,297-299`). Nothing in
        `gripper_action_controller` clamps it.
      * The station's own `gello_publisher.py` publishes
        `process_gripper_position()` — documented "percentage (0-1)", clamped
        to [0, 1] — at 30 Hz, and the corpus column
        `…target_gripper_width_percent_value` spans 0.2025..1.0
        (docs/realdata/01_DATASET_ANALYSIS.md:491). No row is ever above 1.

    What the old 0..100 scaling did: 1.0 open -> wire 100.0, 0.30 -> 30.0,
    both of which become `1 - 100 = -99 rad` / `1 - 30 = -29 rad`, clamped to
    register 0 = FULLY OPEN. The gripper therefore sat open at both ends of
    the T1(c) sine while every value looked plausible on the wire — and each
    20 Hz tick beat the 0.02 dedup, so all 8011 `GripperCommand_Result(
    position=0.0, reached_goal=False)` lines in the companion log are
    preempt-cancelled default results, not feedback.

    Invert only when CAMELO_GRIPPER_INVERT says the driver is the other way
    around — do not reuse the sim republisher default.

    Scalar in -> ``float`` out (a ROS message field wants a Python float);
    array in -> ``ndarray`` out, because this is also the ``width_percent``
    entry of ``camelo.policy.adapters.s27a15.GRIPPER_COMMAND_CONVENTIONS``,
    which maps whole chunks. One definition on purpose: the policy-side
    decode and the ROS-side publish of "how open" must not be able to drift
    apart.
    """
    value = np.clip(np.asarray(open_fraction, dtype=np.float64), 0.0, 1.0)
    if invert:
        value = 1.0 - value
    return float(value) if value.ndim == 0 else value


def _parse_env_file(path) -> dict:
    values = {}
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip("\"'")
    except OSError:
        pass
    return values


def gripper_command_invert(world: str | None = None) -> bool:
    """Resolve whether gripper commands must be inverted.

    Order: CAMELO_GRIPPER_INVERT env var if set; else False on the real
    robot (the wire value is an OPEN fraction, 1.0 = open); else the benchmark's
    task2_isaacsim/.env (falling back to .env.example)
    REPUBLISHER_GRIPPER_INVERT — i.e. follow whatever the running helper
    stack was configured with; else True (the shipped task2 default).
    verify_gripper_polarity() in camelo/ros/command_publisher.py checks this
    guess against the live sim before dataset-critical runs.
    """
    override = os.environ.get("CAMELO_GRIPPER_INVERT")
    if override is not None:
        return override.strip().lower() in ("1", "true", "yes")
    if (world or default_world()) == WORLD_REAL:
        return False
    from camelo.benchmark import find_benchmark_root

    root = find_benchmark_root(required=False)
    if root is not None:
        for name in (".env", ".env.example"):
            values = _parse_env_file(root / "task2_isaacsim" / name)
            if "REPUBLISHER_GRIPPER_INVERT" in values:
                return values["REPUBLISHER_GRIPPER_INVERT"].lower() in ("1", "true", "yes")
    return True


def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def image_msg_to_array(msg) -> np.ndarray:
    """sensor_msgs/Image (rgb8/bgr8/bgra8/rgba8/mono8) -> HxWx3 uint8 RGB.

    Stride-safe. ZED colour is often bgra8; treating it as 3-channel silently
    scrambles the row. 4-channel encodings drop alpha; mono is stacked to RGB.
    """
    enc = (getattr(msg, "encoding", None) or "rgb8").lower()
    if enc in ("bgra8", "rgba8", "8uc4"):
        channels = 4
    elif enc in ("mono8", "8uc1"):
        channels = 1
    else:
        channels = 3
    data = np.frombuffer(msg.data, dtype=np.uint8)
    array = data.reshape(msg.height, msg.step)[:, : msg.width * channels]
    array = array.reshape(msg.height, msg.width, channels)
    if channels == 4:
        array = array[:, :, :3]
        if enc.startswith("bgr"):
            array = array[:, :, ::-1]
    elif channels == 1:
        array = np.repeat(array, 3, axis=2)
    elif enc in ("bgr8", "bgra8"):
        array = array[:, :, ::-1]
    return np.ascontiguousarray(array)


def build_action(snap: dict) -> np.ndarray:
    """Canonical 20-dim action from a recorder-shaped snapshot dict."""
    action = np.full(ACTION_DIM, np.nan, dtype=np.float32)
    action[A_BASE] = snap["cmd_vel"]
    applied = snap["applied_commands"]
    measured = snap["joint_states"]
    for i, name in enumerate(LEFT_JOINTS + RIGHT_JOINTS):
        value = resolve_joint(applied, name)
        if not math.isfinite(value):
            value = resolve_joint(measured, name)  # hold current position
        action[3 + i] = value
    action[A_LEFT_GRIP] = gripper_open_fraction(resolve_joint(applied, LEFT_GRIPPER_DRIVER, 0.0))
    action[A_RIGHT_GRIP] = gripper_open_fraction(resolve_joint(applied, RIGHT_GRIPPER_DRIVER, 0.0))
    spine_target = resolve_joint(applied, SPINE_JOINT)
    if not math.isfinite(spine_target):
        spine_target = resolve_joint(measured, SPINE_JOINT, 0.0)
    action[A_SPINE] = spine_target
    return action


def build_state(snap: dict) -> np.ndarray:
    """Canonical 37-dim state from a recorder-shaped snapshot dict."""
    state = np.full(STATE_DIM, np.nan, dtype=np.float32)
    for offset, side in ((0, "left"), (7, "right")):
        pose = snap["ee_poses"].get(side)
        if pose is not None:
            state[offset : offset + 7] = pose
    measured = snap["joint_states"]
    for i, name in enumerate(LEFT_JOINTS + RIGHT_JOINTS):
        state[14 + i] = resolve_joint(measured, name)
    state[S_SPINE] = resolve_joint(measured, SPINE_JOINT, 0.0)
    state[S_LEFT_GRIP] = gripper_open_fraction(resolve_joint(measured, LEFT_GRIPPER_DRIVER, 0.0))
    state[S_RIGHT_GRIP] = gripper_open_fraction(resolve_joint(measured, RIGHT_GRIPPER_DRIVER, 0.0))
    odom = snap["odom"]
    if odom is not None:
        x, y, _, qx, qy, qz, qw, vx, vy, _, wz = odom
        state[S_BASE_ODOM] = (x, y, quat_to_yaw(qx, qy, qz, qw))
        state[S_BASE_VEL] = (vx, vy, wz)
    return state


# ---------------------------------------------------------------------------
# Startup verification against the live benchmark checkout
# ---------------------------------------------------------------------------
def verify_topics() -> None:
    """Fail loudly if the mirrored topic names drifted from topics.yaml.

    Call from every ROS entry point. Raises BenchmarkNotFound when there is
    no checkout (ROS-side runs always have one) and AssertionError on drift.
    """
    from camelo.benchmark import load_topics

    live = load_topics()
    expect = {
        CLOCK_TOPIC: live["clock"],
        PEDAL_STATE_TOPIC: live["teleop"]["pedal_state"],
        FULL_STATES_TOPIC: live["recording"]["joint_states_full"],
        APPLIED_COMMANDS_TOPIC: live["recording"]["applied_joint_commands"],
        ODOM_TOPIC: live["recording"]["odom"],
        CMD_VEL_APPLIED_TOPIC: live["recording"]["cmd_vel_applied"],
        EE_POSE_TOPICS["left"]: live["recording"]["ee_pose"]["left"],
        EE_POSE_TOPICS["right"]: live["recording"]["ee_pose"]["right"],
        SCENE_RESET_TOPIC: live["ground_truth"]["scene_reset"],
        SCENE_RESET_REQUEST_TOPIC: live["ground_truth"]["scene_reset_request"],
        OBJECT_POSES_TOPIC: live["ground_truth"]["object_poses"],
        PAD_POINTS_TOPIC: live["ground_truth"]["pad_points"],
    }
    sub = live["cameras"]["subtopics"]["image"]
    info_sub = live["cameras"]["subtopics"]["camera_info"]
    for key, cam in CAMERAS.items():
        entry = live["cameras"]["robot"][key]
        expect[cam["image_topic"]] = f"{entry['namespace']}/{sub}"
        expect_shape = tuple(entry["shape"])
        if expect_shape != cam["shape"]:
            raise AssertionError(
                f"camera {key} shape drifted: mirrored {cam['shape']}, "
                f"topics.yaml says {expect_shape}"
            )
    expect[HEAD_CAMERA_INFO_TOPIC] = (
        f"{live['cameras']['robot']['head']['namespace']}/{info_sub}"
    )
    drifted = {mirrored: live for mirrored, live in expect.items() if mirrored != live}
    if drifted:
        lines = "\n".join(
            f"  mirrored {mirrored!r} != live {live!r}" for mirrored, live in drifted.items()
        )
        raise AssertionError(
            "camelo.contracts drifted from the benchmark topic contract:\n" + lines
        )
