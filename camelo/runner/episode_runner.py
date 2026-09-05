"""Episode loop: reset -> settle -> rollout -> score.

Timing model: the rclpy node spins in a background thread (so the pedal
repeat timer and subscriptions stay alive during blocking policy
inference); the control loop runs in the caller's thread, paced by wall
clock but stamped with sim time from /isaac/clock. Rollout length is
measured in SIM seconds, so a deformable scene running below real time
gets the same task time as a fast one.

Scoring calls the benchmark's eval service (std_srvs/Trigger on
/isaac/eval_camera/evaluate; the eval stack from
scripts/evaluation/task2/run.sh must be up) and reads the newest
eval_camera_iou_*.json it wrote.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import shutil
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from camelo import contracts as C
from camelo.control.approach import (
    TASK2_APPROACH_WAYPOINTS,
    ApproachController,
    body_frame_errors,
)
from camelo.control.async_inference import AsyncInference
from camelo.control.chunk_executor import ChunkExecutor
from camelo.control.grasp_gate import GraspObserver
from camelo.control.gripper_latch import RELEASE_SUSTAIN_S, GripperLatch
from camelo.control.image_age import (
    ImageAgeMeter,
    StaleImageError,
    format_ages,
    stale_image_report,
)
from camelo.control.pad_metrics import PadTracker
from camelo.control.state_age import (
    StaleStateError,
    max_state_age,
    stale_state_report,
)
from camelo.control.traj_log import (
    DEFAULT_PAD_CSV,
    DEFAULT_TCP_CSV,
    DEFAULT_TRAJ_CSV,
    BaseTrajectoryLog,
    PadTrajectoryLog,
    TcpTrajectoryLog,
)
from camelo.policy.adapters.s27a15 import CANONICAL_SPACE, chunk_for_executor
from camelo.policy.backend import PolicyBackend
from camelo.runner.chunk_dump import (
    OBS_STATE_AT_ARRIVAL,
    OBS_STATE_AT_INFERENCE,
    ChunkDumpRecorder,
)
from camelo.runner.joint_log import ROLLOUT_PHASE, JointTrackingLog
from camelo.runner.real_arms import START_POSE_TOL_RAD, verify_start_pose
from camelo.runner.recenter import (
    ARM_SETTLE_TOL_RAD,
    REACHED,
    STUCK,
    ArmSettleMonitor,
    pose_error,
    report_unreached,
    wound_joints,
)

if TYPE_CHECKING:
    # Annotation-only: importing these at module scope pulls in ROS, which
    # is not installed on every box that needs to import this module (this
    # Mac included) — AGENTS.md hard rule 3. `from __future__ import
    # annotations` above keeps every annotation below unevaluated, so the
    # names only need to exist for a type checker, never at runtime.
    from camelo.ros.command_publisher import CommandPublisher
    from camelo.ros.obs_collector import ObsCollector

log = logging.getLogger(__name__)

# What t0 an arriving synchronous chunk is stamped with (docs/realdata/16
# §T6 plan-then-execute). `observation` is every result on record: `t0` is
# the sim time of the observation `backend.infer` was handed, so on a slow
# (real) inference the wall index at the NEXT tick is already
# `infer_s / dt` steps in — the arm held still while the rows it never
# played were burned. `arrival` re-bases `t0` to the sim time of the tick
# the chunk is actually installed on, so playback starts at row 0 exactly
# where the arm is, then advances one row per tick from there. See
# `run_rollout`'s `chunk_time_base` parameter.
CHUNK_TIME_BASES = ("observation", "arrival")

# Pedal tokens are fixed-speed and latched for exactly one control tick, so the
# loop period *looks* like a motion quantiser — but the sim applies the twist
# once per render step and the swerve ramps in after steering aligns (F-93).
# Short pulses land ~1 mm / ~0.01°; continuous hold is a different regime.
# Keep approach at 50 Hz so a short engage/coast/brake trim can fire 50 ms bursts.
APPROACH_RATE_HZ = 50.0

# The Path.home() fallback is only right where the eval stack writes to THIS
# machine's home — in the camelo-ros container home is /root while the
# artifacts land on the host, so compose.yaml mounts the host dir and sets
# EVAL_OUTPUT_DIR explicitly (F-42).
EVAL_OUTPUT_DIR = Path(
    os.environ.get(
        "EVAL_OUTPUT_DIR",
        Path(os.environ.get("ISAAC_DOCKER_ROOT", Path.home() / "docker" / "ebim-challenge"))
        / "eval-task2"
        / "evaluate",
    )
)


# The thermal pad's key on OBJECT_POSES_TOPIC. Confirmed against the
# benchmark's own scene capture (task2_isaacsim/scripts/recording/
# scene_capture.py:46,74). The names on that topic are USD prim names
# enumerated at runtime under /World/Scene/task_objects, so an unconfirmed
# key does not error — `object_xyz` just returns None forever and the
# metric silently reads "never observed".
PAD_OBJECT_NAME = "thermalpad"

# The pad's destination. CONFIRMED 2026-08-25 against a live payload on
# OBJECT_POSES_TOPIC from the running room scene, which publishes exactly
# ['board_0', 'board_1', 'board_2', 'board_target', 'thermalpad',
# 'thermalpad_base'] -- board_target at (2.150, 1.950, 0.750), 400 mm from
# the pad at (1.750, 1.950, 0.850). Read off the wire, not guessed: a key
# that MISSES leaves the projection None, but a key that hits the WRONG
# prim yields a plausible wrong millimetre number.
TARGET_OBJECT_NAME = "board_target"


def _grasp_observer(grasp_gate=None) -> GraspObserver:
    """The protocol's §0 metric-1 detector, on the SAME envelope as the gate.

    When a gate is driving the rollout the instrument must measure against
    the numbers the actuator is firing on. This repo currently holds two
    envelopes — `outputs/probes/grasp_pose_envelope.json` says 40.0 mm while
    `grasp_gate.DEFAULT_MAX_DIST_M` is 45 mm, and `from_envelope()` prefers
    the JSON — so an observer built on the module defaults would score a
    gated rollout at a threshold that rollout never used, and a with/without
    -gate comparison (protocol C1) would be taken at two different bars
    without saying so. Copying the gate's own attributes makes that
    impossible whichever value wins, and `GraspObserver.stats()` echoes what
    it used, so the summary states the bar it scored against.

    With no gate there is nothing to copy and the module defaults stand.
    """
    if grasp_gate is None:
        return GraspObserver()
    return GraspObserver(
        max_dist_m=grasp_gate.max_dist_m,
        max_angle_deg=grasp_gate.max_angle_deg,
        reference_quat_xyzw=grasp_gate.reference_quat_xyzw,
    )


def wait_for_obs(collector: ObsCollector, timeout_s: float = 30.0, require_images: bool = True):
    t0 = time.monotonic()
    last_log = t0
    while time.monotonic() - t0 < timeout_s:
        obs = collector.get_obs(require_images=require_images)
        if obs is not None:
            return obs
        now = time.monotonic()
        if now - last_log >= 5.0:
            status = collector.debug_status()
            log.info(
                "waiting for obs (%.0fs): missing_images=%s n_joints=%d "
                "workers=%s joint_names=%s rates=%s",
                now - t0,
                status["missing_images"],
                status["n_joints"],
                status["workers"],
                status["joint_names"],
                status["rates"],
            )
            last_log = now
        time.sleep(0.1)
    rates = collector.rates()
    status = collector.debug_status()
    world = getattr(getattr(collector, "topics", None), "world", "sim")
    if world == "real":
        hint = (
            "are the station nodes up (topics in record_bag.bash)? "
            f"missing_images={status['missing_images']} "
            f"camera_topics={status['camera_topics']} "
            f"workers={status['workers']} "
            f"joint_names={status['joint_names']} rates={rates}"
        )
    else:
        hint = (
            "is the sim up with --record (cameras + recording topics)? "
            f"rates: {rates}"
        )
    log.error("no complete observation within %.0fs — %s", timeout_s, hint)
    raise TimeoutError(
        f"no complete observation within {timeout_s:.0f}s — {hint}"
    )


def _arm_progress_s(approach: ApproachController, state) -> str:
    """Per-arm placement progress: distance to this leg, to the final ready
    pose, the joint holding it up, and the flange height — the last one is
    what shows a gripper dropping instead of reaching out over the table."""
    wp, final = approach.arm_waypoint, approach.arm_path[-1]
    parts = []
    for tag, joints, ee, leg_target, ready_target in (
        ("L", C.S_LEFT_ARM, C.S_LEFT_EE, wp.left, final.left),
        ("R", C.S_RIGHT_ARM, C.S_RIGHT_EE, wp.right, final.right),
    ):
        measured = np.asarray(state[joints], dtype=np.float64)
        leg = np.abs(measured - np.asarray(leg_target, dtype=np.float64))
        ready = np.abs(measured - np.asarray(ready_target, dtype=np.float64))
        parts.append(
            f"{tag}(leg {math.degrees(leg.max()):.1f}° j{int(np.argmax(leg)) + 1}, "
            f"ready {math.degrees(ready.max()):.1f}°, z={float(state[ee][2]):.3f})"
        )
    return f"arms={wp.name}/{approach.arm_leg_ticks}t " + " ".join(parts)


def _log_start_pose(approach: ApproachController, state) -> None:
    """Log the handover pose, then a summary of every max deviation vs its limit."""
    e = approach.start_pose_errors(state)
    x, y, yaw = (float(v) for v in state[C.S_BASE_ODOM])
    ready = approach.arm_path[-1]
    tol_left, tol_right = approach._arm_tolerances(ready)
    tol_xy, tol_yaw = approach._handover_tols()
    xy_lim_mm = tol_xy * 1000.0
    ok = approach.stats.get("start_pose_ok", True)
    (log.info if ok else log.warning)(
        "start position %s: base=(%.3f, %.3f, %+.4f rad) %.0f mm / %+.4f rad from goal "
        "(%.0f mm of drift since navigate) | spine=%.3f m (cmd %.2f) | "
        "arms at ready within L %.1f° R %.1f° | flange z L %.3f R %.3f m",
        "reached" if ok else "off (continuing)",
        x, y, yaw,
        e["xy_err_m"] * 1000.0,
        e["yaw_err_rad"],
        e["drift_m"] * 1000.0,
        e["spine_m"],
        approach.spine_target_m,
        math.degrees(e["arm_err_left_rad"]),
        math.degrees(e["arm_err_right_rad"]),
        e["flange_z_left_m"],
        e["flange_z_right_m"],
    )
    left_deg = [math.degrees(v) for v in e["arm_joint_err_left_rad"]]
    right_deg = [math.degrees(v) for v in e["arm_joint_err_right_rad"]]
    left_tol_deg = [math.degrees(v) for v in tol_left]
    right_tol_deg = [math.degrees(v) for v in tol_right]
    log.info(
        "final position max deviations:\n"
        "  base xy      %+6.1f mm  (lim ±%.0f mm)  "
        "dx=%+.1f dy=%+.1f  body front=%+.1f left=%+.1f mm\n"
        "  base yaw     %+.4f rad (lim ±%.4f rad)\n"
        "  nav drift    %6.1f mm  (since navigate settled)\n"
        "  spine        %+6.1f mm vs cmd %.0f mm  (measured %.3f m, gate ≥%.2f m)\n"
        "  arm L max    %6.1f deg at j%d  (tol %.1f°)  per-joint=%s\n"
        "  arm R max    %6.1f deg at j%d  (tol %.1f°)  per-joint=%s\n"
        "  gripper L/R  %+.3f / %+.3f  (target open=%.2f)\n"
        "  flange z L/R %.3f / %.3f m",
        e["xy_err_m"] * 1000.0,
        xy_lim_mm,
        e["x_err_m"] * 1000.0,
        e["y_err_m"] * 1000.0,
        e["body_front_m"] * 1000.0,
        e["body_left_m"] * 1000.0,
        e["yaw_err_rad"],
        tol_yaw,
        e["drift_m"] * 1000.0,
        e["spine_err_m"] * 1000.0,
        approach.spine_target_m * 1000.0,
        e["spine_m"],
        approach.spine_measured_min_m,
        math.degrees(e["arm_err_left_rad"]),
        e["arm_err_left_joint"],
        left_tol_deg[e["arm_err_left_joint"] - 1] if e["arm_err_left_joint"] > 0 else math.nan,
        "[" + ", ".join(f"{v:.1f}" for v in left_deg) + "]",
        math.degrees(e["arm_err_right_rad"]),
        e["arm_err_right_joint"],
        right_tol_deg[e["arm_err_right_joint"] - 1] if e["arm_err_right_joint"] > 0 else math.nan,
        "[" + ", ".join(f"{v:.1f}" for v in right_deg) + "]",
        e["gripper_err_left"],
        e["gripper_err_right"],
        approach.gripper_open,
        e["flange_z_left_m"],
        e["flange_z_right_m"],
    )


def _log_approach_tick(approach: ApproachController, state, command, t_sim: float) -> None:
    x, y, yaw = (float(v) for v in state[C.S_BASE_ODOM])
    spine = float(state[C.S_SPINE])
    pose_hud = getattr(approach, "pose_hud", None)
    if callable(pose_hud):
        pose_s = pose_hud(state)
    else:
        pose_s = "pose=(odom n/a)"
        if all(math.isfinite(v) for v in (x, y, yaw)):
            pose_s = f"pose=({x:.3f}, {y:.3f}, {yaw:+.4f} rad)"
            if approach.stage in ("navigate", "finegrained_start_position"):
                gx, gy, gyaw = approach.goal_x, approach.goal_y, approach.goal_yaw
                front, left, eyaw = body_frame_errors(x, y, yaw, gx, gy, gyaw)
                if (
                    approach.stage == "finegrained_start_position"
                    and approach.fine_trim is not None
                ):
                    tol_xy = approach.fine_trim.tol_xy_m
                    tol_yaw = approach.fine_trim.tol_yaw_rad
                else:
                    tol_xy, tol_yaw = approach.pos_tol_m, approach.yaw_tol_rad
                pose_s += (
                    f" wp=({gx:.3f}, {gy:.3f}, {gyaw:+.4f} rad) "
                    f"err=({(gx - x) * 1000:+.0f}mm, {(gy - y) * 1000:+.0f}mm, "
                    f"{eyaw:+.4f} rad) "
                    f"body=(front {front:+.3f}, left {left:+.3f}) "
                    f"tol=({tol_xy * 1000:.0f}mm, {tol_yaw:.4f} rad)"
                )
    if approach.stage == "place_arms":
        pose_s = f"{pose_s} {_arm_progress_s(approach, state)}"
    vx, vy, wz = command.base_twist
    log.info(
        "approach next=%s t=%.1f %s spine=%.3f (cmd %.2f) action=%s duty=%.2f "
        "twist=(%.2f, %.2f, %.2f)",
        approach.target_label(),
        t_sim,
        pose_s,
        spine if math.isfinite(spine) else float("nan"),
        approach.spine_target_m,
        command.base_token,
        getattr(approach.drive, "last_duty", 0.0),
        vx,
        vy,
        wz,
    )


def _desk_goal(approach: ApproachController | None) -> tuple[float, float, float]:
    """Final approach waypoint — the pose errors in the trajectory CSV are vs this."""
    wp = approach.waypoints[-1] if approach is not None else TASK2_APPROACH_WAYPOINTS[-1]
    return wp.x, wp.y, wp.yaw


def _record_base(
    traj_log: BaseTrajectoryLog | None, t_sim: float, state, stage: str = ""
) -> None:
    if traj_log is None:
        return
    x, y, yaw = (float(v) for v in state[C.S_BASE_ODOM])
    traj_log.record(t_sim, x, y, yaw, stage=stage)


def _record_tcp(
    tcp_log: TcpTrajectoryLog | None, kin, t_sim: float, state
) -> None:
    if tcp_log is None or kin is None:
        return
    from camelo.control.kinematics import RIGHT_TCP

    q = np.asarray(state[C.S_RIGHT_ARM], dtype=np.float64)
    base = np.asarray(state[C.S_BASE_ODOM], dtype=np.float64)
    spine = float(state[C.S_SPINE])
    pos, _ = kin.fk(q, base, spine, frame=RIGHT_TCP)
    tcp_log.record(
        t_sim,
        float(pos[0]),
        float(pos[1]),
        float(pos[2]),
        float(base[0]),
        float(base[1]),
        float(base[2]),
        spine,
        q,
    )


def _record_pad(
    pad_log: PadTrajectoryLog | None,
    t_sim: float,
    pad_xyz,
    obj_xyz,
    target_xyz,
) -> None:
    if pad_log is None:
        return
    pad_log.record(t_sim, pad_xyz, obj_xyz, target_xyz)


def run_approach(
    collector: ObsCollector,
    publisher: CommandPublisher,
    approach: ApproachController,
    timeout_s: float = 60.0,
    rate_hz: float = APPROACH_RATE_HZ,
    on_tick=None,
    log_period_s: float = 0.5,
    arm_log_period_s: float = 0.1,
    traj_log: BaseTrajectoryLog | None = None,
) -> dict:
    """Spine settle + navigate + place arms; does not count against ``rollout_s``.

    Spine is stage 0 so a post-reset pin-ramp (eval) finishes under this
    sim-time budget before the base moves — same settled geometry a
    no-reset ``run-policy`` session already has.

    Perception approach (``needs_images``) waits on the head camera and
    passes frames into ``step``; pose approach still skips images.
    """
    need_images = bool(getattr(approach, "needs_images", False))
    obs = wait_for_obs(collector, require_images=need_images)
    if getattr(approach, "needs_base_wire", False):
        # base_nudge.py waits for a subscriber before its first command:
        # DDS matching takes a moment and unmatched commands are simply lost.
        subscribed = getattr(publisher, "base_subscribed", lambda: None)()
        if subscribed is False:
            raise RuntimeError(
                "nothing subscribes to the base command topic — is start_base "
                "running on the companion (swerve_drive_controller active)?"
            )
        if subscribed is None:
            log.warning("cannot tell whether the base controller listens; driving anyway")
    approach.reset()
    t_start, wall_t0, tick = obs.t_sim, time.monotonic(), 1.0 / rate_hz
    log.info(
        "approach: %s  wps=%s  spine>=%.2f",
        "→".join(approach.stages),
        ",".join(f"{wp.name}" for wp in approach.waypoints),
        approach.spine_measured_min_m,
    )

    owned = traj_log is None
    if owned:
        gx, gy, gyaw = _desk_goal(approach)
        traj_log = BaseTrajectoryLog(DEFAULT_TRAJ_CSV, gx, gy, gyaw)

    last_log_t, last_token, last_target = -float("inf"), None, None
    try:
        while not approach.done:
            loop_t0 = time.monotonic()
            obs = collector.get_obs(require_images=need_images)
            if obs is None:
                time.sleep(tick)
                continue
            if obs.t_sim - t_start >= timeout_s:
                _record_base(traj_log, obs.t_sim, obs.state, approach.stage)
                raise TimeoutError(
                    f"approach timed out after {timeout_s:.0f} sim-s in "
                    f"{approach.target_label()}: {approach.stage_hint(obs.state)} "
                    f"(stats={approach.stats})"
                )
            if obs.t_sim < t_start - 1.0:
                raise RuntimeError("sim clock rebased during approach")

            odom_n = collector.counts.get("odom")
            stage = approach.stage
            if need_images:
                command = approach.step(
                    obs.state,
                    odom_n=odom_n,
                    t_sim=obs.t_sim,
                    images=obs.images,
                    odom_at_head=obs.odom_at_image.get("head"),
                    head_k=obs.head_k,
                    head_d=obs.head_d,
                    head_t_sim=obs.image_t_sim.get("head"),
                )
            else:
                command = approach.step(obs.state, odom_n=odom_n, t_sim=obs.t_sim)
            publisher.publish(command)
            _record_base(traj_log, obs.t_sim, obs.state, stage)
            target, token = approach.target_label(), command.base_token
            # The arm legs are over in ~1 sim-s; at the navigation period they
            # would be three log lines and invisible. Fine trim is sparse.
            if approach.stage == "place_arms":
                period = arm_log_period_s
            elif approach.stage == "finegrained_start_position":
                # Log every 0.2 sim-s: denser than navigate (0.5) so single-step
                # pulses show up, sparser than place_arms (0.1).
                period = 0.2
            else:
                period = log_period_s
            if target != last_target or token != last_token or (obs.t_sim - last_log_t) >= period:
                _log_approach_tick(approach, obs.state, command, obs.t_sim - t_start)
                last_log_t, last_token, last_target = obs.t_sim, token, target
            if on_tick is not None:
                on_tick()
            elapsed = time.monotonic() - loop_t0
            if elapsed < tick:
                time.sleep(tick - elapsed)
    finally:
        # Always stop the base at approach exit (timeout, error, or done).
        publisher.safe_stop()
        if owned:
            traj_log.close()

    _log_start_pose(approach, obs.state)
    stats = dict(approach.stats)
    stats.update(
        sim_seconds=float(obs.t_sim - t_start),
        wall_seconds=float(time.monotonic() - wall_t0),
        phase=approach.stage,
    )
    log.info(
        "approach done: arms→ready L %.1f°→%.1f° R %.1f°→%.1f° in %d ticks (%.1f sim-s)",
        math.degrees(stats.get("arm_err_start_left_rad") or math.nan),
        math.degrees(stats.get("arm_err_left_rad") or math.nan),
        math.degrees(stats.get("arm_err_start_right_rad") or math.nan),
        math.degrees(stats.get("arm_err_right_rad") or math.nan),
        stats.get("arm_ticks") or 0,
        stats["sim_seconds"],
    )
    return stats


def format_splice(executor: ChunkExecutor) -> str:
    """The per-arrival chunk hand-over numbers, for the `policy chunk` line.

    Empty until there is something to say: an arrival is only measurable
    once a command has been issued to splice against. `jump` is the
    pre-clamp discontinuity the arm is asked to swallow on this arrival
    (docs/realdata/16 U-31), so it stays comparable across policies —
    under `index` it pins to `--max-delta` on every arrival, which is
    exactly the wiggle R-68 measured. `lead` is |command - measured| over
    the 14 arm joints at the last executor tick (U-32): the distance the
    splice is reasoning across, bounded by `--leash-rad` when that is on.
    It is reported on its own, because it exists from the first tick and
    not only on a measurable arrival.
    """
    shift = executor.last_splice_shift
    jump = executor.last_arrival_jump_rad
    lead = executor.last_cmd_lead_rad
    parts = []
    if shift is not None and jump is not None:
        parts.append(f" splice={shift:+.1f} jump={jump:.3f}")
    if lead is not None:
        parts.append(f" lead={lead:.3f}")
    return "".join(parts)


def run_rollout(
    collector: ObsCollector,
    publisher: CommandPublisher,
    executor: ChunkExecutor,
    backend: PolicyBackend,
    rollout_s: float,
    rate_hz: float = 20.0,
    on_tick=None,
    approach: ApproachController | None = None,
    approach_timeout_s: float = 30.0,
    approach_only: bool = False,
    traj_csv: Path | None = None,
    tcp_csv: Path | None = None,
    pad_csv: Path | None = None,
    joint_csv: Path | None = None,
    chunk_dump: Path | None = None,
    chunk_dump_args: dict | None = None,
    grasp_gate=None,
    grasp_observer: GraspObserver | None = None,
    gripper_latch: GripperLatch | None = None,
    pad_tracker: PadTracker | None = None,
    target_object: str | None = TARGET_OBJECT_NAME,
    action_space: str = CANONICAL_SPACE,
    left_gripper_hold: float = 1.0,
    max_image_age_s: float | None = None,
    max_state_age_s: float | None = None,
    async_inference: bool = False,
    chunk_time_base: str = "observation",
) -> dict:
    """Drive the policy for ``rollout_s`` sim seconds; returns loop stats.

    Optional ``approach`` runs first (own sim-time budget) before inference.
    ``on_tick`` runs after each publish (eval MP4 sampler).
    The base pose is written every tick to ``traj_csv`` (default
    ``outputs/approach/base_trajectory.csv``). Measured right TCP is
    written every *rollout* tick to ``tcp_csv`` (sibling
    ``tcp_trajectory.csv``). The thermal pad's per-tick position (live
    mesh centroid, frozen object-pose reading, and target pose) is
    written every *rollout* tick to ``pad_csv`` (sibling
    ``pad_trajectory.csv``).

    The protocol's sim intermediate metrics (GRASP_EXPERIMENT_PROTOCOL.md §0)
    are measured on EVERY rollout, gate or no gate, and land in the returned
    stats. ``grasp_observer`` / ``pad_tracker`` let a caller own the pair so
    it can reset them per episode (F-86); a caller that does not gets a fresh
    pair built here.

    ``action_space`` is the DECLARED layout of the chunks the backend
    returns: ``canonical20`` (every sim adapter) or ``s27a15`` (the Munich
    rig's 15-dim vector, widened onto the executor's 20 slots by
    `camelo.policy.adapters.s27a15`). It is declared, never inferred from the
    width — and both mismatches raise. ``left_gripper_hold`` is the canonical
    open fraction the rig's absent left-gripper channel is filled with.

    ``gripper_latch`` (default None = off, byte-identical passthrough) is
    the rig-side `camelo.control.gripper_latch.GripperLatch` lever: once the
    policy's RIGHT gripper command (after any ``grasp_gate`` override) drops
    below its ``close_below``, the published command is held at
    ``min(policy, 0.20)`` until a sustained re-open releases it. It sits
    AFTER the executor and any grasp-gate override and BEFORE the publish,
    so `chunk_dump` still records the policy's raw value and the observer
    below still sees the value that actually left on the wire. Left gripper
    is never touched. When the latch is built with a ``near_pose`` (CLI
    ``--gripper-latch-near``), each tick's measured ``C.S_RIGHT_ARM`` is
    passed through so the ENGAGE transition can also be gated on joint-space
    proximity to that pose.

    ``joint_csv`` (default None = off, and nothing is written) turns on the
    T5 joint trace: one row per rollout control tick of commanded vs
    measured arm joints and grippers, plus the per-tick clamp / inference /
    publish-gap telemetry, followed by a printed per-joint max-error and
    ``clamped_pct`` block (`camelo.runner.joint_log`). It is the only
    measured-joint trace that survives the real rig, where the base and
    spine are NaN and ``TcpTrajectoryLog`` therefore writes no rows at all.

    ``chunk_dump`` (default None = off, byte-identical either way) turns on
    the `camelo.runner.chunk_dump` diagnostic: one record per chunk that
    reaches `ChunkExecutor.set_chunk` (synchronous and
    `camelo.control.async_inference` alike) -- the chunk number, its t0,
    the sim time and wall/spliced index at arrival, the jump/lead the
    `policy chunk #N` log line already prints, the observation state it
    was computed from (or, under async inference where that observation is
    not returned by the worker, the measured state at arrival -- named
    accordingly), the executor's last commanded arm vector, and the full
    (H, ACTION_DIM) chunk. Written as one ``.npz`` at the end of the run
    (also on the guarded exception path), never raising into the control
    loop. ``chunk_dump_args`` is folded into the npz's JSON sidecar
    unchanged -- `scripts/run_policy.py` passes ``vars(args)``.

    ``max_image_age_s`` (default None = OFF, which is every sim path) is the
    U-23 stale-image guard: at each INFERENCE — not each control tick; the
    replan cadence is ~3.75 Hz against a 20 Hz loop — every camera's newest
    frame must have arrived within that many WALL seconds, or the rollout
    raises ``StaleImageError``. It exists because nothing else notices a
    camera dying: ``ObsCollector`` never clears a cached frame, so the policy
    would keep consuming the last one indefinitely with no fault line.
    `scripts/run_policy.py` sets it from ``--max-image-age-s`` (0.5 s on
    ``--world real``) and its ``finally`` deactivates the arms on the way out.

    ``max_state_age_s`` (default None = OFF, which is every sim path) is the
    U-27 stale-JOINT-STATE guard, the same shape for the other half of the
    observation — but sampled on EVERY control tick, not at the inference
    seam: joint states arrive at ~1 kHz per arm on the rig, the frozen pose
    they leave behind corrupts every published command rather than every
    fourth one, and a 20 Hz check against a 0.2 s bar is free. Both driven
    and HELD arms are watched: a held arm is still published to and still
    feeds the policy's state vector. Raises ``StaleStateError`` out of the
    rollout, exactly as above. Its measurement also fills ``--joint-csv``'s
    ``state_age_s`` column, so a post-hoc reader can see the tick the stream
    died on (docs/realdata/16 R-64).
    ``async_inference`` (default False = the synchronous path, byte for
    byte what it always was) runs ``backend.infer`` on its own thread
    (`camelo.control.async_inference`), so the control loop never blocks on
    a round trip. `scripts/run_policy.py` turns it on for ``--backend
    remote`` on ``--world real``, where the measured 0.45 s RTT was costing
    the 20 Hz loop 9 of every 20 ticks (docs/realdata/16 U-29). The loop
    then: polls for a reply every tick (an exception from the worker is
    re-raised HERE, so it still leaves through the guarded finally);
    installs an arriving chunk at its own ``t0``, so the executor starts it
    at the index the elapsed round trip already consumed rather than
    replaying stale steps; issues the next request as soon as
    ``needs_replan`` fires against that ``t0`` (never against the arrival
    time) and only while nothing is in flight, because a second request
    would be dropped rather than queued (``dropped_requests``, 0 on a
    healthy run); and if a chunk runs out before its successor lands,
    HOLDS the last commanded target (``starved_ticks``) rather than
    extrapolating off the end.

    ``target_object`` is the pad's destination on OBJECT_POSES_TOPIC, needed
    for `pad_disp_toward_target_mm`. It defaults to the CONFIRMED
    ``board_target`` (see the constant above — read off a live payload, not
    guessed). Pass None to disable the projection; the metric then reports
    None rather than a direction.

    ``chunk_time_base`` (default ``"observation"`` -- today's behaviour,
    byte-identical everywhere) is what sim time an arriving chunk's ``t0``
    is stamped with, in `CHUNK_TIME_BASES`. ``"observation"`` uses the
    chunk's own ``t0`` (the sim time of the observation ``backend.infer``
    was handed) exactly as it always has. ``"arrival"`` is the
    plan-then-execute mode for the SYNCHRONOUS real-robot loop: the arm
    holds during inference (keep-alive covers it), so charging the elapsed
    ``infer_s`` against the observation's ``t0`` makes the wall index
    already ``infer_s / dt`` steps in in-loop of the executor's next
    ``step`` and rows ``0..that`` are skipped for an arm that never moved
    (the ~0.3-0.45 s remote round trip skips rows 0..6-9). ``"arrival"``
    re-samples the collector right after ``backend.infer`` returns and
    stamps the chunk's ``t0`` (and ``set_chunk``'s ``t_sim_now``) with THAT
    sim time instead, so the wall index is 0 at installation and every
    executor-derived quantity that counts from it --
    ``needs_replan``/``exhausted``/the clamp/the leash/the chunk dump's
    ``wall_idx``/``spliced_idx`` -- counts from the arrival, not the stale
    observation. It also applies under `async_inference`, where it simply
    re-bases the installed chunk to the arrival tick (``t0 = obs.t_sim``,
    the same value already used for ``t_sim_now`` there) -- harmless, but
    it collapses ``chunk_arrival_index_max`` to ~0 and is not what the flag
    is for; it exists asynchronously only so the parameter behaves
    uniformly rather than raising. It is meant for the SYNCHRONOUS path.
    """
    if chunk_time_base not in CHUNK_TIME_BASES:
        raise ValueError(
            f"chunk_time_base must be one of {CHUNK_TIME_BASES}, got {chunk_time_base!r}"
        )
    gx, gy, gyaw = _desk_goal(approach)
    traj_log = BaseTrajectoryLog(traj_csv or DEFAULT_TRAJ_CSV, gx, gy, gyaw)
    if tcp_csv is None:
        tcp_csv = Path(traj_log.path).with_name("tcp_trajectory.csv")
        if traj_csv is None:
            tcp_csv = DEFAULT_TCP_CSV
    tcp_log = TcpTrajectoryLog(tcp_csv)
    if pad_csv is None:
        pad_csv = Path(traj_log.path).with_name("pad_trajectory.csv")
        if traj_csv is None:
            pad_csv = DEFAULT_PAD_CSV
    pad_log = PadTrajectoryLog(pad_csv)
    from camelo.control.kinematics import MobileFR3Kinematics

    approach_stats = None
    joint_log: JointTrackingLog | None = None
    # Bound before the try for the same reason joint_log is: the finally
    # has to be able to close it however the try left.
    inference: AsyncInference | None = None
    # Bound before the try for the same reason: the finally writes the npz
    # however the try left, including on a raised guard exception.
    chunk_dump_recorder: ChunkDumpRecorder | None = None
    try:
        kin = MobileFR3Kinematics()
        # The instruments are NOT the ablation and are never gated on it. The
        # gate actuates, the observer measures, and protocol C1 compares a
        # gated arm against an ungated one — a detector that only existed in
        # the gated arm could not make that comparison.
        observer = grasp_observer if grasp_observer is not None else _grasp_observer(grasp_gate)
        tracker = pad_tracker if pad_tracker is not None else PadTracker()
        if approach is not None:
            # Approach keeps its own (faster) rate regardless of the policy rollout rate.
            approach_stats = run_approach(
                collector, publisher, approach,
                timeout_s=approach_timeout_s, on_tick=on_tick,
                traj_log=traj_log,
            )
            executor.reset()
            if approach_only:
                # --approach-only: the base is parked, the arms were never
                # touched; zeros on the base wire and out — no hold phase,
                # no activation, no policy.
                publisher.safe_stop()
                log.info("approach complete — --approach-only, exiting before the policy")
                return {"approach": approach_stats, "approach_only": True, "ticks": 0}
            log.info("approach complete — starting manipulation policy")

        obs = wait_for_obs(collector)
        # Per-ROLLOUT publish telemetry (T6): zero the counters here so the
        # numbers below describe this rollout rather than everything the
        # process has published since it started — the hold phase can
        # legitimately keep-alive for minutes while an operator activates.
        reset_publish_stats = getattr(publisher, "reset_publish_stats", None)
        if reset_publish_stats is not None:
            reset_publish_stats()
        t_start, wall_t0, tick = obs.t_sim, time.monotonic(), 1.0 / rate_hz
        infer_times: list[float] = []
        # Async inference (U-29): the worker is started HERE, after the
        # approach, so its lifetime is exactly this rollout's — and closed
        # in the finally below, before run_policy.py's own finally closes
        # the backend the worker is holding.
        inference = AsyncInference(backend) if async_inference else None
        arrival_index_max = 0.0  # chunk steps the round trip had already eaten
        # N2b step 1: image age in SIM seconds at the moment the policy
        # consumes it, plus frames per sim-second over the rollout window.
        age_meter = ImageAgeMeter()
        frame_marker = collector.frame_marker()
        # Opened HERE, after the approach and after the first complete
        # observation: `t` in the trace is monotonic seconds since this
        # line, so it is seconds since the rollout began — not since the
        # process started, and not since a wait_for_obs of unknown length.
        if joint_csv is not None:
            joint_log = JointTrackingLog(joint_csv)
        if chunk_dump is not None:
            # Named for which case applies to THIS run: async inference
            # never returns the observation it submitted (only the chunk),
            # so the field that would otherwise be "computed from" is
            # honestly the arrival-tick measurement instead.
            chunk_dump_recorder = ChunkDumpRecorder(
                OBS_STATE_AT_ARRIVAL if inference is not None else OBS_STATE_AT_INFERENCE
            )
        tick_n = 0
        publish_stats_fn = getattr(publisher, "publish_stats", None)
        # U-27. Read at all only when something wants it — the guard, or the
        # CSV column that records it — so a sim rollout (guard OFF, no joint
        # CSV) touches the collector exactly as it did before this landed.
        state_wall_fn = getattr(collector, "last_state_wall", None)
        watch_state = state_wall_fn is not None and (
            max_state_age_s is not None or joint_log is not None
        )

        while True:
            loop_t0 = time.monotonic()
            tick_infer_s: float | None = None
            tick_state_age_s: float | None = None
            obs = collector.get_obs()
            if obs is None:
                time.sleep(tick)
                continue
            if obs.t_sim - t_start >= rollout_s:
                break
            if obs.t_sim < t_start - 1.0:  # scene reset mid-rollout
                log.warning("sim clock rebased mid-rollout; stopping early")
                break

            if watch_state:
                # U-27: EVERY tick, unlike the image guard's inference-seam
                # sampling. `obs.state`'s arm block was just read out of a
                # cache the collector never clears, so this is the only line
                # between a dead publisher and 30 s of published commands
                # computed from one frozen pose (docs/realdata/16 R-64).
                now_wall = time.monotonic()
                last_state_wall = state_wall_fn()
                tick_state_age_s = max_state_age(last_state_wall, now_wall)
                fault = stale_state_report(
                    last_state_wall, now_wall, max_state_age_s
                )
                if fault is not None:
                    # Logged as well as raised, for the same reason as the
                    # stale-image fault above: the exception's own path out
                    # (run_policy's finally -> deactivate -> exit non-zero)
                    # is correct but silent in the ROS log.
                    log.error("%s", fault)
                    raise StaleStateError(fault)

            if inference is not None:
                # ASYNC (U-29). Poll first, then ask: a chunk installed on
                # this tick is the one `needs_replan` must be judged
                # against, and its t0 is the sim time of the obs that
                # produced it — so the next request goes out replan_steps
                # after THAT, never after the arrival.
                chunk = inference.poll()  # re-raises the worker's exception
                if chunk is not None:
                    infer_s = inference.last_infer_s
                    infer_times.append(infer_s)
                    tick_infer_s = infer_s
                    # The executor indexes by (t_sim - t0)/dt, so it starts
                    # this chunk at the step the round trip already ate
                    # rather than replaying the elapsed part.
                    #
                    # `chunk_time_base == "arrival"` re-bases `t0` to
                    # `obs.t_sim` — the sim time of THIS tick, already what
                    # `t_sim_now` uses below — which collapses the wall
                    # index to 0 on every arrival. That is a legitimate
                    # re-basing (the flag is documented uniformly across
                    # both branches) but not what it is FOR: async already
                    # starts the chunk at the round-trip-consumed index, and
                    # zeroing that out loses `chunk_arrival_index_max`'s
                    # signal for free. The flag exists for the SYNCHRONOUS
                    # loop; see run_rollout's docstring.
                    chunk_t0 = obs.t_sim if chunk_time_base == "arrival" else chunk.t0
                    arrival_index = (obs.t_sim - chunk_t0) / chunk.dt
                    arrival_index_max = max(arrival_index_max, arrival_index)
                    actions_for_executor = chunk_for_executor(
                        chunk.actions, action_space, left_gripper_hold
                    )
                    executor.set_chunk(
                        chunk_t0,
                        actions_for_executor,
                        chunk.dt,
                        t_sim_now=obs.t_sim,
                    )
                    if chunk_dump_recorder is not None:
                        chunk_dump_recorder.record(
                            chunk_num=len(infer_times),
                            t0=chunk_t0,
                            t_sim_arrival=obs.t_sim,
                            wall_idx=arrival_index,
                            spliced_idx=arrival_index + (executor.last_splice_shift or 0.0),
                            jump=executor.last_arrival_jump_rad,
                            lead=executor.last_cmd_lead_rad,
                            obs_state=obs.state,  # measured AT ARRIVAL, not at inference
                            last_cmd_arms=executor.last_commanded_arms,
                            actions=actions_for_executor,
                        )
                    log.info(
                        "policy chunk #%d t0=%.2f arrival_idx=%.1f/%d infer=%.3fs dt=%.3f%s",
                        len(infer_times),
                        chunk_t0 - t_start,
                        arrival_index,
                        len(chunk.actions),
                        infer_s,
                        chunk.dt,
                        format_splice(executor),
                    )
                # Ask only when nothing is out: a second request would be
                # dropped rather than queued (a queued one comes back
                # stamped at an obs the robot has already driven past), and
                # asking every tick would spend an observation and a
                # stale-image check on a refusal. In the measured steady
                # state the trigger (8 steps) has long since fired when the
                # reply lands (9-10), so the next request goes out on the
                # arrival tick and the cadence is the round trip itself.
                if executor.needs_replan(obs.t_sim) and not inference.in_flight:
                    ages = age_meter.consume(obs)
                    if max_image_age_s is not None:
                        # U-23, at the SUBMISSION point and on the obs being
                        # submitted — that is where a frame is consumed
                        # here, exactly as it is below.
                        fault = stale_image_report(
                            collector.last_image_wall(), time.monotonic(), max_image_age_s
                        )
                        if fault is not None:
                            log.error("%s", fault)
                            raise StaleImageError(fault)
                    inference.submit(obs)
                    log.info(
                        "policy request t_sim=%.2f img_age_s=[%s]",
                        obs.t_sim - t_start,
                        format_ages(ages),
                    )
            elif executor.needs_replan(obs.t_sim):
                ages = age_meter.consume(obs)  # sampled on the Obs infer() sees
                if max_image_age_s is not None:
                    # U-23: sampled HERE, beside the age meter, because this
                    # is where a frame is actually consumed — a 20 Hz-tick
                    # check would measure a cadence no policy depends on.
                    fault = stale_image_report(
                        collector.last_image_wall(), time.monotonic(), max_image_age_s
                    )
                    if fault is not None:
                        # Logged as well as raised: the exception's own path
                        # out (run_policy's finally -> deactivate -> exit
                        # non-zero) is correct but silent in the ROS log.
                        log.error("%s", fault)
                        raise StaleImageError(fault)
                t_inf = time.monotonic()
                chunk = backend.infer(obs)
                infer_s = time.monotonic() - t_inf
                infer_times.append(infer_s)
                tick_infer_s = infer_s
                actions_for_executor = chunk_for_executor(
                    chunk.actions, action_space, left_gripper_hold
                )
                if chunk_time_base == "arrival":
                    # Plan-then-execute: the arm held (keep-alive covers it)
                    # for `infer_s` while this chunk was computed, so row 0
                    # is not stale against `chunk.t0` (the observation) — it
                    # is stale against NOW. Re-sample the collector (a cheap
                    # cached read, like every other `get_obs()` call in this
                    # loop) to get the sim time of the tick this chunk is
                    # actually installed on, and stamp it there instead —
                    # `t0 == t_sim_now` makes the wall index 0 at
                    # installation, so `step` plays row 0 first rather than
                    # skipping the rows an arm that never moved never played.
                    arrival_obs = collector.get_obs()
                    t_sim_install = arrival_obs.t_sim if arrival_obs is not None else obs.t_sim
                    chunk_t0 = t_sim_install
                else:
                    t_sim_install = obs.t_sim
                    chunk_t0 = chunk.t0
                executor.set_chunk(
                    chunk_t0,
                    actions_for_executor,
                    chunk.dt,
                    t_sim_now=t_sim_install,
                )
                if chunk_dump_recorder is not None:
                    wall_idx = (
                        (t_sim_install - chunk_t0) / chunk.dt if chunk.dt else 0.0
                    )
                    chunk_dump_recorder.record(
                        chunk_num=len(infer_times),
                        t0=chunk_t0,
                        t_sim_arrival=t_sim_install,
                        wall_idx=wall_idx,
                        spliced_idx=wall_idx + (executor.last_splice_shift or 0.0),
                        jump=executor.last_arrival_jump_rad,
                        lead=executor.last_cmd_lead_rad,
                        obs_state=obs.state,  # SAME obs just handed to backend.infer
                        last_cmd_arms=executor.last_commanded_arms,
                        actions=actions_for_executor,
                    )
                log.info(
                    "policy chunk #%d t_sim=%.2f infer=%.3fs H=%d dt=%.3f img_age_s=[%s]%s",
                    len(infer_times),
                    obs.t_sim - t_start,
                    infer_s,
                    len(chunk.actions),
                    chunk.dt,
                    format_ages(ages),
                    format_splice(executor),
                )

            # The LIVE pad pose, read ONCE per tick and above both conditions
            # below. It used to sit inside `if grasp_gate is not None` (itself
            # inside `if command is not None`), which would have made the
            # protocol's pad metric exist only when --grasp-gate was passed and
            # only on ticks the executor spoke. Metric 2 does not depend on
            # metric 1's flag, and where the pad is does not depend on either.
            pad_xyz = collector.object_xyz(PAD_OBJECT_NAME)
            # The clamp is per-TICK, and ExecutorStats only counts it
            # cumulatively — so read the counter either side of the step
            # rather than re-deriving the clamp here, where a second copy
            # of the rule could disagree with the one that actuated.
            clamped_before = executor.stats.clamped_ticks
            if inference is not None and executor.exhausted(obs.t_sim):
                # The reply is still on the wire and the chunk has run out:
                # hold the last commanded target. `step` would instead keep
                # ramping toward the chunk's final row, which is a target
                # the policy stopped vouching for.
                command = executor.hold(obs.t_sim)
            else:
                command = executor.step(obs.t_sim, obs.state)
            tick_n += 1
            tick_clamped = (
                None if command is None
                else executor.stats.clamped_ticks > clamped_before
            )
            # False by default (no command this tick, or no latch configured
            # at all) rather than None — unlike tick_clamped, "did the latch
            # force this tick's value" has an honest 0 answer when there is
            # nothing to force, so the CSV column never needs an empty cell.
            tick_gripper_latched = False
            if command is not None:
                if grasp_gate is not None:
                    # F-97: the checkpoints never predict the close. Supply it
                    # geometrically from the LIVE pad pose. None = defer to the
                    # policy, so the gate can only ever ADD a close.
                    grip = grasp_gate.update(
                        obs.state[C.S_RIGHT_ARM],
                        obs.state[C.S_BASE_ODOM],
                        float(obs.state[C.S_SPINE]),
                        pad_xyz,
                        obs.t_sim,
                    )
                    if grip is not None:
                        if grasp_gate.fired_at_sim_s == obs.t_sim:
                            log.info(
                                "grasp gate FIRED at t_sim %.2f s: dist %.1f mm, "
                                "angle %.2f deg (envelope %.0f mm / %.0f deg)",
                                obs.t_sim,
                                grasp_gate.last_dist_m * 1000.0,
                                grasp_gate.last_angle_deg,
                                grasp_gate.max_dist_m * 1000.0,
                                grasp_gate.max_angle_deg,
                            )
                        command = replace(command, right_gripper=grip)
                if gripper_latch is not None:
                    # Sits AFTER the executor and any grasp-gate override,
                    # BEFORE the publish — so chunk dumps upstream still
                    # record the policy's raw value, and the wire gets
                    # whatever the latch decides. Right gripper only; the
                    # left is never passed through this.
                    was_engaged = gripper_latch.engaged
                    engaged_ticks_before = gripper_latch.engaged_ticks
                    latched_grip = gripper_latch.update(
                        command.right_gripper,
                        obs.t_sim,
                        right_arm_rad=obs.state[C.S_RIGHT_ARM],
                    )
                    tick_gripper_latched = gripper_latch.engaged_ticks > engaged_ticks_before
                    if not was_engaged and gripper_latch.engaged:
                        if gripper_latch.near_pose is not None:
                            log.info(
                                "gripper latch ENGAGED at t_sim %.2f s (tick %d): "
                                "commanded %.3f < close_below %.3f, near_rad=%.3f "
                                "(radius %.3f)",
                                obs.t_sim, tick_n, command.right_gripper,
                                gripper_latch.close_below,
                                gripper_latch.near_dist_at_engage,
                                gripper_latch.near_radius_rad,
                            )
                        else:
                            log.info(
                                "gripper latch ENGAGED at t_sim %.2f s (tick %d): "
                                "commanded %.3f < close_below %.3f",
                                obs.t_sim, tick_n, command.right_gripper,
                                gripper_latch.close_below,
                            )
                    elif was_engaged and not gripper_latch.engaged:
                        log.info(
                            "gripper latch RELEASED at t_sim %.2f s (tick %d): "
                            "commanded %.3f > release_above %.3f for %.1f s",
                            obs.t_sim, tick_n, command.right_gripper,
                            gripper_latch.release_above, RELEASE_SUSTAIN_S,
                        )
                    command = replace(command, right_gripper=latched_grip)
                publisher.publish(command)
                # Read the gripper where the command LEAVES: after any gate
                # override and beside the publish, so `command.right_gripper`
                # (the canonical open fraction, chunk_executor.py:141 — 1.0
                # open, 0.0 closed) is the value that actually reached the
                # wire this tick, not the policy's overridden prediction. A
                # close the gate forced is therefore a close the observer
                # sees. A tick with no command sent nothing, so there is
                # nothing to read on it; a stand-in "open" would manufacture a
                # falling edge at the next real close, which is exactly the
                # false grasp attempt the edge rule exists to avoid (a channel
                # pinned closed from tick 0 is F-97's ABSENT signature and
                # must not read as an attempt).
                observer.update(
                    obs.state[C.S_RIGHT_ARM],
                    obs.state[C.S_BASE_ODOM],
                    float(obs.state[C.S_SPINE]),
                    pad_xyz,
                    command.right_gripper,
                    obs.t_sim,
                )
            if joint_log is not None:
                # Read at the publish site, one tick after the fact: on a
                # gated tick `command` is the REPLACED command, so the
                # gripper column is what left, not what the policy asked
                # for. The arm targets are the executor's clamped
                # ROBOT-frame values — see joint_log's module docstring for
                # why the publisher's wire values are deliberately not the
                # thing compared against a measured joint.
                joint_log.record(
                    tick=tick_n,
                    phase=ROLLOUT_PHASE,
                    t_sim=obs.t_sim,
                    measured_arms=np.concatenate(
                        [obs.state[C.S_LEFT_ARM], obs.state[C.S_RIGHT_ARM]]
                    ),
                    measured_grippers=(
                        obs.state[C.S_LEFT_GRIP], obs.state[C.S_RIGHT_GRIP]
                    ),
                    commanded_arms=(
                        None if command is None
                        else np.concatenate([command.left_arm, command.right_arm])
                    ),
                    commanded_grippers=(
                        None if command is None
                        else (command.left_gripper, command.right_gripper)
                    ),
                    clamped=tick_clamped,
                    infer_s=tick_infer_s,
                    max_publish_gap_s=(
                        None if publish_stats_fn is None
                        else publish_stats_fn().get("max_publish_gap_s")
                    ),
                    state_age_s=tick_state_age_s,
                    gripper_latched=tick_gripper_latched,
                )
            # Unconditional, unlike the observer above: where the pad is, is
            # a fact about the scene rather than about whether the executor
            # produced a command. NOTE the source: `collector.pad_xyz()`, the
            # mesh centroid, NOT `object_xyz(PAD_OBJECT_NAME)` -- the pad is a
            # deformable body whose prim transform is written at reset and
            # never again, so the object-poses value reports "never moved"
            # about an episode that carried the pad to the target (measured
            # 2026-08-25, run 20260825_092018). The target IS a rigid prim, so
            # it comes from object poses.
            pad_live_xyz = collector.pad_xyz()
            target_xyz = (
                None if target_object is None else collector.object_xyz(target_object)
            )
            tracker.update(pad_live_xyz, obs.t_sim, target_xyz)
            _record_base(traj_log, obs.t_sim, obs.state, "rollout")
            _record_tcp(tcp_log, kin, obs.t_sim, obs.state)
            _record_pad(pad_log, obs.t_sim, pad_live_xyz, pad_xyz, target_xyz)
            if on_tick is not None:
                on_tick()
            elapsed = time.monotonic() - loop_t0
            if elapsed < tick:
                time.sleep(tick - elapsed)

        publisher.safe_stop()
        stats = executor.stats.as_dict()
        stats["sim_seconds"] = float(obs.t_sim - t_start)
        stats["wall_seconds"] = float(time.monotonic() - wall_t0)
        stats["inferences"] = len(infer_times)
        stats["infer_mean_s"] = float(np.mean(infer_times)) if infer_times else None
        # Median and max beside the mean: the rig's RTT spread is 0.33-0.53 s
        # (U-29) and it is the MAX that decides whether a chunk outlives its
        # successor's flight time — a mean cannot answer that.
        stats["infer_median_s"] = float(np.median(infer_times)) if infer_times else None
        stats["infer_max_s"] = float(np.max(infer_times)) if infer_times else None
        stats["async_inference"] = inference is not None
        # Beside the jump/shift aggregates as_dict() already carries: which
        # policy produced them, so a stored report cannot be read as the
        # wrong one (U-31).
        stats["chunk_splice"] = executor.chunk_splice
        # U-32, beside the leash counters as_dict() already carries: what the
        # leash was set to, so `leash_active_pct: 0.0` cannot be read as
        # "the command never ran ahead" when it really means "off".
        stats["leash_rad"] = executor.leash_rad
        if inference is not None:
            stats.update(inference.stats())
            # Chunk steps the round trip had already consumed when the reply
            # landed (≈ RTT/dt). The horizon must cover TWICE this, not
            # once: the successor is requested on the arrival tick, so a
            # chunk is replaced around index 2 x this — ~19 of ACT's 21 at
            # a 0.45 s RTT, which is where `starved_ticks` comes from.
            stats["chunk_arrival_index_max"] = round(arrival_index_max, 2)
        # image_* keys: sim-time staleness (camelo/control/image_age.py).
        # `image_lag_frames30` is the P3-grid number the N2b step-1 verdict
        # reads (<= 2.7 => the render floor was a units artifact).
        stats.update(
            age_meter.stats(
                collector.frames_since(frame_marker),
                stats["sim_seconds"],
                stats["wall_seconds"],
            )
        )
        stats.update(backend.stats())
        # `keepalive_republished` + `max_publish_gap_s`. The gap is what the
        # rig runbook's T6 asks for: the companion's
        # joint_impedance_controller calls rclcpp::shutdown() after 2.0 s
        # without a sample, so the worst gap IS the margin, and a mean would
        # hide the single stall that decides whether the arm launch survived.
        publish_stats = getattr(publisher, "publish_stats", None)
        if publish_stats is not None:
            stats.update(publish_stats())
        if approach_stats is not None:
            stats["approach"] = approach_stats
        stats["base_trajectory_csv"] = str(traj_log.path)
        stats["tcp_trajectory_csv"] = str(tcp_log.path)
        stats["pad_trajectory_csv"] = str(pad_log.path)
        if joint_log is not None:
            # The whole per-joint summary, not a headline: every number in
            # the printed T5 block is here too, and every one of them is
            # re-derivable from the CSV's rows. Guarded the same way as the
            # finally block's report below: the rows are the measurement,
            # this is a read of them, and a failure reading them must not
            # cost the caller the rest of stats.
            try:
                stats["joint_tracking"] = joint_log.summary()
            except Exception as exc:
                log.error("joint trace summary failed: %r", exc)
        if chunk_dump_recorder is not None:
            stats["chunk_dump_npz"] = str(chunk_dump)
            stats["chunk_dump_chunks"] = chunk_dump_recorder.n
        # Protocol §0 metrics 1 and 2, on every rollout path — merged where
        # the instruments are read so no caller can measure and then discard
        # the reading (`graspgate_n2` carried the gate's distances in its rows
        # and reported mean_iou 0.0 as the whole result). batch_eval.py
        # aggregates these under the `loop_` prefix run_episode adds.
        stats.update(observer.stats())
        stats.update(tracker.stats())
        if gripper_latch is not None:
            stats.update(gripper_latch.stats())
        return stats
    finally:
        # Before the logs, the inference worker: it holds the backend, and
        # run_policy.py's own finally closes that (and deactivates the
        # arms) the moment this returns. Guarded and time-boxed — a wedged
        # request must not delay a deactivation.
        if inference is not None:
            with contextlib.suppress(Exception):
                inference.close()
        # Close the joint log FIRST, and unconditionally: the rows are
        # already on disk (every one was flushed as it was written), so
        # the close only has to release the handle, and the three
        # pre-existing closes below must not be skipped by anything that
        # goes wrong in the summary this log also produces.
        if joint_log is not None:
            with contextlib.suppress(Exception):
                joint_log.close()
            # The summary is a convenience READ of the rows already safely
            # on disk (AGENTS.md: "the summary lies, not the measurement")
            # — a failure computing or printing it must not cost an
            # operator's Ctrl+C the traj/tcp/pad closes still to come.
            try:
                report = joint_log.format_summary()
                print(report, flush=True)
                log.info("%s", report)
            except Exception as exc:
                log.error("joint trace summary failed: %r", exc)
        # The chunk dump, same guard: it must survive whatever the try body
        # left (including a StaleImageError/StaleStateError raised before
        # `stats` above was ever built), and its own failure must not cost
        # the traj/tcp/pad closes still to come.
        if chunk_dump_recorder is not None:
            try:
                chunk_dump_recorder.write(chunk_dump, chunk_dump_args)
            except Exception as exc:
                log.error("chunk dump write failed: %r", exc)
        # These three were already mutually fragile (one raising would
        # skip the rest) — now guarded the same way as the joint log above.
        with contextlib.suppress(Exception):
            traj_log.close()
        with contextlib.suppress(Exception):
            tcp_log.close()
        with contextlib.suppress(Exception):
            pad_log.close()


class EpisodeRunner:
    def __init__(
        self,
        node,
        collector: ObsCollector,
        publisher: CommandPublisher,
        executor: ChunkExecutor,
        backend: PolicyBackend,
        task: str,
        settle_s: float = 3.0,
        eval_output_dir: Path | None = None,
        approach: ApproachController | None = None,
        approach_timeout_s: float = 30.0,
        start_pose: tuple[Sequence[float], Sequence[float]] | None = None,
        grasp_gate=None,
        grasp_observer: GraspObserver | None = None,
        start_pose_tol: float = START_POSE_TOL_RAD,
        start_pose_check: bool = True,
        action_space: str = CANONICAL_SPACE,
    ):
        self.node = node
        self.collector = collector
        self.publisher = publisher
        self.executor = executor
        self.backend = backend
        self.task = task
        self.settle_s = settle_s
        self.eval_output_dir = eval_output_dir or EVAL_OUTPUT_DIR
        self.approach = approach
        self.approach_timeout_s = approach_timeout_s
        # Where recenter_arms() parks the arms before a rollout. Default is
        # the benchmark's ARM_READY_POSE, which is what F-86 shipped — but
        # see the warning there and in contracts.py:226: that is the
        # spine-DOWN pose, and no demo contains it with the spine at the SOP.
        # Whether the caller ASKED for this pose decides what an unreached
        # pose means: on the default it is one bad episode, on an explicit
        # --start-pose it invalidates the comparison the run exists to make.
        self.start_pose_is_default = start_pose is None
        self.world = getattr(getattr(collector, "topics", None), "world", C.WORLD_SIM)
        self.start_pose_tol = float(start_pose_tol)
        self.start_pose_check = bool(start_pose_check)
        self.action_space = action_space
        if start_pose is None and self.world == C.WORLD_REAL:
            # NO default pose on the real robot. `ARM_READY_POSE` is a SIM
            # pose and `recenter_arms` would drive two Franka arms to it;
            # there is nothing to substitute, so there is nothing to check.
            self.start_pose = None
        else:
            left, right = start_pose or (C.ARM_READY_POSE, C.ARM_READY_POSE)
            self.start_pose = (
                np.asarray(left, dtype=np.float32),
                np.asarray(right, dtype=np.float32),
            )
            for name, pose in (
                ("left", self.start_pose[0]), ("right", self.start_pose[1])
            ):
                if pose.shape != (len(C.ARM_READY_POSE),):
                    raise ValueError(
                        f"start_pose {name} must be ({len(C.ARM_READY_POSE)},), "
                        f"got {pose.shape}"
                    )
        # F-97 ablation: geometric gripper close. None = the policy owns the
        # channel, which is the default and the honest configuration.
        self.grasp_gate = grasp_gate
        # Built unconditionally, and on the GATE's envelope when there is one
        # (see _grasp_observer): the ablation decides what the robot does, the
        # instruments decide what gets measured, and protocol C1 needs the
        # detector live in the WITHOUT-gate arm too. Owned here rather than in
        # run_rollout so run_episode can reset them per episode.
        # A caller-supplied observer (--grasp-envelope) is how the gate-OFF
        # arm gets the same 45 mm / 7 deg bar as the gate-on arm. Without it
        # the fallback has no reference attitude and scores "inside" on
        # distance alone -- C1 would look one-variable and not be.
        self.grasp_observer = grasp_observer or _grasp_observer(grasp_gate)
        self.pad_tracker = PadTracker()
        self.last_eval_json: Path | None = None

        from std_srvs.srv import Trigger

        self._eval_client = node.create_client(Trigger, C.EVAL_SERVICE)
        self._Trigger = Trigger

    def reset_scene(self, timeout_s: float = 30.0) -> None:
        self.collector.drain_reset_events()
        self.collector.request_scene_reset()
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            if self.collector.drain_reset_events():
                time.sleep(self.settle_s)
                return
            time.sleep(0.1)
        raise TimeoutError(
            "scene reset not acknowledged on "
            f"{C.SCENE_RESET_TOPIC} — was the sim launched with --record?"
        )

    def _measured_arms(self, obs) -> np.ndarray:
        return np.concatenate([obs.state[C.S_LEFT_ARM], obs.state[C.S_RIGHT_ARM]])

    def wait_for_spine(self, timeout_s: float = 20.0, wall_timeout_s: float = 120.0) -> float:
        """Hold the reset pose until the spine is back at the SOP. Returns
        the measured height [m].

        A scene reset drops the spine to ~0.05 m; the launch pin ramps it
        back to ~0.50 m over ~4.4 SIM seconds (measured on the DGX rig
        2026-08-15). ``recenter_arms`` used to command the start pose
        immediately, i.e. squarely inside that window — and every
        non-default start pose is a spine-UP pose (they put the flange
        ~0.19 m over the table at the SOP, contracts.py:223). Commanded
        0.44 m too low they drive the hands into the desk, where they JAM
        and stay jammed after the spine rises: measured left j6 0.994 rad
        short and motionless for 25 sim-s, flange z frozen to the
        millimetre, while ``/isaac/applied_joint_commands`` carried the
        exact target we asked for. Over-commanding that joint by a further
        1.23 rad moved it 0.02 rad — a contact, not a slow controller, so
        no timeout could have saved it. The three --start-pose runs of
        2026-08-15 are all this.

        ``ARM_READY_POSE`` is safe to hold at any height — it is the pose
        the reset teleports to — and holding it here also overwrites the
        previous episode's stale controller target immediately, which is
        what F-86 wanted in the first place.
        """
        ready = np.asarray(C.ARM_READY_POSE, dtype=np.float32)
        wall_deadline = time.monotonic() + wall_timeout_s
        t0 = None
        spine = float("nan")
        while time.monotonic() < wall_deadline:
            self.publisher.publish_arms(ready, ready)
            time.sleep(0.1)
            obs = self.collector.get_obs(require_images=False)
            if obs is None:
                continue
            spine = float(obs.state[C.S_SPINE])
            if spine >= C.SPINE_SOP_MEASURED_MIN_M:
                if t0 is not None:
                    log.info(
                        "spine back at the SOP: %.3f m after %.1f sim-s",
                        spine, obs.t_sim - t0,
                    )
                return spine
            if t0 is None or obs.t_sim < t0:
                t0 = obs.t_sim  # first sample, or a reset rebased the clock
            if obs.t_sim - t0 >= timeout_s:
                break
        log.warning(
            "spine still at %.3f m (gate %.2f m) after waiting — recentring anyway, "
            "but a spine-UP start pose commanded this low will collide",
            spine, C.SPINE_SOP_MEASURED_MIN_M,
        )
        return spine

    def recenter_arms(
        self,
        timeout_s: float = 20.0,
        tol: float = ARM_SETTLE_TOL_RAD,
        wall_timeout_s: float = 300.0,
    ) -> float:
        """Command ``self.start_pose`` until the arms measurably hold it (F-86).

        ``timeout_s`` is in SIM seconds, like ``rollout_s`` and the approach
        budget: the arms move in sim time and the sim runs 5-9x slower than
        wall under eval load, so the old wall-clock budget shrank to ~6 sim-s
        exactly when episodes were hardest. ``wall_timeout_s`` is only a
        backstop for a dead sim clock.

        Convergence is a plateau, not a threshold crossing — see
        camelo/runner/recenter.py for the rule and the measurements behind
        ``tol``. Returns the final max per-joint error [rad].

        A pose that was explicitly requested and not reached RAISES
        ``StartPoseNotReached``: the episode did not start from the
        condition the run is testing, and three runs were already reported
        as a comparison of start poses that were never reached. The default
        pose still only warns, so one stubborn episode cannot abort a batch.

        The pose matters as much as the holding. The default
        ``ARM_READY_POSE`` is spine-DOWN, and the trimmed training corpora
        contain **zero** frames within 0.5 rad of it — see
        docs/research/PI05_STABILITY_INVESTIGATION.md. Pass ``start_pose`` to park
        the arms somewhere the policy has actually seen.

        **On the real robot this commands nothing.** It verifies instead —
        see `verify_start_pose_only`.
        """
        if self.world == C.WORLD_REAL:
            return self.verify_start_pose_only()
        left, right = self.start_pose
        target = np.concatenate([left, right])
        # Which pose this batch ran from is not otherwise recoverable from
        # results.csv, and it changes the result — say it once per episode.
        log.info(
            "recentring arms to %s (max %.3f rad from the benchmark ARM_READY_POSE)",
            "the benchmark ARM_READY_POSE" if self.start_pose_is_default
            else "a non-default start pose",
            float(np.max(np.abs(target - np.tile(C.ARM_READY_POSE, 2)))),
        )
        self.wait_for_spine()

        monitor = ArmSettleMonitor(tol_rad=tol)
        wall_deadline = time.monotonic() + wall_timeout_s
        t0, error, verdict, measured = None, float("inf"), None, None
        while time.monotonic() < wall_deadline:
            self.publisher.publish_arms(left, right)
            time.sleep(0.1)
            obs = self.collector.get_obs(require_images=False)
            if obs is None:
                continue
            measured = self._measured_arms(obs)
            error = pose_error(measured, target)
            if t0 is None or obs.t_sim < t0:
                t0 = obs.t_sim
            verdict = monitor.update(obs.t_sim, error)
            if verdict == REACHED:
                log.info(
                    "arms recentred: max joint error %.4f rad, stable for %.0f sim-s",
                    error, monitor.window_s,
                )
                return error
            if verdict == STUCK or obs.t_sim - t0 >= timeout_s:
                break

        if measured is None:
            t = getattr(self.collector, "topics", None)
            if t is not None and t.world == "real":
                where = "station joint states (record_bag.bash)"
            else:
                where = f"{C.FULL_STATES_TOPIC} and {C.CLOCK_TOPIC}"
            raise TimeoutError(
                f"no observation in {wall_timeout_s:.0f}s while recentring the arms — "
                f"is the robot publishing {where}? "
                f"rates: {self.collector.rates()}"
            )
        wound = wound_joints(measured)
        if wound:
            # Not distance-from-target at all: the joint has wound past a full
            # revolution in the sim and a reset did not unwind it. Separate
            # fault, separate message — folding it into the pose error would
            # report a 16 rad "miss" for a pose that is 1 rad away.
            log.error(
                "arm joints %s read past +/-2 pi (%s rad) — wound up in the sim, "
                "not off-pose; this episode's state is not comparable to any other",
                wound, [round(float(measured[i]), 3) for i in wound],
            )
        # Raises when the pose was explicitly requested; returns the warning
        # text for the default pose (camelo/runner/recenter.py).
        message = report_unreached(
            measured, target, verdict,
            is_default=self.start_pose_is_default, tol=tol, timeout_s=timeout_s,
        )
        log.warning(
            "arms did NOT settle at the ready pose — this episode starts off-pose "
            "(F-86): %s", message,
        )
        return error

    def verify_start_pose_only(self) -> float:
        """Real robot: CHECK the arms against `--start-pose`, never move them.

        `recenter_arms` exists because a sim reset leaves a stale controller
        target behind (F-86). A real robot has no reset, no stale target, and
        no pose this repo is entitled to choose: commanding the sim's
        `ARM_READY_POSE` into two Franka arms standing at the teleop home
        pose is a large unplanned motion through whatever is on the table.
        So the flag becomes an assertion about where the operator left them,
        and the run refuses to start from anywhere else.

        Returns the max per-joint error [rad] (0.0 when no pose was given).
        """
        obs = self.collector.get_obs(require_images=False)
        if obs is None:
            obs = wait_for_obs(self.collector, require_images=False)
        left = obs.state[C.S_LEFT_ARM]
        right = obs.state[C.S_RIGHT_ARM]
        if self.start_pose is None:
            log.info(
                "real robot: no --start-pose, nothing verified and nothing "
                "commanded. Measured left=%s right=%s",
                [round(float(v), 4) for v in left],
                [round(float(v), 4) for v in right],
            )
            return 0.0
        errors = verify_start_pose(
            left, right, self.start_pose, self.start_pose_tol, self.start_pose_check
        )
        return errors["max"]

    def evaluate(self, timeout_s: float = 60.0) -> dict:
        if not self._eval_client.wait_for_service(timeout_sec=10.0):
            raise TimeoutError(
                f"eval service {C.EVAL_SERVICE} unavailable — start it with "
                "bash <benchmark>/scripts/evaluation/task2/run.sh up"
            )
        before = self._newest_eval_json()
        future = self._eval_client.call_async(self._Trigger.Request())
        t0 = time.monotonic()
        while not future.done():
            if time.monotonic() - t0 > timeout_s:
                raise TimeoutError("eval service call timed out")
            time.sleep(0.1)
        response = future.result()
        if not response.success:
            raise RuntimeError(f"eval service failed: {response.message}")

        for _ in range(50):  # the JSON lands moments after the service returns
            newest = self._newest_eval_json()
            if newest is not None and newest != before:
                self.last_eval_json = newest
                return json.loads(newest.read_text())
            time.sleep(0.2)
        raise FileNotFoundError(
            f"no new eval_camera_iou_*.json under {self.eval_output_dir} — "
            "check EVAL_OUTPUT_DIR / ISAAC_DOCKER_ROOT"
        )

    def _newest_eval_json(self) -> Path | None:
        if not self.eval_output_dir.is_dir():
            return None
        files = sorted(self.eval_output_dir.glob(f"{C.EVAL_IOU_PREFIX}*.json"))
        return files[-1] if files else None

    def collect_artifacts(self, dest: Path) -> list[Path]:
        """Copy the last evaluation's full artifact set (the IoU JSON plus
        the frames sharing its timestamp) next to the run ledger, so a
        scored run stays with the evidence that explains it (F-31)."""
        if self.last_eval_json is None:
            return []
        dest.mkdir(parents=True, exist_ok=True)
        copied = []
        for path in C.eval_artifact_paths(self.last_eval_json):
            shutil.copy2(path, dest / path.name)
            copied.append(dest / path.name)
        return copied

    def run_episode(
        self,
        index: int,
        rollout_s: float,
        rate_hz: float = 20.0,
        artifact_dir: Path | None = None,
    ) -> dict:
        log.info("episode %d: reset", index)
        self.reset_scene()
        # F-72: nothing on a policy launch's reset path restores the
        # gripper (ARM_READY_POSE has no gripper joints; the bridge's
        # reopen is keyboard-teleop-only) and the drives hold their last
        # applied target — a grasped episode would start the next one
        # closed, a state no demo contains. Command open explicitly;
        # travel (~3 s, F-27) overlaps the settle already served.
        for _ in range(3):
            self.publisher.publish_grippers(1.0, 1.0)
            time.sleep(0.1)
        # F-86: the ARMS have the identical problem, and it is worse because
        # a reset LOOKS like it fixed them. The reset teleports the joints to
        # ARM_READY_POSE, then the position controller re-applies the target
        # the previous episode left behind and pulls them back out within
        # ~5 s. Measured: +2 s after reset 0.001 rad from ready, +5 s 0.594
        # rad from ready and 0.006 from the stale target, held indefinitely.
        # Without this, every episode after the first starts wherever the
        # last one ended — the episodes are not independent trials.
        self.recenter_arms()
        self.executor.reset()
        # The gate LATCHES closed by design (the dwell inside the envelope is
        # short). Reset it here or every episode after the first starts already
        # closed — the same "episodes are not independent trials" failure the
        # recentre above exists to prevent.
        if self.grasp_gate is not None:
            self.grasp_gate.reset()
        # Same reason, and unconditionally: a PadTracker still holding episode
        # 1's baseline would measure episode 2's displacement from the wrong
        # origin, and a GraspObserver still holding episode 1's last gripper
        # command would read episode 2's first close as a falling edge that
        # never happened. Episodes are not independent trials if state leaks
        # across them (F-86).
        self.grasp_observer.reset()
        self.pad_tracker.reset()
        self.backend.reset(self.task)
        log.info("episode %d: rollout %.0f sim-s", index, rollout_s)
        recorder = None
        episode_dir = (
            Path(artifact_dir) / f"episode_{index:03d}" if artifact_dir is not None else None
        )
        if episode_dir is not None:
            from camelo.ros.eval_camera_recorder import EvalCameraRecorder

            recorder = EvalCameraRecorder()
            episode_dir.mkdir(parents=True, exist_ok=True)
        traj_csv = (
            episode_dir / "base_trajectory.csv" if episode_dir is not None else None
        )
        tcp_csv = (
            episode_dir / "tcp_trajectory.csv" if episode_dir is not None else None
        )
        try:
            stats = run_rollout(
                self.collector,
                self.publisher,
                self.executor,
                self.backend,
                rollout_s,
                rate_hz,
                on_tick=recorder.poll if recorder is not None else None,
                approach=self.approach,
                approach_timeout_s=self.approach_timeout_s,
                traj_csv=traj_csv,
                tcp_csv=tcp_csv,
                grasp_gate=self.grasp_gate,
                grasp_observer=self.grasp_observer,
                pad_tracker=self.pad_tracker,
                action_space=self.action_space,
            )
            # `stats` already carries the instruments' keys (grasp_env_*,
            # grasp_close_*, pad_*): run_rollout merged them into this same
            # dict where it read them, so they arrive whether or not a gate
            # ran. The gate's own keys are the ablation's, and only exist
            # when the ablation did.
            if self.grasp_gate is not None:
                stats.update(self.grasp_gate.stats())
            if recorder is not None and episode_dir is not None:
                # The recorder samples from on_tick, and run_rollout hands the
                # SAME on_tick to run_approach — so the frames span approach +
                # rollout, not the rollout alone. Sizing fps on rollout_s only
                # stamped every eval mp4 (approach+rollout)/rollout too fast:
                # 68.7-80.7 sim-s of footage declared as 45.0, i.e. 1.5-1.8x,
                # which is enough to misread a rollout by eye.
                # No approach (--skip-approach) contributes 0.0.
                rollout_sim = float(stats.get("sim_seconds") or rollout_s)
                approach_sim = float((stats.get("approach") or {}).get("sim_seconds") or 0.0)
                for key, mp4 in recorder.write(
                    episode_dir,
                    duration_s=rollout_sim + approach_sim,
                ).items():
                    stats[f"{key}_mp4"] = str(mp4)
        finally:
            if recorder is not None:
                recorder.close()

        # Attach loop_* before evaluate: a scoring failure must not erase harness
        # timing (mean_rollout_* in summary.json).
        row = {
            "episode": index,
            "iou": None,
            "orientation_correct": None,
            "orientation_case": None,
            **{f"loop_{k}": v for k, v in stats.items()},
        }
        log.info("episode %d: evaluate", index)
        try:
            result = self.evaluate()
        except Exception as exc:
            row["error"] = str(exc)
            log.error("episode %d: evaluate failed: %s", index, exc)
            return row
        if episode_dir is not None:
            copied = self.collect_artifacts(episode_dir)
            log.info("episode %d: %d eval artifacts copied", index, len(copied))
        row.update(
            {
                "iou": result.get("iou_thermalpad_vs_target_current"),
                "orientation_correct": result.get("is_orientation_correct"),
                "orientation_case": result.get("orientation_case"),
            }
        )
        log.info("episode %d: iou=%s orientation=%s", index, row["iou"], row["orientation_case"])
        return row
