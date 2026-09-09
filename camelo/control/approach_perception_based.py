"""Task 2 approach using the head camera instead of world odom for navigate.

Stages come from ``camelo.control.approach`` (imported, not copied).
Stages: spine → navigate → place_arms → start_pose → done — the finegrained
trim stage is disabled here (§1.5 of the design doc); place_arms already
keeps vision alive via ``align_edges``. Navigate seeds a Kalman pose filter
with a rough start pose (``--approach-start-xy-yaw``, sim default: Task 2
spawn), then every tick predicts from relative odometry and corrects from
the head camera when it locks — so a fused pose, and a plan, exist from the
very first tick instead of behind a blocking yaw search (§1, §1.2 of the
design doc).

The vision lives in ``camelo.control.perception`` (one module per stage);
this file is only the state machine. Design notes and the acceptance
criteria are in ``camelo/control/perception_based_navigation.md``.

Opt-in: ``--approach perception``. Pose-based ``ApproachController`` stays
the default.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from camelo import contracts as C
from camelo.control.approach import (
    ApproachController,
    Waypoint,
    body_frame_errors,
    wrap_angle,
)
from camelo.control.base_quantizer import BaseQuantizer
from camelo.control.chunk_executor import Command
from camelo.control.perception import viz
from camelo.control.perception.features import Features, detect, undistort
from camelo.control.perception.filter import PoseFilter
from camelo.control.perception.geometry import (
    BASE_RADIUS_M,
    T_base_cam,
    T_world_base,
    head_intrinsics,
    hfov_deg_from_intrinsics,
    project,
)
from camelo.control.perception.localize import EgoEstimate, TableLocalizer
from camelo.control.perception.planner import PathSample, SplinePlanner
from camelo.control.perception.profile import (
    SIM_PROFILE,
    PerceptionProfile,
    load_profile,
)
from camelo.control.perception.table import TableModel
from camelo.control.perception.walls import ROOM_WALLS

log = logging.getLogger(__name__)

__all__ = (
    "PerceptionApproachController",
    "format_xy_yaw",
)

_SPINE_HOLD_TICKS = 8
_DEFAULT_DT_S = 1.0 / 50.0
_MAX_DT_S = 0.2
_TRAIL_MAX = 4000

# Arrival tolerance on the final sample (§1.4) — the same 5 cm / 0.05 rad gate
# the pose-based goal waypoint used (approach.py's _FINE_XY/_FINE_YAW), so
# place_arms inherits a base that is actually at the handover pose. Only
# the final sample is gated at all: the path is refit fresh every tick from
# wherever the robot actually is, so every other sample is a live ~50 cm
# lookahead, not a fixed via to arrive at and advance past.
_GOAL_XY_M, _GOAL_YAW_RAD = 0.05, 0.05

# The base class (approach.py) still speaks Waypoint for its own
# tolerance/jitter/handover bookkeeping, but navigate here never walks a
# route — SplinePlanner fits one fresh every tick from wherever the robot
# actually is (perception_based_navigation.md §1.3). So, unlike pose-based
# nav's TASK2_APPROACH_WAYPOINTS, there is no fixed transit line or
# overshoot via to bake in: just the single Task 2 goal, wrapped for the
# base class's API.
# The shared contract goal (2.10, 3.05, −90°). Named separately so a
# perception-specific offset could be introduced without touching the
# pose-based controller — none is applied today; the wall margin there is
# 2.2 cm (design doc §1.3.2).
_GOAL_XY_YAW = C.TASK2_APPROACH_XY_YAW
_GOAL_WAYPOINT: tuple[Waypoint, ...] = (
    Waypoint("goal", *_GOAL_XY_YAW, _GOAL_XY_M, _GOAL_YAW_RAD),
)

# place_arms never moves the base (approach.py's _step_place_arms always
# commands "NONE"), so if the rear table edge isn't visible by the time
# navigate is ready to hand off, nothing downstream can fix that — this has
# to happen here, before the handoff. Each failed check retreats the final
# target one step farther (past _GOAL_XY_YAW; that's fine, the goal is
# nominal, not a hard stop — see forward_from_horizontal_rim). Capped so a
# detector that never finds the rim (bad lighting, whatever) can't retreat
# the base indefinitely — past the cap, settle at the nominal goal anyway.
_REAR_RIM_BACKOFF_STEP_M = 0.15
_REAR_RIM_BACKOFF_MAX_M = 0.90

# Sim's wire has no slow gear: BaseQuantizer only ever plays a token at full
# PEDAL_SPEED, so a continuous P response that is still "correcting" right up
# to the tolerance boundary drives at full speed until the token releases,
# overshoots by the base's real momentum, and reverses — MEASURED: a
# navigate/drive[5/5] sample oscillated with the SAME (gt, est) pair
# recurring every ~5.4 s for 20+ s, front swinging +-60-130 mm against a
# 50 mm tolerance, never once satisfying `_waypoint_ok`'s simultaneous
# in-tolerance + near-zero-velocity gate (the two conditions are
# anti-correlated in a limit cycle: the base is only stopped at its
# direction reversals, which is exactly where |err| peaks).
# Inside this distance of the target on whichever axis is actually on the
# wire, alternate fire/coast one control tick at a time instead of letting
# BaseQuantizer's engage/release hysteresis hold the token continuously —
# one tick of motion, one tick to bleed off momentum, re-measure, repeat.
# This does not touch `base_twist`: the real wire takes a genuine continuous
# twist with no quantization step, so it already tapers smoothly near the
# goal and gets none of this.
# 2x _GOAL_XY_M/_GOAL_YAW_RAD (both 0.05): the measured oscillation swung
# past a 0.05 near-zone before ever triggering it (front up to 130mm), so
# gating needs to start outside the arrival tolerance, not at its edge, to
# catch the approach before it overshoots rather than only after.
_NEAR_M = 0.10

# kp is sized against BaseQuantizer's ENGAGE threshold (0.5 of PEDAL_SPEED,
# base_quantizer.py), not a sim plant model this file has no access to.
# Release (0.3) only HOLDS a token that is already moving; starting from
# NONE needs engage, so sizing against release leaves a deadband wider than
# the tolerance and the base stops short of the goal forever — MEASURED: at
# kp 3.0/8.0 the last sample sat 9 cm out commanding twist (-0.24, -0.15)
# that quantized to NONE for 3725 straight ticks, and navigate never handed
# off. kp * tol >= 0.5 * PEDAL_SPEED at the tightest (final-sample:
# _GOAL_XY_M, _GOAL_YAW_RAD) tolerance gives linear 5.0 (0.05 m -> 0.25 m/s)
# and yaw 12.0 (0.05 rad -> 0.6 rad/s). These three constants are sized
# together: tightening _GOAL_YAW_RAD without rescaling kp_yaw reopens the
# same dead-base bug at the new, tighter tolerance.
# Consecutive rejections with nothing ever accepted ⇒ the seed is suspect.
_REJECT_WARN_N = 5
_PID_KP_LINEAR = 5.0
_PID_KP_YAW = 12.0
_MAX_LINEAR_MPS = C.PEDAL_LINEAR_SPEED
_MAX_ANGULAR_RADPS = C.PEDAL_ANGULAR_SPEED

# kp_yaw (12.0) is pinned to the tightest tolerance in play (_GOAL_YAW_RAD,
# via the anti-dead-base invariant above), so it saturates to the full
# PEDAL_ANGULAR_SPEED the instant |yaw_err| clears ~0.1 rad — a mobile base
# spinning at 1.2 rad/s (69 deg/s) reads as a much more violent motion than
# the 0.5 m/s the linear axes ever reach, even though both are, by
# construction, "at the cap". Ease the CAP itself down across a wider band
# instead of touching kp_yaw (which would reopen the dead-base bug at the
# tight goal tolerance): linear ramp from the floor at yaw_err=0 up to the
# full cap at _YAW_EASE_RAD. The floor is exactly the engage-threshold value
# the anti-dead-base proof already relies on (0.5 * PEDAL_ANGULAR_SPEED at
# _GOAL_YAW_RAD, kp_yaw * _GOAL_YAW_RAD == that same floor) — so this cannot
# stall the final turn, it only slows the approach into it. Via legs (15°
# tolerance) barely feel this: kp_yaw * 15° is already ~6x the floor, so the
# ramp only binds close to the (tighter) goal.
_YAW_EASE_RAD = math.radians(30.0)
_YAW_EASE_FLOOR_RADPS = 0.5 * C.PEDAL_ANGULAR_SPEED


def _yaw_speed_cap(yaw_err_abs: float) -> float:
    if yaw_err_abs >= _YAW_EASE_RAD:
        return _MAX_ANGULAR_RADPS
    frac = yaw_err_abs / _YAW_EASE_RAD
    return _YAW_EASE_FLOOR_RADPS + (_MAX_ANGULAR_RADPS - _YAW_EASE_FLOOR_RADPS) * frac


class _AxisPID:
    """P(I)(D) on one body-frame error axis, ``ki``/``kd`` unused for now.

    Real terms rather than hardcoded P so a continuous-twist real wire can
    tune them later without touching the call site — the quantized sim wire
    only plays one of seven fixed-speed tokens (nothing for a derivative to
    sharpen), and the unchanged fine-trim stage owns the final millimetres.
    """

    def __init__(self, kp: float, ki: float = 0.0, kd: float = 0.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self._integral = 0.0
        self._prev_err: float | None = None

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_err = None

    def step(self, err: float, dt_s: float) -> float:
        self._integral += err * dt_s
        deriv = 0.0
        if self._prev_err is not None and dt_s > 0.0:
            deriv = (err - self._prev_err) / dt_s
        self._prev_err = err
        return self.kp * err + self.ki * self._integral + self.kd * deriv


def _clip_twist(
    vx: float,
    vy: float,
    wz: float,
    max_linear: float = _MAX_LINEAR_MPS,
    max_angular: float = _MAX_ANGULAR_RADPS,
) -> tuple[float, float, float]:
    """Cap to the base's physical speed (pedal in sim, swerve clamp on the rig)."""
    speed = math.hypot(vx, vy)
    if speed > max_linear:
        scale = max_linear / speed
        vx, vy = vx * scale, vy * scale
    wz = max(-max_angular, min(max_angular, wz))
    return vx, vy, wz


def format_xy_yaw(x: float | None, y: float | None, yaw: float | None) -> str:
    """``(x, y, yaw)`` or ``(?, ?, ?)`` when any component is missing."""
    if x is None or y is None or yaw is None:
        return "(?, ?, ?)"
    if not all(math.isfinite(v) for v in (x, y, yaw)):
        return "(?, ?, ?)"
    return f"({x:.3f}, {y:.3f}, {yaw:+.4f})"


def _num(value) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _uv(uv) -> list[float] | None:
    if uv is None:
        return None
    u, v = _num(uv[0]), _num(uv[1])
    if u is None or v is None:
        return None
    return [round(u, 2), round(v, 2)]


def _est_record(estimate: EgoEstimate | None) -> dict | None:
    if estimate is None:
        return None
    return {
        "x": _num(estimate.x),
        "y": _num(estimate.y),
        "yaw": _num(estimate.yaw),
        "residual_px": _num(estimate.residual_px),
        "n_corr": int(estimate.n_corr),
        "n_edges": int(estimate.n_edges),
        "source": estimate.source,
        "edge_support": _num(estimate.edge_support),
        "unexplained": int(estimate.unexplained),
        "inlier_uv": [_uv(uv) for uv in estimate.inlier_uv],
        "tabletop_iou": _num(estimate.tabletop_iou),
        "tabletop_recall": _num(estimate.tabletop_recall),
        "instance_size_xy": None
        if estimate.instance_size_xy is None
        else [_num(v) for v in estimate.instance_size_xy],
        "instance_fits_table": estimate.instance_fits_table,
    }


class PerceptionApproachController(ApproachController):
    """Head-camera navigate, then the inherited waypoint walker.

    Main methods:
    - step(state, images=) — spine hold, search until located, then drive
    - localize(head) — edge features → gated absolute ego estimate, or None
    - dump(head) — PNG overlay|BEV plus a JSON sidecar of bbox + detections

    The estimate is absolute: ``solve``/``align_edges`` never read
    ``S_BASE_ODOM`` and never refine a previous fit, so a bad frame cannot
    poison the fused pose beyond the filter's own mirror-rejection (§1.2).
    Odom is display-only, printed beside the estimate as ``gt=``.
    """

    needs_images = True

    def __init__(
        self,
        *args,
        dump_dir: str | Path | None = None,
        start_xy_yaw: tuple[float, float, float] | None = None,
        spine_hold_ticks: int = _SPINE_HOLD_TICKS,
        table: TableModel | None = None,
        localizer: TableLocalizer | None = None,
        planner: SplinePlanner | None = None,
        quantizer: BaseQuantizer | None = None,
        spine_assume_m: float | None = None,
        vision_min_period_s: float = 0.0,
        profile: PerceptionProfile | str | Path | None = None,
        **kwargs,
    ):
        # Parent __init__ calls self.reset() before returning; everything
        # _reset_perception touches must exist before that call.
        # Scene profile: stand-in K/D for a camera with no live CameraInfo
        # (the Munich rig publishes none), that scene's tabletop thresholds,
        # and — on a rig — the drive limits, the landmark and the goal. A
        # path loads configs/rig/*.yaml; None is the sim default, i.e. live
        # CameraInfo (or the generic sim-FOV fallback), the floor-relative
        # surface rule, the frozen Task 2 table and the pedal path.
        if profile is None:
            self.profile = SIM_PROFILE
        elif isinstance(profile, PerceptionProfile):
            self.profile = profile
        else:
            self.profile = load_profile(profile)
        rig = self.profile.rig
        self.rig = rig
        self.spine_hold_ticks = max(1, int(spine_hold_ticks))
        # Assumed height when no /spine/joint_states exists at all (common
        # off-rig): C.SPINE_SOP_M is the SIM's standard-operating height,
        # not necessarily what a given real capture was actually at. A
        # profile's own ``spine_m`` (the rig's fixed height) wins over it.
        self._spine_assume_m = C.SPINE_SOP_M if spine_assume_m is None else float(spine_assume_m)
        # Vision cap for fast cameras (rig ZED ~20 Hz): at most one head
        # frame per period is detected/solved; 0 = every new frame.
        self.vision_min_period_s = max(0.0, float(vision_min_period_s))
        seed = start_xy_yaw if start_xy_yaw is not None else self.profile.start_xy_yaw
        self.start_xy_yaw = (
            tuple(float(v) for v in seed) if seed is not None else C.TASK2_SPAWN_XY_YAW
        )
        if table is not None:
            self.table = table
        elif self.profile.table is not None:
            self.table = self.profile.table.to_model()
        else:
            # Sim: the frozen Task 2 landmark. Rig without a measured table:
            # no landmark, so no vision — start seed + odometry only.
            self.table = None if rig is not None else TableModel()
        if localizer is not None:
            self.localizer = localizer
        else:
            self.localizer = TableLocalizer(self.table) if self.table is not None else None
        self.vision_enabled = self.localizer is not None
        self.walls = () if self.profile.walls == "none" else ROOM_WALLS
        goal = self.profile.goal_xy_yaw
        if rig is not None and goal is None:
            raise ValueError(
                "rig profile has no goal_xy_yaw: the parked base pose in the table "
                "frame has to be measured on site before the approach can drive "
                f"(profile {self.profile.source or '<in-memory>'})"
            )
        self._goal = tuple(float(v) for v in goal) if goal is not None else _GOAL_XY_YAW
        self.goal_xy_m = rig.goal_xy_m if rig is not None else _GOAL_XY_M
        self.goal_yaw_rad = rig.goal_yaw_rad if rig is not None else _GOAL_YAW_RAD
        self.continuous_twist = bool(rig is not None and rig.continuous_twist)
        self._max_linear = rig.max_linear_mps if rig is not None else _MAX_LINEAR_MPS
        self._max_angular = rig.max_angular_radps if rig is not None else _MAX_ANGULAR_RADPS
        self._kp_linear = rig.kp_linear if rig is not None else _PID_KP_LINEAR
        self._kp_yaw = rig.kp_yaw if rig is not None else _PID_KP_YAW
        self.needs_base_wire = rig is not None
        self._warned_no_odom = False
        self._warned_no_dump = False
        self.planner = planner or SplinePlanner()
        self.quantizer = quantizer or BaseQuantizer()
        self.dump_dir = Path(dump_dir) if dump_dir is not None else None
        if self.dump_dir is not None:
            self.dump_dir.mkdir(parents=True, exist_ok=True)
        # Dumps go to dump_dir/ep001, ep002, … — one subdirectory per run
        # that actually dumps (eval_batch reuses one controller across
        # episodes and reset() restarts the tick counter, so flat files
        # from episode N overwrote episode N-1).
        self._dump_runs = 0
        # Finegrained trim is a pose-only-nav stage: place_arms already keeps
        # vision alive here via align_edges, so there is nothing left for a
        # dedicated finegrained stage to correct (§1.5 of the design doc).
        kwargs.setdefault(
            "waypoints",
            (Waypoint("goal", *self._goal, self.goal_xy_m, self.goal_yaw_rad),),
        )
        if rig is not None:
            # Approach loop runs at APPROACH_RATE_HZ (20 Hz) on the rig.
            kwargs.setdefault("settle_ticks", max(1, int(round(rig.settle_s * 20.0))))
        super().__init__(
            *args,
            finegrained_start_position=False,
            base_only=bool(rig is not None and rig.base_only),
            **kwargs,
        )

    # -- lifecycle ---------------------------------------------------------

    def _reset_perception(self) -> None:
        self._spine_hold = 0
        self._dump_i = 0
        self._dump_episode_dir: Path | None = None
        self._last_t_sim: float | None = None
        self.filter = PoseFilter()
        self._est: EgoEstimate | None = None
        self._trails: dict[str, list[tuple[float, float]]] = {
            "gt": [],
            "odometry": [],
            "perception": [],
            "fused": [],
        }
        self._features = Features()
        self._spine_m = self._spine_assume_m
        self._last_state: np.ndarray | None = None
        self._odom_at_head: tuple[float, float, float] | None = None
        self._head_k: tuple[float, float, float, float] | None = None
        self._head_d: tuple[float, ...] = ()
        self._k_active: tuple[float, float, float, float] | None = None
        self._last_token = "NONE"
        self._last_twist: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.last_hud = ""
        self._path: tuple[PathSample, ...] = ()
        self._warned_rejects = False
        self._path_settle = 0
        self._rim_backoff_m = 0.0
        self._nav_image_shape: tuple[int, ...] | None = None
        self._last_head_t_sim: float | None = None
        self.stats["head_frames"] = 0
        self.stats["stale_head_ticks"] = 0
        self.stats["throttled_head_frames"] = 0
        self._pid_front = _AxisPID(self._kp_linear)
        self._pid_left = _AxisPID(self._kp_linear)
        self._pid_yaw = _AxisPID(self._kp_yaw)
        self._creep_coast = False
        self.quantizer.reset()
        self.stats["localize_residual_px"] = None

    def reset(self) -> None:
        super().reset()
        self._reset_perception()

    # -- reporting ---------------------------------------------------------

    def _path_target_index(self) -> int:
        """The sample to drive toward: ``self._path`` is refit fresh every
        tick (``_replan``) from wherever the fused pose is right now, so
        index 0 is always the current position (trivially "reached") and
        the live target is always the next one — ~50 cm ahead — except once
        that next sample is itself the final one, i.e. the caller's goal."""
        return min(1, len(self._path) - 1) if self._path else 0

    def target_label(self) -> str:
        if self.stage == "navigate":
            last = max(len(self._path) - 1, 0)
            return f"navigate/drive[{min(self._path_target_index(), last)}/{last}]"
        return super().target_label()

    def stage_hint(self, state: np.ndarray) -> str:
        if self.stage == "navigate":
            if self._est is None or not self._path:
                return "navigate/drive: no fused pose yet"
            index = self._path_target_index()
            target = self._path[index]
            x, y, yaw = self._est.xy_yaw
            return (
                f"navigate/drive: sample {index}/{len(self._path) - 1} "
                f"{math.hypot(target.x - x, target.y - y) * 1000:.0f} mm / "
                f"{wrap_angle(target.yaw - yaw):+.4f} rad to go"
            )
        return super().stage_hint(state)

    def pose_hud(self, state: np.ndarray | None = None) -> str:
        """Distance from the fused pose to the fixed goal constant. ``"unseeded"`` otherwise."""
        del state
        err = self._goal_err()
        if err is None:
            return "unseeded"
        dist, dyaw = err
        return f"goal_err=({dist * 1000:.0f}mm, {dyaw:+.4f} rad)"

    def _goal_err(self) -> tuple[float, float] | None:
        """``(dist_m, dyaw_rad)``: fused pose vs the fixed goal constant. None unseeded."""
        if self._est is None:
            return None
        gx, gy, gyaw = self._goal
        dist = math.hypot(gx - self._est.x, gy - self._est.y)
        return dist, wrap_angle(gyaw - self._est.yaw)

    # -- perception --------------------------------------------------------

    def localize(self, head: np.ndarray) -> EgoEstimate | None:
        """Features → gated absolute ego estimate, or ``None``."""
        if not self.vision_enabled:
            return None
        self._features = detect(head, surface_params=self.profile.surface)
        height, width = head.shape[:2]
        return self.localizer.solve(
            self._features,
            spine_m=self._spine_m,
            image_shape=head.shape,
            intrinsics=self._k(width, height),
        )

    def _gate_measurement(
        self, measurement: EgoEstimate | None, image_shape: tuple[int, ...]
    ) -> EgoEstimate | None:
        """Drop a candidate the live instance mask contradicts (border sliver).

        Independent of the odometry-vs-measurement mirror rejection the
        filter does one layer later (§1.2) — this is the one live check
        against the *current* frame, for both ``solve`` and ``align_edges``.
        """
        if measurement is None:
            return None
        height, width = image_shape[0], image_shape[1]
        agrees = self.localizer.instance_agrees(
            measurement.xy_yaw,
            self._features,
            spine_m=self._spine_m,
            image_shape=image_shape,
            intrinsics=self._k(width, height),
        )
        return measurement if agrees else None

    def _k(self, width: int, height: int) -> tuple[float, float, float, float]:
        """Intrinsics of the frame the pipeline sees (post-undistortion).

        Everything downstream of the delivered ``head`` image — localize,
        gate, dump/viz — must agree on which K it was drawn/projected in.
        ``_k_active`` is set in ``step()`` to whatever K undistort() just
        used as its new camera matrix (``k_out`` when a distorted profile
        is active, else the same K undistort was called with); falling back
        to ``_k_in`` covers every tick before the first image arrives.
        """
        if self._k_active is not None:
            return self._k_active
        return self._k_in(width, height)

    def _k_in(self, width: int, height: int) -> tuple[float, float, float, float]:
        """Intrinsics of the frame as delivered: CameraInfo, profile, or fallback."""
        if self._head_k is not None:
            return self._head_k
        if self.profile.camera is not None:
            return self.profile.camera.k
        return head_intrinsics(width, height)

    def _replan(self) -> None:
        """Fit fresh, every navigate tick, from wherever the fused pose is
        right now — not once at the start and then walked from a cache.

        A path fit from a stale pose is a path the robot isn't actually on:
        the moment odometry or a vision correction moves the fused pose,
        the old plan's clearance guarantees (§1.3) are about a route the
        robot no longer follows, and the ``--approach-dump`` / BEV
        visualisation of ``self._path`` shows a route that no longer
        matches where the robot is trying to go. Refitting every tick keeps
        both honest at the (small — see ``SplinePlanner``/``_dense_polyline``)
        cost of one more spline fit per tick.

        A rough seed is the whole point of ``--approach-start-xy-yaw``, and
        a rough seed (or a mid-route vision correction) can land inside a
        wall's clearance envelope — the planner cannot move its own
        endpoint, so it raises. That must not kill the episode: keep
        driving the last path that *did* fit (``self._path`` is left
        untouched) and try again next tick, by which time odometry or a
        vision correction may have moved the estimate somewhere feasible.
        """
        assert self._est is not None
        try:
            path = self.planner.plan(
                self._est.xy_yaw,
                self._goal,
                walls=self.walls,
                table=self.table,
                radius_m=BASE_RADIUS_M,
            )
        except ValueError as exc:
            self.stats["plan_failures"] = self.stats.get("plan_failures", 0) + 1
            log.warning("replan from %s failed: %s", format_xy_yaw(*self._est.xy_yaw), exc)
            return
        if not path:
            return
        self._path = path

    def _reset_pid(self) -> None:
        self._pid_front.reset()
        self._pid_left.reset()
        self._pid_yaw.reset()
        self._creep_coast = False

    def _nav_record(self) -> dict:
        """Current path sample index + distance from ``est`` to the goal constant."""
        err = self._goal_err()
        if self._est is None or not self._path:
            return {
                "wp": None,
                "wp_index": 0,
                "goal_dist_m": None if err is None else _num(err[0]),
                "goal_yaw_err_rad": None if err is None else _num(err[1]),
            }
        index = self._path_target_index()
        target = self._path[index]
        return {
            "wp": f"path{index}",
            "wp_index": index,
            "n_wp": len(self._path),
            "wp_xy_yaw": [_num(target.x), _num(target.y), _num(target.yaw)],
            "goal_dist_m": None if err is None else _num(err[0]),
            "goal_yaw_err_rad": None if err is None else _num(err[1]),
        }

    def _dt_s(self, t_sim: float | None) -> float:
        dt = _DEFAULT_DT_S
        if t_sim is not None and self._last_t_sim is not None:
            dt = max(0.0, min(_MAX_DT_S, float(t_sim) - self._last_t_sim))
        if t_sim is not None:
            self._last_t_sim = float(t_sim)
        return dt

    def _state_with_est(self, state: np.ndarray) -> np.ndarray:
        """Inherited stages read odom internally — hand them the estimate."""
        out = np.array(state, copy=True)
        if self._est is not None:
            out[C.S_BASE_ODOM] = self._est.xy_yaw
        return out

    # -- dump --------------------------------------------------------------

    def _gt_xy_yaw(
        self, state: np.ndarray | None = None
    ) -> tuple[float, float, float] | None:
        """Odom pose that took the current head frame. Display only — never control.

        Head RGB is ~1.6 Hz; ``S_BASE_ODOM`` is the latest sample (~50 Hz).
        Projecting that onto a stale image shears the GT box by tens of
        pixels. Prefer the pose interpolated at the image stamp; fall back
        to the tick's state.
        """
        if self._odom_at_head is not None:
            x, y, yaw = (float(v) for v in self._odom_at_head)
            if all(math.isfinite(v) for v in (x, y, yaw)):
                return x, y, yaw
        src = self._last_state if state is None else state
        if src is None:
            return None
        x, y, yaw = (float(v) for v in src[C.S_BASE_ODOM])
        if not all(math.isfinite(v) for v in (x, y, yaw)):
            return None
        return x, y, yaw

    def _channel_poses(self) -> dict:
        """The §1.2 channels present this tick, keyed like ``viz.CHANNELS``.

        Vision-derived channels are handed over as the ``EgoEstimate``, not
        as a bare triple: ``viz`` normalises either, but only the estimate
        carries ``inlier_uv``, and dropping it silently costs the overlay
        its "corner used in fit" dots while the legend still promises them.
        """
        channels = self.filter.channels
        raw = {
            "gt": self._gt_xy_yaw(),
            "odometry": None if channels.odometry is None else channels.odometry.xy_yaw,
            "perception": channels.perception,
            "fused": self._est,
        }
        return {name: pose for name, pose in raw.items() if pose is not None}

    @staticmethod
    def _as_xy_yaw(pose) -> tuple[float, float, float]:
        return tuple(getattr(pose, "xy_yaw", pose))

    def _append_trails(self) -> None:
        """One (x, y, yaw) per pose channel per tick, from the first navigate tick.

        Generalises the old single ``_trail`` (§2, §3.4): GT, odometry, and
        perception can each be absent on a given tick (no image, no lock);
        fused is present on every tick once seeded.
        """
        if self._est is None:
            return
        for name, pose in self._channel_poses().items():
            trail = self._trails[name]
            if len(trail) < _TRAIL_MAX:
                trail.append(self._as_xy_yaw(pose))

    def dump(self, head: np.ndarray | None) -> None:
        """Refresh the HUD; write the composite and a JSON sidecar."""
        spine = self._spine_m
        spine_s = f"{spine:.3f}" if math.isfinite(spine) else "n/a"
        self.last_hud = (
            f"{self.target_label()}  spine={spine_s}  "
            f"{self.pose_hud()}  {self._last_token}"
        )
        if self.dump_dir is None or head is None:
            return
        if self.table is None:
            if not self._warned_no_dump:
                self._warned_no_dump = True
                log.warning("--approach-dump: no table model in the profile, overlays skipped")
            return
        rgb = np.ascontiguousarray(head[:, :, :3], dtype=np.uint8)
        poses = self._channel_poses()
        height, width = rgb.shape[:2]
        k = self._k(width, height)
        hfov = hfov_deg_from_intrinsics(k, width)
        annotated = viz.draw_head_overlay(
            rgb,
            self._features,
            table=self.table,
            poses=poses,
            spine_m=self._spine_m,
            hud=self.last_hud,
            intrinsics=k,
            walls=ROOM_WALLS,
            path=self._path,
            cmd=self._last_twist,
        )
        bev = viz.draw_bev(
            height=annotated.size[1],
            table=self.table,
            waypoints=self.waypoints,
            wp_index=len(self.waypoints) - 1,
            poses=poses,
            trails=self._trails,
            walls=ROOM_WALLS,
            path=self._path,
            hfov_deg=hfov,
            cmd=self._last_twist,
        )
        phase = self.target_label().replace("/", "_")
        stem = f"{self._dump_i:06d}_{phase}_{self._last_token}"
        self._dump_i += 1
        # Default PNG compression costs ~400 ms on a 2400x720 composite and
        # this runs every tick; level 1 is 3.6x faster for 12 % more bytes.
        if self._dump_episode_dir is None:
            self._dump_runs += 1
            self._dump_episode_dir = self.dump_dir / f"ep{self._dump_runs:03d}"
            self._dump_episode_dir.mkdir(parents=True, exist_ok=True)
        out = self._dump_episode_dir
        viz.composite(annotated, bev).save(out / f"{stem}.png", compress_level=1)
        (out / f"{stem}.json").write_text(
            json.dumps(self._record(rgb.shape), indent=2) + "\n"
        )

    def _record(self, image_shape: tuple[int, ...]) -> dict:
        """Bbox + detections for this tick — the PNG is not enough to debug.

        ``est`` is the fused pose; ``perception``/``odometry`` are the other
        two filter channels (§1.2); ``gt`` is odom, display-only. ``rejected``
        is the filter's mirror-lock veto count so it is measurable, not
        invisible (§1.2).
        """
        height, width = int(image_shape[0]), int(image_shape[1])
        table = self.table
        world = table.corners()
        width_m, depth_m, height_m = table.metrics(world)
        bbox = {
            "world": [[_num(v) for v in row] for row in world],
            "uv": [None] * 8,
            "gt_uv": [None] * 8,
            "size_xy": [round(width_m, 4), round(depth_m, 4)],
            "height_m": round(height_m, 4),
            "model_size_xy": [float(table.size_xy[0]), float(table.size_xy[1])],
            "model_height_m": float(table.height_m),
        }
        intrinsics = self._k(width, height)
        if self._est is not None:
            t_world_cam = T_world_base(*self._est.xy_yaw) @ T_base_cam(self._spine_m)
            bbox["uv"] = [_uv(project(row, t_world_cam, intrinsics)) for row in world]
        gt_pose = self._gt_xy_yaw()
        if gt_pose is not None:
            t_gt = T_world_base(*gt_pose) @ T_base_cam(self._spine_m)
            bbox["gt_uv"] = [_uv(project(row, t_gt, intrinsics)) for row in world]
        features = self._features
        surface = getattr(features, "surface", None)
        instance: dict = {"pixels": 0, "bbox_uv": None}
        if surface is not None:
            ys, xs = np.nonzero(surface > 0)
            instance["pixels"] = int(xs.size)
            if xs.size:
                instance["bbox_uv"] = [
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max()),
                    int(ys.max()),
                ]
        if self._est is not None:
            instance["world_size_xy"] = (
                None
                if self._est.instance_size_xy is None
                else [_num(v) for v in self._est.instance_size_xy]
            )
            instance["fits_table"] = self._est.instance_fits_table
            live = self.localizer.overlap_scores(
                self._est.xy_yaw,
                features,
                spine_m=self._spine_m,
                image_shape=image_shape,
                intrinsics=intrinsics,
            )
            if live is not None:
                instance["tabletop_iou"] = _num(live[0])
                instance["tabletop_recall"] = _num(live[1])
            else:
                instance["tabletop_iou"] = None
                instance["tabletop_recall"] = None
        gt = None if gt_pose is None else [_num(v) for v in gt_pose]
        channels = self.filter.channels
        rejected = self.filter.rejected
        return {
            "tick": self._dump_i - 1,
            "phase": self.target_label(),
            "token": self._last_token,
            "cmd": [_num(v) for v in self._last_twist],
            "spine_m": _num(self._spine_m),
            "gt": gt,
            "gt_from": "image" if self._odom_at_head is not None else "latest_odom",
            "intrinsics": [float(v) for v in self._k(width, height)],
            "hfov_deg": round(hfov_deg_from_intrinsics(self._k(width, height), width), 3),
            "est": _est_record(self._est),
            "perception": _est_record(channels.perception),
            "odometry": _est_record(channels.odometry),
            "rejected": {
                "count": len(rejected),
                "poses": [[_num(e.x), _num(e.y), _num(e.yaw)] for e in rejected],
            },
            "nav": self._nav_record(),
            "bbox": bbox,
            "instance": instance,
            "features": {
                "corners": [
                    {
                        "u": _num(c.u),
                        "v": _num(c.v),
                        "kind": c.kind,
                        "support": _num(c.support),
                    }
                    for c in features.corners
                ],
                "outline": [
                    {
                        "x0": _num(s.x0),
                        "y0": _num(s.y0),
                        "x1": _num(s.x1),
                        "y1": _num(s.y1),
                    }
                    for s in features.outline
                ],
                "n_segments": len(features.segments),
            },
        }

    # -- FSM ---------------------------------------------------------------

    def step(
        self,
        state: np.ndarray,
        *,
        odom_n: int | None = None,
        t_sim: float | None = None,
        images: dict | None = None,
        odom_at_head: tuple[float, float, float] | None = None,
        head_k: tuple[float, float, float, float] | None = None,
        head_d: tuple[float, ...] = (),
        head_t_sim: float | None = None,
    ) -> Command:
        """Spine hold, then drive on the fused pose from the first navigate tick.

        ``head_t_sim`` is the head frame's own stamp. The collector hands out
        the *latest* frame on every tick. In sim the Isaac head camera runs
        at ~2 Hz under a 20 Hz loop, so the same image would otherwise be
        detected, solved and fused ~10 times — and re-fusing one measurement
        compounds its gain. A stamp already seen makes this tick vision-less
        (odometry predict only). On the rig the ZED delivers ~20 Hz and
        detect+solve costs ~40–50 ms per real frame, the whole tick budget:
        ``vision_min_period_s`` caps vision to one frame per period. An
        unstamped frame (``None``) keeps the old every-tick behaviour.

        ``odom_at_head`` is odom interpolated at the head image stamp. The
        GT overlay uses it so a stale frame is not projected with a newer
        pose. Control never reads it.

        ``head_k`` is live CameraInfo ``(fx, fy, cx, cy)``. The yaml 90°×60°
        is the real ZED Mini, not the Isaac Camera prim; without this the
        GT box is drawn too small.
        """
        self._last_state = np.asarray(state)
        self._odom_at_head = odom_at_head
        self._head_k = head_k
        cam = self.profile.camera
        # Live CameraInfo distortion wins when it exists; otherwise fall
        # back to the camera profile's self-calibrated D (e.g. the Munich
        # rig, which publishes no camera_info at all).
        if head_d:
            self._head_d = tuple(head_d)
        elif head_k is None and cam is not None:
            self._head_d = tuple(cam.dist)
        else:
            self._head_d = ()
        spine = float(state[C.S_SPINE])
        # contracts.py's resolve_joint(measured, SPINE_JOINT, 0.0) returns a
        # FINITE 0.0 default when the joint is simply absent (no
        # /spine/joint_states at all, e.g. this bag) — math.isfinite alone
        # cannot tell that apart from a genuine reading, and 0.0 is not a
        # physically plausible spine height (SOP range sits near 0.45-0.55
        # m), so treat it as "no measurement" too.
        spine_measured = math.isfinite(spine) and spine > 0.0
        if self.profile.spine_m is not None:
            # Rig: the spine is at a fixed, measured height and the state
            # carries none at all — the profile is the only source.
            self._spine_m = float(self.profile.spine_m)
        else:
            self._spine_m = spine if spine_measured else self._spine_assume_m
        self.stats["final_spine_m"] = spine if spine_measured else None
        head = None if images is None else images.get("head")
        if head is not None:
            head = self._fresh_head(head, head_t_sim)
        if head is not None:
            height, width = head.shape[:2]
            k_in = self._k_in(width, height)
            k_out = k_in
            if head_k is None and cam is not None and self._head_d:
                k_out = cam.k_out
            head = undistort(head, k_in, self._head_d, new_intrinsics=k_out)
            # Everything downstream (localize, gate, dump/viz) must agree on
            # which K the delivered image is actually in — _k() returns
            # this until the next tick's undistort() call replaces it.
            self._k_active = k_out if self._head_d else None

        if self.stage == "spine":
            return self._finish(self._step_spine_perception(state), head)
        if self.stage != "navigate":
            self._refresh_est(state, head)
            cmd = super().step(self._state_with_est(state), odom_n=odom_n, t_sim=t_sim)
            return self._finish(cmd, head)
        return self._finish(self._step_navigate_perception(state, head, t_sim), head)

    def _fresh_head(self, head: np.ndarray, t_sim: float | None) -> np.ndarray | None:
        """``head`` if its stamp is new (or absent) and not throttled, else None."""
        if t_sim is not None and math.isfinite(t_sim):
            last = self._last_head_t_sim
            if last is not None and t_sim <= last:
                self.stats["stale_head_ticks"] += 1
                return None
            if last is not None and t_sim - last < self.vision_min_period_s:
                self.stats["throttled_head_frames"] += 1
                return None
            self._last_head_t_sim = t_sim
        self.stats["head_frames"] += 1
        return head

    def _expected_xy_yaw(self) -> tuple[float, float, float]:
        if self.waypoints:
            goal = self.waypoints[-1]
            return (goal.x, goal.y, goal.yaw)
        return self._goal

    def _refresh_est(self, state: np.ndarray, head: np.ndarray | None) -> bool:
        """Odometry predicts every tick; edge-align corrects when vision agrees.

        Used by ``place_arms`` — navigate has its own version that also
        seeds the filter and (re)plans. Once ``align_edges`` accepts a fit,
        the two single-axis rim refinements sharpen it further (see
        ``_refine_with_rims``): at close range, matched sides alone can
        leave a family of poses rather than pinning down one, and a visible
        rim fixes exactly the axis align_edges is still loose on. Returns
        True on a tick that actually applied a visual correction.
        """
        odom_now = tuple(float(v) for v in state[C.S_BASE_ODOM])
        if not all(math.isfinite(v) for v in odom_now):
            return False
        if head is None:
            self._est = self.filter.update(odom_xy_yaw=odom_now)
            return False
        if not self.vision_enabled:
            # No landmark (rig without a measured table): odometry only.
            self._est = self.filter.update(odom_xy_yaw=odom_now)
            return False
        self._features = detect(head, surface_params=self.profile.surface)
        height, width = head.shape[:2]
        intrinsics = self._k(width, height)
        estimate = self.localizer.align_edges(
            self._features,
            expected=self._expected_xy_yaw(),
            seed=None if self._est is None else self._est.xy_yaw,
            spine_m=self._spine_m,
            image_shape=head.shape,
            intrinsics=intrinsics,
        )
        estimate = self._gate_measurement(estimate, head.shape)
        if estimate is not None:
            estimate = self._refine_with_rims(estimate, head.shape, intrinsics)
        self.stats["localize_residual_px"] = None if estimate is None else estimate.residual_px
        self._est = self.filter.update(odom_xy_yaw=odom_now, measurement=estimate)
        return estimate is not None

    # x (lateral_from_vertical_rim) is off by default; y always runs. Flip
    # this on only once x has its own measured justification the way y's
    # rear-edge-visibility case has — see _refine_with_rims.
    _REFINE_LATERAL_RIM = False

    def _refine_with_rims(
        self,
        estimate: EgoEstimate,
        image_shape: tuple[int, ...],
        intrinsics: tuple[float, float, float, float],
    ) -> EgoEstimate:
        """Sharpen an accepted ``align_edges`` fit's y from the rear rim.

        Only runs on a fit ``align_edges`` already accepted this tick — an
        unanchored rim nudge (no fresh edge match to seed from) would just
        accumulate its own drift instead of refining a real fit. No rim in
        frame simply leaves the estimate as ``align_edges`` found it.
        ``estimate``'s own evidence fields (n_corr, residual_px,
        tabletop_iou) are untouched, so the filter's measurement gain is
        unaffected by this — only x/y/yaw move, and the result is free to
        land past whatever the caller expected (a nominal goal, not a wall).
        """
        xy_yaw = estimate.xy_yaw
        if self._REFINE_LATERAL_RIM:
            lateral = self.localizer.lateral_from_vertical_rim(
                self._features,
                seed=xy_yaw,
                spine_m=self._spine_m,
                image_shape=image_shape,
                intrinsics=intrinsics,
            )
            if lateral is not None:
                xy_yaw = lateral
        forward = self.localizer.forward_from_horizontal_rim(
            self._features,
            seed=xy_yaw,
            spine_m=self._spine_m,
            image_shape=image_shape,
            intrinsics=intrinsics,
        )
        if forward is not None:
            xy_yaw = forward
        if xy_yaw == estimate.xy_yaw:
            return estimate
        x, y, yaw = xy_yaw
        return replace(estimate, x=x, y=y, yaw=yaw)

    def _finish(self, cmd: Command, head: np.ndarray | None) -> Command:
        self._last_token = cmd.base_token
        self._last_twist = cmd.base_twist
        self._append_trails()
        self.dump(head)
        return cmd

    def _step_spine_perception(self, state: np.ndarray) -> Command:
        self._ensure_arm_cmd(state)
        self.stats["spine_ticks"] += 1
        spine = float(state[C.S_SPINE])
        if math.isfinite(spine) and spine >= self.spine_measured_min_m:
            self._spine_hold += 1
        else:
            self._spine_hold = 0
        if self._spine_hold >= self.spine_hold_ticks:
            self._advance_stage()
        return self._command("NONE")

    def _step_navigate_perception(
        self, state: np.ndarray, head: np.ndarray | None, t_sim: float | None
    ) -> Command:
        dt = self._dt_s(t_sim)
        left = np.asarray(state[C.S_LEFT_ARM], dtype=np.float32)
        right = np.asarray(state[C.S_RIGHT_ARM], dtype=np.float32)
        if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            return Command(
                left_arm=np.zeros(7, dtype=np.float32),
                right_arm=np.zeros(7, dtype=np.float32),
                left_gripper=self.gripper_open,
                right_gripper=self.gripper_open,
                base_twist=(0.0, 0.0, 0.0),
                base_token="NONE",
            )
        self._ensure_arm_cmd(state)

        # Reset every tick, not just on a fresh frame: sparse vision means
        # most ticks have none (``step`` drops a head frame whose stamp it
        # has already processed), and _rear_rim_visible must read that as
        # "nothing fresh to check", not stale leftovers from ticks ago.
        self._nav_image_shape = None
        odom_now = tuple(float(v) for v in state[C.S_BASE_ODOM])
        if not all(math.isfinite(v) for v in odom_now) and not self._warned_no_odom:
            self._warned_no_odom = True
            log.warning(
                "navigate: no base odometry in the state (S_BASE_ODOM not finite) — "
                "holding still. On the rig: is start_base running and "
                "/swerve_drive_controller/odom publishing?"
            )
        if all(math.isfinite(v) for v in odom_now):
            if self._est is None:
                # First navigate tick: seed the filter with the rough start
                # pose so a fused estimate — and a plan — exist before
                # vision ever locks (the regression this file exists to fix).
                self.filter.seed(self.start_xy_yaw, odom_now)
            measurement = None
            if head is not None and self.vision_enabled:
                self._nav_image_shape = head.shape
                measurement = self._gate_measurement(self.localize(head), head.shape)
            self._est = self.filter.update(odom_xy_yaw=odom_now, measurement=measurement)
            self.stats["localize_residual_px"] = (
                None if measurement is None else measurement.residual_px
            )
            # A seed whose yaw is wrong by more than the filter's reject
            # threshold makes EVERY true lock look like a mirror twin, and
            # the run then dead-reckons from a bad seed forever while solve
            # keeps returning the truth. Silent is the wrong failure mode:
            # say it once, out loud, not only in the --approach-dump JSON.
            rejected = len(self.filter.rejected)
            self.stats["rejected_locks"] = rejected
            if (
                rejected >= _REJECT_WARN_N
                and self.filter.channels.perception is None
                and not self._warned_rejects
            ):
                self._warned_rejects = True
                log.warning(
                    "%d locks rejected and none ever accepted — is "
                    "--approach-start-xy-yaw yaw wrong by more than the "
                    "filter's reject threshold? driving on odometry alone",
                    rejected,
                )

        self.stats["nav_ticks"] += 1
        if self._est is not None:
            self._replan()
        if self._est is None or not self._path:
            return self._twist_command(0.0, 0.0, 0.0)

        vx, vy, wz = (float(v) for v in state[C.S_BASE_VEL])
        if not all(math.isfinite(v) for v in (vx, vy, wz)):
            vx = vy = wz = 0.0
        return self._drive_path(vx, vy, wz, dt)

    def _drive_path(self, vx: float, vy: float, wz: float, dt: float) -> Command:
        """PID on body-frame error to the path's live target.

        ``self._path`` is refit fresh every tick, always starting exactly
        at the current pose (``_replan``), so there is no persisted via
        index to advance: the target is always ``_path_target_index()`` —
        the next sample ahead, which only settles into the caller's fixed
        goal once the fit is down to its final leg. Arrival (tight
        tolerance, settle-and-advance-stage) only applies on that final
        leg; short of it, the base just keeps driving toward whatever the
        live fit currently says is ~50 cm ahead — there is no discrete
        "reached this via" event to gate on when the via itself moves
        every tick.
        """
        assert self._est is not None
        if not self._path:
            return self._twist_command(0.0, 0.0, 0.0)
        x, y, yaw = self._est.xy_yaw
        index = self._path_target_index()
        last = index == len(self._path) - 1
        target = self._path[index]
        if last and self._rim_backoff_m:
            # Push the final target away from the table by however much the
            # rear-rim search below has retreated so far — the goal is
            # nominal, not a wall to stop short of.
            target = replace(target, y=target.y + self._rim_backoff_m)
        front, left, yaw_err = body_frame_errors(x, y, yaw, target.x, target.y, target.yaw)
        if last:
            sample_wp = Waypoint(
                "path", target.x, target.y, target.yaw, self.goal_xy_m, self.goal_yaw_rad
            )
            if self._waypoint_ok(sample_wp, front, left, yaw_err, vx, vy, wz, require_stopped=True):
                visible = self._rear_rim_visible()
                if visible is False and self._rim_backoff_m < _REAR_RIM_BACKOFF_MAX_M:
                    self._rim_backoff_m = min(
                        self._rim_backoff_m + _REAR_RIM_BACKOFF_STEP_M, _REAR_RIM_BACKOFF_MAX_M
                    )
                    self._reset_pid()
                    self._path_settle = 0
                    return self._twist_command(0.0, 0.0, 0.0)
                self._reset_pid()
                self._path_settle += 1
                if self._path_settle >= self.settle_ticks:
                    self._path_settle = 0
                    gx, gy, gyaw = self._goal
                    self.stats["final_xy_err_m"] = math.hypot(gx - x, gy - y)
                    self.stats["final_yaw_err_rad"] = wrap_angle(gyaw - yaw)
                    self._advance_stage()
                return self._twist_command(0.0, 0.0, 0.0)
        self._path_settle = 0
        wz_cap = min(_yaw_speed_cap(abs(yaw_err)), self._max_angular)
        wz = max(-wz_cap, min(wz_cap, self._pid_yaw.step(yaw_err, dt)))
        cmd = self._twist_command(
            self._pid_front.step(front, dt),
            self._pid_left.step(left, dt),
            wz,
        )
        if self.continuous_twist:
            return cmd  # the twist tapers with the error; nothing to pulse
        return self._creep_gate(cmd, front, left, yaw_err)

    def _rear_rim_visible(self) -> bool | None:
        """Is the rear table edge in this tick's frame? ``None`` when there
        is nothing fresh to check (sparse vision — most ticks have no new
        head frame at all), which callers should treat as "don't know,
        don't block on it" rather than as a failure.
        """
        if self._nav_image_shape is None or self._est is None:
            return None
        height, width = self._nav_image_shape[0], self._nav_image_shape[1]
        pose = self.localizer.forward_from_horizontal_rim(
            self._features,
            seed=self._est.xy_yaw,
            spine_m=self._spine_m,
            image_shape=self._nav_image_shape,
            intrinsics=self._k(width, height),
        )
        return pose is not None

    def _creep_gate(self, cmd: Command, front: float, left: float, yaw_err: float) -> Command:
        """Near the target, pulse the SIM token one tick on / one tick off.

        Only overrides ``base_token``: the real wire's ``base_twist`` is a
        genuine continuous twist with no quantizer, so it already tapers
        toward zero with the error and gets none of this.
        """
        axis_err = {
            "FWD": front, "BACK": front,
            "A": left, "B": left,
            "A+C": yaw_err, "B+C": yaw_err,
        }.get(cmd.base_token)
        near = axis_err is not None and abs(axis_err) <= _NEAR_M
        if not near:
            self._creep_coast = False
            return cmd
        coast, self._creep_coast = self._creep_coast, not self._creep_coast
        return replace(cmd, base_token="NONE") if coast else cmd

    def _twist_command(self, vx: float, vy: float, wz: float) -> Command:
        vx, vy, wz = _clip_twist(vx, vy, wz, self._max_linear, self._max_angular)
        # Rig: the continuous twist IS the wire; no pedal token, no quantizer.
        token = "NONE" if self.continuous_twist else self.quantizer.quantize(vx, vy, wz)
        assert self._left_cmd is not None and self._right_cmd is not None
        left_grip, right_grip = self._gripper_hold()
        return Command(
            left_arm=self._left_cmd.astype(np.float32),
            right_arm=self._right_cmd.astype(np.float32),
            left_gripper=left_grip,
            right_gripper=right_grip,
            base_twist=(vx, vy, wz),
            base_token=token,
        )

    def _gripper_hold(self) -> tuple[float, float]:
        """Sim opens the grippers for the approach; the rig holds them as found."""
        if self.rig is None or self._last_state is None:
            return self.gripper_open, self.gripper_open
        left = float(self._last_state[C.S_LEFT_GRIP])
        right = float(self._last_state[C.S_RIGHT_GRIP])
        return (
            left if math.isfinite(left) else self.gripper_open,
            right if math.isfinite(right) else self.gripper_open,
        )
