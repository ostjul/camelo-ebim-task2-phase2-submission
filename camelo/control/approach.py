"""Task 2 approach FSM: walk stages, drive waypoints, place arms, optional trim.

Default stages: spine → navigate → place_arms → finegrained_start_position →
start_pose → done. Spine runs first so a scene reset (eval) can finish its
pin-ramp before the base moves — matching a normal ``run-policy`` session
where the spine is already settled. Navigation holds arms as-found;
``place_arms`` walks ``TASK2_ARM_PLACE_PATH`` arm by arm and ends at the
dataset ready poses (``TASK2_ARM_READY_LEFT/RIGHT``). The optional
``finegrained_start_position`` stage (``PulseSettleTrim``) trims the base
after the arms are out with engage/coast/brake, then an in-zone dwell —
disable with ``ApproachController(finegrained_start_position=False)``.

The spine stage only *verifies*: there is no ROS topic that commands the
spine (PLAN.md; the benchmark's ``SpineKeyboardController`` owns it), so the
height is pinned at sim launch with ``--spine-keyboard-min/max`` (F-50) and
this FSM waits (sim-time, under ``--approach-timeout``) for measured height
to clear the gate after every reset. Measured sits ~15 mm under commanded at
steady state (F-57 droop), which is why the gate threshold is on the measured
value and looser than the SOP.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace

import numpy as np

from camelo import contracts as C
from camelo.control.chunk_executor import Command

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Waypoints / stages (authoritative numbers — CONTRACTS.md points here)
# ---------------------------------------------------------------------------

AXES = ("left", "front", "yaw")
# Arrival tolerance per joint.
_ARM_TOL = math.radians(2.0)

# Navigate goal tolerances (bang-bang). Coarse waypoints are looser ("close
# enough, move on"). Super-fine gates live on PulseSettleTrim (F-93).
_COARSE_XY, _COARSE_YAW = 0.1, 0.1
_FINE_XY, _FINE_YAW = 0.05, 0.05
# Post-arm engage/coast/brake targets. No timeout-accept; episode
# --approach-timeout is the backstop.
#
# The linear gate must stay ABOVE the smallest motion the drive will make,
# or the band is unreachable by a pulse and the trim limit-cycles until the
# approach timeout fires. The floor hold is 20 ms and the pedal token
# commands PEDAL_LINEAR_SPEED (0.5 m/s), so the smallest commandable step is
# ~10 mm — twice the old 5 mm gate. Measured in eval 20260816_105253/ep1
# from the benchmark spawn: 7085 fine ticks, 82 direction reversals, 1.315 m
# of travel to net 0.082 m (16x), inside the band on only 9.9 % of ticks and
# never long enough to finish the stopped dwell; timed out 30 mm from goal.
# 0.012 m clears the quantum by 20 %. Yaw is unchanged: its quantum
# (A+C ~0.000175 rad per burst) sits well under the 5 mrad gate.
_SUPER_FINE_XY, _SUPER_FINE_YAW = 0.012, 0.005
# hold_s = clip(k * |err| / v, floor, cap). v is the F-93 50 ms quantum rate
# (lin ~0.5 mm, yaw A+C ~0.000175 rad). Holds below ~20 ms never engage the
# swerve. Cap is that same 50 ms quantum: 300 ms bursts catch after steering
# aligns and overshoot a 0.001 rad gate by 10–80× (live 2026-08-13).
_PULSE_V_LIN_MPS = 0.0005 / 0.05
_PULSE_V_YAW_RADPS = 0.000175 / 0.05
_PULSE_HOLD_FLOOR_S = 0.02
_PULSE_HOLD_CAP_S = 0.05
_PULSE_K = 1.0 / 3.0
# When t_sim is omitted (unit tests), hold at least this many ticks.
_PULSE_HOLD_MIN_TICKS = 3
# In-zone + stopped dwell before start_pose (~2× F-93 72 ms settle).
_DWELL_S = 0.15
_DWELL_LEAVE = 1.25  # abort dwell if any |err| exceeds this × gate
_DWELL_MIN_TICKS = 8  # when t_sim is omitted
# |v| above this while still on a locked axis is F-93 catch, not a clean pulse.
# Clean 50 ms yaw peaks near the stop gate; live catch hit ~0.11 rad/s.
_CATCH_LIN_MPS = 0.05
_CATCH_YAW_RADPS = 0.04

_LIN_STOP_MPS = 0.02  # Body-frame linear speed below which the base counts as stopped.
_YAW_STOP_RADPS = 0.015  # Body-frame angular speed below which the base counts as stopped.
_STOP_S = 0.08  # release→stop; median ~72 ms measured 2026-08-12 (F-93)

STAGE_FINE = "finegrained_start_position"


def build_stages(
    *,
    finegrained_start_position: bool = True,
    skip_spine: bool = False,
    base_only: bool = False,
) -> tuple[str, ...]:
    """Assemble the approach stage list; fine trim is opt-in/out in one place.

    ``base_only`` (real robot) is the strongest cut: navigate → done. No
    spine hold (the rig's spine state resolves to a finite 0.0 and its
    height is fixed), and no arm stages — arms are activated and parked by
    the T6 flow afterwards.

    ``skip_spine`` drops "spine" from the list entirely (not just a gate
    that passes instantly): ``reset()`` sets ``self.stage = self.stages[0]``,
    so removing it here is what actually makes the FSM start at the next
    stage — mutating ``self.stage`` after construction does not survive
    ``run_approach``'s own ``approach.reset()`` call at the top of the loop.
    """
    if base_only:
        return ("navigate", "done")
    stages = ["navigate", "place_arms"] if skip_spine else ["spine", "navigate", "place_arms"]
    if finegrained_start_position:
        stages.append(STAGE_FINE)
    stages.extend(["start_pose", "done"])
    return tuple(stages)


STAGES: tuple[str, ...] = build_stages(finegrained_start_position=True)


@dataclass(frozen=True)
class Waypoint:
    name: str
    x: float
    y: float
    yaw: float
    tol_xy_m: float
    tol_yaw_rad: float
    axes: tuple[str, ...] = AXES


TASK2_APPROACH_WAYPOINTS: tuple[Waypoint, ...] = (
    Waypoint("wp1", 4.4, 2.9, math.radians(-90.0), _COARSE_XY, _COARSE_YAW),
    Waypoint("wp2", 3.5, 2.9, math.radians(-90.0), _COARSE_XY, _COARSE_YAW),
    Waypoint("wp3", 3.5, 3.05, math.radians(-90.0), _COARSE_XY, _COARSE_YAW),
    # Waypoint("goal", 2.10, 3.04, math.radians(-90.0), _FINE_XY, _FINE_YAW),
    # Changed from 3.05 to 3.04 as the arm is always colliding with the holder
    Waypoint("goal", 2.10, 3.05, math.radians(-90.0), _FINE_XY, _FINE_YAW),
)

_ARM_SETTLE_EPS = math.radians(0.2)


@dataclass(frozen=True)
class ArmWaypoint:
    """One leg of the arm placement path — absolute joint targets in rad.

    ``settle_ticks=None`` takes the controller's ``arm_settle_ticks``; a leg
    that is only passed through wants a small dwell and a loose ``tol_rad``,
    the final one wants both tight.
    """

    name: str
    left: tuple[float, ...]
    right: tuple[float, ...]
    tol_rad: float = _ARM_TOL
    settle_ticks: int | None = None


_VIA_TOL = math.radians(5.0)  # legs to pass through, not to land on
_VIA_SETTLE_TICKS = 5

TASK2_ARM_VIA_LEGS: tuple[tuple[str, tuple[float, ...], tuple[float, ...]], ...] = (
    (
        "lower1",
        (-0.0247, -0.8693, 0.2680, -2.5805, 0.2025, 1.7197, 0.9012),
        (-0.1911, -0.8387, -0.1089, -2.5856, -0.0814, 1.7409, 0.5388),
    ),
    (
        "lower2",
        (0.0333, -0.8185, 0.5698, -2.6983, 0.4317, 1.9282, 1.0662),
        (-0.3342, -0.7241, -0.3062, -2.7138, -0.2188, 1.9975, 0.3162),
    ),
    (
        "lower3",
        (0.0075, -0.5828, 0.9283, -2.6697, 0.5766, 2.2038, 1.2561),
        (-0.4081, -0.4187, -0.5336, -2.6931, -0.2747, 2.3024, 0.0863),
    ),
)


def default_arm_path(ready_left, ready_right) -> tuple[ArmWaypoint, ...]:
    """The demo descent legs, then ``ready_*``."""
    via = tuple(
        ArmWaypoint(name, left, right, tol_rad=_VIA_TOL, settle_ticks=_VIA_SETTLE_TICKS)
        for name, left, right in TASK2_ARM_VIA_LEGS
    )
    return (*via, ArmWaypoint("ready", tuple(ready_left), tuple(ready_right)))


TASK2_ARM_PLACE_PATH: tuple[ArmWaypoint, ...] = default_arm_path(
    C.TASK2_ARM_READY_LEFT, C.TASK2_ARM_READY_RIGHT
)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def wrap_angle(rad: float) -> float:
    return (rad + math.pi) % (2.0 * math.pi) - math.pi


def body_frame_errors(
    x: float, y: float, yaw: float, goal_x: float, goal_y: float, goal_yaw: float
) -> tuple[float, float, float]:
    dx, dy = goal_x - x, goal_y - y
    c, s = math.cos(yaw), math.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy, wrap_angle(goal_yaw - yaw)


def _err(axis: str, front: float, left: float, yaw_err: float) -> float:
    return {"left": left, "front": front, "yaw": yaw_err}[axis]


def _tol(axis: str, wp: Waypoint) -> float:
    return wp.tol_yaw_rad if axis == "yaw" else wp.tol_xy_m


def _vel(axis: str, vx: float, vy: float, wz: float) -> float:
    return {"left": vy, "front": vx, "yaw": wz}[axis]


def _token(axis: str, sign: float) -> str:
    if sign == 0.0:
        return "NONE"
    return {
        "left": ("A", "B"),
        "front": ("FWD", "BACK"),
        "yaw": ("A+C", "B+C"),
    }[axis][0 if sign > 0 else 1]


# ---------------------------------------------------------------------------
# Pedal drive
# ---------------------------------------------------------------------------


class AxisDrive:
    """Bang-bang drive for the fixed-speed pedal (one token per tick).

    The pedal is full speed or nothing, so the time-optimal law per axis is
    to drive flat out until releasing now would already coast it inside
    tolerance. Only one token fits on the wire per tick; across axes that
    still want to move, arbitration picks the largest error normalised by
    its own tolerance, so the leading axis is driven down and the others
    naturally take over without any accumulated state.
    """

    def __init__(self, stop_s: float = _STOP_S):
        self.stop_s = stop_s
        self.last_duty = 0.0

    def reset(self) -> None:
        self.last_duty = 0.0

    def _wants(self, err: float, tol: float, body_vel: float) -> bool:
        """Drive flat out unless releasing now already coasts into tol."""
        if abs(err) <= tol:
            return False
        closing = body_vel * math.copysign(1.0, err)
        if closing <= 0.0:
            return True
        return abs(err) - 0.5 * closing * self.stop_s > tol

    def step(
        self,
        axes: tuple[str, ...],
        front: float,
        left: float,
        yaw_err: float,
        tols: dict[str, float],
        vx: float,
        vy: float,
        wz: float,
    ) -> str:
        best_axis: str | None = None
        best_norm = -1.0
        best_err = 0.0
        for axis in axes:
            e = _err(axis, front, left, yaw_err)
            tol = tols[axis]
            if not self._wants(e, tol, _vel(axis, vx, vy, wz)):
                continue
            norm = abs(e) / max(tol, 1e-9)
            if norm > best_norm:
                best_norm, best_axis, best_err = norm, axis, e
        self.last_duty = 1.0 if best_axis is not None else 0.0
        if best_axis is None:
            return "NONE"
        return _token(best_axis, math.copysign(1.0, best_err))


# ---------------------------------------------------------------------------
# Super-fine engage/coast/brake trim (optional stage)
# ---------------------------------------------------------------------------


@dataclass
class PulseSettleTrim:
    """Engage/coast/brake base trim after the arms are placed (F-93).

    Idle (stopped + new odom) locks the worst ``|err|/tol`` axis and fires a
    ≤50 ms pulse. Guide then runs every tick: coast if the stop prediction
    lands in the gate, brake (same-axis reverse) on catch/overshoot or a
    delayed F-93 catch (|v| above the clean-pulse peak). Inside the gate:
    never fire, never reverse. Outside: yaw reverse only when opening
    (``err`` and ``wz`` opposite signs; ``err`` is goal−yaw). Re-pulse only
    once that axis has stopped for one settle constant on a *new* odom.
    Handoff is an in-zone stopped dwell, not a single in-tol sample.
    No timeout-accept.
    """

    goal_x: float
    goal_y: float
    goal_yaw: float
    tol_xy_m: float = _SUPER_FINE_XY
    tol_yaw_rad: float = _SUPER_FINE_YAW
    k: float = _PULSE_K
    hold_floor_s: float = _PULSE_HOLD_FLOOR_S
    hold_cap_s: float = _PULSE_HOLD_CAP_S
    v_lin_mps: float = _PULSE_V_LIN_MPS
    v_yaw_radps: float = _PULSE_V_YAW_RADPS
    lin_stop_mps: float = _LIN_STOP_MPS
    yaw_stop_radps: float = _YAW_STOP_RADPS
    stop_s: float = _STOP_S
    hold_min_ticks: int = _PULSE_HOLD_MIN_TICKS
    dwell_s: float = _DWELL_S
    dwell_leave: float = _DWELL_LEAVE
    dwell_min_ticks: int = _DWELL_MIN_TICKS
    catch_lin_mps: float = _CATCH_LIN_MPS
    catch_yaw_radps: float = _CATCH_YAW_RADPS

    _phase: str = field(default="idle", init=False, repr=False)
    _token: str | None = field(default=None, init=False, repr=False)
    _hold_s: float = field(default=0.0, init=False, repr=False)
    _fire_t0_sim: float | None = field(default=None, init=False, repr=False)
    _fire_ticks: int = field(default=0, init=False, repr=False)
    _last_odom_n: int | None = field(default=None, init=False, repr=False)
    _t0_sim: float | None = field(default=None, init=False, repr=False)
    _last_t_sim: float | None = field(default=None, init=False, repr=False)
    _locked_axis: str | None = field(default=None, init=False, repr=False)
    _pulse_err_sign: float | None = field(default=None, init=False, repr=False)
    _dwell_t0_sim: float | None = field(default=None, init=False, repr=False)
    _dwell_ticks: int = field(default=0, init=False, repr=False)
    _observe_t0_sim: float | None = field(default=None, init=False, repr=False)
    _observe_ticks: int = field(default=0, init=False, repr=False)
    last_axis: str | None = field(default=None, init=False)
    pulses: int = field(default=0, init=False)
    done: bool = field(default=False, init=False)
    last_duty: float = field(default=0.0, init=False)
    _allow_fire: bool = field(default=True, init=False, repr=False)

    def reset(self) -> None:
        self._phase = "idle"
        self._token = None
        self._hold_s = 0.0
        self._fire_t0_sim = None
        self._fire_ticks = 0
        self._last_odom_n = None
        self._t0_sim = None
        self._last_t_sim = None
        self._locked_axis = None
        self._pulse_err_sign = None
        self._dwell_t0_sim = None
        self._dwell_ticks = 0
        self._observe_t0_sim = None
        self._observe_ticks = 0
        self.last_axis = None
        self.pulses = 0
        self.done = False
        self.last_duty = 0.0
        self._allow_fire = True

    def elapsed_sim_s(self, t_sim: float | None = None) -> float:
        if self._t0_sim is None:
            return 0.0
        if t_sim is None:
            t_sim = self._last_t_sim
        if t_sim is None:
            return 0.0
        return max(0.0, float(t_sim) - self._t0_sim)

    def final_tols(self) -> tuple[float, float]:
        return self.tol_xy_m, self.tol_yaw_rad

    def _tol(self, axis: str) -> float:
        return self.tol_yaw_rad if axis == "yaw" else self.tol_xy_m

    def _v(self, axis: str) -> float:
        return self.v_yaw_radps if axis == "yaw" else self.v_lin_mps

    def _axis_stop_lim(self, axis: str) -> float:
        return self.yaw_stop_radps if axis == "yaw" else self.lin_stop_mps

    def _catch_lim(self, axis: str) -> float:
        return self.catch_yaw_radps if axis == "yaw" else self.catch_lin_mps

    def _rel(self, axis: str, err: float) -> float:
        return abs(err) / max(self._tol(axis), 1e-12)

    def _measure(self, x: float, y: float, yaw: float) -> dict[str, float]:
        front, left, yaw_err = body_frame_errors(
            x, y, yaw, self.goal_x, self.goal_y, self.goal_yaw
        )
        return {"front": front, "left": left, "yaw": yaw_err}

    def _in_band(self, errs: dict[str, float], scale: float = 1.0) -> bool:
        return all(abs(errs[axis]) <= scale * self._tol(axis) for axis in AXES)

    def _axis_with_max_rel_err(self, errs: dict[str, float]) -> str | None:
        best_axis, best_rel = None, -1.0
        for axis in AXES:
            err = errs[axis]
            if abs(err) <= self._tol(axis):
                continue
            rel = self._rel(axis, err)
            if rel > best_rel:
                best_axis, best_rel = axis, rel
        return best_axis

    def _hold_s_for(self, axis: str, err: float) -> tuple[float, float]:
        """Return ``(hold_s, raw)`` for ``hold_s = clip(k * |err| / v, floor, cap)``."""
        raw = self.k * abs(err) / max(self._v(axis), 1e-12)
        return min(self.hold_cap_s, max(self.hold_floor_s, raw)), raw

    def _still_holding(self, t_sim: float | None) -> bool:
        if t_sim is not None and self._fire_t0_sim is not None:
            return float(t_sim) - self._fire_t0_sim < self._hold_s
        return self._fire_ticks < max(1, self.hold_min_ticks)

    def _stopped(self, vx: float, vy: float, wz: float) -> bool:
        return math.hypot(vx, vy) <= self.lin_stop_mps and abs(wz) <= self.yaw_stop_radps

    def _new_odom(self, odom_n: int) -> bool:
        return self._last_odom_n is None or odom_n > self._last_odom_n

    def _err_after_coast(self, err: float, vel: float) -> float:
        return err - 0.5 * vel * self.stop_s

    def _closing(self, err: float, vel: float) -> float:
        if err == 0.0:
            return 0.0
        return vel * math.copysign(1.0, err)

    def _will_coast_in(self, err: float, vel: float, tol: float) -> bool:
        if self._closing(err, vel) <= 0.0:
            return False
        return abs(self._err_after_coast(err, vel)) <= tol

    def _will_overshoot(self, err: float, vel: float, tol: float) -> bool:
        if self._closing(err, vel) <= 0.0:
            return False
        future = self._err_after_coast(err, vel)
        return future * err < 0.0 and abs(future) > tol

    def _allow_brake(self, axis: str, err: float, vel: float) -> bool:
        """Outside the gate only. Yaw: only when opening (flying away)."""
        if abs(err) <= self._tol(axis):
            return False
        return axis != "yaw" or self._closing(err, vel) <= 0.0

    def _dwell_held(self, t_sim: float | None) -> bool:
        if t_sim is not None and self._dwell_t0_sim is not None:
            return float(t_sim) - self._dwell_t0_sim >= self.dwell_s
        return self._dwell_ticks >= max(1, self.dwell_min_ticks)

    def _observe_done(self, t_sim: float | None) -> bool:
        """Wait one settle constant after release so delayed catch can show."""
        if t_sim is not None and self._observe_t0_sim is not None:
            return float(t_sim) - self._observe_t0_sim >= self.stop_s
        return self._observe_ticks >= max(1, self.hold_min_ticks)

    def _begin_observe(self, odom_n: int, t_sim: float | None) -> None:
        self._last_odom_n = odom_n
        self._observe_t0_sim = float(t_sim) if t_sim is not None else None
        self._observe_ticks = 0

    def _enter_fire(self, axis: str, err: float, t_sim: float | None) -> str:
        token = _token(axis, math.copysign(1.0, err))
        hold_s, raw = self._hold_s_for(axis, err)
        if raw < self.hold_floor_s:
            tag = " (floor)"
        elif raw > self.hold_cap_s:
            tag = " (cap)"
        else:
            tag = ""
        unit = "rad" if axis == "yaw" else "m"
        log.info(
            "finegrained_start_position: hold_s = clip(k*|err|/v, floor, cap) "
            "= clip(%.3f*%.6f/%.6f, %.3f, %.3f) raw=%.3f → %.3f s%s  "
            "axis=%s token=%s err=%+.6f %s rel=%.2f",
            self.k,
            abs(err),
            self._v(axis),
            self.hold_floor_s,
            self.hold_cap_s,
            raw,
            hold_s,
            tag,
            axis,
            token,
            err,
            unit,
            self._rel(axis, err),
        )
        self.last_axis = axis
        self._locked_axis = axis
        self._pulse_err_sign = math.copysign(1.0, err)
        self._token = token
        self._hold_s = hold_s
        self._fire_t0_sim = float(t_sim) if t_sim is not None else None
        self._fire_ticks = 0
        self._phase = "fire"
        self.pulses += 1
        self.last_duty = 1.0
        return token

    def _enter_brake(self, axis: str, vel: float, err: float, t_sim: float | None) -> str:
        if abs(vel) > self._axis_stop_lim(axis):
            token = _token(axis, -math.copysign(1.0, vel))
        elif err != 0.0:
            token = _token(axis, math.copysign(1.0, err))
        else:
            self._phase = "guide"
            return "NONE"
        log.info(
            "finegrained_start_position: brake axis=%s token=%s err=%+.6f vel=%+.4f",
            axis,
            token,
            err,
            vel,
        )
        self.last_axis = axis
        self._locked_axis = axis
        self._token = token
        self._hold_s = self.hold_cap_s
        self._fire_t0_sim = float(t_sim) if t_sim is not None else None
        self._fire_ticks = 0
        self._phase = "brake"
        self.last_duty = 1.0
        return token

    def _enter_dwell(self, t_sim: float | None) -> str:
        log.info(
            "finegrained_start_position: dwell start t=%.2f pulses=%d",
            self.elapsed_sim_s(t_sim),
            self.pulses,
        )
        self._phase = "dwell"
        self._locked_axis = None
        self._pulse_err_sign = None
        self._token = None
        self._dwell_t0_sim = float(t_sim) if t_sim is not None else None
        self._dwell_ticks = 0
        return "NONE"

    def hint(self, x: float, y: float, yaw: float, t_sim: float | None = None) -> str:
        if self.done:
            return "fine trim done"
        if t_sim is None:
            t_sim = self._last_t_sim
        clock = f"t={self.elapsed_sim_s(t_sim):.1f}s pulses={self.pulses}"
        if not all(math.isfinite(v) for v in (x, y, yaw)):
            return f"fine odom n/a {clock}"
        errs = self._measure(x, y, yaw)
        axis = self._locked_axis or self._axis_with_max_rel_err(errs)
        rels = " ".join(f"{a}={self._rel(a, errs[a]):.2f}" for a in AXES)
        pick = axis or "in-band"
        return f"fine/{pick} rel[{rels}] phase={self._phase} {clock}"

    def _step_fire(self, vx: float, vy: float, wz: float, odom_n: int, t_sim) -> str:
        self._fire_ticks += 1
        axis = self._locked_axis
        engaged = (
            axis is not None and abs(_vel(axis, vx, vy, wz)) > self._axis_stop_lim(axis)
        )
        if not engaged and self._still_holding(t_sim) and self._token is not None:
            self.last_duty = 1.0
            return self._token
        self._phase = "guide"
        self._begin_observe(odom_n, t_sim)
        return "NONE"

    def _step_brake(self, vx: float, vy: float, wz: float, odom_n: int, t_sim) -> str:
        self._fire_ticks += 1
        axis = self._locked_axis
        vel = 0.0 if axis is None else _vel(axis, vx, vy, wz)
        dumped = axis is None or abs(vel) <= self._axis_stop_lim(axis)
        if not dumped and self._still_holding(t_sim) and self._token is not None:
            self.last_duty = 1.0
            return self._token
        self._phase = "guide"
        self._begin_observe(odom_n, t_sim)
        return "NONE"

    def _step_dwell(
        self,
        errs: dict[str, float],
        vx: float,
        vy: float,
        wz: float,
        odom_n: int,
        t_sim: float | None,
    ) -> str:
        if not self._in_band(errs, self.dwell_leave) or not self._stopped(vx, vy, wz):
            self._phase = "idle"
            self._dwell_t0_sim = None
            self._dwell_ticks = 0
            return "NONE"
        if not self._new_odom(odom_n):
            return "NONE"
        self._last_odom_n = odom_n
        self._dwell_ticks += 1
        if self._dwell_held(t_sim):
            log.info(
                "finegrained_start_position: dwell complete t=%.2f pulses=%d",
                self.elapsed_sim_s(t_sim),
                self.pulses,
            )
            self.done = True
        return "NONE"

    def _guide(
        self,
        errs: dict[str, float],
        vx: float,
        vy: float,
        wz: float,
        odom_n: int,
        t_sim: float | None,
    ) -> str:
        axis = self._locked_axis
        if axis is None:
            self._phase = "idle"
            return self._idle(errs, vx, vy, wz, odom_n, t_sim)
        err, vel, tol = errs[axis], _vel(axis, vx, vy, wz), self._tol(axis)

        if self._in_band(errs) and self._stopped(vx, vy, wz):
            if not self._new_odom(odom_n):
                return "NONE"
            self._last_odom_n = odom_n
            return self._enter_dwell(t_sim)

        if abs(err) <= tol and self._stopped(vx, vy, wz):
            self._locked_axis = None
            self._pulse_err_sign = None
            self._phase = "idle"
            if self._new_odom(odom_n):
                self._last_odom_n = odom_n
            return "NONE"

        flipped = self._pulse_err_sign is not None and err * self._pulse_err_sign < 0.0
        catching = abs(vel) > self._catch_lim(axis)
        if flipped or self._will_overshoot(err, vel, tol) or catching:
            if self._allow_brake(axis, err, vel):
                return self._enter_brake(axis, vel, err, t_sim)
        if (
            self._allow_brake(axis, err, vel)
            and self._closing(err, vel) <= 0.0
            and abs(vel) > self._axis_stop_lim(axis)
        ):
            return self._enter_brake(axis, vel, err, t_sim)
        if self._will_coast_in(err, vel, tol):
            return "NONE"
        if self._stopped(vx, vy, wz):
            if not self._observe_done(t_sim):
                if self._new_odom(odom_n):
                    self._last_odom_n = odom_n
                    self._observe_ticks += 1
                return "NONE"
            if abs(err) > tol and self._new_odom(odom_n):
                if not self._allow_fire:
                    return "NONE"
                self._last_odom_n = odom_n
                return self._enter_fire(axis, err, t_sim)
        return "NONE"

    def _idle(
        self,
        errs: dict[str, float],
        vx: float,
        vy: float,
        wz: float,
        odom_n: int,
        t_sim: float | None,
    ) -> str:
        if not self._stopped(vx, vy, wz) or not self._new_odom(odom_n):
            return "NONE"
        self._last_odom_n = odom_n
        if self._in_band(errs):
            return self._enter_dwell(t_sim)
        axis = self._axis_with_max_rel_err(errs)
        if axis is None:
            return self._enter_dwell(t_sim)
        if not self._allow_fire:
            return "NONE"
        return self._enter_fire(axis, errs[axis], t_sim)

    def step(
        self,
        x: float,
        y: float,
        yaw: float,
        vx: float,
        vy: float,
        wz: float,
        odom_n: int | None,
        t_sim: float | None = None,
        *,
        allow_fire: bool = True,
    ) -> str:
        """One control tick. Returns a pedal token (usually ``NONE``).

        ``t_sim`` sizes pulse holds and the in-zone dwell. Omit in unit tests
        that use the tick-count fallback. There is no timeout-accept.
        """
        self.last_duty = 0.0
        self._allow_fire = bool(allow_fire)
        if self.done:
            return "NONE"
        if t_sim is not None and math.isfinite(t_sim):
            self._last_t_sim = float(t_sim)
            if self._t0_sim is None:
                self._t0_sim = float(t_sim)
        if not all(math.isfinite(v) for v in (x, y, yaw)):
            return "NONE"

        # Unit tests / no collector: treat every call as a fresh odom sample.
        if odom_n is None:
            odom_n = (self._last_odom_n or 0) + 1

        if self._phase == "fire" and self._token is not None:
            return self._step_fire(vx, vy, wz, odom_n, t_sim)
        if self._phase == "brake" and self._token is not None:
            return self._step_brake(vx, vy, wz, odom_n, t_sim)

        errs = self._measure(x, y, yaw)
        if self._phase == "dwell":
            return self._step_dwell(errs, vx, vy, wz, odom_n, t_sim)
        if self._phase == "guide":
            return self._guide(errs, vx, vy, wz, odom_n, t_sim)
        return self._idle(errs, vx, vy, wz, odom_n, t_sim)


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

# Data-augmentation base-goal jitter: kept small because the arm-place path
# and fine trim are only validated near the nominal Task 2 goal.
_GOAL_JITTER_XY_MAX_M = 0.05
_GOAL_JITTER_YAW_MAX_RAD = math.radians(3.0)


class ApproachController:
    """FSM: spine SOP → navigate → place arms → optional fine trim → handover.

    ``place_arms`` walks ``arm_path`` leg by leg (default: unfold forward,
    then lower onto ``arm_ready_left/right``), advancing when the measured
    joints are inside the leg's tolerance and have stayed there for its
    settle window. Spine/navigate hold the as-found arm pose so the base can
    move without forcing ready early. ``finegrained_start_position`` (default
    on) engage/coast/brakes the base after the arms are out; pass
    ``finegrained_start_position=False`` to drop that stage entirely.
    ``set_goal_jitter`` offsets the final waypoint a few cm / degrees for
    per-episode data-augmentation diversity without changing task geometry.
    """

    def __init__(
        self,
        waypoints: tuple[Waypoint, ...] = TASK2_APPROACH_WAYPOINTS,
        arm_ready_left=C.TASK2_ARM_READY_LEFT,
        arm_ready_right=C.TASK2_ARM_READY_RIGHT,
        arm_path: tuple[ArmWaypoint, ...] | None = None,
        arm_tol_left=None,
        arm_tol_right=None,
        gripper_open: float = C.TASK2_GRIPPER_OPEN,
        spine_measured_min_m: float = C.SPINE_SOP_MEASURED_MIN_M,
        spine_target_m: float = C.SPINE_SOP_M,
        settle_ticks: int = 8,
        arm_settle_ticks: int = 25,
        lin_stop_mps: float = _LIN_STOP_MPS,
        yaw_stop_radps: float = _YAW_STOP_RADPS,
        drive: AxisDrive | None = None,
        finegrained_start_position: bool = True,
        base_only: bool = False,
        super_fine_xy_m: float = _SUPER_FINE_XY,
        super_fine_yaw_rad: float = _SUPER_FINE_YAW,
        fine_trim: PulseSettleTrim | None = None,
        skip_spine: bool = False,
    ):
        if not waypoints:
            raise ValueError("waypoints must be non-empty")
        self.waypoints = waypoints
        self._nominal_goal = self.waypoints[-1]
        self.finegrained_start_position = bool(finegrained_start_position)
        self.base_only = bool(base_only)
        self.stages = build_stages(
            finegrained_start_position=self.finegrained_start_position,
            skip_spine=bool(skip_spine),
            base_only=self.base_only,
        )
        self.arm_ready_left = np.asarray(arm_ready_left, dtype=np.float64)
        self.arm_ready_right = np.asarray(arm_ready_right, dtype=np.float64)
        n_l, n_r = self.arm_ready_left.shape[0], self.arm_ready_right.shape[0]
        self.arm_tol_left = (
            np.full(n_l, _ARM_TOL, dtype=np.float64)
            if arm_tol_left is None
            else np.asarray(arm_tol_left, dtype=np.float64)
        )
        self.arm_tol_right = (
            np.full(n_r, _ARM_TOL, dtype=np.float64)
            if arm_tol_right is None
            else np.asarray(arm_tol_right, dtype=np.float64)
        )
        if self.arm_tol_left.shape != self.arm_ready_left.shape:
            raise ValueError("arm_tol_left must match arm_ready_left shape")
        if self.arm_tol_right.shape != self.arm_ready_right.shape:
            raise ValueError("arm_tol_right must match arm_ready_right shape")
        self.arm_path = arm_path or default_arm_path(self.arm_ready_left, self.arm_ready_right)
        if not self.arm_path:
            raise ValueError("arm_path must be non-empty")
        self.gripper_open = float(gripper_open)
        self.spine_measured_min_m = spine_measured_min_m
        self.spine_target_m = spine_target_m
        self.settle_ticks = settle_ticks
        self.arm_settle_ticks = arm_settle_ticks
        self.lin_stop_mps = lin_stop_mps
        self.yaw_stop_radps = yaw_stop_radps
        self.drive = drive or AxisDrive()
        goal = self.waypoints[-1]
        self.fine_trim = fine_trim
        if self.finegrained_start_position and self.fine_trim is None:
            self.fine_trim = PulseSettleTrim(
                goal_x=goal.x,
                goal_y=goal.y,
                goal_yaw=goal.yaw,
                tol_xy_m=super_fine_xy_m,
                tol_yaw_rad=super_fine_yaw_rad,
                lin_stop_mps=lin_stop_mps,
                yaw_stop_radps=yaw_stop_radps,
            )
        self.reset()

    def reset(self) -> None:
        self.stage = self.stages[0]
        self.wp_index = 0
        self.arm_index = 0
        self.arm_leg_ticks = 0
        self._arm_hold_ticks = 0
        self._arm_hold_span = [0.0, 0.0]
        self._settle_left = 0
        self._nav_end_xy: tuple[float, float] | None = None
        self._left_cmd: np.ndarray | None = None
        self._right_cmd: np.ndarray | None = None
        self.drive.reset()
        if self.fine_trim is not None:
            # Goal may have been swapped in tests after construct — keep tols.
            goal = self.waypoints[-1]
            self.fine_trim.goal_x = goal.x
            self.fine_trim.goal_y = goal.y
            self.fine_trim.goal_yaw = goal.yaw
            self.fine_trim.reset()
        self.stats = {
            "nav_ticks": 0,
            "spine_ticks": 0,
            "arm_ticks": 0,
            "fine_ticks": 0,
            "fine_pulses": 0,
            "final_xy_err_m": None,
            "final_yaw_err_rad": None,
            "final_spine_m": None,
            # Start vs final arm error is what separates "placed the arms"
            # from "the arms were already there" in the log.
            "arm_err_start_left_rad": None,
            "arm_err_start_right_rad": None,
            "arm_err_left_rad": None,
            "arm_err_right_rad": None,
            "drift_after_nav_m": None,
        }

    def set_goal_jitter(self, dx_m: float, dy_m: float, dyaw_rad: float) -> None:
        """Offset the final waypoint from the nominal (frozen) Task 2 goal.

        Apply per episode, before ``reset()`` / ``run_episode`` — the offset
        only reaches ``fine_trim.goal_*`` on the next ``reset()`` (this does
        not call ``reset()`` itself). ``(0.0, 0.0, 0.0)`` restores the
        nominal waypoint exactly. Capped at ±5 cm / ±3°: the arm-place path
        and fine trim are only validated near the nominal goal, and a larger
        jitter would strand the fine trim outside its working band.
        """
        if abs(dx_m) > _GOAL_JITTER_XY_MAX_M or abs(dy_m) > _GOAL_JITTER_XY_MAX_M:
            raise ValueError(
                f"goal jitter dx/dy must be within ±{_GOAL_JITTER_XY_MAX_M} m "
                f"(got dx_m={dx_m}, dy_m={dy_m})"
            )
        if abs(dyaw_rad) > _GOAL_JITTER_YAW_MAX_RAD:
            raise ValueError(
                f"goal jitter dyaw must be within ±{_GOAL_JITTER_YAW_MAX_RAD:.6f} rad "
                f"(got dyaw_rad={dyaw_rad})"
            )
        nominal = self._nominal_goal
        jittered = replace(
            nominal,
            x=nominal.x + dx_m,
            y=nominal.y + dy_m,
            yaw=nominal.yaw + dyaw_rad,
        )
        self.waypoints = (*self.waypoints[:-1], jittered)

    @property
    def done(self) -> bool:
        return self.stage == "done"

    @property
    def goal_x(self) -> float:
        return self.waypoints[min(self.wp_index, len(self.waypoints) - 1)].x

    @property
    def goal_y(self) -> float:
        return self.waypoints[min(self.wp_index, len(self.waypoints) - 1)].y

    @property
    def goal_yaw(self) -> float:
        return self.waypoints[min(self.wp_index, len(self.waypoints) - 1)].yaw

    @property
    def pos_tol_m(self) -> float:
        return self.waypoints[min(self.wp_index, len(self.waypoints) - 1)].tol_xy_m

    @property
    def yaw_tol_rad(self) -> float:
        return self.waypoints[min(self.wp_index, len(self.waypoints) - 1)].tol_yaw_rad

    @property
    def arm_waypoint(self) -> ArmWaypoint:
        return self.arm_path[min(self.arm_index, len(self.arm_path) - 1)]

    def target_label(self) -> str:
        if self.stage == "navigate":
            return f"navigate/{self.waypoints[self.wp_index].name}"
        if self.stage == "place_arms":
            return f"place_arms/{self.arm_waypoint.name}"
        if self.stage == STAGE_FINE and self.fine_trim is not None:
            axis = self.fine_trim.last_axis
            return f"{STAGE_FINE}/{axis}" if axis else STAGE_FINE
        return self.stage

    def _handover_tols(self) -> tuple[float, float]:
        """xy / yaw limits used at ``start_pose`` warning time."""
        if self.finegrained_start_position and self.fine_trim is not None:
            return self.fine_trim.final_tols()
        goal = self.waypoints[-1]
        return goal.tol_xy_m, goal.tol_yaw_rad

    def _ensure_arm_cmd(self, state: np.ndarray) -> None:
        """Hold as-found arms until ``place_arms`` (seed once from measured)."""
        if self._left_cmd is not None:
            return
        left = np.asarray(state[C.S_LEFT_ARM], dtype=np.float64)
        right = np.asarray(state[C.S_RIGHT_ARM], dtype=np.float64)
        if not (np.all(np.isfinite(left)) and np.all(np.isfinite(right))):
            left = self.arm_ready_left.copy()
            right = self.arm_ready_right.copy()
        self._left_cmd, self._right_cmd = left, right

    def _command_arm_waypoint(self, wp: ArmWaypoint) -> None:
        self._left_cmd = np.asarray(wp.left, dtype=np.float64)
        self._right_cmd = np.asarray(wp.right, dtype=np.float64)

    def _arm_tolerances(self, wp: ArmWaypoint) -> tuple[np.ndarray, np.ndarray]:
        """Per-joint tolerance for ``wp``: the last leg is the ready pose, so
        it gets the caller's ``arm_tol_*``; the rest are passed through."""
        if wp is self.arm_path[-1]:
            return self.arm_tol_left, self.arm_tol_right
        return (
            np.full_like(self.arm_tol_left, wp.tol_rad),
            np.full_like(self.arm_tol_right, wp.tol_rad),
        )

    def _arms_at_waypoint(self, state: np.ndarray, wp: ArmWaypoint) -> bool:
        left = np.asarray(state[C.S_LEFT_ARM], dtype=np.float64)
        right = np.asarray(state[C.S_RIGHT_ARM], dtype=np.float64)
        if not (np.all(np.isfinite(left)) and np.all(np.isfinite(right))):
            return False
        tol_left, tol_right = self._arm_tolerances(wp)
        return bool(
            np.all(np.abs(left - np.asarray(wp.left)) <= tol_left)
            and np.all(np.abs(right - np.asarray(wp.right)) <= tol_right)
        )

    def arm_joint_errors(
        self, state: np.ndarray, wp: ArmWaypoint | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-joint absolute error (rad) to ``wp`` for left and right arms."""
        wp = wp or self.arm_waypoint
        out = []
        for sl, target in ((C.S_LEFT_ARM, wp.left), (C.S_RIGHT_ARM, wp.right)):
            measured = np.asarray(state[sl], dtype=np.float64)
            target_a = np.asarray(target, dtype=np.float64)
            if np.all(np.isfinite(measured)):
                out.append(np.abs(measured - target_a))
            else:
                out.append(np.full(target_a.shape, math.nan, dtype=np.float64))
        return out[0], out[1]

    def arm_errors(self, state: np.ndarray, wp: ArmWaypoint | None = None) -> tuple[float, float]:
        """Worst measured joint error to ``wp`` (default: current leg), per side."""
        left, right = self.arm_joint_errors(state, wp)
        return (
            float(np.nanmax(left)) if left.size else math.nan,
            float(np.nanmax(right)) if right.size else math.nan,
        )

    def stage_hint(self, state: np.ndarray) -> str:
        """Why the current stage has not finished — for timeout messages."""
        if self.stage == "spine":
            spine = float(state[C.S_SPINE])
            return (
                f"spine measured {spine:.3f} m < {self.spine_measured_min_m:.2f} m; "
                f"nothing here can command it — relaunch the sim with "
                f"--spine-keyboard-min/max {self.spine_target_m:.2f} (F-50)"
            )
        if self.stage == "start_pose":
            e = self.start_pose_errors(state)
            return (
                f"base {e['xy_err_m'] * 1000:.0f} mm / {e['yaw_err_rad']:+.4f} rad "
                f"from goal, spine {e['spine_m']:.3f} m, arms L "
                f"{math.degrees(e['arm_err_left_rad']):.1f}° R "
                f"{math.degrees(e['arm_err_right_rad']):.1f}° to ready"
            )
        if self.stage == "place_arms":
            left, right = self.arm_errors(state)
            return (
                f"arms not reaching leg '{self.arm_waypoint.name}' (worst joint: left "
                f"{math.degrees(left):.1f}°, right {math.degrees(right):.1f}°) — "
                "is the helper stack up in position mode (make helpers-up)?"
            )
        if self.stage == STAGE_FINE and self.fine_trim is not None:
            x, y, yaw = (float(v) for v in state[C.S_BASE_ODOM])
            return self.fine_trim.hint(x, y, yaw)
        x, y, yaw = (float(v) for v in state[C.S_BASE_ODOM])
        return (
            f"{self.target_label()}: {math.hypot(self.goal_x - x, self.goal_y - y) * 1000:.0f} mm "
            f"/ {wrap_angle(self.goal_yaw - yaw):+.4f} rad to go"
        )

    def _command(self, token: str) -> Command:
        assert self._left_cmd is not None and self._right_cmd is not None
        return Command(
            left_arm=self._left_cmd.astype(np.float32),
            right_arm=self._right_cmd.astype(np.float32),
            left_gripper=self.gripper_open,
            right_gripper=self.gripper_open,
            base_twist=C.token_to_twist(token),
            base_token=token,
        )

    def _advance_stage(self) -> None:
        i = self.stages.index(self.stage)
        self.stage = self.stages[i + 1] if i + 1 < len(self.stages) else "done"

    def _waypoint_ok(
        self,
        wp: Waypoint,
        front: float,
        left: float,
        yaw_err: float,
        vx: float,
        vy: float,
        wz: float,
        *,
        require_stopped: bool,
    ) -> bool:
        for axis in wp.axes:
            if abs(_err(axis, front, left, yaw_err)) > _tol(axis, wp):
                return False
            lim = self.yaw_stop_radps if axis == "yaw" else self.lin_stop_mps
            if require_stopped and abs(_vel(axis, vx, vy, wz)) > lim:
                return False
        return True

    def _step_spine(self, state: np.ndarray) -> Command:
        self.stats["spine_ticks"] += 1
        self._ensure_arm_cmd(state)
        spine = float(state[C.S_SPINE])
        self.stats["final_spine_m"] = spine if math.isfinite(spine) else None
        if math.isfinite(spine) and spine >= self.spine_measured_min_m:
            self._advance_stage()
        return self._command("NONE")

    def _step_place_arms(self, state: np.ndarray) -> Command:
        """Walk ``arm_path`` leg by leg; the last leg is the ready pose."""
        self.stats["arm_ticks"] += 1
        self.arm_leg_ticks += 1
        wp = self.arm_waypoint
        self._command_arm_waypoint(wp)
        left_err, right_err = self.arm_errors(state, self.arm_path[-1])
        if self.stats["arm_err_start_left_rad"] is None:
            self.stats["arm_err_start_left_rad"] = left_err
            self.stats["arm_err_start_right_rad"] = right_err
        self.stats["arm_err_left_rad"] = left_err
        self.stats["arm_err_right_rad"] = right_err

        # Accepting a leg needs two things, and in-tolerance alone is neither.
        # The first ticks' measured state predates the command (joints report
        # at ~10 Hz against a 50 Hz loop), so hold it for a settle window —
        # otherwise "arms moved there" reads the same as "arms were already
        # there". And the arm must have stopped converging: taking the first
        # in-tolerance tick banks the whole tolerance as residual error, which
        # on the final leg is real height (1.8° of shoulder ≈ 15 mm of flange,
        # measured 2026-08-10) between us and the pose the demos hand over in.
        leg_err = max(self.arm_errors(state, wp))
        settle = self.arm_settle_ticks if wp.settle_ticks is None else wp.settle_ticks
        if not self._arms_at_waypoint(state, wp):
            self._arm_hold_ticks = 0
            return self._command("NONE")
        # Span over the window, not its endpoints: an arm swinging through the
        # target comes back to the error it started at.
        if self._arm_hold_ticks == 0:
            self._arm_hold_span = [leg_err, leg_err]
        self._arm_hold_span = [
            min(self._arm_hold_span[0], leg_err),
            max(self._arm_hold_span[1], leg_err),
        ]
        self._arm_hold_ticks += 1
        if self._arm_hold_ticks > settle:
            if self._arm_hold_span[1] - self._arm_hold_span[0] < _ARM_SETTLE_EPS:
                self.arm_leg_ticks = self._arm_hold_ticks = 0
                if self.arm_index + 1 < len(self.arm_path):
                    self.arm_index += 1
                else:
                    self._advance_stage()
            else:
                self._arm_hold_ticks = 0  # still converging — start a new window
        return self._command("NONE")

    def start_pose_errors(self, state: np.ndarray) -> dict:
        """Everything the handover check looks at, measured in one place."""
        x, y, yaw = (float(v) for v in state[C.S_BASE_ODOM])
        goal = self.waypoints[-1]
        ready = self.arm_path[-1]
        left_joints, right_joints = self.arm_joint_errors(state, ready)
        left_err = float(np.nanmax(left_joints)) if left_joints.size else math.nan
        right_err = float(np.nanmax(right_joints)) if right_joints.size else math.nan
        left_j = int(np.nanargmax(left_joints)) + 1 if np.any(np.isfinite(left_joints)) else -1
        right_j = int(np.nanargmax(right_joints)) + 1 if np.any(np.isfinite(right_joints)) else -1
        front, left, _eyaw = (
            body_frame_errors(x, y, yaw, goal.x, goal.y, goal.yaw)
            if all(math.isfinite(v) for v in (x, y, yaw))
            else (math.nan, math.nan, math.nan)
        )
        spine = float(state[C.S_SPINE])
        return {
            "xy_err_m": math.hypot(goal.x - x, goal.y - y),
            "x_err_m": goal.x - x,
            "y_err_m": goal.y - y,
            "yaw_err_rad": wrap_angle(goal.yaw - yaw) if math.isfinite(yaw) else math.nan,
            "body_front_m": front,
            "body_left_m": left,
            "spine_m": spine,
            "spine_err_m": (
                spine - self.spine_target_m if math.isfinite(spine) else math.nan
            ),
            "arm_err_left_rad": left_err,
            "arm_err_right_rad": right_err,
            "arm_err_left_joint": left_j,
            "arm_err_right_joint": right_j,
            "arm_joint_err_left_rad": left_joints,
            "arm_joint_err_right_rad": right_joints,
            "gripper_left": float(state[C.S_LEFT_GRIP]),
            "gripper_right": float(state[C.S_RIGHT_GRIP]),
            "gripper_err_left": float(state[C.S_LEFT_GRIP]) - self.gripper_open,
            "gripper_err_right": float(state[C.S_RIGHT_GRIP]) - self.gripper_open,
            "flange_z_left_m": float(state[C.S_LEFT_EE][2]),
            "flange_z_right_m": float(state[C.S_RIGHT_EE][2]),
            "drift_m": (
                math.hypot(x - self._nav_end_xy[0], y - self._nav_end_xy[1])
                if self._nav_end_xy is not None
                else math.nan
            ),
        }

    def _step_finegrained(
        self, state: np.ndarray, x, y, yaw, vx, vy, wz, odom_n, t_sim, *, allow_fire=True
    ) -> Command:
        """Engage/coast/brake trim until an in-zone stopped dwell."""
        assert self.fine_trim is not None
        self._ensure_arm_cmd(state)
        self.stats["fine_ticks"] += 1
        token = self.fine_trim.step(
            x, y, yaw, vx, vy, wz, odom_n, t_sim=t_sim, allow_fire=allow_fire
        )
        self.stats["fine_pulses"] = self.fine_trim.pulses
        self.drive.last_duty = self.fine_trim.last_duty
        if self.fine_trim.done:
            self.stats["final_xy_err_m"] = math.hypot(
                self.fine_trim.goal_x - x, self.fine_trim.goal_y - y
            )
            self.stats["final_yaw_err_rad"] = wrap_angle(self.fine_trim.goal_yaw - yaw)
            self._advance_stage()
        return self._command(token)

    def _step_start_pose(self, state: np.ndarray) -> Command:
        """Log the whole start pose at handover; warn on drift, never abort.

        Nav already required stopped + in-tol before ``place_arms``. Base can
        still creep while arms move; that is visible here but not a hard fail —
        the policy still starts so eval/replay can proceed.
        """
        errors = self.start_pose_errors(state)
        tol_xy, tol_yaw = self._handover_tols()
        warnings = []
        if errors["xy_err_m"] > tol_xy:
            warnings.append(
                f"base {errors['xy_err_m'] * 1000:.0f} mm from the goal "
                f"(limit {tol_xy * 1000:.0f} mm, "
                f"drifted {errors['drift_m'] * 1000:.0f} mm since navigate)"
            )
        if abs(errors["yaw_err_rad"]) > tol_yaw:
            warnings.append(
                f"yaw {errors['yaw_err_rad']:+.4f} rad off "
                f"(limit ±{tol_yaw:.4f} rad)"
            )
        if not (errors["spine_m"] >= self.spine_measured_min_m):
            warnings.append(
                f"spine {errors['spine_m']:.3f} m < {self.spine_measured_min_m:.2f} m"
            )
        if not self._arms_at_waypoint(state, self.arm_path[-1]):
            warnings.append(
                f"arms off ready (worst joint: left "
                f"{math.degrees(errors['arm_err_left_rad']):.1f}°, right "
                f"{math.degrees(errors['arm_err_right_rad']):.1f}°)"
            )
        if warnings:
            log.warning(
                "start position off at handover: %s — starting policy anyway",
                "; ".join(warnings),
            )
        self.stats.update(
            final_xy_err_m=errors["xy_err_m"],
            final_yaw_err_rad=errors["yaw_err_rad"],
            final_spine_m=errors["spine_m"],
            drift_after_nav_m=errors["drift_m"],
            start_pose_ok=not warnings,
        )
        self._advance_stage()
        return self._command("NONE")

    def _step_navigate(self, state: np.ndarray, x, y, yaw, vx, vy, wz) -> Command:
        self.stats["nav_ticks"] += 1
        self._ensure_arm_cmd(state)
        if not all(math.isfinite(v) for v in (x, y, yaw)):
            return self._command("NONE")

        wp = self.waypoints[self.wp_index]
        front, left, yaw_err = body_frame_errors(x, y, yaw, wp.x, wp.y, wp.yaw)
        tols = {a: _tol(a, wp) for a in wp.axes}
        last = self.wp_index == len(self.waypoints) - 1

        if self._waypoint_ok(wp, front, left, yaw_err, vx, vy, wz, require_stopped=last):
            if not last:
                self.drive.reset()
                self.wp_index += 1
                return self._command("NONE")
            self._settle_left += 1
            if self._settle_left >= self.settle_ticks:
                self.stats["final_xy_err_m"] = math.hypot(wp.x - x, wp.y - y)
                self.stats["final_yaw_err_rad"] = yaw_err
                self._nav_end_xy = (x, y)
                self._settle_left = 0
                self.drive.reset()
                self._advance_stage()
            return self._command("NONE")

        self._settle_left = 0
        return self._command(self.drive.step(wp.axes, front, left, yaw_err, tols, vx, vy, wz))

    def step(
        self,
        state: np.ndarray,
        *,
        odom_n: int | None = None,
        t_sim: float | None = None,
        allow_fire: bool = True,
    ) -> Command:
        """One approach tick.

        ``odom_n`` is the collector's cumulative odom message count — idle
        re-pulses and the dwell clock advance at most once per new sample.
        ``t_sim`` sizes pulse holds and the dwell. Omit either in unit tests.
        """
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

        x, y, yaw = (float(v) for v in state[C.S_BASE_ODOM])
        vx, vy, wz = (float(v) for v in state[C.S_BASE_VEL])
        if not all(math.isfinite(v) for v in (vx, vy, wz)):
            vx = vy = wz = 0.0

        if self.stage == "navigate":
            return self._step_navigate(state, x, y, yaw, vx, vy, wz)
        if self.stage == "spine":
            return self._step_spine(state)
        if self.stage == "place_arms":
            return self._step_place_arms(state)
        if self.stage == STAGE_FINE:
            return self._step_finegrained(
                state, x, y, yaw, vx, vy, wz, odom_n, t_sim, allow_fire=allow_fire
            )
        if self.stage == "start_pose":
            return self._step_start_pose(state)
        self._ensure_arm_cmd(state)
        return self._command("NONE")
