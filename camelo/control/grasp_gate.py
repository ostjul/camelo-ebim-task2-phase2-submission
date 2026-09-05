"""Close the gripper where a demonstrator would have (F-97/F-98).

The fine-tuned checkpoints never predict the gripper close. F-97 measured
it at scale: 0 of 1120 chunks put a close inside the executed window for
`lora_v2`, and the `expert`'s rare closes carry no timing information
(2.08 % near the grasp against 2.19 % far from it, Fisher p = 1.000). The
channel echoes the gripper state and ignores the phase, so nothing
downstream — RTC, more samples, a longer rollout — can recover a close
that was never learned.

This supplies that one bit geometrically. It is a **diagnostic ablation**,
not a policy result: it measures what the learned arm is worth if the
gripper worked. Say so in any number it produces.

The envelope is measured from the demos rather than chosen
(`tools/dgx_probes/grasp_pose_envelope.py`, 22 episodes):

    distance TCP->pad at the grasp    27.9 - 36.8 mm
    orientation vs the median grasp    0.0 -  6.96 deg

so a gate at 45 mm / 7 deg fires only where a demonstrator did in fact
grasp. Both halves earn their place: across four post-F-96 rollouts the
arm was oriented 0.56-1.69 deg from the demo attitude — orientation never
binds — but a position-only gate would eventually fire with the gripper
beside or above the pad in the wrong attitude, and this costs nothing to
check.

Numpy-only on purpose: the decision is unit-testable with no ROS, no sim
and no policy. The caller supplies the pad pose, which
`camelo/ros/obs_collector.py` reads LIVE from `OBJECT_POSES_TOPIC` — a
constant would drift as soon as the assembly is nudged, which happens.

The same envelope is also the **instrument** for the grasp protocol's first
sim intermediate metric (GRASP_EXPERIMENT_PROTOCOL.md §0: "a gripper close
fired while the right TCP was inside the F-98 envelope"). `GraspObserver`
is that instrument: the passive counterpart that measures and never
commands. It is a separate object rather than more `stats()` on the gate
because the gate cannot honestly report on itself — it latches, so it stops
recomputing the distance the moment it fires, and `grasp_gate_last_dist_m`
is therefore a LAST-TICK sample, not an approach. It reads 2.77 m and
2.41 m in outputs/eval/graspgate_n2/results.csv: true of the final tick,
and silent on whether the arm ever reached the pad. `envelope_state()`
below is the one definition of "inside the envelope" both share, so the
detector and the actuator can never drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from camelo.control.kinematics import (
    MobileFR3Kinematics,
    quat_xyzw_angle_deg,
    quat_xyzw_to_rot,
)

# Rounded out from the measured demo envelope above. 45 rather than 40:
# one of four rollouts reached 41.8 mm and would otherwise be excluded by
# 1.8 mm, and the demos never grasp closer than 27.9 mm either, so the
# band is generous in both directions by construction.
DEFAULT_MAX_DIST_M = 0.045
DEFAULT_MAX_ANGLE_DEG = 7.0

# Once closed, stay closed. The dwell inside the envelope is short — 22 to
# 115 ticks of ~3400 in the measured rollouts — so a gate that reopened on
# leaving would drop the pad the moment the arm lifted it.
LATCH = True

# Threshold on the CANONICAL open fraction (1.0 open, 0.0 closed — see
# contracts.gripper_open_fraction) that separates "open" from "closed" when
# reading the gripper channel back. The action channel is effectively binary
# already (the protocol retired A2 for exactly that reason), so the midpoint
# is a safe split rather than a tuned one.
CLOSED_OPEN_FRACTION = 0.5


def envelope_state(
    kin: MobileFR3Kinematics,
    arm_q: np.ndarray,
    base_xy_yaw: np.ndarray,
    spine_m: float,
    pad_xyz: np.ndarray | None,
    reference_quat_xyzw: np.ndarray | None,
    max_dist_m: float,
    max_angle_deg: float,
) -> tuple[bool, float, float] | None:
    """(inside, dist_m, angle_deg). None when pad_xyz is None.

    The single definition of "inside the F-98 envelope" in this codebase.
    Both the actuator (`GraspGate`) and the instrument (`GraspObserver`)
    call it, so a rollout's "did it grasp in the right place" reading is
    the same geometric test the gate would have fired on — two copies of
    this rule would be two silently different envelopes.

    ``None`` (no pad pose yet) is deliberately distinct from ``inside is
    False``: not knowing where the pad is, is not the same as knowing the
    TCP is far from it. Callers must not collapse the two.
    """
    if pad_xyz is None:  # no ground-truth pose yet — never guess one
        return None
    xyz, quat = kin.fk(arm_q, base_xy_yaw, float(spine_m))
    dist_m = float(np.linalg.norm(xyz - np.asarray(pad_xyz, dtype=np.float64)))
    # No reference attitude configured = orientation does not vote. 0.0 here
    # is a sentinel for "not checked", not a measured alignment; GraspObserver
    # reports it as None rather than as a perfect 0.0 for that reason.
    angle_deg = (
        0.0 if reference_quat_xyzw is None else quat_xyzw_angle_deg(quat, reference_quat_xyzw)
    )
    inside = bool(dist_m <= max_dist_m and angle_deg <= max_angle_deg)
    return inside, dist_m, angle_deg


@dataclass
class GraspGate:
    """Decide the right gripper from geometry when the policy will not.

    ``update()`` returns the commanded open fraction (1.0 open, 0.0 closed)
    or ``None`` to leave the policy's own value alone — which is what it
    does before the envelope is first entered, so the ablation only ever
    *adds* a close.
    """

    max_dist_m: float = DEFAULT_MAX_DIST_M
    max_angle_deg: float = DEFAULT_MAX_ANGLE_DEG
    reference_quat_xyzw: np.ndarray | None = None
    latch: bool = LATCH
    kin: MobileFR3Kinematics = field(default_factory=MobileFR3Kinematics)

    closed: bool = field(default=False, init=False)
    fired_at_sim_s: float | None = field(default=None, init=False)
    last_dist_m: float = field(default=float("nan"), init=False)
    last_angle_deg: float = field(default=float("nan"), init=False)

    def reset(self) -> None:
        self.closed = False
        self.fired_at_sim_s = None
        self.last_dist_m = float("nan")
        self.last_angle_deg = float("nan")

    def update(
        self,
        arm_q: np.ndarray,
        base_xy_yaw: np.ndarray,
        spine_m: float,
        pad_xyz: np.ndarray | None,
        t_sim: float,
    ) -> float | None:
        """None = defer to the policy; 0.0 = close; 1.0 = open."""
        # The latch short-circuits BEFORE the geometry: once closed the gate
        # stops measuring entirely, which is why last_dist_m/last_angle_deg
        # below are last-tick samples and why the honest approach reading
        # lives in GraspObserver instead.
        if self.latch and self.closed:
            return 0.0
        state = envelope_state(
            self.kin,
            arm_q,
            base_xy_yaw,
            spine_m,
            pad_xyz,
            self.reference_quat_xyzw,
            self.max_dist_m,
            self.max_angle_deg,
        )
        if state is None:  # no ground-truth pose yet — never guess one
            return None
        inside, self.last_dist_m, self.last_angle_deg = state
        if inside:
            if not self.closed:
                self.fired_at_sim_s = float(t_sim)
            self.closed = True
            return 0.0
        return None

    def stats(self) -> dict:
        return {
            "grasp_gate_fired": self.closed,
            "grasp_gate_fired_at_sim_s": self.fired_at_sim_s,
            "grasp_gate_last_dist_m": self.last_dist_m,
            "grasp_gate_last_angle_deg": self.last_angle_deg,
        }


def pad_in_tcp_frame_mm(
    kin: MobileFR3Kinematics,
    arm_q: np.ndarray,
    base_xy_yaw: np.ndarray,
    spine_m: float,
    pad_xyz: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """(offset_mm (3,), quat_xyzw) of the pad expressed in the right-TCP frame.

    x = finger-closing axis, y = across the fingers, z = approach axis,
    positive = ahead of the TCP -- the same convention
    ``tools/dgx_probes/grasp_pose_envelope.py`` measures ``tcp_frame_mm``
    in, so a caller comparing a live pad against that box is comparing two
    readings taken in the same frame rather than world xyz against a
    TCP-relative box, which would silently rotate with every arm pose.

    ``None`` when ``pad_xyz`` is ``None``, for the same reason
    ``envelope_state`` returns ``None``: not knowing where the pad is must
    never collapse into a manufactured offset.
    """
    if pad_xyz is None:
        return None
    tcp_xyz, quat = kin.fk(arm_q, base_xy_yaw, float(spine_m))
    rot = quat_xyzw_to_rot(quat)
    offset_mm = rot.T @ (np.asarray(pad_xyz, dtype=np.float64) - tcp_xyz) * 1000.0
    return offset_mm, quat


def box_excess_mm(
    offset_mm: np.ndarray, box_min_mm: np.ndarray, box_max_mm: np.ndarray
) -> float:
    """Euclidean distance from ``offset_mm`` to the axis-aligned box; 0.0 inside.

    The "how far from grasp geometry" reading for a tick the box test
    rejected -- clamp each axis into the box, then take the distance to
    the clamped point. Not the test itself (that stays a hard per-axis
    <= / >=): this is diagnostic bookkeeping for ``min_box_excess_mm``,
    the number that answers "how close did the pad-in-box condition get".
    """
    offset = np.asarray(offset_mm, dtype=np.float64)
    lo = np.asarray(box_min_mm, dtype=np.float64)
    hi = np.asarray(box_max_mm, dtype=np.float64)
    clamped = np.clip(offset, lo, hi)
    return float(np.linalg.norm(offset - clamped))


@dataclass
class ConditionedGraspGate(GraspGate):
    """C1b: fire on finger GEOMETRY and/or a DWELL, not the isotropic sphere.

    Measured offline (protocol C1b): the plain `GraspGate` fired with the pad
    beside a finger, 40 mm ahead of the TCP (close on air, still inside the
    45 mm ball), or flying through at 343 mm/s. The demos instead close with
    the pad, in the TCP frame, inside a small box near the fingers, nearly
    stationary, after dwelling -- three separate facts the isotropic
    sphere-and-angle test cannot express because it has no notion of
    direction or of time.

    This is a SUPERSET, not a rewrite: with ``box_min_mm=None,
    box_max_mm=None, dwell_s=0.0`` every code path collapses back to
    `GraspGate`'s (`ok = inside_iso`, fire on the first ok tick), so the
    "entry" mode this produces is BYTE-FOR-BYTE the C1 gate and every
    existing `GraspGate` test doubles as a parity check on it.

    ``mode`` is a label only (set by `conditioned_from_envelope`) -- it
    changes nothing here, but it is echoed in `stats()` so a results row
    states which bar the gate fired on without the reader having to infer
    it from whether `grasp_gate_box_min_mm` happens to be null.
    """

    box_min_mm: np.ndarray | None = None
    box_max_mm: np.ndarray | None = None
    dwell_s: float = 0.0
    mode: str = "entry"

    # Pre-fire bookkeeping only -- like the base class, this gate stops
    # measuring the instant it latches (see update()), so none of these
    # describe the approach, only the tick that satisfied (or nearly
    # satisfied) the condition.
    ok_since_sim_s: float | None = field(default=None, init=False)
    last_offset_mm: np.ndarray | None = field(default=None, init=False)
    fire_offset_mm: np.ndarray | None = field(default=None, init=False)
    fire_dist_m: float | None = field(default=None, init=False)
    fire_angle_deg: float | None = field(default=None, init=False)
    min_box_excess_mm: float | None = field(default=None, init=False)
    min_box_excess_sim_s: float | None = field(default=None, init=False)
    min_box_excess_offset_mm: np.ndarray | None = field(default=None, init=False)
    max_ok_run_s: float = field(default=0.0, init=False)

    def reset(self) -> None:
        super().reset()
        self.ok_since_sim_s = None
        self.last_offset_mm = None
        self.fire_offset_mm = None
        self.fire_dist_m = None
        self.fire_angle_deg = None
        self.min_box_excess_mm = None
        self.min_box_excess_sim_s = None
        self.min_box_excess_offset_mm = None
        self.max_ok_run_s = 0.0

    def update(
        self,
        arm_q: np.ndarray,
        base_xy_yaw: np.ndarray,
        spine_m: float,
        pad_xyz: np.ndarray | None,
        t_sim: float,
    ) -> float | None:
        """None = defer to the policy; 0.0 = close.

        Condition: the isotropic sphere+angle test when there is no box
        (identical to the base class); otherwise the box REPLACES the
        distance half while the angle veto stays, because the box already
        encodes "close enough" -- adding a second, looser distance test on
        top of it would let the sphere override a box rejection. Then a
        dwell: `ok` must hold for `dwell_s` consecutive sim seconds before
        firing, so a fly-through tick that happens to satisfy the box for
        one sample cannot latch a close on its own.
        """
        # Same short-circuit as the base class, and for the same reason:
        # once closed the gate stops measuring, so everything below this
        # point is a PRE-FIRE reading, never an account of the approach.
        if self.latch and self.closed:
            return 0.0
        state = envelope_state(
            self.kin,
            arm_q,
            base_xy_yaw,
            spine_m,
            pad_xyz,
            self.reference_quat_xyzw,
            self.max_dist_m,
            self.max_angle_deg,
        )
        if state is None:  # no ground-truth pose yet — never guess one
            return None
        inside_iso, self.last_dist_m, self.last_angle_deg = state

        # pad_xyz is not None here (state was not None), so this cannot be
        # None either -- computed unconditionally, box or not, so entry and
        # dwell mode also get to report where the pad was on fire.
        offset_mm, _ = pad_in_tcp_frame_mm(self.kin, arm_q, base_xy_yaw, spine_m, pad_xyz)
        self.last_offset_mm = offset_mm

        if self.box_min_mm is None:
            ok = inside_iso
        else:
            excess = box_excess_mm(offset_mm, self.box_min_mm, self.box_max_mm)
            if self.min_box_excess_mm is None or excess < self.min_box_excess_mm:
                self.min_box_excess_mm = excess
                self.min_box_excess_sim_s = float(t_sim)
                self.min_box_excess_offset_mm = offset_mm.copy()
            ok = bool(
                self.last_angle_deg <= self.max_angle_deg
                and np.all(offset_mm >= self.box_min_mm)
                and np.all(offset_mm <= self.box_max_mm)
            )

        t = float(t_sim)
        if ok:
            if self.ok_since_sim_s is None:  # first ok tick of a fresh run
                self.ok_since_sim_s = t
            run_s = t - self.ok_since_sim_s
            if run_s > self.max_ok_run_s:
                self.max_ok_run_s = run_s
        else:
            # Any not-ok tick breaks the run: a fly-through that grazes the
            # box for one sample must not bank credit toward the dwell.
            self.ok_since_sim_s = None

        if ok and (t - self.ok_since_sim_s) >= self.dwell_s:
            if not self.closed:
                self.fired_at_sim_s = t
            self.closed = True
            self.fire_offset_mm = offset_mm.copy()
            self.fire_dist_m = self.last_dist_m
            self.fire_angle_deg = self.last_angle_deg
            return 0.0
        return None

    def stats(self) -> dict:
        """Base `GraspGate.stats()` plus the C1b reading. JSON/CSV friendly:
        no numpy arrays, only lists/floats/None/str.
        """
        stats = super().stats()
        stats.update({
            "grasp_gate_mode": self.mode,
            "grasp_gate_dwell_s": float(self.dwell_s),
            "grasp_gate_box_min_mm": (
                None if self.box_min_mm is None
                else np.asarray(self.box_min_mm, dtype=np.float64).tolist()
            ),
            "grasp_gate_box_max_mm": (
                None if self.box_max_mm is None
                else np.asarray(self.box_max_mm, dtype=np.float64).tolist()
            ),
            "grasp_gate_fire_offset_mm": (
                None if self.fire_offset_mm is None
                else np.asarray(self.fire_offset_mm, dtype=np.float64).tolist()
            ),
            "grasp_gate_fire_dist_m": self.fire_dist_m,
            "grasp_gate_fire_angle_deg": self.fire_angle_deg,
            "grasp_gate_min_box_excess_mm": self.min_box_excess_mm,
            "grasp_gate_min_box_excess_sim_s": self.min_box_excess_sim_s,
            "grasp_gate_min_box_excess_offset_mm": (
                None if self.min_box_excess_offset_mm is None
                else np.asarray(self.min_box_excess_offset_mm, dtype=np.float64).tolist()
            ),
            "grasp_gate_max_ok_run_s": float(self.max_ok_run_s),
        })
        return stats


def envelope_kwargs(payload: dict, **overrides) -> dict:
    """Thresholds + reference attitude for a gate or an observer.

    **The thresholds are the frozen constants, not the JSON's
    `suggested_gate`** (ruled 2026-08-25; CONTRACTS.md). This inverts what
    this module used to do, and the inversion is the point: the generator
    derives `suggested_gate.max_dist_mm` as `ceil(36.77/5)*5 = 40` from the
    demo set alone, but 45 comes from a ROLLOUT observation the generator
    cannot see -- an arm that reached 41.8 mm, the case the gate exists to
    catch. So 40 is not a measurement that disagrees with 45; it is a
    number that cannot express 45. Deriving the threshold from the demos
    silently excluded the very case the docstring above claims it catches.

    The reference attitude IS taken from the payload: that one is measured,
    and there is no constant to freeze it against.

    A disagreement is announced rather than silently resolved, because the
    artifact on disk still says 40 and someone will wonder why the gate
    fires at 45.
    """
    gate = payload.get("suggested_gate", {})
    ref = payload.get("reference_quat_xyzw")
    suggested_mm = gate.get("max_dist_mm")
    if suggested_mm is not None and abs(float(suggested_mm) / 1000.0 - DEFAULT_MAX_DIST_M) > 1e-9:
        print(
            f"[camelo] grasp envelope: using the frozen "
            f"{DEFAULT_MAX_DIST_M * 1000:.0f} mm, not the artifact's "
            f"suggested_gate.max_dist_mm = {float(suggested_mm):.0f} mm "
            f"(CONTRACTS.md; the demo-derived rounding cannot express 45)"
        )
    kwargs = {
        "max_dist_m": DEFAULT_MAX_DIST_M,
        "max_angle_deg": DEFAULT_MAX_ANGLE_DEG,
        "reference_quat_xyzw": None if ref is None else np.asarray(ref, dtype=np.float64),
    }
    kwargs.update(overrides)
    return kwargs


def from_envelope(payload: dict, **overrides) -> GraspGate:
    """Build a gate from `grasp_pose_envelope.py`'s JSON."""
    return GraspGate(**envelope_kwargs(payload, **overrides))


GRASP_GATE_MODES = ("entry", "dwell", "geom", "geom+dwell")


def conditioned_from_envelope(
    payload: dict,
    *,
    mode: str,
    dwell_s: float = 0.0,
    box_margin_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    **overrides,
) -> ConditionedGraspGate:
    """Build the C1b `ConditionedGraspGate` `mode` names, from the same JSON `from_envelope` uses.

    Four modes isolate which half of the C1 gate's failure (fires beside a
    finger, on air 40 mm ahead, or mid-flight at 343 mm/s) each fix buys
    back, one variable at a time:

        entry       box None,  dwell 0.0   -- IS `from_envelope`, byte for byte
        dwell       box None,  dwell_s     -- only "hold still" added
        geom        box,       dwell 0.0   -- only "in finger geometry" added
        geom+dwell  box,       dwell_s     -- both

    so a run's log line and stats state which bar it fired on, and neither
    knob can silently smuggle in the other one's effect.

    The box (modes containing "geom") comes from the payload's
    ``tcp_frame_mm.min/max`` -- MEASURED from the demos, like
    ``reference_quat_xyzw``, unlike the frozen `DEFAULT_MAX_DIST_M` -- widened
    by `box_margin_mm` per axis (min side subtracts, max side adds), because
    the finger box was measured at grasp frames only and a margin of zero
    would reject a demonstrator's own approach a tick either side of it.
    """
    if mode not in GRASP_GATE_MODES:
        raise SystemExit(
            f"--grasp-gate-mode must be one of {', '.join(GRASP_GATE_MODES)}, got {mode!r}"
        )
    kwargs = envelope_kwargs(payload, **overrides)
    box_min_mm: np.ndarray | None = None
    box_max_mm: np.ndarray | None = None
    if "geom" in mode:
        tcp_frame = payload.get("tcp_frame_mm") or {}
        if "min" not in tcp_frame or "max" not in tcp_frame:
            raise SystemExit(
                f"--grasp-gate-mode {mode!r} needs payload['tcp_frame_mm']['min'/'max'] -- "
                "regenerate outputs/probes/grasp_pose_envelope.json with "
                "tools/dgx_probes/grasp_pose_envelope.py"
            )
        margin = np.asarray(box_margin_mm, dtype=np.float64)
        box_min_mm = np.asarray(tcp_frame["min"], dtype=np.float64) - margin
        box_max_mm = np.asarray(tcp_frame["max"], dtype=np.float64) + margin
    return ConditionedGraspGate(
        box_min_mm=box_min_mm,
        box_max_mm=box_max_mm,
        dwell_s=float(dwell_s) if "dwell" in mode else 0.0,
        mode=mode,
        **kwargs,
    )


def observer_from_envelope(payload: dict, **overrides) -> GraspObserver:
    """Build a passive observer on the SAME envelope a gate would use.

    This exists so C1's gate-OFF arm can measure against 45 mm AND 7 deg.
    Without it the observer falls back to `reference_quat_xyzw = None`, and
    `envelope_state` then takes the "orientation did not vote" path -- so
    "inside the envelope" is decided on DISTANCE ALONE while the gate-on
    arm tests both. That is the same one-variable break as the 40-vs-45 mm
    split, one unit over, and it would survive fixing the millimetres.
    """
    return GraspObserver(**envelope_kwargs(payload, **overrides))


@dataclass
class GraspObserver:
    """Passive counterpart to GraspGate: measures, never commands, never latches.

    Answers the protocol's first sim intermediate metric — *did a gripper
    close fire while the right TCP was inside the F-98 envelope* — for a
    rollout the gate is not driving. It touches no command: `update()`
    returns nothing, so wiring it into the loop cannot change what reaches
    the wire and the number it produces is a measurement of the policy, not
    of the ablation.

    Two properties are load-bearing and both are the gate's bugs inverted:

    * **No latch.** The gate stops recomputing geometry once it fires
      (`grasp_gate_last_dist_m` = 2.77 m in the graspgate_n2 run — the last
      tick, long after the arm left). The observer keeps measuring for the
      whole episode, so `grasp_env_min_dist_m` is the *closest approach*.
    * **A close is an EDGE, not a level.** `grasp_close_any` counts a
      falling crossing of CLOSED_OPEN_FRACTION, so a channel that sits
      closed from tick 0 — which is exactly what a state-copying policy
      produces when the state starts closed (F-97's ABSENT signature) —
      never reads as a grasp attempt.

    Anything never observed reports ``None``, never ``0.0``: a 0.0 min
    distance would claim the TCP was on the pad. The floored-IoU episode
    (`c18c4b9`, protocol §0) is the cost of a sentinel that reads like a
    measurement.
    """

    max_dist_m: float = DEFAULT_MAX_DIST_M
    max_angle_deg: float = DEFAULT_MAX_ANGLE_DEG
    reference_quat_xyzw: np.ndarray | None = None
    kin: MobileFR3Kinematics = field(default_factory=MobileFR3Kinematics)

    observed_ticks: int = field(default=0, init=False)
    min_dist_m: float | None = field(default=None, init=False)
    min_angle_deg: float | None = field(default=None, init=False)
    ticks_inside: int = field(default=0, init=False)
    first_entry_sim_s: float | None = field(default=None, init=False)
    close_any_sim_s: float | None = field(default=None, init=False)
    close_in_env_sim_s: float | None = field(default=None, init=False)
    prev_gripper_cmd: float | None = field(default=None, init=False)

    def reset(self) -> None:
        self.observed_ticks = 0
        self.min_dist_m = None
        self.min_angle_deg = None
        self.ticks_inside = 0
        self.first_entry_sim_s = None
        self.close_any_sim_s = None
        self.close_in_env_sim_s = None
        # None, not 1.0: without a previous tick there is no edge to detect,
        # so episode 2 cannot inherit episode 1's last command as a crossing.
        self.prev_gripper_cmd = None

    def update(
        self,
        arm_q: np.ndarray,
        base_xy_yaw: np.ndarray,
        spine_m: float,
        pad_xyz: np.ndarray | None,
        right_gripper_cmd: float,
        t_sim: float,
    ) -> None:
        """Record one tick. Returns nothing — this instrument never commands.

        ``right_gripper_cmd`` is the canonical open fraction ACTUALLY SENT
        to the wire this tick (1.0 open, 0.0 closed), not the policy's raw
        chunk value: what we are asking is whether a close reached the
        robot, so the reading must be taken where the command leaves.
        """
        cmd = float(right_gripper_cmd)
        prev = self.prev_gripper_cmd
        self.prev_gripper_cmd = cmd
        closing = (
            prev is not None and prev >= CLOSED_OPEN_FRACTION and cmd < CLOSED_OPEN_FRACTION
        )
        # The FIRST crossing is the grasp attempt; later ones are regrips or
        # chatter and must not overwrite when the attempt happened. Recorded
        # even with no pad pose — a close on empty air is still a close, and
        # hiding it would make an untimed policy look like it never fired.
        if closing and self.close_any_sim_s is None:
            self.close_any_sim_s = float(t_sim)

        state = envelope_state(
            self.kin,
            arm_q,
            base_xy_yaw,
            spine_m,
            pad_xyz,
            self.reference_quat_xyzw,
            self.max_dist_m,
            self.max_angle_deg,
        )
        if state is None:
            return  # pad pose unknown: nothing was observed, so record nothing
        inside, dist_m, angle_deg = state
        self.observed_ticks += 1
        if self.min_dist_m is None or dist_m < self.min_dist_m:
            self.min_dist_m = dist_m
        if self.reference_quat_xyzw is not None:
            if self.min_angle_deg is None or angle_deg < self.min_angle_deg:
                self.min_angle_deg = angle_deg
        if inside:
            self.ticks_inside += 1
            if self.first_entry_sim_s is None:
                self.first_entry_sim_s = float(t_sim)
            if closing and self.close_in_env_sim_s is None:
                self.close_in_env_sim_s = float(t_sim)

    def stats(self) -> dict:
        """The §0 metric-1 row. Thresholds are echoed on purpose.

        A distance without the envelope it was judged against is not a
        result: this repo currently holds two envelopes — 40.0 mm in
        outputs/probes/grasp_pose_envelope.json's `suggested_gate` and
        DEFAULT_MAX_DIST_M = 45 mm here — and a run that does not say which
        one it used cannot be compared with one that does.
        """
        return {
            "grasp_env_max_dist_m": float(self.max_dist_m),
            "grasp_env_max_angle_deg": float(self.max_angle_deg),
            "grasp_env_observed_ticks": int(self.observed_ticks),
            "grasp_env_min_dist_m": self.min_dist_m,
            "grasp_env_min_angle_deg": self.min_angle_deg,
            "grasp_env_ticks_inside": int(self.ticks_inside),
            "grasp_env_entered": self.ticks_inside > 0,
            "grasp_env_first_entry_sim_s": self.first_entry_sim_s,
            "grasp_close_any": self.close_any_sim_s is not None,
            "grasp_close_any_sim_s": self.close_any_sim_s,
            "grasp_close_in_env": self.close_in_env_sim_s is not None,
            "grasp_close_in_env_sim_s": self.close_in_env_sim_s,
        }
