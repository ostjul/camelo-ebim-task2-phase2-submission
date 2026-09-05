"""When has a commanded arm pose actually been *reached*?

Split out of ``episode_runner`` because that module imports rclpy: this one
is numpy-only, so the settling rule is unit-testable with no ROS, no sim
and no GPU.

The rule is behavioural, not a magic number. The position controller does
not drive the error to zero — it parks the arms at a per-pose steady state
and holds it — so "reached" is defined as *the error has stopped changing*
(a plateau), and the tolerance only classifies that plateau as a settle or
a jam. Measured on the DGX rig (2026-08-15), max joint error after a real
scene reset, held flat for the rest of the observation:

    benchmark ARM_READY_POSE    0.0050 rad
    TASK2_ARM_READY_LEFT/RIGHT  0.0339 rad
    demo pose (start_pose_demo_ep0.json)  0.0138 rad

versus the jammed failure mode this replaced, 0.94-1.07 rad, plateaued
just as flatly. ``ARM_SETTLE_TOL_RAD`` sits between those two clusters
with an order of magnitude of clearance either side; the old 0.02 default
was *below* the TASK2 steady state, so that pose could never pass and
burned its whole timeout by construction.
"""

from __future__ import annotations

import math

import numpy as np

# 3x the worst measured steady-state error above, ~10x below the smallest
# observed jam. Do not tighten below 0.04 without re-measuring: TASK2 settles
# at 0.0339 and stays there.
ARM_SETTLE_TOL_RAD = 0.10

# A plateau must outlast the F-86 snap-back (the stale controller target
# re-asserting itself for ~5 s after a reset), or the snap-back itself could
# be mistaken for a settle. Sim seconds — the arms move in sim time, and the
# sim runs 5-9x slower than wall under eval load.
ARM_SETTLE_WINDOW_S = 5.0
ARM_SETTLE_EPS_RAD = 0.01

SETTLING = "settling"
REACHED = "reached"
STUCK = "stuck"

# No FR3 arm joint can turn a full revolution; a reading past +/-2 pi means
# the joint has wound up in the sim (measured: right j5 at -18.3 rad, still
# spinning to -40.6 by the end of the rollout), not that the arm is 18 rad
# from its target.
JOINT_WIND_LIMIT_RAD = 2.0 * math.pi


class StartPoseNotReached(RuntimeError):
    """An explicitly requested start pose was not reached — the episode did
    not start from the condition the run claims to be testing, so the run is
    invalid and must abort rather than be scored and reported."""


def pose_error(measured, target) -> float:
    """Max per-joint |measured - target| [rad], ignoring NaN joints."""
    measured = np.asarray(measured, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(np.nanmax(np.abs(measured - target)))


def wound_joints(measured) -> list[int]:
    """Indices of joints reading past a full revolution (sim wind-up)."""
    measured = np.asarray(measured, dtype=np.float64)
    return [int(i) for i in np.flatnonzero(np.abs(measured) > JOINT_WIND_LIMIT_RAD)]


def joint_error_detail(measured, target) -> str:
    """Per-joint errors for the log — one number cannot say WHICH joint, and
    that is exactly what made the original failures unreadable."""
    measured = np.asarray(measured, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    err = np.abs(measured - target)
    half = err.size // 2
    worst = int(np.nanargmax(err))
    side, joint = ("L", worst + 1) if worst < half else ("R", worst - half + 1)
    return (
        f"worst {side}j{joint} {err[worst]:.4f} rad "
        f"(measured {measured[worst]:+.4f}, target {target[worst]:+.4f}); "
        f"L={np.round(err[:half], 4).tolist()} R={np.round(err[half:], 4).tolist()}"
    )


def report_unreached(
    measured,
    target,
    verdict: str,
    *,
    is_default: bool,
    tol: float = ARM_SETTLE_TOL_RAD,
    timeout_s: float = 0.0,
) -> str:
    """Explain a start pose the arms never reached.

    Returns the text to warn with when the pose was the default, and RAISES
    ``StartPoseNotReached`` when the caller explicitly asked for it. The
    asymmetry is the whole point: on the default pose one stubborn episode
    should not abort a batch, but an explicit ``--start-pose`` names the
    condition the run exists to measure, and three runs have already been
    completed and reported as a comparison of start poses that were never
    reached.
    """
    error = pose_error(measured, target)
    reason = (
        f"the error plateaued at {error:.4f} rad — more than {tol:.2f} from the "
        "target and no longer improving, so the arms are jammed, not slow"
        if verdict == STUCK
        else f"still {error:.4f} rad away after {timeout_s:.0f} sim-s"
    )
    detail = joint_error_detail(measured, target)
    if is_default:
        return f"{reason}. {detail}"
    raise StartPoseNotReached(
        f"the requested --start-pose was not reached: {reason}. {detail}. "
        "This episode would not start from the pose the run is testing, so the "
        "batch is aborted rather than scored — check the spine height and the "
        "arm path after the reset."
    )


class ArmSettleMonitor:
    """Classifies an error trace as settling / reached / stuck.

    Feed it SIM timestamps: the plateau window has to mean the same thing
    whether the sim is running at 1x or the 0.2x it drops to under eval
    load, and a stalled sim clock then reads as "no plateau yet" rather
    than as a settle.
    """

    def __init__(
        self,
        tol_rad: float = ARM_SETTLE_TOL_RAD,
        window_s: float = ARM_SETTLE_WINDOW_S,
        eps_rad: float = ARM_SETTLE_EPS_RAD,
    ):
        self.tol_rad = float(tol_rad)
        self.window_s = float(window_s)
        self.eps_rad = float(eps_rad)
        self._samples: list[tuple[float, float]] = []

    def reset(self) -> None:
        self._samples.clear()

    def update(self, t_sim: float, error: float) -> str:
        """Add a sample, return SETTLING / REACHED / STUCK."""
        if self._samples and t_sim < self._samples[-1][0]:
            self._samples.clear()  # scene reset rebased the clock mid-wait
        self._samples.append((float(t_sim), float(error)))
        cutoff = t_sim - self.window_s
        # Keep one sample from before the cutoff so the window genuinely
        # SPANS window_s rather than merely containing recent samples.
        while len(self._samples) > 2 and self._samples[1][0] <= cutoff:
            self._samples.pop(0)
        if self._samples[0][0] > cutoff:
            return SETTLING
        errors = [e for _, e in self._samples]
        if max(errors) - min(errors) > self.eps_rad:
            return SETTLING
        return REACHED if error <= self.tol_rad else STUCK
