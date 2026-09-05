"""Pose filter: start pose + relative odometry + occasional vision (§1.2).

The head camera never sees the table from spawn, so a filter that starts
empty and waits for a lock never starts at all — the robot yaw-searches
until it parks in a wall. The fix: seed a rough start pose at ``t=0`` and
propagate it by relative odometry every tick, so a fused pose exists from
the first tick whether or not vision ever locks. A measurement corrects
the fused estimate (gain weighted by the fit's own evidence) when it
clears a rejection test; nothing here waits on ``solve``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from camelo.control.perception.localize import EgoEstimate

# The 180°-symmetric twin's signature (§1.1) is a ~pi yaw error. Genuine
# noise — a rough start yaw, odometry drift over the approach, a noisy fit —
# is nowhere near that; pi/2 sits well clear of both, so it separates the
# twin from a real lock without being tuned to either edge case.
REJECT_YAW_RAD_DEFAULT = math.pi / 2.0
# MEASURED (outputs/approach_dump/2215, tick 171): a marginal fit
# (n_corr=3, residual_px=14, tabletop_iou=0.87) disagreed with the
# odometry-propagated pose by 2.33 m in xy while its yaw (~89.2°) slipped in
# just under REJECT_YAW_RAD_DEFAULT, so it was accepted and the fused pose
# jumped 2.33 m in one tick, permanently (§1.2 gain is no longer flat, see
# _measurement_gain, but the gate is still the first line of defense).
# Replaying reject_xy_m down from the old 6.0 m (the cubicle diagonal,
# ~6.15 m, §1.3.1) across every clean recorded run showed 1.5 m rejects
# that specific measurement with zero effect on any other accepted lock in
# the dataset. Trade-off: this also tightens the "first lock from a rough
# start can be far off" allowance (§1.1) to 1.5 m — a genuinely rough CLI
# start pose beyond that now needs its first lock's *yaw* alone to carry it
# (see test_a_first_lock_beyond_reject_xy_m_is_now_rejected).
REJECT_XY_M_DEFAULT = 1.5

# Quality-weighted correction gain (§1.2), replacing the old flat constant.
# MEASURED (outputs/approach_dump/2215 vs 1738, replayed offline): a flat
# gain forces a single trade-off between "responsive when a lock is good"
# and "safe when it isn't" — dropping it to damp the tick-171 outlier above
# also muted every good correction in the high-lock-frequency runs. Scaling
# gain by the fit's own evidence instead reduced the tick-171 disagreement
# to a 0.41 m nudge (vs a full 2.33 m snap) *and* improved the busiest
# recorded run (1738, 574 corrections): rmse 0.227 m -> 0.193 m, worst
# single-tick jump 0.254 m -> 0.039 m — better on both ends than any single
# flat gain tested.
_GAIN_N_CORR_SATURATE = 6.0  # n_corr at/above this scores full marks
_GAIN_RESIDUAL_DECAY_PX = 15.0  # e-fold falloff of reprojection residual
GAIN_MIN = 0.05  # a passed-gate measurement still nudges the estimate


def _measurement_gain(measurement: EgoEstimate) -> float:
    """How much of ``measurement`` to blend in, from its own fit evidence.

    Three signals already computed by ``TableLocalizer.solve`` and carried
    on every ``EgoEstimate``, multiplied so a fit has to be good on all of
    them to earn a high gain: correspondence count (``n_corr``, saturating
    at ``_GAIN_N_CORR_SATURATE`` — more points fit is more constrained),
    reprojection residual (``residual_px``, exponential falloff — a tight
    fit is trusted, a loose one is not), and tabletop IoU (``tabletop_iou``,
    already in [0, 1] — how much of the detected instance the projected
    model actually explains). Floored at ``GAIN_MIN`` rather than 0: a
    measurement that cleared ``_is_rejected`` is still evidence, just weak
    evidence, so it should nudge the estimate rather than be a no-op.
    """
    n_corr_score = min(1.0, measurement.n_corr / _GAIN_N_CORR_SATURATE)
    residual_score = math.exp(-max(0.0, measurement.residual_px) / _GAIN_RESIDUAL_DECAY_PX)
    iou_score = max(0.0, min(1.0, measurement.tabletop_iou))
    return max(GAIN_MIN, n_corr_score * residual_score * iou_score)


def _wrap(yaw: float) -> float:
    return (yaw + math.pi) % (2.0 * math.pi) - math.pi


def _relative_delta(
    a_xy_yaw: tuple[float, float, float], b_xy_yaw: tuple[float, float, float]
) -> tuple[float, float, float]:
    """``a ⊖ b``: the motion from ``a`` to ``b``, expressed in ``a``'s frame.

    Both poses are readings in the *same* frame (e.g. two odometry samples).
    Subtracting components would give the delta in that shared frame, not in
    ``a``'s local frame — rotate by ``-yaw_a`` so the result composes
    correctly with a *different* base orientation in ``_compose_xy_yaw``.
    """
    xa, ya, yawa = a_xy_yaw
    xb, yb, yawb = b_xy_yaw
    dx, dy = xb - xa, yb - ya
    c, s = math.cos(yawa), math.sin(yawa)
    local_dx = c * dx + s * dy
    local_dy = -s * dx + c * dy
    return local_dx, local_dy, _wrap(yawb - yawa)


def _compose_xy_yaw(
    base_xy_yaw: tuple[float, float, float], delta: tuple[float, float, float]
) -> tuple[float, float, float]:
    """``base ⊕ delta``: apply a local-frame delta at ``base``'s orientation.

    This is the SE(2) composition that makes ``p_odom = start ⊕ (odom(t) ⊖
    odom(t0))`` correct: the delta is rotated into ``base``'s frame before
    being added, so a robot seeded at yaw −90° that drives "forward" in
    odom (local +x) actually moves in −y world, not +x. A componentwise
    ``start + delta`` only matches this when ``base``'s yaw happens to be
    zero — it passes a straight-line test and fails every rotated one.
    """
    x0, y0, yaw0 = base_xy_yaw
    dx, dy, dyaw = delta
    c, s = math.cos(yaw0), math.sin(yaw0)
    x = x0 + c * dx - s * dy
    y = y0 + s * dx + c * dy
    return x, y, _wrap(yaw0 + dyaw)


@dataclass(frozen=True)
class PoseChannels:
    """The four display/dump channels of §1.2 (GT lives outside the filter)."""

    fused: EgoEstimate | None
    odometry: EgoEstimate | None
    perception: EgoEstimate | None  # last ACCEPTED measurement


class PoseFilter:
    """Predict on relative odometry, correct on an accepted measurement.

    Unseeded, ``update`` returns ``None`` and does nothing — there is no
    start pose to propagate yet. After ``seed``, ``pose`` is defined on
    every subsequent tick even if vision never locks.
    """

    def __init__(
        self,
        *,
        reject_yaw_rad: float = REJECT_YAW_RAD_DEFAULT,
        reject_xy_m: float = REJECT_XY_M_DEFAULT,
    ) -> None:
        """Configure the measurement rejection gate.

        An accepted measurement's ``x, y, yaw`` is blended into the
        odometry-predicted pose in ``_correct`` at a gain computed per
        measurement by ``_measurement_gain`` from its own fit evidence
        (``n_corr``, ``residual_px``, ``tabletop_iou``) — a strong fit pulls
        close to ``predicted -> measurement``, a marginal one (like the one
        that motivated this) only nudges it. This is no longer a constructor
        knob: a flat gain forced a single choice between "responsive when a
        lock is good" and "safe when it isn't" (see ``_measurement_gain``'s
        docstring for the measured comparison). Odometry is otherwise
        weighted implicitly at 1.0 — every tick it fully propagates the
        fused pose via ``_compose_xy_yaw`` before any correction is applied.
        Any gain below 1.0 lets a correction persist across ticks rather
        than being overwritten, since the fused channel is carried forward
        from ``self._fused`` (not reset to the odometry pose) on each
        predict.

        ``reject_yaw_rad`` and ``reject_xy_m`` gate which measurements reach
        ``_correct`` at all (see ``_is_rejected``): a measurement whose yaw
        or xy disagrees with the predicted pose by more than these
        thresholds is dropped for that tick (recorded in ``rejected``) and
        the predicted pose is kept unchanged, regardless of gain.
        """
        self._reject_yaw_rad = reject_yaw_rad
        self._reject_xy_m = reject_xy_m
        self._start_xy_yaw: tuple[float, float, float] | None = None
        self._odom0_xy_yaw: tuple[float, float, float] | None = None
        self._last_odom_xy_yaw: tuple[float, float, float] | None = None
        self._fused: EgoEstimate | None = None
        self._odom_pose: EgoEstimate | None = None
        self._perception: EgoEstimate | None = None
        self._rejected: tuple[EgoEstimate, ...] = ()

    def seed(
        self,
        start_xy_yaw: tuple[float, float, float],
        odom_xy_yaw: tuple[float, float, float],
    ) -> None:
        """Start pose at ``t=0``, plus the odom reading at that instant."""
        x, y, yaw = start_xy_yaw
        seed_est = EgoEstimate(
            x=x, y=y, yaw=_wrap(yaw), residual_px=0.0, n_corr=0, source="seed"
        )
        self._start_xy_yaw = (x, y, _wrap(yaw))
        self._odom0_xy_yaw = odom_xy_yaw
        self._last_odom_xy_yaw = odom_xy_yaw
        self._fused = seed_est
        self._odom_pose = seed_est
        self._perception = None
        self._rejected = ()

    def update(
        self,
        *,
        odom_xy_yaw: tuple[float, float, float],
        measurement: EgoEstimate | None = None,
    ) -> EgoEstimate | None:
        """Predict to ``odom_xy_yaw``, then correct on ``measurement``."""
        if self._start_xy_yaw is None or self._fused is None:
            return None  # not seeded yet — nothing to propagate

        # Pure open-loop channel: always start⊕(odom(t)⊖odom(t0)), so it
        # never carries a vision correction and stays comparable across
        # ticks regardless of what the fused channel does.
        odom_x, odom_y, odom_yaw = _compose_xy_yaw(
            self._start_xy_yaw, _relative_delta(self._odom0_xy_yaw, odom_xy_yaw)
        )
        self._odom_pose = EgoEstimate(
            x=odom_x, y=odom_y, yaw=odom_yaw, residual_px=0.0, n_corr=0,
            source="dead-reckon",
        )

        # Fused predict: carry the *fused* pose forward by the odometry
        # increment since the last tick (not reset to p_odom), so a partial
        # correction (gain < 1) stays incorporated instead of being erased
        # by the next predict.
        fused_x, fused_y, fused_yaw = _compose_xy_yaw(
            self._fused.xy_yaw,
            _relative_delta(self._last_odom_xy_yaw, odom_xy_yaw),
        )
        predicted = replace(
            self._fused, x=fused_x, y=fused_y, yaw=fused_yaw, source="dead-reckon"
        )
        self._last_odom_xy_yaw = odom_xy_yaw

        if measurement is None:
            self._fused = predicted
            return self._fused

        if self._is_rejected(predicted, measurement):
            self._rejected = (*self._rejected, measurement)
            self._fused = predicted
            return self._fused

        self._perception = measurement
        self._fused = self._correct(predicted, measurement)
        return self._fused

    def _is_rejected(self, predicted: EgoEstimate, measurement: EgoEstimate) -> bool:
        dyaw = abs(_wrap(measurement.yaw - predicted.yaw))
        dxy = math.hypot(measurement.x - predicted.x, measurement.y - predicted.y)
        return dyaw > self._reject_yaw_rad or dxy > self._reject_xy_m

    def _correct(self, predicted: EgoEstimate, measurement: EgoEstimate) -> EgoEstimate:
        """Blend pose toward ``measurement`` by its own quality-derived gain.

        A high-evidence fit (``_measurement_gain`` near 1.0) makes
        ``x, y, yaw`` land close to the measurement; ``source`` flips to
        "fit" regardless of gain so the fused channel is visibly distinct
        from the raw perception channel (``channels.perception``), which
        keeps the measurement's own source label untouched.
        """
        g = _measurement_gain(measurement)
        yaw = _wrap(predicted.yaw + g * _wrap(measurement.yaw - predicted.yaw))
        x = predicted.x + g * (measurement.x - predicted.x)
        y = predicted.y + g * (measurement.y - predicted.y)
        return replace(measurement, x=x, y=y, yaw=yaw, source="fit")

    @property
    def pose(self) -> EgoEstimate | None:
        return self._fused

    @property
    def channels(self) -> PoseChannels:
        return PoseChannels(
            fused=self._fused, odometry=self._odom_pose, perception=self._perception
        )

    @property
    def rejected(self) -> tuple[EgoEstimate, ...]:
        return self._rejected
