"""Planar odometry helpers shared by the ROS collector and perception.

numpy-only and importable without ``camelo.control.perception`` — that
package pulls in cv2 and PIL at import time, and ``camelo/ros/obs_collector``
needs only these two functions (AGENTS.md hard rule 3: ``camelo/ros`` must
not drag image libraries into every ROS process).
"""

from __future__ import annotations

import math


def wrap_pi(angle: float) -> float:
    """Map radians onto (−π, π]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def interpolate_xy_yaw(
    samples: list[tuple[float, float, float, float]], t: float
) -> tuple[float, float, float] | None:
    """Lerp ``(x, y, yaw)`` at time ``t`` from ``(t, x, y, yaw)`` samples.

    Yaw takes the short arc. Times outside the buffer clamp to the nearest
    sample — the overlay would rather be a little stale than empty.
    """
    if not samples or not math.isfinite(t):
        return None
    if t <= samples[0][0]:
        return samples[0][1], samples[0][2], samples[0][3]
    if t >= samples[-1][0]:
        return samples[-1][1], samples[-1][2], samples[-1][3]
    lo, hi = 0, len(samples) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if samples[mid][0] <= t:
            lo = mid
        else:
            hi = mid
    t0, x0, y0, yaw0 = samples[lo]
    t1, x1, y1, yaw1 = samples[hi]
    span = t1 - t0
    if span <= 1e-9:
        return x1, y1, yaw1
    alpha = (t - t0) / span
    return (
        x0 + alpha * (x1 - x0),
        y0 + alpha * (y1 - y0),
        wrap_pi(yaw0 + alpha * wrap_pi(yaw1 - yaw0)),
    )
