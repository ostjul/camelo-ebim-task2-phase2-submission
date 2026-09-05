"""Fit a spline from ego to goal, sampled every 50 cm, clear of ``ROOM_WALLS``
and the table.

Replaces the fixed 6-waypoint route: the path is fit fresh on every
``plan()`` call from wherever the robot actually is to the fixed Task 2
goal, instead of baking a transit line and an overshoot via into module
constants. See ``perception_based_navigation.md`` §1.3.

Position ``(x, y)`` is a clamped Catmull-Rom through ``start`` and
``goal``, plus any via points the clearance deformation below inserts;
sample yaw is the path's own tangent (forward-looking), except the final
sample, which is pinned to the caller's exact goal yaw so the base turns
onto its final orientation only once it has arrived, not en route. Both
position and yaw choices are deliberately plain — the goal margin against
the north partition is 2.2 cm (§1.3.2), smaller than the localization
error the solve gates tolerate, so precision belongs in staying honest
about that number, not in curve-fitting sophistication.

Velocity ``(vx, vy, wz)`` is the numerical derivative of the fitted path
with respect to arc length, i.e. it assumes unit-speed (1 m/s) travel
along the spline. The PID controller (§1.4) decides how fast to actually
go; this is a normalized heading / turn-rate reference, not a speed claim.

Clearance is checked against ``walls`` (the room partitions) AND, when a
``table`` is passed to ``plan()``, the table's own footprint. The two use
different geometry on purpose: a wall is long and a path only ever meets
one tangentially, so nudging a via directly away from the wall converges
in a step or two; the table is a compact, convex obstacle a path can meet
head-on, where that same nudge has no notion of "which side to go around"
and can walk the crossing point from one edge to the opposite one and
back. ``_table_bypass_vias`` picks a side up front instead; the per-point
push (``_push_via_point`` / ``_nudge_via_from_table``) still runs
afterward on both, as a general-purpose refinement pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from camelo.control.perception.geometry import BASE_RADIUS_M, wrap_pi
from camelo.control.perception.table import TableModel
from camelo.control.perception.walls import ROOM_WALLS, WallSegment

SAMPLE_SPACING_M = 0.50

# Deformation is a bounded nudge, not a planner: give up after this many
# via-point insertions rather than searching indefinitely.
_MAX_DEFORM_ITERS = 6
# Push a deformed via point this far past the exact clearance boundary so
# the next iteration's fine-grained recheck does not immediately re-trip
# on float slop.
_PUSH_MARGIN_M = 0.03
# Resolution the fitted curve is walked at for the clearance check — well
# under the 2.2 cm goal-pose margin (§1.3.2), so a corner a 50 cm output
# sample would skip over is still caught.
_CHECK_RESOLUTION_M = 0.02

# Sample yaw is the path's own tangent (forward-looking — the base faces the
# direction it is driving), not a straight interpolation from start_yaw to
# goal_yaw: that older law set yaw independently of the curve, so a spawn
# and goal that happen to share a yaw (Task 2's does: both -90°) produced a
# path that never commanded any rotation at all while still translating
# sideways the whole route — unintuitive and exactly the behaviour this
# fixes. Every via keeps the pure tangent; only the last sample is pinned to
# the caller's exact ``goal_yaw``, so the base drives the whole route facing
# where it is going and only turns onto the final required orientation once
# it is at the goal position — the last leg, not spread across several vias.
# The PID (§1.4) is what makes that final turn gentle, via its own near-goal
# angular-speed ease-down; the planner's job is only to say where to end up.


@dataclass(frozen=True)
class PathSample:
    x: float
    y: float
    yaw: float
    vx: float
    vy: float
    wz: float


def _catmull_rom(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: float
) -> np.ndarray:
    """Uniform Catmull-Rom position at ``t`` in the ``p1``→``p2`` segment."""
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        2.0 * p1
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


def _dense_polyline(
    control_points: list[np.ndarray], resolution_m: float = _CHECK_RESOLUTION_M
) -> list[tuple[int, float, float]]:
    """Walk the Catmull-Rom curve at ~``resolution_m``.

    Clamped ends (each boundary point duplicated as its own neighbour) so
    the curve passes exactly through ``control_points[0]`` at t=0 and
    ``control_points[-1]`` at the very last t=1 — a straight line falls
    out for granted for the 2-point case with no vias. Each entry is
    ``(segment_index, x, y)`` so a via point can be reinserted after the
    control point pair a violation was found between.

    ``resolution_m`` is a parameter (not always the module default) because
    ``_table_bypass_vias`` below calls this many times per ``plan()`` — once
    per candidate route — to score candidates it then discards; the default
    2 cm resolution there is precision the discarded candidates never need,
    and since ``plan()`` now runs every control tick (not just once, or on a
    rare re-plan), that search cost is paid every tick a route around the
    table is active. The final accepted route is still re-walked at the
    default (fine) resolution by the deficit-check loop in ``plan()``.
    """
    ext = [control_points[0], *control_points, control_points[-1]]
    out: list[tuple[int, float, float]] = []
    last = len(control_points) - 2
    for i in range(len(control_points) - 1):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        chord = float(np.linalg.norm(p2 - p1))
        n = max(20, int(chord / resolution_m) + 1)
        for t in np.linspace(0.0, 1.0, n, endpoint=(i == last)):
            xy = _catmull_rom(p0, p1, p2, p3, float(t))
            out.append((i, float(xy[0]), float(xy[1])))
    return out


def _closest_point(
    px: float, py: float, x0: float, y0: float, x1: float, y1: float
) -> tuple[float, float]:
    dx, dy = x1 - x0, y1 - y0
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return x0, y0
    t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / length_sq))
    return x0 + t * dx, y0 + t * dy


def _worst_violation(
    dense: list[tuple[int, float, float]], walls, radius_m: float
) -> tuple[float, WallSegment, tuple[float, float], int] | None:
    """Largest clearance deficit over every dense point × wall, or ``None``."""
    worst: tuple[float, WallSegment, tuple[float, float], int] | None = None
    for seg_idx, x, y in dense:
        for wall in walls:
            required = radius_m + wall.thickness_m / 2.0
            fx, fy = _closest_point(x, y, wall.x0, wall.y0, wall.x1, wall.y1)
            deficit = required - math.hypot(x - fx, y - fy)
            if deficit > 0.0 and (worst is None or deficit > worst[0]):
                worst = (deficit, wall, (x, y), seg_idx)
    return worst


def _push_via_point(
    control_points: list[np.ndarray],
    violation: tuple[float, WallSegment, tuple[float, float], int],
) -> list[np.ndarray]:
    """Insert a via point pushed off the violated wall, away from its centreline.

    Pushes along the line from the wall's closest point through the
    violating point — i.e. further onto whichever side the path is
    already on, rather than flipping it across the wall — by exactly the
    deficit plus ``_PUSH_MARGIN_M``.
    """
    deficit, wall, (px, py), seg_idx = violation
    fx, fy = _closest_point(px, py, wall.x0, wall.y0, wall.x1, wall.y1)
    away = np.array((px - fx, py - fy))
    norm = float(np.linalg.norm(away))
    if norm < 1e-9:
        # On the centreline itself (e.g. a straight run along the wall):
        # any perpendicular works, so use the wall's own normal.
        away = np.array((-(wall.y1 - wall.y0), wall.x1 - wall.x0))
        norm = float(np.linalg.norm(away))
    away = away / norm
    via = np.array((px, py)) + away * (deficit + _PUSH_MARGIN_M)
    out = list(control_points)
    out.insert(seg_idx + 1, via)
    return out


def _rect_clearance(
    px: float, py: float, cx: float, cy: float, half_w: float, half_h: float
) -> float:
    """Signed distance from ``(px, py)`` to an axis-aligned ``half_w x half_h``
    rectangle centred at ``(cx, cy)``: positive outside (Euclidean distance
    to the nearest boundary point), negative inside (penetration depth).

    Unlike treating the table as a handful of boundary line segments (what
    the wall-clearance check above does, correctly, for a wall — a path
    only ever meets those roughly tangentially), a path can approach the
    table head-on. A line-segment check pushes perpendicular to whichever
    edge it happens to cross, which for a head-on approach points along the
    direction of travel, not out and around the obstacle — the deformation
    loop then just slides the crossing point to the *other* edge and back,
    never converging. Signed distance to the rectangle, and pushing away
    from its centre (below), is well-defined for approach from any angle.
    """
    dx, dy = abs(px - cx) - half_w, abs(py - cy) - half_h
    if dx > 0.0 or dy > 0.0:
        return math.hypot(max(dx, 0.0), max(dy, 0.0))
    return max(dx, dy)


def _worst_table_violation(
    dense: list[tuple[int, float, float]], table: TableModel, radius_m: float
) -> tuple[float, tuple[float, float], int] | None:
    """Largest clearance deficit over every dense point vs. the table footprint."""
    cx, cy = table.origin_xy
    half_w, half_h = table.size_xy[0] / 2.0, table.size_xy[1] / 2.0
    worst: tuple[float, tuple[float, float], int] | None = None
    for seg_idx, x, y in dense:
        deficit = radius_m - _rect_clearance(x, y, cx, cy, half_w, half_h)
        if deficit > 0.0 and (worst is None or deficit > worst[0]):
            worst = (deficit, (x, y), seg_idx)
    return worst


def _table_bypass_vias(
    start_xy: np.ndarray, goal_xy: np.ndarray, table: TableModel, radius_m: float
) -> list[np.ndarray]:
    """Via points routing ``start_xy -> goal_xy`` around the table, or
    ``[]`` if the straight line already clears it.

    The table is one convex, compact obstacle a path can approach from any
    angle (unlike the room's several long partition walls, which a path
    only ever meets tangentially, and which the iterative per-point
    deficit push below already handles well — it has no notion of "which
    side to go around", so on a head-on approach it can walk the crossing
    point from one edge of the table to the opposite edge and back, never
    converging within the deform budget). Picking a route up front avoids
    that.

    A single corner of the radius-expanded footprint is not always enough:
    with start and goal on roughly opposite sides of the table (one south,
    one north — or one west, one east), the *straight legs* through one
    diagonal corner can each individually clear it while the actual
    clamped-end Catmull-Rom curve — whose tangent at the start is pulled
    toward goal's direction, not the via's — still cuts across a corner.
    Routing via **both** corners of one full side (e.g. south-east then
    north-east to pass the east side, or south-west then south-east to
    duck under the south side) keeps the curve's tangents pointed along
    that side the whole way around, so both the four single-corner routes
    and the four side-pair routes are checked, and a single corner is only
    used when it is enough on its own.
    """
    cx, cy = table.origin_xy
    half_w = table.size_xy[0] / 2.0 + radius_m + _PUSH_MARGIN_M
    half_h = table.size_xy[1] / 2.0 + radius_m + _PUSH_MARGIN_M

    def _worst_clearance(control_points: list[np.ndarray]) -> float:
        # Coarser than the module default: this only scores candidates that
        # get thrown away, and `plan()` re-walks whichever one wins at full
        # resolution anyway (see `_dense_polyline`'s docstring).
        dense = _dense_polyline(control_points, resolution_m=5.0 * _CHECK_RESOLUTION_M)
        return min(
            _rect_clearance(x, y, cx, cy, table.size_xy[0] / 2.0, table.size_xy[1] / 2.0)
            for _, x, y in dense
        )

    if _worst_clearance([start_xy, goal_xy]) >= radius_m:
        return []

    corner = lambda sx, sy: np.array((cx + sx * half_w, cy + sy * half_h))  # noqa: E731
    corners = {(sx, sy): corner(sx, sy) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)}
    candidates: list[list[np.ndarray]] = [[c] for c in corners.values()]
    for sx in (-1.0, 1.0):
        pair = [corners[(sx, -1.0)], corners[(sx, 1.0)]]  # full west/east side
        candidates += [pair, pair[::-1]]
    for sy in (-1.0, 1.0):
        pair = [corners[(-1.0, sy)], corners[(1.0, sy)]]  # full south/north side
        candidates += [pair, pair[::-1]]

    def _length(vias: list[np.ndarray]) -> float:
        pts = [start_xy, *vias, goal_xy]
        return sum(float(np.linalg.norm(b - a)) for a, b in zip(pts, pts[1:], strict=False))

    def _score(vias: list[np.ndarray]) -> tuple[float, float]:
        deficit = max(0.0, radius_m - _worst_clearance([start_xy, *vias, goal_xy]))
        return (deficit, _length(vias))

    return min(candidates, key=_score)


def _nudge_via_from_table(
    control_points: list[np.ndarray],
    violation: tuple[float, tuple[float, float], int],
    table: TableModel,
) -> list[np.ndarray]:
    """Refinement only: ``_table_bypass_vias`` already chose which side to
    go around, so a small radial push off the table's centre is enough to
    clear whatever residual clip the fitted curve still has near a corner.
    """
    deficit, (px, py), seg_idx = violation
    cx, cy = table.origin_xy
    away = np.array((px - cx, py - cy))
    norm = float(np.linalg.norm(away))
    if norm < 1e-9:
        away, norm = np.array((0.0, 1.0)), 1.0
    away = away / norm
    via = np.array((px, py)) + away * (deficit + _PUSH_MARGIN_M)
    out = list(control_points)
    out.insert(seg_idx + 1, via)
    return out


def _resample_arclength(
    dense: list[tuple[int, float, float]], spacing: float
) -> list[tuple[float, float]]:
    """Points at ~``spacing`` along the polyline; first and last are exact endpoints."""
    xy = [(x, y) for _, x, y in dense]
    cum = [0.0]
    for a, b in zip(xy, xy[1:], strict=False):
        cum.append(cum[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    total = cum[-1]
    n = max(1, round(total / spacing))
    out = []
    j = 0
    for k in range(n + 1):
        s = total * k / n
        while j < len(cum) - 2 and cum[j + 1] < s:
            j += 1
        seg_len = cum[j + 1] - cum[j]
        alpha = 0.0 if seg_len < 1e-12 else (s - cum[j]) / seg_len
        x = xy[j][0] + alpha * (xy[j + 1][0] - xy[j][0])
        y = xy[j][1] + alpha * (xy[j + 1][1] - xy[j][1])
        out.append((x, y))
    out[0] = xy[0]
    out[-1] = xy[-1]
    return out


class SplinePlanner:
    """Fits a wall-clearing spline from ``start`` to ``goal`` each call."""

    def plan(
        self,
        start: tuple[float, float, float],
        goal: tuple[float, float, float],
        *,
        walls=ROOM_WALLS,
        table: TableModel | None = None,
        radius_m: float = BASE_RADIUS_M,
    ) -> tuple[PathSample, ...]:
        start_xy = np.array(start[:2], dtype=float)
        goal_xy = np.array(goal[:2], dtype=float)
        start_yaw, goal_yaw = float(start[2]), float(goal[2])

        if math.hypot(*(goal_xy - start_xy)) < 1e-9:
            return (PathSample(start[0], start[1], start_yaw, 0.0, 0.0, 0.0),)

        def _worst(dense):
            worst_wall = _worst_violation(dense, walls, radius_m)
            worst_table = None if table is None else _worst_table_violation(dense, table, radius_m)
            if worst_table is None:
                return "wall", worst_wall
            if worst_wall is None or worst_table[0] > worst_wall[0]:
                return "table", worst_table
            return "wall", worst_wall

        control_points = [start_xy, goal_xy]
        if table is not None:
            vias = _table_bypass_vias(start_xy, goal_xy, table, radius_m)
            if vias:
                control_points = [start_xy, *vias, goal_xy]
        dense = _dense_polyline(control_points)
        for _ in range(_MAX_DEFORM_ITERS):
            kind, worst = _worst(dense)
            if worst is None:
                break
            control_points = (
                _push_via_point(control_points, worst)
                if kind == "wall"
                else _nudge_via_from_table(control_points, worst, table)
            )
            dense = _dense_polyline(control_points)
        else:
            kind, worst = _worst(dense)
            if worst is not None and kind == "wall":
                deficit, wall, point, _ = worst
                raise ValueError(
                    f"SplinePlanner: cannot clear wall ({wall.x0:.2f}, {wall.y0:.2f})-"
                    f"({wall.x1:.2f}, {wall.y1:.2f}) near {point} after "
                    f"{_MAX_DEFORM_ITERS} deform attempts; short by {deficit:.3f} m"
                )
            if worst is not None and kind == "table":
                deficit, point, _ = worst
                raise ValueError(
                    f"SplinePlanner: cannot clear the table near {point} after "
                    f"{_MAX_DEFORM_ITERS} deform attempts; short by {deficit:.3f} m"
                )

        points = _resample_arclength(dense, SAMPLE_SPACING_M)
        cum = [0.0]
        for a, b in zip(points, points[1:], strict=False):
            cum.append(cum[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))

        last = len(points) - 1
        lo_hi = []
        for i in range(len(points)):
            if i == 0:
                lo_hi.append((0, min(1, last)))
            elif i == last:
                lo_hi.append((max(0, last - 1), last))
            else:
                lo_hi.append((i - 1, i + 1))

        yaws = [
            math.atan2(points[hi][1] - points[lo][1], points[hi][0] - points[lo][0])
            for lo, hi in lo_hi
        ]
        yaws[-1] = goal_yaw

        samples = []
        for i, (x, y) in enumerate(points):
            lo, hi = lo_hi[i]
            ds = cum[hi] - cum[lo]
            vx = (points[hi][0] - points[lo][0]) / ds
            vy = (points[hi][1] - points[lo][1]) / ds
            wz = wrap_pi(yaws[hi] - yaws[lo]) / ds
            samples.append(PathSample(x, y, yaws[i], vx, vy, wz))
        return tuple(samples)
