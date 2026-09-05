"""Debug rendering: annotated head frame on the left, BEV floor plan right.

Everything derived from an estimate is drawn **only when there is one** —
no ego wedge, no reprojected box, no trail before the first accepted solve.
A dump that shows a box therefore also shows a pose that explained it.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from camelo.control.perception.geometry import (
    HEAD_HFOV_DEG,
    T_base_cam,
    T_world_base,
    head_intrinsics,
)
from camelo.control.perception.table import TOP_INDICES

SEGMENT_RGB = (90, 120, 150)
SURFACE_RGB = (255, 240, 120)
FOOT_RGB = (0, 220, 255)
TOP_RGB = (255, 0, 200)
BOX_RGB = (255, 160, 0)
TABLETOP_RGB = (255, 220, 70)
GT_BOX_RGB = (180, 80, 255)
INLIER_RGB = (60, 255, 120)
GOAL_RGB = (255, 0, 0)
TRAIL_RGB = (150, 150, 150)
WALL_RGB = (200, 120, 90)
PATH_RGB = (0, 255, 200)
TEXT_RGB = (255, 255, 255)
# §1.2 pose channels not already covered by GT_BOX_RGB (gt) / BOX_RGB (fused):
# picked far from both in hue and from every colour above so a channel is
# readable by colour alone before marker shape (draw order) is even needed.
PERCEPTION_RGB = (255, 90, 240)
ODOMETRY_RGB = (120, 255, 90)

# Cubicle interior (walls.TASK2_CUBICLE_WALLS / doc §1.3.1): x 0.24..5.49,
# y 0.34..3.54, holding the table (2.05, 1.95) and the spawn pose (4.4, 2.6).
# A prior crop narrower than this room clipped walls and off-route
# excursions out of frame. +0.5 m margin on every side, rounded to clean
# numbers — NOT walls.ROOM_BOUNDS_XY (the 25x15 m outer shell): at dump
# resolution the 1.25 m table would render only a few pixels wide.
BEV_X = (-0.3, 6.0)
BEV_Y = (-0.2, 4.0)
_BEV_BG = (24, 28, 32)
_BEV_TABLE_FILL = (70, 78, 62)
_BEV_TABLE_EDGE = (210, 210, 180)
_BEV_WP = (180, 180, 80)
_BEV_WP_CUR = (255, 220, 40)
_BEV_FOV = (80, 180, 255, 70)
_BEV_GT_FOV = (180, 80, 255, 70)
_FOV_REACH_M = 2.4
_GOAL_CIRCLE_M = 0.20
_TRAIL_DASH_M = 0.06

_AABB_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
_TOP_LOOP = ((4, 5), (5, 6), (6, 7), (7, 4))
_Z_NEAR = 1e-3
_LEFT, _RIGHT, _ABOVE, _BELOW = 1, 2, 4, 8


def _outcode(u: float, v: float, width: int, height: int) -> int:
    code = 0
    if u < 0.0:
        code |= _LEFT
    elif u > width - 1:
        code |= _RIGHT
    if v < 0.0:
        code |= _ABOVE
    elif v > height - 1:
        code |= _BELOW
    return code


def clip_segment(
    u0: float, v0: float, u1: float, v1: float, width: int, height: int
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Cohen–Sutherland clip of a projected edge to the image rectangle.

    Corners behind the border still contribute the *visible* part of the
    edge — that is the whole point of drawing the box when the table is
    only partly in frame.
    """
    xmin, xmax = 0.0, float(width - 1)
    ymin, ymax = 0.0, float(height - 1)
    c0, c1 = _outcode(u0, v0, width, height), _outcode(u1, v1, width, height)
    for _ in range(8):
        if not (c0 | c1):
            return (u0, v0), (u1, v1)
        if c0 & c1:
            return None
        code = c0 or c1
        if code & _LEFT:
            v = v0 + (v1 - v0) * (xmin - u0) / (u1 - u0)
            u = xmin
        elif code & _RIGHT:
            v = v0 + (v1 - v0) * (xmax - u0) / (u1 - u0)
            u = xmax
        elif code & _ABOVE:
            u = u0 + (u1 - u0) * (ymin - v0) / (v1 - v0)
            v = ymin
        else:
            u = u0 + (u1 - u0) * (ymax - v0) / (v1 - v0)
            v = ymax
        if code == c0:
            u0, v0 = u, v
            c0 = _outcode(u0, v0, width, height)
        else:
            u1, v1 = u, v
            c1 = _outcode(u1, v1, width, height)
    return None


def _world_to_cam(xyz: np.ndarray, inverse: np.ndarray) -> np.ndarray:
    point = inverse @ np.array((xyz[0], xyz[1], xyz[2], 1.0))
    return point[:3]


def _cam_to_uv(
    cam: np.ndarray, intrinsics: tuple[float, float, float, float]
) -> tuple[float, float] | None:
    fx, fy, cx, cy = intrinsics
    if cam[2] <= 1e-9:
        return None
    return fx * cam[0] / cam[2] + cx, fy * cam[1] / cam[2] + cy


def _clip_near(c0: np.ndarray, c1: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    z0, z1 = float(c0[2]), float(c1[2])
    if z0 >= _Z_NEAR and z1 >= _Z_NEAR:
        return c0, c1
    if z0 < _Z_NEAR and z1 < _Z_NEAR:
        return None
    t = (_Z_NEAR - z0) / (z1 - z0)
    hit = c0 + t * (c1 - c0)
    return (hit, c1) if z0 < _Z_NEAR else (c0, hit)


def project_edge(
    p0: np.ndarray,
    p1: np.ndarray,
    inverse: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Visible fragment of a 3D edge, clipped to the image.

    An endpoint behind the camera or outside the frame does not hide the
    rest of the edge: clip to the near plane, project, then clip to the
    pixel rectangle.
    """
    clipped = _clip_near(_world_to_cam(p0, inverse), _world_to_cam(p1, inverse))
    if clipped is None:
        return None
    uv0 = _cam_to_uv(clipped[0], intrinsics)
    uv1 = _cam_to_uv(clipped[1], intrinsics)
    if uv0 is None or uv1 is None:
        return None
    return clip_segment(uv0[0], uv0[1], uv1[0], uv1[1], width, height)


def _visible_tabletop(
    corners: np.ndarray,
    inverse: np.ndarray,
    intrinsics: tuple[float, float, float, float],
) -> list[tuple[float, float]]:
    """Tabletop polygon in pixels, clipped against the camera near plane."""
    cams = [_world_to_cam(corners[i], inverse) for i in TOP_INDICES]
    clipped: list[np.ndarray] = []
    for i, cam in enumerate(cams):
        nxt = cams[(i + 1) % 4]
        inside, nxt_inside = cam[2] >= _Z_NEAR, nxt[2] >= _Z_NEAR
        if inside:
            clipped.append(cam)
        if inside != nxt_inside:
            t = (_Z_NEAR - cam[2]) / (nxt[2] - cam[2])
            clipped.append(cam + t * (nxt - cam))
    out = []
    for cam in clipped:
        uv = _cam_to_uv(cam, intrinsics)
        if uv is not None:
            out.append(uv)
    return out


def _legend(draw, items, xy: tuple[int, int]) -> None:
    x, y = xy
    pad, row_h, width = 6, 18, 188
    draw.rectangle((x, y, x + width, y + pad * 2 + row_h * len(items)), fill=(16, 16, 16))
    for color, label in items:
        draw.rectangle((x + pad, y + pad, x + pad + 14, y + pad + 12), fill=color)
        draw.text((x + pad + 20, y + pad - 1), label, fill=TEXT_RGB)
        y += row_h


def _draw_aabb_wire(
    draw,
    world: np.ndarray,
    inverse: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    width: int,
    height: int,
    *,
    top_rgb: tuple[int, int, int],
    box_rgb: tuple[int, int, int],
    top_width: int = 3,
    box_width: int = 2,
) -> None:
    """Clipped AABB edges. Top first, then posts and rim."""
    for a, b in _TOP_LOOP:
        fragment = project_edge(world[a], world[b], inverse, intrinsics, width, height)
        if fragment is not None:
            draw.line(fragment, fill=top_rgb, width=top_width)
    for a, b in _AABB_EDGES:
        if (a, b) in _TOP_LOOP:
            continue
        fragment = project_edge(world[a], world[b], inverse, intrinsics, width, height)
        if fragment is not None:
            draw.line(fragment, fill=box_rgb, width=box_width)


def _pose_inverse(xy_yaw: tuple[float, float, float], spine_m: float) -> np.ndarray:
    return np.linalg.inv(T_world_base(*xy_yaw) @ T_base_cam(spine_m))


# §1.2 pose channels, in draw order: gt and odometry are context (drawn
# first, so their marks sit underneath); fused — what the robot actually
# drives on — is drawn last so it wins when channels coincide, which is the
# common case for the first few ticks of a run before odometry/vision have
# had a chance to diverge from the seeded start pose.
CHANNELS = ("gt", "odometry", "perception", "fused")
_CHANNEL_RGB = {
    "gt": GT_BOX_RGB,
    "odometry": ODOMETRY_RGB,
    "perception": PERCEPTION_RGB,
    "fused": BOX_RGB,
}
# fused (or perception, before fused has ever been seeded) is the "located"
# channel: gold tabletop fill + full AABB in the head overlay (§3.4). Every
# other present channel there draws as a thin same-colour wire only, or the
# perspective view turns into four stacked boxes.
_PRIMARY_ORDER = ("fused", "perception")


def _xy_yaw(pose) -> tuple[float, float, float]:
    """A plain ``(x, y, yaw)`` tuple, or anything with ``.xy_yaw`` (``EgoEstimate``)."""
    return pose.xy_yaw if hasattr(pose, "xy_yaw") else tuple(pose)


def _resolve_poses(poses, legacy_gt, legacy_ego) -> dict:
    """``poses`` wins outright; otherwise fall back to ``gt``/``ego`` so old
    call sites keep rendering unchanged while callers migrate to ``poses``.
    """
    raw = poses if poses is not None else {"gt": legacy_gt, "fused": legacy_ego}
    return {name: pose for name, pose in raw.items() if pose is not None}


def _draw_path(draw, segments, markers, *, width: int = 3, radius: int = 3) -> None:
    """Polyline + per-sample marker from pixel-space segments/points.

    Shared by the head overlay (segments pre-clipped to the frame) and the
    BEV (segments are the raw consecutive pairs; PIL clips off-canvas draws
    for free, same as the existing trail/wedge drawing).
    """
    for a, b in segments:
        draw.line((*a, *b), fill=PATH_RGB, width=width)
    for u, v in markers:
        draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=PATH_RGB)


def _draw_path_head(draw, path, inverse, intrinsics, width: int, height: int) -> None:
    """Path samples at z=0, projected and clipped like the AABB edges."""
    points = [np.array((s.x, s.y, 0.0)) for s in path]
    segments = (
        project_edge(p0, p1, inverse, intrinsics, width, height)
        for p0, p1 in zip(points, points[1:], strict=False)
    )
    markers = []
    for p in points:
        cam = _world_to_cam(p, inverse)
        if cam[2] >= _Z_NEAR:
            uv = _cam_to_uv(cam, intrinsics)
            if uv is not None:
                markers.append(uv)
    _draw_path(draw, (s for s in segments if s is not None), markers)


def _cmd_lines(cmd, cmd_applied) -> list[str]:
    """Commanded-vs-applied twist readout for §1.4: two lines, either optional."""
    lines = []
    if cmd is not None:
        lines.append("cmd     vx={:+.2f} vy={:+.2f} wz={:+.2f}".format(*cmd))
    if cmd_applied is not None:
        lines.append("applied vx={:+.2f} vy={:+.2f} wz={:+.2f}".format(*cmd_applied))
    return lines


_WIRE_LABEL = {
    "gt": "gt AABB (odom)",
    "odometry": "odometry AABB",
    "perception": "perception AABB (lock)",
}
_PRIMARY_LABEL = {"fused": "table AABB (located)", "perception": "table AABB (perception lock)"}


def draw_head_overlay(
    rgb: np.ndarray,
    features,
    *,
    table,
    ego=None,
    gt=None,
    spine_m: float,
    hud: str = "",
    intrinsics: tuple[float, float, float, float] | None = None,
    walls=(),
    path=(),
    cmd: tuple[float, float, float] | None = None,
    cmd_applied: tuple[float, float, float] | None = None,
    poses=None,
) -> Image.Image:
    """Head frame with segments, corners, and one AABB per present pose channel.

    ``poses`` (§1.2: "gt" / "odometry" / "perception" / "fused") supersedes
    ``ego``/``gt`` outright — pass one or the other, never both, or a channel
    would draw twice. Only the primary channel (fused, or perception before
    fused is ever seeded) gets the gold tabletop fill + full AABB; the rest
    draw as a thin same-colour wire, or four boxes in one perspective frame
    turn to noise.
    """
    canvas = Image.fromarray(
        np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8).copy(), "RGB"
    )
    draw = ImageDraw.Draw(canvas)
    height, width = rgb.shape[:2]
    if intrinsics is None:
        intrinsics = head_intrinsics(width, height)
    world = table.corners()
    raw = _resolve_poses(poses, gt, ego)
    primary = next((n for n in _PRIMARY_ORDER if n in raw), None)

    surface = getattr(features, "surface", None)
    if surface is not None:
        tint = np.asarray(canvas, dtype=np.uint16)
        mask = surface > 0
        tint[mask] = (tint[mask] * 3 + np.array(SURFACE_RGB, np.uint16)) // 4
        canvas = Image.fromarray(tint.astype(np.uint8), "RGB")
        draw = ImageDraw.Draw(canvas)

    for segment in features.segments:
        draw.line((segment.x0, segment.y0, segment.x1, segment.y1), fill=SEGMENT_RGB)

    # Non-primary channels first (debug context, e.g. "the gt box is here
    # and the estimate is not"), so the primary's gold fill ends up on top.
    view_inv: np.ndarray | None = None
    for name in CHANNELS:
        if name not in raw or name == primary:
            continue
        inverse = _pose_inverse(_xy_yaw(raw[name]), spine_m)
        if view_inv is None:
            view_inv = inverse
        color = _CHANNEL_RGB[name]
        _draw_aabb_wire(
            draw, world, inverse, intrinsics, width, height,
            top_rgb=color, box_rgb=color, top_width=2, box_width=2,
        )

    if primary is not None:
        inverse = _pose_inverse(_xy_yaw(raw[primary]), spine_m)
        view_inv = inverse
        top = _visible_tabletop(world, inverse, intrinsics)
        if len(top) >= 3:
            overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            ImageDraw.Draw(overlay).polygon(top, fill=(*TABLETOP_RGB, 80))
            canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
            draw = ImageDraw.Draw(canvas)
        _draw_aabb_wire(
            draw, world, inverse, intrinsics, width, height,
            top_rgb=TABLETOP_RGB, box_rgb=BOX_RGB, top_width=3, box_width=2,
        )

    # Walls and path are world-frame, z=0 geometry: project through whichever
    # camera pose is best known (primary over context), same as the AABB
    # above. Without any pose there is nothing to project through — skip.
    if view_inv is not None:
        for wall in walls:
            fragment = project_edge(
                np.array((wall.x0, wall.y0, 0.0)),
                np.array((wall.x1, wall.y1, 0.0)),
                view_inv, intrinsics, width, height,
            )
            if fragment is not None:
                draw.line(fragment, fill=WALL_RGB, width=2)
        if path:
            _draw_path_head(draw, path, view_inv, intrinsics, width, height)

    for corner in features.corners:
        u, v = corner.u, corner.v
        color = FOOT_RGB if corner.kind == "foot" else TOP_RGB
        if corner.kind == "foot":
            draw.ellipse((u - 7, v - 7, u + 7, v + 7), outline=color, width=2)
            draw.line((u - 10, v, u + 10, v), fill=color, width=2)
        else:
            draw.rectangle((u - 7, v - 7, u + 7, v + 7), outline=color, width=2)

    # The corners the primary pose was actually fitted to — the rest are candidates.
    primary_pose = raw.get(primary)
    for u, v in getattr(primary_pose, "inlier_uv", ()) or ():
        draw.ellipse((u - 4, v - 4, u + 4, v + 4), fill=INLIER_RGB)

    draw.text((12, 12), "\n".join([hud, *_cmd_lines(cmd, cmd_applied)]), fill=TEXT_RGB)
    items = [
        (SEGMENT_RGB, "edge segments"),
        (TOP_RGB, "tabletop corner"),
        (FOOT_RGB, "foot corner"),
    ]
    if surface is not None:
        items.insert(1, (SURFACE_RGB, "tabletop surface"))
    for name in CHANNELS:
        if name in raw and name != primary:
            items.append((_CHANNEL_RGB[name], _WIRE_LABEL[name]))
    if primary is not None:
        items.append((INLIER_RGB, "corner used in fit"))
        items.append((TABLETOP_RGB, "tabletop rectangle"))
        items.append((BOX_RGB, _PRIMARY_LABEL[primary]))
    if view_inv is not None and walls:
        items.append((WALL_RGB, "wall"))
    if view_inv is not None and path:
        items.append((PATH_RGB, "planned path"))
    _legend(draw, items, (12, height - 24 - 18 * len(items)))
    return canvas


def _to_bev(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    u = (x - BEV_X[0]) / (BEV_X[1] - BEV_X[0]) * width
    v = (BEV_Y[1] - y) / (BEV_Y[1] - BEV_Y[0]) * height
    return u, v


def _dotted(draw, points, ppm: float, color) -> None:
    """Dashed polyline with a fixed spacing in metres."""
    step = max(2.0, _TRAIL_DASH_M * ppm)
    for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
        span = math.hypot(x1 - x0, y1 - y0)
        for k in range(int(span // step) + 1):
            t = min(1.0, k * step / span) if span > 0.0 else 0.0
            u, v = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
            draw.ellipse((u - 1.5, v - 1.5, u + 1.5, v + 1.5), fill=color)


def _fov_wedge(xy, pose: tuple[float, float, float], hfov_deg: float) -> list:
    ex, ey, eyaw = pose
    half = math.radians(hfov_deg / 2.0)
    return [
        xy(ex, ey),
        xy(
            ex + _FOV_REACH_M * math.cos(eyaw - half),
            ey + _FOV_REACH_M * math.sin(eyaw - half),
        ),
        xy(
            ex + _FOV_REACH_M * math.cos(eyaw + half),
            ey + _FOV_REACH_M * math.sin(eyaw + half),
        ),
    ]


# gt/fused additionally get the translucent FOV cone: they are the two
# channels an operator reads first (truth, and what the robot drives on).
# perception/odometry rely on marker shape + colour alone, or four
# overlapping washes turn the coincident-pose case into a smear.
_BEV_WEDGE_RGBA = {"gt": _BEV_GT_FOV, "fused": _BEV_FOV}
_MARKER_SHAPE = {"gt": "square", "odometry": "triangle", "perception": "diamond", "fused": "circle"}
_BEV_LABEL = {
    "gt": "gt (sim truth)",
    "odometry": "odometry (start ⊕ Δodom)",
    "perception": "perception-only (last lock)",
    "fused": "fused (driven pose)",
}


def _draw_pose_marker(
    draw, xy, pose: tuple[float, float, float], color, shape: str, r: int = 5
) -> None:
    """Small filled heading marker at a channel's pose.

    Early in a run start ⊕ zero-odometry ⊕ the first vision lock can all
    sit within centimetres of each other — shape carries channel identity
    there, since colour alone is hard to read once markers overlap.
    """
    x, y, yaw = pose
    u, v = xy(x, y)
    tx, ty = xy(x + 0.22 * math.cos(yaw), y + 0.22 * math.sin(yaw))
    if shape == "circle":
        draw.ellipse((u - r, v - r, u + r, v + r), fill=color)
    elif shape == "square":
        draw.rectangle((u - r, v - r, u + r, v + r), fill=color)
    elif shape == "diamond":
        draw.polygon([(u, v - r), (u + r, v), (u, v + r), (u - r, v)], fill=color)
    else:  # triangle
        draw.polygon([(u, v - r), (u + r, v + r), (u - r, v + r)], fill=color)
    draw.line((u, v, tx, ty), fill=color, width=2)


def draw_bev(
    *,
    height: int,
    table,
    waypoints=(),
    wp_index: int = 0,
    ego=None,
    gt=None,
    trail=(),
    walls=(),
    path=(),
    hfov_deg: float | None = None,
    cmd: tuple[float, float, float] | None = None,
    cmd_applied: tuple[float, float, float] | None = None,
    poses=None,
    trails=None,
) -> Image.Image:
    """World-fixed floor plan: one marker + trail per present pose channel.

    ``poses``/``trails`` (§1.2: "gt" / "odometry" / "perception" / "fused")
    supersede ``ego``/``gt``/``trail`` outright — pass one style or the
    other, never both, or a channel draws twice.
    """
    ppm = height / (BEV_Y[1] - BEV_Y[0])
    width = max(1, int(round((BEV_X[1] - BEV_X[0]) * ppm)))
    canvas = Image.new("RGBA", (width, height), (*_BEV_BG, 255))
    draw = ImageDraw.Draw(canvas, "RGBA")

    def xy(x, y):
        return _to_bev(x, y, width, height)

    corners = table.corners()[:4, :2]
    draw.polygon(
        [xy(x, y) for x, y in corners], fill=_BEV_TABLE_FILL, outline=_BEV_TABLE_EDGE
    )

    for wall in walls:
        draw.line((*xy(wall.x0, wall.y0), *xy(wall.x1, wall.y1)), fill=WALL_RGB, width=3)

    for i, wp in enumerate(waypoints):
        u, v = xy(wp.x, wp.y)
        r = 7 if i == wp_index else 5
        draw.ellipse(
            (u - r, v - r, u + r, v + r), fill=_BEV_WP_CUR if i == wp_index else _BEV_WP
        )
    if waypoints:
        goal = waypoints[-1]
        gu, gv = xy(goal.x, goal.y)
        radius = (_GOAL_CIRCLE_M / 2.0) * ppm
        draw.ellipse(
            (gu - radius, gv - radius, gu + radius, gv + radius), outline=GOAL_RGB, width=3
        )
        tip = xy(goal.x + 0.25 * math.cos(goal.yaw), goal.y + 0.25 * math.sin(goal.yaw))
        draw.line((gu, gv, *tip), fill=GOAL_RGB, width=3)

    if path:
        points = [xy(s.x, s.y) for s in path]
        _draw_path(draw, zip(points, points[1:], strict=False), points)

    # Legacy single trail keeps its own grey (matches the pre-channel dump
    # sidecars); the new mapping colours each trail like its channel.
    trail_map = trails if trails is not None else ({"fused": trail} if len(trail) >= 2 else {})
    for name in CHANNELS:
        pts = trail_map.get(name, ())
        if len(pts) >= 2:
            color = TRAIL_RGB if trails is None else _CHANNEL_RGB[name]
            _dotted(draw, [xy(x, y) for x, y, _yaw in pts], ppm, color)

    raw = _resolve_poses(poses, gt, ego)
    fov = HEAD_HFOV_DEG if hfov_deg is None else float(hfov_deg)
    for name in CHANNELS:
        if name not in raw:
            continue
        pose = _xy_yaw(raw[name])
        wedge = _BEV_WEDGE_RGBA.get(name)
        if wedge is not None:
            draw.polygon(_fov_wedge(xy, pose, fov), fill=wedge)
        _draw_pose_marker(draw, xy, pose, _CHANNEL_RGB[name], _MARKER_SHAPE[name])

    status = "BEV located" if "fused" in raw else "BEV searching"
    status_lines = [status, *_cmd_lines(cmd, cmd_applied)]
    draw.text((8, 8), "\n".join(status_lines), fill=TEXT_RGB)
    items = [(_BEV_TABLE_EDGE, "table")]
    if waypoints:
        items.append((GOAL_RGB, "goal 20 cm"))
        items.append((_BEV_WP_CUR, "waypoint"))
    for name in CHANNELS:
        if name in raw:
            items.append((_CHANNEL_RGB[name], _BEV_LABEL[name]))
    if trails is None and trail_map:
        items.append((TRAIL_RGB, "est trajectory"))
    if walls:
        items.append((WALL_RGB, "wall"))
    if path:
        items.append((PATH_RGB, "planned path"))
    # Legend sits below the status block, which grows by a line per cmd_out
    # readout — a fixed offset would let a 3-line block bleed into it.
    _legend(draw, items, (8, 8 + 18 * len(status_lines) + 4))
    return canvas.convert("RGB")


def composite(head: Image.Image, bev: Image.Image) -> Image.Image:
    if bev.size[1] != head.size[1]:
        bev = bev.resize((bev.size[0], head.size[1]))
    out = Image.new("RGB", (head.size[0] + bev.size[0], head.size[1]))
    out.paste(head, (0, 0))
    out.paste(bev, (head.size[0], 0))
    return out
