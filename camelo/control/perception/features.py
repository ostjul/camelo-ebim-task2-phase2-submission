"""Table features: edges lead, a surface cue vetoes.

Geometry is the primary reading, because colour thresholds are the first
thing to break when the same policy meets a real room: Canny edges, Hough
segments, then two geometric readings of those segments.

- **Tabletop corners** (preferred): a corner is where two non-parallel
  segments *end* at the same place, which is what the rim of a rectangular
  top looks like from any side.
- **Foot corners** (fallback): a square post is two near-vertical edges a
  post-width apart whose bottoms land at the same height; the floor contact
  is the midpoint of those two bottom endpoints.

Geometry alone is not selective enough on a tiled floor: grout lines cross
everywhere and each crossing looks exactly like a corner (measured on
``perception_11``, where the fit followed grout instead of the table). So a
**surface cue** runs as a veto on top of it — a tabletop corner has to sit
on the boundary of a bright, unsaturated region. It is deliberately a
relative, adaptive test (bright *compared to this frame's floor*) rather
than an absolute colour, and it can be switched off with
``detect(image, use_surface_prior=False)`` to see the pure-geometry
behaviour.

Quality is not the point of this module — the *interface* is. ``detect``
returns pixel corners tagged with the plane they live on, which is all
``localize`` consumes, so a better (or learned) detector drops straight in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from camelo.control.perception.table import FOOT, TOP

_CANNY_LO = 40
_CANNY_HI = 120
_BLUR_KSIZE = 5
_HOUGH_THRESHOLD = 60
_HOUGH_MIN_LEN = 40
_HOUGH_MAX_GAP = 12

_VERTICAL_TOL_DEG = 18.0
_HORIZONTAL_TOL_DEG = 35.0
# Two segments meeting at less than this are the same line seen twice.
_MIN_CORNER_ANGLE_DEG = 25.0
# An intersection is a corner only if it sits this close to an endpoint of both.
_CORNER_ENDPOINT_PX = 28.0
_CORNER_MERGE_PX = 18.0
_MAX_CORNERS_PER_KIND = 12

# Square post at approach range: ~12–70 px wide, bottoms level within 25 px.
_POST_MIN_W_PX = 12.0
_POST_MAX_W_PX = 70.0
_POST_BOTTOM_TOL_PX = 25.0
_POST_MIN_H_PX = 60.0
# Tabletop corners sit above the floor line; feet below the horizon.
_TOP_V_MAX_FRAC = 0.80
_FOOT_V_MIN_FRAC = 0.25

# Surface cue. Thresholds are relative to this frame's floor brightness:
# measured on perception_11, floor median V ~116 and tabletop V ~207, while
# the cubicle wall that produced a false lock sits at ~150 and is excluded.
_SURFACE_V_ABOVE_FLOOR = 45
_SURFACE_S_MAX = 60
_FLOOR_SAMPLE_FRAC = 0.55
# Absolute tabletop when the lower frame *is* the table (close-up at the
# goal). perception_11: table V~207, wall ~150, floor ~116. Relative
# contrast dies because the "floor" sample is already tabletop; this
# threshold still separates the table from the cubicle wall.
_SURFACE_V_ABS = 170
_SURFACE_ABS_MIN_FRAC = 0.15
# A corner is a boundary, so its neighbourhood is part surface, part not.
# All-surface (mid-tabletop) and no-surface (mid-floor) are both rejected.
_SURFACE_DISK_PX = 15
_SURFACE_FRAC_LO = 0.10
_SURFACE_FRAC_HI = 0.90
# Silhouette corners: the *convex* outline of the tabletop blob. The table
# is a rectangle, so a bite taken out of the mask (gripper, a pad) is a
# concavity, not a corner — `perception_14/000009` had three of those along
# the right crop, which starved the pose solve of a usable assignment.
# Vertices on the image border are where the table leaves the frame.
_SURFACE_OPEN_PX = 7
_SURFACE_MIN_AREA_FRAC = 0.02
_POLY_EPS_FRAC = 0.02
_BORDER_PX = 8
_SNAP_PX = 20
_MIN_OUTLINE_PX = 40.0


@dataclass(frozen=True)
class Segment:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def length(self) -> float:
        return math.hypot(self.x1 - self.x0, self.y1 - self.y0)

    @property
    def angle_deg(self) -> float:
        """Orientation in [0, 180) — segments have no direction."""
        return math.degrees(math.atan2(self.y1 - self.y0, self.x1 - self.x0)) % 180.0

    @property
    def endpoints(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (self.x0, self.y0), (self.x1, self.y1)

    @property
    def bottom(self) -> tuple[float, float]:
        return (self.x0, self.y0) if self.y0 > self.y1 else (self.x1, self.y1)


@dataclass(frozen=True)
class Corner:
    u: float
    v: float
    kind: str
    support: float = 0.0
    # Fraction of surface pixels around it; 0 when the cue is switched off.
    surface: float = 0.0


@dataclass(frozen=True)
class Features:
    segments: tuple[Segment, ...] = ()
    corners: tuple[Corner, ...] = ()
    # Sides of the tabletop silhouette. Kept apart from ``segments`` because
    # they are a different quality of evidence: a Hough segment is any line
    # in the scene, an outline side is a boundary of the table itself.
    outline: tuple[Segment, ...] = ()
    # Largest connected component of the surface cue — the table instance.
    surface: np.ndarray | None = field(default=None, compare=False, repr=False)

    def of_kind(self, kind: str) -> tuple[Corner, ...]:
        return tuple(c for c in self.corners if c.kind == kind)


def _angle_gap(a: float, b: float) -> float:
    diff = abs(a - b) % 180.0
    return min(diff, 180.0 - diff)


def _is_vertical(seg: Segment) -> bool:
    return _angle_gap(seg.angle_deg, 90.0) <= _VERTICAL_TOL_DEG


def _is_horizontal(seg: Segment) -> bool:
    return _angle_gap(seg.angle_deg, 0.0) <= _HORIZONTAL_TOL_DEG


def line_segments(image: np.ndarray) -> tuple[Segment, ...]:
    """Grayscale → Canny → probabilistic Hough segments."""
    if image is None or image.ndim < 2:
        return ()
    array = np.ascontiguousarray(image, dtype=np.uint8)
    gray = cv2.cvtColor(array[:, :, :3], cv2.COLOR_RGB2GRAY) if array.ndim == 3 else array
    gray = cv2.GaussianBlur(gray, (_BLUR_KSIZE, _BLUR_KSIZE), 0)
    edges = cv2.Canny(gray, _CANNY_LO, _CANNY_HI)
    lines = cv2.HoughLinesP(
        edges,
        1,
        math.pi / 180.0,
        _HOUGH_THRESHOLD,
        minLineLength=_HOUGH_MIN_LEN,
        maxLineGap=_HOUGH_MAX_GAP,
    )
    if lines is None:
        return ()
    return tuple(
        Segment(float(x0), float(y0), float(x1), float(y1))
        for x0, y0, x1, y1 in lines.reshape(-1, 4)
    )


def surface_mask(
    image: np.ndarray,
    *,
    v_abs: int | None = None,
    ignore: tuple[tuple[int, int, int, int], ...] = (),
) -> np.ndarray | None:
    """Bright, unsaturated pixels — the tabletop, not the floor or a wall.

    The brightness threshold is measured against this frame's own floor, so
    it survives a change of exposure or of floor shade; only the *contrast*
    between tabletop and floor has to hold.

    ``v_abs`` replaces that rule with an absolute ``V >= v_abs`` — for a
    scene where the floor sample is not floor (the Munich workbench sits on
    a dark cabinet, so ``floor + 45`` admits the white partitions behind the
    table and fuses them with it; ``V >= 145`` separates them, measured on
    seven real frames, ``configs/rig/perception_munich.yaml``).
    """
    if image is None or image.ndim != 3 or image.shape[2] < 3:
        return None
    rgb = np.ascontiguousarray(image[:, :, :3], dtype=np.uint8)
    hsv = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    saturation, value = hsv[:, :, 1], hsv[:, :, 2]
    if v_abs is not None:
        mask = ((value >= int(v_abs)) & (saturation <= _SURFACE_S_MAX)).astype(np.uint8) * 255
        # ``ignore`` rectangles: the robot's own shell at the lower image
        # corners is as bright as the tabletop and bridges into its blob.
        for x0, y0, x1, y1 in ignore:
            mask[y0:y1, x0:x1] = 0
        return mask
    floor_v = float(np.median(value[int(value.shape[0] * _FLOOR_SAMPLE_FRAC) :, :]))
    bright = value >= floor_v + _SURFACE_V_ABOVE_FLOOR
    relative = (bright & (saturation <= _SURFACE_S_MAX)).astype(np.uint8) * 255
    if float(np.count_nonzero(relative)) >= _SURFACE_MIN_AREA_FRAC * relative.size:
        return relative
    absolute = (
        (value >= _SURFACE_V_ABS) & (saturation <= _SURFACE_S_MAX)
    ).astype(np.uint8) * 255
    if float(np.count_nonzero(absolute)) >= _SURFACE_ABS_MIN_FRAC * absolute.size:
        return absolute
    return relative


def largest_component(mask: np.ndarray) -> np.ndarray:
    """The tabletop *instance*: the largest connected blob, nothing else.

    HSV is a cue, not an instance. A pad, a wall, a lamp all survive the
    brightness test; the table is the one that fills the most of the frame.
    Downstream (silhouette, IoU) must see only that blob, or they fit the
    union of every bright thing.
    """
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8)
    )
    if n_labels <= 1:
        return mask
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == keep, mask, 0).astype(mask.dtype)


def _surface_fraction(mask: np.ndarray, u: float, v: float) -> float:
    radius = _SURFACE_DISK_PX
    height, width = mask.shape[:2]
    u0, u1 = max(0, int(u) - radius), min(width, int(u) + radius + 1)
    v0, v1 = max(0, int(v) - radius), min(height, int(v) + radius + 1)
    if u1 <= u0 or v1 <= v0:
        return 0.0
    patch = mask[v0:v1, u0:u1]
    return float(np.count_nonzero(patch)) / float(patch.size)


def _intersect(a: Segment, b: Segment) -> tuple[float, float] | None:
    d1 = (a.x1 - a.x0, a.y1 - a.y0)
    d2 = (b.x1 - b.x0, b.y1 - b.y0)
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-9:
        return None
    t = ((b.x0 - a.x0) * d2[1] - (b.y0 - a.y0) * d2[0]) / denom
    return a.x0 + t * d1[0], a.y0 + t * d1[1]


def _near_endpoint(seg: Segment, point: tuple[float, float]) -> bool:
    return any(
        math.hypot(point[0] - ex, point[1] - ey) <= _CORNER_ENDPOINT_PX
        for ex, ey in seg.endpoints
    )


def _merge(corners: list[Corner]) -> tuple[Corner, ...]:
    """Strongest first, dropping anything within a merge radius of a keeper."""
    kept: list[Corner] = []
    for corner in sorted(corners, key=lambda c: -c.support):
        if any(
            math.hypot(corner.u - k.u, corner.v - k.v) <= _CORNER_MERGE_PX for k in kept
        ):
            continue
        kept.append(corner)
        if len(kept) >= _MAX_CORNERS_PER_KIND:
            break
    return tuple(kept)


def top_corners(
    segments, width: int, height: int, surface: np.ndarray | None = None
) -> tuple[Corner, ...]:
    """Corners of the tabletop, from its outline where possible.

    The outline of the bright region is a far better corner source than
    pairs of segments, which on a tiled floor fire at every grout crossing.
    Segment intersections stay as the fallback for when the surface cue is
    unavailable or finds nothing.
    """
    if surface is not None:
        found, _outline = silhouette(surface, segments, width, height)
        if found:
            return found
    return _intersection_corners(segments, width, height, surface)


def _on_border(u: float, v: float, width: int, height: int) -> bool:
    return _border_flags(u, v, width, height) != 0


def _border_flags(u: float, v: float, width: int, height: int) -> int:
    flags = 0
    if u <= _BORDER_PX:
        flags |= 1
    if u >= width - _BORDER_PX:
        flags |= 2
    if v <= _BORDER_PX:
        flags |= 4
    if v >= height - _BORDER_PX:
        flags |= 8
    return flags


def _share_a_border(
    u0: float, v0: float, u1: float, v1: float, width: int, height: int
) -> bool:
    """True when both ends lie on the same image edge (a hull crop chord)."""
    a = _border_flags(u0, v0, width, height)
    b = _border_flags(u1, v1, width, height)
    return bool(a and b and (a & b))


def silhouette(
    surface: np.ndarray, segments, width: int, height: int
) -> tuple[tuple[Corner, ...], tuple[Segment, ...]]:
    """Tabletop blob → its polygon corners and its polygon sides.

    Both readings come from the same outline, which is the point: a corner
    is where two of the table's own edges meet, so detecting them together
    keeps them consistent. The blob is convex-hulled first because the
    table is a rectangle — a gripper cutting the mask is a concavity, not
    extra corners. Vertices on the image border are the crop. A side whose
    both ends share one border is the hull chord along that crop; a side
    that spans opposite borders is the table rim leaving the frame.
    """
    kernel = np.ones((_SURFACE_OPEN_PX, _SURFACE_OPEN_PX), np.uint8)
    mask = cv2.morphologyEx(surface, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return (), ()
    blob = max(contours, key=cv2.contourArea)
    if cv2.contourArea(blob) < _SURFACE_MIN_AREA_FRAC * width * height:
        return (), ()
    hull = cv2.convexHull(blob)
    perimeter = cv2.arcLength(hull, True)
    polygon = cv2.approxPolyDP(hull, _POLY_EPS_FRAC * perimeter, True).reshape(-1, 2)
    crossings = _segment_crossings(segments)

    found = []
    for u, v in polygon.astype(float):
        if _on_border(u, v, width, height):
            continue
        # Prefer the edge intersection over the mask vertex: thresholding
        # rounds a corner off by a few pixels, two fitted lines do not.
        u, v = _snap(u, v, crossings)
        found.append(Corner(u, v, TOP, perimeter, 1.0))

    sides = []
    for i in range(len(polygon)):
        (u0, v0), (u1, v1) = polygon[i].astype(float), polygon[(i + 1) % len(polygon)]
        u1, v1 = float(u1), float(v1)
        # Same-border hull chord (the crop), not a rim that exits left and
        # right. Close-up at the goal is the latter — perception_03/000747.
        if _share_a_border(u0, v0, u1, v1, width, height):
            continue
        if math.hypot(u1 - u0, v1 - v0) < _MIN_OUTLINE_PX:
            continue
        sides.append(Segment(u0, v0, u1, v1))
    return _merge(found), tuple(sides)


def _segment_crossings(segments) -> list[tuple[float, float]]:
    points = []
    for i, a in enumerate(segments):
        for b in segments[i + 1 :]:
            if _angle_gap(a.angle_deg, b.angle_deg) < _MIN_CORNER_ANGLE_DEG:
                continue
            point = _intersect(a, b)
            if point is not None:
                points.append(point)
    return points


def _snap(u: float, v: float, crossings) -> tuple[float, float]:
    best, best_d = (u, v), _SNAP_PX
    for cu, cv in crossings:
        distance = math.hypot(cu - u, cv - v)
        if distance < best_d:
            best, best_d = (cu, cv), distance
    return best


def _intersection_corners(
    segments, width: int, height: int, surface: np.ndarray | None
) -> tuple[Corner, ...]:
    """Vertices where two non-parallel segments end together, on a surface edge."""
    v_max = _TOP_V_MAX_FRAC * height
    found: list[Corner] = []
    for i, a in enumerate(segments):
        for b in segments[i + 1 :]:
            if _angle_gap(a.angle_deg, b.angle_deg) < _MIN_CORNER_ANGLE_DEG:
                continue
            if not (_is_horizontal(a) or _is_horizontal(b)):
                continue
            point = _intersect(a, b)
            if point is None:
                continue
            u, v = point
            if not (4.0 <= u <= width - 4.0) or not (0.0 <= v <= v_max):
                continue
            if not (_near_endpoint(a, point) and _near_endpoint(b, point)):
                continue
            fraction = 0.0
            if surface is not None:
                # Grout crossings live in uniform floor and never straddle a
                # surface boundary; a tabletop corner always does.
                fraction = _surface_fraction(surface, u, v)
                if not _SURFACE_FRAC_LO <= fraction <= _SURFACE_FRAC_HI:
                    continue
            found.append(Corner(u, v, TOP, a.length + b.length, fraction))
    return _merge(found)


def foot_corners(segments, width: int, height: int) -> tuple[Corner, ...]:
    """Floor contacts of square posts: two vertical edges, level bottoms."""
    verticals = [s for s in segments if _is_vertical(s) and s.length >= _POST_MIN_H_PX]
    v_min = _FOOT_V_MIN_FRAC * height
    found: list[Corner] = []
    for i, a in enumerate(verticals):
        ax, ay = a.bottom
        for b in verticals[i + 1 :]:
            bx, by = b.bottom
            span = abs(ax - bx)
            if not (_POST_MIN_W_PX <= span <= _POST_MAX_W_PX):
                continue
            if abs(ay - by) > _POST_BOTTOM_TOL_PX:
                continue
            u, v = 0.5 * (ax + bx), 0.5 * (ay + by)
            if v < v_min or v > height - 2.0 or not (4.0 <= u <= width - 4.0):
                continue
            found.append(Corner(u, v, FOOT, a.length + b.length))
    return _merge(found)


def detect(
    image: np.ndarray,
    *,
    use_surface_prior: bool = True,
    surface_v_abs: int | None = None,
    surface_ignore: tuple[tuple[int, int, int, int], ...] = (),
) -> Features:
    """Head RGB (or gray) → segments plus tabletop and foot corners.

    ``surface_v_abs`` / ``surface_ignore`` go to :func:`surface_mask` (site profile).
    """
    if image is None or image.ndim < 2:
        return Features()
    height, width = image.shape[:2]
    segments = line_segments(image)
    surface = (
        surface_mask(image, v_abs=surface_v_abs, ignore=surface_ignore)
        if use_surface_prior
        else None
    )
    if surface is not None:
        surface = largest_component(surface)
    outline: tuple[Segment, ...] = ()
    tops: tuple[Corner, ...] = ()
    if surface is not None:
        tops, outline = silhouette(surface, segments, width, height)
    if not tops:
        tops = _intersection_corners(segments, width, height, surface)
    corners = tops + foot_corners(segments, width, height)
    return Features(
        segments=segments, corners=corners, outline=outline, surface=surface
    )


def undistort(
    rgb: np.ndarray,
    intrinsics: tuple[float, float, float, float] | None,
    dist: tuple[float, ...] | None,
    *,
    new_intrinsics: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Undo lens distortion when CameraInfo publishes a non-zero ``D``.

    Sim cameras are ideal pinhole (``D = 0``); this is a no-op there. By
    default the same ``K`` is the output camera matrix, so projection stays
    in the undistorted pixel frame. ``new_intrinsics`` (e.g. a scaled focal
    length, ``profile.CameraModel.k_out``) keeps the periphery of a barrel-
    distorted frame inside the image; every projection afterwards must use
    that matrix.
    """
    if rgb is None or intrinsics is None or not dist:
        return rgb
    coeffs = np.asarray(dist, dtype=np.float64).ravel()
    if coeffs.size == 0 or float(np.max(np.abs(coeffs))) < 1e-9:
        return rgb
    if coeffs.size < 4:  # OpenCV wants (k1, k2, p1, p2[, k3, …]); a profile may give k1 only
        coeffs = np.pad(coeffs, (0, 4 - coeffs.size))
    fx, fy, cx, cy = (float(v) for v in intrinsics)
    k_mat = np.array(
        ((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)), dtype=np.float64
    )
    if new_intrinsics is None:
        return cv2.undistort(rgb, k_mat, coeffs)
    nfx, nfy, ncx, ncy = (float(v) for v in new_intrinsics)
    new_k = np.array(
        ((nfx, 0.0, ncx), (0.0, nfy, ncy), (0.0, 0.0, 1.0)), dtype=np.float64
    )
    return cv2.undistort(rgb, k_mat, coeffs, None, new_k)
