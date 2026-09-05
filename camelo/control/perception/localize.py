"""Ego pose from table corners: absolute, gated, no previous-pose feedback.

Three properties make this correct where the old two-point fit was not:

1. **One unknown.** The table pose is given (``TableModel``); only the ego
   pose is solved. Fitting both from one corner pair is underdetermined.
2. **No prior.** Corners are unprojected in the *base* frame, which needs no
   ego pose at all, and the pose comes from a closed-form fit. Nothing from
   the previous tick enters, so error cannot accumulate.
3. **It must explain the frame.** Not just the corners it chose — the whole
   picture. A hypothesis is rejected unless

   - its corners reproject onto their detections (``reproj_gate_px``),
   - the tabletop edges it predicts lie along **detected line segments**,
   - every model corner it puts well inside the frame is actually detected
     there (an unseen corner in open view means the pose is wrong),
   - and the table is inside the field of view it implies.

The final pose is polished in **pixel space** rather than in metres: the
closed-form fit minimises error on unprojected points, where a corner ten
metres out weighs the same as one at arm's length, while the quantity that
actually matters is where the model lands in the image. Both stages use the
same rigid model, so the known table height and edge lengths constrain the
three degrees of freedom throughout.
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np

from camelo.control.perception.features import Corner, Features, Segment, _angle_gap
from camelo.control.perception.geometry import (
    T_base_cam,
    T_world_base,
    T_world_cam,
    head_intrinsics,
    project,
    unproject_to_plane,
)
from camelo.control.perception.table import FOOT, TOP, TOP_INDICES, TableModel

log = logging.getLogger(__name__)

# A seed pair is only tried when its 3D spacing matches an AABB edge.
PAIR_TOL_M = 0.15
# Accepted poses must reproject their corners this well (RMS pixels).
REPROJ_GATE_PX = 30.0
# Two correspondences determine a pose exactly, so they fit perfectly under
# *any* assignment and their residual carries no information — that is how
# the old pipeline produced confident poses in an empty room. A lock needs
# a correspondence the fit did not consume.
MIN_CORRESPONDENCES = 3
# Fraction of the predicted tabletop outline that must lie on real edges.
EDGE_SUPPORT_MIN = 0.35
# The projected tabletop (the rectangle the 3D box is fitted from) has to
# overlap the instance mask. A pose that puts the top on empty floor can
# still nail two corners and two edges; IoU is the check those miss.
TOP_IOU_MIN = 0.30
# The known-size box cannot sit as a small patch on a large table: the
# instance has to fall *inside* the projected top. IoU of 0.30 still
# allows a one-corner lock whose AABB covers ~30 % of the blob
# (`perception_17/000028`).
TOP_RECALL_MIN = 0.70
# Unprojected instance points must land on the documented tabletop. A pose
# that is too far spreads those points over a region larger than 1.25×0.74.
FOOTPRINT_MARGIN_M = 0.20
# Model corners this far inside the frame are in open view: if one of those
# has no detection, the detector saw the scene and disagrees with the pose.
IN_FRAME_MARGIN_PX = 60.0
MAX_UNEXPLAINED = 1
# Looser radius used to grow a seed into more correspondences.
_MATCH_GROW_PX = 80.0
# Edge-alignment tolerances: how close and how parallel a supporting segment
# must be, and how densely the predicted outline is sampled.
_EDGE_ALIGN_PX = 22.0
_EDGE_MATCH_PX = 60.0
_EDGE_ALIGN_DEG = 20.0
_EDGE_SAMPLE_PX = 12.0
_EDGE_MIN_SAMPLES = 8
# Pixel-space polish of the closed-form fit.
_REFINE_ITERS = 8
_REFINE_EPS = (1e-3, 1e-3, 1e-3)  # metres, metres, radians
# Yaw sweep for the pixel-space seed, in radians. Coarse on purpose: the
# Gauss-Newton polish removes the discretisation, so paying for resolution
# here would buy nothing.
_YAW_STEP = math.radians(4.0)
# Work bounds. Assignments grow as (detections choose 2) x (model pairs), so
# a frame full of clutter has to be capped or the solve outlasts the control
# tick. Detections are ranked by support, so the cap keeps the strongest.
_MAX_PER_KIND = 5
_MAX_ASSIGNMENTS = 300
_MAX_SEEDS = 120
_MAX_SINGLE_CORNER_SEEDS = 6
_MAX_SINGLE_CORNER_INPUTS = 6
# Solved poses outside the room are arithmetic, not localisation.
_ROOM_X = (0.5, 6.0)
_ROOM_Y = (0.5, 5.0)
# The table centre has to be in frame, with this much slack past the border.
_IN_VIEW_MARGIN_PX = 120.0

_TOP_EDGES = ((4, 5), (5, 6), (6, 7), (7, 4))
# Finegrained-only: a close-up table is its rims. One matched side is two
# line constraints on a seed that is already near the desk.
_MIN_ALIGN_SIDES = 1
_ALIGN_HOUGH_MIN_PX = 40.0
# Live veto is only the glancing sliver (perception_06/000485): flush to a
# border and thin. A close-up that fills the frame (perception_07/000370,
# 400 kpx, recall 0.66) is not that case — dropping it left search stuck.
_SLIVER_SPAN_FRAC = 0.20
_SLIVER_BORDER_PX = 16.0
# Finegrained lateral: a usable vertical rim is longer than a Hough scrap.
# 35° (not the 18° post test) so a perspective-tilted east edge still counts
# (perception_13 left rim sat at 18.9° and lost to the flush right crop).
_RIM_MIN_PX = 80.0
_RIM_VERTICAL_DEG = 35.0
# Finegrained forward/back: same perspective-tilt allowance as the lateral
# rim above, just measured from horizontal instead of vertical.
_RIM_HORIZONTAL_DEG = 35.0


@dataclass(frozen=True)
class EgoEstimate:
    """Base pose in world metres/radians, with the evidence behind it."""

    x: float
    y: float
    yaw: float
    residual_px: float
    n_corr: int
    source: str
    # The detections that produced it, so a dump can show its own evidence.
    inlier_uv: tuple[tuple[float, float], ...] = ()
    # Fraction of the predicted tabletop outline lying on detected edges.
    edge_support: float = 0.0
    # Model corners in open view that nothing was detected at.
    unexplained: int = 0
    # Tabletop edges matched to a detected outline side and used in the fit.
    n_edges: int = 0
    # IoU of the projected tabletop rectangle against the instance mask.
    tabletop_iou: float = 0.0
    # Fraction of the instance that falls inside the projected top.
    tabletop_recall: float = 0.0
    # Footprint of the instance mask unprojected onto the tabletop plane, and
    # whether it fits inside the (rigid, never re-fitted) TableModel.
    instance_size_xy: tuple[float, float] | None = None
    instance_fits_table: bool | None = None

    @property
    def xy_yaw(self) -> tuple[float, float, float]:
        return self.x, self.y, self.yaw


def _wrap(yaw: float) -> float:
    return (yaw + math.pi) % (2.0 * math.pi) - math.pi


def _is_border_sliver(mask: np.ndarray, width: int, height: int) -> bool:
    """Thin connected blob flush to an image edge — a glancing table, not a fill."""
    ys, xs = np.nonzero(mask[:height, :width] > 0)
    if xs.size == 0:
        return False
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    span_w = x1 - x0 + 1
    span_h = y1 - y0 + 1
    flush = (
        x0 <= _SLIVER_BORDER_PX
        or y0 <= _SLIVER_BORDER_PX
        or x1 >= width - 1 - _SLIVER_BORDER_PX
        or y1 >= height - 1 - _SLIVER_BORDER_PX
    )
    thin = span_w <= _SLIVER_SPAN_FRAC * width or span_h <= _SLIVER_SPAN_FRAC * height
    return flush and thin


def _flush_to_border(seg: Segment, width: int, height: int) -> bool:
    """Both endpoints hug the same image edge — a clipped rim, not an interior one."""
    border = _SLIVER_BORDER_PX
    us = (seg.x0, seg.x1)
    vs = (seg.y0, seg.y1)
    return (
        all(u <= border for u in us)
        or all(u >= width - 1 - border for u in us)
        or all(v <= border for v in vs)
        or all(v >= height - 1 - border for v in vs)
    )


def _outside(uv: tuple[float, float], width: int, height: int) -> bool:
    return not (0.0 <= uv[0] <= width and 0.0 <= uv[1] <= height)


def _line_of(segment: Segment) -> tuple[float, float, float]:
    """Segment → normalised image line ``(nx, ny, d)`` with ``nx²+ny² = 1``."""
    dx, dy = segment.x1 - segment.x0, segment.y1 - segment.y0
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        return 0.0, 0.0, 0.0
    nx, ny = -dy / norm, dx / norm
    return nx, ny, -(nx * segment.x0 + ny * segment.y0)


def _strongest(corners: tuple[Corner, ...]) -> tuple[Corner, ...]:
    if len(corners) <= _MAX_PER_KIND:
        return corners
    return tuple(sorted(corners, key=lambda c: -c.support)[:_MAX_PER_KIND])


def _rank(estimate: EgoEstimate) -> tuple:
    """Most corners explained wins, then edge alignment, then residual.

    A rectangle aliases: a pose shifted by one table width maps some
    detections onto *other* corners and reprojects just as tightly. Only the
    true pose explains **every** detection, so inlier count has to lead.
    """
    return (
        -estimate.n_corr,
        estimate.unexplained,
        -round(estimate.tabletop_recall, 2),
        -round(estimate.tabletop_iou, 2),
        -round(estimate.edge_support, 2),
        estimate.residual_px,
    )


class _SegmentField:
    """Detected segments, arranged for 'is this predicted edge real?' queries."""

    def __init__(self, segments: tuple[Segment, ...]):
        self.empty = not segments
        if self.empty:
            return
        self.start = np.array([(s.x0, s.y0) for s in segments], dtype=np.float64)
        self.delta = np.array([(s.x1 - s.x0, s.y1 - s.y0) for s in segments], dtype=np.float64)
        self.length_sq = np.maximum((self.delta**2).sum(axis=1), 1e-9)
        self.angle = np.array([s.angle_deg for s in segments], dtype=np.float64)

    def support(
        self,
        a: tuple[float, float],
        b: tuple[float, float],
        width: int,
        height: int,
    ) -> tuple[int, int]:
        """(samples on the a→b line lying on a parallel segment, samples in frame)."""
        if self.empty:
            return 0, 0
        span = math.hypot(b[0] - a[0], b[1] - a[1])
        count = max(2, int(span / _EDGE_SAMPLE_PX))
        steps = np.linspace(0.0, 1.0, count)
        points = np.stack((a[0] + steps * (b[0] - a[0]), a[1] + steps * (b[1] - a[1])), axis=1)
        inside = (
            (points[:, 0] >= 0.0)
            & (points[:, 0] < width)
            & (points[:, 1] >= 0.0)
            & (points[:, 1] < height)
        )
        points = points[inside]
        if len(points) == 0:
            return 0, 0
        edge_angle = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180.0
        gap = np.abs(self.angle - edge_angle) % 180.0
        parallel = np.minimum(gap, 180.0 - gap) <= _EDGE_ALIGN_DEG
        if not parallel.any():
            return 0, len(points)
        start, delta = self.start[parallel], self.delta[parallel]
        length_sq = self.length_sq[parallel]
        offset = points[:, None, :] - start[None, :, :]
        t = np.clip((offset * delta[None, :, :]).sum(axis=2) / length_sq, 0.0, 1.0)
        closest = start[None, :, :] + t[:, :, None] * delta[None, :, :]
        distance = np.linalg.norm(points[:, None, :] - closest, axis=2)
        return int((distance.min(axis=1) <= _EDGE_ALIGN_PX).sum()), len(points)


class TableLocalizer:
    """Detections + known table → ``EgoEstimate`` or ``None``."""

    def __init__(
        self,
        table: TableModel | None = None,
        *,
        pair_tol_m: float = PAIR_TOL_M,
        reproj_gate_px: float = REPROJ_GATE_PX,
        edge_support_min: float = EDGE_SUPPORT_MIN,
        tabletop_iou_min: float = TOP_IOU_MIN,
        tabletop_recall_min: float = TOP_RECALL_MIN,
        footprint_margin_m: float = FOOTPRINT_MARGIN_M,
        max_unexplained: int = MAX_UNEXPLAINED,
        min_correspondences: int = MIN_CORRESPONDENCES,
    ):
        self.table = table or TableModel()
        self.pair_tol_m = float(pair_tol_m)
        self.reproj_gate_px = float(reproj_gate_px)
        self.min_correspondences = int(min_correspondences)
        self.edge_support_min = float(edge_support_min)
        self.tabletop_iou_min = float(tabletop_iou_min)
        self.tabletop_recall_min = float(tabletop_recall_min)
        self.footprint_margin_m = float(footprint_margin_m)
        self.max_unexplained = int(max_unexplained)
        self._corners_world = self.table.corners()
        self._edges = self.table.edge_lengths()
        self._basis_cache: dict[float, tuple] = {}
        if not self.table.size_matches(self._corners_world):
            raise ValueError("TableModel corners do not match documented L×W×H")

    def solve(
        self,
        features: Features,
        *,
        spine_m: float,
        image_shape: tuple[int, ...],
        intrinsics: tuple[float, float, float, float] | None = None,
    ) -> EgoEstimate | None:
        """Best gated pose over every detection set.

        Every set is scored and the best is returned, rather than taking the
        first that fits. Two correspondences determine a pose exactly, so
        their residual is near zero whichever corners they are assumed to
        be; only a set that explains *more* detections can break that tie,
        and the tabletop-only set is often not the one that does.

        ``intrinsics`` is live CameraInfo ``(fx, fy, cx, cy)`` when the
        collector has it; otherwise the USD-default 60° square pinhole.
        """
        height, width = image_shape[0], image_shape[1]
        if intrinsics is None:
            intrinsics = head_intrinsics(width, height)
        t_base_cam = T_base_cam(spine_m)
        outline = getattr(features, "outline", ())
        instance = getattr(features, "surface", None)
        field = _SegmentField(tuple(features.segments) + tuple(outline))
        best: EgoEstimate | None = None
        for source, corners in self._candidate_sets(features):
            estimate = self._solve_set(
                corners,
                source,
                t_base_cam,
                intrinsics,
                field,
                outline,
                instance,
                width,
                height,
            )
            if estimate is None:
                continue
            if best is None or _rank(estimate) < _rank(best):
                best = estimate
        return best

    def overlap_scores(
        self,
        pose: tuple[float, float, float],
        features: Features,
        *,
        spine_m: float,
        image_shape: tuple[int, ...],
        intrinsics: tuple[float, float, float, float] | None = None,
    ) -> tuple[float, float] | None:
        """Live ``(iou, recall)`` of this pose's top vs the instance, or None."""
        instance = getattr(features, "surface", None)
        height, width = image_shape[0], image_shape[1]
        if instance is None or not np.any(instance[:height, :width] > 0):
            return None
        if intrinsics is None:
            intrinsics = head_intrinsics(width, height)
        projected = self._project_corners(pose, T_base_cam(spine_m), intrinsics)
        return self._tabletop_overlap(projected, instance, width, height)

    def instance_agrees(
        self,
        pose: tuple[float, float, float],
        features: Features,
        *,
        spine_m: float,
        image_shape: tuple[int, ...],
        intrinsics: tuple[float, float, float, float] | None = None,
    ) -> bool:
        """True unless a border sliver misses the projected top.

        No / empty mask is lost vision. A large close-up blob is not a
        contradiction even if live recall dips under ``TOP_RECALL_MIN`` —
        that is how ``perception_07`` flapped off a visible table. Only a
        thin instance flush to one edge (``perception_06/000485``) vetoes,
        and only when IoU against the box fails.
        """
        instance = getattr(features, "surface", None)
        height, width = image_shape[0], image_shape[1]
        if instance is None or not np.any(instance[:height, :width] > 0):
            return True
        if not _is_border_sliver(instance, width, height):
            return True
        scores = self.overlap_scores(
            pose,
            features,
            spine_m=spine_m,
            image_shape=image_shape,
            intrinsics=intrinsics,
        )
        if scores is None:
            return False
        iou, _rec = scores
        return iou >= self.tabletop_iou_min

    def align_edges(
        self,
        features: Features,
        *,
        expected: tuple[float, float, float],
        seed: tuple[float, float, float] | None = None,
        spine_m: float,
        image_shape: tuple[int, ...],
        intrinsics: tuple[float, float, float, float] | None = None,
    ) -> EgoEstimate | None:
        """Close-up pose from tabletop edges matched to the expected view.

        Navigate ``solve`` still requires corners. At the desk the corners
        leave the frame and the rims stay; this path accepts 1–3 matched
        sides (corners optional) and does not change the navigate gates.
        Correspondence is tried on the expected start-pose projection first,
        then on ``seed`` if too few sides match (a ~20° yaw error sits on
        ``_EDGE_ALIGN_DEG``).
        """
        height, width = image_shape[0], image_shape[1]
        if intrinsics is None:
            intrinsics = head_intrinsics(width, height)
        t_base_cam = T_base_cam(spine_m)
        sides = self._align_sides(features)
        if not sides:
            return None
        lines, n_sides, use = self._correspond_edges(
            expected, seed, sides, width, height, t_base_cam, intrinsics
        )
        if n_sides < 2:
            extra = self._align_sides(features, with_hough=True)
            if extra != sides:
                lines, n_sides, use = self._correspond_edges(
                    expected, seed, extra, width, height, t_base_cam, intrinsics
                )
                sides = extra
        if n_sides < _MIN_ALIGN_SIDES:
            return None
        # Refine from the live seed (or the pose that produced the matches).
        # Do not run `_pose_from_pixels`: a lines-only yaw sweep is aliased
        # when close-up rims are nearly axis-aligned, and that wrecks a
        # seed that Gauss–Newton would have pulled onto the table.
        corners = tuple(features.corners or ())
        starts = []
        if seed is not None:
            starts.append(seed)
        if use not in starts:
            starts.append(use)
        pose = None
        for start in starts:
            pairs = (
                self._match(start, corners, t_base_cam, intrinsics, self.reproj_gate_px)
                if corners
                else []
            )
            trial = self._refine_in_pixels(
                start, pairs, corners, t_base_cam, intrinsics, lines
            )
            rematch, rematch_n = self._match_from(
                trial, sides, width, height, t_base_cam, intrinsics, _EDGE_ALIGN_PX
            )
            if rematch_n < _MIN_ALIGN_SIDES:
                continue
            projected = self._project_corners(trial, t_base_cam, intrinsics)
            if not self._plausible(trial, projected, width, height):
                continue
            pose, lines, n_sides = trial, rematch, rematch_n
            break
        if pose is None:
            return None
        pairs = (
            self._match(pose, corners, t_base_cam, intrinsics, self.reproj_gate_px)
            if corners
            else []
        )
        residual = self._constraint_rms(pose, pairs, corners, t_base_cam, intrinsics, lines)
        if residual is None:
            return None
        inliers = tuple((corners[d].u, corners[d].v) for d, _ in pairs)
        return EgoEstimate(
            pose[0],
            pose[1],
            pose[2],
            residual,
            len(pairs),
            "edges",
            inliers,
            0.0,
            0,
            n_sides,
            0.0,
            0.0,
            None,
            None,
        )

    def lateral_from_vertical_rim(
        self,
        features: Features,
        *,
        seed: tuple[float, float, float],
        spine_m: float,
        image_shape: tuple[int, ...],
        intrinsics: tuple[float, float, float, float] | None = None,
    ) -> tuple[float, float, float] | None:
        """Shift ``seed.x`` so a visible vertical rim lands on a table X-face.

        Close-up ``align_edges`` can snap xy to the expected pose on one
        horizontal side. After yaw is already good, the leftover error is
        left/right: unproject the leftmost interior vertical outline side
        onto the tabletop and slide world X until it matches the east or
        west face. A frame-crop is not a face. ``y`` and ``yaw`` stay put.
        ``None`` when no usable rim.
        """
        height, width = image_shape[0], image_shape[1]
        if intrinsics is None:
            intrinsics = head_intrinsics(width, height)
        side = self._vertical_rim_side(features, width, height)
        if side is None:
            return None
        rim_x = self._rim_face_x(side, width)
        if rim_x is None:
            return None
        mid_u = 0.5 * (side.x0 + side.x1)
        mid_v = 0.5 * (side.y0 + side.y1)
        point = unproject_to_plane(
            mid_u,
            mid_v,
            self.table.plane_z(TOP, self.table.height_m),
            T_world_cam(*seed, spine_m),
            intrinsics,
        )
        if point is None:
            return None
        delta_x = float(point[0]) - rim_x
        if not math.isfinite(delta_x):
            return None
        return (seed[0] - delta_x, seed[1], seed[2])

    def _vertical_rim_side(
        self, features: Features, width: int, height: int
    ) -> Segment | None:
        interior = tuple(
            seg
            for seg in (getattr(features, "outline", None) or ())
            if _angle_gap(seg.angle_deg, 90.0) <= _RIM_VERTICAL_DEG
            and seg.length >= _RIM_MIN_PX
            and not _flush_to_border(seg, width, height)
        )
        if not interior:
            return None
        return min(interior, key=lambda seg: 0.5 * (seg.x0 + seg.x1))

    def _rim_face_x(self, side: Segment, width: int) -> float | None:
        """Left of image → east (+X) face; right → west. Seed does not vote."""
        live_u = 0.5 * (side.x0 + side.x1)
        half_w = 0.5 * self.table.size_xy[0]
        origin_x = self.table.origin_xy[0]
        return origin_x + half_w if live_u < 0.5 * width else origin_x - half_w

    def forward_from_horizontal_rim(
        self,
        features: Features,
        *,
        seed: tuple[float, float, float],
        spine_m: float,
        image_shape: tuple[int, ...],
        intrinsics: tuple[float, float, float, float] | None = None,
    ) -> tuple[float, float, float] | None:
        """Shift ``seed.y`` so a visible rear rim lands on the table's far face.

        The forward/back counterpart to ``lateral_from_vertical_rim``: once
        yaw (and x) are already good, the leftover error is how far the
        base actually is from the table. With the base close up and facing
        it, the rear (far, -Y) edge sits higher in frame than the near edge
        the base is right up against — nearer the image bottom means the
        base has overshot past the rear face, farther from the bottom means
        it hasn't reached it yet — so the topmost interior horizontal
        outline side is the rear rim: unproject its midpoint onto the
        tabletop and slide world Y until it matches that fixed face. ``x``
        and ``yaw`` stay put. ``None`` when no usable rim.
        """
        height, width = image_shape[0], image_shape[1]
        if intrinsics is None:
            intrinsics = head_intrinsics(width, height)
        side = self._horizontal_rim_side(features, width, height)
        if side is None:
            return None
        rim_y = self.table.origin_xy[1] - 0.5 * self.table.size_xy[1]
        mid_u = 0.5 * (side.x0 + side.x1)
        mid_v = 0.5 * (side.y0 + side.y1)
        point = unproject_to_plane(
            mid_u,
            mid_v,
            self.table.plane_z(TOP, self.table.height_m),
            T_world_cam(*seed, spine_m),
            intrinsics,
        )
        if point is None:
            return None
        delta_y = float(point[1]) - rim_y
        if not math.isfinite(delta_y):
            return None
        return (seed[0], seed[1] - delta_y, seed[2])

    def _horizontal_rim_side(
        self, features: Features, width: int, height: int
    ) -> Segment | None:
        interior = tuple(
            seg
            for seg in (getattr(features, "outline", None) or ())
            if _angle_gap(seg.angle_deg, 0.0) <= _RIM_HORIZONTAL_DEG
            and seg.length >= _RIM_MIN_PX
            and not _flush_to_border(seg, width, height)
        )
        if not interior:
            return None
        return min(interior, key=lambda seg: 0.5 * (seg.y0 + seg.y1))

    def _align_sides(
        self, features: Features, *, with_hough: bool = False
    ) -> tuple[Segment, ...]:
        outline = tuple(getattr(features, "outline", ()) or ())
        extra = tuple(
            seg
            for seg in getattr(features, "segments", ()) or ()
            if seg.length >= _ALIGN_HOUGH_MIN_PX
        )
        if with_hough or len(outline) < 2:
            return outline + extra
        return outline

    def _correspond_edges(
        self,
        expected,
        seed,
        sides,
        width: int,
        height: int,
        t_base_cam: np.ndarray,
        intrinsics,
    ):
        """Match sides to the expected view; fall back to ``seed`` if thin."""
        lines, n_sides = self._match_from(
            expected, sides, width, height, t_base_cam, intrinsics
        )
        use = expected
        if n_sides < 2 and seed is not None:
            seed_lines, seed_n = self._match_from(
                seed, sides, width, height, t_base_cam, intrinsics
            )
            if seed_n > n_sides:
                lines, n_sides, use = seed_lines, seed_n, seed
        return lines, n_sides, use

    def _match_from(
        self,
        pose: tuple[float, float, float],
        sides,
        width: int,
        height: int,
        t_base_cam: np.ndarray,
        intrinsics,
        tolerance_px: float | None = None,
    ) -> tuple[tuple, int]:
        projected = self._project_corners(pose, t_base_cam, intrinsics)
        lines, _cost = self._match_edges(projected, sides, width, height, tolerance_px)
        return lines, len(lines) // 2

    def _constraint_rms(
        self, pose, pairs, corners, t_base_cam, intrinsics, lines
    ) -> float | None:
        residual = self._residual_vec(pose, pairs, corners, t_base_cam, intrinsics, lines)
        if residual is None or residual.size == 0:
            return None
        return float(math.sqrt(float(np.mean(residual**2))))

    # -- candidate generation ---------------------------------------------

    def _candidate_sets(self, features: Features) -> tuple[tuple[str, tuple[Corner, ...]], ...]:
        tops = _strongest(features.of_kind(TOP))
        feet = _strongest(features.of_kind(FOOT))
        sets = []
        # A single corner is admitted: on its own it cannot fix a pose, but
        # with the table's edges it can (``_seed_from_one_corner``).
        if tops:
            sets.append((TOP, tops))
        if feet:
            sets.append((FOOT, feet))
        if tops and feet:
            sets.append(("mixed", tops + feet))
        return tuple(sets)

    def _solve_set(
        self,
        corners: tuple[Corner, ...],
        source: str,
        t_base_cam: np.ndarray,
        intrinsics: tuple[float, float, float, float],
        field: _SegmentField,
        outline,
        instance,
        width: int,
        height: int,
    ) -> EgoEstimate | None:
        points, kept = [], []
        for corner in corners:
            z_plane = self.table.plane_z(corner.kind, self.table.height_m)
            point = unproject_to_plane(corner.u, corner.v, z_plane, t_base_cam, intrinsics)
            if point is None:
                continue
            points.append(point)
            kept.append(corner)
        if not points:
            return None

        # Cheap stage: per-assignment fits that survive the reprojection gate.
        seen: set[tuple[int, int, int]] = set()
        candidates = []
        for seed in self._seed_poses(points, kept, t_base_cam, intrinsics):
            pose = self._fit(seed, kept, t_base_cam, intrinsics)
            if pose is None:
                continue
            key = (round(pose[0], 2), round(pose[1], 2), round(pose[2], 2))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(pose)

        # Expensive stage: polish in pixels, then judge against the whole frame.
        if not candidates:
            # No pair of detections agreed on anything. One corner still
            # fixes the position once yaw is chosen, so sweep yaw and let
            # the table's own edges choose it.
            candidates = self._seed_from_one_corner(
                kept, t_base_cam, intrinsics, outline, width, height
            )

        best: EgoEstimate | None = None
        for pose in candidates:
            estimate = self._score(
                pose,
                points,
                kept,
                source,
                t_base_cam,
                intrinsics,
                field,
                outline,
                instance,
                width,
                height,
            )
            if estimate is None:
                continue
            if best is None or _rank(estimate) < _rank(best):
                best = estimate
        if best is not None:
            log.debug(
                "perception localize: %s (%.3f, %.3f, %+.4f) resid=%.1f px n=%d "
                "edges=%.2f unexplained=%d",
                best.source,
                best.x,
                best.y,
                best.yaw,
                best.residual_px,
                best.n_corr,
                best.edge_support,
                best.unexplained,
            )
        return best

    def _seed_poses(
        self, points, corners, t_base_cam: np.ndarray, intrinsics
    ) -> list[tuple[float, float, float]]:
        """Every way two detections could be two model corners, fitted in pixels.

        Spacing is only used to prune **foot** pairs. A tabletop corner near
        the horizon unprojects badly — the ray grazes the plane, so a few
        pixels move it metres — so neither the pruning nor the fit may lean
        on unprojected geometry. Each assignment is solved through the
        projection matrix instead, which is well conditioned everywhere.
        """
        seeds: list[tuple[float, float, float]] = []
        seen: set[tuple[float, float, float]] = set()
        budget = _MAX_ASSIGNMENTS
        for i, j in itertools.combinations(range(len(points)), 2):
            grounded = corners[i].kind == FOOT and corners[j].kind == FOOT
            distance = float(np.linalg.norm(points[i] - points[j]))
            for ia, ib in itertools.permutations(range(8), 2):
                if self.table.kind(ia) != corners[i].kind:
                    continue
                if self.table.kind(ib) != corners[j].kind:
                    continue
                if grounded:
                    model = float(
                        np.linalg.norm(self._corners_world[ia][:2] - self._corners_world[ib][:2])
                    )
                    if abs(distance - model) > self.pair_tol_m:
                        continue
                budget -= 1
                if budget < 0:
                    return seeds
                pose = self._pose_from_pixels(
                    ((corners[i], ia), (corners[j], ib)), t_base_cam, intrinsics
                )
                if pose is None:
                    continue
                # Different assignments routinely land on the same pose;
                # growing each of them separately repeats the same work.
                key = (round(pose[0], 1), round(pose[1], 1), round(pose[2], 1))
                if key in seen:
                    continue
                seen.add(key)
                seeds.append(pose)
                if len(seeds) >= _MAX_SEEDS:
                    return seeds
        return seeds

    def _pose_from_pixels(
        self, assignment, t_base_cam: np.ndarray, intrinsics, lines=()
    ) -> tuple[float, float, float] | None:
        """Pose that puts the assigned model corners on their pixels.

        For a *fixed* yaw the projection is linear in the base position:
        clearing the perspective divide turns ``u = fx·X/Z + cx`` into
        ``fx·X + (cx − u)·Z = 0``, and X, Y, Z are affine in (x, y). So a
        sweep over yaw with a two-unknown least squares at each step covers
        the whole pose space, and the known corner heights and edge lengths
        enter as the rigid model being projected. Two corners suffice.

        ``lines`` adds *edge* evidence in the same linear form: requiring a
        model corner to land on the image line ``nx·u + ny·v + d = 0``
        becomes ``nx·fx·X + ny·fy·Y + (nx·cx + ny·cy + d)·Z = 0`` after the
        same clearing. A corner constrains two degrees of freedom, a corner
        on a line only one — which is exactly the point, because the table's
        edges survive in frames where its corners do not.
        """
        yaws, basis, offset = self._yaw_basis(t_base_cam)
        fx, fy, cx, cy = intrinsics
        rows, rhs, terms, line_terms = [], [], [], []
        for corner, index in assignment:
            world = self._corners_world[index]
            affine = np.einsum("yij,j->yi", basis, world) - offset
            gradient = -basis[:, :, :2]
            terms.append((corner, affine, gradient))
            rows.append(fx * gradient[:, 0, :] + (cx - corner.u) * gradient[:, 2, :])
            rhs.append(-(fx * affine[:, 0] + (cx - corner.u) * affine[:, 2]))
            rows.append(fy * gradient[:, 1, :] + (cy - corner.v) * gradient[:, 2, :])
            rhs.append(-(fy * affine[:, 1] + (cy - corner.v) * affine[:, 2]))
        for index, (nx, ny, d) in lines:
            world = self._corners_world[index]
            affine = np.einsum("yij,j->yi", basis, world) - offset
            gradient = -basis[:, :, :2]
            bias = nx * cx + ny * cy + d
            line_terms.append((nx, ny, d, affine, gradient))
            rows.append(
                nx * fx * gradient[:, 0, :] + ny * fy * gradient[:, 1, :] + bias * gradient[:, 2, :]
            )
            rhs.append(-(nx * fx * affine[:, 0] + ny * fy * affine[:, 1] + bias * affine[:, 2]))
        if len(rows) < 2:
            return None
        design = np.stack(rows, axis=1)
        target = np.stack(rhs, axis=1)
        normal = design.transpose(0, 2, 1) @ design
        moment = np.einsum("ykj,yk->yj", design, target)
        det = normal[:, 0, 0] * normal[:, 1, 1] - normal[:, 0, 1] * normal[:, 1, 0]
        usable = np.abs(det) > 1e-9
        if not usable.any():
            return None
        safe = np.where(usable, det, 1.0)
        x = (normal[:, 1, 1] * moment[:, 0] - normal[:, 0, 1] * moment[:, 1]) / safe
        y = (-normal[:, 1, 0] * moment[:, 0] + normal[:, 0, 0] * moment[:, 1]) / safe

        # Algebraic least squares is depth-weighted; pick the yaw by the
        # reprojection error we actually care about.
        cost = np.zeros(len(yaws))
        position = np.stack((x, y), axis=1)
        for corner, affine, gradient in terms:
            cam = affine + np.einsum("yij,yj->yi", gradient, position)
            depth = cam[:, 2]
            usable &= depth > 1e-6
            safe_depth = np.where(depth > 1e-6, depth, 1.0)
            cost += (fx * cam[:, 0] / safe_depth + cx - corner.u) ** 2
            cost += (fy * cam[:, 1] / safe_depth + cy - corner.v) ** 2
        for nx, ny, d, affine, gradient in line_terms:
            cam = affine + np.einsum("yij,yj->yi", gradient, position)
            depth = cam[:, 2]
            usable &= depth > 1e-6
            safe_depth = np.where(depth > 1e-6, depth, 1.0)
            u = fx * cam[:, 0] / safe_depth + cx
            v = fy * cam[:, 1] / safe_depth + cy
            cost += (nx * u + ny * v + d) ** 2
        if not usable.any():
            return None
        cost = np.where(usable, cost, np.inf)
        best = int(np.argmin(cost))
        return float(x[best]), float(y[best]), _wrap(float(yaws[best]))

    def _seed_from_one_corner(
        self, corners, t_base_cam: np.ndarray, intrinsics, outline, width, height
    ) -> list[tuple[float, float, float]]:
        """One corner plus the table's edges: the close-up case.

        A single corner leaves a one-parameter family — pick yaw and the
        position follows — so yaw is chosen by whichever member of that
        family lays the predicted tabletop outline onto the detected one.
        Scoring against the outline (a handful of sides) rather than every
        Hough segment keeps the sweep cheap.
        """
        if not outline or len(corners) > _MAX_SINGLE_CORNER_INPUTS:
            return []
        yaws, _basis, _offset = self._yaw_basis(t_base_cam)
        seeds, seen = [], set()
        for corner in corners:
            for index in range(8):
                if self.table.kind(index) != corner.kind:
                    continue
                best, best_rank = None, (0, 0.0)
                for yaw in yaws:
                    pose = self._position_for_yaw(corner, index, float(yaw), t_base_cam, intrinsics)
                    if pose is None:
                        continue
                    projected = self._project_corners(pose, t_base_cam, intrinsics)
                    lines, cost = self._match_edges(projected, outline, width, height)
                    rank = (len(lines), -cost)
                    if rank > best_rank:
                        best, best_rank = pose, rank
                if best is None or best_rank[0] < 4:  # at least two edges
                    continue
                key = (round(best[0], 1), round(best[1], 1), round(best[2], 1))
                if key in seen:
                    continue
                seen.add(key)
                seeds.append(best)
                if len(seeds) >= _MAX_SINGLE_CORNER_SEEDS:
                    return seeds
        return seeds

    def _position_for_yaw(
        self, corner: Corner, index: int, yaw: float, t_base_cam: np.ndarray, intrinsics
    ) -> tuple[float, float, float] | None:
        """(x, y) that puts one model corner on one pixel, for a given yaw."""
        fx, fy, cx, cy = intrinsics
        inverse = t_base_cam[:3, :3].T
        cos, sin = math.cos(-yaw), math.sin(-yaw)
        rotation = np.array(((cos, -sin, 0.0), (sin, cos, 0.0), (0.0, 0.0, 1.0)))
        basis = inverse @ rotation
        affine = basis @ self._corners_world[index] - inverse @ t_base_cam[:3, 3]
        gradient = -basis[:, :2]
        design = np.stack(
            (
                fx * gradient[0, :] + (cx - corner.u) * gradient[2, :],
                fy * gradient[1, :] + (cy - corner.v) * gradient[2, :],
            )
        )
        target = np.array(
            (
                -(fx * affine[0] + (cx - corner.u) * affine[2]),
                -(fy * affine[1] + (cy - corner.v) * affine[2]),
            )
        )
        det = design[0, 0] * design[1, 1] - design[0, 1] * design[1, 0]
        if abs(det) < 1e-9:
            return None
        x = (design[1, 1] * target[0] - design[0, 1] * target[1]) / det
        y = (-design[1, 0] * target[0] + design[0, 0] * target[1]) / det
        return float(x), float(y), _wrap(yaw)

    def _yaw_basis(self, t_base_cam: np.ndarray):
        """``R_bc⁻¹·Rz(−yaw)`` sampled over a full turn, cached per spine height."""
        key = round(float(t_base_cam[2, 3]), 6)
        cached = self._basis_cache.get(key)
        if cached is not None:
            return cached
        yaws = np.arange(0.0, 2.0 * math.pi, _YAW_STEP)
        rotation = np.zeros((len(yaws), 3, 3))
        cos, sin = np.cos(-yaws), np.sin(-yaws)
        rotation[:, 0, 0] = cos
        rotation[:, 0, 1] = -sin
        rotation[:, 1, 0] = sin
        rotation[:, 1, 1] = cos
        rotation[:, 2, 2] = 1.0
        inverse = t_base_cam[:3, :3].T
        basis = np.einsum("ij,yjk->yik", inverse, rotation)
        offset = inverse @ t_base_cam[:3, 3]
        self._basis_cache[key] = (yaws, basis, offset)
        return yaws, basis, offset

    def _fit(
        self, seed, corners, t_base_cam: np.ndarray, intrinsics
    ) -> tuple[float, float, float] | None:
        """Grow the seed to every detection it explains, re-fitting in pixels.

        The re-fit is the same vectorised yaw sweep as the seed rather than
        Gauss-Newton: this runs once per assignment, and there are dozens.
        """
        pose = seed
        for gate in (_MATCH_GROW_PX, self.reproj_gate_px):
            pairs = self._match(pose, corners, t_base_cam, intrinsics, gate)
            if len(pairs) < 2:
                return None
            pose = self._pose_from_pixels(
                tuple((corners[d], m) for d, m in pairs), t_base_cam, intrinsics
            )
            if pose is None:
                return None
        return pose

    # -- scoring -----------------------------------------------------------

    def _score(
        self,
        pose,
        points,
        corners,
        source: str,
        t_base_cam: np.ndarray,
        intrinsics,
        field: _SegmentField,
        outline,
        instance,
        width: int,
        height: int,
    ) -> EgoEstimate | None:
        pairs = self._match(pose, corners, t_base_cam, intrinsics, self.reproj_gate_px)
        if not pairs:
            return None

        # Edges join the fit: match them from the current pose, re-solve with
        # corners *and* edges together, then re-match both from the result.
        projected = self._project_corners(pose, t_base_cam, intrinsics)
        lines, _cost = self._match_edges(projected, outline, width, height)
        # Only re-fit when the constraints outnumber the three unknowns.
        # An exactly-determined system fits every yaw equally well, so the
        # sweep would pick one arbitrarily and walk the pose off the seed.
        if 2 * len(pairs) + len(lines) > 3:
            refit = self._pose_from_pixels(
                tuple((corners[d], m) for d, m in pairs), t_base_cam, intrinsics, lines
            )
            if refit is not None:
                pose = refit
            pose = self._refine_in_pixels(pose, pairs, corners, t_base_cam, intrinsics, lines)
        pairs = self._match(pose, corners, t_base_cam, intrinsics, self.reproj_gate_px)
        projected = self._project_corners(pose, t_base_cam, intrinsics)
        # Count the edges *strictly*. The loose tolerance above exists so a
        # seed can reach the answer; it must not be what lets a pose claim
        # the answer. Counted loosely, one corner and two nearly-anything
        # edges locked onto the mirror pose on live frames.
        lines, _cost = self._match_edges(projected, outline, width, height, _EDGE_ALIGN_PX)
        n_edges = len(lines) // 2
        if not self._enough_evidence(len(pairs)):
            return None
        residual = self._residual_px(pose, pairs, corners, t_base_cam, intrinsics)
        if residual is None or residual > self.reproj_gate_px:
            return None
        if not self._plausible(pose, projected, width, height):
            return None

        unexplained = self._unexplained_in_view(projected, corners, pairs, width, height)
        if unexplained > self.max_unexplained:
            return None
        support = self._edge_support(projected, field, width, height)
        if support is not None and support < self.edge_support_min:
            return None

        overlap = self._tabletop_overlap(projected, instance, width, height)
        iou = rec = None
        if overlap is not None:
            iou, rec = overlap
            if iou < self.tabletop_iou_min or rec < self.tabletop_recall_min:
                return None

        footprint = self._instance_footprint(pose, outline, instance, t_base_cam, intrinsics)
        fits = None
        instance_size = None
        if footprint is not None:
            instance_size, fits = footprint
            if not fits:
                return None

        return EgoEstimate(
            pose[0],
            pose[1],
            pose[2],
            residual,
            len(pairs),
            source,
            tuple((corners[d].u, corners[d].v) for d, _ in pairs),
            0.0 if support is None else support,
            unexplained,
            n_edges,
            0.0 if iou is None else iou,
            0.0 if rec is None else rec,
            instance_size,
            fits,
        )

    def _enough_evidence(self, n_corr: int) -> bool:
        """Is there more evidence than the two unknowns a pair fits for free?

        Two correspondences determine a pose exactly under *any* assignment,
        so their residual is near zero whichever corners they are assumed to
        be and carries no information about whether the assignment is right.
        With no prior to veto a wrong assignment on position, correspondences
        are the only currency this gate accepts — matched edges strengthen a
        pose that already has enough of them, but cannot substitute.
        """
        return n_corr >= self.min_correspondences

    def _refine_in_pixels(
        self, pose, pairs, corners, t_base_cam: np.ndarray, intrinsics, lines=()
    ) -> tuple[float, float, float]:
        """Gauss-Newton on (x, y, yaw) against reprojection error.

        The closed-form fit minimises metres between unprojected points,
        which over-weights distant corners because a pixel there covers far
        more ground. Minimising pixels instead puts the error where it is
        actually observed, and keeps the model rigid: the table height and
        edge lengths enter through the projection, not as free parameters.
        """
        current = np.array(pose, dtype=np.float64)
        residual = self._residual_vec(current, pairs, corners, t_base_cam, intrinsics, lines)
        if residual is None:
            return pose
        cost = float(residual @ residual)
        for _ in range(_REFINE_ITERS):
            jacobian = np.zeros((residual.size, 3), dtype=np.float64)
            for axis, eps in enumerate(_REFINE_EPS):
                step = np.zeros(3)
                step[axis] = eps
                plus = self._residual_vec(
                    current + step, pairs, corners, t_base_cam, intrinsics, lines
                )
                minus = self._residual_vec(
                    current - step, pairs, corners, t_base_cam, intrinsics, lines
                )
                if plus is None or minus is None:
                    return tuple(current)
                jacobian[:, axis] = (plus - minus) / (2.0 * eps)
            try:
                delta = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
            except np.linalg.LinAlgError:
                break
            improved = False
            for scale in (1.0, 0.5, 0.25):
                trial = current + scale * delta
                trial_residual = self._residual_vec(
                    trial, pairs, corners, t_base_cam, intrinsics, lines
                )
                if trial_residual is None:
                    continue
                trial_cost = float(trial_residual @ trial_residual)
                if trial_cost < cost:
                    current, residual, cost = trial, trial_residual, trial_cost
                    improved = True
                    break
            if not improved:
                break
        return float(current[0]), float(current[1]), float(current[2])

    def _residual_vec(
        self, pose, pairs, corners, t_base_cam: np.ndarray, intrinsics, lines=()
    ) -> np.ndarray | None:
        t_world_cam = T_world_base(*pose) @ t_base_cam
        out = np.empty(2 * len(pairs) + len(lines), dtype=np.float64)
        for k, (d, m) in enumerate(pairs):
            uv = project(self._corners_world[m], t_world_cam, intrinsics)
            if uv is None:
                return None
            out[2 * k] = uv[0] - corners[d].u
            out[2 * k + 1] = uv[1] - corners[d].v
        for k, (m, (nx, ny, d)) in enumerate(lines):
            uv = project(self._corners_world[m], t_world_cam, intrinsics)
            if uv is None:
                return None
            out[2 * len(pairs) + k] = nx * uv[0] + ny * uv[1] + d
        return out

    def _residual_px(
        self, pose, pairs, corners, t_base_cam: np.ndarray, intrinsics
    ) -> float | None:
        residual = self._residual_vec(pose, pairs, corners, t_base_cam, intrinsics)
        if residual is None:
            return None
        return math.sqrt(float((residual**2).reshape(-1, 2).sum(axis=1).mean()))

    def _match_edges(
        self, projected, outline, width: int, height: int, tolerance_px: float | None = None
    ):
        """Tabletop edges that a detected outline side can vouch for.

        Each match yields two line constraints — one per endpoint of the
        model edge — because both ends must lie on the detected side's line,
        not merely somewhere near it.
        """
        lines, cost = [], 0.0
        for a, b in _TOP_EDGES:
            first, second = projected[a], projected[b]
            if first is None or second is None:
                continue
            if _outside(first, width, height) and _outside(second, width, height):
                continue
            angle = math.degrees(math.atan2(second[1] - first[1], second[0] - first[0]))
            # Generous, like the corner grow radius: a seed is a few degrees
            # of yaw away from the answer, which is tens of pixels out at the
            # far end of an edge. The strict test is `_edge_support`, applied
            # after the fit has been given the chance to close the gap.
            best, best_gap = None, _EDGE_MATCH_PX if tolerance_px is None else tolerance_px
            for side in outline:
                gap = abs(side.angle_deg - angle % 180.0) % 180.0
                if min(gap, 180.0 - gap) > _EDGE_ALIGN_DEG:
                    continue
                line = _line_of(side)
                offset = max(
                    abs(line[0] * first[0] + line[1] * first[1] + line[2]),
                    abs(line[0] * second[0] + line[1] * second[1] + line[2]),
                )
                if offset < best_gap:
                    best, best_gap = line, offset
            if best is not None:
                lines.append((a, best))
                lines.append((b, best))
                cost += best_gap
        return tuple(lines), cost

    def _unexplained_in_view(self, projected, corners, pairs, width: int, height: int) -> int:
        """Model corners sitting in open view that nothing was detected at.

        Two detections can fit a rectangle in more than one way; the wrong
        fits usually put another corner in plain sight where the detector
        found nothing. Only kinds the detector actually produced are judged,
        and only corners well inside the frame, since a corner at the border
        is a plausible miss.
        """
        matched = {m for _d, m in pairs}
        kinds = {corner.kind for corner in corners}
        margin = IN_FRAME_MARGIN_PX
        count = 0
        for index, uv in projected.items():
            if uv is None or index in matched:
                continue
            kind = self.table.kind(index)
            if kind not in kinds:
                continue
            if not margin <= uv[0] <= width - margin:
                continue
            if not margin <= uv[1] <= height - margin:
                continue
            nearest = min(
                (
                    math.hypot(corner.u - uv[0], corner.v - uv[1])
                    for corner in corners
                    if corner.kind == kind
                ),
                default=math.inf,
            )
            if nearest > _MATCH_GROW_PX:
                count += 1
        return count

    def _edge_support(
        self, projected, field: _SegmentField, width: int, height: int
    ) -> float | None:
        """Fraction of the predicted tabletop outline lying on detected edges.

        ``None`` when too little of the outline falls inside the frame to
        judge — an absent measurement, not a score of zero.
        """
        supported = total = 0
        for a, b in _TOP_EDGES:
            if projected[a] is None or projected[b] is None:
                continue
            hits, samples = field.support(projected[a], projected[b], width, height)
            supported += hits
            total += samples
        if total < _EDGE_MIN_SAMPLES:
            return None
        return supported / total

    def _tabletop_overlap(self, projected, mask, width: int, height: int):
        """``(iou, recall)`` of the projected tabletop vs the instance.

        Recall is the one a known-size box cannot cheat: the instance has
        to fall inside the projected rectangle. IoU alone lets a small box
        sit on a large table. ``None`` when there is no instance or too
        little of the top is in front of the camera to rasterise.
        """
        if mask is None:
            return None
        points = [projected[i] for i in TOP_INDICES if projected[i] is not None]
        if len(points) < 3:
            return None
        canvas = np.zeros((height, width), dtype=np.uint8)
        polygon = np.array([(int(round(u)), int(round(v))) for u, v in points], np.int32)
        cv2.fillConvexPoly(canvas, polygon, 255)
        instance = mask[:height, :width] > 0
        n_instance = int(np.count_nonzero(instance))
        if n_instance == 0:
            return None
        overlap = (canvas > 0) & instance
        union = (canvas > 0) | instance
        n_union = int(np.count_nonzero(union))
        if n_union == 0:
            return None
        n_overlap = int(np.count_nonzero(overlap))
        return n_overlap / n_union, n_overlap / n_instance

    def _instance_footprint(
        self, pose, outline, instance, t_base_cam: np.ndarray, intrinsics
    ):
        """Unproject the instance onto the table plane; it must fit L×W.

        Height is the tabletop plane (``table.height_m``) by construction.
        A pose that is too far spreads the same pixels over a region bigger
        than the documented table — that is the ``000028`` failure, where
        one pinned corner made a known-size AABB look like a small patch.
        """
        pixels: list[tuple[float, float]] = []
        for side in outline or ():
            pixels.append((side.x0, side.y0))
            pixels.append((side.x1, side.y1))
        if not pixels and instance is not None:
            ys, xs = np.nonzero(instance > 0)
            if xs.size:
                pixels.extend(
                    (
                        (float(xs.min()), float(ys.min())),
                        (float(xs.max()), float(ys.min())),
                        (float(xs.max()), float(ys.max())),
                        (float(xs.min()), float(ys.max())),
                    )
                )
        if len(pixels) < 2:
            return None
        t_world_cam = T_world_base(*pose) @ t_base_cam
        world = []
        for u, v in pixels:
            point = unproject_to_plane(u, v, self.table.height_m, t_world_cam, intrinsics)
            if point is not None:
                world.append(point[:2])
        if len(world) < 2:
            return None
        xy = np.asarray(world, dtype=np.float64)
        width_m, depth_m, _height = self.table.metrics(
            np.column_stack((xy, np.full(len(xy), self.table.height_m)))
        )
        inside = self.table.contains_xy(xy, margin_m=self.footprint_margin_m)
        fits = bool(inside.all()) and (
            width_m <= self.table.size_xy[0] + self.footprint_margin_m
            and depth_m <= self.table.size_xy[1] + self.footprint_margin_m
        )
        return (width_m, depth_m), fits

    # -- helpers -----------------------------------------------------------

    def _match(
        self,
        pose: tuple[float, float, float],
        corners,
        t_base_cam: np.ndarray,
        intrinsics,
        gate_px: float,
    ) -> list[tuple[int, int]]:
        """Nearest projected landmark per detection, one-to-one, same kind."""
        projected = self._project_corners(pose, t_base_cam, intrinsics)
        candidates = []
        for d, corner in enumerate(corners):
            for m, uv in projected.items():
                if uv is None or self.table.kind(m) != corner.kind:
                    continue
                distance = math.hypot(corner.u - uv[0], corner.v - uv[1])
                if distance <= gate_px:
                    candidates.append((distance, d, m))
        pairs, used_d, used_m = [], set(), set()
        for _distance, d, m in sorted(candidates):
            if d in used_d or m in used_m:
                continue
            used_d.add(d)
            used_m.add(m)
            pairs.append((d, m))
        return pairs

    def _project_corners(
        self, pose: tuple[float, float, float], t_base_cam: np.ndarray, intrinsics
    ) -> dict[int, tuple[float, float] | None]:
        """All eight corners at once, inverting the camera pose once.

        ``project`` inverts per point, and this is the inner loop of every
        sweep and every match — 8x the inversions for no reason.
        """
        fx, fy, cx, cy = intrinsics
        inverse = np.linalg.inv(T_world_base(*pose) @ t_base_cam)
        cam = self._corners_world @ inverse[:3, :3].T + inverse[:3, 3]
        out: dict[int, tuple[float, float] | None] = {}
        for i in range(8):
            depth = cam[i, 2]
            if depth <= 1e-6:
                out[i] = None
                continue
            out[i] = (fx * cam[i, 0] / depth + cx, fy * cam[i, 1] / depth + cy)
        return out

    def _plausible(
        self,
        pose: tuple[float, float, float],
        projected,
        width: int,
        height: int,
    ) -> bool:
        """Room bounds and the table in view — no other veto.

        There is deliberately **no** "robot is on the +Y approach side" test.
        It was true of the goal and false of the start: the robot spawns at
        y ≈ 1.59, beside the table rather than in front of it, so that check
        vetoed every correct pose for a whole run (`perception_12` never left
        search in 843 frames). There is also no distance-from-expected veto:
        the table's AABB is 180°-symmetric, and nothing here breaks that tie
        — a mirrored pose that clears every other gate is published as-is.
        """
        if not all(math.isfinite(v) for v in pose):
            return False
        if not _ROOM_X[0] <= pose[0] <= _ROOM_X[1]:
            return False
        if not _ROOM_Y[0] <= pose[1] <= _ROOM_Y[1]:
            return False
        return self._table_in_view(projected, width, height)

    def _table_in_view(self, projected, width: int, height: int) -> bool:
        """The pose has to actually show table geometry, not a proxy for it.

        This used to project the *mean of the 3D corners* — a point at half
        table height, table-centre in x/y — and gate on that one pixel. Mean
        of 3D points is not mean of their projections: close to the table and
        looking down, that single averaged point can fall far below the
        frame while every real corner sits inside it. `perception_13/000286`
        is exactly that shape — 5.5 px residual, 0.87 edge support, all 8
        corners visibly on the table in the dump — rejected because the
        proxy point landed at v ≈ 892 on a 720-tall image. Check the
        corners themselves: the table is in view if any one of them is.
        """
        margin = _IN_VIEW_MARGIN_PX
        for uv in projected.values():
            if uv is None:
                continue
            if -margin <= uv[0] <= width + margin and -margin <= uv[1] <= height + margin:
                return True
        return False
