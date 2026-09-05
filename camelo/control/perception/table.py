"""The table as a known map landmark: AABB dimensions and frozen world pose.

The benchmark scene puts the table at a fixed place, so its world pose is a
*given*, not something to estimate. Fitting ego pose *is* placing this AABB
in the image — see ``camelo/control/perception_based_navigation.md`` §3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TABLE_ORIGIN_XY = (2.05, 1.95)
TABLE_SIZE_XY = (1.25, 0.74)  # width along X, depth along Y
TABLE_HEIGHT_M = 0.75

FOOT = "foot"
TOP = "top"

# Unit AABB. Near = +Y (the side the robot approaches from), left = −X.
# 0–3 feet (z=0), 4–7 tabletop, both near-left / near-right / far-right / far-left.
_UNIT_CORNERS = np.array(
    (
        (-0.5, 0.5, 0.0),
        (0.5, 0.5, 0.0),
        (0.5, -0.5, 0.0),
        (-0.5, -0.5, 0.0),
        (-0.5, 0.5, 1.0),
        (0.5, 0.5, 1.0),
        (0.5, -0.5, 1.0),
        (-0.5, -0.5, 1.0),
    ),
    dtype=np.float64,
)

FOOT_INDICES = (0, 1, 2, 3)
TOP_INDICES = (4, 5, 6, 7)
CORNER_KINDS = (FOOT,) * 4 + (TOP,) * 4


@dataclass(frozen=True)
class TableModel:
    """Known table landmark. Defaults are the Task 2 scene."""

    origin_xy: tuple[float, float] = TABLE_ORIGIN_XY
    size_xy: tuple[float, float] = TABLE_SIZE_XY
    height_m: float = TABLE_HEIGHT_M
    yaw: float = 0.0

    def corners(self) -> np.ndarray:
        """Eight world corners ``(8, 3)``, feet first, in AABB index order."""
        scale = np.array((self.size_xy[0], self.size_xy[1], self.height_m))
        local = _UNIT_CORNERS * scale
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        rot = np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))
        world = local @ rot.T
        world[:, 0] += self.origin_xy[0]
        world[:, 1] += self.origin_xy[1]
        return world

    def metrics(self, corners: np.ndarray | None = None) -> tuple[float, float, float]:
        """``(width, depth, height)`` in the table frame, metres.

        The box we draw is this rigid AABB. Any recovered footprint — the
        model corners, or an instance unprojected onto the tabletop — has
        to report these extents. A smaller box in the image is a wrong
        *pose*, not a smaller table.
        """
        pts = self.corners() if corners is None else np.asarray(corners, dtype=np.float64)
        dx = pts[:, 0] - self.origin_xy[0]
        dy = pts[:, 1] - self.origin_xy[1]
        c, s = math.cos(-self.yaw), math.sin(-self.yaw)
        local_x = c * dx - s * dy
        local_y = s * dx + c * dy
        return (
            float(local_x.max() - local_x.min()),
            float(local_y.max() - local_y.min()),
            float(pts[:, 2].max() - pts[:, 2].min()),
        )

    def size_matches(self, corners: np.ndarray | None = None, *, tol: float = 1e-6) -> bool:
        """True iff the 8 points are this table's documented L × W × H."""
        width, depth, height = self.metrics(corners)
        return (
            abs(width - self.size_xy[0]) <= tol
            and abs(depth - self.size_xy[1]) <= tol
            and abs(height - self.height_m) <= tol
        )

    def contains_xy(self, xy: np.ndarray, *, margin_m: float = 0.0) -> np.ndarray:
        """Which world ``(N, 2)`` points lie on the tabletop, plus ``margin_m``."""
        pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        dx = pts[:, 0] - self.origin_xy[0]
        dy = pts[:, 1] - self.origin_xy[1]
        c, s = math.cos(-self.yaw), math.sin(-self.yaw)
        local_x = c * dx - s * dy
        local_y = s * dx + c * dy
        hw, hd = 0.5 * self.size_xy[0] + margin_m, 0.5 * self.size_xy[1] + margin_m
        return (np.abs(local_x) <= hw) & (np.abs(local_y) <= hd)

    @staticmethod
    def kind(index: int) -> str:
        return CORNER_KINDS[index]

    @staticmethod
    def plane_z(kind: str, height_m: float = TABLE_HEIGHT_M) -> float:
        return 0.0 if kind == FOOT else height_m

    def indices_of_kind(self, kind: str) -> tuple[int, ...]:
        return FOOT_INDICES if kind == FOOT else TOP_INDICES

    def edge_lengths(self) -> dict[tuple[int, int], float]:
        """Distance between every corner pair, keyed by sorted index pair."""
        corners = self.corners()
        out: dict[tuple[int, int], float] = {}
        for i in range(8):
            for j in range(i + 1, 8):
                out[(i, j)] = float(np.linalg.norm(corners[i] - corners[j]))
        return out

    def robot_is_on_near_side(self, x: float, y: float) -> bool:
        """Is a pose on the approach side (+Y in table frame)?

        Describes the *goal*, and nothing else. It must never gate a pose
        estimate: the robot starts beside the table (y ≈ 1.59), so using
        this as a filter rejects the true pose for the whole approach —
        which it did, for all 843 frames of `perception_12`.
        """
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        dx, dy = x - self.origin_xy[0], y - self.origin_xy[1]
        return (-s * dx + c * dy) > 0.5 * self.size_xy[1]
