"""Room walls as world-frame segments: map constants now, detector later.

Two things live here.

**The room plan** (``TASK2_CUBICLE_WALLS``, ``ROOM_BOUNDS_XY``) is *map*
data, in exactly the sense ``table.py`` is: the room is as known as the
table standing in it, so the planner reads it from constants instead of
waiting on perception. MIRRORED from the benchmark room asset
``assets/robot_room.usd`` — see "provenance" below for how to re-derive it.

**The detector** (``detect_walls``) is still a placeholder that returns
nothing. It stays because a *measured* wall is a different thing from a
mapped one, and a run on the real rig (where the map may be wrong) will
want it. Intended reading, when it is implemented: the floor/wall boundary
is a long near-horizontal edge low in the frame whose two ends unproject
on ``z = 0`` to points far outside the table. ``features.Features.segments``
already carries the raw segments needed for that, which is why they are
passed in.

Provenance
----------
The benchmark declares the table as a Python constant
(``TASK2_TABLE_POSITION`` in ``scripts/scenes/scene_robot_room_keyboard.py``)
but has NO wall constants: the room geometry exists only inside the binary
USD crate ``assets/robot_room.usd``. The numbers below were read out of it
once, with ``pxr`` (``usd-core``)::

    stage = Usd.Stage.Open(".../assets/robot_room.usd")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    cache.ComputeWorldBound(prim).ComputeAlignedRange()   # per Gprim

filtering to prims that start at the floor (``z_min <= 0.20``), stand at
least 0.5 m tall, and are thin (<= 0.35 m) in one horizontal axis and long
(>= 0.40 m) in the other.

The asset's frame IS the world frame: ``configure_robot_room_stage``
references the room at ``/World/Environment/RobotRoom`` with position
(0, 0, 0) and ``reset_asset_xform=True``, and the asset's own root prim is
already identity. Cross-checked independently — the Task 2 desk in the
crate has world bounds x 1.44..2.69, y 1.60..2.34, which is
``TableModel``'s AABB (centre (2.05, 1.95), size (1.25, 0.74)) to within
1.5 cm. ``planner.SplinePlanner.plan`` clears this AABB the same way it
clears these walls when a ``table=`` is passed in.

These are AABBs of authored slabs, so each wall is recorded as its
CENTRELINE plus the slab ``thickness_m``. A path clearance test wants
``geometry.BASE_RADIUS_M + thickness_m / 2`` from the centreline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from camelo.control.perception.features import Features


@dataclass(frozen=True)
class WallSegment:
    """A floor-level wall footprint in world metres.

    ``(x0, y0) -> (x1, y1)`` is the slab's centreline; ``thickness_m`` is
    its full thickness across that line (0.0 when unknown, which is what a
    detector that only recovers a boundary line will report).
    """

    x0: float
    y0: float
    x1: float
    y1: float
    thickness_m: float = 0.0


# Outer shell of the room: four 0.20 m slabs, 3.00 m tall, enclosing
# x -12.49..12.51, y -7.51..7.49. Recorded for completeness — it is >6 m
# from anywhere the Task 2 approach drives, so it never binds the path.
ROOM_BOUNDS_XY = ((-12.49, -7.51), (12.51, 7.49))

# The Task 2 cubicle. Partitions enclosing the workspace that holds the
# table (centre (2.05, 1.95)) and the spawn pose (4.4, 2.6): interior runs
# from x 0.24 to x 5.49 and y 0.34 to y 3.54. Heights in the comments are
# the authored slab tops; all of them are >= 1.18 m, so every one blocks
# the 0.5 m-tall base equally and the base-only clearance model does not
# need to care which.
TASK2_CUBICLE_WALLS = (
    # north partition, 1.18 m tall — the cubicle wall the transit line runs under
    WallSegment(-5.71, 3.66, 5.73, 3.66, 0.24),
    # south partition, 3.00 m tall (full height, glazed)
    WallSegment(-3.54, 0.22, 3.79, 0.22, 0.24),
    # south partition, east stub, 1.18 m tall
    WallSegment(4.99, 0.22, 5.49, 0.22, 0.24),
    # east partition, 1.18 m tall — the wall the spawn strafe backs away from
    WallSegment(5.61, 1.91, 5.61, 3.54, 0.24),
    # east partition, south run, 1.18 m tall
    WallSegment(5.61, -3.80, 5.61, 0.71, 0.24),
    # west partition, 1.18 m tall
    WallSegment(0.12, 2.51, 0.12, 3.54, 0.24),
    # west partition, south run, 3.00 m tall (full height)
    WallSegment(0.12, 0.34, 0.12, 1.31, 0.24),
)

# What the Task 2 approach plans against.
ROOM_WALLS = TASK2_CUBICLE_WALLS


def detect_walls(
    features: Features,
    t_world_base: np.ndarray,
    *,
    spine_m: float | None = None,
) -> list[WallSegment]:
    """Placeholder wall detector. Always empty; never raises.

    Mapped walls come from ``ROOM_WALLS``; this is the not-yet-built
    *measured* path.
    """
    del features, t_world_base, spine_m
    return []
