"""Head-camera perception for the Task 2 approach (numpy + cv2 + PIL only).

Pipeline, one module per stage — see
``camelo/control/perception_based_navigation.md``:

    features   head RGB → tabletop instance → silhouette + corners
    table      known landmark: AABB dims + frozen world pose
    localize   detections + landmark → EgoEstimate (pixel-space bbox)
    walls      room plan as map constants; detector still a placeholder
    filter     solve measurements → aggregated pose estimate
    planner    start + goal + walls → clearance-checked spline samples
    viz        tabletop AABB, wall bottom lines, path; BEV

``geometry`` holds the pinhole and rigid-transform helpers the rest share.
"""

from camelo.control.perception.features import (
    Corner,
    Features,
    Segment,
    detect,
)
from camelo.control.perception.filter import PoseChannels, PoseFilter
from camelo.control.perception.geometry import (
    HEAD_HFOV_DEG,
    HEAD_SHAPE,
    HEAD_VFOV_DEG,
    T_base_cam,
    T_world_base,
    T_world_cam,
    head_intrinsics,
    hfov_deg_from_intrinsics,
    interpolate_xy_yaw,
    project,
    solve_rigid_2d,
    unproject_to_plane,
)
from camelo.control.perception.localize import EgoEstimate, TableLocalizer
from camelo.control.perception.planner import (
    SAMPLE_SPACING_M,
    PathSample,
    SplinePlanner,
)
from camelo.control.perception.table import TableModel
from camelo.control.perception.walls import (
    ROOM_BOUNDS_XY,
    ROOM_WALLS,
    TASK2_CUBICLE_WALLS,
    WallSegment,
    detect_walls,
)

__all__ = (
    "Corner",
    "EgoEstimate",
    "Features",
    "HEAD_HFOV_DEG",
    "HEAD_SHAPE",
    "HEAD_VFOV_DEG",
    "PathSample",
    "PoseChannels",
    "PoseFilter",
    "ROOM_BOUNDS_XY",
    "ROOM_WALLS",
    "SAMPLE_SPACING_M",
    "Segment",
    "SplinePlanner",
    "TableLocalizer",
    "TableModel",
    "TASK2_CUBICLE_WALLS",
    "T_base_cam",
    "T_world_base",
    "T_world_cam",
    "WallSegment",
    "detect",
    "detect_walls",
    "head_intrinsics",
    "hfov_deg_from_intrinsics",
    "interpolate_xy_yaw",
    "project",
    "solve_rigid_2d",
    "unproject_to_plane",
)
