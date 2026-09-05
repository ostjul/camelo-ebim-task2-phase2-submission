"""Pinhole model, base↔camera↔world transforms, and the 2D rigid fit.

The one property everything else leans on: the base rides on the floor and
only yaws, so world z equals base z. Intersecting a camera ray with a
horizontal plane therefore yields the point in **base** coordinates without
any knowledge of where the robot is — see ``unproject_to_plane``. That is
what lets ``localize`` solve the ego pose absolutely instead of nudging a
previous guess.
"""

from __future__ import annotations

import math

import numpy as np

from camelo import contracts as C
from camelo.control.odom_interp import interpolate_xy_yaw as interpolate_xy_yaw
from camelo.control.odom_interp import wrap_pi as wrap_pi

HEAD_SHAPE = C.CAMERAS["head"]["shape"]  # (H, W, 3)
# camera_sensors.yaml documents the real ZED Mini as 90°×60°. The Isaac
# Camera prim is *not* authored with that: ROS2CameraInfoHelper reports the
# USD default film (focalLength 18.14756 / horizontalAperture 20.955) which
# is 60° HFOV and square pixels. Live ``/isaac/head_camera/camera_info``
# overrides this when the collector has it.
HEAD_HFOV_DEG = 60.0
HEAD_VFOV_DEG = None  # None → fy = fx (square pixels)

# URDF ``base_link`` → ``head_camera_mounting_point`` (lula urdf).
_SPINE_FIXED_XYZ = (0.138289, 0.0, 0.350)
_SPINE_PRISMATIC_XYZ = (0.266711, 0.0, 0.1)
_HEAD_XYZ = (0.0, 0.0, 0.167)
_CAM_MOUNT_XYZ = (0.0498, -0.02, 0.2345)
_CAM_MOUNT_PITCH = 0.7156
# camera_link (X forward) → optical (Z forward, Y down). Looks at the floor.
_OPTICAL_RPY = (-math.pi / 2, 0.0, -math.pi / 2)

# --- simple base model, for path clearance -------------------------------
# Half-extents of the mobile base in ``base_link``, read off the mounting
# point joints of the same URDF as the camera chain above
# (``camelo/control/assets/mobile_fr3_duo_v0_2_lula.urdf``):
#   front_/rear_mounting_point_joint  x = ±0.380705
#   left_/right_mounting_point_joint  y = ±0.272712
# The lidar mounts sit inside that box, at (±0.3275, ±0.2175).
BASE_HALF_LENGTH_M = 0.3807
BASE_HALF_WIDTH_M = 0.2727
# The disc that contains the base, about base_link. This is the BASE only:
# the arms reach much further (~1.00 m horizontally at ARM_READY_POSE with
# the spine at SPINE_SOP_M, ~0.98 m at TASK2_ARM_READY_*), but they do so
# at z ≈ 1.0–1.5 m, clearing the 0.75 m table and every 1.18 m partition in
# the Task 2 cubicle. Path clearance against floor-to-ceiling walls is a
# base problem; anything that reasons about arm-height collisions needs the
# arm envelope instead, not this number.
BASE_RADIUS_M = math.hypot(BASE_HALF_LENGTH_M, BASE_HALF_WIDTH_M)  # 0.468 m


def head_intrinsics(
    width: int = HEAD_SHAPE[1],
    height: int = HEAD_SHAPE[0],
    hfov_deg: float = HEAD_HFOV_DEG,
    vfov_deg: float | None = HEAD_VFOV_DEG,
) -> tuple[float, float, float, float]:
    """Pinhole ``(fx, fy, cx, cy)``. Square pixels when ``vfov_deg`` is None."""
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    if vfov_deg is None:
        fy = fx
    else:
        fy = (height / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)
    return fx, fy, width / 2.0, height / 2.0


def hfov_deg_from_intrinsics(
    intrinsics: tuple[float, float, float, float], width: int | None = None
) -> float:
    """Horizontal FOV implied by ``fx`` at this image width."""
    fx, _fy, _cx, _cy = intrinsics
    half = (HEAD_SHAPE[1] if width is None else width) / 2.0
    return math.degrees(2.0 * math.atan(half / fx))


def rot_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array(((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr)))
    ry = np.array(((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp)))
    rz = np.array(((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0)))
    return rz @ ry @ rx


def transform(rotation: np.ndarray, xyz) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rotation
    out[:3, 3] = xyz
    return out


def T_base_cam(spine_m: float = C.SPINE_SOP_M) -> np.ndarray:
    """``base_link`` → head optical. Spine is the prismatic joint (metres)."""
    t = transform(np.eye(3), _SPINE_FIXED_XYZ)
    t = t @ transform(
        np.eye(3), (_SPINE_PRISMATIC_XYZ[0], 0.0, _SPINE_PRISMATIC_XYZ[2] + spine_m)
    )
    t = t @ transform(np.eye(3), _HEAD_XYZ)
    t = t @ transform(rot_rpy(0.0, _CAM_MOUNT_PITCH, 0.0), _CAM_MOUNT_XYZ)
    t = t @ transform(rot_rpy(*_OPTICAL_RPY), (0.0, 0.0, 0.0))
    return t


def T_world_base(x: float, y: float, yaw: float) -> np.ndarray:
    return transform(rot_rpy(0.0, 0.0, yaw), (x, y, 0.0))


def T_world_cam(
    x: float, y: float, yaw: float, spine_m: float = C.SPINE_SOP_M
) -> np.ndarray:
    return T_world_base(x, y, yaw) @ T_base_cam(spine_m)


def unproject_to_plane(
    u: float,
    v: float,
    z_plane: float,
    t_frame_cam: np.ndarray,
    intrinsics: tuple[float, float, float, float],
) -> np.ndarray | None:
    """Ray through ``(u, v)`` intersected with ``z = z_plane``.

    Pass ``T_base_cam`` and the result is in the base frame — no ego pose
    needed, because a yaw+XY motion of the base leaves a horizontal plane
    invariant. Pass ``T_world_cam`` for the world frame. ``None`` when the
    ray runs parallel to the plane or hits it behind the camera.
    """
    fx, fy, cx, cy = intrinsics
    direction = t_frame_cam[:3, :3] @ np.array(((u - cx) / fx, (v - cy) / fy, 1.0))
    origin = t_frame_cam[:3, 3]
    if abs(direction[2]) < 1e-9:
        return None
    lam = (z_plane - origin[2]) / direction[2]
    if lam <= 0.0:
        return None
    point = origin + lam * direction
    return point if np.all(np.isfinite(point)) else None


def project(
    xyz: np.ndarray,
    t_frame_cam: np.ndarray,
    intrinsics: tuple[float, float, float, float],
) -> tuple[float, float] | None:
    """Point (same frame as ``t_frame_cam``) → pixel; ``None`` if behind."""
    fx, fy, cx, cy = intrinsics
    cam = np.linalg.inv(t_frame_cam) @ np.array((xyz[0], xyz[1], xyz[2], 1.0))
    if cam[2] <= 1e-6:
        return None
    return fx * cam[0] / cam[2] + cx, fy * cam[1] / cam[2] + cy


def solve_rigid_2d(
    source: np.ndarray, target: np.ndarray
) -> tuple[float, float, float]:
    """Closed-form 2D rigid fit (Kabsch): ``target ≈ R(yaw) @ source + t``.

    With ``source`` in the base frame and ``target`` the same landmark
    corners in world, the result *is* the base pose ``(x, y, yaw)``. Needs
    at least two non-coincident pairs; more are least-squares averaged.
    """
    src = np.asarray(source, dtype=np.float64)[:, :2]
    dst = np.asarray(target, dtype=np.float64)[:, :2]
    if src.shape != dst.shape or len(src) < 2:
        raise ValueError("solve_rigid_2d needs >= 2 matching 2D points")
    src_c, dst_c = src.mean(axis=0), dst.mean(axis=0)
    cov = (dst - dst_c).T @ (src - src_c)
    yaw = math.atan2(cov[1, 0] - cov[0, 1], cov[0, 0] + cov[1, 1])
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array(((c, -s), (s, c)))
    translation = dst_c - rot @ src_c
    return float(translation[0]), float(translation[1]), yaw
