"""Site-specific perception settings: camera model + detector thresholds.

The sim needs none of this: Isaac publishes ``camera_info`` (ideal pinhole,
``D = 0``) and the tabletop/floor contrast the detector was tuned on. The
Munich rig has no ``camera_info``, an unrectified ZED-M stream with visible
barrel distortion, and a white workbench in front of white partitions.
``configs/rig/perception_munich.yaml`` carries the measured answers; this
module loads them. numpy-only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from camelo import contracts as C


@dataclass(frozen=True)
class CameraModel:
    """Pinhole ``(fx, fy, cx, cy)`` + OpenCV distortion ``dist``.

    ``undistort_scale`` scales the *output* focal length when undistorting
    (``features.undistort(new_intrinsics=…)``): < 1 keeps the periphery of a
    barrel-distorted frame inside the image instead of cropping it. Every
    projection after undistortion must use ``k_out``, not ``k``.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    dist: tuple[float, ...] = ()
    undistort_scale: float = 1.0

    @property
    def k(self) -> tuple[float, float, float, float]:
        return (self.fx, self.fy, self.cx, self.cy)

    @property
    def k_out(self) -> tuple[float, float, float, float]:
        s = self.undistort_scale
        return (self.fx * s, self.fy * s, self.cx, self.cy)

    @property
    def distorted(self) -> bool:
        return any(abs(d) > 1e-12 for d in self.dist)


@dataclass(frozen=True)
class TableSpec:
    """The landmark in the profile's world frame (metres, radians)."""

    origin_xy: tuple[float, float]
    size_xy: tuple[float, float]
    height_m: float
    yaw: float = 0.0

    def to_model(self):
        from camelo.control.perception.table import TableModel

        return TableModel(
            origin_xy=self.origin_xy, size_xy=self.size_xy, height_m=self.height_m, yaw=self.yaw
        )


@dataclass(frozen=True)
class RigDrive:
    """How the approach drives a real base (continuous twist, no pedals).

    Limits default to the swerve controller's own clamps (``contracts
    REAL_BASE_*``). ``kp_*`` are sized for that plant: at 0.1 m/s the base
    saturates beyond 0.1 m of error with ``kp_linear`` 1.0 and tapers below
    it; the sim's 5.0/12.0 were sized against the pedal quantizer's engage
    threshold and would slam this base against its ramp.
    """

    base_only: bool = True
    continuous_twist: bool = True
    max_linear_mps: float = C.REAL_BASE_MAX_LINEAR_MPS
    max_angular_radps: float = C.REAL_BASE_MAX_ANGULAR_RADPS
    kp_linear: float = 1.0
    kp_yaw: float = 1.5
    goal_xy_m: float = 0.03
    goal_yaw_rad: float = 0.035
    settle_s: float = 1.0


@dataclass(frozen=True)
class PerceptionProfile:
    camera: CameraModel | None = None
    # Real-base drive settings; None = the sim pedal path.
    rig: RigDrive | None = None
    # The landmark. None with ``rig`` set = vision off: the fused pose is
    # the operator's start seed + odometry (the dead-reckoning MVP).
    table: TableSpec | None = None
    # Goal base pose in the profile's world frame; required with ``rig``.
    goal_xy_yaw: tuple[float, float, float] | None = None
    # Default filter seed (``--approach-start-xy-yaw`` overrides).
    start_xy_yaw: tuple[float, float, float] | None = None
    # Fixed head-camera spine height when the state carries none (rig).
    spine_m: float | None = None
    # "task2" = the sim cubicle walls; "none" = no wall map (rig).
    walls: str = "task2"
    # Absolute HSV-V threshold for the tabletop mask (None = the sim's
    # floor-relative rule, see ``features.surface_mask``).
    surface_v_abs: int | None = None
    # Pixel rectangles ``(x0, y0, x1, y1)`` in the (undistorted) frame that
    # can never be tabletop — the robot's own shell at the lower corners
    # bridges into the mask otherwise. Zeroed before the instance is taken.
    ignore_regions: tuple[tuple[int, int, int, int], ...] = ()
    source: str = field(default="", compare=False)


def _camera_from_mapping(m: dict) -> CameraModel:
    try:
        cam = CameraModel(
            fx=float(m["fx"]),
            fy=float(m["fy"]),
            cx=float(m["cx"]),
            cy=float(m["cy"]),
            dist=tuple(float(v) for v in m.get("dist", ())),
            undistort_scale=float(m.get("undistort_scale", 1.0)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"camera model needs fx, fy, cx, cy (+ dist, undistort_scale): {exc}"
        ) from exc
    if not all(math.isfinite(v) and v > 0 for v in (cam.fx, cam.fy, cam.undistort_scale)):
        raise ValueError(f"camera fx/fy/undistort_scale must be finite and > 0: {cam}")
    return cam


def load_profile(path: str | Path) -> PerceptionProfile:
    """Read a YAML profile (``camera:`` mapping, ``surface_v_abs:`` int)."""
    import yaml  # core dependency (contracts.verify_topics reads YAML too)

    p = Path(path).expanduser()
    data = yaml.safe_load(p.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{p}: expected a mapping at the top level")
    camera = _camera_from_mapping(data["camera"]) if data.get("camera") else None
    v_abs = data.get("surface_v_abs")
    if v_abs is not None:
        v_abs = int(v_abs)
        if not 0 < v_abs < 256:
            raise ValueError(f"{p}: surface_v_abs must be in 1..255, got {v_abs}")
    rig = None
    if data.get("rig") is not None:
        m = data["rig"] if isinstance(data["rig"], dict) else {}
        try:
            rig = RigDrive(**{k: (bool(v) if k in ("base_only", "continuous_twist") else float(v))
                              for k, v in m.items()})
        except TypeError as exc:
            raise ValueError(f"{p}: unknown rig key: {exc}") from exc
        if not (0 < rig.max_linear_mps <= C.REAL_BASE_MAX_LINEAR_MPS + 1e-9):
            raise ValueError(
                f"{p}: rig.max_linear_mps must be in (0, {C.REAL_BASE_MAX_LINEAR_MPS}]"
            )
        if not (0 < rig.max_angular_radps <= C.REAL_BASE_MAX_ANGULAR_RADPS + 1e-9):
            raise ValueError(
                f"{p}: rig.max_angular_radps must be in (0, {C.REAL_BASE_MAX_ANGULAR_RADPS}]"
            )
    table = None
    if data.get("table") is not None:
        t = data["table"]
        try:
            table = TableSpec(
                origin_xy=(float(t["origin_xy"][0]), float(t["origin_xy"][1])),
                size_xy=(float(t["size_xy"][0]), float(t["size_xy"][1])),
                height_m=float(t["height_m"]),
                yaw=float(t.get("yaw", 0.0)),
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ValueError(
                f"{p}: table needs origin_xy, size_xy, height_m (+ yaw): {exc}"
            ) from exc
        if not all(v > 0 for v in (*table.size_xy, table.height_m)):
            raise ValueError(f"{p}: table size/height must be > 0")

    def _pose(key):
        v = data.get(key)
        if v is None:
            return None
        try:
            x, y, yaw = (float(c) for c in v)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{p}: {key} must be [x, y, yaw] (metres, radians)") from exc
        if not all(math.isfinite(c) for c in (x, y, yaw)):
            raise ValueError(f"{p}: {key} must be finite")
        return (x, y, yaw)

    walls = str(data.get("walls", "task2"))
    if walls not in ("task2", "none"):
        raise ValueError(f"{p}: walls must be 'task2' or 'none', got {walls!r}")
    spine_m = data.get("spine_m")
    spine_m = None if spine_m is None else float(spine_m)
    ignore = []
    for rect in data.get("ignore_regions") or ():
        try:
            x0, y0, x1, y1 = (int(v) for v in rect)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{p}: ignore_regions entries are [x0, y0, x1, y1]: {rect!r}") from exc
        if not (0 <= x0 < x1 and 0 <= y0 < y1):
            raise ValueError(f"{p}: ignore_regions rectangle is empty or negative: {rect!r}")
        ignore.append((x0, y0, x1, y1))
    return PerceptionProfile(
        camera=camera,
        rig=rig,
        table=table,
        goal_xy_yaw=_pose("goal_xy_yaw"),
        start_xy_yaw=_pose("start_xy_yaw"),
        spine_m=spine_m,
        walls=walls,
        surface_v_abs=v_abs,
        ignore_regions=tuple(ignore),
        source=str(p),
    )
