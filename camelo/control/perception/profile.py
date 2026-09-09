"""Scene-specific perception settings: camera model, detector thresholds,
and — for a real rig — how the base is driven and where the landmark is.

The sim needs none of this: Isaac publishes ``camera_info`` (ideal pinhole,
``D = 0``) and the tabletop/floor contrast the detector was tuned on. A real
scene can differ on both counts — the Munich rig publishes no ``camera_info``
at all (and its ``.../rect/image`` topic is NOT rectified despite the name),
its ZED-M has visible barrel distortion, and its white workbench sits in
front of white curtains under an exposure where the sim's floor-relative
brightness rule collapses (see ``configs/rig/perception_munich.yaml``). It
also has no pedal quantizer, no wall map and no table at the sim's place, so
a rig profile additionally carries a ``rig:`` drive block, a ``table:``
landmark and the goal/start poses in that landmark's frame.

The defaults here ARE the sim values, so an unconfigured run behaves exactly
as before; ``configs/rig/perception_sim.yaml`` spells the same numbers out
for the record (``tests/test_perception_profile.py`` fails if the two ever
drift apart). numpy-only — no cv2, so this module stays importable wherever
``camelo.control`` is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

from camelo import contracts as C


@dataclass(frozen=True)
class CameraModel:
    """Pinhole ``(fx, fy, cx, cy)`` + OpenCV radial/tangential distortion.

    ``undistort_scale`` scales the *output* focal length used when
    undistorting (``features.undistort(new_intrinsics=k_out)``): < 1.0 keeps
    the periphery of a barrel-distorted frame inside the image instead of
    cropping it at the original focal length. Every projection downstream of
    the undistorted image must then use ``k_out``, not ``k``.
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
class SurfaceParams:
    """What counts as "tabletop" for ``features.surface_mask``.

    Two rules, in order:

    * **floor-relative** — ``V >= median(bottom floor_sample_frac of the
      frame) + v_above_floor``. Survives a change of exposure or floor
      shade, and is the sim default. It assumes the bottom of the frame is
      *floor*: park close enough that the tabletop fills it and the sample
      is the table itself, the threshold climbs above the table, and the
      mask goes empty (measured on the Munich bag: floor sample V 139-148,
      threshold 184-193, tabletop only 144-168 → nothing survives).
    * **absolute** — ``v_min <= V <= v_max``. What a scene with its own
      measured exposure should use; ``v_max`` rejects blown-out whites
      (curtains) that a lower bound alone keeps.

    ``use_floor_relative=False`` skips straight to the absolute rule.
    ``ignore_regions`` are ``[x0, y0, x1, y1]`` boxes zeroed *before*
    connected components, for fixed foreground the mask must never bridge
    through (e.g. the robot's own white shell at the frame's lower corners).
    """

    #: floor-relative rule
    use_floor_relative: bool = True
    v_above_floor: int = 45
    floor_sample_frac: float = 0.55
    #: minimum share of the frame for the floor-relative mask to be accepted
    min_area_frac: float = 0.02
    #: absolute rule
    v_min: int = 170
    v_max: int = 255
    #: minimum share of the frame for the absolute mask to be accepted
    abs_min_frac: float = 0.15
    #: both rules
    s_max: int = 60
    ignore_regions: tuple[tuple[int, int, int, int], ...] = ()


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
            origin_xy=self.origin_xy,
            size_xy=self.size_xy,
            height_m=self.height_m,
            yaw=self.yaw,
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
    """Everything scene-specific the perception approach reads."""

    camera: CameraModel | None = None
    surface: SurfaceParams = field(default_factory=SurfaceParams)
    #: Real-base drive settings; ``None`` = the sim pedal path.
    rig: RigDrive | None = None
    #: The landmark. ``None`` with ``rig`` set = vision off: the fused pose
    #: is the operator's start seed + odometry (the dead-reckoning MVP).
    table: TableSpec | None = None
    #: Goal base pose in the profile's world frame; required with ``rig``.
    goal_xy_yaw: tuple[float, float, float] | None = None
    #: Default filter seed (``--approach-start-xy-yaw`` overrides).
    start_xy_yaw: tuple[float, float, float] | None = None
    #: Fixed head-camera spine height when the state carries none (rig).
    spine_m: float | None = None
    #: "task2" = the sim cubicle walls; "none" = no wall map (rig).
    walls: str = "task2"
    #: where this came from, for log lines and error messages. Not part of
    #: the value: two profiles that say the same thing ARE the same profile,
    #: which is what lets ``load_profile(<empty yaml>) == PerceptionProfile()``.
    source: str | None = field(default=None, compare=False)


#: The unconfigured default: exactly the sim constants.
SIM_PROFILE = PerceptionProfile(source="<sim defaults>")


def _camera_from(doc: dict, path: Path) -> CameraModel | None:
    cam = doc.get("camera")
    if cam is None:
        return None
    if not isinstance(cam, dict):
        raise ValueError(f"{path}: 'camera:' must be a mapping")
    try:
        model = CameraModel(
            fx=float(cam["fx"]),
            fy=float(cam["fy"]),
            cx=float(cam["cx"]),
            cy=float(cam["cy"]),
            dist=tuple(float(v) for v in cam.get("dist", ())),
            undistort_scale=float(cam.get("undistort_scale", 1.0)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{path}: camera needs fx, fy, cx, cy (+ dist, undistort_scale): {exc}"
        ) from exc
    if not all(math.isfinite(v) and v > 0 for v in (model.fx, model.fy, model.undistort_scale)):
        raise ValueError(f"{path}: camera fx/fy/undistort_scale must be finite and > 0")
    return model


def _surface_from(doc: dict, path: Path) -> SurfaceParams:
    src = doc.get("surface")
    if src is None:
        return SurfaceParams()
    if not isinstance(src, dict):
        raise ValueError(f"{path}: 'surface:' must be a mapping")
    known = set(SurfaceParams.__dataclass_fields__)
    unknown = set(src) - known
    if unknown:
        raise ValueError(f"{path}: unknown surface keys {sorted(unknown)}; known: {sorted(known)}")
    regions = src.get("ignore_regions", ())
    boxes = []
    for box in regions or ():
        if len(box) != 4:
            raise ValueError(f"{path}: ignore_regions entries are [x0, y0, x1, y1], got {box!r}")
        x0, y0, x1, y1 = (int(v) for v in box)
        if not (x0 < x1 and y0 < y1):
            raise ValueError(f"{path}: ignore_regions box is empty or inverted: {box!r}")
        boxes.append((x0, y0, x1, y1))
    params = replace(
        SurfaceParams(),
        **{k: v for k, v in src.items() if k != "ignore_regions"},
        ignore_regions=tuple(boxes),
    )
    if not 0 <= params.v_min <= params.v_max <= 255:
        raise ValueError(
            f"{path}: need 0 <= v_min <= v_max <= 255, got {params.v_min}..{params.v_max}"
        )
    if not 0.0 < params.floor_sample_frac < 1.0:
        raise ValueError(f"{path}: floor_sample_frac must be in (0, 1)")
    return params


def _rig_from(doc: dict, path: Path) -> RigDrive | None:
    src = doc.get("rig")
    if src is None:
        return None
    if not isinstance(src, dict):
        raise ValueError(f"{path}: 'rig:' must be a mapping")
    flags = ("base_only", "continuous_twist")
    try:
        rig = RigDrive(
            **{k: (bool(v) if k in flags else float(v)) for k, v in src.items()}
        )
    except TypeError as exc:
        raise ValueError(f"{path}: unknown rig key: {exc}") from exc
    if not 0 < rig.max_linear_mps <= C.REAL_BASE_MAX_LINEAR_MPS + 1e-9:
        raise ValueError(f"{path}: rig.max_linear_mps must be in (0, {C.REAL_BASE_MAX_LINEAR_MPS}]")
    if not 0 < rig.max_angular_radps <= C.REAL_BASE_MAX_ANGULAR_RADPS + 1e-9:
        raise ValueError(
            f"{path}: rig.max_angular_radps must be in (0, {C.REAL_BASE_MAX_ANGULAR_RADPS}]"
        )
    return rig


def _table_from(doc: dict, path: Path) -> TableSpec | None:
    src = doc.get("table")
    if src is None:
        return None
    try:
        table = TableSpec(
            origin_xy=(float(src["origin_xy"][0]), float(src["origin_xy"][1])),
            size_xy=(float(src["size_xy"][0]), float(src["size_xy"][1])),
            height_m=float(src["height_m"]),
            yaw=float(src.get("yaw", 0.0)),
        )
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError(
            f"{path}: table needs origin_xy, size_xy, height_m (+ yaw): {exc}"
        ) from exc
    if not all(v > 0 for v in (*table.size_xy, table.height_m)):
        raise ValueError(f"{path}: table size/height must be > 0")
    return table


def _pose_from(doc: dict, path: Path, key: str) -> tuple[float, float, float] | None:
    value = doc.get(key)
    if value is None:
        return None
    try:
        x, y, yaw = (float(c) for c in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {key} must be [x, y, yaw] (metres, radians)") from exc
    if not all(math.isfinite(c) for c in (x, y, yaw)):
        raise ValueError(f"{path}: {key} must be finite")
    return (x, y, yaw)


def load_profile(path: str | Path) -> PerceptionProfile:
    """Parse a ``configs/rig/*.yaml`` scene profile.

    Raises rather than silently falling back to the sim defaults: a profile
    that loaded wrong is worse than one that visibly refused to load — the
    run would look configured while using someone else's scene.
    """
    path = Path(path).expanduser()
    doc = yaml.safe_load(path.read_text()) or {}
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: expected a top-level mapping")
    walls = str(doc.get("walls", "task2"))
    if walls not in ("task2", "none"):
        raise ValueError(f"{path}: walls must be 'task2' or 'none', got {walls!r}")
    spine_m = doc.get("spine_m")
    spine_m = None if spine_m is None else float(spine_m)
    return PerceptionProfile(
        camera=_camera_from(doc, path),
        surface=_surface_from(doc, path),
        rig=_rig_from(doc, path),
        table=_table_from(doc, path),
        goal_xy_yaw=_pose_from(doc, path, "goal_xy_yaw"),
        start_xy_yaw=_pose_from(doc, path, "start_xy_yaw"),
        spine_m=spine_m,
        walls=walls,
        source=str(path),
    )
