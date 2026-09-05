"""Right-arm FK/IK for the mobile FR3 duo (vendored Task 2 Lula URDF).

``MobileFR3Kinematics`` walks the URDF chain. ``RightArmIK`` is damped
least-squares on the same geometric Jacobian (xyz only; spine held).
Numpy-only — no Isaac, no ``ebim-benchmark``.

Quaternions at the public API are **xyzw** (dataset).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from camelo import contracts as C

VENDORED_URDF = Path(__file__).resolve().parent / "assets" / "mobile_fr3_duo_v0_2_lula.urdf"

RIGHT_TCP = "right_tcp"
RIGHT_LINK8 = "right_fr3v2_link8"
SUPPORTED_FRAMES = (RIGHT_TCP, RIGHT_LINK8)

_URDF_ROOT_LINK = "base"
_DEFAULT_AXIS = (1.0, 0.0, 0.0)


def default_lula_urdf() -> Path:
    """Vendored Task 2 Lula URDF (kinematics only; no meshes)."""
    if not VENDORED_URDF.is_file():
        raise FileNotFoundError(f"Lula URDF missing: {VENDORED_URDF}")
    return VENDORED_URDF


@dataclass(frozen=True)
class _Joint:
    name: str
    jtype: str
    parent: str
    child: str
    origin: np.ndarray  # (4, 4)
    axis: np.ndarray  # (3,)
    lower: float = -math.inf
    upper: float = math.inf


def _parse_xyz(text: str | None) -> tuple[float, float, float]:
    if not text:
        return (0.0, 0.0, 0.0)
    vals = [float(v) for v in text.split()]
    if len(vals) != 3:
        raise ValueError(f"expected 3 xyz values, got {text!r}")
    return (vals[0], vals[1], vals[2])


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF RPY: fixed-axis X then Y then Z == Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _origin_matrix(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = _rpy_matrix(*rpy)
    t[0, 3], t[1, 3], t[2, 3] = xyz
    return t


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    c1 = 1.0 - c
    return np.array(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ],
        dtype=np.float64,
    )


def _motion_matrix(joint: _Joint, q: float) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    if joint.jtype in ("revolute", "continuous"):
        t[:3, :3] = _axis_angle_matrix(joint.axis, q)
    elif joint.jtype == "prismatic":
        t[:3, 3] = joint.axis * q
    elif joint.jtype != "fixed":
        raise ValueError(f"unsupported joint type {joint.jtype!r} on {joint.name}")
    return t


def _parse_urdf(path: Path) -> dict[str, _Joint]:
    """Map child-link name -> joint that connects it to its parent."""
    root = ET.parse(path).getroot()
    by_child: dict[str, _Joint] = {}
    for node in root.findall("joint"):
        name = node.get("name")
        jtype = node.get("type")
        parent = node.find("parent")
        child = node.find("child")
        if not name or not jtype or parent is None or child is None:
            continue
        parent_link = parent.get("link")
        child_link = child.get("link")
        if not parent_link or not child_link:
            continue
        origin_node = node.find("origin")
        xyz = _parse_xyz(None if origin_node is None else origin_node.get("xyz"))
        rpy = _parse_xyz(None if origin_node is None else origin_node.get("rpy"))
        axis_node = node.find("axis")
        axis = np.asarray(
            _parse_xyz(None if axis_node is None else axis_node.get("xyz"))
            if axis_node is not None
            else _DEFAULT_AXIS,
            dtype=np.float64,
        )
        n = float(np.linalg.norm(axis))
        if n == 0.0:
            raise ValueError(f"zero axis on joint {name}")
        axis = axis / n
        limit_node = node.find("limit")
        lower, upper = -math.inf, math.inf
        if limit_node is not None:
            if limit_node.get("lower") is not None:
                lower = float(limit_node.get("lower"))
            if limit_node.get("upper") is not None:
                upper = float(limit_node.get("upper"))
        by_child[child_link] = _Joint(
            name=name,
            jtype=jtype,
            parent=parent_link,
            child=child_link,
            origin=_origin_matrix(xyz, rpy),
            axis=axis,
            lower=lower,
            upper=upper,
        )
    return by_child


def _chain_to(by_child: dict[str, _Joint], frame: str) -> tuple[_Joint, ...]:
    joints: list[_Joint] = []
    link = frame
    seen: set[str] = set()
    while link != _URDF_ROOT_LINK:
        if link in seen:
            raise ValueError(f"cycle in URDF while walking to {frame}")
        seen.add(link)
        joint = by_child.get(link)
        if joint is None:
            raise ValueError(f"no parent joint for link {link!r} (frame {frame})")
        joints.append(joint)
        link = joint.parent
    joints.reverse()
    return tuple(joints)


def _base_world_matrix(base_xy_yaw: np.ndarray) -> np.ndarray:
    x, y, yaw = (float(v) for v in base_xy_yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    t = np.eye(4, dtype=np.float64)
    t[0, 0], t[0, 1], t[0, 3] = c, -s, x
    t[1, 0], t[1, 1], t[1, 3] = s, c, y
    return t


def offset_base_xy_yaw(
    live_xy_yaw: np.ndarray,
    demo_xy_yaw_idx: np.ndarray,
    demo_xy_yaw_k: np.ndarray,
) -> np.ndarray:
    """Apply the live-vs-demo SE(2) offset at ``idx`` onto the demo pose at ``k``.

    ``T_offset = T_live * inv(T_demo[idx])``; returns ``T_offset * T_demo[k]``
    as ``(x, y, yaw)``. Numpy-only; used by residual ``CORRECT_POSES`` replay.
    """
    t_live = _base_world_matrix(np.asarray(live_xy_yaw, dtype=np.float64).reshape(-1))
    t_idx = _base_world_matrix(np.asarray(demo_xy_yaw_idx, dtype=np.float64).reshape(-1))
    t_k = _base_world_matrix(np.asarray(demo_xy_yaw_k, dtype=np.float64).reshape(-1))
    t_pred = t_live @ np.linalg.inv(t_idx) @ t_k
    return np.array(
        [float(t_pred[0, 3]), float(t_pred[1, 3]), math.atan2(t_pred[1, 0], t_pred[0, 0])],
        dtype=np.float64,
    )


def rot_to_quat_xyzw(rot: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (x, y, z, w)."""
    m = np.asarray(rot, dtype=np.float64)
    t = float(np.trace(m))
    if t > 0.0:
        s = 0.5 / math.sqrt(t + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quat = np.array([x, y, z, w], dtype=np.float64)
    n = float(np.linalg.norm(quat))
    if n == 0.0:
        raise ValueError("degenerate rotation matrix")
    if quat[3] < 0.0:
        quat = -quat
        n = float(np.linalg.norm(quat))
    return quat / n


def quat_xyzw_to_rot(q: np.ndarray) -> np.ndarray:
    """Unit quaternion (x, y, z, w) -> (3, 3) rotation matrix."""
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def quat_xyzw_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Geodesic angle between two xyzw quaternions, degrees."""
    qa = np.asarray(a, dtype=np.float64)
    qb = np.asarray(b, dtype=np.float64)
    qa = qa / np.linalg.norm(qa)
    qb = qb / np.linalg.norm(qb)
    dot = float(np.clip(abs(float(qa @ qb)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(dot))


class MobileFR3Kinematics:
    """Right-arm FK by walking the vendored Task 2 Lula URDF."""

    def __init__(self, urdf_path: str | Path | None = None):
        path = Path(urdf_path) if urdf_path is not None else default_lula_urdf()
        self.urdf_path = path
        by_child = _parse_urdf(path)
        self._chains = {frame: _chain_to(by_child, frame) for frame in SUPPORTED_FRAMES}
        self._arm_joints = tuple(C.RIGHT_JOINTS)
        self._spine_joint = C.SPINE_JOINT
        by_name = {joint.name: joint for joint in self._chains[RIGHT_TCP]}
        missing = [name for name in self._arm_joints if name not in by_name]
        if missing:
            raise RuntimeError(f"TCP chain missing right-arm joints: {missing}")
        self._arm_lower = np.array(
            [by_name[name].lower for name in self._arm_joints], dtype=np.float64
        )
        self._arm_upper = np.array(
            [by_name[name].upper for name in self._arm_joints], dtype=np.float64
        )

    def tcp_joint_xyz(self) -> np.ndarray:
        """Origin xyz of the URDF joint whose child is ``right_tcp``.

        Parent is the gripper base; translation is in that frame (typically
        along gripper z). Not a world-frame offset.
        """
        joint = self._chains[RIGHT_TCP][-1]
        if joint.child != RIGHT_TCP:
            raise RuntimeError(f"TCP chain does not end at {RIGHT_TCP}")
        return joint.origin[:3, 3].copy()

    def fk(
        self,
        arm_q: np.ndarray,
        base_xy_yaw: np.ndarray,
        spine_m: float,
        frame: str = RIGHT_TCP,
    ) -> tuple[np.ndarray, np.ndarray]:
        """World pose of ``frame``: (xyz, quat_xyzw).

        ``arm_q`` is ``right_fr3v2_joint1..7`` (rad). ``base_xy_yaw`` is
        recorded odom (x, y, yaw); root z is 0 (task2 spawn). ``spine_m`` is
        the measured prismatic height.
        """
        if frame not in SUPPORTED_FRAMES:
            raise ValueError(f"frame must be one of {SUPPORTED_FRAMES}, got {frame!r}")
        q_arm = np.asarray(arm_q, dtype=np.float64).reshape(-1)
        if q_arm.shape != (7,):
            raise ValueError(f"arm_q must be (7,), got {np.asarray(arm_q).shape}")
        base = np.asarray(base_xy_yaw, dtype=np.float64).reshape(-1)
        if base.shape != (3,):
            raise ValueError(f"base_xy_yaw must be (3,), got {np.asarray(base_xy_yaw).shape}")
        return self._fk_urdf(q_arm, base, float(spine_m), frame)

    def _fk_urdf(
        self,
        q_arm: np.ndarray,
        base: np.ndarray,
        spine_m: float,
        frame: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        q_map = {
            name: float(val) for name, val in zip(self._arm_joints, q_arm, strict=True)
        }
        q_map[self._spine_joint] = spine_m
        t = _base_world_matrix(base)
        for joint in self._chains[frame]:
            t = t @ joint.origin @ _motion_matrix(joint, q_map.get(joint.name, 0.0))
        return t[:3, 3].copy(), rot_to_quat_xyzw(t[:3, :3])

    def _tcp_pos_and_jacobian(
        self,
        q_arm: np.ndarray,
        base: np.ndarray,
        spine_m: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """World TCP xyz and 3x7 geometric position Jacobian vs right-arm q.

        Spine is held fixed (not a column). Revolute axis ``i`` contributes
        ``axis_world × (p_tcp − origin_world)`` after the joint origin and
        before that joint's motion, matching this URDF's convention.
        """
        q_map = {
            name: float(val) for name, val in zip(self._arm_joints, q_arm, strict=True)
        }
        q_map[self._spine_joint] = spine_m
        t = _base_world_matrix(base)
        axes: list[tuple[str, np.ndarray, np.ndarray]] = []
        for joint in self._chains[RIGHT_TCP]:
            t = t @ joint.origin
            if joint.name in self._arm_joints and joint.jtype in (
                "revolute",
                "continuous",
            ):
                axis_w = t[:3, :3] @ joint.axis
                origin_w = t[:3, 3].copy()
                axes.append((joint.name, axis_w, origin_w))
            t = t @ _motion_matrix(joint, q_map.get(joint.name, 0.0))
        if len(axes) != 7:
            raise RuntimeError(f"expected 7 arm joints in TCP chain, got {len(axes)}")
        p = t[:3, 3].copy()
        jac = np.zeros((3, 7), dtype=np.float64)
        name_to_i = {name: i for i, name in enumerate(self._arm_joints)}
        for name, axis_w, origin_w in axes:
            jac[:, name_to_i[name]] = np.cross(axis_w, p - origin_w)
        return p, jac

    def fk_traj(
        self,
        arm_q: np.ndarray,
        base_xy_yaw: np.ndarray,
        spine_m: np.ndarray,
        frame: str = RIGHT_TCP,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized-by-loop FK. ``arm_q`` is (T, 7); returns (T, 3), (T, 4)."""
        q = np.asarray(arm_q, dtype=np.float64)
        base = np.asarray(base_xy_yaw, dtype=np.float64)
        spine = np.asarray(spine_m, dtype=np.float64).reshape(-1)
        if q.ndim != 2 or q.shape[1] != 7:
            raise ValueError(f"arm_q must be (T, 7), got {q.shape}")
        if base.shape != (q.shape[0], 3):
            raise ValueError(f"base_xy_yaw must be ({q.shape[0]}, 3), got {base.shape}")
        if spine.shape != (q.shape[0],):
            raise ValueError(f"spine_m must be ({q.shape[0]},), got {spine.shape}")
        pos = np.empty((q.shape[0], 3), dtype=np.float64)
        quat = np.empty((q.shape[0], 4), dtype=np.float64)
        for i in range(q.shape[0]):
            pos[i], quat[i] = self.fk(q[i], base[i], float(spine[i]), frame=frame)
        return pos, quat


@dataclass(frozen=True)
class RightArmIKResult:
    """One right-arm IK step.

    ``q`` is ``right_fr3v2_joint1..7`` (rad). ``pos_err`` is |FK(q) − target|
    in metres. ``backend`` is always ``urdf_dls``.
    """

    q: np.ndarray
    succeeded: bool
    pos_err: float
    backend: str


class RightArmIK:
    """Right-arm IK: live base + world TCP xyz → ``right_fr3v2_joint1..7``.

    Open-loop replay of a demo's arm joints fails when the mobile base is not
    on the demo ``(x, y, yaw)``. This solver holds the saved world TCP
    **position** and solves for new right-arm joints given the live base.

    Not solved (held from the caller):

    - spine height — measured, not a free joint
    - left arm, gripper
    - TCP orientation — xyz only

    URDF damped least-squares on the same chain as ``MobileFR3Kinematics.fk``.
    Geometric 3×7 Jacobian; clamp to URDF limits (FR3 j4/j6 are not centered
    at 0). Policy: ``ik.solve(gt_tcp[t], live_base, live_spine, live_arm_q)``.
    """

    def __init__(
        self,
        kin: MobileFR3Kinematics | None = None,
        *,
        pos_tol_m: float = 5e-4,
        max_iters: int = 40,
        damping: float = 1e-3,
    ):
        if pos_tol_m <= 0.0:
            raise ValueError(f"pos_tol_m must be positive, got {pos_tol_m}")
        if max_iters < 1:
            raise ValueError(f"max_iters must be >= 1, got {max_iters}")
        if damping <= 0.0:
            raise ValueError(f"damping must be positive, got {damping}")
        self.kin = kin if kin is not None else MobileFR3Kinematics()
        self.pos_tol_m = float(pos_tol_m)
        self.max_iters = int(max_iters)
        self.damping = float(damping)

    @property
    def backend(self) -> str:
        return "urdf_dls"

    def solve(
        self,
        tcp_xyz: np.ndarray,
        base_xy_yaw: np.ndarray,
        spine_m: float,
        q_seed: np.ndarray,
    ) -> RightArmIKResult:
        """One frame: world TCP xyz + live base/spine + seed joints → arm_q."""
        target = np.asarray(tcp_xyz, dtype=np.float64).reshape(-1)
        if target.shape != (3,):
            raise ValueError(f"tcp_xyz must be (3,), got {np.asarray(tcp_xyz).shape}")
        base = np.asarray(base_xy_yaw, dtype=np.float64).reshape(-1)
        if base.shape != (3,):
            raise ValueError(f"base_xy_yaw must be (3,), got {np.asarray(base_xy_yaw).shape}")
        seed = np.asarray(q_seed, dtype=np.float64).reshape(-1)
        if seed.shape != (7,):
            raise ValueError(f"q_seed must be (7,), got {np.asarray(q_seed).shape}")
        return self._solve_urdf_dls(target, base, float(spine_m), seed)

    def solve_traj(
        self,
        tcp_xyz: np.ndarray,
        base_xy_yaw: np.ndarray,
        spine_m: np.ndarray,
        q_seed0: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-frame ``solve``. Frame 0 seeds from ``q_seed0``; later frames
        seed from the previous solution (perfect joint execution).
        """
        targets = np.asarray(tcp_xyz, dtype=np.float64)
        base = np.asarray(base_xy_yaw, dtype=np.float64)
        spine = np.asarray(spine_m, dtype=np.float64).reshape(-1)
        seed0 = np.asarray(q_seed0, dtype=np.float64).reshape(-1)
        if targets.ndim != 2 or targets.shape[1] != 3:
            raise ValueError(f"tcp_xyz must be (T, 3), got {targets.shape}")
        t_len = targets.shape[0]
        if base.shape != (t_len, 3):
            raise ValueError(f"base_xy_yaw must be ({t_len}, 3), got {base.shape}")
        if spine.shape != (t_len,):
            raise ValueError(f"spine_m must be ({t_len},), got {spine.shape}")
        if seed0.shape != (7,):
            raise ValueError(f"q_seed0 must be (7,), got {seed0.shape}")
        q_out = np.empty((t_len, 7), dtype=np.float64)
        ok = np.empty(t_len, dtype=bool)
        seed = seed0
        for i in range(t_len):
            result = self.solve(targets[i], base[i], float(spine[i]), seed)
            q_out[i] = result.q
            ok[i] = result.succeeded
            seed = result.q
        return q_out, ok

    def _solve_urdf_dls(
        self,
        target: np.ndarray,
        base: np.ndarray,
        spine_m: float,
        q_seed: np.ndarray,
    ) -> RightArmIKResult:
        lower = self.kin._arm_lower
        upper = self.kin._arm_upper
        q = np.clip(q_seed.copy(), lower, upper)
        eye = np.eye(3, dtype=np.float64)
        for _ in range(self.max_iters):
            pos, jac = self.kin._tcp_pos_and_jacobian(q, base, spine_m)
            err = target - pos
            nerr = float(np.linalg.norm(err))
            if nerr < self.pos_tol_m:
                return RightArmIKResult(
                    q=q.copy(),
                    succeeded=True,
                    pos_err=nerr,
                    backend="urdf_dls",
                )
            dq = jac.T @ np.linalg.solve(jac @ jac.T + self.damping * eye, err)
            q = np.clip(q + dq, lower, upper)
        pos, _ = self.kin._tcp_pos_and_jacobian(q, base, spine_m)
        nerr = float(np.linalg.norm(target - pos))
        return RightArmIKResult(
            q=q.copy(),
            succeeded=nerr < self.pos_tol_m,
            pos_err=nerr,
            backend="urdf_dls",
        )
