"""CSV logs of the mobile-base pose vs a goal, and of executed right TCP.

``BaseTrajectoryLog`` columns: ``t, x, y, yaw, err_x, err_y, err_yaw, stage``.
``t`` is elapsed sim seconds from the first sample; errors are world-frame
``goal - pose`` (yaw wrapped to (−π, π]); ``stage`` is the approach FSM
label (or ``rollout``).

``TcpTrajectoryLog`` columns: ``t, t_sim, x, y, z, base_x, base_y, yaw,
spine, q1..q7``. ``t=0`` is the first *rollout* sample (matches demo
indexing / adapter ``_t0``). ``(x, y, z)`` is FK ``right_tcp`` from
measured joints.

``PadTrajectoryLog`` columns: ``t, t_sim, pad_x, pad_y, pad_z, obj_x,
obj_y, obj_z, target_x, target_y, target_z``. ``t=0`` is the first
sample, same convention as ``TcpTrajectoryLog``. ``(pad_x, pad_y,
pad_z)`` is the LIVE thermal-pad mesh centroid; ``(obj_x, obj_y,
obj_z)`` is the pad's (frozen at reset) object-pose reading;
``(target_x, target_y, target_z)`` is the target object's pose. A
reading unavailable on a given tick is written as an empty field, never
``0.0`` — a missing sample must not be mistaken for the origin.

Rows are flushed as they arrive so a timeout or kill still leaves a
usable file.
"""

from __future__ import annotations

import csv
import logging
import math
from pathlib import Path

from camelo.control.approach import wrap_angle

log = logging.getLogger(__name__)

COLUMNS = ("t", "x", "y", "yaw", "err_x", "err_y", "err_yaw", "stage")
DEFAULT_TRAJ_CSV = Path("outputs/approach/base_trajectory.csv")
TCP_COLUMNS = (
    "t",
    "t_sim",
    "x",
    "y",
    "z",
    "base_x",
    "base_y",
    "yaw",
    "spine",
    "q1",
    "q2",
    "q3",
    "q4",
    "q5",
    "q6",
    "q7",
)
DEFAULT_TCP_CSV = Path("outputs/approach/tcp_trajectory.csv")
PAD_COLUMNS = (
    "t",
    "t_sim",
    "pad_x",
    "pad_y",
    "pad_z",
    "obj_x",
    "obj_y",
    "obj_z",
    "target_x",
    "target_y",
    "target_z",
)
DEFAULT_PAD_CSV = Path("outputs/approach/pad_trajectory.csv")


class BaseTrajectoryLog:
    """Append-only CSV of base odometry against a fixed goal pose."""

    def __init__(
        self,
        path: Path | str,
        goal_x: float,
        goal_y: float,
        goal_yaw: float,
    ):
        self.path = Path(path)
        self.goal_x = float(goal_x)
        self.goal_y = float(goal_y)
        self.goal_yaw = float(goal_yaw)
        self._t0: float | None = None
        self.n = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(COLUMNS)
        self._f.flush()
        log.info(
            "base trajectory -> %s  goal=(%.3f, %.3f, %+.4f rad)",
            self.path,
            self.goal_x,
            self.goal_y,
            self.goal_yaw,
        )

    def record(
        self, t_sim: float, x: float, y: float, yaw: float, stage: str = ""
    ) -> None:
        if self._f is None:
            return
        if not all(math.isfinite(v) for v in (t_sim, x, y, yaw)):
            return
        if self._t0 is None:
            self._t0 = float(t_sim)
        t = float(t_sim) - self._t0
        err_x = self.goal_x - x
        err_y = self.goal_y - y
        err_yaw = wrap_angle(self.goal_yaw - yaw)
        self._w.writerow((t, x, y, yaw, err_x, err_y, err_yaw, stage))
        self.n += 1
        self._f.flush()

    def close(self) -> None:
        if self._f is None:
            return
        self._f.flush()
        self._f.close()
        self._f = None
        log.info("base trajectory: %d samples -> %s", self.n, self.path)


class TcpTrajectoryLog:
    """Append-only CSV of measured right-arm TCP (world xyz) during rollout."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._t0: float | None = None
        self.n = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(TCP_COLUMNS)
        self._f.flush()
        log.info("tcp trajectory -> %s", self.path)

    def record(
        self,
        t_sim: float,
        x: float,
        y: float,
        z: float,
        base_x: float,
        base_y: float,
        yaw: float,
        spine: float,
        q,
    ) -> None:
        if self._f is None:
            return
        joints = tuple(float(v) for v in q)
        if len(joints) != 7:
            raise ValueError(f"q must have 7 joints, got {len(joints)}")
        pose = (
            float(t_sim),
            float(x),
            float(y),
            float(z),
            float(base_x),
            float(base_y),
            float(yaw),
            float(spine),
        )
        if not all(math.isfinite(v) for v in (*pose, *joints)):
            return
        if self._t0 is None:
            self._t0 = float(t_sim)
        t = float(t_sim) - self._t0
        self._w.writerow((t, *pose, *joints))
        self.n += 1
        self._f.flush()

    def close(self) -> None:
        if self._f is None:
            return
        self._f.flush()
        self._f.close()
        self._f = None
        log.info("tcp trajectory: %d samples -> %s", self.n, self.path)


class PadTrajectoryLog:
    """Append-only CSV of the thermal pad's per-tick position during rollout.

    ``pad_xyz`` (LIVE mesh centroid), ``obj_xyz`` (frozen object-pose
    reading), and ``target_xyz`` (target object's pose) are each recorded
    as-is; any of them being ``None`` on a tick (an unobserved reading)
    writes an empty field for that triple rather than ``0.0``.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._t0: float | None = None
        self.n = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(PAD_COLUMNS)
        self._f.flush()
        log.info("pad trajectory -> %s", self.path)

    def record(
        self,
        t_sim: float,
        pad_xyz: tuple[float, float, float] | None,
        obj_xyz: tuple[float, float, float] | None,
        target_xyz: tuple[float, float, float] | None,
    ) -> None:
        if self._f is None:
            return
        if not math.isfinite(t_sim):
            return
        if self._t0 is None:
            self._t0 = float(t_sim)
        t = float(t_sim) - self._t0
        row = [t, float(t_sim)]
        for xyz in (pad_xyz, obj_xyz, target_xyz):
            if xyz is None:
                row.extend(("", "", ""))
            else:
                row.extend(float(v) for v in xyz)
        self._w.writerow(row)
        self.n += 1
        self._f.flush()

    def close(self) -> None:
        if self._f is None:
            return
        self._f.flush()
        self._f.close()
        self._f = None
        log.info("pad trajectory: %d samples -> %s", self.n, self.path)
