"""Build and save world-frame right-arm GT trajectories from measured state.

Numpy-only (FK via ``MobileFR3Kinematics``). Parquet loading stays in
``scripts/prepare_replay_actions.py`` / the extract notebook.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from camelo.control.kinematics import RIGHT_LINK8, RIGHT_TCP, MobileFR3Kinematics

GT_TRAJ_KEYS = (
    "t",
    "base_xy_yaw",
    "spine",
    "arm_q",
    "tcp_xyz",
    "tcp_quat_xyzw",
    "fps",
    "episode",
)


def compute_gt_traj(
    *,
    t: np.ndarray,
    base_xy_yaw: np.ndarray,
    spine: np.ndarray,
    arm_q: np.ndarray,
    recorded_link8: np.ndarray,
    episode: int,
    fps: float,
    kin: MobileFR3Kinematics | None = None,
) -> dict[str, np.ndarray]:
    """FK TCP + link8; keep recorded link8 for audits."""
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    base = np.asarray(base_xy_yaw, dtype=np.float64)
    spine_m = np.asarray(spine, dtype=np.float64).reshape(-1)
    q = np.asarray(arm_q, dtype=np.float64)
    rec = np.asarray(recorded_link8, dtype=np.float64)
    n = t.shape[0]
    if base.shape != (n, 3):
        raise ValueError(f"base_xy_yaw must be ({n}, 3), got {base.shape}")
    if spine_m.shape != (n,):
        raise ValueError(f"spine must be ({n},), got {spine_m.shape}")
    if q.shape != (n, 7):
        raise ValueError(f"arm_q must be ({n}, 7), got {q.shape}")
    if rec.shape != (n, 7):
        raise ValueError(f"recorded_link8 must be ({n}, 7), got {rec.shape}")
    solver = kin if kin is not None else MobileFR3Kinematics()
    tcp_xyz, tcp_quat = solver.fk_traj(q, base, spine_m, frame=RIGHT_TCP)
    link8_xyz, link8_quat = solver.fk_traj(q, base, spine_m, frame=RIGHT_LINK8)
    return {
        "t": t,
        "base_xy_yaw": base,
        "spine": spine_m,
        "arm_q": q,
        "tcp_xyz": tcp_xyz,
        "tcp_quat_xyzw": tcp_quat,
        "link8_xyz": link8_xyz,
        "link8_quat_xyzw": link8_quat,
        "recorded_link8_xyz": rec[:, :3].copy(),
        "recorded_link8_quat_xyzw": rec[:, 3:7].copy(),
        "episode": np.int32(episode),
        "fps": np.float64(fps),
    }


def write_gt_traj(path: Path | str, arrays: dict[str, Any]) -> Path:
    """Write ``epXXX_gt_traj.npz``. ``tcp_xyz`` is the IK target."""
    path = Path(path)
    missing = [k for k in GT_TRAJ_KEYS if k not in arrays]
    if missing:
        raise KeyError(f"GT trajectory missing keys {missing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path
