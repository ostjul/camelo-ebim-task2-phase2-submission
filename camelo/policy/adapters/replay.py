"""Open-loop replay of prerecorded canonical action chunks.

Loads a ``(T, 20)`` action sequence prepared by
``scripts/prepare_replay_actions.py`` (from
``hermanprawiro/task2_fixpos_200`` by default) and serves it one
``MODEL_HORIZON`` window at a time, indexed by sim time from the first
``infer`` after ``reset``. No GPU / lerobot / images required — same
wire path as every other adapter.

Default (``correct_poses=False``) is the full 20-dim demo chunk. With
``correct_poses=True``, the full demo still plays; only ``A_RIGHT_ARM`` is
replaced by IK so world TCP xyz matches GT given the live-vs-demo
base/spine offset (seeded from the demo arm trajectory).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from camelo import contracts as C
from camelo.policy.base import Obs, PolicyAdapter

log = logging.getLogger(__name__)

# Hardcoded source for the base "replay" policy. Episode 89 is bundled here;
# prepare_replay_actions.py materialises other episodes (videos never downloaded).
DEFAULT_REPO_ID = "hermanprawiro/task2_fixpos_200"
DEFAULT_EPISODE = 89
DEFAULT_DATASET_DIR = Path("outputs/datasets/ext_hermanprawiro_task2_fixpos_200")
DEFAULT_ACTIONS_DIR = Path("data/replay/task2_fixpos_200")

_GT_KEYS = ("tcp_xyz", "arm_q", "base_xy_yaw", "spine")


def default_actions_path(episode: int = DEFAULT_EPISODE) -> Path:
    return DEFAULT_ACTIONS_DIR / f"ep{int(episode):03d}_actions.npy"


def default_gt_traj_path(episode: int = DEFAULT_EPISODE) -> Path:
    return DEFAULT_ACTIONS_DIR / f"ep{int(episode):03d}_gt_traj.npz"


def _sibling_gt_path(actions_path: Path) -> Path:
    return actions_path.with_name(actions_path.name.replace("_actions.npy", "_gt_traj.npz"))


class ReplayAdapter(PolicyAdapter):
    """Serve a frozen demo as (H, 20) chunks along the live sim clock."""

    name = "replay"
    chunk_dt: float = 1.0 / 30.0

    def __init__(
        self,
        actions_path: str | Path | None = None,
        episode: int = DEFAULT_EPISODE,
        horizon: int = C.MODEL_HORIZON,
        fps: float | None = None,
        correct_poses: bool = False,
        gt_path: str | Path | None = None,
    ):
        path = Path(actions_path) if actions_path else default_actions_path(episode)
        if not path.is_file():
            raise FileNotFoundError(
                f"replay actions not found at {path} — run "
                f"`python scripts/prepare_replay_actions.py --episode {episode}` first"
            )
        actions = np.load(path)
        if actions.ndim != 2 or actions.shape[1] != C.ACTION_DIM:
            raise ValueError(
                f"replay actions must be (T, {C.ACTION_DIM}), got {actions.shape} from {path}"
            )
        self.actions = np.asarray(actions, dtype=np.float32)
        self.horizon = int(horizon)
        if fps is not None:
            self.chunk_dt = 1.0 / float(fps)
        meta_path = path.with_name(path.name.replace("_actions.npy", "_meta.json"))
        if fps is None and meta_path.is_file():
            import json

            meta = json.loads(meta_path.read_text())
            if "fps" in meta:
                self.chunk_dt = 1.0 / float(meta["fps"])
        self._t0: float | None = None
        self.path = path
        self.episode = int(episode)
        self.correct_poses = bool(correct_poses)
        self.gt_path: Path | None = None
        self._tcp_xyz: np.ndarray | None = None
        self._arm_q: np.ndarray | None = None
        self._base_xy_yaw: np.ndarray | None = None
        self._spine: np.ndarray | None = None
        self._ik = None
        if self.correct_poses:
            self._load_gt(gt_path, episode)

    def _load_gt(self, gt_path: str | Path | None, episode: int) -> None:
        path = Path(gt_path) if gt_path else _sibling_gt_path(self.path)
        if not path.is_file():
            raise FileNotFoundError(
                f"replay GT trajectory not found at {path} — run "
                f"`python scripts/prepare_replay_actions.py --episode {episode}` first"
            )
        t_act = len(self.actions)
        loaded: dict[str, np.ndarray] = {}
        with np.load(path) as data:
            missing = [k for k in _GT_KEYS if k not in data.files]
            if missing:
                raise KeyError(
                    f"{path} missing key {missing[0]} — run "
                    f"`python scripts/prepare_replay_actions.py --episode {episode}` first"
                )
            loaded["tcp_xyz"] = np.asarray(data["tcp_xyz"], dtype=np.float64)
            loaded["arm_q"] = np.asarray(data["arm_q"], dtype=np.float64)
            loaded["base_xy_yaw"] = np.asarray(data["base_xy_yaw"], dtype=np.float64)
            loaded["spine"] = np.asarray(data["spine"], dtype=np.float64).reshape(-1)
        tcp = loaded["tcp_xyz"]
        arm_q = loaded["arm_q"]
        base = loaded["base_xy_yaw"]
        spine = loaded["spine"]
        if tcp.ndim != 2 or tcp.shape[1] != 3 or tcp.shape[0] != t_act:
            raise ValueError(
                f"GT tcp_xyz shape {tcp.shape} does not match actions "
                f"T={t_act} — run "
                f"`python scripts/prepare_replay_actions.py --episode {episode}` first"
            )
        if arm_q.shape != (t_act, 7):
            raise ValueError(
                f"GT arm_q shape {arm_q.shape} does not match actions "
                f"T={t_act} — run "
                f"`python scripts/prepare_replay_actions.py --episode {episode}` first"
            )
        if base.shape != (t_act, 3):
            raise ValueError(
                f"GT base_xy_yaw shape {base.shape} does not match actions "
                f"T={t_act} — run "
                f"`python scripts/prepare_replay_actions.py --episode {episode}` first"
            )
        if spine.shape != (t_act,):
            raise ValueError(
                f"GT spine shape {spine.shape} does not match actions "
                f"T={t_act} — run "
                f"`python scripts/prepare_replay_actions.py --episode {episode}` first"
            )
        from camelo.control.kinematics import RightArmIK

        self.gt_path = path
        self._tcp_xyz = tcp
        self._arm_q = arm_q
        self._base_xy_yaw = base
        self._spine = spine
        self._ik = RightArmIK()

    def reset(self, task: str) -> None:
        super().reset(task)
        self._t0 = None

    def infer(self, obs: Obs) -> np.ndarray:
        if self._t0 is None:
            self._t0 = float(obs.t_sim)
        idx = int((float(obs.t_sim) - self._t0) / self.chunk_dt)
        idx = max(0, min(idx, len(self.actions) - 1))
        end = min(idx + self.horizon, len(self.actions))
        chunk = self.actions[idx:end]
        if len(chunk) < self.horizon:
            # Hold the last recorded action once the demo is exhausted.
            pad = np.broadcast_to(self.actions[-1], (self.horizon - len(chunk), C.ACTION_DIM))
            chunk = np.concatenate([chunk, pad], axis=0)
        chunk = np.asarray(chunk, dtype=np.float32)
        if not self.correct_poses:
            return chunk
        return self._correct_chunk(chunk, obs, idx)

    def _correct_chunk(self, chunk: np.ndarray, obs: Obs, idx: int) -> np.ndarray:
        assert self._ik is not None
        assert self._tcp_xyz is not None
        assert self._arm_q is not None
        assert self._base_xy_yaw is not None
        assert self._spine is not None
        from camelo.control.kinematics import offset_base_xy_yaw

        live_base = np.asarray(obs.state[C.S_BASE_ODOM], dtype=np.float64)
        live_spine = float(obs.state[C.S_SPINE])
        t_gt = len(self._tcp_xyz)
        demo_base_idx = self._base_xy_yaw[idx]
        demo_spine_idx = float(self._spine[idx])
        spine_off = live_spine - demo_spine_idx
        out = chunk.copy()
        seed = np.asarray(self._arm_q[idx], dtype=np.float64)
        last_good: np.ndarray | None = None
        n_fail = 0
        first_err = 0.0
        dq0 = 0.0
        for k in range(self.horizon):
            gi = min(idx + k, t_gt - 1)
            base_k = offset_base_xy_yaw(live_base, demo_base_idx, self._base_xy_yaw[gi])
            spine_k = float(self._spine[gi]) + spine_off
            result = self._ik.solve(self._tcp_xyz[gi], base_k, spine_k, seed)
            if k == 0:
                first_err = float(result.pos_err)
            if result.succeeded:
                q = result.q
                last_good = q
            else:
                n_fail += 1
                if last_good is not None:
                    q = last_good
                else:
                    q = np.asarray(chunk[k, C.A_RIGHT_ARM], dtype=np.float64)
            if k == 0:
                dq0 = float(np.max(np.abs(q - self._arm_q[gi])))
            out[k, C.A_RIGHT_ARM] = q
            seed = q
        log.info(
            "correct_poses idx=%d pos_err=%.2f mm fails=%d/%d dq0=%.3f rad",
            idx,
            first_err * 1000.0,
            n_fail,
            self.horizon,
            dq0,
        )
        return np.asarray(out, dtype=np.float32)
