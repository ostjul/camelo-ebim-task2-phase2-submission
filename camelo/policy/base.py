"""Policy-side data types and the adapter interface (no ROS imports here)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from camelo import contracts as C


@dataclass
class Obs:
    """Canonical observation, identical on both sides of the wire.

    ``state`` is the SIM contract's 37-dim recorder vector for every
    ``model16``/``canonical`` run. The Munich rig speaks a different
    contract (27 dims, different robot, different recorder), so a
    real-robot bridge fills ``rig`` instead: named joint/wrench/gripper
    groups that ``camelo.policy.adapters.s27a15`` packs into the policy's
    27-dim slot. Named rather than positional because that rig's
    joint-state field order has never been observed from this repo — see
    that module's docstring. When ``rig`` is set, ``state`` may be the
    already-packed 27-dim vector or empty.
    """

    t_sim: float
    state: np.ndarray  # (37,) float32, recorder layout — or (27,) under s27a15
    images: dict[str, np.ndarray] = field(default_factory=dict)  # HxWx3 uint8 RGB
    # Sim-time stamp (header.stamp) of each image, where known. Lets the
    # runner measure image AGE in sim seconds at the moment of inference
    # (camelo.control.image_age, N2b step 1). Absent keys = unstamped.
    image_t_sim: dict[str, float] = field(default_factory=dict)
    rig: dict | None = None  # s27a15 only: named groups, see adapters/s27a15.py
    # Odom (x, y, yaw) interpolated at each camera's header stamp. Latest
    # S_BASE_ODOM is newer than a 1.6 Hz head frame while the base yaws.
    odom_at_image: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    # Live pinhole from ``/isaac/head_camera/camera_info``. None until the
    # first CameraInfo; perception then falls back to the USD-default 60°.
    head_k: tuple[float, float, float, float] | None = None
    head_d: tuple[float, ...] = ()

    def model_state(self) -> np.ndarray:
        return C.model_state_from_state(self.state)

    def image_ages(self) -> dict[str, float]:
        """t_sim - stamp per image that carries a stamp (sim seconds)."""
        return {
            k: float(self.t_sim - self.image_t_sim[k])
            for k in self.images
            if self.image_t_sim.get(k) is not None
        }

    def image_list(self) -> list[np.ndarray]:
        """Images in the ai_data_contract order (head, wrist_left, wrist_right)."""
        return [self.images[k] for k in C.CAMERA_KEYS if k in self.images]


@dataclass
class ActionChunk:
    t0: float  # sim time of actions[0]
    actions: np.ndarray  # (H, 20) canonical layout — (H, 15) under s27a15
    dt: float = 1.0 / 30.0


class PolicyAdapter(ABC):
    """Owns a loaded model; maps canonical Obs -> canonical action chunks.

    Adapters run wherever the model runs (in-process for LocalBackend, on
    the server for RemoteBackend), so heavyweight imports must stay inside
    __init__/methods — never at module import time.
    """

    name: str = "adapter"
    # Seconds between chunk steps at the checkpoint's NATIVE control rate.
    # Backends stamp this into ActionChunk.dt — a wrong dt replays the chunk
    # at the wrong speed (DROID checkpoints run 15 Hz: 1/30 here would play
    # them at 2x, inflating every delta into the clamp — RUNBOOK §1.2e).
    chunk_dt: float = 1.0 / 30.0

    def reset(self, task: str) -> None:
        """New episode; ``task`` is the language instruction (may be '')."""
        self.task = task

    # Which action vector ``infer`` returns. Every sim adapter emits the
    # canonical 20-dim contract; the real-robot ``s27a15`` layout emits the
    # rig's own 15-dim vector, which the sim executor CANNOT consume. It is
    # an attribute rather than a width test because width never implies a
    # layout (F-45) — 20 and 15 happen to differ, the next one may not.
    action_space: str = "canonical20"

    @abstractmethod
    def infer(self, obs: Obs) -> np.ndarray:
        """Return (H, 20) canonical actions for sim time obs.t_sim onward.

        Under ``action_space == "s27a15"`` this is (H, 15) in the rig's
        layout instead — decode it with
        ``camelo.policy.adapters.s27a15.unpack_action``.
        """
