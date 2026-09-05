"""Model-free adapter: hold the current pose (optionally wave one joint).

Exercises the entire loop — obs, wire, executor, bridge — with no GPU or
model dependencies. ``wave=True`` superimposes a small sinusoid on each
arm's last two joints so motion is visible in the sim viewer.
"""

from __future__ import annotations

import math

import numpy as np

from camelo import contracts as C
from camelo.policy.base import Obs, PolicyAdapter


class DummyAdapter(PolicyAdapter):
    name = "dummy"

    def __init__(self, wave: bool = False, amplitude: float = 0.05, period_s: float = 5.0):
        self.wave = wave
        self.amplitude = amplitude
        self.period_s = period_s

    def infer(self, obs: Obs) -> np.ndarray:
        hold = np.zeros(C.MODEL_ACTION_DIM, dtype=np.float32)
        hold[C.M_LEFT_ARM] = obs.state[C.S_LEFT_ARM]
        hold[C.M_LEFT_GRIP] = obs.state[C.S_LEFT_GRIP]
        hold[C.M_RIGHT_ARM] = obs.state[C.S_RIGHT_ARM]
        hold[C.M_RIGHT_GRIP] = obs.state[C.S_RIGHT_GRIP]

        chunk = np.tile(hold, (C.MODEL_HORIZON, 1))
        if self.wave:
            dt = 1.0 / 30.0
            for k in range(C.MODEL_HORIZON):
                phase = 2.0 * math.pi * (obs.t_sim + k * dt) / self.period_s
                offset = self.amplitude * math.sin(phase)
                for arm in (C.M_LEFT_ARM, C.M_RIGHT_ARM):
                    chunk[k, arm.stop - 2 : arm.stop] += offset
        return C.model_to_canonical_actions(chunk)
