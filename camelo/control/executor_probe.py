"""What the executor can track AT ALL, independent of any checkpoint.

The chunk executor rate-limits every command to `max_delta_per_tick` rad
per tick (F-53: the only rate limiter in the command path). That budget is
per TICK, so the deliverable joint speed is `max_delta_per_tick x control
rate` — and the control rate is not a constant: `run_rollout` calls
`backend.infer()` synchronously inside the loop, so a policy with 1.2 s
inference publishes far fewer commands per sim second than a 0.1 s one
(F-88).

`oracle_rollout` feeds a KNOWN-GOOD action stream — the recorded demo
actions themselves — through the real `ChunkExecutor` at a given control
rate. The "policy" is an oracle: at every replan it hands back the true
future actions, timestamped at the current sim time. Whatever clamping
and tracking error come out are therefore properties of the HARNESS, and
they bound what any checkpoint could possibly achieve at that rate.

Numpy only, no ROS, no torch — this runs offline on any box.
"""

from __future__ import annotations

import numpy as np

from camelo import contracts as C
from camelo.control.chunk_executor import DATASET_FPS, ChunkExecutor


def demo_joint_speeds(actions: np.ndarray, fps: float = DATASET_FPS) -> np.ndarray:
    """Per-frame worst-joint |delta| of an arm action stream, in rad/s.

    This is what the demonstrations ask of the arms; compare it against
    `max_delta_per_tick * control_hz` to see whether the executor could
    reproduce them even in principle.
    """
    actions = np.asarray(actions, dtype=np.float64)
    arms = np.concatenate([actions[:, C.A_LEFT_ARM], actions[:, C.A_RIGHT_ARM]], axis=1)
    if len(arms) < 2:
        return np.zeros(0)
    return np.abs(np.diff(arms, axis=0)).max(axis=1) * fps


def oracle_rollout(
    actions: np.ndarray,
    ticks_per_sim_s: float,
    replan_steps: int = 8,
    max_delta: float = 0.05,
    chunk_size: int = 50,
    fps: float = DATASET_FPS,
    allow_below_nominal: bool = False,
) -> dict:
    """Replay `actions` through the executor with a perfect policy.

    `ticks_per_sim_s` is commands per **SIM** second — `loop_ticks /
    loop_sim_seconds` from a scored run, never `ticks / wall_seconds`. The
    two differ by the sim/wall factor (~6.5x) and using the wall figure
    understates the executor's budget by exactly that much, which inverts
    the conclusion this probe exists to support (F-91).

    Returns the executor's own counters plus the tracking error of the
    command stream against the demo it replays.
    """
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != C.ACTION_DIM:
        raise ValueError(f"expected (T, {C.ACTION_DIM}) canonical actions, got {actions.shape}")
    # The nominal loop is 20 Hz WALL and the sim runs slower than wall, so a
    # sim-time rate below 20 is almost certainly a wall-time number passed
    # by mistake. Refuse rather than return a plausible wrong table.
    if ticks_per_sim_s < 20.0 and not allow_below_nominal:
        raise ValueError(
            f"ticks_per_sim_s={ticks_per_sim_s} is below the 20 Hz nominal loop rate, "
            "which almost always means a wall-time figure was passed (F-91): the sim "
            "runs ~6.5x slower than wall, so ticks per SIM second is ~6.5x LARGER than "
            "ticks per wall second. Pass loop_ticks / loop_sim_seconds. A genuinely "
            "degraded rig can be below 20 — say so with allow_below_nominal=True."
        )

    executor = ChunkExecutor(max_delta_per_tick=max_delta, replan_after_steps=replan_steps)
    # The executor reads only the arm slices of the state, and only to seed
    # its first command from the live pose; everything after is its own
    # integrated command stream.
    state = np.zeros(C.STATE_DIM, dtype=np.float32)
    state[C.S_LEFT_ARM] = actions[0][C.A_LEFT_ARM]
    state[C.S_RIGHT_ARM] = actions[0][C.A_RIGHT_ARM]

    errors: list[float] = []
    t_sim, tick = 0.0, 1.0 / ticks_per_sim_s
    horizon = max(len(actions) - chunk_size, 1) / fps
    while t_sim < horizon:
        if executor.needs_replan(t_sim):
            start = int(t_sim * fps)
            executor.set_chunk(t_sim, actions[start : start + chunk_size], 1.0 / fps)
        command = executor.step(t_sim, state)
        if command is not None:
            frame = actions[min(int(t_sim * fps), len(actions) - 1)]
            want = np.concatenate([frame[C.A_LEFT_ARM], frame[C.A_RIGHT_ARM]])
            got = np.concatenate([command.left_arm, command.right_arm])
            errors.append(float(np.abs(want - got).max()))
        t_sim += tick

    stats = executor.stats
    return {
        "ticks_per_sim_s": ticks_per_sim_s,
        "replan_steps": replan_steps,
        "ticks": stats.ticks,
        "clamp_fraction": stats.clamped_ticks / stats.ticks if stats.ticks else None,
        "max_requested_delta": stats.max_requested_delta,
        "track_p99": float(np.percentile(errors, 99)) if errors else None,
        "track_max": float(np.max(errors)) if errors else None,
    }
