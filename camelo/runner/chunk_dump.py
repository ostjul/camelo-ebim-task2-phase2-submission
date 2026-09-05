"""``--chunk-dump``: archive every policy chunk that reaches the executor.

OFF by default (docs/CONTRACTS.md). A real-robot rollout is expensive to
repeat, and the ``policy chunk #N`` log line only ever shows the jump/lead
numbers for the arrival that just happened — this exists so the whole run
can be re-examined offline: every chunk, the context
``ChunkExecutor.set_chunk`` spliced it against, and the observation it was
computed from where that is reachable.

One record per chunk, on both the synchronous and
``camelo.control.async_inference`` paths (``camelo/runner/episode_runner.py``,
beside the ``policy chunk #N`` log line and the ``executor.set_chunk(...)``
call):

* ``chunk_num`` -- 1-based, matching the printed line.
* ``t0`` -- the chunk's own time base (sim s).
* ``t_sim_arrival`` -- sim time of the tick the chunk was installed on.
* ``wall_idx`` -- ``(t_sim_arrival - t0) / dt``, the same wall-clock index
  ``ChunkExecutor``'s ``_Chunk.wall_index`` computes internally.
* ``spliced_idx`` -- ``wall_idx`` shifted by the executor's splice offset
  (``ChunkExecutor.last_splice_shift``, 0.0 before anything is measurable)
  -- where playback actually resumes.
* ``jump`` -- ``ChunkExecutor.last_arrival_jump_rad``: the pre-clamp arm
  discontinuity this arrival demands. NaN before the first measurable
  arrival (no prior command to jump from).
* ``lead`` -- ``ChunkExecutor.last_cmd_lead_rad``: |command - measured| at
  the executor's last ``step``. NaN before the first ``step``.
* the observation state vector: the ``Obs.state`` the chunk was actually
  computed from, where that is reachable (the SYNCHRONOUS path -- the same
  ``obs`` object passed to ``backend.infer``), else the MEASURED state at
  the tick the chunk arrived (``camelo.control.async_inference``: the
  worker returns only the chunk, never the submitted observation, so the
  one that produced it is not available here). Which is which is named,
  not left to be guessed: the npz key is ``obs_state_at_inference`` or
  ``obs_state_at_arrival`` -- never both, and never a plain ``obs_state``.
* ``last_cmd_arms`` -- ``ChunkExecutor.last_commanded_arms``, (14,): what
  was commanded right before this chunk arrived. ``set_chunk`` never
  touches it, so it is unaffected by which side of the call this is read
  on. NaN row before the executor's first ``step``.
* ``actions`` -- the (H, ``ACTION_DIM``) float32 block actually handed to
  ``set_chunk`` -- i.e. AFTER ``chunk_for_executor`` widens an s27a15
  chunk onto the canonical slots, matching what the executor plays, not
  the adapter's raw output.

Memory is trivial by construction (a real rollout replans a few times a
second; 64 chunks x 21 x 15 floats is nothing), so every field is just
appended to a python list and stacked once, at ``write()``.

``write()`` never raises into the control loop by itself -- but it is the
CALLER's job to wrap it in ``try``/``except`` and log (see
``episode_runner.run_rollout``'s guarded ``finally``), the same convention
every other end-of-rollout report in this package follows.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

#: (14,) -- executor arm dim, used to fill a NaN row when nothing has been
#: commanded yet (the very first chunk of a run).
_ARM_DIM = 14

OBS_STATE_AT_INFERENCE = "obs_state_at_inference"
OBS_STATE_AT_ARRIVAL = "obs_state_at_arrival"


def _stack(arrays: list[np.ndarray], fallback_shape: tuple[int, ...]) -> np.ndarray:
    """Stack ``arrays`` into one ``(N, *shape)`` block, or an object array.

    Chunks have a fixed H per run, so this is normally a plain
    ``np.stack``. If H (or any other per-record shape) varied, a plain
    stack would raise -- fall back to an ``(N,)`` object array instead of
    losing rows.
    """
    if not arrays:
        return np.empty((0, *fallback_shape), dtype=np.float32)
    shapes = {a.shape for a in arrays}
    if len(shapes) == 1:
        return np.stack(arrays).astype(np.float32)
    out = np.empty(len(arrays), dtype=object)
    for i, a in enumerate(arrays):
        out[i] = a
    return out


class ChunkDumpRecorder:
    """Buffers one record per chunk arrival; ``write()`` flushes one .npz."""

    def __init__(self, obs_state_field: str):
        if obs_state_field not in (OBS_STATE_AT_INFERENCE, OBS_STATE_AT_ARRIVAL):
            raise ValueError(
                f"obs_state_field must be {OBS_STATE_AT_INFERENCE!r} or "
                f"{OBS_STATE_AT_ARRIVAL!r}, got {obs_state_field!r}"
            )
        self.obs_state_field = obs_state_field
        self.n = 0
        self._chunk_num: list[int] = []
        self._t0: list[float] = []
        self._t_sim_arrival: list[float] = []
        self._wall_idx: list[float] = []
        self._spliced_idx: list[float] = []
        self._jump: list[float] = []
        self._lead: list[float] = []
        self._obs_state: list[np.ndarray] = []
        self._last_cmd_arms: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []

    def record(
        self,
        *,
        chunk_num: int,
        t0: float,
        t_sim_arrival: float,
        wall_idx: float,
        spliced_idx: float,
        jump: float | None,
        lead: float | None,
        obs_state: np.ndarray,
        last_cmd_arms: np.ndarray | None,
        actions: np.ndarray,
    ) -> None:
        self._chunk_num.append(int(chunk_num))
        self._t0.append(float(t0))
        self._t_sim_arrival.append(float(t_sim_arrival))
        self._wall_idx.append(float(wall_idx))
        self._spliced_idx.append(float(spliced_idx))
        self._jump.append(float("nan") if jump is None else float(jump))
        self._lead.append(float("nan") if lead is None else float(lead))
        self._obs_state.append(np.asarray(obs_state, dtype=np.float32).copy())
        self._last_cmd_arms.append(
            np.full(_ARM_DIM, np.nan, dtype=np.float32)
            if last_cmd_arms is None
            else np.asarray(last_cmd_arms, dtype=np.float32).copy()
        )
        self._actions.append(np.asarray(actions, dtype=np.float32).copy())
        self.n += 1

    def write(self, path: Path | str, args: dict | None = None) -> None:
        """Stack every record into one .npz, with a JSON sidecar field.

        ``args`` (default ``{}``) is whatever the caller wants recorded
        about the run that produced this dump -- ``run_policy.py`` passes
        ``vars(args)``. Embedded as a JSON string (``meta_json``) rather
        than a second file, so ``--chunk-dump PATH`` really is ONE
        artifact.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        obs_state = _stack(self._obs_state, self._obs_state[0].shape if self._obs_state else (0,))
        last_cmd_arms = _stack(self._last_cmd_arms, (_ARM_DIM,))
        actions = _stack(self._actions, self._actions[0].shape if self._actions else (0, 0))
        layout = {
            "chunk_num": "1-based, matches the printed 'policy chunk #N' line",
            "t0": "chunk time base (sim s)",
            "t_sim_arrival": "sim time this chunk was installed on the executor",
            "wall_idx": "(t_sim_arrival - t0) / dt -- the executor's wall-clock index",
            "spliced_idx": "wall_idx shifted by the executor's splice offset",
            "jump": "pre-clamp arm discontinuity this arrival demands [rad]; "
            "NaN before the first measurable arrival",
            "lead": "|command - measured| at the executor's last step [rad]; "
            "NaN before the first step",
            self.obs_state_field: (
                "the Obs.state the chunk was computed from"
                if self.obs_state_field == OBS_STATE_AT_INFERENCE
                else "the MEASURED Obs.state at the tick the chunk arrived -- "
                "under async inference the observation the chunk was computed "
                "from is never returned by the worker, so this is the closest "
                "reachable stand-in"
            ),
            "last_cmd_arms": "(14,) executor's last COMMANDED arm targets "
            "before this chunk arrived; NaN row before the first step",
            "actions": "(H, ACTION_DIM) float32 -- the widened block actually "
            "handed to ChunkExecutor.set_chunk",
        }
        meta = {"layout": layout, "args": args or {}, "n_chunks": self.n}
        arrays = {
            "chunk_num": np.asarray(self._chunk_num, dtype=np.int64),
            "t0": np.asarray(self._t0, dtype=np.float64),
            "t_sim_arrival": np.asarray(self._t_sim_arrival, dtype=np.float64),
            "wall_idx": np.asarray(self._wall_idx, dtype=np.float64),
            "spliced_idx": np.asarray(self._spliced_idx, dtype=np.float64),
            "jump": np.asarray(self._jump, dtype=np.float64),
            "lead": np.asarray(self._lead, dtype=np.float64),
            self.obs_state_field: obs_state,
            "last_cmd_arms": last_cmd_arms,
            "actions": actions,
            "meta_json": np.asarray(json.dumps(meta, default=str)),
        }
        np.savez(path, **arrays)
        log.info("chunk dump: %d chunks -> %s", self.n, path)
