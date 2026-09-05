"""Image staleness instrument — GRASP_EXPERIMENT_PLAN.md N2b step 1.

The render floor that kills ACT (P3, GRASP_EXPERIMENT_PROTOCOL.md §7) is a
SIM-time quantity: the offline lag grid holds an image for ``lag`` dataset
frames at 30 fps, so ``lag`` is a period in 1/30 sim-s. Every camera rate
measured on the rig so far was WALL Hz (``ros2 topic hz``, ``make
check-obs``), and the sim runs at ~0.17 x wall under eval load — so the two
have never been compared in the same units (the AGENTS.md "reading
wall-clock as sim time" trap, in the other direction).

This module measures the sim-time quantities at the two places they exist:

* **arrival** — each frame the camera workers hand the collector, with the
  frame's own ``header.stamp`` (sim time, set by the sim's ROS2CameraHelper),
  the ``/isaac/clock`` value the collector held when it drained the frame,
  and wall time. ``FrameLog`` keeps a bounded per-camera history of these.
* **consumption** — the frame that is actually inside the ``Obs`` handed to
  ``backend.infer``, aged against that ``Obs``'s ``t_sim``.
  ``ImageAgeMeter.consume`` samples it once per inference.

``ImageAgeMeter.stats`` folds both into flat per-episode keys (see
``summarize`` for the list). Two of them answer the plan's question:

* ``image_lag_frames30`` — the P3-comparable number: 30 x the median
  inter-frame stamp gap of the slowest camera, i.e. the rig's delivered
  period expressed on the lag grid. P3's ~80 % retention line is lag <= 2.7
  (>= ~11 Hz sim). Stamp gaps need no clock, so this survives a camera
  stamp that is offset from ``/isaac/clock``. Read it beside
  ``image_stamp_gap_k_min_s`` and ``image_lag_frames30_mean``: the median
  alone cannot tell a slow wire from a fast wire losing frames — the min is
  the wire's period floor (no gap is shorter than one wire period) and the
  mean carries the long gaps a median discards (docs/realdata/16 R-45).
* ``image_age_frames30`` — 30 x the worst camera's mean age at consumption.
  Larger than half a period by whatever the pipeline adds (render->publish
  ->ingest->drain, plus up to one inference blackout, F-88). If it is
  NEGATIVE or wildly larger than the period, the camera stamps and
  ``/isaac/clock`` do not share an origin (the bridge dropped OmniGraph
  time for exactly that reason — see ``image_drain_age_*_median_s``) and
  only the gap-based number is trustworthy.

Both of the above MEASURE. The third thing here GATES: ``stale_image_report``
answers "is the frame the policy is about to consume still alive?" and the
runner turns a yes into ``StaleImageError`` (U-23, docs/realdata/16). That
check deliberately reads the frame's WALL receipt time, not its sim-time age:
``get_obs`` never clears ``ObsCollector.images``, so a camera that dies
mid-rollout keeps serving its last decoded frame forever, and the age that
detects it must not depend on the camera's own stamp origin agreeing with the
clock (on the rig it demonstrably need not — see ``image_drain_age_*``, which
can read negative). Receipt time is recorded by the collector's own drain, so
a dead camera's age grows without bound whatever its stamps say.

Numpy-only, no ROS: the collector feeds it, the runner reads it, tests run
it offline.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Mapping
from typing import NamedTuple

import numpy as np

# Default staleness threshold for a `--world real` rollout, in WALL seconds.
# The bar is the replan cadence, not the 20 Hz control tick: `ChunkExecutor`
# asks for a new chunk every `replan_after_steps = 8` steps at dt = 1/30 s ->
# inference at 3.75 Hz, and this check samples once per inference (U-23). 0.5 s
# is ~2 replan intervals — long enough that a merely slow camera does not trip
# it, short enough that a DEAD one is caught within one chunk.
DEFAULT_MAX_IMAGE_AGE_S = 0.5

# The offline lag grid's frame rate (dataset fps). A sim period of 1/30 s is
# lag 1; the plan's ">= ~11 Hz sim" threshold is lag 2.7 on this grid.
LAG_GRID_FPS = 30.0

# Enough for a 120 sim-s episode at 30 frames/sim-s with headroom; the
# collector lives for a whole batch, so the log must be bounded.
DEFAULT_FRAME_LOG_MAXLEN = 50_000


class FrameArrival(NamedTuple):
    """One frame reaching the collector."""

    t_stamp: float | None  # header.stamp in sim-s; None when unstamped (0)
    t_clock: float | None  # /isaac/clock the collector held at drain time
    t_wall: float  # monotonic wall time at drain


class FrameLog:
    """Bounded per-camera arrival history with a monotonic frame counter.

    Not thread-safe on its own — the collector calls it under its lock.
    """

    def __init__(self, keys: Iterable[str], maxlen: int = DEFAULT_FRAME_LOG_MAXLEN):
        self._logs: dict[str, deque[FrameArrival]] = {
            key: deque(maxlen=maxlen) for key in keys
        }
        self._total: dict[str, int] = dict.fromkeys(self._logs, 0)

    @property
    def keys(self) -> list[str]:
        return list(self._logs)

    def record(
        self, key: str, t_stamp: float | None, t_clock: float | None, t_wall: float
    ) -> None:
        if key not in self._logs:
            self._logs[key] = deque(maxlen=DEFAULT_FRAME_LOG_MAXLEN)
            self._total[key] = 0
        self._logs[key].append(FrameArrival(t_stamp, t_clock, t_wall))
        self._total[key] += 1

    def last_wall(self) -> dict[str, float | None]:
        """Monotonic wall time each camera's NEWEST frame reached the collector.

        ``None`` for a camera that has never delivered one. This is the
        liveness clock behind ``stale_image_report`` — see the module
        docstring for why it is wall receipt time and not the frame's stamp.
        """
        return {key: (log[-1].t_wall if log else None) for key, log in self._logs.items()}

    def marker(self) -> dict[str, int]:
        """Frame counters now — pass back to ``since`` to bracket a window."""
        return dict(self._total)

    def since(self, marker: Mapping[str, int] | None = None) -> dict[str, list[FrameArrival]]:
        """Arrivals per camera after ``marker`` (all of them when None).

        Capped by the deque: a window longer than ``maxlen`` frames returns
        the newest ``maxlen``.
        """
        out: dict[str, list[FrameArrival]] = {}
        for key, log in self._logs.items():
            start = (marker or {}).get(key, 0)
            n = self._total[key] - start
            n = max(0, min(n, len(log)))
            out[key] = list(log)[len(log) - n :] if n else []
        return out


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p90": None, "max": None, "min": None}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
        "min": float(arr.min()),
    }


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def summarize(
    consumed_ages: Mapping[str, list[float]],
    arrivals: Mapping[str, list[FrameArrival]],
    sim_seconds: float | None,
    wall_seconds: float | None,
    n_inferences: int,
) -> dict:
    """Flat per-episode keys from consumption samples and arrival history.

    Per camera ``k``:
      image_age_k_{mean,p90,max,min}_s   age at consumption (t_sim - stamp)
      image_frames_k                     frames that reached the collector
      image_rate_k_sim_hz                frames / sim_seconds
      image_rate_k_wall_hz               frames / wall_seconds
      image_stamp_gap_k_median_s         median gap between consecutive stamps
      image_stamp_gap_k_mean_s           mean of the same gaps
      image_stamp_gap_k_min_s            smallest gap seen — the wire's own
                                         period floor: a 15 fps wire can never
                                         produce a gap below 1/15, so a min of
                                         0.0667 says 30 fps + loss while a
                                         median of 0.0667 says nothing (it
                                         aliases under a slower drain)
      image_stamp_gap_k_p90_s            90th percentile gap — how bad the
                                         slow tail gets when frames are lost
      image_period_k_frames30            the MEDIAN gap on the lag grid (x 30)
      image_drain_age_k_median_s         median (clock - stamp) at drain;
                                         negative => stamp origin != clock
      image_stamped_k                    False when frames carried no stamp
    Across cameras (worst = slowest / oldest):
      image_lag_frames30                 max_k image_period_k_frames30
      image_lag_frames30_mean            the same from MEAN gaps — a median is
                                         blind to a minority of long gaps
                                         (dropped frames), a mean is not, so
                                         mean >> median means loss, not a slow
                                         wire
      image_rate_sim_hz                  min_k image_rate_k_sim_hz
      image_age_worst_mean_s             max_k image_age_k_mean_s
      image_age_frames30                 that x 30
      image_age_samples                  inferences that carried an age
    A quantity nobody measured reads None, never 0.0 (batch_eval.py rule).
    """
    out: dict = {"image_age_samples": int(n_inferences)}
    keys = sorted(set(consumed_ages) | set(arrivals))
    periods30: list[float] = []
    mean_periods30: list[float] = []
    sim_rates: list[float] = []
    mean_ages: list[float] = []
    for k in keys:
        ages = list(consumed_ages.get(k, []))
        st = _stats(ages)
        for name in ("mean", "p90", "max", "min"):
            out[f"image_age_{k}_{name}_s"] = st[name]
        if st["mean"] is not None:
            mean_ages.append(st["mean"])

        frames = list(arrivals.get(k, []))
        n = len(frames)
        out[f"image_frames_{k}"] = n
        out[f"image_rate_{k}_sim_hz"] = n / sim_seconds if sim_seconds else None
        out[f"image_rate_{k}_wall_hz"] = n / wall_seconds if wall_seconds else None
        if out[f"image_rate_{k}_sim_hz"] is not None:
            sim_rates.append(out[f"image_rate_{k}_sim_hz"])

        stamps = [f.t_stamp for f in frames if f.t_stamp is not None]
        out[f"image_stamped_{k}"] = bool(stamps) if n else None
        gaps = [b - a for a, b in zip(stamps, stamps[1:], strict=False) if b > a]
        gap = _median(gaps)
        out[f"image_stamp_gap_{k}_median_s"] = gap
        # The median aliases: under a drain slower than the wire it reports
        # the DRAIN period, and a 30 fps wire losing frames reports the same
        # 0.0667 s as a healthy 15 fps one. min/p90 separate those — no gap
        # can be shorter than the wire's own period, and the p90 carries the
        # loss the median hides (docs/realdata/16 R-45).
        gap_stats = _stats(gaps)
        for name in ("mean", "min", "p90"):
            out[f"image_stamp_gap_{k}_{name}_s"] = gap_stats[name]
        out[f"image_period_{k}_frames30"] = gap * LAG_GRID_FPS if gap is not None else None
        if gap is not None:
            periods30.append(gap * LAG_GRID_FPS)
        if gap_stats["mean"] is not None:
            mean_periods30.append(gap_stats["mean"] * LAG_GRID_FPS)
        drain = [
            f.t_clock - f.t_stamp
            for f in frames
            if f.t_stamp is not None and f.t_clock is not None
        ]
        out[f"image_drain_age_{k}_median_s"] = _median(drain)

    out["image_lag_frames30"] = max(periods30) if periods30 else None
    out["image_lag_frames30_mean"] = max(mean_periods30) if mean_periods30 else None
    out["image_rate_sim_hz"] = min(sim_rates) if sim_rates else None
    out["image_age_worst_mean_s"] = max(mean_ages) if mean_ages else None
    out["image_age_frames30"] = (
        out["image_age_worst_mean_s"] * LAG_GRID_FPS
        if out["image_age_worst_mean_s"] is not None
        else None
    )
    return out


class ImageAgeMeter:
    """Samples image age at the moment a policy consumes an ``Obs``."""

    def __init__(self):
        self._ages: dict[str, list[float]] = {}
        self.n_inferences = 0

    def consume(self, obs) -> dict[str, float]:
        """Record ``obs``'s per-camera ages (t_sim - stamp); returns them."""
        self.n_inferences += 1
        ages = obs.image_ages()
        for key, age in ages.items():
            self._ages.setdefault(key, []).append(float(age))
        return ages

    def stats(
        self,
        arrivals: Mapping[str, list[FrameArrival]],
        sim_seconds: float | None,
        wall_seconds: float | None,
    ) -> dict:
        return summarize(self._ages, arrivals, sim_seconds, wall_seconds, self.n_inferences)


def format_ages(ages: Mapping[str, float]) -> str:
    """Compact ``head:0.12 wl:0.15 wr:0.09`` for the per-chunk log line."""
    short = {"head": "head", "wrist_left": "wl", "wrist_right": "wr"}
    return " ".join(f"{short.get(k, k)}:{v:.2f}" for k, v in sorted(ages.items()))


# --------------------------------------------------------------------------
# the liveness guard (U-23) — a DEAD camera, not a slow one
# --------------------------------------------------------------------------
class StaleImageError(RuntimeError):
    """A camera stopped delivering while the policy was still consuming it.

    Raised by the runner at the inference seam, never by the collector: the
    collector's job is to cache whatever arrives, and a rollout is the only
    context in which a frozen frame is a fault rather than a fact. Handled
    exactly like `camelo.runner.recenter.StartPoseNotReached` — it escapes
    `run_rollout`, `scripts/run_policy.py`'s guarded `finally` deactivates the
    arms on the way out, and the process exits non-zero.
    """


def stale_image_ages(
    last_wall: Mapping[str, float | None],
    now_wall: float,
    max_age_s: float,
) -> dict[str, float]:
    """``{camera: age_s}`` for every camera whose newest frame is too old.

    ``last_wall`` is ``FrameLog.last_wall()`` (monotonic receipt time per
    camera); a camera that has never delivered a frame reads ``inf`` rather
    than being skipped — "no frame at all" is not fresher than "an old frame".
    """
    stale: dict[str, float] = {}
    for key, t_wall in sorted(last_wall.items()):
        age = math.inf if t_wall is None else float(now_wall - t_wall)
        if age > max_age_s:
            stale[key] = age
    return stale


def stale_image_report(
    last_wall: Mapping[str, float | None],
    now_wall: float,
    max_age_s: float | None,
) -> str | None:
    """The fault line for a stale camera, or ``None`` when every camera is live.

    ``max_age_s`` of ``None`` (or <= 0) is the guard switched OFF and always
    returns ``None`` — that is the sim default and the explicit opt-out, and
    it is the ONLY way this returns None while a camera is in fact dead.
    """
    if max_age_s is None or max_age_s <= 0.0:
        return None
    stale = stale_image_ages(last_wall, now_wall, max_age_s)
    if not stale:
        return None
    detail = ", ".join(
        f"{key} {'never delivered a frame' if math.isinf(age) else f'{age:.2f} s old'}"
        for key, age in stale.items()
    )
    live = {k: v for k, v in sorted(last_wall.items()) if k not in stale}
    live_detail = ", ".join(
        f"{k} {now_wall - v:.2f} s" for k, v in live.items() if v is not None
    )
    return (
        f"STALE IMAGE: {detail} (limit {max_age_s:.2f} s wall). The collector "
        "never clears a cached frame, so this camera has been feeding the "
        "policy the SAME image since it died (docs/realdata/16 U-23). Fix the "
        "camera before rerunning; --max-image-age-s 0 disables the guard."
        + (f" Live cameras: {live_detail}." if live_detail else "")
    )
