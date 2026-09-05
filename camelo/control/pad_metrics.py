"""Pad motion over one episode — the signal a floored IoU cannot show.

Scored IoU is floored at 0.0 (`c18c4b9`: the dataset's own actions score
0.586, yet a real 268 mm -> 166 mm improvement still read exactly 0.0), so
IoU cannot rank two failing policies. The grasp protocol's second sim
intermediate metric (GRASP_EXPERIMENT_PROTOCOL.md §0: "pad displacement
toward the target (mm) and pad lifted off the liner") is the "did it
actually move the thing" reading IoU hides — and nothing in the eval loop
records it today. `camelo/runner/episode_runner.py` already reads the live
pad pose (`collector.object_xyz("thermalpad")`), but only when the grasp
gate is enabled, and then throws the value away.

This module is the accumulator for that reading. Numpy-only on purpose
(`camelo/control/` is a numpy-only layer): the whole episode's worth of
bookkeeping is unit-testable with no ROS, no sim and no policy. The caller
supplies the poses; `camelo/ros/obs_collector.py` reads them LIVE from
`OBJECT_POSES_TOPIC`, because the assembly gets nudged and a launch-time
constant would be wrong exactly when it matters.

**Everything here is reported in MILLIMETRES.** The topic speaks metres;
the protocol and F-98 speak mm. The conversion happens once, here, and
nowhere else — two conversion sites is how a summary starts disagreeing
with the data underneath it (AGENTS.md, "the summary lies").

**A quantity that was never observed reads `None`, never `0.0`.** If the
pad pose never arrived, `pad_disp_max_mm` is `None`; if no target was ever
supplied, `pad_disp_toward_target_mm` is `None`. A 0.0 would read as
"measured, and it did not move", which is the floored-IoU failure in a new
costume: a confident wrong answer where there was no measurement at all.

`PAD_MOTION_EPS_MM` is a **chosen** threshold, not a measured one — see its
comment. (House rule: a measured constant cites the finding that produced
it, a chosen one says out loud that it was chosen.)

**Lift (P4, GRASP_EXPERIMENT_PROTOCOL.md §1).** ``pad_lift_max_mm`` — the
max signed Δz of the mesh centroid — is **retired as evidence** (§0, Wave
10): the pad is deformable, so a shove buckles it and its centroid rises
10–14 mm for a few hundred ms while it is dragged along the liner, and
drags and grasps read the same ratio of "lift" to travel (0.13–0.17). It
is still emitted so old and new ``results.csv`` files share columns, but
nothing should be concluded from it. What separates a carried pad from a
shoved one is *time*: a grasped pad stays up for as long as the arm holds
it, a buckled one falls back within a second. So the metric that replaces
it is **sustained height above the liner plane**:

    pad_lift_sustained_mm = max over every window of PAD_LIFT_WINDOW_S
                            continuous sim-seconds of
                            min(height above the rest centroid in that window)

i.e. "the pad was at least this high for a full window". The liner plane
is the pad's own live centroid at its first observation (the same baseline
as displacement), which self-calibrates the ~4.7 mm the centroid sits
below the frozen prim origin (GRASP_W1.md §5 caveat iii). The window is in
SIM time and time-weighted — the executor ticks ~2× faster than the sim
clock's resolution, so ~half of all rows repeat a timestamp and a tick
count would be a harness number. Measured on the six per-tick episodes
that existed when this was written (all non-grasps: closes on air and
shoves, `outputs/eval/c1b_*`), the largest 1 s-sustained height was
3.0 mm and the largest 2 s-sustained height 0.4 mm, against sub-second
peaks of 9–14 mm. ``pad_lifted`` is that value against
``PAD_LIFT_EPS_MM``; both constants are **chosen** and say so.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

# The one metres -> millimetres conversion in this module (see the header).
_M_TO_MM = 1000.0

# Noise floor for "the pad moved at all", in mm. **CHOSEN, not measured**:
# no probe has characterised the ground-truth pose jitter of a pad resting
# on a compliant liner, so this is a judgement call and a later finding that
# measures it should replace this number and cite itself here. 1 mm sits an
# order or two below every displacement this metric exists to separate (the
# c18c4b9 calibration moved 268 mm -> 166 mm; the F-98 grasp envelope is
# 45 mm wide), and above the sub-mm settling wobble a resting object shows,
# so `pad_first_motion_sim_s` marks a push rather than a twitch.
PAD_MOTION_EPS_MM = 1.0

# The sustained-lift window, in SIM seconds. **CHOSEN, not measured**, and
# bracketed by the data in the module header: on every known non-grasp the
# centroid's shove-induced rise is gone within ~0.4 s at 5 mm, and the
# worst 1 s-sustained height is 3.0 mm, while a carried pad stays up for
# the whole transport (seconds). 1.0 s is inside that gap with room on
# both sides; the tool `tools/dgx_probes/pad_lift.py --window-s` sweeps it
# so a later finding can move it and cite itself here.
PAD_LIFT_WINDOW_S = 1.0

# Height above the rest centroid that counts as "lifted", in mm, judged on
# the sustained value above. **CHOSEN, not measured**: 10 mm is 3.3× the
# worst 1 s-sustained non-grasp reading (3.0 mm), an order of magnitude
# above the pad's own thickness (~1 mm), and far below what a carry to the
# target needs. A pad held 10 mm up for a full second is not resting on
# the liner.
PAD_LIFT_EPS_MM = 10.0

XYZ = Sequence[float] | np.ndarray


def sustained_min(
    t_s: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    window_s: float,
) -> tuple[float, float] | None:
    """The largest level a signal HELD for a full ``window_s`` of time.

    Returns ``(level, t_start)``: the maximum, over every window
    ``[t_i, t_i + window_s]`` that the samples fully cover, of the minimum
    of ``values`` inside that window — and the ``t_s`` at which that
    window starts. ``None`` when the samples span less than ``window_s``
    (there is no full window, which is not the same as "held nothing").

    The window is a span of ``t_s``, not a count of samples: repeated
    timestamps (two ticks inside one sim-clock update) add samples but no
    time, and a gap in the log covers time with no samples. A window is
    the samples with ``t_i <= t <= t_i + window_s`` plus the first sample
    at or past the far end, so it always spans at least ``window_s``. The
    signal is treated as piecewise-constant between samples — a dip that
    fell between two ticks is not seen, which is the instrument's
    resolution, not the metric's.

    Samples are sorted by time (stably) before use; a non-finite value
    is a sample that was not observed and is dropped, never treated as
    -inf or 0.
    """
    t = np.asarray(t_s, dtype=np.float64).reshape(-1)
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if t.shape != v.shape:
        raise ValueError(f"t_s and values must match in length, got {t.size} and {v.size}")
    if not window_s > 0.0:
        raise ValueError(f"window_s must be positive, got {window_s}")
    keep = np.isfinite(t) & np.isfinite(v)
    t, v = t[keep], v[keep]
    n = t.size
    if n == 0:
        return None
    order = np.argsort(t, kind="stable")
    t, v = t[order], v[order]
    if t[-1] - t[0] < window_s:
        return None

    # Sliding-window minimum over a window whose two ends both move
    # forward monotonically: a deque of candidate indices with increasing
    # values, O(n) overall.
    best: float | None = None
    best_t = 0.0
    dq: list[int] = []
    head = 0  # dq[head:] is live; popping from the front by index
    j = 0  # first index NOT yet folded into the window
    for i in range(n):
        end = t[i] + window_s
        # Fold in every sample up to and including the first one at/past
        # `end` (the window must span at least window_s).
        while j < n:
            while len(dq) > head and v[dq[-1]] >= v[j]:
                dq.pop()
            dq.append(j)
            j += 1
            if t[j - 1] >= end:
                break
        if t[j - 1] < end:
            break  # ran out of samples before the window closed
        while dq[head] < i:
            head += 1
        level = float(v[dq[head]])
        if best is None or level > best:
            best, best_t = level, float(t[i])
    if best is None:
        return None
    return best, best_t


def _as_xyz(value: XYZ | None) -> np.ndarray | None:
    """(x, y, z) in metres as float64, or None for "no reading this tick".

    A non-finite pose counts as no reading rather than as data: a NaN folded
    into a running max poisons every later tick, and the point of this module
    is that an unobserved quantity says so instead of returning a number. A
    wrong-shaped pose is a caller bug, not a missing reading, so it raises.
    """
    if value is None:
        return None
    xyz = np.asarray(value, dtype=np.float64).reshape(-1)
    if xyz.size != 3:
        raise ValueError(f"pose must be (x, y, z) in metres, got {xyz.size} values")
    if not np.all(np.isfinite(xyz)):
        return None
    return xyz


@dataclass
class PadTracker:
    """Pad motion over one episode: the signal a floored IoU cannot show.

    Call ``update()`` on every control tick and ``stats()`` once at the end.
    The first non-``None`` ``pad_xyz`` is the baseline and everything is
    measured against it, so a rollout that starts before the pose topic has
    published is not penalised — it measures from its first sight of the pad
    rather than from a guessed origin.
    """

    motion_eps_mm: float = PAD_MOTION_EPS_MM
    lift_window_s: float = PAD_LIFT_WINDOW_S
    lift_eps_mm: float = PAD_LIFT_EPS_MM

    baseline_xyz: np.ndarray | None = field(default=None, init=False)
    target_xyz: np.ndarray | None = field(default=None, init=False)
    observed_ticks: int = field(default=0, init=False)
    disp_max_mm: float | None = field(default=None, init=False)
    final_delta_mm: np.ndarray | None = field(default=None, init=False)
    lift_max_mm: float | None = field(default=None, init=False)
    first_motion_sim_s: float | None = field(default=None, init=False)
    # The per-tick height trace the sustained-lift metric needs (P4). The
    # extrema above cannot be re-analysed; this can. `_height_t_sim` holds
    # the sim time of every observed tick including the baseline's, and
    # `_height_mm` the signed height above the baseline centroid.
    _height_t_sim: list[float] = field(default_factory=list, init=False, repr=False)
    _height_mm: list[float] = field(default_factory=list, init=False, repr=False)

    def reset(self) -> None:
        """Start a fresh episode.

        A baseline that survived the reset would measure episode 2 against
        episode 1's pad — the "episodes are not independent trials" failure
        (F-86) that the grasp gate's latch has already had to guard once.
        """
        self.baseline_xyz = None
        self.target_xyz = None
        self.observed_ticks = 0
        self.disp_max_mm = None
        self.final_delta_mm = None
        self.lift_max_mm = None
        self.first_motion_sim_s = None
        self._height_t_sim = []
        self._height_mm = []

    def update(
        self,
        pad_xyz: XYZ | None,
        t_sim: float,
        target_xyz: XYZ | None = None,
    ) -> None:
        """Fold one tick in. Both poses are world metres, or ``None``.

        ``pad_xyz`` is ``None`` before the object-pose topic has published;
        ``target_xyz`` may be ``None`` on every tick of the episode, and then
        the toward-target projection stays ``None`` rather than becoming 0.0.
        """
        target = _as_xyz(target_xyz)
        if target is not None:
            # Latest target wins: the slot pose is read live like the pad, so
            # the freshest one is where the episode was actually aiming. The
            # projection is still anchored at pad(0) (see stats()).
            self.target_xyz = target

        pad = _as_xyz(pad_xyz)
        if pad is None:  # no reading — not a reading of zero motion
            return
        self.observed_ticks += 1

        if self.baseline_xyz is None:
            self.baseline_xyz = pad
            self.final_delta_mm = np.zeros(3, dtype=np.float64)
            self.disp_max_mm = 0.0
            # lift_max_mm stays None here on purpose: the baseline tick's own
            # z - z = 0 is an identity, not an observation, and folding it
            # into the max would clamp every pressed-down pad to 0.0. See
            # stats() for what a baseline-only episode reports.
            # The height trace DOES start here: the sustained-lift window is
            # a span of time, and the baseline tick is the first instant the
            # pad was seen resting on the liner.
            if np.isfinite(t_sim):
                self._height_t_sim.append(float(t_sim))
                self._height_mm.append(0.0)
            return

        delta_mm = (pad - self.baseline_xyz) * _M_TO_MM
        disp_mm = float(np.linalg.norm(delta_mm))
        self.final_delta_mm = delta_mm
        self.disp_max_mm = disp_mm if self.disp_max_mm is None else max(self.disp_max_mm, disp_mm)
        lift_mm = float(delta_mm[2])
        self.lift_max_mm = lift_mm if self.lift_max_mm is None else max(self.lift_max_mm, lift_mm)
        if np.isfinite(t_sim):
            self._height_t_sim.append(float(t_sim))
            self._height_mm.append(lift_mm)
        # A non-finite t_sim is a tick with no clock, and "first motion at
        # nan" is a float that json.dump writes without complaint — the
        # motion is still counted (disp/max above); its TIME waits for the
        # next tick that has one.
        if self.first_motion_sim_s is None and disp_mm > self.motion_eps_mm and np.isfinite(t_sim):
            self.first_motion_sim_s = float(t_sim)

    def stats(self) -> dict:
        """The episode's readings, in mm, for the eval summary."""
        seen = self.baseline_xyz is not None
        return {
            "pad_observed_ticks": int(self.observed_ticks),
            "pad_baseline_seen": bool(seen),
            "pad_disp_max_mm": self.disp_max_mm,
            "pad_disp_final_mm": (
                None if self.final_delta_mm is None else float(np.linalg.norm(self.final_delta_mm))
            ),
            # SIGNED, and a max rather than an abs: "lifted off the liner" is
            # a +z question, so a pad crushed 8 mm DOWN must read -8.0 and not
            # an 8.0 that looks like a lift. A baseline-only episode reports
            # 0.0: one pose arrived and no motion was seen, which is a
            # measurement — unlike no pose at all, which is None.
            # RETIRED as evidence (module header): kept so results.csv
            # columns stay comparable across runs, never to be quoted.
            "pad_lift_max_mm": (0.0 if seen and self.lift_max_mm is None else self.lift_max_mm),
            "pad_disp_toward_target_mm": self._toward_target_mm(),
            "pad_first_motion_sim_s": self.first_motion_sim_s,
            **self._lift_stats(),
        }

    def _lift_stats(self) -> dict:
        """P4: sustained height above the liner plane (module header).

        ``pad_lift_sustained_mm`` is ``None`` when no full window exists —
        the pad was never seen, or was seen for less than the window. That
        is "not measured", distinct from a measured ~0.0 on a pad that sat
        on the liner for the whole episode. ``pad_lifted`` follows it
        (``None`` when it is ``None``). The instrument settings are echoed
        so a row can be read against the window and threshold that judged
        it — the summary carries them up, never averages them.
        """
        held = sustained_min(self._height_t_sim, self._height_mm, self.lift_window_s)
        if held is None:
            sustained_mm: float | None = None
            at_s: float | None = None
            lifted: bool | None = None
        else:
            sustained_mm, at_s = held
            lifted = bool(sustained_mm >= self.lift_eps_mm)
        return {
            "pad_lift_sustained_mm": sustained_mm,
            "pad_lift_sustained_at_sim_s": at_s,
            "pad_lifted": lifted,
            "pad_lift_window_s": float(self.lift_window_s),
            "pad_lift_eps_mm": float(self.lift_eps_mm),
        }

    def _toward_target_mm(self) -> float | None:
        """Signed projection of the FINAL displacement onto pad(0) -> target.

        Positive is progress toward the target, negative is the policy having
        shoved the pad the wrong way — an unsigned distance would score that
        shove as progress.

        ``None``, never 0.0, when there is no target, no baseline, or the
        target sits on top of the baseline: a direction that cannot be
        computed is not a zero-length move along it.
        """
        if self.baseline_xyz is None or self.target_xyz is None or self.final_delta_mm is None:
            return None
        toward_mm = (self.target_xyz - self.baseline_xyz) * _M_TO_MM
        norm_mm = float(np.linalg.norm(toward_mm))
        if norm_mm <= 0.0:
            return None
        return float(self.final_delta_mm @ (toward_mm / norm_mm))
