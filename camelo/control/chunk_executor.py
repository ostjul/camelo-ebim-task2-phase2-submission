"""Turn policy action chunks into smooth per-tick robot commands.

A policy produces chunks of canonical 20-dim actions at dataset framerate
(30 fps sim time). The executor interpolates the chunk at the current sim
time and applies a per-tick arm delta clamp toward the target. The clamp
doubles as ramp-in: the first commands after a reset start from the
measured arm pose, so a chunk that begins far from the current pose is
approached at a bounded rate. The bridge applies raw targets with NO
smoothing or slew limit of its own (F-53: the fr3duo_mobile embodiment
disables all three constants) — this clamp is the ONLY rate limiter in the
entire command path. Never bypass it or raise max_delta_per_tick casually.

Everything is paced on sim time (/isaac/clock), matching the recorder. A
clock that jumps backwards (scene reset) invalidates the current chunk.

`step` assumes the next chunk arrives before the current one runs out,
which is true whenever inference is synchronous (the loop cannot advance
past a reply it is blocked on). Under `camelo.control.async_inference` it
is not, so `exhausted` / `hold` make the shortfall explicit: hold the last
commanded target, count it, and never extrapolate off the end of a chunk.

`chunk_splice` decides where in an arriving chunk playback resumes
(docs/realdata/16 U-31). Indexing a fresh chunk at its wall-clock index
presumes the arm executed the elapsed steps at full speed; behind the
clamp above (and, on the real rig, the companion's own slew cap) it did
not, so the spliced-in target sits wherever the policy *expected* to be
rather than where the arm *is*. Measured on the rig 2026-09-03
(`t6_act_20260903_093858`, 587 ticks / 70 arrivals at a 0.28-0.5 s round
trip): every single arrival demanded more than the clamp allows — max
|delta cmd| = 0.0500 rad on 70 of 70 — so the arm wiggled at the chunk
cadence instead of descending. `nearest` and `offset` close that gap; see
`set_chunk`.

`leash_rad` (docs/realdata/16 U-32) closes the other half of it. The
clamp bounds how fast the command may MOVE but not how far it may drift
from the pose the arm actually holds, so behind the companion's 0.5 rad/s
slew cap (half our clamp) the command runs ahead and every hand-over is
then judged against a target no joint occupies. The leash clips the arm
command to `measured +/- leash_rad` after the clamp, and `nearest`
anchors its search at the MEASURED arm pose rather than the last
commanded one. Measured on the rig 2026-09-03
(`t6_act_20260903_101739`): with `nearest` anchored to the last COMMAND,
every splice shift came out positive (mean +11 rows of 21, max +15) and
the loop held for 413 of 587 ticks. Grippers and the base are never
leashed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from camelo import contracts as C
from camelo.control.base_quantizer import BaseQuantizer

DATASET_FPS = 30.0  # recorder framerate; chunk steps are 1/30 s of sim time apart

SPLICE_POLICIES = ("index", "nearest", "offset")

# `nearest` treats two rows as equally close when their L2 distance (over
# all 14 arm dims) differs by less than this, and then prefers the one
# nearest the wall-clock index. 1e-3 rad is well under one tick of the
# 0.05 rad/tick clamp, so it can only re-order picks that were already
# indistinguishable to the arm (U-32).
NEAREST_TIE_EPS_RAD = 1e-3

# How many rows past the wall-clock index `nearest` may splice FORWARD.
# The chunk is anchored at the pose the policy observed, so the row that
# matches the measured pose sits at or just before the wall index; a pick
# far beyond it is a near-tie artefact, and it costs horizon (playback
# starts that much closer to the end, then starves).
NEAREST_FORWARD_ROWS = 2

# The default command leash on the REAL robot (U-32): 0.3 s of the
# companion's 0.5 rad/s joint_impedance_controller slew, i.e. the furthest
# ahead a command can usefully be while the arm is still catching up. None
# (off) everywhere else, so sim and every synchronous run stay
# byte-identical.
DEFAULT_LEASH_RAD = 0.15


@dataclass
class Command:
    left_arm: np.ndarray  # (7,) rad
    right_arm: np.ndarray  # (7,) rad
    left_gripper: float  # 0..1 open fraction
    right_gripper: float
    base_twist: tuple[float, float, float]
    base_token: str


@dataclass
class ExecutorStats:
    ticks: int = 0
    clamped_ticks: int = 0
    chunks: int = 0
    stale_chunks: int = 0
    # Ticks the loop commanded a HOLD because the chunk ran out before the
    # next one arrived (async inference only; 0 on every synchronous run).
    starved_ticks: int = 0
    max_requested_delta: float = 0.0
    # Chunk hand-over (U-31). An arrival is only *measurable* once something
    # has been commanded and the caller passed `t_sim_now`, so this counts
    # the arrivals the two aggregates below average over and is <= `chunks`.
    # The sums are kept beside the means so every aggregate can be
    # re-derived from what produced it (AGENTS.md, "the summary lies").
    spliced_arrivals: int = 0
    arrival_jump_rad_sum: float = 0.0
    arrival_jump_rad_max: float = 0.0
    splice_shift_sum: float = 0.0
    splice_shift_max: float = 0.0  # the largest-MAGNITUDE shift, signed
    # The command leash (U-32). `leash_active_ticks` counts the ticks the
    # clip actually moved the command; `cmd_lead_rad_max` is the worst
    # |command - measured| over the 14 arm joints AFTER it, and is
    # recorded whether the leash is on or off — with it off it is the
    # diagnostic that says how far the command had run ahead (rig
    # 2026-09-03: mean 0.05, p90 0.12, max 0.28 rad).
    leash_active_ticks: int = 0
    cmd_lead_rad_max: float = 0.0

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        n = self.spliced_arrivals
        d["arrival_jump_rad_mean"] = (self.arrival_jump_rad_sum / n) if n else 0.0
        d["splice_shift_mean"] = (self.splice_shift_sum / n) if n else 0.0
        d["leash_active_pct"] = 100.0 * self.leash_active_ticks / self.ticks if self.ticks else 0.0
        return d


def _arm_targets(actions: np.ndarray) -> np.ndarray:
    """The 14 arm dims of one action row, or of an (n, 20) block.

    The same 14 `step` concatenates and clamps — the executor drives both
    arms and does not know which one the run selected, so the splice
    metric spans all of them.
    """
    return np.concatenate(
        [actions[..., C.A_LEFT_ARM], actions[..., C.A_RIGHT_ARM]], axis=-1
    ).astype(np.float64)


def _nearest_index(
    actions: np.ndarray,
    arms_now: np.ndarray,
    lo: int,
    hi: int | None = None,
    prefer: float | None = None,
    tie_eps: float | None = None,
) -> float:
    """The index in ``[lo, hi]`` whose arm target is closest to ``arms_now``.

    FRACTIONAL, and deliberately so. Rounding to whole rows leaves up to
    half a chunk step of residual discontinuity, and a chunk step is not
    small: on the rig the policy asked for ~0.39 rad per tick against a
    0.05 rad clamp (R-67), so the nearest WHOLE row would still have
    handed the arm several times the clamp and R-68's wiggle would have
    survived the fix. `step` interpolates between rows anyway, so the
    honest question is the closest point on the piecewise-linear path it
    will actually play, not the closest vertex of it.

    L2 over the 14 arm dims — what `step` clamps. ``hi`` defaults to the
    last row; `set_chunk` bounds it forward so a search anchored at the
    measured pose cannot skip most of the horizon (U-32).

    Near-ties are resolved toward ``prefer`` (the wall-clock index), not
    toward the smallest index. This is not cosmetic: when the policy
    hovers, a whole chunk's rows sit within float noise of each other, a
    plain `argmin` lands on an arbitrary row, and on the rig 2026-09-03
    that produced forward picks of +11..+15 rows out of 21 — the chunk was
    exhausted on arrival and the executor held for 413 of 587 ticks. Any
    candidate within ``tie_eps`` of the best distance is treated as equally
    good, and among those the one nearest ``prefer`` wins. With ``prefer``
    None the earliest such index wins, which is the pre-U-32 behaviour for
    a converged tail: splicing to the front of it keeps the most chunk left
    to play.
    """
    tie_eps = NEAREST_TIE_EPS_RAD if tie_eps is None else tie_eps
    horizon = len(actions)
    hi = horizon - 1 if hi is None else int(hi)
    hi = max(0, min(hi, horizon - 1))
    lo = max(0, min(int(lo), hi))
    arms = _arm_targets(actions[lo : hi + 1])  # (n, 14)
    dist = np.linalg.norm(arms - arms_now[None, :], axis=1)
    candidates = [(float(dist[i]), float(lo + i)) for i in range(len(arms))]
    # Refine along the two segments adjoining the best vertex: project the
    # current pose onto each and keep the feet as candidates too.
    best_i = int(np.argmin(dist))
    for a in (best_i - 1, best_i):
        if a < 0 or a + 1 >= len(arms):
            continue
        seg = arms[a + 1] - arms[a]
        denom = float(seg @ seg)
        if denom <= 0.0:
            continue
        frac = float(np.clip((arms_now - arms[a]) @ seg / denom, 0.0, 1.0))
        foot = arms[a] + frac * seg
        candidates.append((float(np.linalg.norm(arms_now - foot)), float(lo + a + frac)))
    best_dist = min(d for d, _ in candidates)
    near = [idx for d, idx in candidates if d <= best_dist + tie_eps]
    if prefer is None:
        return min(near)
    return min(near, key=lambda idx: (abs(idx - prefer), idx))


@dataclass
class _Chunk:
    t0: float
    actions: np.ndarray  # (H, 20)
    dt: float
    # Splice offset in chunk steps (U-31): PLAYBACK starts `offset` steps
    # away from the wall-clock index and then advances at real time. Always
    # 0.0 under `chunk_splice="index"`, negative when `nearest` rewinds to a
    # row closer to the pose the arm actually holds.
    offset: float = 0.0
    steps_consumed_at: float = field(init=False)

    def __post_init__(self):
        self.actions = np.asarray(self.actions, dtype=np.float32)
        if self.actions.ndim != 2 or self.actions.shape[1] != C.ACTION_DIM:
            raise ValueError(f"chunk must be (H, {C.ACTION_DIM}), got {self.actions.shape}")

    def wall_index(self, t_sim: float) -> float:
        """Steps of sim time since `t0` — the age of the OBSERVATION.

        Splice-invariant on purpose: `t0` is when the policy looked, and
        rewinding playback does not make that observation any fresher, so
        the replan trigger is judged on this and never on `index`.
        """
        return (t_sim - self.t0) / self.dt

    def index(self, t_sim: float) -> float:
        """Where playback is — the wall index shifted by the splice."""
        return self.wall_index(t_sim) + self.offset

    def interpolate(self, t_sim: float) -> np.ndarray:
        idx = self.index(t_sim)
        last = len(self.actions) - 1
        if idx <= 0:
            return self.actions[0]
        if idx >= last:
            return self.actions[last]
        lo = int(idx)
        frac = idx - lo
        return (1.0 - frac) * self.actions[lo] + frac * self.actions[lo + 1]


class ChunkExecutor:
    def __init__(
        self,
        max_delta_per_tick: float = 0.05,  # rad per command tick, per joint
        replan_after_steps: int = 8,  # request new chunk after consuming this many
        quantizer: BaseQuantizer | None = None,
        chunk_splice: str = "index",
        splice_ramp_ticks: int = 8,
        leash_rad: float | None = None,
    ):
        if chunk_splice not in SPLICE_POLICIES:
            raise ValueError(f"chunk_splice must be one of {SPLICE_POLICIES}, got {chunk_splice!r}")
        if leash_rad is not None and leash_rad <= 0.0:
            raise ValueError(f"leash_rad must be positive or None (off), got {leash_rad!r}")
        self.max_delta_per_tick = max_delta_per_tick
        self.replan_after_steps = replan_after_steps
        self.quantizer = quantizer or BaseQuantizer()
        self.chunk_splice = chunk_splice
        self.splice_ramp_ticks = splice_ramp_ticks
        # U-32: how far the arm COMMAND may stand from the MEASURED pose,
        # per joint, after the per-tick clamp. None = off (sim and every
        # synchronous run, which stay byte-identical).
        self.leash_rad = leash_rad
        self.stats = ExecutorStats()
        self._chunk: _Chunk | None = None
        self._last_arms: np.ndarray | None = None  # (14,) last commanded arm targets
        # (14,) measured arm joints from the most recent `step` — the leash's
        # reference and, once seen, `nearest`'s splice anchor.
        self._measured_arms: np.ndarray | None = None
        self._last_command: Command | None = None  # what hold() repeats
        # `offset` mode's decaying correction, and the per-arrival numbers
        # the runner prints on its `policy chunk #N` line. None = not
        # measured on this arrival (nothing commanded yet, or no `t_sim_now`).
        self._splice_delta: np.ndarray | None = None
        self._splice_ticks_left: int = 0
        self.last_splice_shift: float | None = None
        self.last_arrival_jump_rad: float | None = None
        self.last_cmd_lead_rad: float | None = None

    @property
    def last_commanded_arms(self) -> np.ndarray | None:
        """(14,) last commanded arm targets, or None before any `step`.

        `set_chunk` never touches `_last_arms` -- reading it right before
        or right after a `set_chunk` call is the same value, "what was
        commanded right before this chunk arrived" (a diagnostic like
        `--chunk-dump` wants exactly that, without reaching into the
        private attribute directly).
        """
        return None if self._last_arms is None else self._last_arms.copy()

    def reset(self) -> None:
        self._chunk = None
        self._last_arms = None
        self._measured_arms = None
        self._last_command = None
        self._splice_delta = None
        self._splice_ticks_left = 0
        self.last_splice_shift = None
        self.last_arrival_jump_rad = None
        self.last_cmd_lead_rad = None
        self.quantizer.reset()
        self.stats = ExecutorStats()

    def set_chunk(
        self,
        t0_sim: float,
        actions: np.ndarray,
        dt: float = 1.0 / DATASET_FPS,
        t_sim_now: float | None = None,
    ):
        """Latest chunk wins; timestamps are sim time.

        ``t_sim_now`` is the sim time of the tick this chunk is being
        installed on — under async inference that is LATER than ``t0_sim``
        by the round trip. It is optional so every synchronous caller keeps
        working unchanged; without it there is no wall index to splice
        against and no jump to measure, so the chunk is installed exactly
        as it always was.

        Splice policies (docs/realdata/16 U-31), selected by
        ``chunk_splice``. All three leave grippers and the base twist
        alone: no correction term is ever added to them (a base twist is a
        VELOCITY — offsetting it is continued motion nobody asked for) and
        they take no part in choosing an index.

        * ``index`` — today's behaviour and the default: play the chunk at
          ``(t_sim - t0)/dt``. Byte-identical to the pre-U-31 executor.
        * ``nearest`` — rewind to the point the arm can actually pick up
          from. Search for the index whose arm target is closest (L2 over
          the 14 arm dims — what `step` clamps) to the MEASURED arm pose
          from the most recent `step`, and store ``chosen - wall_idx`` as a
          per-chunk index offset so playback still advances at real time
          from there. The index is FRACTIONAL (see `_nearest_index`) and
          never past ``horizon - 1``; `exhausted` follows the spliced
          index, so a chunk shorter than the offset still runs out exactly
          as before. The search runs over
          ``[max(0, wall_idx - horizon), min(horizon - 1, wall_idx + 2)]``:
          the chunk's row 0 IS the pose the policy observed, so the row
          matching the measured pose lies at or just before the wall index
          and anything far beyond it is a near-tie artefact that only costs
          horizon (U-32; on the rig it cost 413 held ticks of 587). Until a
          `step` has supplied a measured pose there is nothing to anchor
          to, and the search falls back to the last COMMANDED target over
          the full ``[lo, horizon - 1]`` — the pre-U-32 behaviour.
        * ``offset`` — keep the wall-clock index and instead carry the
          discontinuity as an additive arm correction,
          ``current_cmd - chunk[wall_idx]``, decaying linearly to zero over
          ``splice_ramp_ticks`` COMMAND ticks (not chunk steps).
        """
        chunk = _Chunk(t0_sim, actions, dt)
        self._splice_delta = None
        self._splice_ticks_left = 0
        self.last_splice_shift = None
        self.last_arrival_jump_rad = None

        measurable = t_sim_now is not None and self._last_arms is not None
        if measurable:
            wall_idx = chunk.wall_index(t_sim_now)
            horizon = len(chunk.actions)
            if self.chunk_splice == "nearest":
                lo = int(max(0, np.floor(wall_idx) - horizon))
                lo = min(lo, horizon - 1)
                if self._measured_arms is not None:
                    reference = self._measured_arms
                    hi = int(np.floor(wall_idx)) + NEAREST_FORWARD_ROWS
                    prefer = float(wall_idx)
                else:  # nothing measured yet: the pre-U-32 fallback
                    reference, hi, prefer = self._last_arms, None, None
                chosen = _nearest_index(chunk.actions, reference, lo, hi=hi, prefer=prefer)
                chunk.offset = chosen - wall_idx
            elif self.chunk_splice == "offset" and self.splice_ramp_ticks > 0:
                at_wall = _arm_targets(chunk.interpolate(t_sim_now))
                self._splice_delta = self._last_arms - at_wall
                self._splice_ticks_left = self.splice_ramp_ticks

        self._chunk = chunk
        self.stats.chunks += 1

        if measurable:
            # The target the NEXT tick will aim at, after splicing and
            # before the clamp: the discontinuity the arm is actually
            # asked to swallow. Measured for every policy so `index`'s
            # jump and `nearest`'s stay directly comparable.
            first = _arm_targets(chunk.interpolate(t_sim_now))
            if self._splice_delta is not None:
                first = first + self._splice_delta  # full weight on the arrival tick
            jump = float(np.max(np.abs(first - self._last_arms)))
            shift = float(chunk.offset)
            self.last_arrival_jump_rad = jump
            self.last_splice_shift = shift
            self.stats.spliced_arrivals += 1
            self.stats.arrival_jump_rad_sum += jump
            self.stats.arrival_jump_rad_max = max(self.stats.arrival_jump_rad_max, jump)
            self.stats.splice_shift_sum += shift
            if abs(shift) > abs(self.stats.splice_shift_max):
                self.stats.splice_shift_max = shift

    def needs_replan(self, t_sim: float) -> bool:
        # The WALL index, never the spliced one: this asks how stale the
        # observation behind the chunk is, and a rewind does not refresh
        # it. Judging it on playback position would push the request out
        # by the splice on every arrival and walk the replacement index
        # past the horizon into starvation.
        if self._chunk is None:
            return True
        return self._chunk.wall_index(t_sim) >= self.replan_after_steps

    def exhausted(self, t_sim: float) -> bool:
        """True when the chunk has no unplayed step left at ``t_sim``.

        `_Chunk.interpolate` clamps past the end, so `step` would keep
        driving toward the LAST row forever. That reads as "hold" but is
        really "keep ramping toward a target the policy stopped vouching
        for at t0 + (H-1)*dt". Asynchronous inference asks this question
        explicitly (see `hold`) so a late reply is held rather than papered
        over; the synchronous path never calls it and is unchanged.
        """
        if self._chunk is None:
            return True
        return self._chunk.index(t_sim) >= len(self._chunk.actions) - 1

    def hold(self, t_sim: float) -> Command | None:
        """Repeat the last commanded target — the starvation command.

        Never extrapolates. The arm targets are exactly the ones already
        commanded (a POSITION, so repeating it is standing still) and the
        base twist is zeroed (a VELOCITY, so repeating it would be
        continued motion nobody asked for). Counts a tick, because the
        command really did leave, and a `starved_ticks` beside it.

        ``None`` before anything has ever been commanded: there is no
        target to hold and nothing to starve, which is the same silence
        `step` returns with no chunk.
        """
        del t_sim  # signature parity with step(); the held target has no time
        if self._last_command is None:
            return None
        self.stats.ticks += 1
        self.stats.starved_ticks += 1
        return replace(
            self._last_command,
            base_twist=(0.0, 0.0, 0.0),
            base_token=self.quantizer.quantize(0.0, 0.0, 0.0),
        )

    def step(self, t_sim: float, state: np.ndarray) -> Command | None:
        """One command tick; None when there is nothing (safe) to command."""
        if self._chunk is None:
            return None
        if t_sim < self._chunk.t0 - 1.0:  # clock rebased (scene reset): chunk is stale
            self._chunk = None
            self.stats.stale_chunks += 1
            return None

        target = self._chunk.interpolate(t_sim)
        arms_target = np.concatenate([target[C.A_LEFT_ARM], target[C.A_RIGHT_ARM]])

        if self._splice_delta is not None:
            # `offset` splice: carry the hand-over discontinuity as an
            # additive ARM correction and bleed it off linearly. Full
            # weight on the first tick (so the hand-over is continuous),
            # zero after `splice_ramp_ticks`, then dropped entirely.
            if self._splice_ticks_left > 0:
                frac = self._splice_ticks_left / float(self.splice_ramp_ticks)
                arms_target = arms_target + self._splice_delta * frac
                self._splice_ticks_left -= 1
            else:
                self._splice_delta = None

        # The MEASURED arm pose, read every tick (U-32): the leash's
        # reference below, and `nearest`'s splice anchor in `set_chunk`. On
        # the real rig `state` carries the joints the controller reports; in
        # sim, the sim's own. A non-finite sample is simply not adopted —
        # the last good one stands.
        measured = np.concatenate([state[C.S_LEFT_ARM], state[C.S_RIGHT_ARM]]).astype(np.float64)
        if np.all(np.isfinite(measured)):
            self._measured_arms = measured
        if self._last_arms is None:
            if self._measured_arms is None:
                return None  # no joint state yet; do not guess a start pose
            self._last_arms = self._measured_arms.copy()

        prev_arms = self._last_arms
        delta = arms_target - self._last_arms
        max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
        self.stats.max_requested_delta = max(self.stats.max_requested_delta, max_abs)
        if max_abs > self.max_delta_per_tick:
            delta = np.clip(delta, -self.max_delta_per_tick, self.max_delta_per_tick)
            self.stats.clamped_ticks += 1
        self._last_arms = self._last_arms + delta

        if self.leash_rad is not None and self._measured_arms is not None:
            # U-32, AFTER the per-tick clamp: the clamp bounds how fast the
            # command may MOVE, not how far it may stand from the pose the
            # arm actually holds. Behind the companion's 0.5 rad/s slew cap
            # (half this clamp) the command drifts ahead, and then every
            # chunk hand-over is judged against a target no joint occupies.
            # ARMS ONLY — never the grippers (an open fraction, not a joint
            # this tracks) and never the base (a velocity).
            leashed = np.clip(
                self._last_arms,
                self._measured_arms - self.leash_rad,
                self._measured_arms + self.leash_rad,
            )
            if not np.array_equal(leashed, self._last_arms):
                self.stats.leash_active_ticks += 1
            # The per-tick clamp stays the ONLY rate limiter: the leash's
            # centre is the raw measured sample, which is not itself
            # rate-limited (a stale-then-fresh joint state under the U-27
            # age gate can move it by 0.1 rad between ticks), so the
            # leashed command is re-bounded to the same +-max_delta step.
            self._last_arms = np.clip(
                leashed,
                prev_arms - self.max_delta_per_tick,
                prev_arms + self.max_delta_per_tick,
            )
        if self._measured_arms is not None:
            # Recorded whether the leash is on or off: with it off this is
            # the diagnostic that says how far the command had run ahead.
            lead = float(np.max(np.abs(self._last_arms - self._measured_arms)))
            self.last_cmd_lead_rad = lead
            self.stats.cmd_lead_rad_max = max(self.stats.cmd_lead_rad_max, lead)
        self.stats.ticks += 1

        vx, vy, wz = (float(v) for v in target[C.A_BASE])
        token = self.quantizer.quantize(vx, vy, wz)
        self._last_command = Command(
            left_arm=self._last_arms[:7].astype(np.float32),
            right_arm=self._last_arms[7:].astype(np.float32),
            left_gripper=float(np.clip(target[C.A_LEFT_GRIP], 0.0, 1.0)),
            right_gripper=float(np.clip(target[C.A_RIGHT_GRIP], 0.0, 1.0)),
            base_twist=(vx, vy, wz),
            base_token=token,
        )
        return self._last_command
