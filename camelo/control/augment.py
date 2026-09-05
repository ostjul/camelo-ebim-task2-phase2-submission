"""Synthesize augmented heuristic references from recorded Task 2 demos.

The 200-episode fixpos corpus has one converged trajectory shape and zero
start/approach diversity (GRASP_FAILURE_RESEARCH.md §2.9: "zero pad
variation in all 199 episodes"), which is the documented ceiling on what a
policy can learn from it. This module widens the corpus without touching
the sim: it edits a demo's **world TCP path** — perturbed starts that merge
back, approach arcs, mid-flight disturbances that recover, alternative
transfer paths, time warps, cross-episode splices — re-solves the right arm
offline with the same IK the heuristic adapter runs online, and writes
variants in exactly the `epNNN_{actions.npy,gt_traj.npz,meta.json}` layout
`HeuristicAdapter` already consumes. Collection is then the shipped
heuristic policy playing each variant in the sim while the benchmark
recorder taps the wire as usual.

Every variant passes `validate()` before it is allowed to exist on disk:
grasp and place keyframes must hold the demo's pose (the fingers must still
straddle the pad, the drop must still land on the target), per-frame joint
deltas must stay at demo level (the 0.05 rad/tick executor clamp is the
only rate limiter in the command path, F-53), and the TCP must never dive
below the demo's own floor. Editing the path between those anchors is the
point; editing the anchors is a failed variant, not a looser one.

Numpy-only on purpose (`camelo/control/` layering): generation and
validation run with no ROS, no sim, no GPU, so the whole family is
unit-testable offline and a variant is known-feasible before it costs a
single Spark episode.

Chosen thresholds say so; measured ones cite their source (house rule).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from camelo import contracts as C
from camelo.control.kinematics import RightArmIK, quat_xyzw_angle_deg

# --- validation budgets ----------------------------------------------------
# MEASURED from ep089 of hermanprawiro/task2_fixpos_200 (the shipped
# heuristic's own reference): GT per-frame joint delta max 0.131 rad, the
# recorded action stream's max 0.159 rad. The executor demonstrably tracks
# that, so demo level is the budget — a variant that needs more per frame
# than the demos ever did would saturate the 0.05 rad/tick clamp (F-53).
MAX_JOINT_DELTA_RAD = 0.16
# CHOSEN: IK must land on the intended path to within 2 mm everywhere.
# The online adapter re-solves anyway; this bounds what "the path we
# validated" can differ from "the path the adapter will target".
MAX_TARGET_ERR_M = 0.002
# CHOSEN: how far the variant may sit from the source pose while the
# gripper actually closes / opens. The demos themselves grasp the fixed pad
# at 27.9-36.8 mm TCP-to-pad (F-98) — a ±4.5 mm natural spread — and the
# gate envelope is 45 mm / 7°, so 8 mm + 5° keeps a deliberate grasp jitter
# inside what a demonstrator did while staying far from the envelope edge.
GRASP_HOLD_TOL_M = 0.008
PLACE_HOLD_TOL_M = 0.008
HOLD_MAX_ANGLE_DEG = 5.0
# Frames around close/open where the pose must hold (30 fps: close-5..+15
# covers the finger travel; open-10..+5 covers the drop).
GRASP_HOLD_PRE, GRASP_HOLD_POST = 5, 15
PLACE_HOLD_PRE, PLACE_HOLD_POST = 10, 5
# CHOSEN: never dive more than 5 mm below the source's own lowest TCP —
# the demo already clears the table/holder by construction.
Z_FLOOR_MARGIN_M = 0.005
# Threshold separating "open" from "closed" on the canonical gripper
# channel — same midpoint split grasp_gate.CLOSED_OPEN_FRACTION uses.
_GRIP_SPLIT = 0.5
# Frames of safety margin keeping disturbances away from grasp/place.
EVENT_MARGIN_FRAMES = 20

GENERATOR_VERSION = 1

FAMILIES = (
    "start_offset",
    "approach",
    "disturbance",
    "transfer",
    "timewarp",
    "splice",
)


# ---------------------------------------------------------------------------
# Reference container and IO
# ---------------------------------------------------------------------------


@dataclass
class Reference:
    """One heuristic reference: the (T, 20) action stream plus its GT.

    ``tcp_xyz`` is the world right-TCP path the adapter's residual IK
    targets; ``arm_q`` seeds that IK. All arrays share T.
    """

    actions: np.ndarray  # (T, 20) float32
    t: np.ndarray  # (T,) seconds
    base_xy_yaw: np.ndarray  # (T, 3)
    spine: np.ndarray  # (T,)
    arm_q: np.ndarray  # (T, 7)
    tcp_xyz: np.ndarray  # (T, 3)
    tcp_quat_xyzw: np.ndarray  # (T, 4)
    fps: float
    episode: int
    source: str = ""

    def __post_init__(self):
        t_len = self.actions.shape[0]
        if self.actions.ndim != 2 or self.actions.shape[1] != C.ACTION_DIM:
            raise ValueError(f"actions must be (T, {C.ACTION_DIM}), got {self.actions.shape}")
        shapes = {
            "t": (self.t, (t_len,)),
            "base_xy_yaw": (self.base_xy_yaw, (t_len, 3)),
            "spine": (self.spine, (t_len,)),
            "arm_q": (self.arm_q, (t_len, 7)),
            "tcp_xyz": (self.tcp_xyz, (t_len, 3)),
            "tcp_quat_xyzw": (self.tcp_quat_xyzw, (t_len, 4)),
        }
        for name, (arr, want) in shapes.items():
            if arr.shape != want:
                raise ValueError(f"{name} must be {want}, got {arr.shape}")

    @property
    def frames(self) -> int:
        return int(self.actions.shape[0])

    def copy(self) -> Reference:
        return Reference(
            actions=self.actions.copy(),
            t=self.t.copy(),
            base_xy_yaw=self.base_xy_yaw.copy(),
            spine=self.spine.copy(),
            arm_q=self.arm_q.copy(),
            tcp_xyz=self.tcp_xyz.copy(),
            tcp_quat_xyzw=self.tcp_quat_xyzw.copy(),
            fps=self.fps,
            episode=self.episode,
            source=self.source,
        )


def load_reference(actions_path: str | Path, gt_path: str | Path | None = None) -> Reference:
    """Load a prepared episode (see scripts/prepare_heuristic_actions.py)."""
    actions_path = Path(actions_path)
    if gt_path is None:
        gt_path = actions_path.with_name(
            actions_path.name.replace("_actions.npy", "_gt_traj.npz")
        )
    gt_path = Path(gt_path)
    actions = np.asarray(np.load(actions_path), dtype=np.float32)
    with np.load(gt_path) as data:
        ref = Reference(
            actions=actions,
            t=np.asarray(data["t"], dtype=np.float64),
            base_xy_yaw=np.asarray(data["base_xy_yaw"], dtype=np.float64),
            spine=np.asarray(data["spine"], dtype=np.float64),
            arm_q=np.asarray(data["arm_q"], dtype=np.float64),
            tcp_xyz=np.asarray(data["tcp_xyz"], dtype=np.float64),
            tcp_quat_xyzw=np.asarray(data["tcp_quat_xyzw"], dtype=np.float64),
            fps=float(data["fps"]),
            episode=int(data["episode"]),
            source=actions_path.stem.replace("_actions", ""),
        )
    return ref


def write_variant(out_dir: str | Path, name: str, ref: Reference, meta: dict) -> dict:
    """Write ``<name>_{actions.npy,gt_traj.npz,meta.json}``; returns the paths.

    The npz carries exactly the keys `HeuristicAdapter._load_gt` reads plus
    the bookkeeping keys `gt_traj.GT_TRAJ_KEYS` requires, so a variant is
    loadable by the unmodified adapter via ``heuristic:<actions_path>``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    actions_path = out_dir / f"{name}_actions.npy"
    gt_path = out_dir / f"{name}_gt_traj.npz"
    meta_path = out_dir / f"{name}_meta.json"
    np.save(actions_path, np.asarray(ref.actions, dtype=np.float32))
    np.savez(
        gt_path,
        t=ref.t,
        base_xy_yaw=ref.base_xy_yaw,
        spine=ref.spine,
        arm_q=ref.arm_q,
        tcp_xyz=ref.tcp_xyz,
        tcp_quat_xyzw=ref.tcp_quat_xyzw,
        fps=np.float64(ref.fps),
        episode=np.int32(ref.episode),
    )
    payload = {
        "generator_version": GENERATOR_VERSION,
        "fps": ref.fps,
        "frames": ref.frames,
        "episode": ref.episode,
        "actions_path": str(actions_path),
        "gt_traj_path": str(gt_path),
        **meta,
    }
    meta_path.write_text(json.dumps(payload, indent=2))
    return {"actions": str(actions_path), "gt": str(gt_path), "meta": str(meta_path)}


# ---------------------------------------------------------------------------
# Keyframes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Keyframes:
    """Grasp/place anchors of one reference, in frame indices.

    ``close``/``open`` are the gripper edges; ``lift`` is the first frame
    the TCP has risen 20 mm off the grasp; ``descend`` is the last carry
    frame before the final 20 mm drop onto the target.
    """

    close: int
    open: int
    lift: int
    descend: int


def _grip_edges(grip: np.ndarray) -> tuple[list[int], list[int]]:
    closed = grip < _GRIP_SPLIT
    step = np.diff(closed.astype(np.int8))
    closes = [int(i) + 1 for i in np.where(step == 1)[0]]
    opens = [int(i) + 1 for i in np.where(step == -1)[0]]
    return closes, opens


def find_keyframes(ref: Reference, lift_eps_m: float = 0.02) -> Keyframes:
    """Anchors from the gripper channel + TCP height. Raises on a channel
    that does not close exactly once and reopen exactly once, in order —
    that reference is not a Task 2 pick-and-place and must not be augmented.
    """
    closes, opens = _grip_edges(np.asarray(ref.actions[:, C.A_RIGHT_GRIP], dtype=np.float64))
    if len(closes) != 1 or len(opens) != 1 or not closes[0] < opens[0]:
        raise ValueError(
            f"right gripper channel must close once then open once, got "
            f"closes={closes} opens={opens}"
        )
    close, open_ = closes[0], opens[0]
    z = ref.tcp_xyz[:, 2]
    risen = np.where(z[close:open_] - z[close] >= lift_eps_m)[0]
    if risen.size == 0:
        raise ValueError("TCP never lifts 20 mm after the close — not a pick")
    lift = close + int(risen[0])
    high = np.where(z[lift:open_] - z[open_] >= lift_eps_m)[0]
    descend = (lift + int(high[-1])) if high.size else lift
    return Keyframes(close=close, open=open_, lift=lift, descend=descend)


# ---------------------------------------------------------------------------
# Smooth profiles (C1 in the offset; per-frame deltas are what the budget
# actually bounds, and validate() measures those directly)
# ---------------------------------------------------------------------------


def _minjerk(s: np.ndarray) -> np.ndarray:
    s = np.clip(s, 0.0, 1.0)
    return s**3 * (10.0 + s * (-15.0 + 6.0 * s))


def decay_profile(t_len: int, merge: int) -> np.ndarray:
    """1 at frame 0, min-jerk down to 0 at ``merge``, 0 after."""
    w = np.zeros(t_len, dtype=np.float64)
    merge = max(1, min(merge, t_len))
    w[:merge] = 1.0 - _minjerk(np.arange(merge, dtype=np.float64) / merge)
    return w


def bump_profile(t_len: int, i0: int, i1: int) -> np.ndarray:
    """0 outside [i0, i1], sin² inside — zero-valued and zero-sloped ends."""
    w = np.zeros(t_len, dtype=np.float64)
    i0, i1 = max(0, i0), min(t_len, i1)
    n = i1 - i0
    if n > 1:
        w[i0:i1] = np.sin(np.pi * np.arange(n, dtype=np.float64) / (n - 1)) ** 2
    return w


def plateau_profile(t_len: int, i0: int, i1: int, ramp: int) -> np.ndarray:
    """0 → 1 over [i0, i0+ramp], hold 1, 1 → 0 over [i1-ramp, i1]."""
    w = np.zeros(t_len, dtype=np.float64)
    i0, i1 = max(0, i0), min(t_len, i1)
    if i1 <= i0:
        return w
    ramp = max(1, min(ramp, (i1 - i0) // 2))
    up = _minjerk(np.arange(ramp, dtype=np.float64) / ramp)
    w[i0 : i0 + ramp] = up
    w[i0 + ramp : i1 - ramp] = 1.0
    w[i1 - ramp : i1] = up[::-1]
    return w


def _unit(rng: np.random.Generator, z_damp: float = 0.5) -> np.ndarray:
    """Random unit direction with the vertical component damped: the arm
    works above a table, so lateral diversity is worth more than vertical
    and large +z excursions walk out of the wrist's comfortable reach."""
    v = rng.normal(size=3)
    v[2] *= z_damp
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0])
    return v / n


def _perp_unit(
    rng: np.random.Generator, tangent: np.ndarray, z_damp: float = 0.35
) -> np.ndarray:
    """Random unit vector perpendicular to ``tangent``, vertical damped.

    The damping matters here too: near-vertical approach arcs pitch the
    wrist and were the reproducible way to force the IK across a
    configuration flip (attitude 70°+ off at the grasp — rejected, but a
    wasted draw).
    """
    t_norm = float(np.linalg.norm(tangent))
    if t_norm < 1e-9:
        return _unit(rng)
    t_hat = tangent / t_norm
    for _ in range(8):
        v = rng.normal(size=3)
        v[2] *= z_damp
        v -= (v @ t_hat) * t_hat
        n = float(np.linalg.norm(v))
        if n > 1e-6:
            return v / n
    return _unit(rng)


# ---------------------------------------------------------------------------
# Families: each returns (target Reference, params). ``tcp_xyz`` is the
# INTENDED path; resolve() then earns the arm joints (and the achieved
# tcp/quat) with the same IK the adapter runs online.
# ---------------------------------------------------------------------------


def aug_start_offset(
    ref: Reference, kf: Keyframes, rng: np.random.Generator, scale: float = 1.0
) -> tuple[Reference, dict]:
    """Family 1: begin the episode off the demo path, merge back before the
    grasp — the recovery-from-OOD-start data the corpus has none of."""
    mag = float(rng.uniform(0.02, 0.08)) * scale
    direction = _unit(rng)
    merge = int(rng.uniform(0.35, 0.60) * kf.close)
    merge = min(merge, kf.close - EVENT_MARGIN_FRAMES)
    out = ref.copy()
    out.tcp_xyz = ref.tcp_xyz + np.outer(decay_profile(ref.frames, merge), mag * direction)
    return out, {
        "offset_m": mag,
        "direction": direction.tolist(),
        "merge_frame": merge,
    }


def aug_approach(
    ref: Reference, kf: Keyframes, rng: np.random.Generator, scale: float = 1.0
) -> tuple[Reference, dict]:
    """Family 2: a different road to the same grasp — arc the pre-grasp
    path sideways and optionally shift the grasp itself a few mm (the demos
    themselves spread ±4.5 mm, F-98)."""
    i0 = int(0.30 * kf.close)
    arc_mag = float(rng.uniform(0.02, 0.06)) * scale
    mid = (i0 + kf.close) // 2
    a, b = max(0, mid - 2), min(ref.frames - 1, mid + 2)
    tangent = ref.tcp_xyz[b] - ref.tcp_xyz[a]
    arc_dir = _perp_unit(rng, tangent)
    jit_mag = float(rng.uniform(0.0, 0.004)) * scale
    jit_dir = _unit(rng, z_damp=0.3)
    out = ref.copy()
    arc = np.outer(bump_profile(ref.frames, i0, kf.close - EVENT_MARGIN_FRAMES), arc_mag * arc_dir)
    # Grasp jitter ramps in over the approach, holds through the fingers'
    # travel, and is gone by the middle of the carry so the place is
    # untouched.
    hold_end = kf.lift + EVENT_MARGIN_FRAMES
    release_end = min((hold_end + kf.descend) // 2, kf.descend - EVENT_MARGIN_FRAMES)
    ramp_in = _minjerk(
        (np.arange(ref.frames, dtype=np.float64) - i0) / max(1, kf.close - i0)
    )
    ramp_in[hold_end:] = 1.0
    ramp_out = 1.0 - _minjerk(
        (np.arange(ref.frames, dtype=np.float64) - hold_end) / max(1, release_end - hold_end)
    )
    jitter = np.outer(np.minimum(ramp_in, ramp_out), jit_mag * jit_dir)
    out.tcp_xyz = ref.tcp_xyz + arc + jitter
    return out, {
        "arc_m": arc_mag,
        "arc_direction": arc_dir.tolist(),
        "arc_window": [i0, kf.close - EVENT_MARGIN_FRAMES],
        "grasp_jitter_m": jit_mag,
        "grasp_jitter_direction": jit_dir.tolist(),
    }


def aug_disturbance(
    ref: Reference, kf: Keyframes, rng: np.random.Generator, scale: float = 1.0
) -> tuple[Reference, dict]:
    """Family 3: a smooth push off the path and back — the recovery signal
    a perfect demo never contains. Placed clear of the grasp and the drop;
    gentler while the pad is in hand."""
    # Sample the desired length first, then clamp to what the reference can
    # host — a short source gets a shorter disturbance, not a ValueError.
    length = int(rng.uniform(0.8, 2.0) * ref.fps)
    spans = {
        "pre_grasp": (10, kf.close - EVENT_MARGIN_FRAMES),
        "carry": (kf.lift + EVENT_MARGIN_FRAMES, kf.descend - EVENT_MARGIN_FRAMES),
    }
    min_len = int(0.5 * ref.fps)
    zones = {name: (lo, hi) for name, (lo, hi) in spans.items() if hi - lo > min_len}
    if not zones:
        raise ValueError("no window long enough for a disturbance in this reference")
    length = min(length, max(hi - lo for lo, hi in zones.values()) - 1)
    fitting = [(name, lo, hi) for name, (lo, hi) in zones.items() if hi - lo > length]
    zone_name, lo, hi = fitting[int(rng.integers(len(fitting)))]
    i0 = int(rng.integers(lo, hi - length))
    amp_range = (0.02, 0.06) if zone_name == "pre_grasp" else (0.015, 0.04)
    amp = float(rng.uniform(*amp_range)) * scale
    direction = _unit(rng)
    out = ref.copy()
    out.tcp_xyz = ref.tcp_xyz + np.outer(
        bump_profile(ref.frames, i0, i0 + length), amp * direction
    )
    return out, {
        "zone": zone_name,
        "window": [i0, i0 + length],
        "amplitude_m": amp,
        "direction": direction.tolist(),
    }


def aug_transfer(
    ref: Reference, kf: Keyframes, rng: np.random.Generator, scale: float = 1.0
) -> tuple[Reference, dict]:
    """Family 4: a different road from the gripped pad to the target —
    carry height plateau plus a lateral arc, endpoints pinned to the demo's
    lift and final descent. Height goes UP only: the pad hangs below the
    TCP, the scene between grasp and target cannot be collision-checked
    offline, and validate() holds every carry to the source's own floor —
    a lower carry is a variant that cannot pass, so it is never drawn."""
    i0 = kf.lift + EVENT_MARGIN_FRAMES // 2
    i1 = kf.descend - EVENT_MARGIN_FRAMES // 2
    if i1 - i0 < int(1.0 * ref.fps):
        raise ValueError("carry segment too short to reshape")
    height = float(rng.uniform(0.0, 0.06)) * scale
    lateral = float(rng.uniform(0.02, 0.08)) * scale
    azimuth = float(rng.uniform(0.0, 2.0 * math.pi))
    lat_dir = np.array([math.cos(azimuth), math.sin(azimuth), 0.0])
    ramp = int(0.5 * ref.fps)
    out = ref.copy()
    shape = plateau_profile(ref.frames, i0, i1, ramp)
    offset = np.outer(shape, np.array([0.0, 0.0, height]))
    offset += np.outer(bump_profile(ref.frames, i0, i1), lateral * lat_dir)
    out.tcp_xyz = ref.tcp_xyz + offset
    return out, {
        "carry_window": [i0, i1],
        "height_delta_m": height,
        "lateral_m": lateral,
        "lateral_azimuth_rad": azimuth,
    }


def _resample(ref: Reference, src_idx: np.ndarray) -> Reference:
    """Rebuild every channel at fractional source indices ``src_idx``.

    Continuous channels interpolate linearly; the two gripper channels take
    the PREVIOUS sample, because interpolating a binary edge invents
    half-open frames no demonstrator produced and shifts the close the
    validator anchors on.
    """
    n = src_idx.shape[0]
    t_src = np.arange(ref.frames, dtype=np.float64)

    def lerp(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 1:
            return np.interp(src_idx, t_src, arr)
        return np.stack([np.interp(src_idx, t_src, arr[:, j]) for j in range(arr.shape[1])], 1)

    prev = np.clip(np.floor(src_idx).astype(int), 0, ref.frames - 1)
    actions = lerp(ref.actions.astype(np.float64)).astype(np.float32)
    for ch in (C.A_LEFT_GRIP, C.A_RIGHT_GRIP):
        actions[:, ch] = ref.actions[prev, ch]
    quat = lerp(ref.tcp_quat_xyzw)
    norms = np.linalg.norm(quat, axis=1, keepdims=True)
    quat = quat / np.clip(norms, 1e-9, None)
    return Reference(
        actions=actions,
        t=np.arange(n, dtype=np.float64) / ref.fps,
        base_xy_yaw=lerp(ref.base_xy_yaw),
        spine=lerp(ref.spine),
        arm_q=lerp(ref.arm_q),
        tcp_xyz=lerp(ref.tcp_xyz),
        tcp_quat_xyzw=quat,
        fps=ref.fps,
        episode=ref.episode,
        source=ref.source,
    )


def aug_timewarp(
    ref: Reference, kf: Keyframes, rng: np.random.Generator, scale: float = 1.0
) -> tuple[Reference, dict]:
    """Family 5: same road, different pace — resampled onto the 30 fps grid.

    Only the approach, the carry, and the tail change pace. The windows
    around the close (through the lift) and around the final descent/open
    stay at 1.0×: the Robotiq's finger travel takes its own fixed sim time,
    so speeding the reference up THERE would lift the pad off the fingers
    before they grip — a physical failure, not a validation artifact. The
    per-frame joint-delta budget in validate() is what keeps the warped
    segments honest.
    """
    lo, hi = 1.0 - 0.25 * scale, 1.0 + 0.35 * scale
    speeds = [float(rng.uniform(lo, hi)) for _ in range(3)]
    m = EVENT_MARGIN_FRAMES
    speed = np.ones(ref.frames, dtype=np.float64)
    speed[: max(0, kf.close - m)] = speeds[0]
    speed[kf.lift + m : max(kf.lift + m, kf.descend - m)] = speeds[1]
    speed[min(ref.frames, kf.open + m) :] = speeds[2]
    # New playback time of each source frame; sampling that monotone map at
    # the integer 30 fps grid yields the fractional source indices.
    t_new = np.concatenate([[0.0], np.cumsum(1.0 / speed[:-1])])
    n_new = int(math.floor(t_new[-1])) + 1
    src_idx = np.interp(
        np.arange(n_new, dtype=np.float64), t_new, np.arange(ref.frames, dtype=np.float64)
    )
    out = _resample(ref, src_idx)
    return out, {"segment_speeds": speeds, "pinned_margin_frames": m}


def aug_splice(
    ref_a: Reference,
    kf_a: Keyframes,
    ref_b: Reference,
    kf_b: Keyframes,
    rng: np.random.Generator,
    scale: float = 1.0,
) -> tuple[Reference, dict]:
    """Family 6: approach+grasp from A, carry+place from B.

    The handoff sits ``EVENT_MARGIN_FRAMES`` after A's close — the grasp is
    entirely A's, the place entirely B's, both demonstrated poses — and the
    two timelines crossfade over the early carry, where the fixpos demos
    are within a few cm of each other anyway. ``scale`` is accepted for
    generator symmetry; the crossfade window is what varies.
    """
    del scale
    hold = EVENT_MARGIN_FRAMES
    fade = int(rng.uniform(0.5, 1.0) * ref_a.fps)
    ja, jb = kf_a.close + hold, kf_b.close + hold
    tail = ref_b.frames - (jb + fade)
    if ja + fade >= ref_a.frames or tail <= 0:
        raise ValueError("references too short to splice at close+hold")
    n = ja + fade + tail
    w = _minjerk(np.arange(fade, dtype=np.float64) / fade)

    def stitch(arr_a: np.ndarray, arr_b: np.ndarray) -> np.ndarray:
        head = arr_a[:ja]
        wa = w.reshape(-1, *([1] * (arr_a.ndim - 1)))
        blend = (1.0 - wa) * arr_a[ja : ja + fade] + wa * arr_b[jb : jb + fade]
        return np.concatenate([head, blend, arr_b[jb + fade :]], axis=0)

    actions = stitch(
        ref_a.actions.astype(np.float64), ref_b.actions.astype(np.float64)
    ).astype(np.float32)
    # Gripper channels are edges, not signals to blend: A owns the wire up
    # to the end of the fade, B after — both are "closed" throughout it.
    for ch in (C.A_LEFT_GRIP, C.A_RIGHT_GRIP):
        actions[: ja + fade, ch] = ref_a.actions[: ja + fade, ch]
        actions[ja + fade :, ch] = ref_b.actions[jb + fade :, ch]
    quat = stitch(ref_a.tcp_quat_xyzw, ref_b.tcp_quat_xyzw)
    quat /= np.clip(np.linalg.norm(quat, axis=1, keepdims=True), 1e-9, None)
    out = Reference(
        actions=actions,
        t=np.arange(n, dtype=np.float64) / ref_a.fps,
        base_xy_yaw=stitch(ref_a.base_xy_yaw, ref_b.base_xy_yaw),
        spine=stitch(ref_a.spine, ref_b.spine),
        arm_q=stitch(ref_a.arm_q, ref_b.arm_q),
        tcp_xyz=stitch(ref_a.tcp_xyz, ref_b.tcp_xyz),
        tcp_quat_xyzw=quat,
        fps=ref_a.fps,
        episode=ref_a.episode,
        source=f"{ref_a.source}+{ref_b.source}",
    )
    return out, {
        "source_a": ref_a.source,
        "source_b": ref_b.source,
        "handoff_frame_a": ja,
        "handoff_frame_b": jb,
        "fade_frames": fade,
    }


# ---------------------------------------------------------------------------
# IK resolve + validation
# ---------------------------------------------------------------------------


@dataclass
class ResolveResult:
    ref: Reference
    ik_ok_fraction: float
    max_target_err_m: float


def resolve(target: Reference, ik: RightArmIK | None = None) -> ResolveResult:
    """Earn the right-arm joints for an edited TCP path.

    Each frame is seeded from the source's OWN arm pose at that frame —
    exactly how `HeuristicAdapter._correct_chunk` seeds online — not
    chained from the previous solution: position-only IK on a 7-DOF arm
    has a 4-dim null space, and chained solving measurably wanders through
    it (re-solving the unmodified ep089 path chained drifts up to 2.9 rad
    per joint and ~76° of TCP attitude while holding position to 0.5 mm).
    The demo seed pins every frame to the demonstrated configuration
    branch. The returned reference carries the ACHIEVED tcp/quat from FK
    of the solved joints, so what the validator judges is exactly what the
    adapter will target at run time — validating the intent while shipping
    a different path would be the fixture-built-from-the-doc trap
    (AGENTS.md).
    """
    # More iterations than the online default: offline milliseconds are
    # free, and a 6-8 cm offset from the demo seed can need them.
    solver = ik if ik is not None else RightArmIK(max_iters=120)
    t_len = target.frames
    q = np.empty((t_len, 7), dtype=np.float64)
    ok = np.empty(t_len, dtype=bool)
    jump_tol = MAX_JOINT_DELTA_RAD
    for i in range(t_len):
        result = solver.solve(
            target.tcp_xyz[i], target.base_xy_yaw[i], float(target.spine[i]), target.arm_q[i]
        )
        if i > 0 and (
            not result.succeeded or float(np.max(np.abs(result.q - q[i - 1]))) > jump_tol
        ):
            # Retry from the previous solved frame: mid-detour that seed is
            # millimetres from this target while the demo seed is the full
            # detour away, and a demo-seeded solve can also land on the far
            # side of a configuration branch the previous frame did not —
            # the 0.5+ rad single-frame jumps the delta budget rejects. The
            # demo seed stays the FIRST try so the branch anchor is never
            # given up, only rescued.
            retry = solver.solve(
                target.tcp_xyz[i], target.base_xy_yaw[i], float(target.spine[i]), q[i - 1]
            )
            retry_jump = float(np.max(np.abs(retry.q - q[i - 1])))
            if retry.succeeded and (
                not result.succeeded or retry_jump < float(np.max(np.abs(result.q - q[i - 1])))
            ):
                result = retry
        q[i] = result.q
        ok[i] = result.succeeded
    achieved_xyz, achieved_quat = solver.kin.fk_traj(q, target.base_xy_yaw, target.spine)
    err = float(np.max(np.linalg.norm(achieved_xyz - target.tcp_xyz, axis=1)))
    out = target.copy()
    out.arm_q = q
    out.tcp_xyz = achieved_xyz
    out.tcp_quat_xyzw = achieved_quat
    out.actions = out.actions.copy()
    out.actions[:, C.A_RIGHT_ARM] = q.astype(np.float32)
    return ResolveResult(ref=out, ik_ok_fraction=float(np.mean(ok)), max_target_err_m=err)


@dataclass
class Budgets:
    """Validation limits. Defaults are the module constants; a test that
    needs to prove a check fires can tighten one in isolation."""

    max_joint_delta_rad: float = MAX_JOINT_DELTA_RAD
    max_target_err_m: float = MAX_TARGET_ERR_M
    grasp_hold_tol_m: float = GRASP_HOLD_TOL_M
    place_hold_tol_m: float = PLACE_HOLD_TOL_M
    hold_max_angle_deg: float = HOLD_MAX_ANGLE_DEG
    z_floor_margin_m: float = Z_FLOOR_MARGIN_M


@dataclass
class ValidationReport:
    passed: bool
    reasons: list[str]
    metrics: dict

    def as_dict(self) -> dict:
        return {"passed": self.passed, "reasons": self.reasons, "metrics": self.metrics}


def _hold_deviation(
    var: Reference, var_anchor: int, src: Reference, src_anchor: int, pre: int, post: int
) -> tuple[float, float]:
    """(max position m, max angle deg) between the two anchored windows."""
    max_pos, max_ang = 0.0, 0.0
    for j in range(-pre, post + 1):
        vi = min(max(var_anchor + j, 0), var.frames - 1)
        si = min(max(src_anchor + j, 0), src.frames - 1)
        max_pos = max(max_pos, float(np.linalg.norm(var.tcp_xyz[vi] - src.tcp_xyz[si])))
        max_ang = max(
            max_ang, quat_xyzw_angle_deg(var.tcp_quat_xyzw[vi], src.tcp_quat_xyzw[si])
        )
    return max_pos, max_ang


def validate(
    resolved: ResolveResult,
    grasp_src: Reference,
    place_src: Reference | None = None,
    budgets: Budgets | None = None,
) -> ValidationReport:
    """Judge one resolved variant against its source(s).

    ``grasp_src`` anchors the grasp checks; ``place_src`` the place checks
    (defaults to ``grasp_src`` — a splice passes its B side). A variant
    that fails ANY check is discarded, never loosened: the anchors are what
    make "this will still grasp and still place" an argument instead of a
    hope.
    """
    budgets = budgets or Budgets()
    place_src = place_src or grasp_src
    var = resolved.ref
    reasons: list[str] = []
    metrics: dict = {
        "ik_ok_fraction": resolved.ik_ok_fraction,
        "max_target_err_mm": resolved.max_target_err_m * 1000.0,
    }
    if resolved.ik_ok_fraction < 1.0:
        reasons.append(f"IK failed on {(1.0 - resolved.ik_ok_fraction) * 100:.1f}% of frames")
    if resolved.max_target_err_m > budgets.max_target_err_m:
        reasons.append(
            f"achieved TCP off target by {resolved.max_target_err_m * 1000:.1f} mm "
            f"(> {budgets.max_target_err_m * 1000:.1f})"
        )

    try:
        kf_var = find_keyframes(var)
    except ValueError as exc:
        return ValidationReport(False, [*reasons, str(exc)], metrics)
    kf_grasp = find_keyframes(grasp_src)
    kf_place = find_keyframes(place_src)

    delta = float(np.max(np.abs(np.diff(var.arm_q, axis=0)))) if var.frames > 1 else 0.0
    metrics["max_joint_delta_rad"] = delta
    if delta > budgets.max_joint_delta_rad:
        reasons.append(
            f"per-frame joint delta {delta:.3f} rad exceeds the demo-level "
            f"budget {budgets.max_joint_delta_rad:.3f}"
        )

    pos, ang = _hold_deviation(
        var, kf_var.close, grasp_src, kf_grasp.close, GRASP_HOLD_PRE, GRASP_HOLD_POST
    )
    metrics["grasp_hold_dev_mm"], metrics["grasp_hold_dev_deg"] = pos * 1000.0, ang
    if pos > budgets.grasp_hold_tol_m:
        reasons.append(f"grasp hold drifts {pos * 1000:.1f} mm off the source grasp")
    if ang > budgets.hold_max_angle_deg:
        reasons.append(f"grasp attitude {ang:.1f}° off the source (F-98 margin)")

    pos, ang = _hold_deviation(
        var, kf_var.open, place_src, kf_place.open, PLACE_HOLD_PRE, PLACE_HOLD_POST
    )
    metrics["place_hold_dev_mm"], metrics["place_hold_dev_deg"] = pos * 1000.0, ang
    if pos > budgets.place_hold_tol_m:
        reasons.append(f"place hold drifts {pos * 1000:.1f} mm off the source place")
    if ang > budgets.hold_max_angle_deg:
        reasons.append(f"place attitude {ang:.1f}° off the source")

    floor = min(float(grasp_src.tcp_xyz[:, 2].min()), float(place_src.tcp_xyz[:, 2].min()))
    z_min = float(var.tcp_xyz[:, 2].min())
    metrics["z_min_m"], metrics["z_floor_m"] = z_min, floor
    if z_min < floor - budgets.z_floor_margin_m:
        reasons.append(
            f"TCP dives to {z_min:.3f} m, below the source floor "
            f"{floor:.3f} m − {budgets.z_floor_margin_m * 1000:.0f} mm"
        )
    carry_src = float(
        place_src.tcp_xyz[kf_place.lift : max(kf_place.lift + 1, kf_place.descend), 2].min()
    )
    carry_var = float(
        var.tcp_xyz[kf_var.lift : max(kf_var.lift + 1, kf_var.descend), 2].min()
    )
    metrics["carry_z_min_m"] = carry_var
    if carry_var < carry_src - budgets.z_floor_margin_m:
        reasons.append(
            f"carry dips to {carry_var:.3f} m, below the source carry floor {carry_src:.3f} m"
        )

    end_err = float(np.linalg.norm(var.tcp_xyz[-1] - place_src.tcp_xyz[-1]))
    metrics["end_err_mm"] = end_err * 1000.0
    if end_err > 0.02:
        reasons.append(f"final TCP {end_err * 1000:.0f} mm from the source's rest pose")

    return ValidationReport(passed=not reasons, reasons=reasons, metrics=metrics)


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------


@dataclass
class Variant:
    name: str
    family: str
    ref: Reference
    params: dict
    report: ValidationReport
    sources: list[str] = field(default_factory=list)


_SINGLE_SOURCE_FAMILIES = {
    "start_offset": aug_start_offset,
    "approach": aug_approach,
    "disturbance": aug_disturbance,
    "transfer": aug_transfer,
    "timewarp": aug_timewarp,
}


def generate(
    refs: list[Reference],
    family: str,
    count: int,
    seed: int,
    scale: float = 1.0,
    ik: RightArmIK | None = None,
    budgets: Budgets | None = None,
) -> list[Variant]:
    """Draw ``count`` variants of ``family`` (validated AND failed — the
    caller keeps the passing ones and logs the rest; silently dropping
    failures would hide a mis-tuned magnitude behind a short manifest).

    Deterministic per (seed, family, k): re-running with the same inputs
    reproduces every variant bit-for-bit.
    """
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")
    if not refs:
        raise ValueError("need at least one source reference")
    if family == "splice" and len(refs) < 2:
        raise ValueError("splice needs at least two source references")
    solver = ik if ik is not None else RightArmIK()
    keyframes = [find_keyframes(r) for r in refs]
    out: list[Variant] = []
    family_id = FAMILIES.index(family)
    for k in range(count):
        # Not hash(family): Python string hashes are salted per process and
        # would make "deterministic per (seed, family, k)" silently false.
        rng = np.random.default_rng([seed, family_id, k])
        if family == "splice":
            ia = int(rng.integers(len(refs)))
            ib = int(rng.integers(len(refs) - 1))
            ib = ib + 1 if ib >= ia else ib
            target, params = aug_splice(
                refs[ia], keyframes[ia], refs[ib], keyframes[ib], rng, scale
            )
            grasp_src, place_src = refs[ia], refs[ib]
            sources = [refs[ia].source, refs[ib].source]
        else:
            i = int(rng.integers(len(refs)))
            target, params = _SINGLE_SOURCE_FAMILIES[family](refs[i], keyframes[i], rng, scale)
            grasp_src = place_src = refs[i]
            sources = [refs[i].source]
        resolved = resolve(target, ik=solver)
        report = validate(resolved, grasp_src, place_src, budgets=budgets)
        name = f"{sources[0]}_aug_{family}_{k:03d}"
        out.append(
            Variant(
                name=name,
                family=family,
                ref=resolved.ref,
                params={"seed": seed, "k": k, "scale": scale, **params},
                report=report,
                sources=sources,
            )
        )
    return out
