"""Hold the right-gripper close once the policy commands it (rig-side lever).

The ACT checkpoint's gripper channel copies its own gripper state (F-97):
after a full close it re-opens to ~0.45 and hovers half-open rather than
staying shut. Measured on the rig today, `outputs/rig/munich_2026-09-01/
t6a_a03_150950`: command 0.20 / measured 0.02 from 25.8 to 28.8 s at the
demo grasp pose, then command back to 0.44-0.7 for the remaining 90 s. The
grasp never fails geometrically -- the channel just does not hold what it
already achieved.

`camelo.control.grasp_gate.GraspGate` is the geometric fix for the sibling
problem (the checkpoint never PREDICTS a close at all), but it needs a live
pad pose from `OBJECT_POSES_TOPIC` to decide when to fire -- and the real
rig has no such pose (`pad_observed_ticks: 0`, docs/realdata/16). This is
therefore an independent, simpler filter: it watches the policy's OWN
right-gripper command and holds the close once the command itself says
"closed", with no geometry and no pad. off (the default) is a byte-identical
passthrough -- the same "None = defer to the policy" shape `GraspGate.update`
uses, just returning the value unchanged instead of None.

CANONICAL open fraction throughout (1.0 = open, 0.0 = closed -- see
`camelo.contracts.gripper_open_fraction`), matching every other gripper
reading in this codebase.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

# Tracked corpus of per-slot demo grasp poses (docs/realdata/16). Only the
# "slot:xNNN" POSE_SPEC form reads this file.
SLOT_GRASP_POSES_PATH = Path("outputs/rig/munich_2026-09-01/slot_grasp_poses.json")

# Default threshold on the policy's OWN commanded open fraction that counts
# as "closing" -- below this the latch engages. 0.35 sits well above the
# corpus's closed value (0.21) and well below its open rest values, so a
# policy command that is genuinely trying to close crosses it long before
# the state-copy channel's re-open drifts back up past it.
DEFAULT_CLOSE_BELOW = 0.35

# Minimum dwell once engaged, sim seconds. 30 s is "the rest of a 30 s
# rollout" by construction -- the lever exists to stop a demo-length rollout
# from ever seeing the re-open at all.
DEFAULT_HOLD_S = 30.0

# The policy must command above this, continuously, before the latch will
# even consider releasing. High enough that the observed 0.44-0.7 re-open
# never qualifies -- only a policy that is unambiguously trying to open the
# hand again releases it.
DEFAULT_RELEASE_ABOVE = 0.9

# How long the release condition must hold continuously (sim seconds) before
# the latch actually lets go. Guards a single glitchy tick above
# RELEASE_ABOVE from releasing a grasp mid-transport.
RELEASE_SUSTAIN_S = 1.0

# What the latch clamps the published command to while engaged -- the
# corpus's own closed value (0.21), rounded down so "at most this open"
# reads as fully closed rather than as a number someone has to look up.
LATCHED_VALUE = 0.20


@dataclass
class GripperLatch:
    """Hold the right gripper closed once the policy's own command says so.

    ``update()`` is called every control tick with the RIGHT gripper command
    that would otherwise be published (canonical open fraction, post
    grasp-gate if one is also configured) and returns the value to publish
    instead. It never touches the left gripper -- callers must not run this
    on ``command.left_gripper``.

    Sequence: below ``close_below`` -> engage, publish ``min(cmd,
    LATCHED_VALUE)`` every tick (so a policy re-open attempt like the
    measured 0.44-0.7 is clamped back down, not merely overridden once) for
    at least ``hold_s`` seconds; thereafter, once the policy commands above
    ``release_above`` for ``RELEASE_SUSTAIN_S`` seconds continuously, release
    and pass the policy's own value through again. A latch that releases can
    re-engage on a later close -- there is nothing one-shot about it, the
    same way the policy's own channel can close more than once in a rollout.
    """

    close_below: float = DEFAULT_CLOSE_BELOW
    hold_s: float = DEFAULT_HOLD_S
    release_above: float = DEFAULT_RELEASE_ABOVE
    # Joint-space proximity gate on the ENGAGE transition only (U-40 / a04):
    # a threshold-only latch fires on the ACT checkpoint's pre-shape dip
    # (gripper command 0.4-0.6 for many seconds during the approach) while
    # the arm is still far from the grasp pose. When set, the latch may only
    # start holding while the measured right arm (7 joints, L2) is within
    # ``near_radius_rad`` of ``near_pose``; a command below ``close_below``
    # outside that radius passes through unchanged, exactly like a command
    # at or above it. Hold and release are unaffected -- this only gates
    # whether a NEW engage is allowed to start. Default None = off, the
    # original threshold-only behaviour, byte-identical.
    near_pose: tuple[float, ...] | None = None
    near_radius_rad: float | None = None

    engaged: bool = field(default=False, init=False)
    engage_t: float | None = field(default=None, init=False)
    release_t: float | None = field(default=None, init=False)
    engaged_ticks: int = field(default=0, init=False)
    # L2 distance (rad) to near_pose at the tick the latch last engaged, or
    # None when no near-gate is configured (or it has never engaged yet).
    near_dist_at_engage: float | None = field(default=None, init=False)
    # Ticks where the command was below close_below but the near-gate
    # refused the engage -- "would engage but out of range".
    blocked_ticks: int = field(default=0, init=False)
    # Sim time the sustained-open run currently being timed started, or None
    # when the policy is not (right now) above release_above.
    _open_since_t: float | None = field(default=None, init=False)

    def reset(self) -> None:
        self.engaged = False
        self.engage_t = None
        self.release_t = None
        self.engaged_ticks = 0
        self.near_dist_at_engage = None
        self.blocked_ticks = 0
        self._open_since_t = None

    def update(
        self,
        right_gripper_cmd: float,
        t_sim: float,
        right_arm_rad=None,
    ) -> float:
        """The right-gripper value to publish this tick.

        ``right_gripper_cmd`` is whatever would otherwise reach the wire
        this tick (the policy's raw value, or a grasp-gate override) --
        the latch sits after both and before the publisher. Returns that
        same value unchanged whenever it is not engaged and does not engage
        this tick, so a caller that never sees ``cmd < close_below`` gets
        exactly its own input back.

        ``right_arm_rad`` is the tick's measured right-arm state
        (``contracts.S_RIGHT_ARM``, 7 floats) -- required on every call when
        ``near_pose`` is configured, ignored otherwise.
        """
        cmd = float(right_gripper_cmd)
        t = float(t_sim)
        if not self.engaged:
            if cmd >= self.close_below:
                return cmd
            if self.near_pose is not None:
                if right_arm_rad is None:
                    raise ValueError(
                        "GripperLatch configured with near_pose requires "
                        "right_arm_rad on every update() call"
                    )
                dist = _l2(right_arm_rad, self.near_pose)
                if dist > self.near_radius_rad:
                    self.blocked_ticks += 1
                    return cmd
                self.near_dist_at_engage = dist
            self.engaged = True
            self.engage_t = t
            self._open_since_t = None
        # Engaged now, whether just entered above or already was.
        if t - self.engage_t >= self.hold_s:
            if cmd > self.release_above:
                if self._open_since_t is None:
                    self._open_since_t = t
                elif t - self._open_since_t >= RELEASE_SUSTAIN_S:
                    self.engaged = False
                    self.release_t = t
                    self._open_since_t = None
                    return cmd  # the releasing tick's own value passes through
            else:
                self._open_since_t = None  # any dip below breaks the sustain run
        self.engaged_ticks += 1
        return min(cmd, LATCHED_VALUE)

    def stats(self) -> dict:
        stats = {
            "gripper_latch_engaged_ticks": self.engaged_ticks,
            "gripper_latch_engage_t": self.engage_t,
            "gripper_latch_release_t": self.release_t,
        }
        # Only present when the near-gate is configured, so a plain
        # `--gripper-latch` run's stats dict stays byte-identical.
        if self.near_pose is not None:
            stats["gripper_latch_near_rad"] = self.near_dist_at_engage
            stats["gripper_latch_blocked_ticks"] = self.blocked_ticks
        return stats


def parse_gripper_latch_spec(spec: str) -> tuple[float, float, float]:
    """``"CLOSE_BELOW[:HOLD_S[:RELEASE_ABOVE]]"`` -> the three floats.

    Trailing fields are optional and fall back to the module defaults, so
    ``"0.35"`` and ``"0.35:30.0:0.9"`` build the same latch. Raises
    ``ValueError`` (the CLI wraps it as ``SystemExit``) on anything else --
    an unparsable ``--gripper-latch`` must not fall back to "off" silently.
    """
    parts = spec.split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError(
            f"--gripper-latch must be 'CLOSE_BELOW[:HOLD_S[:RELEASE_ABOVE]]', got {spec!r}"
        )
    defaults = (DEFAULT_CLOSE_BELOW, DEFAULT_HOLD_S, DEFAULT_RELEASE_ABOVE)
    try:
        values = tuple(
            float(p) if p != "" else d for p, d in zip(parts, defaults, strict=False)
        )
    except ValueError:
        raise ValueError(
            f"--gripper-latch fields must be floats, got {spec!r}"
        ) from None
    values = values + defaults[len(values):]
    return values  # type: ignore[return-value]


def _l2(a, b) -> float:
    """L2 distance between two equal-length joint-angle sequences (rad)."""
    a = list(a)
    b = list(b)
    if len(a) != len(b):
        raise ValueError(f"joint vectors must be the same length, got {len(a)} vs {len(b)}")
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b, strict=True)))


def parse_gripper_latch_near_spec(spec: str) -> tuple[str, float]:
    """``"POSE_SPEC:RADIUS_RAD"`` -> ``(pose_spec, radius)``.

    POSE_SPEC itself always starts with ``file:``, ``slot:`` or ``list:`` and
    may contain further colons (a file path) or commas (an inline list), so
    only the trailing RADIUS_RAD field is peeled off the right with
    ``rpartition`` -- the rest, colons and all, is handed to
    `resolve_gripper_latch_near_pose` unexamined.
    """
    pose_spec, sep, radius_str = spec.rpartition(":")
    if not sep or not pose_spec:
        raise ValueError(
            "--gripper-latch-near must be 'POSE_SPEC:RADIUS_RAD', got "
            f"{spec!r}"
        )
    try:
        radius = float(radius_str)
    except ValueError:
        raise ValueError(
            f"--gripper-latch-near RADIUS_RAD must be a float, got {spec!r}"
        ) from None
    return pose_spec, radius


def resolve_gripper_latch_near_pose(pose_spec: str) -> tuple[float, ...]:
    """``POSE_SPEC`` -> the 7-float right-arm reference pose it names.

    Three forms:

    * ``file:PATH`` -- a JSON file with key ``"right"`` (7 floats), the same
      shape as ``outputs/rig/t5/start_pose_s27a15_ep163.json``.
    * ``slot:xNNN`` -- ``mean_right_arm_rad`` of that slot in the tracked
      ``outputs/rig/munich_2026-09-01/slot_grasp_poses.json``.
    * ``list:a,b,c,d,e,f,g`` -- the 7 floats given inline.

    Raises ``ValueError`` (the CLI wraps it as ``SystemExit``) on an unknown
    form, a missing file/slot, or anything but exactly 7 floats.
    """
    kind, sep, rest = pose_spec.partition(":")
    if not sep:
        raise ValueError(
            "--gripper-latch-near POSE_SPEC must be 'file:PATH', "
            f"'slot:xNNN' or 'list:a,b,c,d,e,f,g', got {pose_spec!r}"
        )
    if kind == "file":
        path = Path(rest)
        if not path.is_file():
            raise ValueError(f"--gripper-latch-near file not found: {path}")
        payload = json.loads(path.read_text())
        if "right" not in payload:
            raise ValueError(f"{path} has no \"right\" key")
        pose = payload["right"]
    elif kind == "slot":
        if not SLOT_GRASP_POSES_PATH.is_file():
            raise ValueError(
                f"--gripper-latch-near slot poses not found: {SLOT_GRASP_POSES_PATH}"
            )
        slots = json.loads(SLOT_GRASP_POSES_PATH.read_text())
        if rest not in slots:
            raise ValueError(
                f"unknown slot {rest!r} in {SLOT_GRASP_POSES_PATH} "
                f"(have {sorted(slots)})"
            )
        pose = slots[rest]["mean_right_arm_rad"]
    elif kind == "list":
        pose = rest.split(",")
    else:
        raise ValueError(
            "--gripper-latch-near POSE_SPEC must be 'file:PATH', "
            f"'slot:xNNN' or 'list:a,b,c,d,e,f,g', got {pose_spec!r}"
        )
    try:
        pose = tuple(float(v) for v in pose)
    except (TypeError, ValueError):
        raise ValueError(
            f"--gripper-latch-near pose values must be floats, got {pose_spec!r}"
        ) from None
    if len(pose) != 7:
        raise ValueError(
            f"--gripper-latch-near pose must have 7 joints, got {len(pose)}: {pose_spec!r}"
        )
    return pose
