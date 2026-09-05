"""Joint-state liveness guard — docs/realdata/16 U-27.

The sibling of `camelo.control.image_age`'s U-23 guard, for the other half
of the observation. `ObsCollector` caches joint positions in a dict that is
only ever OVERWRITTEN (`obs_collector.py`'s `joint_states`, fed by
`_ingest_joints`), never cleared or expired — exactly like `images`. So a
`JointState` stream that stops mid-rollout leaves the last measured array in
place forever: `get_obs` keeps returning a complete observation, the policy
keeps being fed a frozen pose, `--joint-csv`'s "measured" columns keep
recording numbers that are no longer measurements, and nothing raises.

MEASURED 2026-09-02 17:29 on the Munich rig (R-64): during a
`run_policy --world real --backend remote --arms right --activate-arms` run
the companion's right `ros2_control_node` hit
`[FATAL] Timeout: No valid joint states received from Gello` 0.5 s after
activation — a clock-skew message-age rejection — and took the whole
upper-body launch down with it. `/right/franka_robot_state_broadcaster/
measured_joint_states`, `/right/gripper/joint_states` and the right wrench
all stopped at that instant. The runner noticed nothing: it drove the full
30 s rollout against the frozen state and only the final deactivation
service call timed out. Silent, not loud — the failure class AGENTS.md's
drift-alarm rule exists to prevent.

Like the image guard, the age read here is the message's WALL RECEIPT time
(`time.monotonic()` when the collector's callback ran), never a header stamp
or `t_sim`: the incident was itself a CLOCK-SKEW rejection, so an alarm that
trusted the publisher's own notion of time would be reasoning with the
quantity that broke. Receipt time is recorded by our own callback and grows
without bound the moment the stream stops, whatever the stamps say.

Numpy-free and ROS-free: the collector feeds it, the runner and the
real-arm session read it, tests run it offline.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

from camelo import contracts as C

#: Default staleness threshold for a `--world real` rollout, in WALL seconds.
#: The arm broadcasters publish at ~1081 Hz per side and the grippers at
#: 100 Hz (MEASURED on the rig, docs/realdata/16 R-43/R-37), so 0.2 s is
#: ~200 missed arm messages — far outside any scheduling hiccup, and well
#: inside the companion controller's own 0.5 s `max_gello_message_age`
#: rejection window, so the runner notices in the same order of magnitude
#: the controller does rather than a rollout later.
DEFAULT_MAX_STATE_AGE_S = 0.2

#: The two sides this guard watches. Deliberately NOT every entry of
#: `TopicMap.joint_state_topics`: `/spine/joint_states` has never come up on
#: the Munich station (R-07's base stage, R-24's silent spine) and the sim's
#: single un-sided `/isaac/joint_states_full` belongs to a world where the
#: guard is off anyway — including either would refuse healthy rigs.
ARM_SIDES = ("left", "right")


def state_liveness_key(topic: str, side: str) -> str | None:
    """The guard's key for one `joint_state_topics` entry, or None to skip.

    Reuses `ObsCollector`'s existing counter vocabulary so a fault line and
    a `rates()` table name the same stream: `joint_states_left` /
    `joint_states_right` for the arm broadcasters, `gripper_js_left` /
    `gripper_js_right` for the gripper `JointState`s that share their side.
    """
    if side not in ARM_SIDES:
        return None
    if C.is_gripper_joint_state_topic(topic):
        return f"gripper_js_{side}"
    return f"joint_states_{side}"


def state_liveness_keys(joint_state_topics: Iterable[tuple[str, str]]) -> list[str]:
    """Every key the guard watches for a topic map, in subscription order."""
    keys: list[str] = []
    for topic, side in joint_state_topics:
        key = state_liveness_key(topic, side)
        if key is not None and key not in keys:
            keys.append(key)
    return keys


class StaleStateError(RuntimeError):
    """A joint-state stream stopped while the policy was still consuming it.

    Raised by the runner (every control tick) and by `RealArmSession`
    (once, before activation), never by the collector: the collector's job
    is to cache whatever arrives, and only a rollout makes a frozen pose a
    fault rather than a fact. Handled exactly like
    `camelo.control.image_age.StaleImageError` and
    `camelo.runner.recenter.StartPoseNotReached` — it escapes `run_rollout`,
    `scripts/run_policy.py`'s guarded `finally` deactivates the arms on the
    way out, and the process exits non-zero.
    """


def stale_state_ages(
    last_wall: Mapping[str, float | None],
    now_wall: float,
    max_age_s: float,
) -> dict[str, float]:
    """``{stream: age_s}`` for every stream whose newest message is too old.

    ``last_wall`` is `ObsCollector.last_state_wall()` (monotonic receipt time
    per stream); a stream that has never delivered a message reads ``inf``
    rather than being skipped — "no joint state at all" is not fresher than
    a frozen one.
    """
    stale: dict[str, float] = {}
    for key, t_wall in sorted(last_wall.items()):
        age = math.inf if t_wall is None else float(now_wall - t_wall)
        if age > max_age_s:
            stale[key] = age
    return stale


def max_state_age(
    last_wall: Mapping[str, float | None],
    now_wall: float,
) -> float | None:
    """Worst age across the watched streams — the `--joint-csv` column.

    ``None`` only when nothing is watched at all (never 0.0: the batch_eval
    rule that an unmeasured quantity must not read as a measured zero).
    ``inf`` when some stream has never delivered — the same "no message is
    not fresh" convention as `stale_state_ages`.
    """
    if not last_wall:
        return None
    return max(
        math.inf if t_wall is None else float(now_wall - t_wall)
        for t_wall in last_wall.values()
    )


def stale_state_report(
    last_wall: Mapping[str, float | None],
    now_wall: float,
    max_age_s: float | None,
) -> str | None:
    """The fault line for a dead joint-state stream, or None when all are live.

    ``max_age_s`` of ``None`` (or <= 0) is the guard switched OFF and always
    returns ``None`` — that is the sim default and the explicit
    ``--max-state-age-s 0`` opt-out, and it is the ONLY way this returns
    None while a stream is in fact dead.
    """
    if max_age_s is None or max_age_s <= 0.0:
        return None
    stale = stale_state_ages(last_wall, now_wall, max_age_s)
    if not stale:
        return None
    detail = ", ".join(
        f"{key} {'never delivered a message' if math.isinf(age) else f'{age:.2f} s old'}"
        for key, age in stale.items()
    )
    live = {k: v for k, v in sorted(last_wall.items()) if k not in stale}
    live_detail = ", ".join(
        f"{k} {now_wall - v:.2f} s" for k, v in live.items() if v is not None
    )
    return (
        f"STALE JOINT STATE: {detail} (limit {max_age_s:.2f} s wall). The "
        "collector never clears a cached joint position, so this arm has been "
        "feeding the policy — and the --joint-csv 'measured' columns — the "
        "SAME frozen pose since its stream stopped (docs/realdata/16 U-27). A "
        "controller that shut its arm launch down looks exactly like this. "
        "Check the companion's ros2_control_node before rerunning; "
        "--max-state-age-s 0 disables the guard."
        + (f" Live streams: {live_detail}." if live_detail else "")
    )
