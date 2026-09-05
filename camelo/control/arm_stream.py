"""Was the published arm stream — and the arm itself — standing still?

The `--arm-command-frame gello` map is `g = g0 + dir*(q_target - q0)`, and
the companion's `joint_impedance_controller` undoes it with
`q_goal = q0 + dir*(g - g0)` using ITS OWN `(q0, g0)`, captured at
`on_activate`. The two only cancel if our `(q0, g0)` name the same instant as
the controller's. Nothing on this side observes that instant: the runner
polls `list_controllers` at `poll_s`, so `on_activate` may have fired up to
one full poll interval before we notice it.

So the reference cannot be *verified*, only *bounded* — and the bound is
stationarity. If neither the published command stream nor the measured arm
moved across the whole window between the controller's possible capture
instant and ours, then any `(q0, g0)` taken inside that window is the same
pose to within the spread, whichever instant the controller picked.

Two numbers say that, and both are needed:

  * **spread** — per-joint `max - min` of everything PUBLISHED inside the
    window. This is the half that a "compare the last publish against the
    value we published two lines ago" check cannot see: those are the same
    number by construction and their difference is always ~0.
  * **delta** — newest published value vs the MEASURED pose handed in. The
    arm can lag or sag while the stream repeats a constant.

Above `ACTIVATION_DRIFT_TOL_RAD` the reference is not a single pose. In the
`robot` frame that is a warning (the wire carries absolute targets; a wrong
reference changes nothing that goes out). In the `gello` frame it is fatal:
`(q0, g0)` multiply every command, and joints 1, 2 and 7 have `dir = -1`, so
a wrong reference sends them the wrong way with a perfectly valid message.

The same ring buffer answers a second, independent question — **how fast was
the stream going into the switch?** (`guard_publish_cadence`). Stationarity
says the stream repeated one pose; cadence says it repeated it often enough.
Both were true yesterday and only the first was true during the 2026-09-02
T6 attempt, where "4 publishes in the last 0.60s" was logged and the runner
activated anyway (docs/realdata/16 U-28).

Numpy + stdlib only — no ROS, so the runner's fakes can drive the real guard.
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

log = logging.getLogger(__name__)

#: Per-joint tolerance for the activation reference, radians. Below the
#: arms' own steady-state wobble; above it the stream/arm moved.
ACTIVATION_DRIFT_TOL_RAD = 0.01
#: How much published history the publisher keeps, seconds. Two seconds
#: covers any plausible `poll_s + margin` window with room to spare, and is
#: also the controller's own liveness gap — nothing older can matter.
ARM_HISTORY_S = 2.0
#: Added to the caller's poll interval to get the window: the controller may
#: have activated just before our previous poll, plus scheduling slop.
ACTIVATION_WINDOW_MARGIN_S = 0.1
#: Default `--min-activation-publish-hz`: the publish cadence below which the
#: activation switch is REFUSED. MEASURED on the rig 2026-09-02 (the first T6
#: attempt, docs/realdata/16 U-28): with three camera workers draining into
#: the same executor the hold loop's own cadence had already collapsed to
#: 6.7 Hz — the activation report read "4 publishes in the last 0.60s" — and
#: the controller answered `on_activate` with "Rejecting GELLO state: message
#: too old (age 1.075 s)" 0.485 s later, then FATAL. 8 Hz sits just under the
#: 10 Hz keep-alive floor and far above the 2 Hz the controller's 0.5 s
#: stamp-age window strictly needs, so it fires only on a stream that is
#: already sick — never on a healthy one.
MIN_ACTIVATION_PUBLISH_HZ = 8.0


class ActivationReferenceDrift(RuntimeError):
    """The published stream or the arm moved across the capture window."""


class PublishCadenceTooLow(RuntimeError):
    """The command stream was too slow to hand an activating controller.

    Raised BEFORE the `switch_controller` call, never after: the whole point
    is that no switch happens. A controller that goes ACTIVE onto a starved
    stream rejects the first sample it sees as too old and takes the arm
    launch down with it.
    """


class ArmPublishHistory:
    """Ring buffer of ``(t_monotonic, left(7), right(7))`` arm publishes.

    Every arm publish goes in — policy ticks, hold ticks and keep-alive
    repeats alike, because the question is what the CONTROLLER saw, and it
    cannot tell those apart either.
    """

    def __init__(self, keep_s: float = ARM_HISTORY_S):
        self.keep_s = float(keep_s)
        self._entries: deque = deque()

    def append(self, t: float, left, right) -> None:
        t = float(t)
        self._entries.append(
            (t, [float(v) for v in left], [float(v) for v in right])
        )
        cutoff = t - self.keep_s
        while self._entries and self._entries[0][0] < cutoff:
            self._entries.popleft()

    def newest(self) -> tuple | None:
        return self._entries[-1] if self._entries else None

    def window(self, now: float, window_s: float) -> list:
        cutoff = float(now) - float(window_s)
        return [entry for entry in self._entries if entry[0] >= cutoff]

    def __len__(self) -> int:
        return len(self._entries)


def _worst(values) -> tuple:
    """(side, joint number, value) of the largest of two 7-vectors."""
    left, right = values
    side = "left" if float(np.max(left)) >= float(np.max(right)) else "right"
    arm = left if side == "left" else right
    joint = int(np.argmax(arm))
    return side, joint + 1, float(arm[joint])


def publish_cadence_hz(samples: int, window_s: float) -> float:
    """Publishes per second over a window. One place, so the gate and the
    activation report cannot report different numbers for the same stream."""
    window_s = float(window_s)
    return float(samples) / window_s if window_s > 0.0 else 0.0


def guard_publish_cadence(
    history: ArmPublishHistory,
    now: float,
    window_s: float,
    min_hz: float = MIN_ACTIVATION_PUBLISH_HZ,
) -> dict:
    """Refuse to activate onto a command stream that has already collapsed.

    The controller rejects any sample stamped more than 0.5 s in the past and
    needs a fresh one within 0.5 s of `on_activate`, continuously afterwards.
    A stream that is publishing at 6.7 Hz *before* the switch is not merely
    slow — it is a stream whose publisher is not being scheduled, and the
    activation instant is precisely when that stops being survivable.

    Counts PUBLISHES, not a rate estimate: `min_hz * window_s` publishes have
    to be in the window. `min_hz <= 0` measures and logs without a verdict
    (`--min-activation-publish-hz 0`). Returns the measurement either way, so
    the caller's stats carry the number that was actually seen.
    """
    samples = len(history.window(now, window_s))
    hz = publish_cadence_hz(samples, window_s)
    report = {
        "publish_window_s": float(window_s),
        "publish_samples": samples,
        "publish_hz": hz,
        "min_publish_hz": float(min_hz),
    }
    measured = (
        f"{samples} publishes in the last {float(window_s):.2f}s ({hz:.1f} Hz)"
    )
    if min_hz <= 0.0:
        log.info(
            "activation publish cadence: %s (gate OFF, "
            "--min-activation-publish-hz 0)", measured,
        )
        return report
    if samples < float(min_hz) * float(window_s):
        error = PublishCadenceTooLow(
            f"the command stream is too slow to activate onto: {measured}, "
            f"below the required {float(min_hz):.1f} Hz. The companion's "
            "joint_impedance_controller wants a sample stamped less than "
            "0.5 s old within 0.5 s of on_activate and continuously after "
            "it; a publisher that is already being starved (rig 2026-09-02: "
            "camera drain callbacks on the same executor, docs/realdata/16 "
            "U-28) will not recover at the switch. Nothing was switched. "
            "Reduce --cameras, raise --keepalive-hz, or lower "
            "--min-activation-publish-hz deliberately"
        )
        # The measurement travels WITH the refusal: a caller that only ever
        # sees the exception should still be able to put the numbers in its
        # stats rather than only the sentence.
        error.report = report
        raise error
    log.info("activation publish cadence OK: %s (min %.1f Hz)", measured, float(min_hz))
    return report


def activation_reference_report(
    history: ArmPublishHistory,
    measured_left,
    measured_right,
    now: float,
    window_s: float,
) -> dict:
    """Per-joint spread of the published window + delta against the arms.

    Pure measurement — the verdict is `guard_activation_reference`'s. Returns
    the rows as well as the maxima (AGENTS.md, "the summary lies"): a single
    max hides which joint moved, and on a 14-joint robot that is the only
    thing worth knowing.
    """
    measured = []
    for name, arm in (("left", measured_left), ("right", measured_right)):
        values = np.asarray(arm, dtype=np.float64).ravel()
        if values.shape != (7,):
            raise ValueError(f"measured {name} arm must be 7 joints, got {values.shape}")
        measured.append(values)
    left, right = measured

    entries = history.window(now, window_s)
    spreads = [np.zeros(7), np.zeros(7)]
    if len(entries) >= 2:
        for index in (0, 1):
            published = np.array([e[index + 1] for e in entries], dtype=np.float64)
            spreads[index] = published.max(axis=0) - published.min(axis=0)
    spread_side, spread_joint, spread = _worst(spreads)

    newest = history.newest()
    if newest is None:
        deltas = [np.zeros(7), np.zeros(7)]
    else:
        deltas = [
            np.abs(np.asarray(newest[1], dtype=np.float64) - left),
            np.abs(np.asarray(newest[2], dtype=np.float64) - right),
        ]
    delta_side, delta_joint, delta = _worst(deltas)

    return {
        "window_s": float(window_s),
        "samples": len(entries),
        "publish_hz": publish_cadence_hz(len(entries), window_s),
        "spread_rad": spread,
        "spread_joint": (spread_side, spread_joint),
        "spread_left": [float(v) for v in spreads[0]],
        "spread_right": [float(v) for v in spreads[1]],
        "delta_rad": delta,
        "delta_joint": (delta_side, delta_joint),
        "delta_left": [float(v) for v in deltas[0]],
        "delta_right": [float(v) for v in deltas[1]],
    }


def format_activation_report(report: dict, tol: float = ACTIVATION_DRIFT_TOL_RAD) -> str:
    spread_side, spread_joint = report["spread_joint"]
    delta_side, delta_joint = report["delta_joint"]
    return (
        f"published spread {report['spread_rad']:.4f} rad (worst {spread_side} "
        f"j{spread_joint}) over {report['samples']} publishes in the last "
        f"{report['window_s']:.2f}s ({report['publish_hz']:.1f} Hz); "
        f"newest-publish vs measured delta "
        f"{report['delta_rad']:.4f} rad (worst {delta_side} j{delta_joint}); "
        f"tol {tol:.4f}"
    )


def guard_activation_reference(
    history: ArmPublishHistory,
    measured_left,
    measured_right,
    now: float,
    window_s: float,
    arm_command_frame: str,
    tol: float = ACTIVATION_DRIFT_TOL_RAD,
) -> dict:
    """Measure, then WARN (`robot` frame) or RAISE (`gello` frame).

    Returns the report either way, so the caller can put the numbers in its
    stats instead of only in a log line.
    """
    report = activation_reference_report(
        history, measured_left, measured_right, now, window_s
    )
    message = format_activation_report(report, tol)
    if report["samples"] < 2:
        # Not a drift finding — a coverage one. One sample cannot show a
        # spread, so the "0.0000" below is an absence of evidence.
        log.warning(
            "activation reference window carried %d publish(es): stationarity "
            "of the command stream is NOT established (%s)",
            report["samples"],
            message,
        )
    if report["spread_rad"] <= tol and report["delta_rad"] <= tol:
        log.info("activation reference stationary: %s", message)
        return report
    detail = (
        f"the activation reference is not a single pose: {message}. The "
        "controller's own (q0, g0) are captured at on_activate, which may "
        "have fired up to one poll interval before this check — so the stream "
        "and the arm both have to be still across the whole window, and they "
        "were not"
    )
    if arm_command_frame == "gello":
        raise ActivationReferenceDrift(
            f"{detail}. --arm-command-frame gello multiplies (q0, g0) into "
            "EVERY command (g = g0 + dir*(q_target - q0)) and dir is -1 on "
            "joints 1, 2 and 7, so a wrong reference drives them the wrong "
            "way with a valid message. Hold the arms still and re-run, or "
            "use --arm-command-frame robot"
        )
    log.warning("%s. Frame is 'robot' (absolute targets), so this is a warning", detail)
    return report
