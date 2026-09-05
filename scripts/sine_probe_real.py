#!/usr/bin/env python3
"""Real-robot sine probe = scripts/sine_actor.py + evidence.

    python3 -u scripts/sine_probe_real.py --seconds 120 --amplitude 0.03

Same command path as the policy runner (CommandPublisher / ObsCollector, ONE DDS
participant, no camera workers), REAL world only. Differences from sine_actor.py:

  * logs, once per second, the COMMANDED vs MEASURED offset of the last two joints
    of each arm and the gripper open-fractions, so "the arm followed" is a number,
    not an impression; writes the same rows to --csv;
  * prints the captured start pose BEFORE the first command, so an operator can
    eyeball it against joint limits before activating the impedance controllers;
  * gripper swing is bounded (--gripper-span, default 0.5 -> cycles 1.0..0.5 open),
    --no-gripper disables it; --arms selects which arms wiggle (others are held);
  * summary at the end: max |measured offset| per arm, final drift after the
    return-to-start, and a verdict (FOLLOWED / HELD / NOT_RECEIVING).

Safety envelope is the sine actor's: the arms only ever receive start_pose +/- amplitude
on the --joints selection; on ANY exit the start pose is republished and safe_stop()
runs. The controller-side rate limit (max_goal_velocity) and the impedance gains are
untouched.

Choreography (RUNBOOK S3/S4 on the station): start THIS first, let --hold-s seconds of
ZERO-offset publishing go by (the measured pose, i.e. a zero-delta GELLO stand-in), THEN
activate_arms.py on the companion, and DEACTIVATE before this process exits -
joint_impedance_controller shuts the whole arm stack down if its command stream stops
while it is active. The hold is not padding: on_activate refuses without a sample inside
its 2.0 s window and captures (q0, g0) at that instant, so activating DURING the hold is
what makes g0 == q0 and the wiggle a pure offset from the pose the arm is already in.

`--activate-arms` does that switching from HERE instead, which is what T1's exit
criterion asks for — the same sine, driven by the runner's own hold/activate/deactivate
path (`camelo/ros/arm_activation.py`, the sequence in `camelo/runner/real_arms.py`), on
this process's node and its existing DDS participant. The choreography then is: hold at
zero offset, publish for a second so `on_activate` has a stream to accept, switch
(best-effort then strict) the `--arms` selection, poll `list_controllers` — still
publishing on every tick — until it says active, print `t_act` and `offset(t_act)`, and
only THEN start the wiggle. `--hold-s` has to be long enough to contain all of that or
the run is refused before anything is published. On the way out — normal end, --seconds
elapsed, Ctrl+C or any exception — the arms are returned to the start pose, the
controllers are deactivated while the stream is STILL up, and `safe_stop()` runs only
after `list_controllers` confirms they are inactive. That teardown is not optional and
not interruptible: there is no `--no-deactivate-on-exit` (its only effect would have been
to stop the stream on an active controller), and SIGINT is IGNORED for the WHOLE teardown,
from the CSV write through `collector.close()` — not just across the deactivation — so no
reflex second (or third) Ctrl+C can unwind out of a service call and leave `safe_stop()` to
do the killing, or skip the evidence. Three more things make that survivable: `camelo/
ros/session.py` initialises rclpy with `SignalHandlerOptions.NO` (T1(b)), so Ctrl+C no
longer calls `rcl_shutdown()` before this code runs and the context is still valid while
the teardown talks to ROS; every teardown step runs independently, so whichever one fails
next cannot skip the ones after it, and that independence now covers `KeyboardInterrupt`
too, not only ordinary exceptions — MEASURED on the rig 2026-09-02 (t1r_123718): activation
and deactivation both worked, and a Ctrl+C that landed AFTER the deferred window (which used
to cover only the deactivation) hit the `sleep(0.5)` "settle" step as a bare
`KeyboardInterrupt` and walked out with a traceback, no summary, and no verdict — the rows
were fine (written first), but the file that says what they mean was not; and a PROVISIONAL
verdict is written to `<--csv>.summary.txt` immediately behind the CSV, before deactivation
is even attempted, so a hard kill anywhere later in the teardown (which cannot honour a
deferred signal) still leaves a verdict on disk, not just rows — the final call overwrites
it with the real one. The operator's job during `--activate-arms` is unchanged in the way
that matters: hand on the E-stop.

Pipe it with `| tee -i`, never a bare `| tee` — MEASURED 2026-09-02, T1(a) 11:07-11:10:
Ctrl+C goes to the whole foreground process group, plain `tee` dies first, and the very
next print here raised BrokenPipeError, which took the CSV write, the summary and the
verdict down with it. The rows had been collected and were lost. `tee -i` ignores SIGINT;
`python3 -u ... > outputs/rig/t1a.log 2>&1` plus `tail -f` in another pane avoids the
shared process group altogether. The probe no longer depends on either: the CSV is
written first thing in the teardown, every line after the loop also lands in
`<--csv>.summary.txt`, and a dead stdout can no longer skip the deactivation, the
`safe_stop()` or the exit code.

T1b (does the controller-side joint-direction transform need undoing?) is
`--joints 1 --amplitude 0.02`: joint 1 has dir = -1, so a wrong frame moves the arm the
other way and says so in one cycle.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from camelo import contracts as C
from camelo.cli import setup_logging
from camelo.contracts import topics_for
from camelo.runner.real_arms import (
    ACTIVATION_STREAM_LEAD_S,
    SWITCH_CONFIRM_TIMEOUT_S,
    HoldActivation,
    deactivate_controllers_guarded,
    keepalive_gap_report,
    sigint_deferred,
)

#: How much longer than `--activation-timeout-s` the hold has to be: the
#: stream lead-in before the switch, plus a second of margin. The wiggle may
#: not start before the controller is active, so a hold that cannot contain
#: the worst-case activation is refused rather than silently stretched.
ACTIVATION_HOLD_MARGIN_S = ACTIVATION_STREAM_LEAD_S + 1.0


#: The stdout object we last failed to write to. Identity, not a boolean, so
#: a test (or a caller) that installs a fresh stream is not permanently
#: silenced by an earlier one's death.
_DEAD_STDOUT = None


def _mute_broken_stdout() -> None:
    """Point fd 1 at /dev/null once stdout is gone.

    Without this, CPython tries to flush the dead stream at interpreter exit,
    fails, and turns a deliberate `return 130` into exit status 120 — the
    Ctrl+C exit code the operator reads off the terminal would be a lie about
    a run whose teardown actually succeeded.
    """
    try:
        fileno = sys.stdout.fileno()
    except Exception:  # a non-file stdout (a test double): nothing to mute
        return
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, fileno)
        os.close(devnull)
    except OSError:
        pass


def _say(text: str = "", flush: bool = True) -> bool:
    """print() that a dead stdout cannot turn into a lost run.

    MEASURED on the rig 2026-09-02 (T1(a)): Ctrl+C reaches the whole
    foreground group, `tee` exits first, and the next print raised
    BrokenPipeError — which skipped the CSV write, the summary and the
    verdict, i.e. every piece of evidence the run existed to produce. Nothing
    this script prints is worth an exception, so this never raises: stdout,
    then stderr, then silence.

    Once a stream has failed it is not tried again. That is not an
    optimisation: `_mute_broken_stdout` points the fd at /dev/null so the
    interpreter can exit cleanly, which makes every LATER write to it
    succeed — silently — and the fallback to stderr would never fire again.
    The most important line this script ever prints ("DEACTIVATION FAILED …
    deactivate by hand NOW") comes after the first broken one. The dead
    stream is remembered by identity, so a caller that installs a fresh
    stdout gets a fresh chance.
    """
    global _DEAD_STDOUT
    streams = []
    if sys.stdout is not _DEAD_STDOUT:
        streams.append(sys.stdout)
    streams.append(sys.stderr)
    for stream in streams:
        try:
            stream.write(f"{text}\n")
            if flush:
                stream.flush()
            return True
        except (BrokenPipeError, OSError, ValueError):
            if stream is sys.stdout:
                _DEAD_STDOUT = sys.stdout
                _mute_broken_stdout()
    return False


def teardown_step(out, what: str, action):
    """Run one teardown step; report and CONTINUE if it fails.

    MEASURED on the rig 2026-09-02 (T1(b), the both-arms run): a Ctrl+C that had
    already invalidated
    the rclpy context made the first line of the teardown — a publish — raise,
    and everything after it was skipped, including the deactivation. The steps
    are in priority order and each one matters more than the previous one
    failing, so no step may be able to cancel the ones after it. Returns the
    action's value, or None when it failed.

    `camelo/ros/session.py` now keeps the context valid through Ctrl+C, which
    is the actual fix; this is what makes the teardown survive the NEXT reason
    a publish can raise.

    MEASURED on the rig 2026-09-02 (t1r_123718): the operator's SECOND Ctrl+C
    landed after the deferred window (see `sigint_deferred` around the caller's
    whole teardown) and hit `sleep(0.5)` in the "settle" step as a bare
    `KeyboardInterrupt` — which this function used to let straight through,
    skipping the summary and turning a completed run into a traceback.
    `KeyboardInterrupt` is a `BaseException`, not an `Exception`, so it needs
    its own clause; `SystemExit` gets one too, explicitly, so a future change
    to this except-chain cannot accidentally start swallowing it.
    """
    try:
        return action()
    except KeyboardInterrupt:
        out(f"teardown: {what} interrupted — continuing")
        return None
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - the whole point is not to stop
        out(f"teardown: {what} FAILED ({exc!r}) — continuing")
        return None


def teardown_deactivate(publisher, controllers, out=_say):
    """The runner's guarded deactivation, with nothing left to raise.

    `deactivate_controllers_guarded` (camelo/runner/real_arms.py) is the
    shared piece: SIGINT deferred across the switch, one retry. What is
    probe-specific is the verdict when it fails anyway — returns True/False
    from `list_controllers`, or None when the switch could not be made at
    all. That last case is not lost evidence: a controller left ACTIVE faults
    its arm launch 2.0 s after this process stops publishing, so it is
    reported as an instruction and threaded into the summary file.
    """
    try:
        return deactivate_controllers_guarded(publisher, controllers, out=out)
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001
        out(f"DEACTIVATION FAILED ({exc!r}) — joint_impedance_controller "
            "may still be ACTIVE, and it shuts its arm launch down ~2s "
            "after this process stops publishing. Deactivate it by hand "
            "NOW: activate_arms.py --deactivate", flush=True)
        return None


def summary_path(csv_path) -> Path:
    """`<--csv>.summary.txt` — the verdict, on disk, next to the rows."""
    csv_path = Path(csv_path)
    return csv_path.with_name(csv_path.name + ".summary.txt")


def write_rows(csv_path, rows) -> str | None:
    """Rows to disk, FIRST thing in the teardown. Never raises.

    They are the whole measurement, they exist only in memory until this
    runs, and everything after it — a print, a service call, a robot — can
    fail. Returns the line to report, or None when there was nothing to
    write; an unwritable path is reported, never raised, because the arm
    teardown that follows matters more than the file.
    """
    if csv_path is None or not rows:
        return None
    try:
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return f"wrote {len(rows)} rows to {path}"
    except OSError as exc:
        return f"FAILED to write {csv_path}: {exc}"


def write_summary(csv_path, lines) -> bool:
    """The post-run lines to `<--csv>.summary.txt`. Never raises either."""
    if csv_path is None or not lines:
        return False
    try:
        path = summary_path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(str(line) for line in lines) + "\n")
        return True
    except OSError:
        return False


def build_summary_lines(
    args, wiggle, max_meas, t_act, offset_at_act,
    t_deact, deactivate_confirmed, final, left0, right0,
    activated_arms=None,
) -> list:
    """The verdict lines shared by the provisional and the final summary.

    Takes the individual pieces (`max_meas`, `t_act`, ...) rather than the
    `run_probe` result dict, so a PROVISIONAL call — made from inside the
    teardown, before deactivation has even been attempted — cannot reach for
    a field (`t_deact`, `final`) that does not exist yet. `summarise()` is the
    only caller that has the full dict; it unpacks it into these same
    arguments so the two call sites can never drift into different wording.
    """
    lines = []
    if t_act is not None:
        # T1b asks for offset(t_act) explicitly. It is 0 by construction —
        # activation happens inside the zero-offset hold — and printing the
        # constructed value is how a run where it ISN'T 0 becomes visible
        # instead of assumed.
        lines.append(f"activation: t_act={t_act:.2f}s "
                     f"offset(t_act)={offset_at_act:+.4f} rad "
                     f"arms={list(activated_arms or [])}")
    if t_deact is not None:
        lines.append(f"deactivation: t_deact={t_deact:.2f}s "
                     f"confirmed={deactivate_confirmed} "
                     f"(the command stream stopped only after it)")
        if deactivate_confirmed is not True:
            # The one line in this file that is an instruction rather than a
            # measurement, and the one most likely to be printed into a dead
            # terminal — so it goes on disk with the rest of the summary.
            # `list_controllers` did not agree that the controller is down:
            # it faults its arm launch 2.0 s after this process stopped
            # publishing, which by now it has.
            lines.append(
                "DEACTIVATION NOT CONFIRMED — deactivate by hand: "
                "activate_arms.py --deactivate"
            )

    verdicts = {}
    for arm in ("left", "right"):
        m = max_meas[arm]
        if arm not in wiggle:
            verdicts[arm] = f"HELD (max |meas offset| {m:.4f} rad)"
        elif m >= 0.5 * args.amplitude:
            verdicts[arm] = f"FOLLOWED (max |meas offset| {m:.4f} of {args.amplitude} rad cmd)"
        elif m >= 0.005:
            verdicts[arm] = f"PARTIAL (max |meas offset| {m:.4f} of {args.amplitude} rad cmd)"
        else:
            verdicts[arm] = "NOT_RECEIVING/INACTIVE (arm never moved)"
    if final is not None:
        dl = float(np.max(np.abs(final.state[C.S_LEFT_ARM] - left0)))
        dr = float(np.max(np.abs(final.state[C.S_RIGHT_ARM] - right0)))
        lines.append(
            f"final drift from start after return: left {dl:.4f} rad, right {dr:.4f} rad"
        )
    for arm, verdict in verdicts.items():
        lines.append(f"verdict {arm}: {verdict}")
    return lines


def provisional_summary_header(args, wiggle, interrupted: bool) -> str:
    """The header for the mid-teardown summary — same shape as the final
    one's, marked so an operator (or a script) reading the file mid-run
    cannot mistake it for the finished verdict."""
    return (
        f"# sine_probe_real summary (interrupted={interrupted}, "
        f"amplitude={args.amplitude}, joints={args.joints}, "
        f"arms={sorted(wiggle)}) "
        f"[PROVISIONAL — teardown in progress, deactivation/final observation "
        f"not yet attempted]"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--amplitude", type=float, default=0.03,
                        help="rad on the --joints selection")
    parser.add_argument("--period", type=float, default=5.0, help="s per sine cycle")
    parser.add_argument("--gripper-period", type=float, default=10.0)
    parser.add_argument("--gripper-span", type=float, default=0.5,
                        help="open-fraction swing below fully open (0.5 -> 1.0..0.5)")
    parser.add_argument("--no-gripper", action="store_true", help="hold grippers open")
    parser.add_argument("--arms", default="left,right", help="comma list of arms to wiggle")
    parser.add_argument("--joints", default="6,7",
                        help="comma list of 1-based joints to wiggle (default 6,7 — the "
                             "wrist, the smallest thing that moves visibly). T1b uses "
                             "--joints 1 --amplitude 0.02: joint 1's gello direction is "
                             "-1, so a wrong command frame is visible in one cycle")
    parser.add_argument("--hold-s", type=float, default=10.0,
                        help="seconds of ZERO-offset publishing before the wiggle starts "
                             "(default 10). Activate the controllers DURING it: "
                             "on_activate captures (q0, g0) then, and the hold is what "
                             "makes g0 == q0. 0 starts wiggling immediately")
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--csv", default="outputs/rig/sine_probe_real.csv")
    parser.add_argument("--max-amplitude", type=float, default=0.05,
                        help="refuse a larger --amplitude (real robot guard)")
    parser.add_argument(
        "--activate-arms", action="store_true",
        help="OFF by default: activate joint_impedance_controller on the "
             "--arms selection FROM HERE, during the hold, the way the runner "
             "does (all four controller_manager services first, then "
             "best-effort/strict switch, then poll list_controllers while "
             "still publishing). The wiggle does not start until every "
             "selected arm reports active. Off means an operator runs the "
             "station's activate_arms.py during the hold, as before",
    )
    parser.add_argument(
        "--activation-timeout-s", type=float, default=SWITCH_CONFIRM_TIMEOUT_S,
        help="--activate-arms only: how long list_controllers may take to "
             f"agree with the switch (default {SWITCH_CONFIRM_TIMEOUT_S:.0f}, "
             "the runner's confirm timeout). --hold-s must be at least this "
             f"plus {ACTIVATION_HOLD_MARGIN_S:.0f}s, or the run is refused",
    )
    return parser


def parse_joints(spec: str) -> list:
    """'6,7' -> [5, 6] (0-based). Raises ValueError on anything else.

    1-based on the command line because that is how the joints are named
    everywhere else on this robot (`fr3_joint1..7`); an off-by-one here would
    wiggle a different joint than the operator watched.
    """
    values = [v.strip() for v in str(spec).split(",") if v.strip()]
    if not values:
        raise ValueError("--joints must name at least one joint")
    out = []
    for value in values:
        if not value.lstrip("+").isdigit() or not 1 <= int(value) <= 7:
            raise ValueError(f"--joints entries must be 1..7, got {value!r}")
        index = int(value) - 1
        if index not in out:
            out.append(index)
    return out


def validate(args) -> tuple:
    """(wiggle arms, 0-based joints, error message or None) — no ROS needed."""
    if args.amplitude < 0 or args.amplitude > args.max_amplitude:
        return (), (), (
            f"refusing amplitude {args.amplitude} rad (guard {args.max_amplitude})"
        )
    if args.hold_s < 0:
        return (), (), f"--hold-s must be >= 0, got {args.hold_s}"
    if getattr(args, "activate_arms", False):
        if args.activation_timeout_s <= 0:
            return (), (), (
                f"--activation-timeout-s must be > 0, got {args.activation_timeout_s}"
            )
        needed = args.activation_timeout_s + ACTIVATION_HOLD_MARGIN_S
        if args.hold_s < needed:
            return (), (), (
                f"--hold-s {args.hold_s:g} is too short for --activate-arms: the "
                f"whole activation has to fit inside the hold ({ACTIVATION_STREAM_LEAD_S:g}s "
                f"of stream before the switch + up to --activation-timeout-s "
                f"{args.activation_timeout_s:g}s of polling + 1s margin), because the "
                f"wiggle must not start on an inactive controller. Pass --hold-s "
                f"{needed:g} or more (or a smaller --activation-timeout-s)"
            )
    wiggle = {a.strip() for a in args.arms.split(",") if a.strip()}
    if not wiggle or not wiggle <= {"left", "right"}:
        return (), (), f"--arms must be a subset of left,right; got {sorted(wiggle)}"
    try:
        joints = parse_joints(args.joints)
    except ValueError as exc:
        return (), (), str(exc)
    return wiggle, joints, None


def run_probe(
    args,
    wiggle,
    joints,
    publisher,
    collector,
    left0,
    right0,
    controllers=None,
    csv_path=None,
    clock=time.monotonic,
    sleep=time.sleep,
    out=_say,
) -> dict:
    """The publish/measure loop, with everything ROS-shaped injected.

    `publisher`, `collector` and `controllers` arrive as arguments so the
    ORDER this function enforces — hold, activate, wiggle, return, deactivate,
    stop — can be asserted with no rclpy and no robot. That order is the whole
    safety argument (`camelo/runner/real_arms.py`): `safe_stop()` before the
    controller is inactive is a 2.0 s gap on an ACTIVE controller, which takes
    the arm launch down with it, and a live smoke test would only ever reveal
    that by breaking something expensive.

    `controllers` is None unless `--activate-arms`, and with it None every
    line below behaves exactly as it did before the flag existed.

    The CSV is written HERE, first thing in the teardown, rather than by the
    caller after this returns: on the rig (2026-09-02, T1(a)) a BrokenPipeError
    from the first print after Ctrl+C propagated past the caller's write and
    the rows were lost. Nothing between the last row and the file is allowed
    to be able to fail.
    """
    activation = (
        None if controllers is None
        else HoldActivation(controllers, timeout_s=args.activation_timeout_s)
    )
    rows = []
    max_meas = {"left": 0.0, "right": 0.0}
    t_wall0 = clock()
    next_log = 0.0
    interrupted = False
    csv_note = None
    t_act = None
    t_deact = None
    deactivate_confirmed = None
    offset_at_act = None
    final = None
    try:
        while clock() - t_wall0 < args.hold_s + args.seconds:
            t = clock() - t_wall0
            # The hold lasts until --hold-s AND until the controller we are
            # activating ourselves says it is active: the wiggle must never
            # be the first thing an arm hears.
            holding = t < args.hold_s or (
                activation is not None and not activation.done
            )
            if holding:
                # Zero-delta stand-in: the captured start pose, unchanged.
                # This is the sample on_activate needs, and the pose its
                # g0 is captured from.
                offset = 0.0
            else:
                phase_t = t - args.hold_s
                offset = args.amplitude * math.sin(
                    2.0 * math.pi * phase_t / args.period
                )
            left = left0.copy()
            right = right0.copy()
            if "left" in wiggle:
                left[joints] += offset
            if "right" in wiggle:
                right[joints] += offset
            if args.no_gripper or holding:
                opening = 1.0
            else:
                phase = 0.5 + 0.5 * math.cos(2.0 * math.pi * t / args.gripper_period)
                opening = 1.0 - args.gripper_span * (1.0 - phase)
            publisher.publish_arms(left, right)
            publisher.publish_grippers(opening, opening)

            if activation is not None and not activation.done:
                # AFTER this tick's publish, always: the switch and every
                # list_controllers poll are blocking service calls, and the
                # stream the controller is watching must not pause for the
                # question we are asking it.
                if activation.step(t):
                    t_act, offset_at_act = t, offset
                    out(f"ACTIVATED at t={t:.2f}s (probe clock), "
                        f"offset(t_act)={offset:+.4f} rad — 0 by construction: "
                        f"activation happens inside the zero-offset hold, which "
                        f"is what makes the controller's g0 == q0", flush=True)

            if t >= next_log:
                next_log = t + 1.0
                cur = collector.get_obs(require_images=False)
                if cur is not None:
                    ml = cur.state[C.S_LEFT_ARM] - left0
                    mr = cur.state[C.S_RIGHT_ARM] - right0
                    gl, gr = float(cur.state[C.S_LEFT_GRIP]), float(cur.state[C.S_RIGHT_GRIP])
                    if not holding:
                        # The HOLD must not count toward "the arm
                        # followed": it commands zero offset, so any
                        # motion during it is drift, not tracking.
                        max_meas["left"] = max(
                            max_meas["left"], float(np.max(np.abs(ml[joints])))
                        )
                        max_meas["right"] = max(
                            max_meas["right"], float(np.max(np.abs(mr[joints])))
                        )
                    rates = collector.rates()
                    row = {
                        "t": round(t, 2),
                        "phase": "hold" if holding else "wiggle",
                        "cmd_offset": round(offset, 4),
                        **{
                            f"left_j{j + 1}": round(float(ml[j]), 4) for j in joints
                        },
                        **{
                            f"right_j{j + 1}": round(float(mr[j]), 4) for j in joints
                        },
                        "cmd_grip_open": round(opening, 3),
                        "left_grip": round(gl, 3), "right_grip": round(gr, 3),
                        "jsl_hz": round(rates.get("joint_states_left", 0.0), 1),
                        "jsr_hz": round(rates.get("joint_states_right", 0.0), 1),
                        # Appended at the END so every column an existing
                        # reader knows keeps its position. The gripper's own
                        # joint_states is a separate topic from the arm's: a
                        # gripper that never moves and a gripper whose
                        # feedback never arrives look identical without it.
                        "gjsl_hz": round(rates.get("gripper_js_left", 0.0), 1),
                        "gjsr_hz": round(rates.get("gripper_js_right", 0.0), 1),
                    }
                    rows.append(row)
                    meas_l = " ".join(
                        f"j{j + 1}={row[f'left_j{j + 1}']:+.4f}" for j in joints
                    )
                    meas_r = " ".join(
                        f"j{j + 1}={row[f'right_j{j + 1}']:+.4f}" for j in joints
                    )
                    out(f"t={row['t']:6.1f} [{row['phase']}] "
                        f"cmd={row['cmd_offset']:+.4f} "
                        f"meas L({meas_l}) R({meas_r}) "
                        f"grip cmd={row['cmd_grip_open']:.2f} "
                        f"L={row['left_grip']:.2f} R={row['right_grip']:.2f} "
                        f"js_hz L={row['jsl_hz']} R={row['jsr_hz']} "
                        f"gjs_hz L={row['gjsl_hz']} R={row['gjsr_hz']}", flush=True)
            sleep(1.0 / args.rate)
    except KeyboardInterrupt:
        interrupted = True
        out("interrupted — returning to start pose and summarising", flush=True)
    finally:
        # MEASURED on the rig 2026-09-02 (t1r_123718): activation AND
        # deactivation both worked, and then the operator's reflex SECOND
        # Ctrl+C — after the window `deactivate_controllers_guarded` defers
        # on its own — hit the "settle" sleep as a bare KeyboardInterrupt and
        # walked straight out of this function, skipping the summary and
        # exiting via a traceback instead of exit 130. Deferring SIGINT for
        # only the deactivation was never enough: the CSV write, the
        # provisional summary and the final observation are evidence too,
        # and a second Ctrl+C anywhere in this block must not be able to
        # skip any of them. So the deferral now spans the whole teardown,
        # from here to `collector.close()` — everything this function still
        # owes the operator regardless of how many more times they press
        # Ctrl+C. `publisher.close()` is deliberately outside it: it is the
        # very last thing, cancelling the keep-alive timers, and owes
        # nothing to a signal that has already been ignored for the rest of
        # the teardown.
        out("teardown running — Ctrl+C ignored until the evidence is written",
            flush=True)
        with sigint_deferred() as deferred:
            if not deferred:
                out("note: SIGINT could not be deferred for the teardown "
                    "(not the main thread)")
            # The rows FIRST, before a print, a service call or a robot gets
            # a chance to fail: they exist only in memory until this line
            # runs.
            csv_note = write_rows(csv_path, rows)
            # And a PROVISIONAL verdict right behind it — computed from
            # max_meas as measured so far, with no deactivation and no final
            # observation yet (both None below) — so a hard kill anywhere
            # later in this block (a killed process cannot honour a deferred
            # signal) still leaves a verdict on disk, not just rows. The
            # final call to `summarise()` overwrites this with the real one.
            write_summary(csv_path, [
                provisional_summary_header(args, wiggle, interrupted),
                *([csv_note] if csv_note else []),
                *build_summary_lines(
                    args, wiggle, max_meas, t_act, offset_at_act,
                    None, None, None, left0, right0,
                    activated_arms=None if controllers is None else controllers.arms,
                ),
            ])
            # Then each step on its own, in priority order. None of them may
            # be able to skip the next: on the rig (T1(b)) a single raising
            # publish took the deactivation, the safe_stop and the evidence
            # with it.
            teardown_step(out, "start-pose republish",
                          lambda: publisher.publish_arms(left0, right0))
            teardown_step(out, "gripper release",
                          lambda: publisher.publish_grippers(1.0, 1.0))
            # Deactivate BEFORE safe_stop, on every path including the
            # exception one, and only if we are the side that switched it on
            # (an operator's own activate_arms.py stays the operator's).
            # Neither optional nor a flag: a controller THIS process
            # switched on, whose stream THIS process is about to stop, has
            # to come down first.
            # `--no-deactivate-on-exit` used to exist here and named nothing
            # but the way to kill the arm launch.
            if activation is not None and activation.switched_at is not None:
                t_deact = clock() - t_wall0
                deactivate_confirmed = teardown_deactivate(publisher, controllers, out)
                out(f"DEACTIVATED at t={t_deact:.2f}s (probe clock), "
                    f"list_controllers confirmed={deactivate_confirmed}", flush=True)
            # Only now: stopping the stream on an ACTIVE controller is what
            # shuts the whole arm launch down.
            teardown_step(out, "safe_stop", publisher.safe_stop)
            teardown_step(out, "settle", lambda: sleep(0.5))
            final = teardown_step(out, "final observation",
                                  lambda: collector.get_obs(require_images=False))
            teardown_step(out, "collector close", collector.close)
        # LAST: cancel the keep-alive/pedal timers now that nothing will
        # publish again in this process. MEASURED on the rig 2026-09-02
        # (T1(c)): rclpy printed "The following exception was never
        # retrieved: cannot use Destroyable because destruction was
        # requested" twice at exit, once per timer — a callback already
        # queued on the executor's worker pool can still race
        # `ros_session`'s destroy_node(). This is not `safe_stop()`
        # because `safe_stop()` also runs mid-episode elsewhere in the
        # codebase, where the next phase's command stream still needs
        # both timers alive.
        teardown_step(out, "publisher close", publisher.close)
    return {
        "rows": rows,
        "csv_note": csv_note,
        "max_meas": max_meas,
        "interrupted": interrupted,
        "final": final,
        "t_act": t_act,
        "t_deact": t_deact,
        "deactivate_confirmed": deactivate_confirmed,
        "offset_at_act": offset_at_act,
    }


def summarise(args, result, wiggle, csv_path, left0, right0,
              activated_arms=None, out=_say) -> int:
    """Build every post-run line, put it on DISK, then print it. Exit code.

    That order is the lesson from T1(a): the terminal is the least durable
    place this run's verdict can live, and on 2026-09-02 it was the only
    one. `<--csv>.summary.txt` is written before anything is printed, and
    `_say` cannot raise, so neither a dead pipe nor a full disk can now cost
    the operator the verdict — or turn Ctrl+C into a mystery exit code.
    """
    lines = []
    if result.get("csv_note"):
        lines.append(result["csv_note"])
    lines.extend(build_summary_lines(
        args, wiggle, result["max_meas"], result["t_act"], result["offset_at_act"],
        result["t_deact"], result.get("deactivate_confirmed"), result["final"],
        left0, right0, activated_arms=activated_arms,
    ))

    write_summary(csv_path, [
        f"# sine_probe_real summary (interrupted={result['interrupted']}, "
        f"amplitude={args.amplitude}, joints={args.joints}, arms={sorted(wiggle)})",
        *lines,
    ])
    for line in lines:
        out(line)
    return 130 if result["interrupted"] else 0


def main() -> int:
    args = build_parser().parse_args()
    setup_logging()

    wiggle, joints, error = validate(args)
    if error is not None:
        print(error)
        return 2

    # C-11 (docs/realdata/16 R-38): the runner's activation client needs
    # controller_manager_msgs, absent from both rig environments on
    # 2026-09-02. Refuse before creating a DDS participant.
    from camelo.ros.real_imports import missing_real_imports, report_real_imports

    if missing_real_imports():
        print(report_real_imports())
        return 4

    from camelo.ros.command_publisher import CommandPublisher
    from camelo.ros.obs_collector import ObsCollector
    from camelo.ros.session import ros_session
    from camelo.runner.episode_runner import wait_for_obs

    topics = topics_for("real")
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with ros_session("camelo_sine_probe_real", world="real") as node:
        collector = ObsCollector(node, camera_keys=[], topics=topics)
        publisher = CommandPublisher(node, topics=topics)
        obs = wait_for_obs(collector, require_images=False)
        left0 = obs.state[C.S_LEFT_ARM].copy()
        right0 = obs.state[C.S_RIGHT_ARM].copy()
        if not (np.all(np.isfinite(left0)) and np.all(np.isfinite(right0))):
            print(f"start pose has NaN — refusing to publish. left={left0} right={right0}")
            collector.close()
            return 3
        grip0 = (float(obs.state[C.S_LEFT_GRIP]), float(obs.state[C.S_RIGHT_GRIP]))
        print(f"start pose left ={np.round(left0, 4).tolist()}")
        print(f"start pose right={np.round(right0, 4).tolist()}")
        print(f"gripper open-fractions at start left/right={grip0}")
        print(f"command topics: {topics.left_arm_cmd} {topics.right_arm_cmd} "
              f"{topics.left_gripper_cmd} ({topics.gripper_cmd_kind})")
        print(f"wiggling {sorted(wiggle)} +/-{args.amplitude} rad on joints "
              f"{[j + 1 for j in joints]}, period {args.period}s, for "
              f"{args.seconds:.0f}s after a {args.hold_s:.0f}s zero-offset hold; "
              f"gripper "
              f"{'HELD OPEN' if args.no_gripper else f'1.0..{1.0 - args.gripper_span:.2f} open'}")
        controllers = None
        if args.activate_arms:
            # Refuse BEFORE creating any service client: nothing publishes
            # across a blocking controller_manager call except the
            # publisher's keep-alive timer, and the first switch would
            # already be an activation with no stream behind it.
            gap = keepalive_gap_report(publisher, topics)
            if gap is not None:
                print(gap)
                collector.close()
                return 4
            print("keep-alive 10 Hz covers the activation switch: "
                  "controller_manager calls block per side and publish "
                  "nothing themselves; the publisher's timer keeps the stream "
                  "alive across them on the executor's spin thread")
            # The runner's convention, mirrored: only the arms this run
            # drives get switched. An arm outside --arms is the operator's
            # (or the station's) business, exactly as in run_policy.py.
            from camelo.ros.arm_activation import ArmControllers

            controllers = ArmControllers(
                node, arms=tuple(a for a in ("left", "right") if a in wiggle)
            )
        if args.hold_s > 0 and controllers is None:
            print(f"HOLD: publishing the start pose with zero offset for "
                  f"{args.hold_s:.0f}s — activate the controllers NOW "
                  f"(activate_arms.py); on_activate captures q0/g0 during this "
                  f"window, which is what makes g0 == q0", flush=True)
        elif controllers is not None:
            print(f"HOLD: publishing the start pose with zero offset for "
                  f"{args.hold_s:.0f}s and activating "
                  f"{list(controllers.arms)} joint_impedance_controller from "
                  f"here after {ACTIVATION_STREAM_LEAD_S:.0f}s of stream "
                  f"(timeout {args.activation_timeout_s:.0f}s). Do NOT run "
                  f"activate_arms.py; keep a hand on the E-stop. The wiggle "
                  f"starts only once list_controllers says active", flush=True)

        result = run_probe(
            args, wiggle, joints, publisher, collector, left0, right0,
            controllers=controllers, csv_path=csv_path,
        )
        return summarise(
            args, result, wiggle, csv_path, left0, right0,
            activated_arms=None if controllers is None else controllers.arms,
        )


if __name__ == "__main__":
    sys.exit(main())
