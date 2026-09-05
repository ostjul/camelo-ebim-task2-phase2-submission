"""The real-robot arm choreography: verify -> hold -> activate -> deactivate.

Nothing in here commands a POSE. That is the whole design: on the real rig
`recenter_arms()` would drive the sim's `ARM_READY_POSE` into two Franka
arms that are already where the operator parked them, so the start pose is
*verified* and never *reached*, and the only thing this module ever puts on
the wire before the policy speaks is the arms' own measured pose.

Why the pose gets published at all — MEASURED on the rig 2026-09-01, the
companion's `joint_impedance_controller`:

  * refuses `on_activate` unless a valid GELLO sample arrived within 2.0 s,
    and captures `q0` (robot) / `g0` (command) at that instant;
  * shuts the entire arm launch down (`rclcpp::shutdown()`) if the stream
    stops for 2.0 s while it is ACTIVE.

So the order is forced. The hold publishes the measured pose (a zero-delta
GELLO stand-in) BEFORE activation, which both satisfies the liveness gate
and makes `g0 == q0` — the reference `--arm-command-frame gello` needs, and
which is otherwise unknowable from this side. Not knowable *exactly*, at
that: `on_activate` runs inside the controller and we see it only at the
next `list_controllers` poll, so the reference is BOUNDED rather than
verified — the publisher checks that the published stream and the arm both
stood still across that whole window (`camelo/control/arm_stream.py`).

Deactivation happens the same way in reverse: the stream stays up across
the switch, and only once `list_controllers` confirms the controller is no
longer active does the publisher stop.

The service calls live in `camelo/ros/arm_activation.py` (one participant,
the runner's own node). This module is the sequence, and the pure pose
comparison below is unit-tested with no ROS.
"""

from __future__ import annotations

import contextlib
import logging
import signal
import time

import numpy as np

from camelo import contracts as C
from camelo.control.arm_stream import (
    ACTIVATION_WINDOW_MARGIN_S,
    MIN_ACTIVATION_PUBLISH_HZ,
    PublishCadenceTooLow,
)
from camelo.control.state_age import StaleStateError, stale_state_report
from camelo.runner.recenter import StartPoseNotReached

log = logging.getLogger(__name__)

#: Default `--start-pose-tol`. Same order as `recenter.ARM_SETTLE_TOL_RAD`:
#: below the arms' own steady-state wobble nothing would ever pass.
START_POSE_TOL_RAD = 0.10
#: How often `list_controllers` is polled while holding. This is also the
#: worst-case age of the activation we react to — `on_activate` can fire
#: immediately after a poll — so it sets the drift guard's window.
ACTIVATION_POLL_S = 0.5
#: How long `list_controllers` may take to agree with a switch we just asked
#: for, in EITHER direction. One number on purpose: the question ("did the
#: controller actually change state?") and the answer's cost are the same
#: going in as coming out.
SWITCH_CONFIRM_TIMEOUT_S = 10.0
#: `wait_for_services` budget. All of them before any switch — a service
#: discovered mid-switch is the burst that can fault a live FCI loop.
SERVICE_TIMEOUT_S = 30.0
#: Seconds of command stream that must precede the activation switch:
#: `on_activate` refuses without a valid sample inside its 2.0 s window, and
#: one publish ago is not a stream.
ACTIVATION_STREAM_LEAD_S = 1.0
ACTIVE_STATE = "active"


def warm_up_backend(backend, obs, task: str | None = None) -> dict:
    """`reset` + ONE discarded inference, BEFORE the controller switch.

    MEASURED on the rig 2026-09-02 (the first T6 attempt, docs/realdata/16
    U-28): with `--backend remote` the first `infer()` of a run took
    **1.443 s** where every later one took 0.33–0.53 s — model warm-up,
    allocator, the first large image payload over the websocket. It was paid
    AFTER activation, so the controller's first 1.6 s of life had nothing but
    keep-alive repeats of a pose nobody had re-decided, and it faulted.

    The fix is ordering, not speed: pay the first inference while nothing is
    active and nothing can be hurt by it, then activate, then run a rollout
    whose first inference is already steady-state. The chunk is DISCARDED —
    it never reaches the executor and never reaches the wire, so the
    executor's first real chunk still starts at the rollout's own t_sim 0.

    If this raises (server down, websocket timeout) the run ends here, before
    any switch — which is the whole point of running it before the switch.

    The warm-up's own RTT is trimmed back off the backend's sample lists: a
    1.4 s outlier would become `rtt_max_s` and misreport the steady state the
    rollout actually saw. It is not lost — it is returned as
    ``warmup_infer_s`` and logged, which is where a one-off belongs.
    """
    lengths = {}
    # RemoteBackend's own per-request samples. `wire_bytes`/`encode_s` are
    # trimmed for the same reason as `rtts`: the warm-up's encode is the one
    # that pays for `describe()`'s extra baseline JPEG, so leaving it in
    # would inflate `wire_encode_s` for the whole run.
    for name in ("rtts", "server_infer_s", "wire_bytes", "encode_s"):
        seq = getattr(backend, name, None)
        if isinstance(seq, list):
            lengths[name] = len(seq)
    t0 = time.monotonic()
    if task is not None:
        backend.reset(task)
    reset_s = time.monotonic() - t0
    t1 = time.monotonic()
    backend.infer(obs)  # discarded ON PURPOSE — see the docstring
    infer_s = time.monotonic() - t1
    for name, before in lengths.items():
        del getattr(backend, name)[before:]
    log.info(
        "backend warm-up before activation: reset %.3f s, first inference "
        "%.3f s (chunk discarded, nothing published, no controller active)",
        reset_s,
        infer_s,
    )
    return {"warmup_reset_s": float(reset_s), "warmup_infer_s": float(infer_s)}


class HoldActivation:
    """Activate from INSIDE a caller's own publishing loop, one tick at a time.

    `RealArmSession.hold_until_activated` owns both the publishing and the
    activation, which is right for the runner (the hold has no other job).
    A probe — `scripts/sine_probe_real.py` — already has a loop with its own
    waveform and its own clock, and the one thing it must never do is stop
    publishing to wait for a controller. So the same state machine is
    offered here as a `step(t)` that publishes nothing, sleeps never, and
    returns True once every selected arm reports ``active``:

        activation = HoldActivation(controllers)
        while ...:
            publisher.publish_arms(...)       # the caller's stream, unbroken
            if not activated and activation.step(t):
                activated = True

    ``t`` is the caller's own clock, in seconds from the first publish — the
    lead-in and the timeout are both measured on it, so the caller's log
    timestamps and this one cannot disagree.
    """

    def __init__(
        self,
        controllers,
        timeout_s: float = SWITCH_CONFIRM_TIMEOUT_S,
        poll_s: float = ACTIVATION_POLL_S,
        lead_s: float = ACTIVATION_STREAM_LEAD_S,
        service_timeout_s: float = SERVICE_TIMEOUT_S,
    ):
        self.controllers = controllers
        self.arms = tuple(getattr(controllers, "arms", ()))
        self.timeout_s = float(timeout_s)
        self.poll_s = float(poll_s)
        self.lead_s = float(lead_s)
        self.service_timeout_s = float(service_timeout_s)
        self.switched_at: float | None = None
        self.active_at: float | None = None
        self.states: dict = {}
        self._next_poll = 0.0

    @property
    def done(self) -> bool:
        return self.active_at is not None

    def step(self, t: float) -> bool:
        """One tick at caller-clock time ``t``; True once every arm is active.

        Raises `TimeoutError` rather than returning False forever: the
        caller's next phase (a wiggle, a rollout) assumes an ACTIVE
        controller, and running it against an inactive one produces the most
        expensive answer there is — a probe that says "the arm never moved"
        because nobody was listening.
        """
        if self.active_at is not None:
            return True
        if self.switched_at is None:
            if t < self.lead_s:
                return False
            # Everything below BLOCKS, per side and one side after the other:
            # `wait_for_services`, then `switch`, then `states()` — up to
            # `call_timeout_s` (5 s) each, with no publish of our own in
            # between. With `--arms left,right` the left controller can
            # therefore be ACTIVE while the right side's call is still in
            # flight, which is exactly the 2.0 s liveness gap that shuts an
            # arm launch down. What covers it is the publisher's keep-alive
            # (10 Hz) on its OWN thread — not this loop, and not the ROS
            # executor, which is itself blocked under camera load (U-28). The
            # caller asserts `publisher.keepalive_active()` before switching.
            if not self.controllers.wait_for_services(self.service_timeout_s):
                raise TimeoutError(
                    "controller_manager services not discovered: "
                    f"{self.controllers.service_names()}"
                )
            log.info("activating %s joint_impedance_controller", list(self.arms))
            self.controllers.switch(activate=True)
            self.switched_at = float(t)
            self._next_poll = float(t)
        if t < self._next_poll:
            return False
        self._next_poll = float(t) + self.poll_s
        self.states = self.controllers.states()
        if self.states and all(
            state == ACTIVE_STATE for state in self.states.values()
        ):
            self.active_at = float(t)
            log.info(
                "controllers ACTIVE on %s at t=%.2fs (%.2fs after the switch)",
                list(self.arms), self.active_at, self.active_at - self.switched_at,
            )
            return True
        if t - self.switched_at >= self.timeout_s:
            raise TimeoutError(
                f"joint_impedance_controller not active on {list(self.arms)} "
                f"within {self.timeout_s:.0f}s of the switch. States: "
                f"{self.states}. The command stream was up the whole time, so "
                "this is a refusal, not a liveness gap — check the joint names "
                "and that the arm launch is running"
            )
        return False


def deactivate_controllers(
    publisher,
    controllers,
    timeout_s: float = SWITCH_CONFIRM_TIMEOUT_S,
) -> bool:
    """Deactivate while the stream is STILL up, and confirm it took.

    Deliberately does not stop publishing: every caller stops in its own
    `finally`, and the one ordering that must never happen is `safe_stop()`
    before the controller is inactive — that is a 2.0 s gap on an ACTIVE
    controller, i.e. `rclcpp::shutdown()` and the whole arm launch with it.
    """
    # The sides come from the controllers themselves: an `arms` argument here
    # would only ever describe the switch, never select it, and a log line
    # that can disagree with what happened is worse than no log line.
    log.info(
        "deactivating %s joint_impedance_controller",
        list(getattr(controllers, "arms", ())),
    )
    # One explicit publish before the (blocking) service call, so the stream
    # is fresh even with the keep-alive turned off.
    publisher.repeat_last_arm_command()
    controllers.switch(activate=False)
    confirmed = controllers.wait_until(
        active=False,
        timeout_s=timeout_s,
        on_poll=publisher.repeat_last_arm_command,
    )
    if confirmed:
        log.info("joint_impedance_controller inactive — stopping commands")
    else:
        log.error(
            "joint_impedance_controller did NOT confirm inactive within %.0fs "
            "(%s). Stopping the command stream anyway; the controller will "
            "shut its arm launch down 2 s later, which is the safe failure "
            "but needs a station restart",
            timeout_s,
            controllers.states(),
        )
    return bool(confirmed)


def start_pose_errors(measured_left, measured_right, target) -> dict:
    """Per-joint |measured - target| for both arms, plus the maxima.

    ``target`` is ``(left(7), right(7))``. Pure numpy — the verdict is the
    caller's, and the per-joint rows are returned rather than summarised
    because a single max is exactly the aggregate that hides which joint is
    off (AGENTS.md, "the summary lies"). Always computed for BOTH arms
    regardless of which are driven — `verify_start_pose` is the one that
    decides which side's error can refuse a run.
    """
    target_left = np.asarray(target[0], dtype=np.float64).ravel()
    target_right = np.asarray(target[1], dtype=np.float64).ravel()
    left = np.asarray(measured_left, dtype=np.float64).ravel()
    right = np.asarray(measured_right, dtype=np.float64).ravel()
    for name, arm in (
        ("left", left), ("right", right),
        ("target left", target_left), ("target right", target_right),
    ):
        if arm.shape != (7,):
            raise ValueError(f"{name} arm must be 7 joints, got {arm.shape}")
    errors_left = np.abs(left - target_left)
    errors_right = np.abs(right - target_right)
    both = np.concatenate([errors_left, errors_right])
    return {
        "left": [float(v) for v in errors_left],
        "right": [float(v) for v in errors_right],
        "max_left": float(np.max(errors_left)),
        "max_right": float(np.max(errors_right)),
        "max": float(np.max(both)) if np.all(np.isfinite(both)) else float("inf"),
        "worst_joint": (
            "left" if np.max(errors_left) >= np.max(errors_right) else "right",
            int(np.argmax(errors_left if np.max(errors_left) >= np.max(errors_right)
                          else errors_right)) + 1,
        ),
    }


def format_start_pose_errors(errors: dict, tol: float, arms=("left", "right")) -> str:
    def row(side):
        held = " HELD, not verified" if side not in arms else ""
        return " ".join(
            f"j{i + 1}={v:+.4f}" for i, v in enumerate(errors[side])
        ) + held

    side, joint = errors["worst_joint"]
    return (
        f"max {errors['max']:.4f} rad (tol {tol:.4f}, worst {side} j{joint}); "
        f"left [{row('left')}] right [{row('right')}]"
    )


def verify_start_pose(
    measured_left,
    measured_right,
    target,
    tol: float = START_POSE_TOL_RAD,
    require: bool = True,
    arms=("left", "right"),
) -> dict:
    """Compare the MEASURED arms against the requested start pose.

    Verify-only, by construction: nothing is commanded, here or by any caller
    on the real map. Raises `StartPoseNotReached` when the arms are further
    than ``tol`` from the pose and ``require`` — a rollout that starts
    somewhere else is not the run the operator asked for, and on a real robot
    the difference between "0.3 rad off" and "on pose" is whether the first
    chunk drives through the table.

    ``arms`` restricts the refusal to the DRIVEN sides (default both, so an
    omitted ``arms`` behaves exactly as before). A side outside ``arms`` is
    HELD at its measured pose for the whole run (`RealArmSession.held_arms`)
    and never moves under policy or GELLO command, so its distance from
    ``target`` cannot make the run unsafe — it is reported at INFO, not
    refused on. ``errors["max"]`` itself is still the max over both arms
    (`start_pose_errors` never changes); this recomputes the max over only
    the selected sides for the pass/fail decision and the message.
    """
    errors = start_pose_errors(measured_left, measured_right, target)
    # np.max, not the builtin: it propagates NaN regardless of which operand
    # carries it, so an unwired driven arm still refuses no matter the order
    # `arms` lists the sides in (the builtin's NaN result is order-dependent).
    driven_max = float(np.max([errors[f"max_{side}"] for side in arms]))
    message = format_start_pose_errors(errors, tol, arms=arms)
    for side in ("left", "right"):
        if side not in arms:
            log.info(
                "start pose: %s arm is held, not verified (max %.4f rad from "
                "target)", side, errors[f"max_{side}"],
            )
    if driven_max <= tol:
        log.info("start pose OK: %s", message)
        return errors
    if not require:
        log.warning(
            "start pose MISMATCH (--no-start-pose-check, running anyway): %s", message
        )
        return errors
    raise StartPoseNotReached(
        f"the arms are not at the requested start pose: {message}. Nothing is "
        "commanded on the real robot — move the arms there (teleop / the "
        "corpus frame-0 pose the file was lifted from), widen "
        "--start-pose-tol, or pass --no-start-pose-check to run off-pose "
        "deliberately"
    )


def _log_line(text: str, flush: bool = True) -> None:
    """Default `out` for the guarded teardown: the log, not stdout.

    The runner has no stdout contract; `scripts/sine_probe_real.py` passes
    its own BrokenPipe-safe `_say` instead. Same words either way, so the
    two teardowns cannot drift into describing different procedures.
    """
    log.info("%s", text)


@contextlib.contextmanager
def sigint_deferred():
    """Ignore SIGINT for the duration of the arm teardown.

    A second Ctrl+C — the operator's reflex when the first one looks like it
    did nothing — used to land inside the deactivation service call, unwind
    past it, and leave the caller's `finally` to stop the command stream on a
    controller that was still ACTIVE: the one sequence that takes the arm
    launch down (2.0 s without a sample -> `rclcpp::shutdown()`). It also
    skipped everything after the switch in the caller's own teardown. The
    window is short and bounded (one switch plus a confirm timeout), so the
    signal is dropped rather than deferred to after it.

    Yields False when it could not be installed (not the main thread), so the
    caller can say so rather than quietly promising a guarantee it lacks.
    """
    previous = None
    installed = False
    try:
        previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
        installed = True
    except (ValueError, OSError, AttributeError):
        pass
    try:
        yield installed
    finally:
        if installed:
            with contextlib.suppress(ValueError, OSError):
                signal.signal(signal.SIGINT, previous)


def deactivate_controllers_guarded(
    publisher,
    controllers,
    timeout_s: float = SWITCH_CONFIRM_TIMEOUT_S,
    out=None,
) -> bool:
    """`deactivate_controllers` that a Ctrl+C cannot interrupt.

    Used by EVERY real-robot teardown — `RealArmSession.shutdown()` and the
    sine probe — because the failure it prevents is identical in both: the
    switch never lands, the caller's `finally` stops the stream anyway, and
    the controller takes its arm launch down 2.0 s later. Anything other than
    KeyboardInterrupt still propagates: a service that answers with an error
    is the caller's to report, and swallowing it here would turn a failed
    deactivation into a silent one.
    """
    say = _log_line if out is None else out
    say("deactivating — Ctrl+C ignored until the controller is down "
        "(stopping the stream on an ACTIVE controller shuts the whole arm "
        "launch down)")
    with sigint_deferred() as deferred:
        if not deferred:
            say("note: SIGINT could not be deferred for the teardown "
                "(not the main thread)")
        try:
            return deactivate_controllers(publisher, controllers, timeout_s)
        except KeyboardInterrupt:
            # Only reachable when the guard could not be installed. One more
            # attempt; a second interrupt there propagates and the caller's
            # own finally still stops the stream.
            say("Ctrl+C during the deactivation — finishing it anyway")
            return deactivate_controllers(publisher, controllers, timeout_s)


def keepalive_gap_report(publisher, topics) -> str | None:
    """Refuse to activate when nothing publishes across the switch.

    `ArmControllers` switches and polls one side after the other, blocking up
    to `call_timeout_s` (5 s) per call and publishing nothing of its own; with
    `--arms left,right` the LEFT controller is already ACTIVE while the right
    side's call is in flight. Nothing in any caller's loop publishes during
    that window — the publisher's keep-alive thread does. Without it the very
    act of activating the arms is a >2.0 s liveness gap on an active
    controller, and `--keepalive-hz 0` is a real flag.

    Returns the message to refuse with, or None when it is covered.
    """
    if topics.world != C.WORLD_REAL:
        return (
            f"arm activation is real-robot only, and the topic map says "
            f"world={topics.world!r}"
        )
    if not publisher.keepalive_active():
        return (
            "activating the arms needs the command publisher's keep-alive "
            "thread, and it is OFF (--keepalive-hz 0). controller_manager "
            "calls block "
            "up to 5s per side with no publish of their own; the controller "
            "shuts its whole arm launch down after 2.0s without a sample"
        )
    return None


class RealArmSession:
    """Owns the real-robot choreography around a rollout.

    Usage (the runner's shape):

        session = RealArmSession(collector, publisher, controllers, ...)
        session.prepare()          # verify -> hold -> activated
        try:      run_rollout(...)
        finally:  session.shutdown()   # deactivate -> safe_stop

    ``controllers`` is an `arm_activation.ArmControllers` (or None, which
    turns activation waiting/switching off and leaves only the hold).
    """

    def __init__(
        self,
        collector,
        publisher,
        controllers=None,
        arms=("left", "right"),
        start_pose=None,
        start_pose_tol: float = START_POSE_TOL_RAD,
        start_pose_check: bool = True,
        wait_for_activation: bool = True,
        activate_arms: bool = False,
        deactivate_on_exit: bool = True,
        max_state_age_s: float | None = None,
        rate_hz: float = 20.0,
        hold_timeout_s: float = 600.0,
        service_timeout_s: float = SERVICE_TIMEOUT_S,
        deactivate_timeout_s: float = SWITCH_CONFIRM_TIMEOUT_S,
        min_activation_publish_hz: float = MIN_ACTIVATION_PUBLISH_HZ,
    ):
        self.collector = collector
        self.publisher = publisher
        self.controllers = controllers
        self.arms = tuple(arms)
        self.held_arms = tuple(a for a in ("left", "right") if a not in self.arms)
        self.start_pose = start_pose
        self.start_pose_tol = float(start_pose_tol)
        self.start_pose_check = bool(start_pose_check)
        self.wait_for_activation = bool(wait_for_activation)
        self.activate_arms = bool(activate_arms)
        self.deactivate_on_exit = bool(deactivate_on_exit)
        self.max_state_age_s = (
            None if max_state_age_s is None else float(max_state_age_s)
        )
        self.tick = 1.0 / float(rate_hz)
        self.hold_timeout_s = float(hold_timeout_s)
        self.service_timeout_s = float(service_timeout_s)
        self.deactivate_timeout_s = float(deactivate_timeout_s)
        self.min_activation_publish_hz = float(min_activation_publish_hz)
        self.activated = False
        self.left_gripper_hold = 1.0
        # The controller's own on_activate can fire immediately after a poll,
        # so the reference has to be stationary across a whole poll interval
        # (plus scheduling slop), not just at the instant we noticed.
        self.activation_window_s = ACTIVATION_POLL_S + ACTIVATION_WINDOW_MARGIN_S
        self.stats: dict = {}

    # -- helpers -----------------------------------------------------------
    def _measured_arms(self, obs):
        return (
            np.asarray(obs.state[C.S_LEFT_ARM], dtype=np.float64),
            np.asarray(obs.state[C.S_RIGHT_ARM], dtype=np.float64),
        )

    def _obs(self, timeout_s: float = 30.0):
        """Block until the rig groups are complete. Images are NOT waited for.

        The arms' liveness gate is what this phase serves, and waiting for
        three camera workers first would leave the controller without a
        stream for exactly as long as they take to come up. Deliberately not
        `episode_runner.wait_for_obs`: that module imports the ROS layer at
        module scope, and everything here is unit-tested with no ROS.
        """
        deadline = time.monotonic() + timeout_s
        last_log = 0.0
        while time.monotonic() < deadline:
            obs = self.collector.get_obs(require_images=False)
            if obs is not None:
                return obs
            now = time.monotonic()
            if now - last_log >= 5.0:
                last_log = now
                missing = getattr(self.collector, "missing_rig_groups", lambda: [])()
                log.info(
                    "waiting for a complete real observation: missing=%s rates=%s",
                    missing, self.collector.rates(),
                )
            time.sleep(0.05)
        missing = getattr(self.collector, "missing_rig_groups", lambda: [])()
        raise TimeoutError(
            f"no complete real observation within {timeout_s:.0f}s — missing "
            f"{missing}; rates {self.collector.rates()}. The arms, the right "
            "gripper and both wrenches all have to be live before anything is "
            "published (s27a15 has no zero default for a missing group)"
        )

    def _publish_hold(self, left, right) -> None:
        """Zero-delta stand-in: the measured pose, with NO frame mapping.

        Unmapped on purpose. `--arm-command-frame gello` needs `(q0, g0)`,
        which does not exist until activation, and the value that MAKES
        `g0 == q0` is the raw measured pose. Mapping it would be circular.
        """
        self.publisher.publish_arms_unmapped(left, right)

    # -- phases ------------------------------------------------------------
    def verify(self) -> dict:
        """Check the arms against `--start-pose` (never command it)."""
        obs = self._obs()
        left, right = self._measured_arms(obs)
        self.left_gripper_hold = float(obs.state[C.S_LEFT_GRIP])
        if self.start_pose is None:
            log.info(
                "no --start-pose on real: start-pose check skipped. Measured "
                "left=%s right=%s",
                [round(float(v), 4) for v in left],
                [round(float(v), 4) for v in right],
            )
            return {}
        errors = verify_start_pose(
            left, right, self.start_pose, self.start_pose_tol, self.start_pose_check,
            arms=self.arms,
        )
        self.stats["start_pose_max_err_rad"] = errors["max"]
        return errors

    def check_state_liveness(self) -> None:
        """Refuse to activate against a dead joint-state stack (U-27).

        The same decision `run_rollout` makes on every tick, asked ONCE here
        so the refusal lands BEFORE the switch rather than one rollout tick
        after it. `verify()` above has just proven a complete observation
        exists, but "complete" is a statement about the collector's cache,
        not about the wire: the cache is never cleared, so a stack that died
        between two runs of this process — or during a long hold — still
        assembles a full 27-dim observation out of frozen values. Only the
        receipt clock separates the two.

        No-op when the guard is off (`--max-state-age-s 0`, and every sim
        path) or when the collector predates `last_state_wall`.
        """
        if self.max_state_age_s is None:
            return
        last_state_wall = getattr(self.collector, "last_state_wall", None)
        if last_state_wall is None:
            return
        fault = stale_state_report(
            last_state_wall(), time.monotonic(), self.max_state_age_s
        )
        if fault is not None:
            log.error("%s", fault)
            raise StaleStateError(fault)

    def hold_until_activated(self) -> bool:
        """Publish the measured pose until the controller is ACTIVE on both arms.

        Returns True once every selected arm is active (or immediately, when
        activation waiting is off). The publishing never stops in between:
        that stream is the liveness gate `on_activate` refuses without.
        """
        obs = self._obs()
        left, right = self._measured_arms(obs)
        self._publish_hold(left, right)
        for side in self.held_arms:
            # An arm this run does not drive is frozen at the pose it is in
            # now, for the whole run — including through the rollout, where
            # the policy's prediction for it is discarded.
            self.publisher.set_held_arm(side, left if side == "left" else right)
            log.info("arm %s is HELD at its measured pose for the whole run", side)

        if self.controllers is None or not (
            self.wait_for_activation or self.activate_arms
        ):
            log.warning(
                "activation waiting is OFF: publishing the measured pose and "
                "starting the policy without checking that "
                "joint_impedance_controller is active. The arms will not move "
                "if it is not, and nothing here will say so"
            )
            self._capture_activation(left, right)
            return False

        if not self.controllers.wait_for_services(self.service_timeout_s):
            raise TimeoutError(
                "controller_manager services not discovered: "
                f"{self.controllers.service_names()}. All of them are waited "
                "for BEFORE any switch (activate_arms.py's rule: a new "
                "participant mid-session is a discovery burst that can fault a "
                "live FCI loop)"
            )

        switched = False
        t0 = time.monotonic()
        last_log = 0.0
        # Poll on the FIRST iteration: an arm an operator already activated
        # must not cost the run half a second of holding to notice.
        last_poll = -1.0
        while time.monotonic() - t0 < self.hold_timeout_s:
            obs = self.collector.get_obs(require_images=False)
            if obs is not None:
                left, right = self._measured_arms(obs)
            self._publish_hold(left, right)
            now = time.monotonic() - t0
            if self.activate_arms and not switched and now >= ACTIVATION_STREAM_LEAD_S:
                # >= 1 s of stream first: on_activate refuses without a
                # sample inside its 2.0 s window, and one publish ago is not
                # a stream. And the stream has to be a FAST one — the gate
                # below raises `PublishCadenceTooLow` before any switch.
                self._guard_publish_cadence()
                log.info("activating %s joint_impedance_controller", list(self.arms))
                self.controllers.switch(activate=True)
                switched = True
            if now - last_poll >= ACTIVATION_POLL_S:
                last_poll = now
                states = self.controllers.states()
                if states and all(state == "active" for state in states.values()):
                    self._capture_activation(left, right)
                    # D: from here the controller is ACTIVE and every publish
                    # until the policy's first command is a REPEAT. The
                    # publisher measures that window and folds it into
                    # `max_publish_gap_s`, where the 2.0 s fault lives.
                    self.publisher.mark_activation()
                    log.info("controllers ACTIVE on %s after %.1fs", list(self.arms), now)
                    self.activated = True
                    self.stats["activation_wait_s"] = now
                    return True
                if now - last_log >= 5.0:
                    last_log = now
                    log.info(
                        "holding the measured pose (%.0fs) — waiting for "
                        "joint_impedance_controller: %s%s",
                        now,
                        states,
                        "" if self.activate_arms else
                        " (run the station's activate_arms.py, or pass "
                        "--activate-arms)",
                    )
            time.sleep(self.tick)
        raise TimeoutError(
            f"joint_impedance_controller not active on {list(self.arms)} within "
            f"{self.hold_timeout_s:.0f}s of holding. States: "
            f"{self.controllers.states()}"
        )

    def _guard_publish_cadence(self) -> None:
        """Refuse to switch onto a command stream that has already collapsed.

        Called immediately before `switch(activate=True)` and nowhere else.
        The number it reads is the same "N publishes in the last 0.60s" the
        activation report prints — which on 2026-09-02 read 4 (6.7 Hz, the
        camera workers' drain callbacks starving the executor the keep-alive
        timer then still lived on) and was logged, not acted on. The
        controller went ACTIVE, rejected the first sample as 1.076 s old, and
        called `rclcpp::shutdown()` 0.5 s later.

        Raises `PublishCadenceTooLow` (arm_stream), so the caller's `finally`
        runs having switched nothing at all.
        """
        try:
            report = self.publisher.check_publish_cadence(
                self.activation_window_s, self.min_activation_publish_hz
            )
        except PublishCadenceTooLow as exc:
            # Recorded on the refusal path too: the sentence says what
            # happened, the stats say what was measured, and a run that was
            # refused is exactly the one whose numbers get argued about.
            self._record_cadence(getattr(exc, "report", {}))
            raise
        self._record_cadence(report)

    def _record_cadence(self, report: dict) -> None:
        if report:
            self.stats["activation_publish_hz"] = report["publish_hz"]
            self.stats["activation_publish_samples"] = report["publish_samples"]

    def _capture_activation(self, left, right) -> None:
        """Freeze (q0, g0) the moment we see the controller go active.

        `g0` is the last thing we PUBLISHED and `q0` what we last MEASURED,
        and the publisher takes both from its own ring buffer rather than
        from anything this side asserts. The window it checks is a full poll
        interval wide because that is how stale our knowledge of
        `on_activate` can be — comparing the newest publish against the value
        we published one line earlier would compare a number to itself and
        pass for every possible robot.
        """
        report = self.publisher.capture_activation_reference(
            left, right, window_s=self.activation_window_s
        )
        if report:
            self.stats["activation_ref_window_s"] = report["window_s"]
            self.stats["activation_ref_samples"] = report["samples"]
            self.stats["activation_ref_spread_rad"] = report["spread_rad"]
            self.stats["activation_q0_g0_drift_rad"] = report["delta_rad"]

    def prepare(self, warm_up=None) -> dict:
        """verify -> (optional backend warm-up) -> hold -> activated.

        ``warm_up`` is a zero-arg callable run AFTER the verify and BEFORE
        the hold — the one place in the sequence where a multi-second
        blocking call is free: nothing has been published, no controller is
        active, and the arms are exactly where the operator left them. A dict
        it returns is merged into the stats (`warm_up_backend` returns
        ``warmup_infer_s``). Anything it raises ends the run here, before the
        switch. See `warm_up_backend` for why this exists at all.
        """
        self.verify()
        self.check_state_liveness()
        if warm_up is not None:
            result = warm_up()
            if isinstance(result, dict):
                self.stats.update(result)
        self.hold_until_activated()
        return self.stats

    # -- teardown ----------------------------------------------------------
    def shutdown(self) -> dict:
        """Deactivate (stream still up), confirm, THEN stop commanding.

        Called from the runner's `finally`, so it runs on the exception path
        too — which is the path that matters: dropping the command stream on
        an ACTIVE controller is what takes the whole arm launch down.
        """
        try:
            if (
                self.deactivate_on_exit
                and self.controllers is not None
                and (self.activated or self.activate_arms)
            ):
                self.stats["deactivate_confirmed"] = (
                    deactivate_controllers_guarded(
                        self.publisher,
                        self.controllers,
                        timeout_s=self.deactivate_timeout_s,
                    )
                )
        finally:
            self.publisher.safe_stop()
        # The publisher's own telemetry belongs in the session summary too:
        # `keepalive_max_interval_s` (did the watchdog keep ITS cadence?) and
        # `activation_to_first_cmd_s` (how long the controller held a repeat
        # before the policy spoke) are both facts about this choreography,
        # and the rollout stats they also appear in do not exist on the paths
        # where the rollout never started.
        publish_stats = getattr(self.publisher, "publish_stats", None)
        if publish_stats is not None:
            with contextlib.suppress(Exception):
                self.stats.update(publish_stats())
        return self.stats
