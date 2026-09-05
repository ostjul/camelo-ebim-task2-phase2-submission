"""Publish policy commands — a drop-in GELLO stand-in.

Sim (``TopicMap.world == sim``): JointState on ``/bridge/*`` plus a pedal
token on ``/pedal/state``, matching the benchmark's gello_to_bridge.py.
Real (``world == real``): the TMR station topics in record_bag.bash —
GELLO JointState per arm, ``target_gripper_width_percent``
(**std_msgs/Float32**, 0..1 open fraction, 1.0 = open — MEASURED 2026-09-02
(T1(c)); the topic name is a misnomer — measured 2026-09-01: the rig's
``robotiq_gripper_client.py`` subscribes Float32 and the station's own
GELLO publisher publishes Float32, so a Float64 publisher is simply never
matched and the gripper never moves), and a Twist on
``/swerve_drive_controller/cmd_vel``.

Two real-robot behaviours have no sim counterpart and are OFF in sim:

  * a **keep-alive** that republishes the last arm + gripper command with a
    fresh stamp whenever nothing was published for ``1/keepalive_hz``.
    The companion's ``joint_impedance_controller`` calls
    ``rclcpp::shutdown()`` — taking the whole arm launch down — if no valid
    GELLO sample arrives for 2.0 s while it is ACTIVE, and it rejects
    samples stamped more than 0.5 s in the past. A remote-inference stall is
    therefore not a pause but a fault, and repeating the last command is what
    a real GELLO would do anyway (it publishes continuously, whether or not
    the operator moves it). Never on sim: the /bridge path has its own
    watchdog semantics and byte-identical sim behaviour is the rule.

    It runs on its OWN daemon thread, not on a ROS timer — MEASURED on the
    rig 2026-09-02 (the first T6 attempt, docs/realdata/16 U-28). As a timer
    it was dispatched by the session's `MultiThreadedExecutor`, the same
    pool that drains three camera workers' decoded frames at 20 Hz; under
    that load it fired 159 times in 30 s instead of ~300 (≈5 Hz), and the
    hold loop's own cadence had already collapsed to 6.7 Hz before
    activation. A watchdog whose period depends on the load it exists to
    survive is not a watchdog. On its own thread the cadence holds whatever
    the executor and the (blocked) main thread are doing, and
    ``keepalive_max_interval_s`` reports how well it actually held.
  * an **arm command frame**. ``robot`` (default) publishes the joint target
    as-is. ``gello`` publishes ``g = g0 + dir * (q_target - q0)``, the
    inverse of the controller's own ``q_goal = q0 + dir * (g - g0)`` map,
    with ``(q0, g0)`` captured at activation. Which one is right is a rig
    measurement (T1b), so it is a switch rather than a constant.

Gripper semantics are canonical open-fraction (1 = open) EVERYWHERE inside
camelo; the wire value is converted per world. Sim default
`REPUBLISHER_GRIPPER_INVERT=true` means the /bridge wire carries a CLOSE
fraction (docs/DGX_FINDINGS.md F-18). Real ``width_percent`` is a 0..1 open
fraction, 1.0 = open — MEASURED 2026-09-02 (T1(c)); the topic name is a
misnomer — unless CAMELO_GRIPPER_INVERT overrides. The sim pedal token
must be republished faster than the bridge's 1 s watchdog; a repeat timer
keeps the last token (or last twist) alive between policy ticks and
safe_stop() switches it to NONE / zero.
"""

from __future__ import annotations

import contextlib
import logging
import math
import threading
import time

import numpy as np
from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, String

from camelo import contracts as C
from camelo.control.arm_stream import (
    ACTIVATION_DRIFT_TOL_RAD,
    MIN_ACTIVATION_PUBLISH_HZ,
    ArmPublishHistory,
    guard_activation_reference,
    guard_publish_cadence,
)
from camelo.control.chunk_executor import Command
from camelo.ros.qos import DEFAULT_DEPTH

log = logging.getLogger(__name__)

#: MEASURED on the rig 2026-09-01: the companion's
#: ``gello_joint_directions``. See docs/CONTRACTS.md "Real robot".
DEFAULT_GELLO_JOINT_DIRECTIONS = (-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0)
ARM_COMMAND_FRAMES = ("robot", "gello")
#: Republish rate for the keep-alive. 10 Hz leaves 20x margin on the
#: controller's 2.0 s liveness gap and 5x on its 0.5 s stamp-age reject.
DEFAULT_KEEPALIVE_HZ = 10.0
#: Queue depth of the REAL arm command publishers. One, not `DEFAULT_DEPTH`.
#:
#: A command stream is newest-wins: an old target is not worth delivering,
#: and on this rig it is worse than nothing. `create_publisher(..., 10)` is
#: KEEP_LAST(10) RELIABLE, so the writer holds up to ten unacknowledged
#: samples for a slow reader; at the runner's 20 Hz that history is exactly
#: **0.5 s deep — the companion's own stamp-age reject limit**. A reader that
#: stalls for a moment therefore does not miss samples, it receives a BACKLOG
#: whose oldest entry is at or just past the limit, and every one of those is
#: rejected on arrival.
#:
#: MEASURED on the rig 2026-09-02 17:41 (diagnostic run, `--arms right`, left
#: arm HELD, left controller INACTIVE): "Rejecting GELLO state: message too
#: old (timestamp age 0.505783 s, limit 0.500 s)" 0.1 s after the right-arm
#: activation and again at "age 0.500000 s" 4.7 s later — twice in 20 s,
#: while the ACTIVE right controller (consuming at its own 1 kHz) saw none.
#: Ages pinned at 10 ticks of a 19.95 Hz stream are a queue depth, not a
#: stamp: both arms are stamped from their own `now()` read inside the same
#: `_publish_arm_wire_locked` call below, so a cached or re-sent message is
#: ruled out by construction (and `test_command_publisher_control.py` pins
#: it). Depth 1 caps what a stalled reader can be handed at one tick.
#:
#: Depth is not part of QoS compatibility, so this cannot unmatch the
#: companion's subscription the way a reliability change could. Sim keeps
#: `DEFAULT_DEPTH` — byte-identical sim behaviour is the rule.
ARM_CMD_DEPTH_REAL = 1


def clip_base_twist(
    twist: tuple[float, float, float], max_linear: float, max_angular: float
) -> tuple[tuple[float, float, float], bool]:
    """Scale the planar speed to ``max_linear`` (direction kept), clip yaw rate."""
    vx, vy, wz = twist
    clamped = False
    speed = math.hypot(vx, vy)
    if speed > max_linear:
        vx, vy = vx * max_linear / speed, vy * max_linear / speed
        clamped = True
    if abs(wz) > max_angular:
        wz = math.copysign(max_angular, wz)
        clamped = True
    return (vx, vy, wz), clamped


class CommandPublisher:
    def __init__(
        self,
        node,
        pedal_repeat_hz: float = 10.0,
        gripper_invert: bool | None = None,
        topics: C.TopicMap | None = None,
        arm_command_frame: str = "robot",
        gello_joint_directions=DEFAULT_GELLO_JOINT_DIRECTIONS,
        keepalive_hz: float = DEFAULT_KEEPALIVE_HZ,
    ):
        self.node = node
        self.topics = topics if topics is not None else C.topics_for()
        if arm_command_frame not in ARM_COMMAND_FRAMES:
            raise ValueError(
                f"arm_command_frame must be one of {ARM_COMMAND_FRAMES}, "
                f"got {arm_command_frame!r}"
            )
        self.arm_command_frame = arm_command_frame
        self.gello_directions = np.asarray(gello_joint_directions, dtype=np.float64)
        if self.gello_directions.shape != (7,):
            raise ValueError(
                "gello_joint_directions must be 7 values, got "
                f"{self.gello_directions.shape}"
            )
        # (q0, g0) per side, captured at controller activation. Set by
        # `capture_activation_reference`; required by the gello frame.
        self._activation_reference: dict[str, tuple] = {}
        # Arms this run does not drive (`--arms`): frozen at their measured
        # pose for the whole run.
        self._held_arms: dict = {}
        # Guards the last-command bookkeeping AND the publish calls it
        # describes: the keep-alive fires on a timer thread, and a check that
        # is not atomic with its own publish can double-send or skip.
        self._cmd_lock = threading.Lock()
        self._last_arm_wire: tuple | None = None
        self._last_grippers: tuple | None = None
        self._last_arm_pub: float = 0.0
        # Every arm publish over the last ARM_HISTORY_S, for the activation
        # reference guard (camelo/control/arm_stream.py).
        self._arm_history = ArmPublishHistory()
        self.keepalive_republished = 0
        self._max_publish_gap_s = 0.0
        # The keep-alive's OWN cadence (see the module docstring): how far
        # apart its wakeups actually landed. A repeat count alone cannot tell
        # "the loop was healthy so there was nothing to repeat" from "the
        # watchdog itself was not scheduled".
        self._keepalive_max_interval_s = 0.0
        self._keepalive_last_wake = 0.0
        self._keepalive_errors = 0
        # D: the activation -> first-rollout-command window. Keep-alive
        # repeats bridge it, so every gap BETWEEN publishes stays small while
        # the controller waits minutes for a first real command; the window
        # itself is the number that matters and nothing else measures it.
        self._activation_t: float | None = None
        self._activation_to_first_cmd_s: float | None = None
        self.gripper_invert = (
            C.gripper_command_invert(world=self.topics.world)
            if gripper_invert is None
            else gripper_invert
        )
        log.info(
            "command topics world=%s invert=%s (1.0 canonical = open)",
            self.topics.world,
            self.gripper_invert,
        )
        t = self.topics
        pub = node.create_publisher
        arm_depth = (
            ARM_CMD_DEPTH_REAL if t.world == C.WORLD_REAL else DEFAULT_DEPTH
        )
        self._left_arm = pub(JointState, t.left_arm_cmd, arm_depth)
        self._right_arm = pub(JointState, t.right_arm_cmd, arm_depth)
        if t.gripper_cmd_kind == "width_percent":
            # Float32 is the rig's type (module docstring). A message-type
            # mismatch never raises: the subscription simply never matches.
            self._left_grip = pub(Float32, t.left_gripper_cmd, DEFAULT_DEPTH)
            self._right_grip = pub(Float32, t.right_gripper_cmd, DEFAULT_DEPTH)
        else:
            self._left_grip = pub(JointState, t.left_gripper_cmd, DEFAULT_DEPTH)
            self._right_grip = pub(JointState, t.right_gripper_cmd, DEFAULT_DEPTH)
        self._pedal = None
        self._twist_pub = None
        self._twist_stamped = bool(getattr(t, "base_cmd_stamped", False))
        # Real base: clamp what the controller would clamp anyway, so the
        # numbers in our logs are the numbers on the wire (contracts
        # REAL_BASE_*). None = no clamp (sim pedal path has its own quantizer).
        self._base_limits = (
            (C.REAL_BASE_MAX_LINEAR_MPS, C.REAL_BASE_MAX_ANGULAR_RADPS)
            if t.world == C.WORLD_REAL
            else None
        )
        self.base_clamped = 0
        self.keepalive_base_republished = 0
        self._last_base_pub = 0.0
        if t.base_cmd_kind == "twist" and t.base_cmd:
            twist_type = TwistStamped if self._twist_stamped else Twist
            self._twist_pub = pub(twist_type, t.base_cmd, DEFAULT_DEPTH)
        elif t.base_cmd:
            self._pedal = pub(String, t.base_cmd, DEFAULT_DEPTH)
        self._token = "NONE"
        self._twist = (0.0, 0.0, 0.0)
        # The real twist wire needs a fresh stamp inside the controller's
        # 0.5 s window: repeat at REAL_BASE_CMD_HZ, not the sim pedal rate.
        base_repeat_hz = (
            C.REAL_BASE_CMD_HZ if (self._twist_pub is not None and t.world == C.WORLD_REAL)
            else pedal_repeat_hz
        )
        self._base_repeat_period = 1.0 / base_repeat_hz
        self._pedal_timer = node.create_timer(self._base_repeat_period, self._repeat_base)
        # Real only, and only ever a REPEAT of what was already commanded.
        # Its own thread, NOT a node timer — see the module docstring (U-28).
        self._keepalive_period = 1.0 / keepalive_hz if keepalive_hz > 0 else None
        self._keepalive_stop = threading.Event()
        self._keepalive_thread = None
        if t.world == C.WORLD_REAL and self._keepalive_period is not None:
            self._keepalive_thread = threading.Thread(
                target=self._keepalive_loop,
                name="camelo-keepalive",
                daemon=True,
            )
            self._keepalive_thread.start()
        elif t.world == C.WORLD_REAL:
            log.warning(
                "keep-alive is DISABLED on the real robot (--keepalive-hz 0): "
                "any gap over 2.0 s in the command stream — a slow remote "
                "inference, a blocking service call — makes the companion's "
                "joint_impedance_controller shut the whole arm launch down"
            )
        if t.world == C.WORLD_REAL and self.arm_command_frame == "robot":
            log.warning(
                "arm command frame is 'robot' (identity): the joint targets go "
                "to %s exactly as the policy predicted them, and the "
                "companion's joint_impedance_controller then applies "
                "q_goal = q0 + dir*(g - g0) with dir=%s on top. That is correct "
                "only if the corpus's action frame already matches the ROBOT "
                "frame (it does, offline: slope +1 on all 14 joints) AND the "
                "controller's transform is not applied twice. T1b decides it on "
                "the rig — pass --arm-command-frame gello to pre-invert",
                t.left_arm_cmd,
                list(self.gello_directions),
            )

    def _joint_state(self, names, positions) -> JointState:
        """A NEW message with a stamp read HERE, for every single publish.

        Load-bearing, and not only for the driven arm: the companion rejects
        any GELLO sample stamped more than 0.5 s in the past, on BOTH sides,
        and a `--arms right` run still publishes the left arm (frozen at its
        measured pose, `_side_wire`). Nothing is cached and no message is
        ever re-sent — the held side goes through this same call and gets its
        own clock read — so "the held arm's stamp is old" can only ever be a
        delivery delay, never a stale stamp (see `ARM_CMD_DEPTH_REAL`).
        """
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = list(names)
        msg.position = [float(p) for p in positions]
        return msg

    def _repeat_base(self):
        if self._twist_pub is not None:
            vx, vy, wz = self._twist
            if self._twist_stamped:
                msg = TwistStamped()
                # A NEW stamp per publish (base_nudge.py): the controller
                # ages the message by this field, never by arrival time.
                msg.header.stamp = self.node.get_clock().now().to_msg()
                twist = msg.twist
            else:
                msg = Twist()
                twist = msg
            twist.linear.x, twist.linear.y, twist.angular.z = vx, vy, wz
            self._twist_pub.publish(msg)
            self._last_base_pub = time.monotonic()
            return
        if self._pedal is not None:
            msg = String()
            msg.data = self._token
            self._pedal.publish(msg)

    def _repeat_pedal(self):
        # Name kept for the sim path / tests that poke the timer callback.
        self._repeat_base()

    # -- the arm command frame (T1b) ---------------------------------------
    def capture_activation_reference(
        self, left_q0, right_q0, left_g0=None, right_g0=None, window_s=None
    ) -> dict:
        """Freeze ``(q0, g0)`` per side — the controller's own capture instant.

        The companion captures ``q0`` (measured robot pose) and ``g0`` (the
        GELLO command it was holding) at ``on_activate`` and maps every later
        sample ``q_goal = q0 + dir * (g - g0)``. The hold phase publishes the
        MEASURED pose, so at that instant ``g0 == q0`` and both frames agree
        — which is exactly why the hold exists: it makes the reference
        knowable from this side instead of guessed.

        ``g0`` defaults to the newest value actually PUBLISHED (not to the
        caller's idea of it), and with ``window_s`` the drift guard in
        `camelo.control.arm_stream` runs over the ring buffer: the whole
        window has to be stationary, because `on_activate` may have fired up
        to one poll interval before the caller noticed. The caller passes its
        poll interval plus `ACTIVATION_WINDOW_MARGIN_S`. Returns the guard's
        report (empty without ``window_s``); raises in the ``gello`` frame,
        warns in ``robot``.
        """
        report: dict = {}
        with self._cmd_lock:
            newest = self._arm_history.newest()
            if window_s is not None:
                report = guard_activation_reference(
                    self._arm_history,
                    left_q0,
                    right_q0,
                    time.monotonic(),
                    float(window_s),
                    self.arm_command_frame,
                    ACTIVATION_DRIFT_TOL_RAD,
                )
        if newest is not None:
            # What the controller last SAW is g0. An explicit value still
            # wins (the hold-skipped path, and the unit tests that pin the
            # offset), but nothing here invents one.
            left_g0 = newest[1] if left_g0 is None else left_g0
            right_g0 = newest[2] if right_g0 is None else right_g0
        for side, q0, g0 in (
            ("left", left_q0, left_g0), ("right", right_q0, right_g0)
        ):
            if q0 is None:
                continue
            q0 = np.asarray(q0, dtype=np.float64).ravel()
            g0 = q0.copy() if g0 is None else np.asarray(g0, dtype=np.float64).ravel()
            if q0.shape != (7,) or g0.shape != (7,):
                raise ValueError(
                    f"activation reference for {side} must be 7 joints, got "
                    f"q0={q0.shape} g0={g0.shape}"
                )
            self._activation_reference[side] = (q0, g0)
            log.info(
                "activation reference %s: q0=%s g0=%s (frame=%s)",
                side,
                [round(float(v), 4) for v in q0],
                [round(float(v), 4) for v in g0],
                self.arm_command_frame,
            )
        return report

    def _arm_wire(self, side: str, target) -> list:
        """Joint target -> the numbers that go on the GELLO topic."""
        values = np.asarray(target, dtype=np.float64).ravel()
        if self.arm_command_frame == "robot":
            return [float(v) for v in values]
        reference = self._activation_reference.get(side)
        if reference is None:
            raise RuntimeError(
                f"--arm-command-frame gello needs the {side} activation "
                "reference (q0, g0), and none was captured. Only the hold ->"
                " activate choreography can produce it (the controller's own "
                "q0/g0 are taken at on_activate); publishing without it would "
                "send an absolute pose through a relative map"
            )
        q0, g0 = reference
        return [float(v) for v in g0 + self.gello_directions * (values - q0)]

    def _publish_arm_wire(self, left_wire, right_wire, repeat: bool = False) -> None:
        """Put already-transformed positions on the wire, with a fresh stamp."""
        with self._cmd_lock:
            self._publish_arm_wire_locked(left_wire, right_wire, repeat=repeat)

    def _publish_arm_wire_locked(
        self, left_wire, right_wire, repeat: bool = False
    ) -> None:
        """`_publish_arm_wire`'s body, with `_cmd_lock` already held.

        The publish itself is inside the lock, not just the bookkeeping after
        it: the keep-alive runs on its own thread and decides whether to
        publish from `_last_arm_pub`. With only the bookkeeping locked, that
        decision could be made against a timestamp the in-flight publish had
        not written yet. Publishes are cheap, so the lock spans them.

        ``repeat`` marks a keep-alive / service-call republish — the same
        values again. It changes no wire behaviour at all, only the
        activation window in `mark_activation`: a repeat is what covers that
        window, so it cannot also be what closes it.
        """
        self._left_arm.publish(
            self._joint_state(self.topics.left_arm_joint_names, left_wire)
        )
        self._right_arm.publish(
            self._joint_state(self.topics.right_arm_joint_names, right_wire)
        )
        now = time.monotonic()
        if self._last_arm_pub > 0.0:  # 0.0 = nothing published yet, no gap
            self._max_publish_gap_s = max(
                self._max_publish_gap_s, now - self._last_arm_pub
            )
        if (
            not repeat
            and self._activation_t is not None
            and self._activation_to_first_cmd_s is None
        ):
            # D: the controller went ACTIVE at `_activation_t` and this is the
            # first NEW command it gets. Folded into the same max as any other
            # gap because it is the same physical thing — how long the arm was
            # held by a repeat of a pose nobody had re-decided. The 2026-09-02
            # attempt reported max_publish_gap_s 0.206 s across a 1.6 s window
            # of exactly that (backend.reset + a 1.443 s first inference).
            self._activation_to_first_cmd_s = now - self._activation_t
            self._max_publish_gap_s = max(
                self._max_publish_gap_s, self._activation_to_first_cmd_s
            )
        self._last_arm_wire = (list(left_wire), list(right_wire))
        self._last_arm_pub = now
        self._arm_history.append(now, left_wire, right_wire)

    def publish(self, command: Command) -> None:
        self.publish_arms(command.left_arm, command.right_arm)
        self.publish_grippers(command.left_gripper, command.right_gripper)
        self._token = command.base_token
        if self._twist_pub is not None:
            twist = tuple(float(v) for v in command.base_twist)
            if self._base_limits is not None:
                twist, clamped = clip_base_twist(twist, *self._base_limits)
                if clamped:
                    self.base_clamped += 1
            self._twist = twist
        self._repeat_base()

    def base_subscribed(self) -> bool | None:
        """Whether anything listens on the base command topic (None = unknown).

        base_nudge.py waits for this before the first command: DDS matching
        takes a moment and commands published before it are lost, so a move
        is shorter than requested. The approach controller checks it once
        before its first navigate tick on the real robot.
        """
        if self._twist_pub is None:
            return None
        count = getattr(self._twist_pub, "get_subscription_count", None)
        if count is None:
            return None
        return int(count()) > 0

    def publish_arms(self, left_arm, right_arm) -> None:
        """Arms only (sine actor and other low-level probes).

        A HELD arm (`set_held_arm`, i.e. one this run does not drive) ignores
        the caller's value entirely and republishes its frozen pose, so the
        policy's prediction for an arm nobody selected cannot reach the wire
        through some other call site.
        """
        self._publish_arm_wire(
            self._side_wire("left", left_arm), self._side_wire("right", right_arm)
        )

    def publish_arms_unmapped(self, left_arm, right_arm) -> None:
        """Publish joint values with NO arm-frame mapping (the hold phase).

        The gello frame is defined by `(q0, g0)`, which do not exist until
        activation — and the publishing that makes `g0 == q0` is precisely
        this one. Mapping it would be circular, so it is not mapped.
        """
        self._publish_arm_wire(
            self._held_or("left", left_arm), self._held_or("right", right_arm)
        )

    def set_held_arm(self, side: str, positions) -> None:
        """Freeze one arm at ``positions`` for the rest of the run (`--arms`)."""
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        values = np.asarray(positions, dtype=np.float64).ravel()
        if values.shape != (7,):
            raise ValueError(f"held {side} arm must be 7 joints, got {values.shape}")
        self._held_arms[side] = values
        log.info("arm %s held at %s", side, [round(float(v), 4) for v in values])

    def _held_or(self, side: str, values) -> list:
        held = self._held_arms.get(side)
        return [float(v) for v in (held if held is not None else values)]

    def _side_wire(self, side: str, values) -> list:
        held = self._held_arms.get(side)
        if held is not None:
            # A held arm sits at the pose its own g0 was captured from, so the
            # raw value is a no-op in BOTH frames; mapping it would not be.
            return [float(v) for v in held]
        return self._arm_wire(side, values)

    def keepalive_active(self) -> bool:
        """Is the keep-alive thread republishing the last command?

        Public because callers make a SAFETY decision on it: a blocking
        controller_manager call (`_call`, up to 5 s per side, and the sides
        are switched one after the other) publishes nothing itself, so what
        keeps the stream alive across it is this thread. False here on the
        real robot means the next service call can outlast the controller's
        2.0 s liveness gap. `is_alive()` and not just "was created": a thread
        that died on an exception must not read as a live watchdog.
        """
        return self._keepalive_thread is not None and self._keepalive_thread.is_alive()

    def check_publish_cadence(
        self, window_s: float, min_hz: float = MIN_ACTIVATION_PUBLISH_HZ
    ) -> dict:
        """Measure the recent publish cadence; raise `PublishCadenceTooLow`.

        The runner calls this immediately before the activation switch
        (`camelo/runner/real_arms.py`). It reads the same ring buffer the
        activation-reference guard reads, under the same lock, so the number
        it refuses on is the number the report prints.
        """
        with self._cmd_lock:
            return guard_publish_cadence(
                self._arm_history, time.monotonic(), float(window_s), float(min_hz)
            )

    def mark_activation(self, t: float | None = None) -> None:
        """Open the activation -> first-command window (D).

        Called the instant `list_controllers` confirms ACTIVE. The next
        non-repeat arm publish closes it and folds its length into
        `max_publish_gap_s`; until then `activation_to_first_cmd_s` is None,
        which is itself the honest answer ("the policy has not spoken yet").
        """
        with self._cmd_lock:
            self._activation_t = time.monotonic() if t is None else float(t)
            self._activation_to_first_cmd_s = None

    def last_arm_command(self) -> tuple | None:
        """The last (left, right) positions put on the wire, or None."""
        with self._cmd_lock:
            return None if self._last_arm_wire is None else (
                list(self._last_arm_wire[0]), list(self._last_arm_wire[1])
            )

    # -- telemetry ---------------------------------------------------------
    def publish_stats(self) -> dict:
        """Keep-alive repeats and the worst gap between two arm publishes.

        The gap is the number the rig runbook's T6 asks for: the controller
        dies at 2.0 s without a sample, so "how close did this rollout get"
        is the margin, and a mean would hide the one stall that matters.

        `keepalive_max_interval_s` is the WATCHDOG's own worst period —
        distinct from the gap above, which the watchdog's repeats keep small
        by construction. `activation_to_first_cmd_s` is None until the first
        real command after an activation.
        """
        with self._cmd_lock:
            return {
                "keepalive_republished": int(self.keepalive_republished),
                "keepalive_base_republished": int(self.keepalive_base_republished),
                "base_clamped": int(self.base_clamped),
                "max_publish_gap_s": float(self._max_publish_gap_s),
                "keepalive_max_interval_s": float(self._keepalive_max_interval_s),
                "activation_to_first_cmd_s": (
                    None if self._activation_to_first_cmd_s is None
                    else float(self._activation_to_first_cmd_s)
                ),
            }

    def reset_publish_stats(self) -> None:
        """Zero the counters so they mean 'this rollout', not 'this process'.

        `_last_arm_pub` and `_keepalive_last_wake` are deliberately NOT
        reset: the first gap / interval is therefore measured from the last
        event before the rollout began (on real, at most one keep-alive
        period ago). Both can only come out too large, never too small, which
        is the right direction for a margin. `_activation_t` survives too —
        the activation is what the window is measured FROM, and the rollout's
        first command is what closes it, in that order.
        """
        with self._cmd_lock:
            self.keepalive_republished = 0
            self.keepalive_base_republished = 0
            self.base_clamped = 0
            self._max_publish_gap_s = 0.0
            self._keepalive_max_interval_s = 0.0

    def repeat_last_arm_command(self) -> None:
        """Republish the last arm command NOW, with a fresh stamp.

        Used while blocking on a controller_manager service: the stream must
        not go quiet for 2.0 s while we ask the controller a question about
        itself.
        """
        arms = self.last_arm_command()
        if arms is not None:
            self._publish_arm_wire(*arms, repeat=True)

    # -- keep-alive (real only) --------------------------------------------
    def _keepalive_loop(self) -> None:
        """The keep-alive's own thread: a monotonic schedule nothing shares.

        Deliberately not a ROS timer (module docstring, U-28) and
        deliberately not `sleep(period)`: the schedule is absolute, so a slow
        publish shortens the next wait instead of accumulating drift. A wake
        that arrived late re-bases rather than firing a burst of catch-up
        repeats — the controller wants a fresh sample, not N of them.

        Nothing here may raise: this thread IS the liveness guarantee, and a
        watchdog that dies silently on a transient publish error is worse
        than none, because `keepalive_active()` would still have said True.
        """
        period = self._keepalive_period
        if period is None:  # pragma: no cover - never started without one
            return
        next_at = time.monotonic() + period
        while not self._keepalive_stop.wait(max(0.0, next_at - time.monotonic())):
            now = time.monotonic()
            if self._keepalive_last_wake > 0.0:
                with self._cmd_lock:
                    self._keepalive_max_interval_s = max(
                        self._keepalive_max_interval_s,
                        now - self._keepalive_last_wake,
                    )
            self._keepalive_last_wake = now
            next_at = next_at + period
            if next_at <= now:
                next_at = now + period
            try:
                self._keepalive()
            except Exception:
                self._keepalive_errors += 1
                if self._keepalive_errors <= 3:
                    log.exception(
                        "keep-alive republish failed (%d); the thread keeps "
                        "running — stopping it would drop the command stream "
                        "on a possibly ACTIVE controller",
                        self._keepalive_errors,
                    )

    def _keepalive(self) -> None:
        """Republish the last command if the loop has gone quiet.

        Fresh stamp, identical values. The controller rejects samples older
        than 0.5 s by ITS clock and shuts the arm stack down after 2.0 s
        without one, so a slow remote inference must not read as a dead
        publisher. Nothing new is ever commanded here — a repeat cannot move
        the robot, and the executor's slew clamp still owns every change.
        """
        if self.topics.world != C.WORLD_REAL:
            return  # sim keeps its exact previous behaviour
        if self._keepalive_period is None:
            return
        # Base backstop: the pedal-repeat ROS timer shares the executor with
        # the camera drains and can starve past the swerve's 0.5 s window;
        # this thread does not. A repeat is the SAME value with a new stamp.
        if (
            self._twist_pub is not None
            and self._last_base_pub > 0.0  # never before the first base command
            and time.monotonic() - self._last_base_pub >= self._base_repeat_period
        ):
            self._repeat_base()
            self.keepalive_base_republished += 1
        with self._cmd_lock:
            arms = self._last_arm_wire
            grippers = self._last_grippers
            if arms is None:
                return
            # Re-read the clock HERE, under the lock, immediately before
            # publishing: between a check outside the lock and the publish,
            # the control loop can have sent a fresh command, and this would
            # then double-publish a stale value on top of it.
            if time.monotonic() - self._last_arm_pub < self._keepalive_period:
                return
            self._publish_arm_wire_locked(*arms, repeat=True)
            if grippers is not None:
                self._publish_grippers_locked(*grippers)
            self.keepalive_republished += 1

    def publish_grippers(self, left: float, right: float) -> None:
        """Canonical open fractions (1 = open); wire polarity applied here."""
        with self._cmd_lock:
            self._publish_grippers_locked(left, right)

    def _publish_grippers_locked(self, left: float, right: float) -> None:
        """`publish_grippers`' body, with `_cmd_lock` already held."""
        self._last_grippers = (float(left), float(right))
        if self.topics.gripper_cmd_kind == "width_percent":
            left_msg = Float32()
            left_msg.data = C.gripper_width_percent(left, self.gripper_invert)
            right_msg = Float32()
            right_msg.data = C.gripper_width_percent(right, self.gripper_invert)
            self._left_grip.publish(left_msg)
            self._right_grip.publish(right_msg)
            return
        self._left_grip.publish(
            self._joint_state(
                [C.LEFT_GRIPPER_OPENING], [C.gripper_wire_value(left, self.gripper_invert)]
            )
        )
        self._right_grip.publish(
            self._joint_state(
                [C.RIGHT_GRIPPER_OPENING], [C.gripper_wire_value(right, self.gripper_invert)]
            )
        )

    def safe_stop(self) -> None:
        """Base to NONE / zero twist; arms/grippers hold their last applied targets.

        On the stamped real wire the zeros are spread over ~0.3 s the way
        base_nudge.py does it: one burst can be reordered behind an in-flight
        repeat, and the controller otherwise holds the last twist until its
        0.5 s watchdog — another ~5 cm of travel at 0.1 m/s.
        """
        self._token = "NONE"
        self._twist = (0.0, 0.0, 0.0)
        repeats = 6 if self._twist_stamped else 3
        for i in range(repeats):
            self._repeat_base()
            if self._twist_stamped and i + 1 < repeats:
                time.sleep(0.05)

    def close(self) -> None:
        """Stop the keep-alive thread and cancel the pedal-repeat timer.

        Call ONCE, at the
        very end of a process's life — never between episodes or phases,
        where `safe_stop()` already runs and the NEXT command stream still
        needs both watchdogs watching it.

        MEASURED on the rig 2026-09-02 (T1(c)): at process exit rclpy
        printed "The following exception was never retrieved: cannot use
        Destroyable because destruction was requested" twice, once per
        timer. `ros_session` (camelo/ros/session.py) already stops the
        executor and joins the spin thread before `destroy_node()` (F-16
        order), but a timer callback already handed to the
        MultiThreadedExecutor's worker pool can still be in flight when
        that teardown starts, racing the rcl handle it destroys. Cancelling
        the pedal timer here — the last thing this process ever does with
        it — means it cannot hand the executor a new callback once the
        teardown begins; a cancelled timer is simply never "ready".

        The keep-alive is a THREAD rather than a timer now (U-28), and it is
        joined here for the same reason: it publishes, so it has to be
        finished publishing before anything destroys the publishers it uses.
        It is a daemon, so a join that times out is logged rather than fatal.

        Never raises: a timer that is already gone (or a fake with no
        `cancel`) must not skip whatever teardown runs after this.
        """
        self._keepalive_stop.set()
        thread = self._keepalive_thread
        if thread is not None:
            with contextlib.suppress(Exception):
                thread.join(timeout=2.0)
            if thread.is_alive():
                log.warning(
                    "the keep-alive thread did not stop within 2s; it is a "
                    "daemon so interpreter exit is not blocked, but it may "
                    "still be publishing"
                )
        for timer in (self._pedal_timer,):
            if timer is None:
                continue
            with contextlib.suppress(Exception):
                timer.cancel()


def browser_conflict(node, topics: C.TopicMap | None = None) -> list[str]:
    """Browser command topics with live publishers (they'd fight the policy)."""
    check = (topics or C.topics_for()).browser_cmd_topics
    conflicts = []
    for topic in check:
        if node.get_publishers_info_by_topic(topic):
            conflicts.append(topic)
    return conflicts


def verify_gripper_polarity(
    collector,
    publisher,
    dwell_s: float = 8.0,
) -> None:
    """Live polarity self-test: command open then closed, assert the SWING.

    Protects every dataset (DGX_FINDINGS.md F-18). Design per F-27/F-29:
    no convergence detection at all — the gripper has ~0.5 s of dead time
    before it moves, which a stability predicate cannot tell apart from
    "settled" (that bug made the gate fail 0/5). Instead: keep commanding
    for a generous fixed dwell (travel is ~3 s here incl. multi-second sim
    stalls), read once at the end of each phase, and assert the RELATIVE
    swing `closed - open > 0.5 * stroke` — which is polarity by definition
    (larger driver angle = more closed) and immune to the starting pose.
    Raises RuntimeError on a genuine mismatch; requires the sim + helper
    stack up and the collector already receiving /isaac/joint_states_full.
    """

    def measured_driver() -> float:
        snap = collector.snapshot()
        return C.resolve_joint(snap["joint_states"], C.LEFT_GRIPPER_DRIVER)

    def command_and_dwell(open_fraction: float, dwell: float) -> float:
        deadline = time.monotonic() + dwell
        while time.monotonic() < deadline:
            publisher.publish_grippers(open_fraction, open_fraction)
            time.sleep(0.2)  # /isaac/joint_states_full arrives at ~10 Hz
        return measured_driver()

    open_rad = command_and_dwell(1.0, dwell_s)
    closed_rad = command_and_dwell(0.0, dwell_s)
    command_and_dwell(1.0, 2.0)  # leave the grippers opening (scene default)
    swing = closed_rad - open_rad
    if not swing > 0.5 * C.GRIPPER_CLOSED_RAD:
        raise RuntimeError(
            "gripper polarity check FAILED: commanded open=1.0 -> driver "
            f"{open_rad:.3f} rad, closed=0.0 -> {closed_rad:.3f} rad, swing "
            f"{swing:.3f} (needs > {0.5 * C.GRIPPER_CLOSED_RAD:.2f}; "
            f"invert={publisher.gripper_invert}). Either the republisher's "
            "REPUBLISHER_GRIPPER_INVERT and camelo disagree (set "
            "CAMELO_GRIPPER_INVERT or fix the stack env) or the sim stalled "
            "through the dwell — retry with a larger dwell before recording."
        )
    log.info(
        "gripper polarity OK: open=1.0 -> %.3f rad, closed=0.0 -> %.3f rad, "
        "swing %.3f (invert=%s)",
        open_rad, closed_rad, swing, publisher.gripper_invert,
    )
