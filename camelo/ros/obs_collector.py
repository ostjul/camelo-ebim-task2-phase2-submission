"""Latest-value observation cache over the task2 bridge topics.

Mirrors the benchmark recorder's subscription set (record_task2.py
Task2RecorderNode) minus the ground-truth streams, and assembles the same
canonical 37-dim state via camelo.contracts — so what a policy sees at
inference is exactly what training saw from the recorded datasets.

Cameras are NOT subscribed in this process: rclpy image ingest is
per-process GIL-bound (~0.8 Hz for three cameras in one process vs the
~5.6 Hz the wire carries — DGX_FINDINGS.md F-30). Each camera runs in its
own subscriber subprocess (camelo.ros.camera_workers); a node timer
drains their decoded frames into the cache, so ``images`` holds numpy
arrays, not ROS messages.

Attaches to an existing rclpy Node so one process can hold both the
collector and the command publisher.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import threading
import time
from collections import deque

from geometry_msgs.msg import PoseStamped, Twist, TwistStamped, WrenchStamped
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, JointState
from std_msgs.msg import Float32MultiArray, String

from camelo import contracts as C
from camelo.control.image_age import FrameArrival, FrameLog
from camelo.control.odom_interp import interpolate_xy_yaw
from camelo.control.state_age import state_liveness_key, state_liveness_keys
from camelo.policy.base import Obs
from camelo.ros.camera_workers import CameraWorkers
from camelo.ros.qos import DEFAULT_DEPTH, SENSOR_QOS

log = logging.getLogger(__name__)

# Head RGB is ~1.6 Hz while odom is ~50 Hz. Keep enough samples to rewind the
# GT overlay to the image stamp while the base is yawing (B+C is 1.2 rad/s).
_ODOM_HIST_MAX = 300


class ObsCollector:
    def __init__(
        self,
        node,
        camera_keys=None,
        topics: C.TopicMap | None = None,
        wrench_declared_absent: bool = False,
    ):
        self.node = node
        # s27a15 raises on a MISSING wrench group and there is no zero
        # default (12 of 27 trained dims are wrench, F-63). A rig that
        # genuinely has no wrench topic says so here — `--no-wrench` — and
        # the declaration travels with the observation instead of arriving
        # as silent zeros.
        self.wrench_declared_absent = bool(wrench_declared_absent)
        self.topics = topics if topics is not None else C.topics_for()
        self.camera_keys = list(C.CAMERA_KEYS if camera_keys is None else camera_keys)
        unknown = [key for key in self.camera_keys if key not in self.topics.cameras]
        if unknown:
            raise ValueError(
                f"cameras {unknown} are not in the {self.topics.world} topic map; "
                f"valid: {list(self.topics.cameras)}"
            )
        self.lock = threading.Lock()
        self.sim_time = None
        self.joint_states: dict[str, float] = {}
        # Monotonic WALL receipt time of the newest JointState per arm
        # stream, `None` until the first one lands. `joint_states` above is
        # only ever overwritten — never cleared, never expired — so this is
        # the ONLY thing that can say a stream died mid-rollout (U-27, the
        # joint-state twin of `frame_log`'s liveness role for cameras). Keyed
        # by `camelo.control.state_age`; the spine and the sim's un-sided
        # topic are deliberately absent (see ARM_SIDES there).
        self._joint_state_wall: dict[str, float | None] = dict.fromkeys(
            state_liveness_keys(self.topics.joint_state_topics), None
        )
        self.applied_commands: dict[str, float] = {}
        self.odom = None
        self._odom_hist: deque = deque(maxlen=_ODOM_HIST_MAX)
        self.cmd_vel = (0.0, 0.0, 0.0)
        self.ee_poses = {"left": None, "right": None}
        # Real only: force xyz + torque xyz in the stiffness frame, per arm.
        # None until the first message, so "not wired" and "measured all
        # zero" stay distinguishable — an exactly-zero wrench is real data
        # here (4,359 of 121,828 corpus frames), so the guard has to be
        # structural (adapters/s27a15.py).
        self.wrenches: dict[str, tuple | None] = {"left": None, "right": None}
        self.images: dict = {key: None for key in self.camera_keys}  # numpy arrays
        # Sim-time stamp of the cached frame per camera (None = unstamped or
        # no frame yet) and the bounded arrival history behind the N2b
        # staleness numbers (camelo.control.image_age).
        self.image_t_sim: dict[str, float | None] = {key: None for key in self.camera_keys}
        # Real only: measured frame.shape per camera, once it has disagreed
        # with the contract's declared `shape` (docs/realdata/16 R-42 — the
        # station's ZED publishes VGA while the corpus and contracts.py's
        # real TopicMap both say HD720). Populated on the FIRST mismatch per
        # camera; absence means either sim (never checked) or agreement.
        self.camera_shape_mismatch: dict[str, tuple] = {}
        self.frame_log = FrameLog(self.camera_keys)
        self.head_k: tuple[float, float, float, float] | None = None
        self.head_d: tuple[float, ...] = ()
        self.reset_events: list[str] = []
        self.object_poses: dict[str, list[float]] = {}
        self.pad_centroid: tuple[float, float, float] | None = None
        self.counts: dict[str, int] = {}
        self._t_start = time.monotonic()
        self._reset_request_pub = None
        # Every entity this instance owns on `node` that needs an explicit
        # teardown in `close()` (see there for why): subscription handles,
        # and the two timers this class itself creates. `None` until
        # created — `close()` must not assume either timer exists (no
        # cameras -> no drain timer; sim/`/isaac/clock` -> no wall-clock
        # timer).
        self._subs: list = []
        self._drain_timer = None
        self._wall_clock_timer = None
        self._closed = False

        self._subscribe(node)
        cam_topics = {key: self.topics.cameras[key]["image_topic"] for key in self.camera_keys}
        self._workers = CameraWorkers(cam_topics)
        if cam_topics:
            log.info(
                "obs world=%s cameras=%s",
                self.topics.world,
                cam_topics,
            )
        if self.camera_keys:
            self._drain_timer = node.create_timer(0.05, self._drain_cameras)

    def _subscribe(self, node) -> None:
        t = self.topics
        # Real Franka measured_joint_states is BEST_EFFORT; a RELIABLE
        # subscriber is silently unmatched (RELIABILITY incompatible).
        qos = SENSOR_QOS if t.world == C.WORLD_REAL else DEFAULT_DEPTH

        def sub(*args):
            # Every subscription handle this instance creates, tracked so
            # `close()` can destroy each one explicitly instead of leaving
            # it live on the executor through teardown.
            handle = node.create_subscription(*args)
            self._subs.append(handle)
            return handle

        if t.clock:
            sub(Clock, t.clock, self._on_clock, qos)
        else:
            # Real robot: no /isaac/clock. Drive t_sim from ROS time so the
            # rest of the pipeline (rollout length, chunk executor) keeps
            # working; on a live robot ROS time is wall time.
            self._wall_clock_timer = node.create_timer(0.02, self._on_wall_clock)
        for topic, side in t.joint_state_topics:
            # The gripper's joint_states is a JointState on the same side as
            # the arm broadcaster, so it used to land in the same
            # `joint_states_{side}` counter — and a probe reading `js_hz`
            # could not tell whether gripper feedback was arriving at all
            # (T1(c), 2026-09-02). Count it again under its own key; the
            # existing keys are untouched.
            extra = (
                f"gripper_js_{side or 'unsided'}"
                if C.is_gripper_joint_state_topic(topic)
                else None
            )
            # U-27's key for this stream (None for the spine and the sim's
            # un-sided topic), bound like `extra` so the lambda captures the
            # value rather than the loop variable.
            liveness = state_liveness_key(topic, side)
            sub(
                JointState,
                topic,
                lambda m, s=side, k=extra, w=liveness: (
                    self._on_joint_states(m, s, k, w)
                ),
                qos,
            )
        for topic, side in t.applied_command_topics:
            sub(
                JointState,
                topic,
                lambda m, s=side: self._on_applied(m, s),
                qos,
            )
        for side, topic in (("left", t.left_wrench), ("right", t.right_wrench)):
            if topic:
                sub(
                    WrenchStamped,
                    topic,
                    lambda m, s=side: self._on_wrench(s, m),
                    SENSOR_QOS,  # broadcaster output is BEST_EFFORT
                )
        if t.odom:
            sub(Odometry, t.odom, self._on_odom, qos)
        if t.cmd_vel_applied:
            # The real swerve echoes what it applied as the same stamped type
            # it accepts (contracts REAL_BASE_*); the sim bridge is plain Twist.
            applied_type = TwistStamped if getattr(t, "base_cmd_stamped", False) else Twist
            sub(applied_type, t.cmd_vel_applied, self._on_cmd_vel, qos)
        for side, topic in t.ee_poses:
            sub(PoseStamped, topic, lambda m, s=side: self._on_ee_pose(s, m), qos)
        if t.scene_reset:
            sub(String, t.scene_reset, self._on_scene_reset, qos)
        if t.object_poses:
            sub(String, t.object_poses, self._on_object_poses, qos)
        if t.pad_points:
            sub(Float32MultiArray, t.pad_points, self._on_pad_points, qos)
        if t.scene_reset_request:
            self._reset_request_pub = node.create_publisher(
                String, t.scene_reset_request, DEFAULT_DEPTH
            )
        if t.world == C.WORLD_SIM and "head" in self.camera_keys:
            # Live pinhole for perception (camelo.control.perception): Isaac
            # publishes this BEST_EFFORT same as images (camelo/ros/qos.py).
            sub(CameraInfo, C.HEAD_CAMERA_INFO_TOPIC, self._on_head_camera_info, SENSOR_QOS)

    # -- callbacks ---------------------------------------------------------
    def _count(self, key):
        self.counts[key] = self.counts.get(key, 0) + 1

    def _on_clock(self, msg):
        with self.lock:
            self.sim_time = msg.clock.sec + msg.clock.nanosec * 1e-9
            self._count("clock")

    def _on_wall_clock(self):
        now = self.node.get_clock().now()
        with self.lock:
            self.sim_time = now.nanoseconds * 1e-9
            self._count("clock")

    def _ingest_joints(self, dest: dict, msg, side: str) -> None:
        for idx, name in enumerate(msg.name):
            if idx < len(msg.position):
                dest[C.canonicalize_joint_name(name, side)] = float(msg.position[idx])

    def _on_joint_states(
        self,
        msg,
        side: str = "",
        extra_count: str | None = None,
        liveness_key: str | None = None,
    ):
        with self.lock:
            self._ingest_joints(self.joint_states, msg, side)
            self._count("joint_states_full")
            if side:
                self._count(f"joint_states_{side}")
            if extra_count:
                self._count(extra_count)
            if liveness_key is not None:
                # U-27: receipt time, taken here rather than from the
                # message's header — the rig incident that motivated this
                # guard WAS a clock-skew message-age rejection, so the
                # publisher's own notion of time is the broken quantity.
                self._joint_state_wall[liveness_key] = time.monotonic()

    def _on_full_states(self, msg):
        # Kept for probes that call the old name; same as un-sided ingest.
        self._on_joint_states(msg, "")

    def _on_applied(self, msg, side: str = ""):
        with self.lock:
            self._ingest_joints(self.applied_commands, msg, side)
            self._count("applied_commands")

    def _on_odom(self, msg):
        pos, ori = msg.pose.pose.position, msg.pose.pose.orientation
        lin, ang = msg.twist.twist.linear, msg.twist.twist.angular
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        yaw = C.quat_to_yaw(ori.x, ori.y, ori.z, ori.w)
        with self.lock:
            self.odom = (
                pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w, lin.x, lin.y, lin.z, ang.z,
            )
            if stamp <= 0.0:
                stamp = self.sim_time
            if stamp is not None:
                if self._odom_hist and stamp < self._odom_hist[-1][0] - 1.0:
                    self._odom_hist.clear()  # sim clock rebased
                self._odom_hist.append(
                    (float(stamp), float(pos.x), float(pos.y), float(yaw))
                )
            self._count("odom")

    def _on_head_camera_info(self, msg):
        """Live pinhole. The yaml 90°×60° is the real ZED, not this USD Camera."""
        k = msg.k
        fx, fy, cx, cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])
        if fx <= 1.0 or fy <= 1.0:
            return
        d = tuple(float(v) for v in msg.d)
        with self.lock:
            self.head_k = (fx, fy, cx, cy)
            self.head_d = d
            self._count("head_camera_info")

    def _on_cmd_vel(self, msg):
        twist = getattr(msg, "twist", msg)  # TwistStamped wraps the Twist
        with self.lock:
            self.cmd_vel = (twist.linear.x, twist.linear.y, twist.angular.z)
            self._count("cmd_vel")

    def _on_wrench(self, side, msg):
        w = msg.wrench
        with self.lock:
            self.wrenches[side] = (
                w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z,
            )
            self._count(f"wrench_{side}")

    def _on_ee_pose(self, side, msg):
        p, o = msg.pose.position, msg.pose.orientation
        with self.lock:
            self.ee_poses[side] = (p.x, p.y, p.z, o.x, o.y, o.z, o.w)
            self._count(f"ee_{side}")

    def _drain_cameras(self):
        if self._closed:
            return  # close() cancelled this timer, but a queued call can
            # still be in flight on the executor's worker pool (see close())
        latest = self._workers.poll()
        if not latest:
            return
        now_wall = time.monotonic()
        with self.lock:
            for key, frame in latest.items():
                self.images[key] = frame.array
                self.image_t_sim[key] = frame.t_stamp
                # t_clock is the clock value held NOW, at drain — the frame
                # was decoded up to one drain period (50 ms wall) earlier.
                self.frame_log.record(key, frame.t_stamp, self.sim_time, now_wall)
                self._count(f"image_{key}")
                self._check_camera_shape(key, frame.array)

    def _check_camera_shape(self, key: str, array) -> None:
        """Real world only: loud drift alarm when a camera's own frame shape
        disagrees with the contract's declared shape (docs/realdata/16
        R-42 — the station's ZED was moved to VGA on 2026-08-31 to save
        bandwidth while contracts.py's real TopicMap and the s27a15 corpus
        both still say HD720). Ingest reshapes from the message's own
        height/width and never compared against the contract before this,
        so a policy trained on 720x1280 would silently receive 376x672.

        Logs once per camera (first mismatch only) and records the measured
        shape in ``self.camera_shape_mismatch`` for callers — check_obs.py,
        run_policy.py's refusal, debug_status() — to act on. Must be called
        with ``self.lock`` held.
        """
        if self.topics.world != C.WORLD_REAL:
            return
        if key in self.camera_shape_mismatch:
            return  # already logged for this camera
        expected = self.topics.cameras.get(key, {}).get("shape")
        if expected is None:
            return
        expected = tuple(expected)
        measured = tuple(array.shape)
        if measured == expected:
            return
        self.camera_shape_mismatch[key] = measured
        log.error(
            "camera %s shape %s != contract %s — the station's ZED is not "
            "at the corpus resolution (docs/realdata/16 R-42); fix the "
            "launcher (resolution HD720), do not resize here",
            key, measured, expected,
        )

    def _on_scene_reset(self, msg):
        with self.lock:
            self.reset_events.append(msg.data)
            self._count("scene_reset")

    def _on_object_poses(self, msg):
        """Ground-truth object poses, JSON in a String.

        Kept as a latest-value cache like the rest: the grasp gate reads the
        pad LIVE because the assembly gets nudged mid-episode, and a gate
        measured against a launch-time constant drifts with it.
        """
        try:
            objects = json.loads(msg.data).get("objects", {})
        except (ValueError, AttributeError):
            return  # malformed frame: keep the last good one rather than crash
        with self.lock:
            self.object_poses = objects
            self._count("object_poses")

    def object_xyz(self, name: str) -> tuple[float, float, float] | None:
        """World (x, y, z) of a ground-truth object, or None if unseen."""
        with self.lock:
            pose = (self.object_poses or {}).get(name)
        if not pose or len(pose) < 3:
            return None
        return (float(pose[0]), float(pose[1]), float(pose[2]))

    def _on_pad_points(self, msg):
        """Thermal-pad mesh vertices -> a centroid, the only live pad position.

        The pad is a DEFORMABLE body: its prim transform on
        OBJECT_POSES_TOPIC is written at scene reset and never again.
        Measured 2026-08-25 -- frozen at (1.750, 1.950, 0.850) for all 7183
        ticks of a rollout that scored IoU 0.87, so `object_xyz("thermalpad")`
        reports "the pad never moved" about an episode that carried it to the
        target. Mesh vertices are the ground truth that actually moves.

        Layout is [sim_time, n_points, x0, y0, z0, x1, ...] in world frame.
        The centroid is a proxy for position, not a pose: a deformable body
        has no rigid transform, and squeezing it moves vertices without
        moving the pad. It is the right quantity for "did the pad travel",
        which is what the protocol's displacement metric asks.
        """
        data = getattr(msg, "data", None)
        if data is None or len(data) < 2:
            return
        n = int(data[1])
        flat = data[2 : 2 + 3 * n]
        if n <= 0 or len(flat) < 3 * n:
            return  # partial frame: keep the last good centroid
        xs = flat[0::3]
        ys = flat[1::3]
        zs = flat[2::3]
        centroid = (sum(xs) / n, sum(ys) / n, sum(zs) / n)
        with self.lock:
            self.pad_centroid = centroid
            self._count("pad_points")

    def pad_xyz(self) -> tuple[float, float, float] | None:
        """Live thermal-pad position (mesh centroid), or None if unseen.

        Use this, never `object_xyz("thermalpad")`, for anything that must
        notice the pad MOVING -- see `_on_pad_points`.
        """
        with self.lock:
            return self.pad_centroid

    # -- API ---------------------------------------------------------------
    def snapshot(self) -> dict:
        """Recorder-shaped snapshot (same keys build_state/build_action use)."""
        with self.lock:
            return {
                "sim_time": self.sim_time,
                "joint_states": dict(self.joint_states),
                "applied_commands": dict(self.applied_commands),
                "odom": self.odom,
                "odom_hist": list(self._odom_hist),
                "cmd_vel": tuple(self.cmd_vel),
                "ee_poses": dict(self.ee_poses),
                "wrenches": dict(self.wrenches),
                "images": dict(self.images),
                "image_t_sim": dict(self.image_t_sim),
                "head_k": self.head_k,
                "head_d": self.head_d,
            }

    def frame_marker(self) -> dict[str, int]:
        """Per-camera frame counters now; bracket a window with ``frames_since``."""
        with self.lock:
            return self.frame_log.marker()

    def frames_since(self, marker: dict[str, int] | None = None) -> dict[str, list[FrameArrival]]:
        """Per-camera arrivals (stamp, clock at drain, wall) after ``marker``."""
        with self.lock:
            return self.frame_log.since(marker)

    def last_image_wall(self) -> dict[str, float | None]:
        """Monotonic receipt time of each camera's newest frame (None = none yet).

        The liveness clock for the stale-image guard (U-23): `get_obs` never
        clears `self.images`, so the age of the CACHE, not of the observation,
        is what says a camera died mid-rollout. Read against
        `time.monotonic()` by `camelo.control.image_age.stale_image_report`.
        """
        with self.lock:
            return self.frame_log.last_wall()

    def last_state_wall(self) -> dict[str, float | None]:
        """Monotonic receipt time of each arm stream's newest JointState.

        `None` for a stream that has never delivered one; empty when the
        topic map has no sided arm streams at all (sim). The liveness clock
        for the stale-joint-state guard (U-27), and the exact twin of
        `last_image_wall`: `self.joint_states` is never cleared, so the age
        of the CACHE — not of the observation — is what says an arm's
        publisher died mid-rollout. Read against `time.monotonic()` by
        `camelo.control.state_age.stale_state_report`.
        """
        with self.lock:
            return dict(self._joint_state_wall)

    def rig_groups(self, snap: dict | None = None) -> dict | None:
        """s27a15 NAMED observation groups, or None while one is missing.

        The real bridge's half of `adapters/s27a15.py::RigObservation
        .from_mapping`: `left_arm(7)`, `right_arm(7)`, `right_gripper_rad`
        (RAW knuckle radians — the adapter owns the
        `1 - clip(rad,0,0.8)/0.8` conversion, so converting here would apply
        it twice), `left_wrench(6)`, `right_wrench(6)`.

        A key is set only once its data has ARRIVED. That is the whole
        point: `from_mapping` raises on a missing group, and a zero-filled
        stand-in for an unwired wrench is F-63 in its purest form — 12 of
        the 27 trained dims are wrench and an exactly-zero reading is real
        data on this rig. So an incomplete observation returns None here and
        `get_obs` keeps waiting, rather than handing the policy a hole.
        """
        snap = self.snapshot() if snap is None else snap
        joints = snap["joint_states"]
        groups: dict = {}
        for key, names in (("left_arm", C.LEFT_JOINTS), ("right_arm", C.RIGHT_JOINTS)):
            values = [C.resolve_joint(joints, name) for name in names]
            if any(not math.isfinite(v) for v in values):
                return None
            groups[key] = values
        knuckle = C.resolve_joint(joints, C.RIGHT_GRIPPER_DRIVER)
        if not math.isfinite(knuckle):
            return None
        groups["right_gripper_rad"] = float(knuckle)
        if self.wrench_declared_absent:
            # Declared, not inferred: the flag travels with the observation
            # so the adapter logs an off-contract run instead of silently
            # normalizing zeros as if they were forces.
            groups["left_wrench"] = [0.0] * 6
            groups["right_wrench"] = [0.0] * 6
            groups["wrench_declared_absent"] = True
            return groups
        for key, side in (("left_wrench", "left"), ("right_wrench", "right")):
            wrench = snap["wrenches"].get(side)
            if wrench is None:
                return None
            groups[key] = [float(v) for v in wrench]
        return groups

    def missing_rig_groups(self, snap: dict | None = None) -> list[str]:
        """Which s27a15 groups have not arrived yet — for the wait logs."""
        snap = self.snapshot() if snap is None else snap
        joints = snap["joint_states"]
        missing = []
        for key, names in (("left_arm", C.LEFT_JOINTS), ("right_arm", C.RIGHT_JOINTS)):
            if any(not math.isfinite(C.resolve_joint(joints, n)) for n in names):
                missing.append(key)
        if not math.isfinite(C.resolve_joint(joints, C.RIGHT_GRIPPER_DRIVER)):
            missing.append("right_gripper_rad")
        if not self.wrench_declared_absent:
            for key, side in (("left_wrench", "left"), ("right_wrench", "right")):
                if snap["wrenches"].get(side) is None:
                    missing.append(key)
        return missing

    def get_obs(self, require_images: bool = True) -> Obs | None:
        """Canonical Obs, or None while streams are still incomplete."""
        snap = self.snapshot()
        if snap["sim_time"] is None or not snap["joint_states"]:
            return None
        if require_images and any(snap["images"].get(k) is None for k in self.camera_keys):
            return None
        rig = None
        if self.topics.world == C.WORLD_REAL:
            # Real runs are s27a15 runs: no complete observation until every
            # named group is live (see rig_groups). The 37-dim `state` is
            # built exactly as before beside it, so nothing sim-side moves.
            rig = self.rig_groups(snap)
            if rig is None:
                return None
        images = {  # already-decoded arrays from the camera workers
            key: array for key, array in snap["images"].items() if array is not None
        }
        image_t_sim = {
            key: t for key, t in snap["image_t_sim"].items() if key in images and t is not None
        }
        odom_at_image = {}
        hist = snap["odom_hist"]
        for key, stamp in image_t_sim.items():
            pose = interpolate_xy_yaw(hist, float(stamp))
            if pose is not None:
                odom_at_image[key] = pose
        return Obs(
            t_sim=snap["sim_time"],
            state=C.build_state(snap),
            images=images,
            image_t_sim=image_t_sim,
            rig=rig,
            odom_at_image=odom_at_image,
            head_k=snap["head_k"],
            head_d=tuple(snap["head_d"] or ()),
        )

    def debug_status(self) -> dict:
        """Why get_obs() is still None — for wait_for_obs logs, not the policy."""
        snap = self.snapshot()
        missing_images = [k for k in self.camera_keys if snap["images"].get(k) is None]
        return {
            "rates": self.rates(),
            "missing_rig": (
                self.missing_rig_groups(snap)
                if self.topics.world == C.WORLD_REAL
                else []
            ),
            "n_joints": len(snap["joint_states"]),
            "joint_names": sorted(snap["joint_states"])[:16],
            "missing_images": missing_images,
            "camera_topics": {
                k: self.topics.cameras[k]["image_topic"] for k in self.camera_keys
            },
            "camera_shape_mismatch": dict(self.camera_shape_mismatch),
            "workers": self._workers.status(),
        }

    def close(self) -> None:
        """Terminate the camera worker processes and destroy every rclpy
        entity this instance owns. Call ONCE, at the very end of a
        process's life — idempotent, and never raises.

        MEASURED on the rig 2026-09-02 (`scripts/check_obs.py --world real
        --seconds 10`, native Humble rclpy, deployed a20277f): after the
        camera-worker-side fix, the PARENT still printed "The following
        exception was never retrieved: cannot use Destroyable because
        destruction was requested" twice at exit. `check_obs.py` creates no
        CommandPublisher, so both came from ObsCollector's own entities
        still live on the `MultiThreadedExecutor` when `ros_session`
        (camelo/ros/session.py) calls `destroy_node()`: the 0.05 s camera
        drain timer, the real-robot 0.02 s wall-clock timer (there is no
        `/isaac/clock` on the rig — see `_subscribe`), and every stored
        subscription. A callback already handed to the executor's worker
        pool can still be in flight when teardown starts, racing the rcl
        handle it destroys — same class of race as
        `CommandPublisher.close()` (this morning's fix). Cancelling both
        timers FIRST means neither can hand the executor a new callback
        once teardown begins; `self._closed` additionally makes a queued
        `_drain_cameras` call that is already in flight a no-op instead of
        touching `self._workers` after `CameraWorkers.close()` has run.
        Camera workers are stopped next (their own subprocess-side
        teardown is `camera_workers.py`'s concern, not this race), and the
        subscriptions are destroyed last — order follows the same
        reasoning as `CommandPublisher.close()`: stop anything that could
        still enqueue work before destroying the entities that work would
        touch. Every step is wrapped so one entity already gone (or a fake
        with no `cancel`/`destroy_*`) never skips the rest.
        """
        if self._closed:
            return
        self._closed = True
        for timer in (self._drain_timer, self._wall_clock_timer):
            if timer is None:
                continue
            with contextlib.suppress(Exception):
                timer.cancel()
            with contextlib.suppress(Exception):
                self.node.destroy_timer(timer)
        with contextlib.suppress(Exception):
            self._workers.close()
        for sub in self._subs:
            with contextlib.suppress(Exception):
                self.node.destroy_subscription(sub)

    def request_scene_reset(self) -> None:
        if self._reset_request_pub is None:
            return
        self._reset_request_pub.publish(String())

    def drain_reset_events(self) -> list[str]:
        with self.lock:
            events, self.reset_events = self.reset_events, []
        return events

    def rates(self) -> dict[str, float]:
        elapsed = max(time.monotonic() - self._t_start, 1e-6)
        with self.lock:
            return {key: count / elapsed for key, count in sorted(self.counts.items())}
