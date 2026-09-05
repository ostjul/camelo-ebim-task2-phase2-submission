#!/usr/bin/env python3
"""Verify every topic a LeRobot episode needs is not just ADVERTISED but actually SENDING
real data, from ONE DDS participant.

    python3 verify_topics.py            # 10 s sample of all 28 topics
    python3 verify_topics.py --secs 20  # longer, for slow publishers
    python3 verify_topics.py --no-video # skip the three cameras

WHY THIS EXISTS
record_bag.bash --check answers "does a publisher exist?". That is not the same question
as "will this column contain data?", and its own header says so: a controller can advertise
a topic and never publish to it, and `--check` cannot see the difference. LABS'
TemporalSynchronizer then fails the WHOLE episode, not just that column. Three bugs of
exactly this kind were found on 2026-08-23.

ONE PARTICIPANT, DELIBERATELY
Every `ros2` CLI invocation creates a new DDS participant, and participant creation is a
15-25 s discovery burst on this network - which is enough to make a 1 kHz FCI loop miss
its deadline and fault the hardware. Checking 28 topics with 28 CLI calls would be 28
bursts aimed at a live robot. This subscribes to all of them from a single node.

VERDICTS
  ok        messages arriving and the payload carries data
  EMPTY     messages arriving but the payload is degenerate (zero-length array/image)
  ZEROS     messages arriving, all-numeric-zero - expected when idle, suspicious when not
  SILENT    a publisher exists but sent nothing in the sample window
  ABSENT    no publisher at all
"""

import argparse
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from rosidl_runtime_py.utilities import get_message

STATE = [
    "/left/franka_robot_state_broadcaster/measured_joint_states",
    "/left/franka_robot_state_broadcaster/external_joint_torques",
    "/left/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame",
    "/left/gripper/joint_states",
    "/right/franka_robot_state_broadcaster/measured_joint_states",
    "/right/franka_robot_state_broadcaster/external_joint_torques",
    "/right/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame",
    "/right/gripper/joint_states",
    "/mobile_base/pose",
    "/mobile_base/twist",
    "/swerve_drive_controller/cmd_vel_out",
    "/spine/joint_states",
]
ACTION = [
    "/left/gello/joint_states",
    "/right/gello/joint_states",
    "/left/gripper/gripper_client/target_gripper_width_percent",
    "/right/gripper/gripper_client/target_gripper_width_percent",
    "/swerve_drive_controller/cmd_vel",
    "/spine/target_height",
]
VIDEO = [
    "/wrist_camera_left/camera/color/image_raw",
    "/wrist_camera_right/camera/color/image_raw",
    "/head_camera/zed_node/rgb/color/rect/image",
]
LIDAR = ["/lidar_front/scan", "/lidar_rear/scan"]
EXTRA = ["/tf", "/tf_static", "/pedal/state", "/teleop/pedal_mode",
         "/swerve_drive_controller/odom"]


def payload_verdict(msg):
    """Is there anything in here, or is it a structurally-valid empty shell?"""
    # length-bearing payloads: an empty one is a silently dropped dataset column
    for attr in ("data", "position", "ranges", "transforms", "effort"):
        v = getattr(msg, attr, None)
        if v is None:
            continue
        if isinstance(v, (str, bytes, bytearray, list, tuple)) or hasattr(v, "__len__"):
            return "ok" if len(v) > 0 else "EMPTY"
        # scalar `data` (Float32 etc) - any value is real, including 0.0
        return "ok"

    # numeric structures (Twist/Pose/Wrench): all-zero is legal but worth flagging,
    # because an idle robot and a dead publisher look identical here.
    nums = []

    def walk(o, depth=0):
        if depth > 4:
            return
        for f in getattr(o, "get_fields_and_field_types", lambda: {})():
            val = getattr(o, f, None)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                nums.append(float(val))
            elif hasattr(val, "get_fields_and_field_types"):
                walk(val, depth + 1)

    walk(msg)
    if nums and all(abs(n) < 1e-12 for n in nums):
        return "ZEROS"
    return "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--secs", type=float, default=10.0, help="sample window")
    ap.add_argument("--no-video", action="store_true", help="skip the cameras")
    args = ap.parse_args()

    groups = [("STATE", STATE), ("ACTION", ACTION)]
    if not args.no_video:
        groups.append(("VIDEO", VIDEO))
    groups += [("LIDAR (optional)", LIDAR), ("EXTRA (optional)", EXTRA)]
    required = set(STATE) | set(ACTION) | (set() if args.no_video else set(VIDEO))

    rclpy.init()
    node = Node("verify_topics")

    # Let discovery settle before resolving types - Fast DDS is slow to converge here and
    # a short wait reports half the graph as ABSENT.
    deadline = time.time() + 8.0
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)

    available = dict(node.get_topic_names_and_types())
    counts, last = {}, {}

    def make_cb(topic):
        def cb(msg):
            counts[topic] = counts.get(topic, 0) + 1
            last[topic] = msg
        return cb

    subs = 0
    for _, topics in groups:
        for t in topics:
            types = available.get(t)
            if not types:
                continue
            try:
                cls = get_message(types[0])
            except Exception:
                continue
            # BEST_EFFORT, always. Reliability is matched offered>=requested, so a
            # BEST_EFFORT subscriber receives from BOTH reliable and best-effort
            # publishers - while a RELIABLE subscriber silently receives NOTHING from a
            # best-effort one. franka_robot_state_broadcaster publishes best-effort, and
            # an earlier RELIABLE version of this script reported all six of its topics
            # as SILENT on a perfectly healthy robot, with only a QoS warning to show
            # for it. Do not "tighten" this back to RELIABLE.
            #
            # tf_static and the latched mode topic are TRANSIENT_LOCAL; a VOLATILE
            # subscriber never sees their retained sample and would report SILENT.
            depth = 10
            qos = QoSProfile(depth=depth, history=HistoryPolicy.KEEP_LAST,
                             reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL
                             if t in ("/tf_static", "/teleop/pedal_mode")
                             else DurabilityPolicy.VOLATILE)
            node.create_subscription(cls, t, make_cb(t), qos)
            subs += 1

    print(f"subscribed to {subs} topics from ONE participant; sampling {args.secs:.0f}s\n")
    end = time.time() + args.secs
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)

    bad_required, warn = [], []
    for name, topics in groups:
        print(f"{name}")
        for t in topics:
            n = counts.get(t, 0)
            if t not in available:
                verdict, detail = "ABSENT", "no publisher"
            elif n == 0:
                verdict, detail = "SILENT", "publisher exists, sent nothing"
            else:
                verdict = payload_verdict(last[t])
                detail = f"{n} msgs, {n / args.secs:5.1f} Hz"
            flag = " " if verdict == "ok" else "!"
            print(f"  {flag} {t:<62s} {verdict:<7s} {detail}")
            if verdict in ("ABSENT", "SILENT", "EMPTY"):
                (bad_required if t in required else warn).append((t, verdict))
            elif verdict == "ZEROS":
                warn.append((t, verdict))
        print()

    node.destroy_node()
    rclpy.try_shutdown()

    if bad_required:
        print(f"{len(bad_required)} REQUIRED topic(s) would record no usable data:")
        for t, v in bad_required:
            print(f"  {v:<7s} {t}")
        print("\nRecording now produces an episode LABS cannot convert.")
    else:
        print("All required topics are publishing real data.")
    if warn:
        print(f"\n{len(warn)} non-blocking note(s):")
        for t, v in warn:
            print(f"  {v:<7s} {t}")
        print("  ZEROS on cmd_vel / twist / pose is normal while the robot is idle -")
        print("  re-check with the pedals pressed if you expect motion in the episode.")
    return 1 if bad_required else 0


if __name__ == "__main__":
    raise SystemExit(main())
