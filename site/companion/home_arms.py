#!/usr/bin/env python3
"""Send BOTH arms to the teleop home pose from a SINGLE ROS node.

WHY THIS EXISTS

The previous version shelled out to `ros2 action send_goal` once per arm. Each call creates
a new DDS participant, and participant discovery on this robot is a 15-25 s exchange over a
link already carrying three 1 kHz Franka FCI streams. The second call's burst repeatedly
lost the first one's reply:

    [right.action_server.rclcpp_action]: Failed to send goal response ... (timeout):
    client will not receive response

That is the same failure that made two `ros2 control set_controller_state` calls kill the
first arm - see activate_arms.py. The fix is the same: ONE participant, and discover
everything BEFORE any arm starts moving.

Retries here are cheap because they reuse the existing participant: a lost goal response is
a messaging failure, not a motion failure, and re-sending an already satisfied PTP goal is
harmless because the arm is already there.

USAGE
    python3 ~/home_arms.py [--file ~/teleop_home_pose.yaml] [--side left|right]

Called by start_robot.bash after the grippers and before the base. THIS MOVES THE ARMS.
"""
import argparse
import sys
import time

import rclpy
import yaml
from rclpy.action import ActionClient
from rclpy.node import Node

from controller_manager_msgs.srv import SwitchController
from sensor_msgs.msg import JointState
from franka_msgs.action import PTPMotion

CONTROLLER = "joint_impedance_controller"
DISCOVERY_TIMEOUT = 15.0  # servers are up by the time this runs; 15 s is generous
MOTION_TIMEOUT = 30.0     # measured: ~15 s at 0.15 rad/s from ~2 rad away
MAX_VEL = 0.15          # rad/s; ~15 s over the ~2 rad from a typical pose
GOAL_TOLERANCE = 0.01   # rad
ATTEMPTS = 2
AT_HOME_TOL = 0.05      # rad; skip homing if every joint is already this close
STATE_TIMEOUT = 3.0     # s to wait for one /<side>/franka/joint_states message


def current_q(node, side):
    """Current joint1..7 for `side`, or None if no joint_states arrived in time.

    /<side>/franka/joint_states is NOT ordered joint1..7 - joint_state_publisher aggregates
    and does not sort. Observed orders were [1,2,3,4,5,7,6] on the left and [3,4,7,6,2,1,5]
    on the right. Reading the array positionally silently returns another joint's value, so
    always zip name with position and order by the jointN suffix.
    """
    got = {}

    def cb(msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            got[name] = pos

    sub = node.create_subscription(JointState, f"/{side}/franka/joint_states", cb, 10)
    deadline = time.time() + STATE_TIMEOUT
    try:
        while time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            picked = [v for k, v in sorted(got.items()) if "joint" in k]
            if len(picked) >= 7:
                break
    finally:
        node.destroy_subscription(sub)

    ordered = []
    for i in range(1, 8):
        match = [v for k, v in got.items() if k.endswith(f"joint{i}")]
        if len(match) != 1:
            return None
        ordered.append(match[0])
    return ordered


def spin_until(node, fut, timeout):
    rclpy.spin_until_future_complete(node, fut, timeout_sec=timeout)
    return fut.result()


def home_side(node, side, switch_cli, act_cli, q, force=False) -> bool:
    log = node.get_logger()

    # Skip a needless PTP goal when the arm is already there. This is not only faster: every
    # PTP goal is another FCI Move cycle, and those are what trip the
    # communication_constraints_violation reflex on this robot.
    if not force:
        now = current_q(node, side)
        if now is None:
            log.warn(f"{side}: could not read joint_states; homing anyway")
        else:
            err = max(abs(a - b) for a, b in zip(now, q))
            if err <= AT_HOME_TOL:
                log.info(f"{side}: already at home (max joint error {err:.4f} rad); skipping.")
                return True
            log.info(f"{side}: max joint error {err:.4f} rad from home; homing.")

    # joint_impedance_controller holds the command interfaces, so PTP cannot run while it
    # is active. It spawns inactive, so this is normally a no-op and matters on a re-run.
    req = SwitchController.Request()
    req.deactivate_controllers = [CONTROLLER]
    req.strictness = SwitchController.Request.BEST_EFFORT
    spin_until(node, switch_cli.call_async(req), 10.0)

    goal = PTPMotion.Goal()
    goal.goal_joint_configuration = list(q)
    goal.maximum_joint_velocities = [MAX_VEL] * len(q)
    goal.goal_tolerance = GOAL_TOLERANCE

    for attempt in range(1, ATTEMPTS + 1):
        gh = spin_until(node, act_cli.send_goal_async(goal), 10.0)
        if gh is None or not gh.accepted:
            log.warn(f"{side}: goal not accepted (attempt {attempt}/{ATTEMPTS})")
            continue
        res = spin_until(node, gh.get_result_async(), MOTION_TIMEOUT)
        if res is None:
            # Classic lost-response case: the motion may well have completed.
            log.warn(f"{side}: no result received (attempt {attempt}/{ATTEMPTS}); "
                     "the motion may still have completed")
            continue
        err = getattr(res.result, "error_message", "") or ""
        if err:
            log.error(f"{side}: PTP reported: {err}")
            return False
        log.info(f"{side}: at home.")
        return True

    log.error(f"{side}: homing did not confirm after {ATTEMPTS} attempts. Verify the pose "
              f"by hand before activating impedance control; "
              f"/{side}/action_server/error_recovery is available.")
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="/home/tmr-user/teleop_home_pose.yaml")
    ap.add_argument("--side", choices=["left", "right"], action="append",
                    help="home only this side (repeatable); default both")
    ap.add_argument("--force", action="store_true",
                    help="send the PTP goal even if the arm is already at home")
    args = ap.parse_args()

    sides = args.side or ["left", "right"]
    poses = yaml.safe_load(open(args.file))

    rclpy.init()
    node: Node = rclpy.create_node("home_arms")
    log = node.get_logger()

    switch, act, q = {}, {}, {}
    for s in sides:
        key = s.upper()
        if key not in poses or "arm_joint_positions" not in poses[key]:
            log.error(f"{args.file} has no {key}.arm_joint_positions")
            return 1
        q[s] = poses[key]["arm_joint_positions"]
        if len(q[s]) != 7:
            log.error(f"{key}.arm_joint_positions has {len(q[s])} values, expected 7")
            return 1
        switch[s] = node.create_client(SwitchController,
                                       f"/{s}/controller_manager/switch_controller")
        act[s] = ActionClient(node, PTPMotion, f"/{s}/action_server/ptp_motion")

    # Discover EVERYTHING first: no new DDS discovery may happen once an arm is moving.
    for s in sides:
        log.info(f"waiting for {s} controller_manager and ptp_motion ...")
        if not switch[s].wait_for_service(timeout_sec=DISCOVERY_TIMEOUT):
            log.error(f"/{s}/controller_manager/switch_controller not available")
            return 1
        if not act[s].wait_for_server(timeout_sec=DISCOVERY_TIMEOUT):
            log.error(f"/{s}/action_server/ptp_motion not available")
            return 1
    log.info("all servers discovered; homing now")

    rc = 0
    for s in sides:
        if not home_side(node, s, switch[s], act[s], q[s], force=args.force):
            rc = 1

    node.destroy_node()
    rclpy.shutdown()
    return rc


if __name__ == "__main__":
    # Without this, Ctrl+C during a spin raises inside the retry loop and the script simply
    # moves on to the next attempt - which is what made homing feel frozen on 2026-08-23.
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nhoming interrupted", file=sys.stderr)
        try:
            rclpy.shutdown()
        except Exception:
            pass
        sys.exit(130)
