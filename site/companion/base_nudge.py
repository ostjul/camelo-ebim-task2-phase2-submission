#!/usr/bin/env python3
"""Drive the TMR base directly for a fixed time, bypassing the pedals entirely.

THIS MOVES THE ROBOT. Stand clear and keep the duration short.

    python3 configs/base_nudge.py                 # 2 s forward at 0.1 m/s (~86 mm)
    python3 configs/base_nudge.py --y 0.1         # strafe left
    python3 configs/base_nudge.py --yaw 0.1       # rotate CCW
    python3 configs/base_nudge.py --duration 5

Why this exists rather than `ros2 topic pub`: SwerveDriveController ages every command
against its own clock (swerve_drive_controller.cpp:78) and silently substitutes zeros for
anything older than cmd_vel_timeout, 0.5 s. `ros2 topic pub` leaves header.stamp at 0, so
the age computes as ~1.8 billion seconds and EVERY message is discarded - the topic looks
perfectly healthy at 20 Hz and the base never moves. This stamps each message, exactly as
base_bridge does.

It also publishes zeros on the way out, so the base stops immediately rather than coasting
until the watchdog fires.
"""

import argparse
import time

import rclpy
from geometry_msgs.msg import TwistStamped

# The controller clamps to these and limits acceleration to 0.1 m/s^2 / 0.1 rad/s^2,
# so larger values here buy nothing.
MAX_LINEAR = 0.1
MAX_ANGULAR = 0.1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x", type=float, default=0.0, help="m/s forward (+) / back (-)")
    parser.add_argument("--y", type=float, default=0.0, help="m/s left (+) / right (-)")
    parser.add_argument("--yaw", type=float, default=0.0, help="rad/s CCW (+) / CW (-)")
    parser.add_argument("--duration", type=float, default=2.0, help="seconds to command")
    parser.add_argument("--rate", type=float, default=20.0, help="Hz")
    parser.add_argument("--topic", default="/swerve_drive_controller/cmd_vel")
    args = parser.parse_args()

    for name, value, limit in (("x", args.x, MAX_LINEAR),
                               ("y", args.y, MAX_LINEAR),
                               ("yaw", args.yaw, MAX_ANGULAR)):
        if abs(value) > limit:
            print(f"note: {name}={value} exceeds the controller's {limit} ceiling; it will be clamped.")

    rclpy.init()
    node = rclpy.create_node("base_nudge")
    pub = node.create_publisher(TwistStamped, args.topic, 10)

    # Give DDS time to match the controller's subscription, or the first commands are
    # published into the void and the move is shorter than requested.
    deadline = time.time() + 5.0
    while pub.get_subscription_count() == 0 and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if pub.get_subscription_count() == 0:
        print(f"WARNING: nothing is subscribed to {args.topic}. Is the base bringup running,")
        print("         and on the same ROS_DOMAIN_ID? Commanding anyway.")

    def send(x, y, yaw):
        msg = TwistStamped()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.twist.linear.x = x
        msg.twist.linear.y = y
        msg.twist.angular.z = yaw
        pub.publish(msg)

    period = 1.0 / max(args.rate, 1.0)
    print(f"Commanding x={args.x} y={args.y} yaw={args.yaw} for {args.duration}s "
          f"at {args.rate} Hz. Ctrl+C stops.")
    end = time.time() + args.duration
    try:
        while time.time() < end:
            send(args.x, args.y, args.yaw)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(period)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        # Explicit stop: without it the base holds the last command until the 0.5 s
        # watchdog fires, which is another ~5 cm of travel.
        for _ in range(int(0.3 * args.rate) or 1):
            send(0.0, 0.0, 0.0)
            time.sleep(period)
        print("stopped (zeros sent).")
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
