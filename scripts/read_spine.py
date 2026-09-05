#!/usr/bin/env python3
"""Print the live Franka spine height and compare it with the corpus value.

Read-only: one subscription to /spine/joint_states (joint ``spine_z``, metres,
published by the station's spine_state_publisher) and, if nothing arrives,
one call to /franka_spine_node/get_position. The task-2 corpus recorded a
constant spine target of 434 (mm, spine reach 0..770 mm) on every frame of all
238 episodes (docs/realdata/16 R-69), i.e. 0.434 m.

    python -u scripts/read_spine.py            # pixi shell on ebimHP (section A env)
"""

from __future__ import annotations

import argparse
import sys
import time

CORPUS_SPINE_M = 0.434


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--topic", default="/spine/joint_states")
    ap.add_argument("--service", default="/franka_spine_node/get_position")
    ap.add_argument("--seconds", type=float, default=5.0, help="listen this long")
    ap.add_argument("--tol-m", type=float, default=0.01)
    args = ap.parse_args()

    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState

    rclpy.init()
    node = rclpy.create_node("camelo_read_spine")
    seen: list[tuple[str, float]] = []

    def cb(msg: JointState) -> None:
        for n, p in zip(msg.name, msg.position, strict=False):
            seen.append((n, float(p)))

    qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
    node.create_subscription(JointState, args.topic, cb, qos)
    t0 = time.monotonic()
    while time.monotonic() - t0 < args.seconds and not seen:
        rclpy.spin_once(node, timeout_sec=0.2)

    height = None
    if seen:
        name, height = seen[-1]
        dt = time.monotonic() - t0
        print(f"{args.topic}: {name} = {height:.4f} m  ({len(seen)} samples in {dt:.1f}s)")
    else:
        print(f"no message on {args.topic} in {args.seconds:.0f}s — trying {args.service}")
        try:
            from franka_spine_msgs.srv import GetPosition
        except ImportError as e:
            print(f"cannot import franka_spine_msgs: {e}")
        else:
            cli = node.create_client(GetPosition, args.service)
            if cli.wait_for_service(timeout_sec=5.0):
                fut = cli.call_async(GetPosition.Request())
                rclpy.spin_until_future_complete(node, fut, timeout_sec=5.0)
                resp = fut.result()
                print(f"{args.service} -> {resp}")
                for field in ("position", "height", "position_in_m", "position_in_mm"):
                    if resp is not None and hasattr(resp, field):
                        v = float(getattr(resp, field))
                        height = v / 1000.0 if "mm" in field or v > 5 else v
                        break
            else:
                print(f"service {args.service} not available")
    node.destroy_node()
    rclpy.shutdown()

    if height is None:
        print("SPINE HEIGHT UNKNOWN")
        return 2
    diff = height - CORPUS_SPINE_M
    verdict = "OK" if abs(diff) <= args.tol_m else "MISMATCH"
    print(
        f"spine {height * 1000:.0f} mm vs corpus {CORPUS_SPINE_M * 1000:.0f} mm: "
        f"{diff * 1000:+.0f} mm -> {verdict}"
    )
    return 0 if verdict == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
