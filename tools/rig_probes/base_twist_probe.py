#!/usr/bin/env python3
"""Move the real base through camelo's own publisher — the WP1 on-site check.

THIS MOVES THE ROBOT. Stand clear; default is 2 s forward at 0.1 m/s
(~86 mm with the controller's ramp), then the same back.

    python3 tools/rig_probes/base_twist_probe.py                  # forward + back
    python3 tools/rig_probes/base_twist_probe.py --x 0.05 --s 3   # slower, longer
    python3 tools/rig_probes/base_twist_probe.py --yaw 0.1 --no-return

Same contract as the site's base_nudge.py (TwistStamped, fresh stamps, 20 Hz,
zeros on the way out), but through ``CommandPublisher`` on the real TopicMap —
so what this proves is the path the perception approach will use, not just
that the base can move. Prints the odometry delta over the move (expect the
commanded distance within a few cm) and what ``cmd_vel_out`` echoed.
Refuses to command while nothing subscribes to the command topic.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from camelo import contracts as C
from camelo.control.chunk_executor import Command


def _xy_yaw(state):
    x, y, yaw = (float(v) for v in state[C.S_BASE_ODOM])
    return x, y, yaw


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--x", type=float, default=0.1, help="m/s forward (+) / back (-)")
    ap.add_argument("--y", type=float, default=0.0, help="m/s left (+) / right (-)")
    ap.add_argument("--yaw", type=float, default=0.0, help="rad/s CCW (+)")
    ap.add_argument("--s", type=float, default=2.0, help="seconds per leg")
    ap.add_argument("--no-return", action="store_true", help="do not drive the inverse leg")
    ap.add_argument("--rate", type=float, default=C.REAL_BASE_CMD_HZ)
    args = ap.parse_args()

    from camelo.ros.command_publisher import CommandPublisher
    from camelo.ros.obs_collector import ObsCollector
    from camelo.ros.session import ros_session
    from camelo.runner.episode_runner import wait_for_obs

    topics = C.topics_for("real")
    legs = [(args.x, args.y, args.yaw)]
    if not args.no_return:
        legs.append((-args.x, -args.y, -args.yaw))
    with ros_session("camelo_base_twist_probe", world="real") as node:
        collector = ObsCollector(node, camera_keys=[], topics=topics)
        publisher = CommandPublisher(node, topics=topics)
        try:
            obs = wait_for_obs(collector, require_images=False)
            deadline = time.monotonic() + 5.0
            while publisher.base_subscribed() is False and time.monotonic() < deadline:
                time.sleep(0.1)
            if publisher.base_subscribed() is False:
                print(f"nothing subscribes to {topics.base_cmd} — is start_base running? refusing")
                return 3
            x0, y0, yaw0 = _xy_yaw(obs.state)
            if not all(math.isfinite(v) for v in (x0, y0, yaw0)):
                print("no odometry in the state (S_BASE_ODOM not finite) — refusing")
                return 3
            print(f"wire: {topics.base_cmd} stamped={topics.base_cmd_stamped} rate={args.rate} Hz")
            print(f"odom start x={x0:.3f} y={y0:.3f} yaw={yaw0:.3f}")
            arms = np.zeros(7, dtype=np.float32)  # never published: publish_arms holds
            period = 1.0 / max(args.rate, 1.0)
            for vx, vy, wz in legs:
                print(f"leg: vx={vx} vy={vy} wz={wz} for {args.s} s")
                end = time.monotonic() + args.s
                while time.monotonic() < end:
                    publisher.publish(Command(arms, arms, 1.0, 1.0, (vx, vy, wz), "NONE"))
                    time.sleep(period)
                publisher.safe_stop()
                time.sleep(0.5)
                obs = collector.get_obs(require_images=False)
                x, y, yaw = _xy_yaw(obs.state) if obs is not None else (math.nan,) * 3
                print(
                    f"  odom now x={x:.3f} y={y:.3f} yaw={yaw:.3f}  "
                    f"delta={math.hypot(x - x0, y - y0):.3f} m, {yaw - yaw0:+.3f} rad  "
                    f"cmd_vel_out={getattr(collector, 'cmd_vel', None)}"
                )
            print("stats:", publisher.publish_stats())
            return 0
        finally:
            publisher.safe_stop()
            publisher.close()
            collector.close()


if __name__ == "__main__":
    raise SystemExit(main())
