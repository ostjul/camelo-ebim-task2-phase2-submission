#!/usr/bin/env python3
"""M1.2 — prove the command path: wiggle wrist joints, cycle grippers.

    python scripts/sine_actor.py --seconds 20 --amplitude 0.05

Requires the task2 sim (NO keyboard-teleop flags, --no-browser) and the
helper stack in position mode. Watch the sim viewer: both arms' last two
joints oscillate and the grippers slowly open/close. Note which direction
gripper=1.0 moves — first data point of the polarity checklist
(docs/runbooks/DATA_COLLECTION.md).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from camelo import contracts as C
from camelo.cli import setup_logging
from camelo.contracts import default_world, topics_for


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--amplitude", type=float, default=0.05, help="rad")
    parser.add_argument("--period", type=float, default=5.0, help="s per sine cycle")
    parser.add_argument("--gripper-period", type=float, default=10.0)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument(
        "--world",
        choices=("sim", "real"),
        default=default_world(),
        help="topic contract (sim = /bridge/*; real = record_bag.bash GELLO topics)",
    )
    args = parser.parse_args()
    setup_logging()

    from camelo.ros.command_publisher import CommandPublisher, browser_conflict
    from camelo.ros.obs_collector import ObsCollector
    from camelo.ros.session import ros_session
    from camelo.runner.episode_runner import wait_for_obs

    topics = topics_for(args.world)
    with ros_session("camelo_sine_actor", world=args.world) as node:
        conflicts = browser_conflict(node, topics)
        if conflicts:
            print(f"WARNING: browser controller is publishing on {conflicts} — "
                  "it will fight this actor; restart the helper stack with --no-browser")
        # No images needed (joint state only) — skip the camera workers.
        collector = ObsCollector(node, camera_keys=[], topics=topics)
        publisher = CommandPublisher(node, topics=topics)
        obs = wait_for_obs(collector)
        left0 = obs.state[C.S_LEFT_ARM].copy()
        right0 = obs.state[C.S_RIGHT_ARM].copy()
        print(f"start pose captured at t_sim={obs.t_sim:.2f}; wiggling for {args.seconds:.0f}s")

        t_wall0 = time.monotonic()
        try:
            while time.monotonic() - t_wall0 < args.seconds:
                t = time.monotonic() - t_wall0
                offset = args.amplitude * math.sin(2.0 * math.pi * t / args.period)
                left = left0.copy()
                right = right0.copy()
                left[-2:] += offset
                right[-2:] += offset
                opening = 0.5 + 0.5 * math.cos(2.0 * math.pi * t / args.gripper_period)
                publisher.publish_arms(left, right)
                publisher.publish_grippers(opening, opening)
                time.sleep(1.0 / args.rate)
        finally:
            publisher.publish_arms(left0, right0)
            publisher.safe_stop()
            collector.close()  # uniform teardown (F-32b)

        final = collector.get_obs()
        if final is not None:
            drift = np.max(np.abs(final.state[C.S_LEFT_ARM] - left0))
            print(f"done; left-arm max offset from start: {drift:.3f} rad")
        return 0


if __name__ == "__main__":
    sys.exit(main())
