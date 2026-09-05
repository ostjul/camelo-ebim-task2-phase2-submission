#!/usr/bin/env python3
"""The topic list a rig bag must carry for camelo to replay it offline.

DERIVED from ``camelo.contracts.topics_for("real")``, never hardcoded. A bag
is only useful for debugging the runner if it carries exactly what
``ObsCollector`` subscribes to; a hand-maintained second copy of the manifest
would drift from the contract silently, and the failure mode is the one
AGENTS.md's drift-alarm rule exists to prevent — an empty column that reads
as "the topic was quiet", not as "we never recorded it".

Groups:
  state   joint states (both arms, both grippers, spine), wrenches, odom,
          applied cmd_vel — everything ObsCollector reads.
  action  the GELLO / gripper / base / spine COMMAND topics, so a bag can
          also be replayed as a demonstration.
  video   the three camera image topics.
  extra   /tf + /tf_static. Deliberately NOT /pedal/state or
          /teleop/pedal_mode: those carry site-custom message types, which
          would make the bag unplayable on a stock ROS install (see
          tools/rig_probes/play_rig_bag.sh).

Every type emitted here is a stock sensor_msgs / geometry_msgs / nav_msgs /
std_msgs / tf2_msgs type, so `ros2 bag play` works off-rig with no franka,
robotiq or LABS packages installed.

    python3 tools/rig_probes/rig_bag_topics.py --groups state,video
    python3 tools/rig_probes/rig_bag_topics.py --cameras head
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from camelo import contracts as C  # noqa: E402

GROUPS = ("state", "action", "video", "extra")


def topics(groups: list[str], cameras: list[str]) -> list[str]:
    tm = C.topics_for(C.WORLD_REAL)
    out: list[str] = []

    if "state" in groups:
        out += [t for t, _side in tm.joint_state_topics]
        # Wrench is 12 of the 27 dims s27a15 feeds the policy and the adapter
        # raises on a missing group rather than substituting zeros (F-63) —
        # a bag without these cannot drive a real-layout rollout at all.
        out += [t for t in (tm.left_wrench, tm.right_wrench) if t]
        out += [t for t in (tm.odom, tm.cmd_vel_applied) if t]

    if "action" in groups:
        out += [
            tm.left_arm_cmd,
            tm.right_arm_cmd,
            tm.left_gripper_cmd,
            tm.right_gripper_cmd,
        ]
        out += [t for t in (tm.base_cmd, tm.spine_cmd) if t]

    if "video" in groups:
        for key in cameras:
            if key not in tm.cameras:
                raise SystemExit(
                    f"unknown camera {key!r}; contract has {list(tm.cameras)}"
                )
            out.append(tm.cameras[key]["image_topic"])

    if "extra" in groups:
        out += ["/tf", "/tf_static"]

    # Preserve order, drop duplicates (a topic can serve two groups).
    seen: set[str] = set()
    return [t for t in out if not (t in seen or seen.add(t))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--groups",
        default="state,action,video,extra",
        help=f"comma-separated subset of {GROUPS} (default: all)",
    )
    ap.add_argument(
        "--cameras",
        default=",".join(C.CAMERA_KEYS),
        help="comma-separated camera keys, or 'none' (default: all three)",
    )
    ap.add_argument(
        "--bytes-per-second",
        action="store_true",
        help="print the estimated image bandwidth instead of the topic list",
    )
    args = ap.parse_args()

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    bad = [g for g in groups if g not in GROUPS]
    if bad:
        raise SystemExit(f"unknown group(s) {bad}; choose from {GROUPS}")
    cams = (
        []
        if args.cameras.strip().lower() in ("", "none")
        else [c.strip() for c in args.cameras.split(",") if c.strip()]
    )

    if args.bytes_per_second:
        # Raw sensor_msgs/Image is uncompressed, so h*w*3 per frame is exact.
        # 30 Hz is the wire rate camelo's DDS profile actually delivers (R-49);
        # without that profile the wrists arrive at 10-15 Hz and this
        # over-estimates, which is the safe direction for a disk check.
        tm = C.topics_for(C.WORLD_REAL)
        total = 0
        for key in cams:
            h, w, ch = tm.cameras[key]["shape"]
            total += h * w * ch * 30
        print(total)
        return 0

    for t in topics(groups, cams):
        print(t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
