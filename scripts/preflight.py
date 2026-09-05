#!/usr/bin/env python3
"""Recording-session preflight: refuse to record into a session that will be wasted.

    python scripts/preflight.py                      # before a recording block
    python scripts/preflight.py --dataset <path>     # after, on what was recorded

Every check here exists because the failure it catches is SILENT — the sim
keeps running, the recorder keeps writing, and the benchmark's own validator
still reports "All checks passed" (DGX_FINDINGS.md F-47/F-48):

  cameras   the render pipeline can collapse mid-session with no error and no
            log line; physics keeps running (the clock roughly DOUBLES) while
            every RGB stream goes silent. Recording then yields episodes with
            no usable video.
  spine     has no ROS command path and resets with the scene, so a restart
            silently drops it to the USD default. Two throwaway episodes were
            recorded at 0.0000 m instead of the frozen SOP height and passed
            validation, because the validator has no notion of the SOP.
  state     a non-finite state column poisons every downstream consumer.

--dataset re-checks the same spine fact against what actually landed on disk,
which is the only proof that matters once the episodes exist.
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from camelo import contracts as C
from camelo.cli import setup_logging

# The frozen SOP height (docs/CONTRACTS.md). Kept here as a default only;
# --spine overrides it for a deliberately different session.
# 0.50 m since 2026-08-07 (was 0.38): the reference Task 2 demos on the Hub
# work at 0.500 m — 96.8 % of 20 869 frames sit exactly there — so matching it
# keeps our episodes geometrically comparable with theirs (F-58).
SPINE_SOP_M = 0.50
SPINE_TOL_M = 0.005

# The spine drive does not reach its target: measured sat ~15 mm below
# commanded at the old 0.38 setpoint (0.3652, F-57). The SOP is defined on the
# COMMANDED value, because that is what `action[19]` records (contracts.py
# reads the applied command, falling back to measured). So the live gate checks
# commanded, and treats the measured gap as a separate "is the drive tracking
# at all" signal — a dead drive shows up as a large gap, not a small offset.
# Droop may differ at 0.50; the band below is deliberately generous.
SPINE_DROOP_WARN_M = 0.05


def _spine_commanded(node, timeout_s: float = 5.0) -> float | None:
    """Latest spine target off the applied-commands topic — the value that
    lands in `action[19]`, and therefore the one the SOP is defined on."""
    import rclpy
    from sensor_msgs.msg import JointState

    seen: list[float] = []

    def _cb(msg):
        if C.SPINE_JOINT in msg.name:
            seen.append(float(msg.position[msg.name.index(C.SPINE_JOINT)]))

    sub = node.create_subscription(JointState, C.APPLIED_COMMANDS_TOPIC, _cb, 10)
    try:
        deadline = time.monotonic() + timeout_s
        while not seen and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
        return seen[-1] if seen else None
    finally:
        node.destroy_subscription(sub)


def _fail(msg: str) -> None:
    print(f"  FAIL  {msg}")


def _ok(msg: str) -> None:
    print(f"  ok    {msg}")


def check_live(seconds: float, spine_target: float, cameras: list[str], world: str) -> int:
    from camelo.ros.obs_collector import ObsCollector
    from camelo.ros.session import ros_session

    topics = C.topics_for(world)
    problems: list[str] = []
    with ros_session("camelo_preflight", world=world) as node:
        collector = ObsCollector(node, camera_keys=cameras, topics=topics)
        try:
            print(f"sampling {seconds:.0f}s (world={world}) ...\n")
            time.sleep(seconds)
            rates = collector.rates()
            obs = collector.get_obs(require_images=False)

            print("cameras")
            for key in collector.camera_keys:
                rate = rates.get(f"image_{key}", 0.0)
                if rate > 0.5:
                    _ok(f"{key:<12} {rate:.1f} Hz")
                else:
                    _fail(f"{key:<12} {rate:.1f} Hz — render pipeline may have collapsed")
                    problems.append(f"camera {key} silent")

            clock = rates.get("clock", 0.0)
            print("\nclock")
            # Real-robot t_sim is a 50 Hz ROS-time timer, not the sim clock.
            if world == "sim" and clock > 15.0:
                _fail(f"{clock:.1f} Hz — suspiciously fast; is anything rendering?")
                problems.append("clock abnormally fast (rendering may have stopped)")
            else:
                _ok(f"{clock:.1f} Hz")

            print("\nstate")
            min_finite = C.STATE_DIM if world == "sim" else C.STATE_DIM - 14
            if obs is None:
                _fail("no observation (clock or joint_states_full missing)")
                problems.append("no observation")
            else:
                finite = int(np.isfinite(obs.state).sum())
                if finite >= min_finite:
                    _ok(f"{finite}/{C.STATE_DIM} finite")
                else:
                    bad = np.where(~np.isfinite(obs.state))[0].tolist()
                    _fail(f"{finite}/{C.STATE_DIM} finite — non-finite indices {bad}")
                    problems.append("non-finite state")

                print("\nspine")
                measured = float(obs.state[C.S_SPINE])
                if world == "real":
                    if abs(measured - spine_target) <= SPINE_DROOP_WARN_M:
                        _ok(f"measured  {measured:.4f} m (SOP {spine_target:.4f} m)")
                    else:
                        _fail(
                            f"measured {measured:.4f} m but SOP is {spine_target:.4f} m"
                        )
                        problems.append("spine not at SOP")
                else:
                    commanded = _spine_commanded(node)
                    if commanded is None:
                        _fail("no spine target on the applied-commands topic")
                        problems.append("no spine command")
                    elif abs(commanded - spine_target) <= SPINE_TOL_M:
                        _ok(f"commanded {commanded:.4f} m (SOP {spine_target:.4f} m)")
                    else:
                        _fail(
                            f"commanded {commanded:.4f} m but SOP is {spine_target:.4f} m — "
                            "launch the scene with "
                            f"--spine-keyboard-min {spine_target} --spine-keyboard-max "
                            f"{spine_target}, or arrow it there in the sim window"
                        )
                        problems.append("spine not at SOP")

                    if commanded is not None:
                        gap = commanded - measured
                        if abs(gap) <= SPINE_DROOP_WARN_M:
                            _ok(f"measured  {measured:.4f} m (droop {gap:+.4f} m, expected)")
                        else:
                            _fail(
                                f"measured {measured:.4f} m is {gap:+.4f} m off the "
                                "commanded target — the spine drive is not tracking"
                            )
                            problems.append("spine drive not tracking")
        finally:
            collector.close()  # reap camera workers or we never exit (F-32b)

    return _verdict(problems)


def check_dataset(path: Path, spine_target: float) -> int:
    import pandas as pd

    # Only data/ — meta/ parquets have a different schema and break the stack.
    files = sorted(glob.glob(str(path / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        print(f"  FAIL  no data parquet under {path}/data")
        return 1

    frame = pd.concat([pd.read_parquet(f) for f in files])
    actions = np.stack(frame["action"].to_numpy())
    states = np.stack(frame["observation.state"].to_numpy())
    episodes = sorted(int(e) for e in frame["episode_index"].unique())
    print(f"{path.name}: {len(actions)} frames, episodes {episodes}\n")

    problems: list[str] = []

    print("spine")
    spine = actions[:, C.A_SPINE]
    if spine.std() > 1e-3:
        _fail(f"NOT constant: min={spine.min():.4f} max={spine.max():.4f} — "
              "the height moved mid-dataset; episodes are geometrically inconsistent")
        problems.append("spine varies within dataset")
    elif abs(float(np.nanmean(spine)) - spine_target) > SPINE_TOL_M:
        _fail(f"constant at {float(np.nanmean(spine)):.4f} m but SOP is "
              f"{spine_target:.4f} m — re-record")
        problems.append("spine not at SOP")
    else:
        _ok(f"constant at {float(np.nanmean(spine)):.4f} m")

    print("\nfinite")
    n_bad = int(np.isnan(actions).sum()) + int(np.isnan(states).sum())
    if n_bad:
        _fail(f"{n_bad} NaNs across action/state")
        problems.append("NaNs")
    else:
        _ok("no NaNs in action or state")

    # Not a gate — the grippers legitimately stay put in some episodes, but a
    # dataset with no gripper motion at all usually means a teleop mis-wire.
    print("\ngrippers (informational)")
    for name, idx in (("left", C.A_LEFT_GRIP), ("right", C.A_RIGHT_GRIP)):
        col = actions[:, idx]
        print(f"  {name:<6} range [{col.min():.2f}, {col.max():.2f}]")

    return _verdict(problems)


def _verdict(problems: list[str]) -> int:
    if problems:
        print(f"\nPREFLIGHT FAILED: {'; '.join(problems)}")
        return 1
    print("\nPREFLIGHT OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=None,
                        help="check a recorded dataset instead of the live sim")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--spine", type=float, default=SPINE_SOP_M,
                        help=f"expected spine height in m (default: SOP {SPINE_SOP_M})")
    parser.add_argument("--cameras", default="head,wrist_left,wrist_right")
    parser.add_argument(
        "--world",
        choices=("sim", "real"),
        default=C.default_world(),
        help="topic contract (sim = /isaac/*; real = record_bag.bash)",
    )
    args = parser.parse_args()
    setup_logging()

    if args.dataset is not None:
        return check_dataset(args.dataset, args.spine)
    cameras = [k.strip() for k in args.cameras.split(",") if k.strip()]
    return check_live(args.seconds, args.spine, cameras, args.world)


if __name__ == "__main__":
    sys.exit(main())
