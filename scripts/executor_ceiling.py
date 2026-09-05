#!/usr/bin/env python3
"""Offline: the best IoU-relevant tracking ANY checkpoint could get, per
control rate. No sim, no GPU, no server.

    python scripts/executor_ceiling.py --dataset outputs/datasets/ext_fixpos200_trim_model16_v1

Replays the recorded demo actions through the real ChunkExecutor with a
perfect policy (see camelo/control/executor_probe.py). Everything it
reports is a harness property: if the demos themselves cannot be executed
at the control rate a policy achieves on the rig, no amount of training
will produce a good rollout, and a large `max_requested_delta` is evidence
about latency rather than about the checkpoint.

Read it alongside the measured ticks/episode from a scored run
(`summary.json: mean_control_hz_sim`) — that is the row that applies to you.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camelo.control.executor_probe import demo_joint_speeds, oracle_rollout

# Control rates measured on the DGX rig, one per rung (F-88/F-91), in
# **ticks per SIM second** — the only unit in which this comparison is
# valid, because demo speeds are rad per sim second.
#
# ⚠️ TICKS PER WALL SECOND IS A DIFFERENT NUMBER AND USING IT HERE INVERTS
# THE CONCLUSION (F-91, and it cost us a round trip). The loop paces itself
# on wall time while the sim runs ~6.5x slower, so roughly 6.5x MORE ticks
# fit into one sim second than into one wall second: pi0.5's clean run is
# 6.7 ticks/wall-s but 43.7 ticks/sim-s. Divide ticks by `sim_seconds`,
# never by `wall_seconds` — `summary.json: mean_control_hz_sim` already does.
#
# Nominal ceiling for reference: 20 Hz wall at sim/wall 0.153 is ~131
# ticks/sim-s, i.e. a 6.55 rad/s budget. Synchronous inference is what
# separates that from the 43.7 actually achieved.
MEASURED_RATES = (
    (75.0, "act"),
    (71.2, "smolvla"),
    (45.9, "pi0"),
    (43.7, "pi05"),
    (10.5, "pi05-degraded"),
)


def episode_actions(root: Path, limit: int) -> list[np.ndarray]:
    import pyarrow.parquet as pq

    files = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"no parquet under {root}/data")
    episodes: list[np.ndarray] = []
    for path in files:
        table = pq.read_table(path, columns=["action", "episode_index"])
        index = np.asarray(table["episode_index"])
        actions = np.stack([np.asarray(r) for r in table["action"].to_pylist()])
        for episode in np.unique(index):
            episodes.append(actions[index == episode].astype(np.float32))
            if len(episodes) >= limit:
                return episodes
    return episodes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-delta", type=float, default=0.05)
    parser.add_argument("--replan-steps", type=int, default=8)
    parser.add_argument(
        "--rates",
        default=None,
        help="Comma-separated ticks/SIM-second to report instead of the measured "
        "per-rung table, e.g. '60' for a rate no rung has been measured at. Same "
        "unit warning applies — sim seconds, never wall (F-91).",
    )
    args = parser.parse_args()

    rates = MEASURED_RATES
    if args.rates:
        rates = tuple((float(r), "custom") for r in args.rates.split(","))

    episodes = episode_actions(args.dataset, args.episodes)
    print(f"{args.dataset.name}: {len(episodes)} episodes\n")

    speeds = np.concatenate([demo_joint_speeds(a) for a in episodes])
    print("what the DEMOS ask of the arms (worst joint per frame, rad/s)")
    for q in (50, 90, 99):
        print(f"  p{q:<3} {np.percentile(speeds, q):6.3f}")
    print(f"  max  {speeds.max():6.3f}")

    print(f"\nwhat the EXECUTOR can deliver at {args.max_delta} rad/tick")
    for hz, _ in rates:
        budget = args.max_delta * hz
        over = (speeds > budget).mean()
        print(f"  {hz:5.1f} Hz -> {budget:5.3f} rad/s ; {over:6.2%} of demo frames exceed it")

    def mean(rows: list[dict], key: str) -> float:
        return float(np.mean([r[key] for r in rows if r[key] is not None]))

    print(f"\nORACLE POLICY through the real executor (replan={args.replan_steps})")
    header = (
        f"{'Hz sim':>7} {'rung':>8} {'ticks':>7} {'clamped':>8} "
        f"{'maxdelta':>9} {'track p99':>10}"
    )
    print(header)
    print("-" * len(header))
    for hz, rung in rates:
        rows = [
            oracle_rollout(
                a,
                hz,
                replan_steps=args.replan_steps,
                max_delta=args.max_delta,
                # The degraded row is deliberately below the nominal loop
                # rate — that is the point of including it.
                allow_below_nominal=hz < 20.0,
            )
            for a in episodes
        ]
        print(
            f"{hz:7.1f} {rung:>8} {mean(rows, 'ticks'):7.0f} "
            f"{mean(rows, 'clamp_fraction'):7.1%} "
            f"{mean(rows, 'max_requested_delta'):9.3f} {mean(rows, 'track_p99'):10.3f}"
        )
    print(
        "\nThese are FLOORS on clamping and CEILINGS on tracking: a real policy "
        "also deviates from the demos, which only adds to both."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
