#!/usr/bin/env python3
"""Per-episode spine-ramp trim point — measured, never a blanket frame count.

    python scripts/analyze_ramp_trim.py outputs/datasets/ext_fixpos200_train_v1

Recorded episodes start with the spine at 0.000 and ramp to its plateau
over the first seconds. Eval never reproduces that: the scene pins the
spine at launch and the rollout starts after it settles, so those leading
frames show arm heights the policy cannot encounter at inference.

Trimming them is worth doing and easy to overdo. A single frame count
applied to every episode is wrong twice over — the ramp is a physical
settle whose duration varies, and this corpus deliberately spans several
plateau heights (0.45-0.55, kept as augmentation), so "reaches 0.50" is
not the test either. This measures each episode against **its own**
plateau.

Two safety properties, because over-trimming silently deletes task:
  * the trim point must sit well before that episode's first gripper
    motion — the report fails loudly if the margin is thin;
  * per-episode trims are reported with their spread, so one anomalous
    episode cannot hide inside an average.

Read-only. Prints what a trim WOULD remove; applying it is a separate
step (and, for a video dataset, a re-encode).
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camelo import contracts as C  # noqa: E402

# Fraction of the plateau value the measured spine must be within, and how
# many consecutive frames must hold it, before we call the ramp finished.
SETTLE_TOL_M = 0.005
SETTLE_HOLD = 15
# Refuse to call a trim safe if it lands within this many seconds of the
# episode's first gripper motion.
MIN_MARGIN_S = 3.0


def settle_frame(spine: np.ndarray) -> tuple[int, float]:
    """First frame from which the spine stays within tolerance of its own
    plateau. Returns (frame, plateau)."""
    plateau = float(np.median(spine[len(spine) // 2 :]))
    close = np.abs(spine - plateau) < SETTLE_TOL_M
    # Walk back from the end so a brief early touch of the plateau during
    # the ramp cannot be mistaken for having settled.
    settled = len(spine)
    run = 0
    for i in range(len(spine) - 1, -1, -1):
        if close[i]:
            run += 1
            if run >= SETTLE_HOLD:
                settled = i
        else:
            run = 0
            if settled < len(spine):
                break
    return settled, plateau


def first_gripper_motion(action: np.ndarray) -> int | None:
    grip = action[:, C.A_RIGHT_GRIP]
    moved = np.where(np.abs(grip - grip[0]) > 0.05)[0]
    return int(moved[0]) if len(moved) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--verbose", action="store_true", help="per-episode rows")
    args = parser.parse_args()

    import pandas as pd

    files = sorted(glob.glob(str(args.dataset / "data" / "chunk-*" / "*.parquet")))
    if not files:
        print(f"error: no data parquet under {args.dataset}", file=sys.stderr)
        return 2
    frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    fps = 30.0

    rows = []
    for episode, group in frame.groupby("episode_index"):
        state = np.stack(group["observation.state"].to_numpy())
        action = np.stack(group["action"].to_numpy())
        cut, plateau = settle_frame(state[:, C.S_SPINE])
        grasp = first_gripper_motion(action)
        margin = ((grasp - cut) / fps) if grasp is not None else float("inf")
        rows.append((int(episode), len(group), cut, plateau, grasp, margin))

    cuts = np.array([r[2] for r in rows])
    lengths = np.array([r[1] for r in rows])
    margins = np.array([r[5] for r in rows])
    plateaus = np.array([r[3] for r in rows])

    if args.verbose:
        header = f"{'ep':>4} {'len':>5} {'cut':>5} {'cut_s':>6} {'plateau':>8}"
        print(header + f" {'grasp':>6} {'margin_s':>9}")
        for ep, n, cut, plateau, grasp, margin in rows:
            print(f"{ep:>4} {n:>5} {cut:>5} {cut / fps:>6.2f} {plateau:>8.4f} "
                  f"{(grasp if grasp is not None else -1):>6} {margin:>9.2f}")
        print()

    total = int(lengths.sum())
    trimmed = int(cuts.sum())
    print(f"episodes                 {len(rows)}")
    print(f"frames total             {total}")
    print(f"frames a trim removes    {trimmed}  ({trimmed / total * 100:.1f} %)")
    print(f"frames remaining         {total - trimmed}")
    print(f"per-episode cut frames   min {cuts.min()}  median {int(np.median(cuts))}"
          f"  max {cuts.max()}")
    print(f"per-episode cut seconds  min {cuts.min() / fps:.2f}  median {np.median(cuts) / fps:.2f}"
          f"  max {cuts.max() / fps:.2f}")
    print(f"plateau heights          min {plateaus.min():.4f}  max {plateaus.max():.4f}"
          f"  distinct(2dp) {len(set(np.round(plateaus, 2)))}")

    print("\n-- safety --")
    worst = margins.min()
    print(f"tightest margin to first gripper motion: {worst:.2f} s "
          f"(floor {MIN_MARGIN_S:.1f} s)")
    tight = [r[0] for r in rows if r[5] < MIN_MARGIN_S]
    if tight:
        print(f"FAIL: {len(tight)} episode(s) trim within {MIN_MARGIN_S} s "
              f"of the grasp: {tight[:10]}")
        return 1
    print("ok: every episode's trim lands well before its first gripper motion")

    # A cut far from the pack is the signature of a mis-detected plateau.
    spread = cuts.max() - int(np.median(cuts))
    if spread > 3 * fps:
        outliers = [r[0] for r in rows if r[2] > np.median(cuts) + 3 * fps]
        print(f"WARNING: {len(outliers)} episode(s) cut >3 s beyond the median "
              f"— inspect {outliers[:10]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
