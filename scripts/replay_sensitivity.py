#!/usr/bin/env python3
"""Offline: how far does end-effector placement drift as a policy's
self-disagreement grows? No sim, no GPU, no server.

    python scripts/replay_sensitivity.py \
        --dataset outputs/datasets/ext_fixpos200_trim_model16_v1

## The question this exists to answer

Every rung scores IoU 0.0. The dataset's OWN actions, replayed through the
same executor and scoring service, score 0.586 (F-82). Between those two
numbers sits an untested assumption: that the 0.0 reflects policy quality
at all. Every intervention that provably improved a checkpoint left the
score at exactly 0.0 — the adapter repair halved the sampling spread and
improved 22/22 episodes; both start-pose fixes raised pad engagement;
merging, re-normalizing and re-widening the state all landed on 0.0. A
metric that does not move when the thing it measures demonstrably improves
is a metric worth calibrating before trusting.

So: take the action stream that DOES score, corrupt it by a measured
amount, and see how far the gripper ends up from where it should be. That
gives a displacement-vs-noise curve, and the checkpoints' own measured
spreads can be read off it.

## What it does NOT do

It does not compute IoU. IoU comes from the eval camera in Isaac Sim and
cannot be had offline. This reports **TCP displacement in millimetres**,
which is upstream of IoU: the pad has to be carried to roughly the right
place before overlap is possible. Converting mm to IoU needs the rig, or
at minimum the pad footprint. Treat the output as necessary-not-sufficient
— a displacement far larger than the pad cannot score, but a small one is
not proof that it would.

The other half of the calibration needs the rig and is worth stating
because this script cannot substitute for it: **replay the demo actions
from the POLICY's starting condition.** F-82's 0.586 was obtained with
`--start-frame auto` (seek the frame whose recorded spine matches live,
frame 169) and `--goto-start` (walk the arms to that frame's recorded
joints, converged to 0.0148 rad). The oracle was therefore handed an
in-distribution start; a policy under `recenter_arms` is not, and that
condition appears in 0 of 173 769 recorded frames. If the oracle replay
ALSO scores 0.0 when started where the policy starts, the 0.0 says nothing
about any checkpoint.

## Noise model, and the conversion that matters

Noise is injected **per replan**, on the arm columns of each chunk the
executor is handed — that is what a sampling policy does: a fresh, slightly
different chunk every time it is asked. Injecting per-tick noise instead
would model sensor jitter, which is not the failure mode here.

`spread` as reported by `dataset_action_probe.py` is a **max of ranges over
`--repeat` samples**, NOT a standard deviation. For n samples from
N(0, sigma) the expected range is ~3.735 sigma at n=20, so a spread of
0.553 rad is sigma ~= 0.148 rad. Passing the spread straight in as sigma
overstates the noise ~3.7x. `--spread` does the conversion; `--sigma`
takes the raw value if you want it.

Measured spreads at repeat=20, for reference (job 3817905, 22 episodes):
    pi0.5 baseline  1.0150      lora_v2  0.5531      expert  0.5032
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camelo import contracts as C
from camelo.control.chunk_executor import DATASET_FPS, ChunkExecutor
from camelo.control.kinematics import MobileFR3Kinematics

# Expected range of n draws from a standard normal. Used to convert a
# probe `spread` (max of ranges) into the sigma of the underlying noise.
# n=20 is the frozen probe protocol; the others are here so a spread taken
# at another repeat count is not silently misconverted.
_RANGE_OVER_SIGMA = {2: 1.128, 5: 2.326, 10: 3.078, 20: 3.735, 50: 4.498}

# Eval pins both of these, so they are the right frame to answer an
# eval-relevant question in. Base is F-69's measured stage-2 pose; spine is
# the SOP plateau (contracts.SPINE_SOP_M is the commanded 0.50, the
# measured settle is ~0.4852).
EVAL_BASE_XY_YAW = np.array([2.100, 3.051, np.deg2rad(-90.0)])
EVAL_SPINE_M = 0.4852


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


def replay(
    actions: np.ndarray,
    ticks_per_sim_s: float,
    sigma: float,
    rng: np.random.Generator,
    replan_steps: int = 8,
    max_delta: float = 0.05,
    chunk_size: int = 50,
    fps: float = DATASET_FPS,
    mode: str = "perchunk",
) -> np.ndarray:
    """Replay `actions`, perturbing each replanned chunk by N(0, sigma).

    Returns (executed right-arm command stream [T, 7], clamp fraction).
    The stream is the executor's own integrated output, which is what the
    robot would actually have tracked. sigma=0 reproduces
    `executor_probe.oracle_rollout`'s command stream exactly.
    """
    executor = ChunkExecutor(max_delta_per_tick=max_delta, replan_after_steps=replan_steps)
    state = np.zeros(C.STATE_DIM, dtype=np.float32)
    state[C.S_LEFT_ARM] = actions[0][C.A_LEFT_ARM]
    state[C.S_RIGHT_ARM] = actions[0][C.A_RIGHT_ARM]

    executed: list[np.ndarray] = []
    t_sim, tick = 0.0, 1.0 / ticks_per_sim_s
    horizon = max(len(actions) - chunk_size, 1) / fps
    while t_sim < horizon:
        if executor.needs_replan(t_sim):
            start = int(t_sim * fps)
            chunk = actions[start : start + chunk_size].copy()
            if sigma > 0 and len(chunk):
                # Arms only. The grippers are binary and the base/spine are
                # not commandable, so perturbing them models nothing.
                for sl in (C.A_LEFT_ARM, C.A_RIGHT_ARM):
                    if mode == "perchunk":
                        # One draw per joint per chunk, held across it. This
                        # is what `spread` actually measures: how far apart
                        # two SAMPLED CHUNKS are, not frame-to-frame jitter
                        # within one. A sampled chunk is a smooth trajectory.
                        chunk[:, sl] += rng.normal(0.0, sigma, (chunk[:, sl].shape[1],))
                    else:
                        chunk[:, sl] += rng.normal(0.0, sigma, chunk[:, sl].shape)
            executor.set_chunk(t_sim, chunk, 1.0 / fps)
        command = executor.step(t_sim, state)
        if command is not None:
            executed.append(np.asarray(command.right_arm, dtype=np.float64))
        t_sim += tick
    stream = np.stack(executed) if executed else np.zeros((0, 7))
    stats = executor.stats
    clamp = stats.clamped_ticks / stats.ticks if stats.ticks else float("nan")
    return stream, clamp


def tcp_track(kin: MobileFR3Kinematics, arm_stream: np.ndarray) -> np.ndarray:
    """[T, 7] joint commands -> [T, 3] world TCP xyz."""
    return np.stack(
        [kin.fk(q, EVAL_BASE_XY_YAW, EVAL_SPINE_M)[0] for q in arm_stream]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--rate", type=float, default=43.7, help="ticks per SIM second (F-91)")
    parser.add_argument("--replan-steps", type=int, default=8)
    parser.add_argument("--max-delta", type=float, default=0.05)
    parser.add_argument("--trials", type=int, default=8, help="noise draws per (episode, level)")
    parser.add_argument("--repeat", type=int, default=20, help="probe repeat the spreads came from")
    parser.add_argument(
        "--spread",
        default="0,0.1,0.25,0.5031,0.5531,1.0150,2.0",
        help="probe spreads (max-of-range) to sweep; converted to sigma internally",
    )
    parser.add_argument("--sigma", default=None, help="raw sigmas instead, no conversion")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--noise-mode",
        default="perchunk",
        choices=("perchunk", "perframe"),
        help="perchunk: one offset per joint per chunk (models sampling spread, "
        "the faithful default). perframe: i.i.d. per frame — models jitter, "
        "saturates the rate limiter far earlier, and overstates clamping.",
    )
    args = parser.parse_args()

    if args.repeat not in _RANGE_OVER_SIGMA and args.sigma is None:
        raise SystemExit(
            f"no range/sigma factor for repeat={args.repeat}; "
            f"known: {sorted(_RANGE_OVER_SIGMA)} — or pass --sigma directly"
        )
    if args.sigma is not None:
        levels = [(float(s), float(s)) for s in args.sigma.split(",")]
        label = "sigma"
    else:
        factor = _RANGE_OVER_SIGMA[args.repeat]
        levels = [(float(s), float(s) / factor) for s in args.spread.split(",")]
        label = "spread"

    episodes = episode_actions(args.dataset, args.episodes)
    kin = MobileFR3Kinematics()
    print(f"{args.dataset.name}: {len(episodes)} episodes, rate {args.rate} ticks/sim-s")
    print(f"noise mode: {args.noise_mode}")
    print(f"noise injected per replan on arm columns; {label} -> sigma via /{_RANGE_OVER_SIGMA.get(args.repeat, 1):.3f}")
    print(f"FK frame: base {EVAL_BASE_XY_YAW[:2]} yaw -90deg, spine {EVAL_SPINE_M} m\n")

    # sigma=0 reference per episode
    ref, ref_clamp = [], []
    for a in episodes:
        stream, clamp = replay(a, args.rate, 0.0, np.random.default_rng(0),
                               args.replan_steps, args.max_delta, mode=args.noise_mode)
        ref.append(tcp_track(kin, stream)); ref_clamp.append(clamp)
    print(f"noise-free clamp fraction: {np.mean(ref_clamp):.1%}\n")

    print(f"{'spread':>8}{'sigma':>8}{'final mm':>11}{'p50 mm':>9}{'max mm':>9}{'clamped':>10}")
    print("-" * 55)
    for shown, sigma in levels:
        finals, medians, maxes, clamps = [], [], [], []
        for ei, actions in enumerate(episodes):
            for t in range(args.trials if sigma > 0 else 1):
                rng = np.random.default_rng(args.seed + 1000 * ei + t)
                stream, clamp = replay(actions, args.rate, sigma, rng,
                                       args.replan_steps, args.max_delta,
                                       mode=args.noise_mode)
                track = tcp_track(kin, stream); clamps.append(clamp)
                n = min(len(track), len(ref[ei]))
                if n == 0:
                    continue
                d = np.linalg.norm(track[:n] - ref[ei][:n], axis=1) * 1000.0
                finals.append(d[-1]); medians.append(np.median(d)); maxes.append(d.max())
        print(f"{shown:8.4f}{sigma:8.4f}{np.mean(finals):11.1f}"
              f"{np.mean(medians):9.1f}{np.mean(maxes):9.1f}{np.mean(clamps):10.1%}")

    print("\nfinal mm = TCP displacement from the noise-free replay at the last tick")
    print("           (the place moment — what determines where the pad is put down)")
    print("This is displacement, NOT IoU. Large displacement cannot score; small")
    print("displacement is necessary but not sufficient. See the module docstring")
    print("for the rig-side half of the calibration (replay from the policy's start).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
