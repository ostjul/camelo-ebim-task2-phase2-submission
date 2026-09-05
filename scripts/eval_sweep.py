#!/usr/bin/env python3
"""Score one checkpoint under several EXECUTOR settings in one sim session.

    python scripts/eval_sweep.py --adapter lerobot \
        --checkpoint outputs/runs/pi05_ft_.../checkpoints/last/pretrained_model \
        --action-layout canonical --state-layout model16 \
        --backend remote --server ws://h100:8765 \
        --replan-steps 8,4,2 --episodes 5 --rollout-s 120

Why this exists: a low IoU has two causes the score cannot separate — the
policy asked for the wrong pose, or the executor could not follow the pose
it asked for. The chunk executor's per-tick clamp is the ONLY rate limiter
in the command path (F-53), and between replans the arm follows a chunk
that was planned from an older observation. Both are eval-side, both are
free to vary, and neither requires retraining. This sweeps them and prints
the clamp/stale counters next to the IoU so the two causes are visible in
the same table.

Sweep --replan-steps UPWARDS (F-88). `backend.infer()` runs synchronously
inside the control loop (`episode_runner.py:97`), so a 1.2 s pi0.5
inference stops the arm each time it fires: replanning more often means
fewer commands per sim second and MORE clamping, not fresher control. The
measured clamp fraction tracks the control rate, not plan age. Chunks are
50 steps, so 32 still leaves headroom before one goes stale — and the
table reports ticks/episode and Hz next to the clamp fraction, without
which two variants are not comparable at all.

Requirements are `eval_batch.py`'s exactly (sim with --record --no-browser,
helper stack in position mode, eval stack up); the whole sweep runs in ONE
ROS session and one polarity check, so variants share a scene and a server.

Variants run as consecutive blocks of `--episodes` episodes, not
interleaved: if you suspect the sim itself drifts over a long session,
re-run the sweep with the variant order reversed rather than reading a
small gap as signal. `--num-inference-steps` is deliberately NOT swept —
it lives in the policy process, so with --backend remote it needs a server
restart; sweep it by re-running this script against a server started with
a different value, under a different --run-name.

Writes outputs/eval/<run>/<variant>/results.csv per variant, plus
outputs/eval/<run>/sweep.json holding every summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camelo.cli import (
    add_policy_args,
    cameras_from_args,
    make_backend_from_args,
    setup_logging,
    topics_from_args,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_policy_args(parser)
    parser.add_argument("--episodes", type=int, default=5, help="episodes PER variant")
    parser.add_argument("--rollout-s", type=float, default=120.0, help="sim seconds per episode")
    parser.add_argument("--output", type=Path, default=Path("outputs/eval"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--skip-polarity-check", action="store_true")
    parser.add_argument(
        "--replan-steps-sweep",
        dest="replan_steps_sweep",
        default=None,
        help="comma list of --replan-steps values (default: the single "
        "--replan-steps value, i.e. no sweep)",
    )
    parser.add_argument(
        "--max-delta-sweep",
        dest="max_delta_sweep",
        default=None,
        help="comma list of --max-delta values, 0.05 or lower (default: the "
        "single --max-delta value)",
    )
    args = parser.parse_args()
    setup_logging()

    from camelo.runner.batch_eval import parse_sweep_variants

    variants = parse_sweep_variants(
        args.replan_steps_sweep or str(args.replan_steps),
        args.max_delta_sweep or str(args.max_delta),
    )
    run_name = args.run_name or time.strftime("sweep_%Y%m%d_%H%M%S")
    print(f"sweep {run_name}: {len(variants)} variants x {args.episodes} episodes")

    from camelo.control.chunk_executor import ChunkExecutor
    from camelo.ros.command_publisher import (
        CommandPublisher,
        browser_conflict,
        verify_gripper_polarity,
    )
    from camelo.ros.obs_collector import ObsCollector
    from camelo.ros.session import ros_session
    from camelo.runner.batch_eval import format_diagnostics, run_batch, sweep_table
    from camelo.runner.episode_runner import EpisodeRunner, wait_for_obs

    sweep_dir = Path(args.output) / run_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    backend = make_backend_from_args(args)
    results: list[tuple[str, dict]] = []
    topics = topics_from_args(args)
    with ros_session("camelo_eval_sweep", world=args.world) as node:
        conflicts = browser_conflict(node, topics)
        if conflicts:
            print(f"WARNING: browser controller live on {conflicts}; use --no-browser")
        collector = ObsCollector(node, camera_keys=cameras_from_args(args), topics=topics)
        publisher = CommandPublisher(node, topics=topics)
        try:
            # Once per session, not once per variant: the gate costs a fixed
            # 8 s dwell and measures the bridge, which no variant changes.
            if not args.skip_polarity_check:
                wait_for_obs(collector)
                verify_gripper_polarity(collector, publisher)

            for variant in variants:
                print(
                    f"\n=== variant {variant['label']}: replan_steps="
                    f"{variant['replan_steps']} max_delta={variant['max_delta']} ==="
                )
                sys.stdout.flush()
                runner = EpisodeRunner(
                    node=node,
                    collector=collector,
                    publisher=publisher,
                    executor=ChunkExecutor(
                        max_delta_per_tick=variant["max_delta"],
                        replan_after_steps=variant["replan_steps"],
                    ),
                    backend=backend,
                    task=args.task,
                )
                summary = run_batch(
                    runner, args.episodes, args.rollout_s, sweep_dir,
                    rate_hz=args.rate, run_name=variant["label"],
                )
                summary["replan_steps"] = variant["replan_steps"]
                summary["max_delta"] = variant["max_delta"]
                results.append((variant["label"], summary))
                print(f"mean IoU: {summary['mean_iou']}")
                diagnostics = format_diagnostics(summary)
                if diagnostics:
                    print(diagnostics)
                sys.stdout.flush()
                # Write after every variant: a sweep that dies on variant 3
                # must not lose variants 1 and 2.
                (sweep_dir / "sweep.json").write_text(
                    json.dumps(
                        {
                            "run": run_name,
                            "task_instruction": args.task,
                            "checkpoint": args.checkpoint,
                            "adapter": args.adapter,
                            "episodes_per_variant": args.episodes,
                            "rollout_s": args.rollout_s,
                            "variants": [s for _, s in results],
                        },
                        indent=2,
                    )
                )
        finally:
            backend.close()
            collector.close()  # reap camera workers + queues or the process never exits (F-32b)

    if results:
        print("\n" + sweep_table(results))
        print(f"\nledger: {sweep_dir}/sweep.json")
    return 0


if __name__ == "__main__":
    # Same teardown hazard as eval_batch.py (F-32b class): rclpy /
    # multiprocessing teardown can hang after a finished batch, so the
    # process is ended by force. os._exit skips atexit AND the stdout
    # flush, hence the explicit flush — without it `--help` and every
    # argparse refusal print nothing at all.
    code = 1
    try:
        code = main()
    except SystemExit as exc:  # --help, argparse errors, our own refusals
        if exc.code is None or isinstance(exc.code, int):
            code = exc.code if isinstance(exc.code, int) else 0
        else:
            # A `raise SystemExit("message")` refusal: os._exit skips the
            # interpreter's own print of it, which silently turned refusals
            # into empty exit-0 "successes" (F-12 class). Print it, fail.
            print(exc.code, file=sys.stderr)
            code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
