#!/usr/bin/env python3
"""M1.8 — batch scored evaluation: N episodes of reset -> rollout -> IoU.

    python scripts/eval_batch.py --adapter pi0 --backend remote \
        --server ws://h100:8765 --episodes 5 --rollout-s 120

Requires: task2 sim with --record and --no-browser, helper stack in
position mode, AND the eval stack
(bash <benchmark>/scripts/evaluation/task2/run.sh up). Writes
outputs/eval/<run>/results.csv + summary.json — the experiment ledger —
plus episode_***/eval_camera.mp4 (and demo_cam.mp4 when
/isaac/demo_cam/image_raw is published — optional, skipped otherwise).

**Sim only.** `--world real` is refused before any ROS session is created
(`refuse_real_world` below): this script has none of the real-robot arm
choreography, and the real-robot rung is `run_policy.py --world real`.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camelo.cli import (
    action_space_from_args,
    add_policy_args,
    cameras_from_args,
    make_approach_from_args,
    make_backend_from_args,
    make_executor_from_args,
    make_grasp_gate_from_args,
    make_grasp_observer_from_args,
    make_start_pose_from_args,
    setup_logging,
    topics_from_args,
)


def refuse_real_world(args) -> None:
    """`--world real` is not implemented here — say so before anything runs.

    This script has NONE of the real-robot arm choreography that
    `run_policy.py --world real` performs (`camelo/runner/real_arms.py`):
    no hold before activation, no `list_controllers` wait, no keep-alive
    ownership, no deactivate-confirm-then-stop, and `reset_scene()` /
    `verify_gripper_polarity()` below both command poses that only exist in
    sim. On the real map the topics resolve and every publisher constructs
    happily, so the run would LOOK correct: the arms would sit inactive
    while the gripper gate dwells, or — worse, with the controllers already
    active — the batch would drive the sim's `ARM_READY_POSE` into two
    Franka arms and then drop the command stream at the end of every
    episode, which is the 2.0 s liveness fault that takes the whole arm
    launch down.

    Refusing before the first `ros_session` keeps that from being a live
    discovery: nothing is published, nothing is even created.
    """
    if getattr(args, "world", "sim") == "real":
        raise SystemExit(
            "eval_batch has no real-robot arm choreography; use "
            "run_policy.py --world real. Scored batch evaluation needs the "
            "sim's scene reset and IoU scoring, neither of which exists on "
            "the rig — the real-robot rung is a single rollout "
            "(docs/setup/SETUP.md §7.1)"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_policy_args(parser)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--rollout-s", type=float, default=120.0, help="sim seconds per episode")
    parser.add_argument("--output", type=Path, default=Path("outputs/eval"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--skip-polarity-check", action="store_true")
    args = parser.parse_args()
    setup_logging()
    refuse_real_world(args)

    from camelo.ros.command_publisher import (
        CommandPublisher,
        browser_conflict,
        verify_gripper_polarity,
    )
    from camelo.ros.obs_collector import ObsCollector
    from camelo.ros.session import ros_session
    from camelo.runner.batch_eval import format_diagnostics, run_batch
    from camelo.runner.episode_runner import EpisodeRunner, wait_for_obs

    backend = make_backend_from_args(args)
    topics = topics_from_args(args)
    with ros_session("camelo_eval_batch", world=args.world) as node:
        conflicts = browser_conflict(node, topics)
        if conflicts:
            print(f"WARNING: browser controller live on {conflicts}; use --no-browser")
        collector = ObsCollector(node, camera_keys=cameras_from_args(args), topics=topics)
        publisher = CommandPublisher(node, topics=topics)
        try:
            _gate = make_grasp_gate_from_args(args)
            runner = EpisodeRunner(
                node=node,
                collector=collector,
                publisher=publisher,
                executor=make_executor_from_args(args),
                backend=backend,
                task=args.task,
                approach=make_approach_from_args(args),
                approach_timeout_s=args.approach_timeout,
                start_pose=make_start_pose_from_args(args),
                start_pose_tol=args.start_pose_tol,
                start_pose_check=args.start_pose_check,
                action_space=action_space_from_args(args, backend),
                grasp_gate=_gate,
                grasp_observer=make_grasp_observer_from_args(args, _gate),
            )
            if not args.skip_polarity_check:
                wait_for_obs(collector)
                # Reset BEFORE the gate, not after (F-80). The previous rung
                # can leave the gripper driver jammed outside [0, 0.8] — a
                # pi0 run left it at -0.886 rad — and the gate then fails on
                # stale state instead of a real polarity fault, killing the
                # run before episode 0. A reset clears it, so the gate
                # measures THIS run's stack rather than the last one's mess.
                runner.reset_scene()
                verify_gripper_polarity(collector, publisher)
            summary = run_batch(
                runner, args.episodes, args.rollout_s, args.output,
                rate_hz=args.rate, run_name=args.run_name,
            )
            print(f"mean IoU over {summary['scored']} scored episodes: {summary['mean_iou']}")
            wall = summary.get("mean_rollout_wall_s")
            sim = summary.get("mean_rollout_sim_s")
            if wall is not None and sim is not None:
                print(
                    f"mean rollout time: {wall:.1f} wall s / {sim:.1f} sim s "
                    f"({sim / wall:.2f}x sim/wall)"
                )
            elif wall is not None:
                print(f"mean rollout time: {wall:.1f} wall s")
            diagnostics = format_diagnostics(summary)
            if diagnostics:
                print(diagnostics)
            sys.stdout.flush()
            return 0
        finally:
            backend.close()
            collector.close()  # reap camera workers + queues or the process never exits (F-32b)


if __name__ == "__main__":
    # rclpy / multiprocessing resource-tracker teardown can hang after a
    # finished batch (F-32b class) even when collector.close() ran — force
    # the process to end so launch_policy.sh and docker compose return.
    # os._exit skips interpreter teardown, which also means it skips the
    # default traceback print AND the stdout flush — an exception in main()
    # would otherwise exit 1 with an EMPTY log, indistinguishable from a kill,
    # and `--help`/argparse errors printed nothing at all. Handle both: let a
    # SystemExit carry its own code (that is --help and our own refusals,
    # which are not failures), print anything else, and always flush.
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
    except BaseException:
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
