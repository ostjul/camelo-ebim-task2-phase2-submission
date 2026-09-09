#!/usr/bin/env python3
"""M1.6/M1.7 — run a policy against the live sim (no scoring, no resets).

    python scripts/run_policy.py --adapter dummy --seconds 30          # smoke test
    python scripts/run_policy.py --adapter pi0 --backend remote \
        --server ws://h100:8765 --seconds 60

Requires: task2 sim up WITHOUT keyboard-teleop flags and with --no-browser
(plus --record if you want resets/eval later), helper stack in position
mode. For --backend local the model must be installed here; for remote,
start scripts/serve_policy.py near the GPU first.

Real robot (`--world real`, TMR station topics from record_bag.bash) adds
the arm choreography, in this order and with the command stream never
interrupted (docs/setup/SETUP.md §7):

    verify --start-pose (nothing is commanded)
      -> warm the backend up (reset + ONE discarded inference: the first one
         of a run is seconds slow, and paying it after the switch leaves the
         controller holding a repeat, docs/realdata/16 U-28)
      -> hold the MEASURED pose at --rate (a zero-delta GELLO stand-in)
      -> joint_impedance_controller goes active (operator, or --activate-arms)
      -> policy
      -> deactivate, confirm, then stop publishing

    python3 -u scripts/run_policy.py --world real --backend remote \
        --server ws://gpu:8765 --action-layout s27a15 --state-layout s27a15 \
        --start-pose file:outputs/rig/home_pose.json --seconds 120
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
    arms_from_args,
    async_inference_from_args,
    cameras_from_args,
    chunk_time_base_from_args,
    gello_directions_from_args,
    make_approach_from_args,
    make_backend_from_args,
    make_executor_from_args,
    make_gripper_latch_from_args,
    make_start_pose_from_args,
    max_image_age_from_args,
    max_state_age_from_args,
    setup_logging,
    topics_from_args,
)

#: Same wording as ObsCollector's log.error (camelo/ros/obs_collector.py) —
#: one drift alarm, two surfaces (the log at first-mismatch time, this
#: refusal at rollout-start time).
CAMERA_SHAPE_MISMATCH_HINT = (
    "the station's ZED is not at the corpus resolution (docs/realdata/16 "
    "R-42); fix the launcher (resolution HD720), do not resize here"
)


def refuse_camera_shape_mismatch(collector, allow: bool) -> str | None:
    """The refusal message for a real-world camera shape drift, or ``None``.

    Real world only: ``collector.camera_shape_mismatch`` is populated by
    ObsCollector the moment a camera's measured frame shape disagrees with
    the contract's declared shape (a policy trained on the corpus's 720x1280
    would otherwise silently run on whatever the station happens to publish
    -- e.g. 376x672 after the 2026-08-31 VGA change, docs/realdata/16 R-42).
    ``allow`` is ``args.allow_camera_shape_mismatch`` -- an explicit opt-in
    to run anyway.
    """
    mismatches = getattr(collector, "camera_shape_mismatch", None) or {}
    if not mismatches or allow:
        return None
    lines = []
    for key, measured in sorted(mismatches.items()):
        expected = tuple(collector.topics.cameras[key]["shape"])
        lines.append(
            f"camera {key} shape {tuple(measured)} != contract {expected} — "
            f"{CAMERA_SHAPE_MISMATCH_HINT}"
        )
    lines.append(
        "Pass --allow-camera-shape-mismatch to run anyway (runs the policy "
        "on frames it was not trained on)."
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_policy_args(parser)
    parser.add_argument("--seconds", type=float, default=30.0, help="rollout length [sim s]")
    args = parser.parse_args()
    setup_logging()

    from camelo.ros.command_publisher import CommandPublisher, browser_conflict
    from camelo.ros.obs_collector import ObsCollector
    from camelo.ros.session import ros_session
    from camelo.runner.episode_runner import run_rollout, wait_for_obs

    backend = make_backend_from_args(args)
    if args.dummy_obs:
        from camelo.policy.dummy_obs import FixedObsBackend, layout_for_args

        backend = FixedObsBackend(
            backend, layout_for_args(args.action_layout, args.state_layout)
        )
        print(
            "WARNING --dummy-obs: the policy is being fed a FIXED, FAKE observation "
            "(all-zero joints/wrench, mid-grey images) — NOT real sensor data. Real "
            "measured state still drives the start-pose check, the executor's "
            "max-delta clamp, keepalive, and deactivate-on-exit, so the robot WILL "
            "move, safely rate/delta-limited, but on whatever the checkpoint "
            "predicts from this fake input rather than anything sensible."
        )
    topics = topics_from_args(args)
    real = topics.world == "real"
    action_space = action_space_from_args(args, backend)
    arms = arms_from_args(args)
    start_pose = make_start_pose_from_args(args)
    wait_for_activation = (
        real if args.wait_for_activation is None else args.wait_for_activation
    )
    if real:
        # C-11 (docs/realdata/16 R-38): controller_manager_msgs was missing in
        # BOTH rig environments on 2026-09-02 and the runner's activation
        # path died at import time. Refuse here, before any DDS participant.
        from camelo.ros.real_imports import missing_real_imports, report_real_imports

        if missing_real_imports():
            print(report_real_imports())
            return 4
    print(
        f"world={topics.world}  arm_cmd={topics.left_arm_cmd}  "
        f"head_cam={topics.cameras['head']['image_topic']}  "
        f"action_space={action_space}"
    )
    with ros_session("camelo_policy_bridge", world=args.world) as node:
        conflicts = browser_conflict(node, topics)
        if conflicts:
            print(f"WARNING: browser controller live on {conflicts} — it will fight "
                  "the policy; restart the helper stack with --no-browser")
        collector = ObsCollector(
            node,
            # dummy-obs never looks at real images (the policy gets fake ones
            # instead) — don't subscribe to cameras, and don't let
            # wait_for_obs() block on a stream this run doesn't need.
            camera_keys=[] if args.dummy_obs else cameras_from_args(args),
            topics=topics,
            wrench_declared_absent=real and not args.wrench,
        )
        publisher = CommandPublisher(
            node,
            topics=topics,
            arm_command_frame=args.arm_command_frame,
            gello_joint_directions=gello_directions_from_args(args),
            keepalive_hz=args.keepalive_hz,
        )
        executor = make_executor_from_args(args)
        session = None
        stats = None
        first_obs = None
        if real:
            # First complete observation, before any activation/hold: a
            # camera shape drift (docs/realdata/16 R-42) means whatever the
            # policy is about to see is already off-contract, which is
            # worth refusing before the arm is ever touched.
            # Kept, not discarded: this is also the observation the backend
            # warm-up runs on (below), images included — the first inference
            # has to be paid on a payload the same size as the rollout's.
            first_obs = wait_for_obs(collector)
            refusal = refuse_camera_shape_mismatch(
                collector, args.allow_camera_shape_mismatch
            )
            if refusal is not None:
                print(refusal)
                publisher.close()
                collector.close()
                return 4
            # Service clients on THIS node, created before anything is
            # published: a second DDS participant appearing mid-session is a
            # discovery burst that can fault a live FCI loop.
            controllers = None
            if wait_for_activation or args.activate_arms:
                # Nothing in the hold loop publishes across a blocking
                # controller_manager call (5 s per side, one side after the
                # other) — the publisher's keep-alive thread does.
                # `--keepalive-hz 0` therefore makes the activation itself a
                # >2.0 s liveness gap.
                from camelo.runner.real_arms import keepalive_gap_report

                gap = keepalive_gap_report(publisher, topics)
                if gap is not None:
                    print(f"{gap}. Pass --keepalive-hz 10 (the default), or "
                          "--no-wait-for-activation to run without touching "
                          "the controllers at all")
                    publisher.close()
                    collector.close()
                    return 4
                from camelo.ros.arm_activation import ArmControllers

                controllers = ArmControllers(node, arms=arms)
            from camelo.runner.real_arms import RealArmSession

            session = RealArmSession(
                collector,
                publisher,
                controllers=controllers,
                arms=arms,
                start_pose=start_pose,
                start_pose_tol=args.start_pose_tol,
                start_pose_check=args.start_pose_check,
                wait_for_activation=wait_for_activation,
                activate_arms=args.activate_arms,
                deactivate_on_exit=args.deactivate_on_exit,
                # U-27: refuse the switch when the joint-state stack is
                # already dead, instead of discovering it one rollout tick
                # later (docs/realdata/16 R-64).
                max_state_age_s=max_state_age_from_args(args),
                rate_hz=args.rate,
                min_activation_publish_hz=args.min_activation_publish_hz,
            )
        try:
            if session is not None:
                # verify (never command) -> WARM UP the backend -> hold the
                # measured pose -> active. Inside the try: an operator may
                # have activated one arm before this raised, and that arm
                # still needs deactivating.
                #
                # The warm-up is `backend.reset` plus ONE discarded inference,
                # paid here rather than after the switch: on 2026-09-02 the
                # first remote inference took 1.443 s against 0.33-0.53 s
                # steady state, and paying it AFTER activation left the
                # controller holding a keep-alive repeat for 1.6 s
                # (docs/realdata/16 U-28). Nothing is published from it and
                # the executor never sees the chunk, so the rollout's own
                # t_sim 0 is unchanged.
                from camelo.runner.real_arms import warm_up_backend

                session.prepare(
                    warm_up=lambda: warm_up_backend(backend, first_obs, args.task)
                )
            else:
                backend.reset(args.task)
            stats = run_rollout(
                collector,
                publisher,
                executor,
                backend,
                args.seconds,
                args.rate,
                approach=make_approach_from_args(args),
                approach_timeout_s=args.approach_timeout,
                approach_only=bool(getattr(args, "approach_only", False)),
                action_space=action_space,
                joint_csv=Path(args.joint_csv) if args.joint_csv else None,
                chunk_dump=Path(args.chunk_dump) if args.chunk_dump else None,
                chunk_dump_args=vars(args) if args.chunk_dump else None,
                # Rig-side lever for the state-copying gripper channel
                # (docs/realdata/15_RIG_WINDOW_RUNBOOK.md §3): off unless
                # --gripper-latch is passed.
                gripper_latch=make_gripper_latch_from_args(args),
                left_gripper_hold=(
                    session.left_gripper_hold if session is not None else 1.0
                ),
                # U-23 stale-image guard: real only, 0.5 s by default. Raises
                # StaleImageError out of run_rollout; the finally below is
                # what makes that safe (deactivate, then stop publishing).
                max_image_age_s=max_image_age_from_args(args),
                # U-27 stale-joint-state guard: real only, 0.2 s by default,
                # every tick. Raises StaleStateError out of run_rollout on
                # the same path.
                max_state_age_s=max_state_age_from_args(args),
                # U-29: remote inference off the control thread (ON by
                # default for --backend remote on --world real). The 20 Hz
                # loop keeps commanding through the round trip instead of
                # stopping dead for it.
                async_inference=async_inference_from_args(args),
                # Plan-then-execute (docs/realdata/16 T6): 'arrival' re-bases
                # a synchronous chunk's t0 to the tick it installs on instead
                # of the observation it was computed from, so playback starts
                # at row 0 rather than skipping the rows an inference-bound
                # arm never played. 'observation' (default) is unchanged.
                chunk_time_base=chunk_time_base_from_args(args),
            )
        finally:
            # Real: deactivate while STILL publishing, confirm, then stop.
            # Dropping the stream on an active controller takes the arm
            # launch down, so this must also run on the exception path.
            if session is not None:
                stats_teardown = session.shutdown()
                print(f"real-arm session: {stats_teardown}")
            else:
                publisher.safe_stop()
            backend.close()
            try:
                collector.close()  # reap camera workers or the process never exits (F-32b)
            finally:
                # LAST, and unconditionally: `close()` stops and JOINS the
                # keep-alive thread. It publishes, so it must be finished
                # before `ros_session`'s own teardown destroys the node under
                # it — and it must not stop any EARLIER than this, because
                # `session.shutdown()` above needs the stream alive across
                # the blocking deactivation service call.
                publisher.close()
        print(f"rollout stats: {stats}")
        if stats is not None:
            # T6 (docs/realdata/16_RIG_TEST_PROTOCOL.md): the controller dies
            # after 2.0 s without a sample, so the worst gap in THIS rollout
            # is the margin that was actually left. Printed on its own line
            # rather than left inside the stats dict, where a 2 s gap and a
            # 0.05 s one look identical at a glance.
            gap = stats.get("max_publish_gap_s")
            interval = stats.get("keepalive_max_interval_s")
            first_cmd = stats.get("activation_to_first_cmd_s")

            def _s(value):
                return "n/a" if value is None else format(value, ".3f")

            print(
                "publish telemetry: keepalive_republished="
                f"{stats.get('keepalive_republished')} "
                f"max_publish_gap_s={_s(gap)} "
                # The watchdog's OWN cadence: max_publish_gap_s is kept small
                # by its repeats, so it cannot report a starved watchdog.
                f"keepalive_max_interval_s={_s(interval)} "
                f"activation_to_first_cmd_s={_s(first_cmd)}"
                + (" (controller faults at 2.000)" if real else "")
            )
            # U-32: how far the command stood from the measured pose, and
            # how often the leash had to pull it back. On its own line for
            # the same reason as the gap above — `cmd_lead_rad_max` is the
            # number the chunk splice reasons across, and `leash_rad: None`
            # is what tells a reader a 0 % is "off", not "never needed".
            print(
                "leash telemetry: "
                f"leash_rad={stats.get('leash_rad')} "
                f"leash_active_pct={_s(stats.get('leash_active_pct'))} "
                f"({stats.get('leash_active_ticks')}/{stats.get('ticks')} ticks) "
                f"cmd_lead_rad_max={_s(stats.get('cmd_lead_rad_max'))}"
            )
        sys.stdout.flush()
        return 0


if __name__ == "__main__":
    # os._exit skips the default traceback print as well as teardown; without
    # this an exception exits 1 with an empty log (see eval_batch.py).
    code = 1
    try:
        code = main()
    except SystemExit as exc:  # --help / argparse exit: not a crash, no traceback
        if exc.code is None or isinstance(exc.code, int):
            code = exc.code if isinstance(exc.code, int) else 0
        else:
            print(exc.code, file=sys.stderr)
            code = 1
    except BaseException:
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
