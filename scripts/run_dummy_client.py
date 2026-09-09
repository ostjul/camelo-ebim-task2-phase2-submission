#!/usr/bin/env python3
"""Intermediate step toward the real ROS bridge (docs/setup/SETUP.md §7).

A lightweight ROS node that subscribes to NOTHING — no cameras, no joint
states, no clock — and instead sends one fixed, hand-built observation to
a running policy server on a timer, over the exact same wire format
`run_policy.py --backend remote` uses. The returned action chunk is logged
and discarded; nothing is commanded.

Purpose: prove the ROS-process <-> policy-server round trip (connection,
wire format, server-side inference) before wiring in real sensor
subscriptions. Nothing here ever commands the robot.

To actually publish the resulting actions and move the robot from this same
FIXED observation (verifying the command-publish path before real sensors
are wired in), use `run_policy.py --dummy-obs` instead — it keeps the real
safety choreography (start-pose check, controller activation, the executor's
--max-delta clamp anchored on REAL measured state, keepalive,
deactivate-on-exit) and only swaps what the policy itself sees:

    python3 -u scripts/run_policy.py --world real --backend remote \
        --server ws://127.0.0.1:8765 --dummy-obs \
        --action-layout s27a15 --state-layout s27a15 \
        --task "Pick up the thermal pad and place it on the target RAM board" \
        --start-pose file:outputs/rig/home_pose.json --arms left,right \
        --wait-for-activation --arm-command-frame robot \
        --gello-joint-directions -1,-1,1,1,1,1,-1 \
        --rate 20 --replan-steps 8 --max-delta 0.05 --seconds 60

Once real sensors are wired in for good, replace the obs builder here with
`camelo.ros.obs_collector.ObsCollector` (see run_policy.py) and drop
--dummy-obs there too.

    # default layout is s27a15 (the Munich real-robot checkpoints, 27-in/15-out):
    python scripts/run_dummy_client.py --server ws://127.0.0.1:8765 --rate 2 --seconds 60

    # same box, model-free smoke test against the dummy adapter (still s27a15):
    python scripts/serve_policy.py --adapter dummy --port 8765
    python scripts/run_dummy_client.py --server ws://127.0.0.1:8765 --seconds 30

    # a sim-trained (model16/canonical) checkpoint needs --layout canonical to
    # match --action-layout/--state-layout on the server:
    python scripts/run_dummy_client.py --layout canonical \
        --server ws://127.0.0.1:8765 --rate 2 --seconds 60

    # the U-35 wire probe on the SHRUNK wire VLA-JEPA runs with: same
    # recorded frame, resized client-side to the checkpoint's own 224x224
    # before the JPEG. Row 0 must still match the full-resolution probe
    # (the server's own Resize is the identity on an already-224 frame).
    python scripts/run_dummy_client.py --no-ros --server ws://127.0.0.1:8765 \
        --obs-dir outputs/rig/t5/ep163_frames --frame 0 --wire-image-size 224x224

    # from a dev machine with no ROS 2 / rclpy install (no Docker needed
    # either — just `pip install numpy websockets msgpack pillow`): the
    # rclpy node wrapper is skipped entirely, only the wire protocol runs.
    python scripts/run_dummy_client.py --no-ros --server ws://127.0.0.1:8766 --seconds 30
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camelo.cli import add_wire_image_args, setup_logging, wire_images_from_args
from camelo.policy.dummy_obs import ACTION_FORMATTERS, DEFAULT_TASK, OBS_BUILDERS

log = logging.getLogger("camelo.run_dummy_client")


def _run_obs_dir(backend, obs_dir: Path, frame: int, task: str) -> int:
    """U-35 probe: send ONE recorded corpus observation through the real
    client path (RemoteBackend / wire.py — the same JPEG encode / websocket
    / server decode as a live rollout) and diff the returned chunk against
    the demo's own recorded action_chunk. See
    docs/realdata/16d_T6_LEARNINGS.md and camelo/policy/dummy_obs.py."""
    from camelo.policy.adapters import s27a15
    from camelo.policy.dummy_obs import (
        build_obs_from_recorded_frame,
        chunk_diff_verdict,
        load_recorded_frame,
    )

    _manifest, frame_entry = load_recorded_frame(obs_dir, frame)
    obs = build_obs_from_recorded_frame(obs_dir, frame_entry)

    log.info("connecting to %s", backend.url if hasattr(backend, "url") else "<backend>")
    backend.reset(task)
    log.info(
        "reset ok, layout=s27a15 task=%r; sending RECORDED obs frame=%d from %s",
        task,
        frame,
        obs_dir,
    )
    for camera, image in obs.images.items():
        log.info("  image[%-11s] shape=%s dtype=%s", camera, image.shape, image.dtype)
    state27 = np.asarray(frame_entry["observation.state"], dtype=np.float64)
    log.info(
        "  observation.state (27-dim): %s",
        np.array2string(state27, precision=4, suppress_small=True, max_line_width=200),
    )

    try:
        chunk = backend.infer(obs)
    finally:
        backend.close()

    served = np.asarray(chunk.actions, dtype=np.float64)
    demo_raw = frame_entry.get("action_chunk")
    demo = np.asarray(demo_raw, dtype=np.float64) if demo_raw is not None else None
    log.info(
        "infer done rtt=%.3fs served chunk=%s demo action_chunk=%s",
        backend.last_rtt if hasattr(backend, "last_rtt") else float("nan"),
        served.shape,
        demo.shape if demo is not None else None,
    )

    out_path = Path(obs_dir) / f"served_chunk_f{frame}.npy"
    np.save(out_path, served.astype(np.float32))
    log.info("saved served chunk to %s", out_path)

    def _fmt_row(label: str, row: np.ndarray) -> str:
        right_arm = ", ".join(f"{v:7.4f}" for v in row[s27a15.A_RIGHT_ARM])
        grip = float(row[s27a15.A_RIGHT_GRIP])
        return f"  {label:<16} right_arm=[{right_arm}] gripper={grip:7.4f}"

    log.info("server chunk (right arm + gripper):")
    for r in (0, 10, 20):
        log.info(_fmt_row(f"served row {r}", served[r]))

    if demo is None:
        # e.g. a --capture-live observation, which has no ground-truth
        # action: print the served chunk and stop there, no diff/verdict.
        log.info(
            "manifest frame %d has no action_chunk — skipping diff/verdict",
            frame,
        )
        return 1

    log.info("demo action_chunk (right arm + gripper):")
    for r in (0, 10, 20):
        log.info(_fmt_row(f"demo row {r}", demo[r]))

    per_row_max, verdict = chunk_diff_verdict(served, demo)

    log.info("per-row max|diff| over 15 dims:")
    for r, value in enumerate(per_row_max):
        log.info("  row %2d: %.4f", r, value)

    row0_diff = np.abs(served[0] - demo[0])
    joints = ", ".join(f"j{i + 1}={d:.4f}" for i, d in enumerate(row0_diff[s27a15.A_RIGHT_ARM]))
    log.info("per-joint |diff| at row 0 (right arm): %s", joints)

    log.info("VERDICT: %s", verdict)
    return 1


def _run_mixed(
    backend,
    images_dir: Path,
    images_source: str,
    state_dir: Path,
    state_source: str,
    frame: int,
    task: str,
) -> int:
    """`--images-from`/`--state-from`: build ONE Obs with images from
    `images_dir` and state (arms + gripper + wrenches) from `state_dir` —
    either may be a recorded corpus dir or a `--capture-live` capture, both
    `*_obs.json`-shaped — send it once, and diff the served chunk against
    the STATE source's own action_chunk when it has one (U-35 isolation:
    is a rollout mismatch coming from the live images, or the live
    wrenches?). See camelo/policy/dummy_obs.py:build_obs_from_mixed_frames."""
    from camelo.policy.adapters import s27a15
    from camelo.policy.dummy_obs import (
        build_obs_from_mixed_frames,
        chunk_diff_verdict,
        load_recorded_frame,
    )

    _images_manifest, images_entry = load_recorded_frame(images_dir, frame)
    _state_manifest, state_entry = load_recorded_frame(state_dir, frame)
    obs = build_obs_from_mixed_frames(images_dir, images_entry, state_entry)

    log.info("connecting to %s", backend.url if hasattr(backend, "url") else "<backend>")
    backend.reset(task)
    log.info(
        "reset ok, layout=s27a15 task=%r; sending MIXED obs frame=%d — "
        "images from %s (%s), state from %s (%s)",
        task,
        frame,
        images_dir,
        images_source,
        state_dir,
        state_source,
    )
    for camera, image in obs.images.items():
        log.info("  image[%-11s] shape=%s dtype=%s", camera, image.shape, image.dtype)
    state27 = np.asarray(state_entry["observation.state"], dtype=np.float64)
    log.info(
        "  observation.state (27-dim, from %s): %s",
        state_dir,
        np.array2string(state27, precision=4, suppress_small=True, max_line_width=200),
    )

    try:
        chunk = backend.infer(obs)
    finally:
        backend.close()

    served = np.asarray(chunk.actions, dtype=np.float64)
    log.info(
        "infer done rtt=%.3fs served chunk=%s",
        backend.last_rtt if hasattr(backend, "last_rtt") else float("nan"),
        served.shape,
    )

    def _fmt_row(label: str, row: np.ndarray) -> str:
        right_arm = ", ".join(f"{v:7.4f}" for v in row[s27a15.A_RIGHT_ARM])
        grip = float(row[s27a15.A_RIGHT_GRIP])
        return f"  {label:<16} right_arm=[{right_arm}] gripper={grip:7.4f}"

    log.info("server chunk (right arm + gripper):")
    for r in (0, 10, 20):
        log.info(_fmt_row(f"served row {r}", served[r]))

    demo_raw = state_entry.get("action_chunk")
    if demo_raw is None:
        log.info(
            "state source %s frame %d has no action_chunk — skipping diff/verdict",
            state_dir,
            frame,
        )
        return 1

    demo = np.asarray(demo_raw, dtype=np.float64)
    log.info("demo action_chunk (from state source, right arm + gripper):")
    for r in (0, 10, 20):
        log.info(_fmt_row(f"demo row {r}", demo[r]))

    per_row_max, verdict = chunk_diff_verdict(served, demo)
    log.info("per-row max|diff| over 15 dims:")
    for r, value in enumerate(per_row_max):
        log.info("  row %2d: %.4f", r, value)

    log.info("VERDICT: %s", verdict)
    return 1


def _run_capture_live(node, outdir: Path, obs_dir: Path | None, task: str) -> int:
    """`--capture-live`: ONE live rig observation (all 3 cameras + joint
    states + wrenches), captured via the exact ObsCollector/wait_for_obs
    path `scripts/run_policy.py --world real` uses
    (camelo.ros.obs_collector.ObsCollector, camelo.runner.episode_runner
    .wait_for_obs) — never re-implemented here. Saved into `outdir` in the
    same schema `load_recorded_frame`/`build_obs_from_recorded_frame` read
    (camelo/policy/dummy_obs.py:save_live_frame), with NO action_chunk
    (there is no ground truth for a live pose).

    Never touches the policy server: `task` is only used for the log line,
    no `backend.reset()`/`infer()` happens. If `obs_dir` is given, diffs
    the live state and images against that corpus's OWN frame 0
    (docs/realdata/16d_T6_LEARNINGS.md, U-35 isolation)."""
    from camelo import contracts as C
    from camelo.policy.adapters import s27a15
    from camelo.policy.dummy_obs import (
        format_state27_diff_table,
        load_image_rgb,
        load_recorded_frame,
        mean_abs_pixel_diff,
        save_live_frame,
    )
    from camelo.ros.obs_collector import ObsCollector
    from camelo.runner.episode_runner import wait_for_obs

    topics = C.topics_for("real")
    collector = ObsCollector(node, camera_keys=list(s27a15.CAMERA_KEYS), topics=topics)
    try:
        log.info(
            "capturing ONE live observation (task=%r): waiting up to 10s for all "
            "3 cameras + joint states + wrenches to arrive...",
            task,
        )
        obs = wait_for_obs(collector, timeout_s=10.0)
    finally:
        collector.close()

    state27 = s27a15.RigObservation.from_mapping(obs.rig).pack()
    manifest_path = save_live_frame(outdir, obs, state27, frame=0)
    log.info("saved live observation to %s", manifest_path)
    for camera, image in obs.images.items():
        log.info("  image[%-11s] shape=%s dtype=%s", camera, image.shape, image.dtype)
    log.info(
        "  observation.state (27-dim, LIVE): %s",
        np.array2string(state27, precision=4, suppress_small=True, max_line_width=200),
    )

    if obs_dir is None:
        return 1

    _manifest, recorded_entry = load_recorded_frame(obs_dir, 0)
    recorded_state27 = np.asarray(recorded_entry["observation.state"], dtype=np.float64)
    log.info(
        "live vs recorded (%s frame 0) state:\n%s",
        obs_dir,
        format_state27_diff_table(state27, recorded_state27),
    )
    for camera, info in recorded_entry["images"].items():
        if camera not in obs.images:
            log.warning("  pixel diff[%-11s] missing from live capture, skipping", camera)
            continue
        recorded_image = load_image_rgb(Path(obs_dir) / info["file"])
        live_image = obs.images[camera]
        try:
            diff = mean_abs_pixel_diff(live_image, recorded_image)
        except RuntimeError as exc:
            log.error("  pixel diff[%-11s] %s", camera, exc)
            continue
        log.info(
            "  pixel diff[%-11s] live_shape=%s recorded_shape=%s mean_abs_diff=%.3f",
            camera,
            live_image.shape,
            recorded_image.shape,
            diff,
        )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="ws://127.0.0.1:8765", help="policy server URL")
    parser.add_argument(
        "--layout",
        choices=["canonical", "s27a15"],
        default="s27a15",
        help="s27a15 (default) = the Munich real-robot checkpoints (27 state / 15 "
        "action), named rig groups + real camera resolutions; canonical = sim-trained "
        "(model16/canonical) checkpoints, 37-dim state, sim camera resolutions. Must "
        "match how serve_policy.py's adapter was started "
        "(--action-layout/--state-layout)",
    )
    parser.add_argument(
        "--task", default=None, help="language instruction (default: the layout's frozen caption)"
    )
    parser.add_argument("--rate", type=float, default=1.0, help="dummy observations per second")
    parser.add_argument(
        "--seconds", type=float, default=30.0, help="run length; <= 0 runs until Ctrl-C"
    )
    parser.add_argument("--node-name", default="camelo_dummy_client")
    parser.add_argument(
        "--obs-dir",
        type=Path,
        default=None,
        help="directory with a recorded corpus observation (e.g. "
        "outputs/rig/t5/ep163_frames): send ONE recorded frame's real "
        "images/joints/wrenches through the real client path instead of the "
        "fixed dummy observation, and diff the returned chunk against the "
        "demo's own action_chunk (U-35 probe, "
        "docs/realdata/16d_T6_LEARNINGS.md). Only defined for --layout "
        "s27a15. Ignores --rate/--seconds and sends a single request.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="frame index to send from --obs-dir/--images-from/--state-from "
        "(default 0; e.g. 0, 100, 200)",
    )
    parser.add_argument(
        "--images-from",
        type=Path,
        default=None,
        help="directory with a *_obs.json manifest (a recorded corpus dir OR a "
        "--capture-live output dir — both share the schema): take the IMAGES "
        "from here instead of --obs-dir, for isolating whether a mismatch "
        "comes from images or state (U-35). The other half (state) still "
        "comes from --state-from if given, else --obs-dir; at least one of "
        "the two must resolve. Only defined for --layout s27a15.",
    )
    parser.add_argument(
        "--state-from",
        type=Path,
        default=None,
        help="directory with a *_obs.json manifest: take the STATE (arms + "
        "gripper + wrenches) from here instead of --obs-dir. The other half "
        "(images) still comes from --images-from if given, else --obs-dir. "
        "Only defined for --layout s27a15.",
    )
    parser.add_argument(
        "--capture-live",
        type=Path,
        default=None,
        metavar="OUTDIR",
        help="capture ONE live rig observation (all 3 cameras + joint states + "
        "wrenches, via the same ObsCollector/wait_for_obs path "
        "run_policy.py --world real uses) and save it into OUTDIR in the "
        "recorded-frame schema — no policy-server contact. Requires ROS "
        "(incompatible with --no-ros) and --layout s27a15. If --obs-dir is "
        "also given, diffs the live state/images against that corpus's "
        "frame 0. Ignores --rate/--seconds/--server.",
    )
    # The same two levers `run_policy.py --backend remote` runs with, so the
    # probe can send EXACTLY what a rollout sends: a resized-wire rollout
    # whose row 0 is only checked against the demo at full resolution would
    # be checking the wrong pipeline.
    add_wire_image_args(parser)
    parser.add_argument(
        "--no-ros",
        action="store_true",
        help="skip rclpy entirely (no ros_session, no ROS node) — for proving the "
        "wire protocol from a machine with no ROS 2 install, e.g. a dev laptop. "
        "The real deployment on the robot must NOT use this flag.",
    )
    args = parser.parse_args()
    if args.obs_dir is not None and args.layout != "s27a15":
        parser.error("--obs-dir is only defined for --layout s27a15 (the Munich rig layout)")
    if (args.images_from is not None or args.state_from is not None) and args.layout != "s27a15":
        parser.error("--images-from/--state-from are only defined for --layout s27a15")
    if args.capture_live is not None and args.layout != "s27a15":
        parser.error("--capture-live is only defined for --layout s27a15")
    if args.capture_live is not None and args.no_ros:
        parser.error(
            "--capture-live needs ROS (it captures a live observation via "
            "ObsCollector) — drop --no-ros"
        )
    images_dir = args.images_from if args.images_from is not None else args.obs_dir
    state_dir = args.state_from if args.state_from is not None else args.obs_dir
    mixed = args.images_from is not None or args.state_from is not None
    if mixed and (images_dir is None or state_dir is None):
        parser.error(
            "--images-from/--state-from needs the OTHER half from itself or "
            "--obs-dir — pass both, or pass --obs-dir for whichever one is missing"
        )
    task = args.task if args.task is not None else DEFAULT_TASK[args.layout]
    make_obs = OBS_BUILDERS[args.layout]
    format_action = ACTION_FORMATTERS[args.layout]
    setup_logging()

    from camelo.policy.adapters import s27a15
    from camelo.policy.backend import RemoteBackend

    wire_images = wire_images_from_args(args, cameras=s27a15.CAMERA_KEYS)

    def _run(backend) -> int:
        if mixed:
            return _run_mixed(
                backend,
                images_dir,
                "--images-from" if args.images_from is not None else "--obs-dir",
                state_dir,
                "--state-from" if args.state_from is not None else "--obs-dir",
                args.frame,
                task,
            )
        if args.obs_dir is not None:
            return _run_obs_dir(backend, args.obs_dir, args.frame, task)
        log.info("connecting to %s", args.server)
        backend.reset(task)
        log.info(
            "reset ok, layout=%s task=%r; sending dummy observations at %.2f Hz",
            args.layout,
            task,
            args.rate,
        )

        period = 1.0 / args.rate
        t_sim = 0.0
        deadline = None if args.seconds <= 0 else time.monotonic() + args.seconds
        n = 0
        try:
            while deadline is None or time.monotonic() < deadline:
                tick = time.monotonic()
                obs = make_obs(t_sim)
                chunk = backend.infer(obs)
                n += 1
                horizon = chunk.actions.shape[0]
                action_lines = format_action(np.asarray(chunk.actions[0]))
                log.info(
                    "infer #%d t_sim=%.2f rtt=%.3fs chunk=%s next_action (step 0/%d):\n%s",
                    n,
                    t_sim,
                    backend.last_rtt,
                    tuple(chunk.actions.shape),
                    horizon,
                    action_lines,
                )
                t_sim += period
                sleep_left = period - (time.monotonic() - tick)
                if sleep_left > 0:
                    time.sleep(sleep_left)
        except KeyboardInterrupt:
            log.info("interrupted")
        finally:
            backend.close()
        log.info("sent %d dummy observations", n)
        return n

    if args.capture_live is not None:
        from camelo.ros.session import ros_session

        # verify_contract=False: this node touches no topics of its own (the
        # ObsCollector it creates subscribes real-rig topics directly), so it
        # has no business requiring the benchmark's topics.yaml.
        with ros_session(args.node_name, verify_contract=False) as node:
            n = _run_capture_live(node, args.capture_live, args.obs_dir, task)
    elif args.no_ros:
        n = _run(RemoteBackend(args.server, wire_images=wire_images))
    else:
        from camelo.ros.session import ros_session

        # verify_contract=False: this node touches no topics, so it has no
        # business requiring the benchmark's topics.yaml to exist or match.
        with ros_session(args.node_name, verify_contract=False):
            n = _run(RemoteBackend(args.server, wire_images=wire_images))
    return 0 if n else 1


if __name__ == "__main__":
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
