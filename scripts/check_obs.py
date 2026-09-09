#!/usr/bin/env python3
"""M1.1 — prove observation reception: per-topic rates + a finite 37-dim state.

    python scripts/check_obs.py --seconds 10
    python scripts/check_obs.py --world real --seconds 10

Requires the task2 sim up with --record (cameras + recording topics) and a
sourced ROS 2 environment — or --world real against the TMR station topics
in record_bag.bash.

Exit codes:
    0 - OK
    1 - a stream is silent, or the state has fewer than the expected finite dims
    3 - real world only: a camera's measured frame shape disagrees with the
        contract's declared shape (docs/realdata/16 R-42 — e.g. the station's
        ZED at VGA against a HD720 contract/corpus). Takes priority over the
        exit-1 checks: it means whatever DID arrive is off-contract, which is
        worse than a slow or incomplete stream.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from camelo import contracts as C
from camelo.cli import setup_logging
from camelo.contracts import default_world, topics_for
from camelo.control.image_age import DEFAULT_MAX_IMAGE_AGE_S, stale_image_report, summarize
from camelo.control.state_age import DEFAULT_MAX_STATE_AGE_S, stale_state_report


def _cell(value: float | None, width: int) -> str:
    return f"{value:>{width}.3f}" if value is not None else f"{'-':>{width}}"


def _cell_int(value: int | None, width: int) -> str:
    return f"{value:>{width}d}" if value is not None else f"{'-':>{width}}"


def _wire_hz(wire_0, wire_1, key: str, span_s: float) -> float | None:
    """Received-frames-per-second for one camera over the bracketed window.

    None (printed as '-') when the counters never reached the parent —
    either the camera shipped nothing at all, or it shipped only before the
    window opened. A rate we could not measure must not read 0.0.
    """
    end = wire_1.get(key)
    if end is None:
        return None
    start = wire_0.get(key)
    recv_0 = start.recv if start is not None else 0
    if end.recv <= recv_0:
        return None
    return (end.recv - recv_0) / span_s


def _dropped_full_delta(wire_0, wire_1, key: str) -> int | None:
    """New ``dropped_full`` frames for one camera over the bracketed window.

    Not a rate: a raw count, so a run of drops from a single parent-drain
    stall (camera_workers._put_or_drop — while the queue stays full, EVERY
    new frame is dropped until the parent drains again) shows up as one
    visibly large number in the table instead of averaging away. None
    (printed as '-') when the counters never reached the parent, same
    convention as ``_wire_hz``.
    """
    end = wire_1.get(key)
    if end is None:
        return None
    start = wire_0.get(key)
    dropped_0 = start.dropped_full if start is not None else 0
    return end.dropped_full - dropped_0


def _shape_str(shape) -> str:
    return "x".join(str(int(v)) for v in shape)


def decide_exit_code(
    *,
    silent: list,
    obs,
    min_finite: int,
    camera_shape_mismatch: dict,
) -> int:
    """Pure exit-code decision (see the module docstring for what each code
    means) — split out from ``main`` so it is testable without a ROS
    session."""
    if camera_shape_mismatch:
        return 3
    if silent:
        return 1
    if obs is None or int(np.isfinite(obs.state).sum()) < min_finite:
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument(
        "--cameras",
        default="head,wrist_left,wrist_right",
        help="comma list (default: all three; each camera has its own "
        "subscriber subprocess, so all should reach the ~2 Hz wire rate)",
    )
    parser.add_argument(
        "--world",
        choices=("sim", "real"),
        default=default_world(),
        help="topic contract (sim = /isaac/*; real = record_bag.bash). "
        "Also CAMELO_WORLD",
    )
    args = parser.parse_args()
    setup_logging()

    from camelo.ros.obs_collector import ObsCollector
    from camelo.ros.session import ros_session

    camera_keys = [key.strip() for key in args.cameras.split(",") if key.strip()]
    topics = topics_for(args.world)
    with ros_session("camelo_check_obs", world=args.world) as node:
        collector = ObsCollector(node, camera_keys=camera_keys, topics=topics)
        try:
            print(f"listening for {args.seconds:.0f}s ...")
            frame_marker = collector.frame_marker()
            # Wire counters ride in on the frames themselves (camera_workers
            # .WireCounts). They are cumulative from worker start, so the
            # window has to be bracketed and timed here — `frames` below
            # counts parent DRAIN ticks, `wire` counts what DDS delivered to
            # the worker, and the two differ by exactly the rate limit plus
            # whatever the drain missed (docs/realdata/16 R-45).
            workers = getattr(collector, "_workers", None)
            wire_0 = workers.wire_counts() if hasattr(workers, "wire_counts") else {}
            t_wire_0 = time.monotonic()
            deadline = time.monotonic() + 5.0  # first clock tick brackets sim time
            while collector.snapshot()["sim_time"] is None and time.monotonic() < deadline:
                time.sleep(0.1)
            t_sim_0 = collector.snapshot()["sim_time"]
            time.sleep(args.seconds)
            t_sim_1 = collector.snapshot()["sim_time"]
            wire_1 = workers.wire_counts() if hasattr(workers, "wire_counts") else {}
            wire_span = max(time.monotonic() - t_wire_0, 1e-6)

            rates = collector.rates()
            print(f"\n{'stream':<24}{'msgs/s':>8}")
            for key, rate in rates.items():
                print(f"{key:<24}{rate:>8.1f}")

            # N2b step 1 (GRASP_EXPERIMENT_PLAN.md): the same cameras in SIM
            # units. msgs/s above is WALL Hz; the offline lag grid (P3) is
            # in 1/30 sim-s, so only these columns compare with it. Idle
            # here; the number that counts is measured under eval load by
            # run_rollout (image_* keys in summary.json).
            sim_span = (
                (t_sim_1 - t_sim_0)
                if t_sim_0 is not None and t_sim_1 is not None and t_sim_1 >= t_sim_0
                else None
            )
            image_stats = summarize(
                {}, collector.frames_since(frame_marker), sim_span, args.seconds, 0
            )
            if sim_span is not None:
                print(f"\nsim advanced {sim_span:.2f} sim-s in {args.seconds:.0f} wall-s "
                      f"({sim_span / args.seconds:.3f}x)")
            print(f"{'camera':<14}{'frames':>7}{'wire Hz':>9}{'dropped':>9}"
                  f"{'sim Hz':>8}{'gap(sim-s)':>12}{'gapmin':>9}{'gapp90':>9}"
                  f"{'lag30':>7}{'clock-stamp':>13}")
            for key in collector.camera_keys:
                sim_hz = image_stats.get(f"image_rate_{key}_sim_hz")
                gap = image_stats.get(f"image_stamp_gap_{key}_median_s")
                lag = image_stats.get(f"image_period_{key}_frames30")
                drain = image_stats.get(f"image_drain_age_{key}_median_s")
                print(f"{key:<14}{image_stats.get(f'image_frames_{key}', 0):>7}"
                      f"{_cell(_wire_hz(wire_0, wire_1, key, wire_span), 9)}"
                      f"{_cell_int(_dropped_full_delta(wire_0, wire_1, key), 9)}"
                      f"{_cell(sim_hz, 8)}{_cell(gap, 12)}"
                      f"{_cell(image_stats.get(f'image_stamp_gap_{key}_min_s'), 9)}"
                      f"{_cell(image_stats.get(f'image_stamp_gap_{key}_p90_s'), 9)}"
                      f"{_cell(lag, 7)}{_cell(drain, 13)}")
            print("wire Hz = frames DDS delivered to the worker process "
                  "(recv); frames = parent drain ticks; dropped = new "
                  "dropped_full frames this window (camera_workers._put_or_"
                  "drop) — a run of these means a parent-drain stall, and "
                  "the delivered frame can be as old as the whole stall")
            for key, measured in sorted(collector.camera_shape_mismatch.items()):
                expected = topics.cameras[key]["shape"]
                print(f"SHAPE MISMATCH {key}: measured {_shape_str(measured)} "
                      f"vs contract {_shape_str(expected)}")
            if topics.world == C.WORLD_REAL:
                # The same U-23 liveness question run_policy gates on, asked
                # once here: a camera that died during the window still shows
                # frames/Hz for the part of it that it was alive.
                fault = stale_image_report(
                    collector.last_image_wall(), time.monotonic(), DEFAULT_MAX_IMAGE_AGE_S
                )
                if fault is not None:
                    print(fault)
                # ... and the U-27 twin for the arms. The msgs/s table above
                # averages over the whole window, so a stream that died
                # halfway through still shows a healthy rate; this reads the
                # newest receipt time instead.
                now_wall = time.monotonic()
                last_state_wall = collector.last_state_wall()
                print("joint-state age (s, wall): " + (
                    "  ".join(
                        f"{k}=" + ("never" if v is None else f"{now_wall - v:.3f}")
                        for k, v in sorted(last_state_wall.items())
                    ) or "no sided arm streams in this topic map"
                ))
                fault = stale_state_report(
                    last_state_wall, now_wall, DEFAULT_MAX_STATE_AGE_S
                )
                if fault is not None:
                    print(fault)
            if any(image_stats.get(f"image_stamped_{k}") is False for k in collector.camera_keys):
                print("WARNING: unstamped camera frames — header.stamp is 0; "
                      "sim-time age cannot be measured")
            lag = image_stats.get("image_lag_frames30")
            if lag is not None:
                lag_mean = image_stats.get("image_lag_frames30_mean")
                mean_str = f", mean-gap {lag_mean:.1f}" if lag_mean is not None else ""
                print(f"worst camera: lag {lag:.1f} frames on the P3 grid{mean_str} "
                      f"(P3's ~80 % retention line is lag <= 2.7)")

            expected = set(topics.expected_obs_streams) | {
                f"image_{k}" for k in collector.camera_keys
            }
            silent = sorted(expected - set(rates))

            # Report the state summary even when a stream is silent — it is the
            # actual point of M1.1 (DGX_FINDINGS.md F-17).
            obs = collector.get_obs(require_images=False)
            # Real robot has no EE-pose topics (14 of 37 dims stay NaN).
            min_finite = C.STATE_DIM if topics.world == "sim" else C.STATE_DIM - 14
            if obs is not None:
                finite = int(np.isfinite(obs.state).sum())
                print(f"\nt_sim={obs.t_sim:.2f}s  state: {finite}/{C.STATE_DIM} finite")
                for key, image in obs.images.items():
                    print(f"image {key}: {image.shape} {image.dtype}")
                if finite < min_finite:
                    nan_idx = np.where(~np.isfinite(obs.state))[0].tolist()
                    print(f"non-finite state indices: {nan_idx}")
            else:
                print("\nno state yet (clock or joint_states_full missing)")

            if silent:
                print(f"\nSILENT streams: {silent}")
                hint = (
                    "Is the sim running with --record? Are you on the right ROS_DOMAIN_ID?"
                    if topics.world == "sim"
                    else "Are the station nodes up on this ROS_DOMAIN_ID? "
                    "Topics must match record_bag.bash."
                )
                print(hint)

            exit_code = decide_exit_code(
                silent=silent,
                obs=obs,
                min_finite=min_finite,
                camera_shape_mismatch=collector.camera_shape_mismatch,
            )
            if exit_code == 0:
                print("\nOK")
            return exit_code
        finally:
            collector.close()  # reap camera workers or the process never exits (F-32b)


if __name__ == "__main__":
    # os._exit, not sys.exit — the same pattern as run_policy.py/eval_batch.py,
    # and here for a MEASURED reason (rig 2026-09-02, t3_checkobs_140915): this
    # script printed its complete report, returned 0, and then sat 6+ minutes
    # in INTERPRETER EXIT. rclpy's MultiThreadedExecutor runs callbacks on a
    # concurrent.futures ThreadPoolExecutor whose threads are NON-daemon, and
    # concurrent.futures registers an atexit hook that joins every one of them
    # with no timeout, so one wedged callback (there: a camera drain blocked in
    # multiprocessing.Queue.get_nowait — fixed in CameraWorkers.close) hangs
    # the process after the work is done. The wedge is fixed; this makes the
    # exit itself deterministic, and keeps the exit code M1.1 is read by.
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
        traceback.print_exc()  # os._exit skips the default traceback print
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
