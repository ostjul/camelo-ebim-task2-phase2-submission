#!/usr/bin/env python3
"""M1.5 — replay a recorded episode's actions through the chunk executor.

    python scripts/replay_episode.py --dataset <path>/task2_thermalpad_v1 --episode 0
    python scripts/replay_episode.py --dataset <path>/ext_hermanprawiro_task2_fixpos_v1 \
        --episode 0 --goto-start --video outputs/videos/replay_ep0.mp4

Closes the whole command path without any policy: a teleoped episode that
replays to a visually similar end state proves topics, units, timing, AND
gripper polarity. It is also the control for any policy failure — if the
dataset's OWN actions do not reproduce the demo, no checkpoint could.

Requires lerobot (dataset loading), the task2 sim with --record, and the
helper stack. Reset the scene first (recorder key 5 or --reset here).

Start pose. --goto-start walks the arms to the episode's first recorded
joint state under the same 0.05 rad/tick clamp before replay begins, so
the trajectory starts where it was recorded instead of wherever the scene
happens to sit. The SPINE cannot be driven this way — it has no ROS
command path and is pinned at launch (SPINE=0.50) — so --start-frame auto
instead seeks the first frame whose RECORDED spine matches the live one.
That matters: these corpora start at spine 0.0 and ramp to ~0.486 over the
first seconds, and replaying a pre-ramp frame at a settled spine would run
the whole arm trajectory ~0.49 m too high.
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from camelo import contracts as C
from camelo.cli import setup_logging

RECORDER = Path(__file__).resolve().parents[1] / "tools" / "dgx_probes" / "record_camera.py"


def load_episode(dataset_root: Path, episode: int) -> tuple[np.ndarray, np.ndarray, float]:
    """(actions [T, 20], states [T, 37], fps) for ONE episode.

    Loads only the two columns it needs — a full-dataset scan decodes every
    video frame (F-22).
    """
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id=dataset_root.name, root=dataset_root, episodes=[episode]
    )
    if dataset.num_frames == 0:  # out-of-range episode only warns upstream
        raise ValueError(f"episode {episode} not found in {dataset_root}")

    def column(name: str) -> np.ndarray:
        rows = dataset.select_columns(name)
        return np.stack(
            [np.asarray(rows[i][name], dtype=np.float32) for i in range(dataset.num_frames)]
        )

    return column("action"), column("observation.state"), float(dataset.fps)


def pick_start_frame(states: np.ndarray, measured_spine: float, tol: float) -> int:
    """First frame whose recorded spine matches the live (pinned) spine.

    The spine is part of the arms' kinematic chain, so replaying at the
    wrong height offsets every reach vertically.
    """
    spine = states[:, C.S_SPINE]
    matches = np.flatnonzero(np.abs(spine - measured_spine) <= tol)
    if matches.size == 0:
        raise SystemExit(
            f"no frame has spine within {tol} of the live {measured_spine:.4f} m "
            f"(episode spans {np.nanmin(spine):.4f}..{np.nanmax(spine):.4f}) — "
            "launch the scene with SPINE pinned to a height this episode reaches, "
            "or pass --start-frame explicitly"
        )
    return int(matches[0])


def arms_of(state: np.ndarray) -> np.ndarray:
    return np.concatenate([state[C.S_LEFT_ARM], state[C.S_RIGHT_ARM]])


def hold_chunk(target_state: np.ndarray) -> np.ndarray:
    """A 2-step canonical chunk that parks the arms/grippers at a recorded state."""
    action = np.zeros(C.ACTION_DIM, dtype=np.float32)
    action[C.A_LEFT_ARM] = target_state[C.S_LEFT_ARM]
    action[C.A_RIGHT_ARM] = target_state[C.S_RIGHT_ARM]
    action[C.A_LEFT_GRIP] = target_state[C.S_LEFT_GRIP]
    action[C.A_RIGHT_GRIP] = target_state[C.S_RIGHT_GRIP]
    action[C.A_SPINE] = np.nan  # no ROS command path; the executor holds it
    return np.stack([action, action])


def goto_state(collector, publisher, target_state, *, max_delta, rate, timeout_s, tol):
    """Walk the arms to a recorded joint pose under the executor's clamp.

    Returns the final max per-joint error [rad]. Uses its own executor so the
    replay's stats stay clean.
    """
    from camelo.control.chunk_executor import ChunkExecutor
    from camelo.runner.episode_runner import wait_for_obs

    executor = ChunkExecutor(max_delta_per_tick=max_delta, replan_after_steps=10**9)
    obs = wait_for_obs(collector)
    executor.set_chunk(obs.t_sim, hold_chunk(target_state), dt=1.0)
    target_arms = arms_of(target_state)

    deadline = time.monotonic() + timeout_s
    error = float("inf")
    while time.monotonic() < deadline:
        obs = collector.get_obs()
        if obs is None:
            time.sleep(1.0 / rate)
            continue
        error = float(np.nanmax(np.abs(arms_of(obs.state) - target_arms)))
        if error <= tol:
            return error
        command = executor.step(obs.t_sim, obs.state)
        if command is not None:
            publisher.publish(command)
        time.sleep(1.0 / rate)
    return error


def wait_for_spine(collector, *, rate: float, timeout_s: float, tol: float = 1e-3) -> float:
    """Block until the spine stops moving; returns the settled height.

    A scene reset drops the spine and the launcher re-drives it to the pinned
    SOP value, a ramp of ~24 wall s (F-72) — far longer than the recorder's
    own settle_s. Replaying during the ramp moves the arms while the ground
    under them is still rising.
    """
    from camelo.runner.episode_runner import wait_for_obs

    previous = float(wait_for_obs(collector).state[C.S_SPINE])
    stable_since = time.monotonic()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        time.sleep(1.0 / rate)
        obs = collector.get_obs()
        if obs is None:
            continue
        current = float(obs.state[C.S_SPINE])
        if abs(current - previous) > tol:
            stable_since = time.monotonic()
        previous = current
        if time.monotonic() - stable_since >= 2.0:
            return current
    return previous


def start_recorder(out: Path, seconds: float, fps: float) -> subprocess.Popen | None:
    """Spawn the streaming eval-camera recorder alongside the replay.

    Separate process on purpose: rclpy image ingest is per-process GIL-bound
    (F-30), and record_camera.py streams to ffmpeg so memory stays flat.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u", str(RECORDER),
        "--out", str(out), "--seconds", str(seconds), "--fps", str(fps),
    ]
    print(f"recording eval camera -> {out}")
    return subprocess.Popen(cmd)


def stop_recorder(proc: subprocess.Popen | None) -> None:
    """SIGINT, then wait: the recorder closes ffmpeg's stdin on the way out,
    and killing it harder strands the file without a moov atom."""
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=90)
    except subprocess.TimeoutExpired:
        proc.terminate()
        proc.wait(timeout=30)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--max-delta", type=float, default=0.05)
    parser.add_argument("--reset", action="store_true", help="request a scene reset first")
    parser.add_argument("--skip-polarity-check", action="store_true")
    parser.add_argument(
        "--start-frame",
        default="0",
        help="first frame to replay: an index, or 'auto' = first frame whose "
        "recorded spine matches the live pinned spine (default: 0)",
    )
    parser.add_argument("--spine-tol", type=float, default=0.01, help="metres, for 'auto'")
    parser.add_argument(
        "--goto-start",
        action="store_true",
        help="walk the arms to the start frame's recorded pose before replaying",
    )
    parser.add_argument("--goto-tol", type=float, default=0.02, help="rad, per joint")
    parser.add_argument("--goto-timeout-s", type=float, default=120.0, help="wall seconds")
    parser.add_argument("--video", type=Path, default=None, help="eval-camera mp4 to write")
    parser.add_argument("--video-fps", type=float, default=10.0, help="playback fps")
    parser.add_argument(
        "--world",
        choices=("sim", "real"),
        default=C.default_world(),
        help="topic contract (sim = /bridge/*; real = record_bag.bash)",
    )
    args = parser.parse_args()
    setup_logging()

    actions, states, fps = load_episode(args.dataset, args.episode)
    print(
        f"episode {args.episode}: {len(actions)} frames @ {fps:g} fps "
        f"({len(actions) / fps:.1f} sim-s); recorded spine "
        f"{states[0, C.S_SPINE]:.4f} -> {states[-1, C.S_SPINE]:.4f} m"
    )

    from camelo.control.chunk_executor import DATASET_FPS, ChunkExecutor
    from camelo.ros.command_publisher import CommandPublisher, verify_gripper_polarity
    from camelo.ros.obs_collector import ObsCollector
    from camelo.ros.session import ros_session
    from camelo.runner.episode_runner import wait_for_obs

    if fps != DATASET_FPS:
        print(f"note: dataset fps {fps:g} != executor default {DATASET_FPS:g}; using {fps:g}")

    recorder = None
    with ros_session("camelo_replay", world=args.world) as node:
        # Replay needs no images (actions + joint state only); skipping the
        # image subscriptions sidesteps the per-process ingest limit (F-30).
        # The eval-camera video is captured by a separate process.
        topics = C.topics_for(args.world)
        collector = ObsCollector(node, camera_keys=[], topics=topics)
        publisher = CommandPublisher(node, topics=topics)
        executor = ChunkExecutor(max_delta_per_tick=args.max_delta, replan_after_steps=10**9)

        wait_for_obs(collector)
        if not args.skip_polarity_check:
            verify_gripper_polarity(collector, publisher)
        if args.reset:
            collector.request_scene_reset()
            time.sleep(5.0)
            settled = wait_for_spine(collector, rate=args.rate, timeout_s=90.0)
            print(f"post-reset spine settled at {settled:.4f} m")

        obs = wait_for_obs(collector)
        live_spine = float(obs.state[C.S_SPINE])
        if args.start_frame == "auto":
            start = pick_start_frame(states, live_spine, args.spine_tol)
            print(
                f"start frame {start} (t={start / fps:.2f}s): recorded spine "
                f"{states[start, C.S_SPINE]:.4f} matches live {live_spine:.4f} m"
            )
        else:
            start = int(args.start_frame)
            if not 0 <= start < len(actions):
                raise SystemExit(f"--start-frame {start} outside 0..{len(actions) - 1}")
            print(f"start frame {start} (t={start / fps:.2f}s)")
            drift = abs(states[start, C.S_SPINE] - live_spine)
            if drift > args.spine_tol:
                print(
                    f"  WARNING: recorded spine {states[start, C.S_SPINE]:.4f} vs live "
                    f"{live_spine:.4f} m ({drift:.4f} m off) — the arm trajectory will "
                    "run at the wrong height; try --start-frame auto"
                )

        target = states[start]
        print(
            f"recorded start base odom {np.round(target[C.S_BASE_ODOM], 4)}, "
            f"live {np.round(obs.state[C.S_BASE_ODOM], 4)}"
        )
        if args.goto_start:
            error = goto_state(
                collector, publisher, target,
                max_delta=args.max_delta, rate=args.rate,
                timeout_s=args.goto_timeout_s, tol=args.goto_tol,
            )
            verdict = "converged" if error <= args.goto_tol else "TIMED OUT"
            print(f"goto-start {verdict}: max joint error {error:.4f} rad")
        else:
            error = float(np.nanmax(np.abs(arms_of(wait_for_obs(collector).state)
                                           - arms_of(target))))
            print(f"start pose offset (not corrected): {error:.4f} rad — pass --goto-start")

        replay = actions[start:]
        if args.video is not None:
            # Sim runs ~0.18x real time; give the recorder generous headroom
            # and stop it explicitly when the replay ends.
            recorder = start_recorder(
                args.video, seconds=len(replay) / fps / 0.18 + 120.0, fps=args.video_fps
            )

        obs = wait_for_obs(collector)
        t0 = obs.t_sim
        executor.set_chunk(t0, replay, dt=1.0 / fps)
        end_t = t0 + len(replay) / fps
        print(f"replaying frames {start}..{len(actions) - 1} "
              f"from t_sim={t0:.2f} to {end_t:.2f}")

        tracking: list[float] = []
        try:
            while True:
                obs = collector.get_obs()
                if obs is None:
                    time.sleep(1.0 / args.rate)
                    continue
                if obs.t_sim >= end_t:
                    break
                # Where SHOULD the arms be right now, per the recording?
                frame = start + int(round((obs.t_sim - t0) * fps))
                if 0 <= frame < len(states):
                    tracking.append(
                        float(np.nanmax(np.abs(arms_of(obs.state) - arms_of(states[frame]))))
                    )
                command = executor.step(obs.t_sim, obs.state)
                if command is not None:
                    publisher.publish(command)
                time.sleep(1.0 / args.rate)
        finally:
            publisher.safe_stop()
            stop_recorder(recorder)
            final = collector.get_obs()
            collector.close()  # no cameras here, but keep teardown uniform (F-32b)

        stats = executor.stats.as_dict()
        print(f"done: {stats}")
        if tracking:
            print(
                f"arm tracking error vs the recording: mean {np.mean(tracking):.4f} rad, "
                f"max {np.max(tracking):.4f} rad over {len(tracking)} ticks"
            )
        if final is not None:
            print(
                f"final base odom {np.round(final.state[C.S_BASE_ODOM], 4)} "
                f"(recorded {np.round(states[-1, C.S_BASE_ODOM], 4)}), "
                f"spine {final.state[C.S_SPINE]:.4f} m"
            )
        if stats["clamped_ticks"] > stats["ticks"] * 0.5:
            print("WARNING: >50% of ticks delta-clamped — replay lagged the recording "
                  "(start pose mismatch or max-delta too small)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
