#!/usr/bin/env python3
"""Batch collection: play augmented heuristic variants in sim, one per episode.

    python scripts/collect_augmented.py \
        --manifest data/heuristic/task2_fixpos_200/aug_v1/manifest.json \
        --rollout-s 120

Each episode swaps `HeuristicAdapter.load_reference()` to the next variant
in `--manifest` (from scripts/generate_augmented_trajectories.py) via the
`run_batch` `episode_setup` hook, optionally jittering the approach's final
waypoint too (`--goal-jitter-xy-mm` / `--goal-jitter-yaw-deg`). Requires:
task2 sim with --record and --no-browser, helper stack in position mode,
AND the eval stack (bash <benchmark>/scripts/evaluation/task2/run.sh up).
Writes outputs/eval/<run>/results.csv + summary.json — each row carries
variant/variant_family/variant_sources[/goal_jitter_*] provenance columns
— plus family_summary.json (per-family mean IoU) next to them.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camelo.cli import (
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
from camelo.control.approach import _GOAL_JITTER_XY_MAX_M, _GOAL_JITTER_YAW_MAX_RAD

# Mirrors ApproachController.set_goal_jitter's own caps (camelo/control/
# approach.py) in the CLI's units (mm / deg) so a bad sweep value is refused
# up front, before any sim time is spent, rather than mid-batch on episode N.
_JITTER_XY_MM_MAX = _GOAL_JITTER_XY_MAX_M * 1000.0
_JITTER_YAW_DEG_MAX = math.degrees(_GOAL_JITTER_YAW_MAX_RAD)


def load_manifest(path: str | Path) -> list[dict]:
    """Parse + validate manifest.json from generate_augmented_trajectories.py.

    Refuses (SystemExit) on: unparsable JSON, no `variants`, or a variant
    missing `name`/`family`/`actions` or whose `actions` file does not exist
    — every failure mode a stale or hand-edited manifest could produce.
    """
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise SystemExit(f"--manifest not found: {manifest_path}")
    try:
        data = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--manifest {manifest_path} is not valid JSON: {exc}") from exc
    variants = data.get("variants")
    if not variants:
        raise SystemExit(f"--manifest {manifest_path} has no variants — nothing to collect")
    for i, variant in enumerate(variants):
        missing = [key for key in ("name", "family", "actions") if key not in variant]
        if missing:
            raise SystemExit(f"--manifest {manifest_path} variant {i} missing {missing[0]!r}")
        actions_path = Path(variant["actions"])
        if not actions_path.is_file():
            raise SystemExit(
                f"--manifest {manifest_path} variant {variant['name']!r}: actions file not "
                f"found: {actions_path}"
            )
    return variants


def build_plan(variants: list[dict], episodes: int, order: str, seed: int) -> list[dict]:
    """`episodes` variant assignments, one per planned episode.

    `episodes<=0` means one episode per manifest variant. `roundrobin`
    cycles the manifest order (`i % len(variants)`); `shuffle` draws ONE
    `numpy.random.default_rng(seed)` permutation and cycles that instead —
    deterministic for a given seed, and simple enough that "the order" is a
    single re-derivable array rather than a reshuffle-per-lap policy.
    """
    n = len(variants)
    total = episodes if episodes > 0 else n
    if order == "roundrobin":
        indices = [i % n for i in range(total)]
    elif order == "shuffle":
        perm = np.random.default_rng(seed).permutation(n)
        indices = [int(perm[i % n]) for i in range(total)]
    else:
        raise ValueError(f"unknown order {order!r}; expected 'roundrobin' or 'shuffle'")
    return [variants[i] for i in indices]


def sample_goal_jitter(
    rng: np.random.Generator, xy_mm: float, yaw_deg: float
) -> tuple[float, float, float]:
    """Uniform draw within ``±xy_mm`` / ``±yaw_deg``, converted to metres/radians
    (``ApproachController.set_goal_jitter``'s own units)."""
    dx_mm = rng.uniform(-xy_mm, xy_mm)
    dy_mm = rng.uniform(-xy_mm, xy_mm)
    dyaw_deg = rng.uniform(-yaw_deg, yaw_deg)
    return dx_mm / 1000.0, dy_mm / 1000.0, math.radians(dyaw_deg)


def family_summary(rows: list[dict]) -> dict:
    """Per-family mean IoU + scored-episode count, keyed by `variant_family`.

    Rows with no numeric `iou` (errored episodes) are skipped from both —
    but a family that scored zero episodes still appears, with
    `mean_iou: None`, rather than being silently absent: absence and
    "measured zero" must not read the same (AGENTS.md, "the summary lies,
    not the measurement").
    """
    families: dict[str, list[float]] = {}
    for row in rows:
        family = row.get("variant_family")
        if not family:
            continue
        families.setdefault(family, [])
        try:
            iou_val = float(row.get("iou"))
        except (TypeError, ValueError):
            continue
        families[family].append(iou_val)
    return {
        family: {
            "episodes": len(values),
            "mean_iou": sum(values) / len(values) if values else None,
        }
        for family, values in sorted(families.items())
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_policy_args(parser)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="manifest.json from scripts/generate_augmented_trajectories.py",
    )
    parser.add_argument(
        "--episodes", type=int, default=0, help="0 = one episode per manifest variant"
    )
    parser.add_argument("--rollout-s", type=float, default=120.0, help="sim seconds per episode")
    parser.add_argument("--output", type=Path, default=Path("outputs/eval"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--skip-polarity-check", action="store_true")
    parser.add_argument(
        "--variant-order",
        choices=["roundrobin", "shuffle"],
        default="roundrobin",
        help="how manifest variants are assigned to episodes (default: roundrobin)",
    )
    parser.add_argument(
        "--order-seed", type=int, default=0, help="--variant-order shuffle only"
    )
    parser.add_argument(
        "--goal-jitter-xy-mm",
        type=float,
        default=0.0,
        help="per-episode uniform +/- draw on the approach goal's x/y (mm); 0 = off",
    )
    parser.add_argument(
        "--goal-jitter-yaw-deg",
        type=float,
        default=0.0,
        help="per-episode uniform +/- draw on the approach goal's yaw (deg); 0 = off",
    )
    parser.add_argument("--goal-jitter-seed", type=int, default=0)
    parser.add_argument(
        "--inter-episode-pause-s", type=float, default=0.0,
        help="wall seconds to sleep before each episode after the first -- headroom "
        "for a lockstep-driven recorder session to finish saving/encoding the "
        "previous episode before this one's scene reset (0 = no pause)",
    )
    args = parser.parse_args()
    setup_logging()

    if abs(args.goal_jitter_xy_mm) > _JITTER_XY_MM_MAX:
        raise SystemExit(
            f"--goal-jitter-xy-mm {args.goal_jitter_xy_mm} exceeds the approach controller's "
            f"±{_JITTER_XY_MM_MAX:.0f} mm cap (ApproachController.set_goal_jitter)"
        )
    if abs(args.goal_jitter_yaw_deg) > _JITTER_YAW_DEG_MAX:
        raise SystemExit(
            f"--goal-jitter-yaw-deg {args.goal_jitter_yaw_deg} exceeds the approach controller's "
            f"±{_JITTER_YAW_DEG_MAX:.0f} deg cap (ApproachController.set_goal_jitter)"
        )
    jitter_enabled = args.goal_jitter_xy_mm > 0.0 or args.goal_jitter_yaw_deg > 0.0
    if jitter_enabled and args.skip_approach:
        print(
            "WARNING: --goal-jitter-* given with --skip-approach: there is no "
            "ApproachController to jitter — the jitter will be skipped every episode"
        )

    if args.backend == "remote":
        # RemoteBackend's model lives in the server process; there is no
        # in-process adapter object here to swap references onto.
        raise SystemExit(
            "--backend remote is not supported by collect_augmented.py: the adapter object "
            "must be reachable in-process to swap references between episodes"
        )
    if args.adapter == "dummy":
        print("[collect_augmented] --adapter not given; defaulting to heuristic")
        args.adapter = "heuristic"
    elif not args.adapter.startswith("heuristic"):
        raise SystemExit(
            f"collect_augmented.py only drives the heuristic adapter (references are swapped "
            f"in-process between episodes); got --adapter {args.adapter!r}"
        )

    variants = load_manifest(args.manifest)
    plan = build_plan(variants, args.episodes, args.variant_order, args.order_seed)
    jitter_rng = np.random.default_rng(args.goal_jitter_seed)

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
    adapter = getattr(backend, "adapter", None)
    if adapter is None:
        raise SystemExit(
            "collect_augmented.py needs a backend with a reachable .adapter to swap references"
        )
    topics = topics_from_args(args)
    with ros_session("camelo_collect_augmented", world=args.world) as node:
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
                grasp_gate=_gate,
                grasp_observer=make_grasp_observer_from_args(args, _gate),
            )
            if not args.skip_polarity_check:
                wait_for_obs(collector)
                # Reset BEFORE the gate, not after (F-80) — see eval_batch.py.
                runner.reset_scene()
                verify_gripper_polarity(collector, publisher)

            def episode_setup(index: int) -> dict:
                if index > 0 and args.inter_episode_pause_s > 0:
                    print(
                        f"[collect_augmented] pausing {args.inter_episode_pause_s:.0f}s "
                        f"before episode {index} (recorder save headroom)",
                        flush=True,
                    )
                    time.sleep(args.inter_episode_pause_s)
                variant = plan[index]
                adapter.load_reference(variant["actions"])
                extra = {
                    "variant": variant["name"],
                    "variant_family": variant["family"],
                    "variant_sources": ";".join(variant.get("sources", [])),
                }
                if jitter_enabled and runner.approach is not None:
                    dx_m, dy_m, dyaw_rad = sample_goal_jitter(
                        jitter_rng, args.goal_jitter_xy_mm, args.goal_jitter_yaw_deg
                    )
                    runner.approach.set_goal_jitter(dx_m, dy_m, dyaw_rad)
                    extra["goal_jitter_dx_mm"] = round(dx_m * 1000.0, 2)
                    extra["goal_jitter_dy_mm"] = round(dy_m * 1000.0, 2)
                    extra["goal_jitter_dyaw_deg"] = round(math.degrees(dyaw_rad), 2)
                return extra

            summary = run_batch(
                runner, len(plan), args.rollout_s, args.output,
                rate_hz=args.rate, run_name=args.run_name,
                episode_setup=episode_setup,
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

            run_dir = Path(summary["results_csv"]).parent
            with open(summary["results_csv"], newline="") as f:
                result_rows = list(csv.DictReader(f))
            fam_summary = family_summary(result_rows)
            (run_dir / "family_summary.json").write_text(json.dumps(fam_summary, indent=2))
            for family, stats in fam_summary.items():
                mean_iou = stats["mean_iou"]
                mean_s = f"{mean_iou:.3f}" if mean_iou is not None else "n/a"
                print(f"family {family}: {stats['episodes']} scored episodes, mean IoU {mean_s}")

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
