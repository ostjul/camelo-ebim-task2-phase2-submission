#!/usr/bin/env python3
"""THE parity test: does the rig's adapter reproduce the training pipeline?

`docs/realdata/15_RIG_WINDOW_RUNBOOK.md` §1.2 — the single most important
check before the first rollout. Nothing scored after it is interpretable
without it, because every way of getting preprocessing wrong here produces
a *plausible* rollout rather than an error: a mirrored gripper, a
normalizer built from another corpus, the left wrist where the right one
belongs, a 224x224 resize done with the wrong filter.

It drives a checkpoint through **the adapter the rig will actually run**
(`LeRobotAdapter`, `--state-layout s27a15 --action-layout s27a15`), fed
from a held-out episode's recorded frames as if they were live
observations, and diffs the result two ways:

**A — live A/B against the training path (the bar).** The reference is
`camelo.eval.offline_probe.build_runtime` + its own prediction loop: the
LeRobotDataset with the run's `image_transforms`, the saved pre/post
processors, `predict_action_chunk`. Both paths run in this process, frame
by frame, over the same frames, with the torch RNG reseeded identically
immediately before each policy call. The diff covers **every frame, every
chunk step and all 15 action dims** — not just the gripper channel the
probe JSON kept.

**B — against the stored probe JSON.** `episode_rows[i].gripper_trace[t]
.pred_g` is the predicted gripper chunk at every probed frame (the only
per-frame prediction the probe stored; the other per-frame rows —
`close_scan`, `demo_relative_grid`, `rollout_right_arm` — are derived
statistics, not raw predictions). `per_dim_mse_raw[15]` is recomputed from
the adapter's own step-0 predictions against the demo's actions and
compared with the stored vector, which is how the other 14 dims get
covered against the recorded run.

**Why A is the bar and B is not, for every policy.** A stored prediction
can only be reproduced if the run that made it was reproducible.
Measured here, not assumed: the tool probes determinism directly (three
forward passes on one frame — same seed twice, then a different seed) and
prints the verdict.

  * ACT is deterministic: B is bit-exact-modulo-float and is enforced.
  * VLA-JEPA's action head is flow matching — `action_head.predict_action`
    starts from `torch.randn(...)` drawn from the **global** RNG, with no
    `generator` argument, and `VLAJEPAPolicy.predict_action_chunk` accepts
    a `noise=` kwarg that it never forwards. The probe seeded nothing, so
    its stored `pred_g` is a sample nobody can redraw — not even by
    re-running the probe. For such a policy B is reported as INFORMATION
    with its distributional summary, and A (same seed, same conditioning,
    same sample) carries the verdict.

**Prove the instrument has teeth before trusting a PASS.**
`--negative-control <fault>` injects one named, realistic fault into the
ADAPTER's input and inverts the verdict: exit 0 now means the check
*caught* it. A parity test that passes on a broken pipeline is worse than
no parity test, and each fault maps to a row of the runbook's table.

**Never widen a tolerance to make this pass.** Read a failure by its
shape, per the runbook's table.

    .venv/bin/python tools/rig_parity_check.py \\
        --probe outputs/probes/act_3983190_100000.json \\
        --checkpoint outputs/runs/train_..._3983190/checkpoints/100000/pretrained_model \\
        --dataset outputs/datasets/ebim_task2_realrobotdata/task2_munich_s27a15_hires \\
        --episode 9 --state-layout s27a15 --action-layout s27a15
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camelo.policy.adapters import s27a15  # noqa: E402

# ---------------------------------------------------------------------------
# Tolerances. Justified, and constant across checkpoints so two parity runs
# are comparable — 15_RIG_WINDOW_RUNBOOK.md §1.2.
# ---------------------------------------------------------------------------

#: Max |Δ| on any action dim, any chunk step, any frame.
#:
#: Same weights, same corpus, one process — the only legitimate residual is
#: float32 reassociation: a different cuDNN convolution algorithm for a
#: batch of 1 vs the probe's batched loader, and a different reduction order
#: in the normalizer. That is 1e-6..1e-5 territory on these magnitudes.
#:
#: 1e-3 is where it stops being noise and starts being a bug, on both
#: channels of this contract:
#:   * gripper (a fraction in [0, 1]): the close threshold is 0.6048 and the
#:     open/closed modes sit at 0.945 / 0.265, so the decision gap is ~0.34.
#:     1e-3 is 0.3 % of it — 300x margin before a Δ could flip a verdict.
#:   * arms (absolute radians): the executor's slew clamp is 0.05 rad/tick,
#:     so 1e-3 rad (0.057°) is 2 % of one tick's motion and cannot change a
#:     trajectory the robot could follow.
#: The stored trace is rounded to 6 dp, which puts a 5e-7 floor under
#: comparison B regardless.
DEFAULT_TOL = 1e-3

#: Max |Δ| between the recomputed and stored `per_dim_mse_raw`. These are
#: squared radians accumulated over hundreds of frames; the runbook fixes
#: 1e-4, which on the largest ACT entry (2.56e-3) is ~4 % — loose enough to
#: absorb float noise, tight enough that a wrong normalizer (which inflates
#: every dim by a FACTOR, not an epsilon) cannot hide under it.
DEFAULT_MSE_TOL = 1e-4

#: Seed used for both paths. Any value works; it is fixed so two parity runs
#: of a stochastic policy are comparable with each other.
DEFAULT_SEED = 0


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable, no torch)
# ---------------------------------------------------------------------------
def episode_row(report: dict[str, Any], episode: int | None) -> dict[str, Any]:
    """The probe row for `episode`, or the first row when none is named."""
    rows = report.get("episode_rows") or []
    if not rows:
        raise SystemExit("probe report carries no episode_rows")
    if episode is None:
        return rows[0]
    for row in rows:
        if int(row["episode"]) == int(episode):
            return row
    have = [int(r["episode"]) for r in rows]
    raise SystemExit(f"episode {episode} is not in this probe (it holds {have})")


def stored_trace(row: dict[str, Any]) -> tuple[list[int], np.ndarray]:
    """(frames, (F, H) stored gripper chunks) from a probe episode row."""
    trace = row.get("gripper_trace")
    if not trace:
        raise SystemExit(
            f"episode {row['episode']} has no gripper_trace — the probe was run "
            "with --no-gripper-trace and there is nothing stored to diff against. "
            "Re-probe that checkpoint without the flag."
        )
    frames = [int(entry["t"]) for entry in trace]
    widths = {len(entry["pred_g"]) for entry in trace}
    if len(widths) != 1:
        raise SystemExit(f"gripper_trace has ragged chunk widths {sorted(widths)}")
    return frames, np.asarray([entry["pred_g"] for entry in trace], dtype=np.float64)


def diff_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """max / mean |a - b| plus where the max sits, as plain floats."""
    delta = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    if delta.size == 0:
        return {"max": float("nan"), "mean": float("nan"), "argmax": []}
    flat = int(np.argmax(delta))
    return {
        "max": float(delta.max()),
        "mean": float(delta.mean()),
        "argmax": [int(x) for x in np.unravel_index(flat, delta.shape)],
    }


def per_dim_diff(a: np.ndarray, b: np.ndarray) -> list[dict[str, float]]:
    """max / mean |Δ| per action dim over (F, H, A) arrays."""
    delta = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    return [
        {"max": float(delta[:, :, d].max()), "mean": float(delta[:, :, d].mean())}
        for d in range(delta.shape[-1])
    ]


def classify_determinism(same_seed: bool, other_seed: bool) -> tuple[str, str]:
    """Verdict from the three-pass probe. `*_seed` == "output was identical"."""
    if same_seed and other_seed:
        return "DETERMINISTIC", "identical output under two seeds — no sampling at inference"
    if same_seed and not other_seed:
        return (
            "STOCHASTIC_SEEDED",
            "output depends on the RNG but is fully determined by the seed",
        )
    return (
        "STOCHASTIC_UNSEEDED",
        "output changed under the SAME seed — sampling is not controlled by "
        "torch.manual_seed (a non-deterministic kernel, or a private generator)",
    )


#: Named faults `--negative-control` can inject into the adapter's input.
#: Each is a real mis-wiring the rig could ship with, not a synthetic
#: perturbation, and each names the runbook §1.2 row it should reproduce.
#:
#: ⚠ `camera_swap` is deliberately kept even though it has NO teeth on ACT.
#: MEASURED 2026-09-01: swapping ACT's two wrists moves its output by
#: 4.8e-07 — float noise. That is not the check going blind, it is a
#: property of the architecture: lerobot's ACT runs ONE shared backbone over
#: the cameras, adds only a 2-D SPATIAL sinusoidal embedding
#: (`ACTSinusoidalPositionEmbedding2d`) and no per-camera embedding, then
#: concatenates the tokens into a permutation-equivariant self-attention
#: encoder (`modeling_act.py:462-493`). Two cameras of the SAME resolution
#: (both wrists are 480x640) therefore produce identical positional
#: embeddings, and the output is exactly invariant to swapping them. So on
#: ACT this control reports a true negative; use `camera_blank` for a
#: camera-path control with teeth on every rung.
NEGATIVE_CONTROLS = {
    "camera_swap": "wrist_left <-> wrist_right (F-64, 'wrong everywhere')",
    "camera_blank": "the checkpoint's FIRST mapped camera zeroed (a dead/stale feed, §1.5)",
    "gripper_flip": "state gripper mirrored about 0.5 ('pred_g mirrored')",
    "state_roll": "the 27-dim state rolled by one ('arm dims wrong')",
}


def inject_fault(obs, fault: str, camera: str | None = None):
    """Return `obs` with one named fault applied to what the adapter sees."""
    import copy

    if fault not in NEGATIVE_CONTROLS:
        known = sorted(NEGATIVE_CONTROLS)
        raise SystemExit(f"unknown --negative-control {fault!r}; known: {known}")
    obs = copy.copy(obs)
    obs.images = dict(obs.images)
    obs.rig = dict(obs.rig)
    if fault == "camera_blank":
        if camera not in obs.images:
            raise SystemExit(f"camera_blank needs {camera!r} in the observation")
        obs.images[camera] = np.zeros_like(obs.images[camera])
    elif fault == "camera_swap":
        left, right = obs.images.get("wrist_left"), obs.images.get("wrist_right")
        if left is None or right is None:
            raise SystemExit("camera_swap needs both wrist cameras in the observation")
        obs.images["wrist_left"], obs.images["wrist_right"] = right, left
    elif fault == "gripper_flip":
        # The polarity mistake §1.4 exists to catch: rad -> 0.8 - rad is
        # exactly `open -> 1 - open` on the packed dim.
        opened = s27a15.open_fraction_from_knuckle_rad(obs.rig["right_gripper_rad"])
        obs.rig["right_gripper_rad"] = float(
            s27a15.knuckle_rad_from_open_fraction(1.0 - opened)
        )
    else:  # state_roll
        obs.rig["left_arm"] = np.roll(np.asarray(obs.rig["left_arm"], dtype=np.float64), 1)
        obs.rig["right_arm"] = np.roll(np.asarray(obs.rig["right_arm"], dtype=np.float64), 1)
    return obs


# ---------------------------------------------------------------------------
# torch / lerobot below this line and inside functions (AGENTS.md rule 3)
# ---------------------------------------------------------------------------
def _seed(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _chw_uint8_to_hwc(tensor) -> np.ndarray:
    """A dataset camera tensor -> the HxWx3 uint8 RGB array a ROS bridge gives.

    Refuses anything but uint8: the whole point of this path is that the
    adapter meets the same bytes the training dataloader met *before* the
    /255 cast, and a float tensor here would mean the assumption behind
    `_real_image_tensor` is wrong and the comparison is meaningless.
    """
    import torch

    if not torch.is_tensor(tensor):
        raise RuntimeError(f"camera value is {type(tensor).__name__}, not a tensor")
    if tensor.dtype != torch.uint8:
        raise RuntimeError(
            f"camera tensor is {tensor.dtype}, not uint8 — the dataset was not "
            "built with return_uint8=True, so this comparison would not be "
            "against the pixels training saw"
        )
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise RuntimeError(f"camera tensor is {tuple(tensor.shape)}, expected (3, H, W)")
    return tensor.permute(1, 2, 0).cpu().numpy()


def build_raw_dataset(runtime: dict[str, Any], dataset_root: Path, episode: int):
    """The episode's RAW frames: no delta_timestamps, no image transforms.

    This is the rig's-eye view — what a camera topic and a joint-state topic
    would deliver. The adapter must reach the training pixels from *here*,
    which is exactly the claim the check is testing.
    """
    from lerobot.datasets import LeRobotDataset

    recipe = runtime["train_config"]
    return LeRobotDataset(
        recipe["repo_id"],
        root=str(dataset_root),
        episodes=[episode],
        delta_timestamps=None,
        image_transforms=None,
        video_backend=recipe.get("video_backend"),
        return_uint8=True,
        tolerance_s=recipe["tolerance_s"],
    )


def observation_from_row(item, t_sim: float):
    """One raw dataset row -> the `Obs` a real-robot bridge would publish.

    The state goes back out through the gripper's RADIAN form on purpose:
    the bridge's job is to hand over raw knuckle feedback, and the adapter's
    job is to turn it into the open fraction training built. Shortcutting
    that would leave the one transform most likely to be wrong on the rig
    untested by the one check that exists to test it.
    """
    state = np.asarray(item["observation.state"], dtype=np.float32).ravel()
    if state.size != s27a15.STATE_DIM:
        raise RuntimeError(f"corpus state is {state.size}-dim, expected {s27a15.STATE_DIM}")
    images = {
        key.rsplit(".", 1)[-1]: _chw_uint8_to_hwc(value)
        for key, value in item.items()
        if key.startswith("observation.images.")
    }
    rig = {
        "left_arm": state[s27a15.S_LEFT_ARM],
        "right_arm": state[s27a15.S_RIGHT_ARM],
        "right_gripper_rad": float(
            s27a15.knuckle_rad_from_open_fraction(state[s27a15.S_RIGHT_GRIP])
        ),
        "left_wrench": state[s27a15.S_LEFT_WRENCH],
        "right_wrench": state[s27a15.S_RIGHT_WRENCH],
    }
    from camelo.policy.base import Obs

    return Obs(t_sim=t_sim, state=state, images=images, rig=rig)


def reference_chunk(runtime: dict[str, Any], ds, frame: int, seed: int) -> np.ndarray:
    """One chunk from the TRAINING path, at `frame`, under `seed`.

    Mirrors `offline_probe.predict_episode_chunks` at batch size 1: the same
    dataset (delta_timestamps + the run's image_transforms), the same uint8
    -> float32/255 staging, the same pre/post processors, the same
    `predict_action_chunk`. Batch size 1 because the adapter's is 1, and a
    flow-matching head draws `randn(B, H, A)` — a different B consumes a
    different slice of the RNG stream and the seeds would not line up.
    """
    import torch
    from torch.utils.data import default_collate

    policy = runtime["policy"]
    pre, post = runtime["preprocessor"], runtime["postprocessor"]
    staged = default_collate([ds[frame]])
    for cam in runtime["camera_keys"]:
        if cam in staged and staged[cam].dtype == torch.uint8:
            staged[cam] = staged[cam].to(dtype=torch.float32) / 255.0
    with torch.no_grad():
        processed = pre(staged)
        _seed(seed)
        chunk = policy.predict_action_chunk(processed)
        if not torch.is_tensor(chunk):
            chunk = torch.as_tensor(chunk)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)
        b, h, a = chunk.shape
        flat = post(chunk.reshape(b * h, a))
        if not torch.is_tensor(flat):
            flat = torch.as_tensor(flat)
        return flat.reshape(b, h, flat.shape[-1])[0].float().cpu().numpy()


def adapter_chunk(adapter, obs, seed: int) -> np.ndarray:
    """One chunk from the ADAPTER path under `seed` — the code the rig runs."""
    _seed(seed)
    return np.asarray(adapter.infer(obs), dtype=np.float64)


def probe_determinism(runtime, ds, frame: int, seed: int) -> dict[str, Any]:
    """Three forward passes: seed, seed, seed+1. Measured, not looked up."""
    first = reference_chunk(runtime, ds, frame, seed)
    again = reference_chunk(runtime, ds, frame, seed)
    other = reference_chunk(runtime, ds, frame, seed + 1)
    same_seed = bool(np.array_equal(first, again))
    other_seed = bool(np.array_equal(first, other))
    verdict, why = classify_determinism(same_seed, other_seed)
    return {
        "frame": int(frame),
        "verdict": verdict,
        "explanation": why,
        "same_seed_identical": same_seed,
        "other_seed_identical": other_seed,
        "same_seed_max_abs_delta": float(np.abs(first - again).max()),
        "other_seed_max_abs_delta": float(np.abs(first - other).max()),
    }


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from camelo.eval.offline_probe import (
        GRIPPER_ACTION_DIM,
        build_runtime,
        load_demo_arrays,
        per_dim_mse,
        resolve_model_dir,
    )
    from camelo.policy.adapters.lerobot_generic import LeRobotAdapter

    report = json.loads(Path(args.probe).read_text())
    row = episode_row(report, args.episode)
    episode = int(row["episode"])
    frames, stored_g = stored_trace(row)
    if args.max_frames and len(frames) > args.max_frames:
        step = max(1, len(frames) // args.max_frames)
        keep = list(range(0, len(frames), step))[: args.max_frames]
        frames = [frames[i] for i in keep]
        stored_g = stored_g[keep]

    model_dir = resolve_model_dir(Path(args.checkpoint).resolve())
    dataset_root = Path(args.dataset or report["dataset"]).resolve()

    print(f"checkpoint : {model_dir}")
    print(f"probe      : {args.probe}  ({report.get('policy_type')})")
    print(f"dataset    : {dataset_root}")
    print(f"episode    : {episode}   frames: {len(frames)}   chunk: {stored_g.shape[1]}")
    print(f"layouts    : state={args.state_layout} action={args.action_layout}")

    runtime = build_runtime(model_dir, dataset_root, args.device)
    ref_ds = runtime["make_dataset"]([episode])
    raw_ds = build_raw_dataset(runtime, dataset_root, episode)
    if len(ref_ds) != len(raw_ds):
        raise SystemExit(
            f"reference dataset has {len(ref_ds)} frames, raw dataset {len(raw_ds)} — "
            "the two are not indexing the same episode"
        )

    adapter = LeRobotAdapter(
        checkpoint=str(model_dir),
        device=args.device,
        name="rig-parity",
        action_layout=args.action_layout,
        state_layout=args.state_layout,
        gripper_command=args.gripper_command,
    )
    caption = args.task or s27a15.TASK_CAPTION
    adapter.reset(caption)
    print(
        f"adapter    : action_space={adapter.action_space} "
        f"n_action_steps={adapter.n_action_steps} chunk_dt={adapter.chunk_dt:.4f}s "
        f"({1.0 / adapter.chunk_dt:.1f} Hz)"
    )
    print(f"cameras    : {adapter.camera_map}")
    print(f"caption    : {caption!r}")

    det = probe_determinism(runtime, ref_ds, frames[0], args.seed)
    print(f"determinism: {det['verdict']} — {det['explanation']}")

    # --- the two paths, frame by frame --------------------------------------
    # The camera `camera_blank` zeroes: the first one the checkpoint maps,
    # i.e. `head` for every rung here. Taken from the resolved map rather
    # than named, so the control follows the checkpoint.
    blank_camera = next(iter(adapter.camera_map.values()), None)

    ref_chunks, ada_chunks = [], []
    for frame in frames:
        ref_chunks.append(reference_chunk(runtime, ref_ds, frame, args.seed))
        obs = observation_from_row(raw_ds[frame], frame / s27a15.FPS)
        if args.negative_control:
            obs = inject_fault(obs, args.negative_control, blank_camera)
        ada_chunks.append(adapter_chunk(adapter, obs, args.seed))
    ref = np.asarray(ref_chunks, dtype=np.float64)
    ada = np.asarray(ada_chunks, dtype=np.float64)
    if ref.shape != ada.shape:
        raise SystemExit(
            f"the two paths returned different chunk shapes: reference {ref.shape} "
            f"vs adapter {ada.shape} — the adapter is not running this checkpoint's "
            "geometry"
        )

    live = diff_stats(ada, ref)
    live_dims = per_dim_diff(ada, ref)
    live_steps = [
        float(np.abs(ada[:, h, :] - ref[:, h, :]).max()) for h in range(ref.shape[1])
    ]

    # --- B: against the stored probe ---------------------------------------
    horizon = min(stored_g.shape[1], ada.shape[1])
    stored_diff = diff_stats(ada[:, :horizon, GRIPPER_ACTION_DIM], stored_g[:, :horizon])

    demos = load_demo_arrays(dataset_root, [episode])
    demo_action = demos[episode]["action"][frames]
    recomputed_mse = per_dim_mse(ada[:, 0, :], demo_action)
    stored_mse = np.asarray(row["per_dim_mse_raw"], dtype=np.float64)
    mse_diff = diff_stats(recomputed_mse, stored_mse)

    # --- verdicts ----------------------------------------------------------
    reproducible = det["verdict"] in ("DETERMINISTIC", "STOCHASTIC_SEEDED")
    live_pass = bool(live["max"] <= args.tol) and reproducible
    # The stored probe is only a BAR for a policy whose sample can be redrawn.
    stored_is_bar = det["verdict"] == "DETERMINISTIC"
    stored_pass = bool(stored_diff["max"] <= args.tol and mse_diff["max"] <= args.mse_tol)
    overall = live_pass and (stored_pass or not stored_is_bar)

    names = runtime["action_names"] or list(s27a15.ACTION_NAMES)
    print()
    print("A — adapter vs the training path (same seed, all frames/steps/dims)")
    print(f"{'dim':>3}  {'name':<62} {'max|Δ|':>10} {'mean|Δ|':>10}")
    for d, (name, stat) in enumerate(zip(names, live_dims, strict=False)):
        print(f"{d:>3}  {name:<62} {stat['max']:>10.2e} {stat['mean']:>10.2e}")
    print(f"     OVERALL max |Δ| = {live['max']:.3e}  (tol {args.tol:.0e})  at "
          f"[frame_i, step, dim] = {live['argmax']}")
    print(f"     worst chunk step = {int(np.argmax(live_steps))} "
          f"(max |Δ| {max(live_steps):.3e})")

    print()
    print("B — adapter vs the stored probe JSON")
    print(f"     gripper chunk  max |Δ| = {stored_diff['max']:.3e} "
          f"mean = {stored_diff['mean']:.3e}  (tol {args.tol:.0e})")
    print(f"     per-dim MSE    max |Δ| = {mse_diff['max']:.3e}  (tol {args.mse_tol:.0e})")
    if not stored_is_bar:
        print("     ^ INFORMATION ONLY: this policy samples at inference and the "
              "probe seeded nothing,")
        print("       so the stored prediction is a draw that cannot be reproduced. "
              "A carries the verdict.")

    if args.negative_control:
        # Verdict INVERTED: the run exists to show the check can fail.
        caught = bool(live["max"] > args.tol)
        print()
        print(f"NEGATIVE CONTROL {args.negative_control!r} — "
              f"{NEGATIVE_CONTROLS[args.negative_control]}")
        print(f"     max |Δ| = {live['max']:.3e} vs tol {args.tol:.0e}")
        print(f"CONTROL: {'PASS — the check caught it' if caught else 'NO EFFECT'}")
        if not caught:
            print("     The adapter's output did not move. Two possibilities, and the")
            print("     tool cannot tell them apart — resolve it before trusting a PASS:")
            print("       1. the model is genuinely INVARIANT to this perturbation")
            print("          (measured: lerobot ACT is exactly invariant to swapping two")
            print("          same-resolution cameras — one shared backbone, spatial-only")
            print("          position embedding, permutation-equivariant encoder), or")
            print("       2. the fault never reached the model, i.e. the check is blind.")
            print("     Distinguish them with a control the model cannot ignore "
                  "(camera_blank,")
            print("     gripper_flip, state_roll). At least one MUST have teeth per rung.")
        return {
            "schema": "camelo.rig_parity_negative_control/1",
            "pass": caught,
            "negative_control": args.negative_control,
            "checkpoint": str(model_dir),
            "episode": episode,
            "n_frames": len(frames),
            "max_abs_delta": live["max"],
            "blank_camera": blank_camera,
            "per_dim": live_dims,
            "tolerance": args.tol,
        }

    print()
    print(f"A (training-path parity) : {'PASS' if live_pass else 'FAIL'}")
    print(
        f"B (stored probe)         : "
        f"{('PASS' if stored_pass else 'FAIL') if stored_is_bar else 'n/a (stochastic)'}"
    )
    print(f"PARITY                   : {'PASS' if overall else 'FAIL'}")
    if not overall:
        print()
        print("Read the failure by its SHAPE (runbook §1.2) — do not widen a tolerance:")
        print("  pred_g mirrored about 0.5     -> gripper polarity on the state side")
        print("  every dim inflated by a factor-> wrong normalizer (not this corpus)")
        print("  gripper fine, arms wrong      -> 27-dim state packing order")
        print("  fine at step 0, worse later   -> frame indexing / action_delta_indices")
        print("  wrong everywhere              -> camera assignment (F-64)")

    result = {
        "schema": "camelo.rig_parity/1",
        "pass": bool(overall),
        "checkpoint": str(model_dir),
        "probe": str(Path(args.probe).resolve()),
        "dataset": str(dataset_root),
        "policy_type": report.get("policy_type"),
        "episode": episode,
        "n_frames": len(frames),
        "frames": frames,
        "chunk_horizon": int(ref.shape[1]),
        "action_dim": int(ref.shape[2]),
        "device": args.device,
        "seed": args.seed,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "layouts": {"state": args.state_layout, "action": args.action_layout},
        "caption": caption,
        "camera_map": adapter.camera_map,
        "n_action_steps": adapter.n_action_steps,
        "chunk_dt": adapter.chunk_dt,
        "gripper_command": args.gripper_command,
        "tolerances": {"abs": args.tol, "per_dim_mse": args.mse_tol},
        "determinism": det,
        "live_ab": {
            "pass": live_pass,
            "overall": live,
            "per_dim": live_dims,
            "per_chunk_step_max": live_steps,
            "action_names": list(names),
        },
        "stored_probe": {
            "is_bar": stored_is_bar,
            "pass": stored_pass,
            "gripper_chunk": stored_diff,
            "per_dim_mse": mse_diff,
            "per_dim_mse_recomputed": [float(x) for x in recomputed_mse],
            "per_dim_mse_stored": [float(x) for x in stored_mse],
        },
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {out}")
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--probe", required=True, help="outputs/probes/<rung>_<job>_<step>.json")
    p.add_argument("--checkpoint", required=True, help="checkpoint dir or its pretrained_model/")
    p.add_argument("--dataset", default=None, help="corpus root; default: the probe's own")
    p.add_argument(
        "--episode", type=int, default=None, help="held-out episode id; default: the probe's first"
    )
    p.add_argument("--state-layout", default=s27a15.LAYOUT)
    p.add_argument("--action-layout", default=s27a15.LAYOUT)
    p.add_argument(
        "--gripper-command",
        default=s27a15.DEFAULT_GRIPPER_COMMAND,
        choices=sorted(s27a15.GRIPPER_COMMAND_CONVENTIONS),
        help="driver units for the decoded chunk; does not affect the diff "
        "(which is on the model's own output) but is recorded in the report",
    )
    p.add_argument("--task", default=None, help=f"caption; default {s27a15.TASK_CAPTION!r}")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--tol", type=float, default=DEFAULT_TOL)
    p.add_argument("--mse-tol", type=float, default=DEFAULT_MSE_TOL)
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="subsample the probed frames (evenly) to bound wall time; default: all",
    )
    p.add_argument(
        "--negative-control",
        default=None,
        choices=sorted(NEGATIVE_CONTROLS),
        help="inject one named fault into the adapter's input and INVERT the "
        "verdict: exit 0 means the check caught it. Run this before trusting a "
        "clean PASS — a parity test that cannot fail is not a test",
    )
    p.add_argument("--out", default=None, help="write the full report JSON here")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return 0 if run(args)["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
