#!/usr/bin/env python3
"""The offline selection probe (docs/realdata/00_PROTOCOL.md §5, 14_PROBE.md).

Ranks trained checkpoints on the frozen held-out episodes of the real-robot
Munich corpus *before* the rig window. Training loss is not a selection
signal here, and neither is held-out loss (00 §5; MEMORY; GRASP §0) — this
is the instrument that decides which checkpoints get rig minutes.

Three metrics, in decreasing order of authority:

**M1 — gripper-close timing verdict (PRIMARY).** Every held-out episode is
replayed open loop: the recorded observations are fed to the policy frame by
frame, nothing is executed, and the predicted action chunk is kept at each
step. A chunk "commands a close" when the right-gripper action dim drops
below the close threshold; it reaches the wire only if that happens inside
the first `n_action_steps` of the chunk (the executed window — a close at
step 30 is discarded at the next replan). The episode's predicted close is
the absolute frame `t* + step` of the first such event, which is exactly
`t_demo` for a policy that reproduces the demonstration. Verdict by
MECHANISM per GRASP §0: TIMED / MISPLACED / UNTIMED / ABSENT (see
`classify_verdict`). The **episode is the independent unit**; no chunk-level
statistic appears in the headline. Every report also scores three **null
models** — a state-hold copycat, an action-hold, and the demonstration
oracle — on the same episodes and the same chunk geometry, because on this
corpus the gripper action is largely recoverable from the gripper *state*
and a verdict count with no null model beside it is not a measurement.

**M2 — per-dim action MSE**, raw (rad², and gripper fraction²) and
normalized by the corpus per-dim std. Tabulated per dim with the dataset's
own feature names. A tiebreak and a broken-checkpoint alarm, never a ranking
(00 §5.2).

**M3 — open-loop chunk rollout error.** At anchor frames every
`--anchor-period-s` seconds, the predicted chunk is compared against the
demonstration's own future actions, giving a per-horizon-step error curve.
This separates "predicts the next pose" from "predicts a coherent plan".

Everything the summary claims is re-derivable from the per-episode /
per-frame rows kept in the JSON (AGENTS.md, "the summary lies, not the
measurement"): an ABSENT verdict must show no close-crossing anywhere in the
per-frame gripper trace, and it does.

CLI::

    .venv/bin/python -m camelo.eval.offline_probe \
        --checkpoint outputs/runs/train_..._3983190/checkpoints/008333 \
        --dataset outputs/datasets/ebim_task2_realrobotdata/task2_munich_s27a15_hires \
        --split heldout \
        --out outputs/probes/act_008333

Writes `<out>.json` (every row) and `<out>.md` (the per-episode verdict
table, then the aggregates).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Corpus constants — 00_PROTOCOL.md §0.2, frozen with the corpus.
# ---------------------------------------------------------------------------

#: `action[14]` = right_follower_gello ... target_gripper_width_percent_value.
GRIPPER_ACTION_DIM = 14
#: `action[7:14]` / `observation.state[7:14]` = the right FR3's seven joints.
RIGHT_ARM_SLICE = slice(7, 14)
#: `action[0:7]` = the left arm: alive but off-task, and it dominates any
#: aggregate without contributing competence (00 §5.2).
LEFT_ARM_SLICE = slice(0, 7)

#: The frozen close threshold. Derived (see `derive_close_threshold`) from
#: the held-out demonstrations' own gripper distribution, which is bimodal
#: at 0.209 (closed) / 1.000 (open) with a 23 % ramp between them. Three
#: independent estimators — 2-means midpoint 0.6048, Otsu 0.5996, midpoint of
#: the cluster medians 0.6047 — all put the separating value at 0.60, which is
#: also what 00 §5.2 specifies (`action[14] < 0.6`).
FROZEN_CLOSE_THRESHOLD = 0.60
#: How far the derived threshold may drift from the frozen one before the
#: run is refused. A moved threshold silently re-scores every checkpoint.
CLOSE_THRESHOLD_DRIFT_TOL = 0.02

#: Timing tolerance for TIMED, in seconds. 1.0 s = 20 frames at 20 Hz — the
#: `-20` near cell of 00 §5.2's offset grid, and ~2x the executed window of
#: the fastest rung. A convention, not a measurement: the signed per-episode
#: timing error is in the JSON, so any other tolerance is re-derivable.
DEFAULT_TIMING_TOL_S = 1.0

#: Right-arm joint-space L2 tolerance for "the arm is at the demo's grasp
#: pose", in radians: the 90th percentile of the displacement the
#: demonstrations' own right arm covers in +/- the timing tolerance around
#: their own close. Derived per run by `derive_pose_tolerance` on the frozen
#: 22-episode held-out split, which produces 0.619 at a 1.0 s tolerance.
FROZEN_POSE_TOL_RAD = 0.62
#: Drift budget on that derivation, same purpose as the threshold's: a moved
#: tolerance silently re-scores every checkpoint ever probed.
POSE_TOL_DRIFT_TOL = 0.05

#: 00 §5.2 / GRASP §0 pre-grasp offset grid, in frames before the demo close.
DEFAULT_OFFSETS = (-60, -40, -20, -12, -8, -4, 0)
#: The two cells a policy must keep clean regardless of what the demo does.
FAR_OFFSETS = (-60, -40)
#: Demo-relative PASS bars (GRASP §0, stochastic-policy ruling 2026-08-25).
NEAR_RATE_MIN = 0.50
FAR_RATE_MAX = 0.05

VERDICTS = ("TIMED", "MISPLACED", "UNTIMED", "ABSENT")

#: Default RNG seed. Every rung on this ladder except ACT has a *stochastic*
#: action head (flow matching / diffusion), so an unseeded probe reports a
#: single draw and calls it a ranking — 14_PROBE.md §9.9 measured two
#: identical EO-1 runs disagreeing on 19 of 22 episodes' predicted close
#: frame and swinging the headline PASS by 2 episodes. A seed does not make
#: a stochastic head deterministic in any deep sense; it makes a report
#: *reproducible*, and it makes the seed-to-seed spread measurable.
DEFAULT_SEED = 0


# ---------------------------------------------------------------------------
# Pure logic. numpy only — unit-tested in tests/test_offline_probe.py with
# neither torch nor lerobot installed.
# ---------------------------------------------------------------------------


def derive_close_threshold(values: np.ndarray) -> dict[str, float]:
    """Split the demo gripper channel into its open/closed modes.

    1-D Lloyd (2-means) seeded at the extremes; the threshold is the midpoint
    of the two converged centres. The corpus's gripper *action* is a target
    width fraction with a real ramp between the modes (23.3 % of held-out
    frames sit strictly between them), so a mid-mode cut is the right
    landmark and a "first value below" test is not enough on its own.

    Returns the threshold plus the two centres and the mass on each side, so
    the number in the report can be re-derived from the report.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        raise ValueError("cannot derive a close threshold from zero frames")
    lo, hi = float(v.min()), float(v.max())
    if hi - lo <= 0.0:
        raise ValueError(
            f"gripper channel is constant at {lo!r}: no open/closed modes to separate"
        )
    centres = np.array([lo, hi], dtype=np.float64)
    for _ in range(200):
        labels = np.abs(v[:, None] - centres[None, :]).argmin(axis=1)
        if not labels.any() or labels.all():
            break
        new = np.array([v[labels == 0].mean(), v[labels == 1].mean()])
        if np.allclose(new, centres):
            centres = new
            break
        centres = new
    threshold = float(centres.mean())
    return {
        "threshold": threshold,
        "centre_closed": float(centres.min()),
        "centre_open": float(centres.max()),
        "frac_below": float((v < threshold).mean()),
        "frac_above": float((v >= threshold).mean()),
        "n_frames": int(v.size),
    }


def first_close_index(series: np.ndarray, threshold: float) -> int | None:
    """Index of the first commanded close in `series`, or None.

    A close is a 1->0 crossing: index `i > 0` qualifies when
    `series[i] < threshold <= series[i-1]`. Index 0 qualifies when the
    series *starts* below the threshold, because a chunk (or a demo) that
    opens already closed is commanding a close from its first step — there
    is simply no earlier frame in which to observe the transition.
    """
    s = np.asarray(series, dtype=np.float64).ravel()
    if s.size == 0:
        return None
    if s[0] < threshold:
        return 0
    below = s < threshold
    hits = np.nonzero(below[1:] & ~below[:-1])[0]
    return int(hits[0]) + 1 if hits.size else None


@dataclass(frozen=True)
class ChunkClose:
    """What one predicted chunk does with the gripper channel."""

    any_step: int | None
    """First chunk step commanding a close, anywhere in the horizon."""
    executed_step: int | None
    """Same, restricted to the executed window; None if outside it."""
    min_value: float
    """Minimum predicted gripper value in the chunk — the graded signal a
    binary test cannot show (a channel pinned at 0.99 is not the same
    failure as one that slides to 0.62)."""


def chunk_close_event(chunk_gripper: np.ndarray, threshold: float, n_executed: int) -> ChunkClose:
    """Score one predicted chunk's gripper channel."""
    g = np.asarray(chunk_gripper, dtype=np.float64).ravel()
    if g.size == 0:
        raise ValueError("empty chunk")
    if n_executed <= 0:
        raise ValueError(f"n_executed must be positive, got {n_executed}")
    idx = first_close_index(g, threshold)
    executed = idx if (idx is not None and idx < n_executed) else None
    return ChunkClose(any_step=idx, executed_step=executed, min_value=float(g.min()))


def scan_close_events(
    frames: list[int],
    events: list[ChunkClose],
) -> dict[str, Any]:
    """Reduce a whole episode's per-frame chunk events to its close landmark.

    `predicted_close_frame` is `t* + step` for the first frame `t*` whose
    chunk commands a close inside the executed window. A policy that
    reproduces the demonstration lands exactly on the demo's close frame,
    whichever frame it was first visible from — which is what makes the
    timing error comparable across policies with different chunk sizes.
    """
    if len(frames) != len(events):
        raise ValueError(f"{len(frames)} frames but {len(events)} events")
    n_any = sum(1 for e in events if e.any_step is not None)
    n_exec = sum(1 for e in events if e.executed_step is not None)
    first_any_frame = next(
        (f for f, e in zip(frames, events, strict=True) if e.any_step is not None), None
    )
    first_exec = next(
        ((f, e) for f, e in zip(frames, events, strict=True) if e.executed_step is not None), None
    )
    out: dict[str, Any] = {
        "n_probed_frames": len(frames),
        "n_frames_any_close": n_any,
        "n_frames_executed_close": n_exec,
        "frac_frames_any_close": (n_any / len(frames)) if frames else 0.0,
        "frac_frames_executed_close": (n_exec / len(frames)) if frames else 0.0,
        "first_any_close_frame": first_any_frame,
        "first_executed_close_frame": None,
        "first_executed_close_step": None,
        "predicted_close_frame": None,
        "min_predicted_gripper": min((e.min_value for e in events), default=float("nan")),
    }
    if first_exec is not None:
        frame, event = first_exec
        assert event.executed_step is not None
        out["first_executed_close_frame"] = frame
        out["first_executed_close_step"] = event.executed_step
        out["predicted_close_frame"] = frame + event.executed_step
    return out


def classify_verdict(
    *,
    n_frames_any_close: int,
    n_frames_executed_close: int,
    predicted_close_frame: int | None,
    demo_close_frame: int | None,
    pose_err_rad: float | None,
    timing_tol_frames: int,
    pose_tol_rad: float,
) -> tuple[str, str]:
    """The GRASP §0 mechanism verdict, applied to an open-loop frame sweep.

    Order is load-bearing:

    1. **ABSENT** — no chunk contains a close anywhere, at any probed frame.
       The probe feeds fresh in-distribution dataset frames, so every ABSENT
       measured here is *supervision*-ABSENT (GRASP §0's scoping note).
    2. **MISPLACED (chunk placement)** — closes exist in the chunks but never
       land inside the executed window. The policy knows *whether*, not
       *where*; π0-FAST's signature. An executor-horizon problem, not a
       supervision problem.
    3. **UNTIMED** — a close reaches the wire, but at the wrong moment:
       `|t_pred - t_demo|` exceeds the tolerance. Closes fire anywhere.
    4. **MISPLACED (pose)** — the close reaches the wire at the right *time*
       but the arm is not at the demonstration's grasp configuration.
    5. **TIMED** — right window, right pose. The target.

    Timing is checked before pose deliberately: a close 10 s early is also
    far from the grasp pose, and calling that MISPLACED would hide the
    timing failure that caused it.
    """
    if n_frames_any_close == 0:
        return "ABSENT", "no chunk commands a close at any probed frame (supervision-ABSENT)"
    if n_frames_executed_close == 0:
        return (
            "MISPLACED",
            "chunk_placement: close present in the chunk, never inside the executed window",
        )
    if predicted_close_frame is None:
        raise ValueError("executed closes counted but no predicted_close_frame given")
    if demo_close_frame is None:
        return "UNTIMED", "the demonstration never closes; any policy close is untimed"
    dt = predicted_close_frame - demo_close_frame
    if abs(dt) > timing_tol_frames:
        return "UNTIMED", f"close fires {dt:+d} frames from the demo close"
    if pose_err_rad is None:
        raise ValueError("pose_err_rad is required once timing is inside tolerance")
    if pose_err_rad > pose_tol_rad:
        return (
            "MISPLACED",
            f"pose: right arm {pose_err_rad:.3f} rad from the demo grasp pose "
            f"(tol {pose_tol_rad:.3f})",
        )
    return "TIMED", f"close {dt:+d} frames, arm {pose_err_rad:.3f} rad from the demo grasp pose"


def demo_executed_close(
    demo_gripper: np.ndarray, frame: int, threshold: float, n_executed: int
) -> int | None:
    """Score the *recorded* actions through the probe's own rule.

    The demonstration's own next `n_executed` actions are its "chunk". This
    is what makes PASS demo-relative rather than absolute: the episode
    ceiling is not 1.0, and an absolute bar reads like 50 % while being 75 %
    of what a perfect imitator can score (GRASP §0).
    """
    g = np.asarray(demo_gripper, dtype=np.float64).ravel()
    if frame < 0 or frame >= g.size:
        return None
    window = g[frame : frame + n_executed]
    return 1 if first_close_index(window, threshold) is not None else 0


def demo_relative_cells(
    *,
    offsets: tuple[int, ...],
    policy_rates: dict[int, float | None],
    demo_cells: dict[int, int | None],
    far_offsets: tuple[int, ...] = FAR_OFFSETS,
    near_rate_min: float = NEAR_RATE_MIN,
    far_rate_max: float = FAR_RATE_MAX,
) -> dict[str, Any]:
    """GRASP §0's demo-relative PASS, cell for cell.

    * near cell where the demo closes in the executed window -> policy rate
      must be >= `near_rate_min` (the modal rollout matches);
    * near cell where the demo does **not** close -> no requirement, the
      policy is penalised neither way (ceiling-limited cells);
    * far cells -> policy rate must be <= `far_rate_max`, regardless of the
      demo. A far close is premature, not necessarily on air (A1d), and the
      bar stands because full-corpus ACT reads 0.000 on 16/16 far cells.

    A cell whose frame falls outside the episode is `null` and constrains
    nothing.
    """
    cells = []
    ok = True
    for off in offsets:
        rate = policy_rates.get(off)
        demo = demo_cells.get(off)
        far = off in far_offsets
        if rate is None:
            required, passed = None, None
        elif far:
            required, passed = f"<= {far_rate_max}", rate <= far_rate_max
        elif demo == 1:
            required, passed = f">= {near_rate_min}", rate >= near_rate_min
        else:
            required, passed = None, None
        if passed is False:
            ok = False
        cells.append(
            {
                "offset": off,
                "policy_rate": rate,
                "demo_cell": demo,
                "required": required,
                "passed": passed,
            }
        )
    return {"cells": cells, "pass": ok}


def per_dim_mse(
    pred: np.ndarray, target: np.ndarray, scale: np.ndarray | None = None
) -> np.ndarray:
    """Mean squared error per action dim over `pred`/`target` of shape (N, A).

    `scale` divides the residual before squaring (pass the corpus per-dim
    std for the normalized table). Never reduce this to one scalar: the left
    arm is alive but off-task and would dominate it (00 §5.2).
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs target {t.shape}")
    if p.ndim != 2:
        raise ValueError(f"expected (N, A), got {p.shape}")
    if p.shape[0] == 0:
        return np.full(p.shape[1], np.nan)
    resid = p - t
    if scale is not None:
        s = np.asarray(scale, dtype=np.float64)
        if s.shape != (p.shape[1],):
            raise ValueError(f"scale must be ({p.shape[1]},), got {s.shape}")
        resid = resid / np.where(s > 0, s, 1.0)
    return (resid**2).mean(axis=0)


def rollout_error_curve(
    pred: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray | None = None,
    dims: slice | None = None,
) -> dict[str, list[float]]:
    """Per-horizon-step error between predicted chunks and the demo's future.

    `pred`/`target` are (N, H, A) — N anchors, H horizon steps, A dims.
    `valid` is (N, H) and false where the demo ran off the end of the
    episode (lerobot's `action_is_pad`). Returns, per horizon step, the mean
    absolute and the root-mean-square error over the selected dims, plus how
    many anchors backed each step.
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs target {t.shape}")
    if p.ndim != 3:
        raise ValueError(f"expected (N, H, A), got {p.shape}")
    if dims is not None:
        p, t = p[:, :, dims], t[:, :, dims]
    if valid is None:
        valid = np.ones(p.shape[:2], dtype=bool)
    v = np.asarray(valid, dtype=bool)
    if v.shape != p.shape[:2]:
        raise ValueError(f"valid must be {p.shape[:2]}, got {v.shape}")
    resid = np.abs(p - t)
    mae, rmse, counts = [], [], []
    for h in range(p.shape[1]):
        m = v[:, h]
        counts.append(int(m.sum()))
        if not m.any():
            mae.append(float("nan"))
            rmse.append(float("nan"))
            continue
        r = resid[m, h, :]
        mae.append(float(r.mean()))
        rmse.append(float(np.sqrt((r**2).mean())))
    return {"mae": mae, "rmse": rmse, "n_anchors": counts}


def derive_pose_tolerance(
    states: list[np.ndarray], close_frames: list[int | None], window: int
) -> dict[str, float]:
    """How far the demonstrations' own right arm travels in +/- `window` frames.

    The tolerance is the 90th percentile of that displacement: "near the demo
    grasp pose" means no further than the demonstration itself moves inside
    the timing tolerance. Derived from the demos, never from a policy, so it
    cannot be tuned by what it rescues.
    """
    dists: list[float] = []
    for state, close in zip(states, close_frames, strict=True):
        if close is None:
            continue
        s = np.asarray(state, dtype=np.float64)
        ref = s[close, RIGHT_ARM_SLICE]
        for off in (-window, window):
            u = int(np.clip(close + off, 0, s.shape[0] - 1))
            dists.append(float(np.linalg.norm(s[u, RIGHT_ARM_SLICE] - ref)))
    if not dists:
        raise ValueError("no demonstration close events to derive a pose tolerance from")
    d = np.asarray(dists)
    return {
        "tolerance_rad": float(np.percentile(d, 90)),
        "median_rad": float(np.median(d)),
        "max_rad": float(d.max()),
        "n_samples": int(d.size),
        "window_frames": int(window),
    }


NULL_MODELS = ("state_hold", "action_hold", "oracle")


def null_model_gripper_chunk(
    demo_action_g: np.ndarray,
    demo_state_g: np.ndarray,
    frame: int,
    horizon: int,
    kind: str,
) -> np.ndarray:
    """The gripper chunk a trivial model would emit at `frame`.

    * `state_hold` — hold the *measured* gripper open fraction
      (`observation.state[14]`) across the horizon. This is the F-99 copycat:
      a policy that has learned nothing but to echo its own proprioception.
    * `action_hold` — hold the current recorded action. A one-step oracle
      with no plan.
    * `oracle` — the demonstration's own future actions. The ceiling.

    These exist because a verdict with no null model is not a measurement.
    """
    if kind == "state_hold":
        return np.full(horizon, float(demo_state_g[frame]))
    if kind == "action_hold":
        return np.full(horizon, float(demo_action_g[frame]))
    if kind == "oracle":
        idx = np.clip(np.arange(frame, frame + horizon), 0, len(demo_action_g) - 1)
        return demo_action_g[idx]
    raise ValueError(f"unknown null model {kind!r}; expected one of {NULL_MODELS}")


def null_model_verdicts(
    demos: dict[int, dict[str, np.ndarray]],
    demo_closes: dict[int, int | None],
    episodes: list[int],
    *,
    threshold: float,
    horizon: int,
    n_executed: int,
    timing_tol_frames: int,
    pose_tol_rad: float,
) -> dict[str, dict[str, Any]]:
    """Score the three trivial models through the probe's own rule.

    Full frame sweep, same episodes, same thresholds, same geometry as the
    policy under test. Costs no GPU: the chunks are constructed from the
    demonstration arrays.
    """
    out: dict[str, dict[str, Any]] = {}
    for kind in NULL_MODELS:
        rows, deltas = [], []
        for e in episodes:
            action_g = demos[e]["action"][:, GRIPPER_ACTION_DIM]
            state_g = demos[e]["state"][:, GRIPPER_ACTION_DIM]
            state = demos[e]["state"]
            n = action_g.shape[0]
            frames = list(range(n))
            events = [
                chunk_close_event(
                    null_model_gripper_chunk(action_g, state_g, t, horizon, kind),
                    threshold,
                    n_executed,
                )
                for t in frames
            ]
            scan = scan_close_events(frames, events)
            t_demo = demo_closes[e]
            t_pred = scan["predicted_close_frame"]
            pose_err = None
            if t_pred is not None and t_demo is not None:
                u = int(np.clip(t_pred, 0, n - 1))
                pose_err = float(
                    np.linalg.norm(state[u, RIGHT_ARM_SLICE] - state[t_demo, RIGHT_ARM_SLICE])
                )
                deltas.append(t_pred - t_demo)
            verdict, _ = classify_verdict(
                n_frames_any_close=scan["n_frames_any_close"],
                n_frames_executed_close=scan["n_frames_executed_close"],
                predicted_close_frame=t_pred,
                demo_close_frame=t_demo,
                pose_err_rad=pose_err,
                timing_tol_frames=timing_tol_frames,
                pose_tol_rad=pose_tol_rad,
            )
            rows.append({"episode": int(e), "verdict": verdict, "timing_error_frames": (
                None if (t_pred is None or t_demo is None) else int(t_pred - t_demo)
            )})
        out[kind] = {
            "verdicts": aggregate_verdicts(rows),
            "median_timing_error_frames": float(np.median(deltas)) if deltas else None,
            "rows": rows,
        }
    return out


def aggregate_verdicts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Verdict counts. The episode is the independent unit — nothing else."""
    counts = dict.fromkeys(VERDICTS, 0)
    for row in rows:
        counts[row["verdict"]] += 1
    return counts


# ---------------------------------------------------------------------------
# Dataset side. pyarrow only (no torch, no lerobot) — the demo arrays the
# verdict is scored against come straight from the parquet.
# ---------------------------------------------------------------------------


def load_demo_arrays(root: Path, episodes: list[int]) -> dict[int, dict[str, np.ndarray]]:
    """Per-episode raw `action` (T, 15) and `observation.state` (T, 27)."""
    import pyarrow.dataset as pads

    files = sorted(Path(root, "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet under {root / 'data'}")
    table = pads.dataset([str(f) for f in files], format="parquet").to_table(
        columns=["episode_index", "frame_index", "action", "observation.state"]
    )
    ep = np.asarray(table.column("episode_index"))
    fr = np.asarray(table.column("frame_index"))
    act = np.stack(table.column("action").to_pylist()).astype(np.float32)
    sta = np.stack(table.column("observation.state").to_pylist()).astype(np.float32)
    out: dict[int, dict[str, np.ndarray]] = {}
    for e in episodes:
        m = ep == e
        if not m.any():
            raise KeyError(f"episode {e} not present in {root}")
        order = np.argsort(fr[m], kind="stable")
        out[e] = {"action": act[m][order], "state": sta[m][order]}
    return out


def read_split(dataset_root: Path, split: str, episodes_file: Path | None) -> list[int]:
    """The frozen held-out list. Never re-derived, never re-seeded (00 §5.1)."""
    path = episodes_file or (dataset_root / "meta" / "splits_camelo.json")
    with open(path) as fh:
        blob = json.load(fh)
    if split == "all":
        return sorted(set(blob["heldout"]) | set(blob["train"]))
    if split not in blob:
        raise KeyError(f"{path} has no '{split}' key (keys: {sorted(blob)})")
    return list(blob[split])


# ---------------------------------------------------------------------------
# Policy side. Every torch / lerobot import lives below this line and inside
# a function (AGENTS.md hard rule 3).
# ---------------------------------------------------------------------------


def resolve_model_dir(checkpoint: Path) -> Path:
    """Accept a checkpoint dir, its `pretrained_model/`, or a run dir."""
    if (checkpoint / "config.json").is_file():
        return checkpoint
    if (checkpoint / "pretrained_model" / "config.json").is_file():
        return checkpoint / "pretrained_model"
    raise FileNotFoundError(
        f"{checkpoint} is not a checkpoint: no config.json and no pretrained_model/config.json"
    )


def _read_train_config(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "train_config.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. The probe reproduces training preprocessing from the run's "
            "own train_config.json and will not guess it."
        )
    with open(path) as fh:
        return json.load(fh)


def _device_override(pipeline_json: Path, device: str) -> dict[str, dict[str, Any]]:
    """`{"device_processor": {"device": ...}}`, but only if that step exists.

    `PolicyProcessorPipeline.from_pretrained` raises on an override key it
    cannot place (its typo guard), and the per-rung pipelines differ — the
    VLA-JEPA checkpoint's postprocessor was rebuilt by hand with two steps
    deleted (13_LAUNCH_LOG.md V-3). So read the saved pipeline and only
    override what is in it.
    """
    if not pipeline_json.is_file():
        return {}
    with open(pipeline_json) as fh:
        blob = json.load(fh)
    names = {step.get("registry_name") for step in blob.get("steps", [])}
    return {"device_processor": {"device": device}} if "device_processor" in names else {}


def build_runtime(
    model_dir: Path,
    dataset_root: Path,
    device: str,
) -> dict[str, Any]:
    """Load the policy, its processors, and the dataset recipe it trained on.

    Everything that decides preprocessing comes from artifacts the run wrote,
    never from a guess:

    * the **policy config** and weights from `config.json` / `model.safetensors`;
    * the **pre/post-processor pipelines** from `policy_preprocessor.json` /
      `policy_postprocessor.json` — which is where the per-rung camera
      `rename_map` (VLA-JEPA's 2-camera `head -> exterior_1_left`,
      `wrist_right -> exterior_2_left`) and the normalizer stats already live,
      so they are reproduced rather than re-specified;
    * `dataset.image_transforms` (Diffusion's 240x320, VLA-JEPA's 224x224,
      GR00T's resize) and `tolerance_s` / `video_backend` /
      `use_imagenet_stats` from `train_config.json`.

    The one thing that is *not* a training artifact is the eval-side dropping
    of image transforms (13_LAUNCH_LOG.md D-2): the probe uses the **train**
    transforms, because they are what the weights saw and what makes the
    heterogeneous camera shapes stackable at all.
    """
    import draccus
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata, resolve_delta_timestamps
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.transforms import ImageTransforms, ImageTransformsConfig
    from lerobot.utils.constants import IMAGENET_STATS

    train_cfg = _read_train_config(model_dir)
    ds_cfg = train_cfg["dataset"]
    repo_id = ds_cfg["repo_id"]
    tolerance_s = float(train_cfg.get("tolerance_s", 1e-4))
    video_backend = ds_cfg.get("video_backend")
    use_imagenet_stats = bool(ds_cfg.get("use_imagenet_stats", True))
    rename_map = dict(train_cfg.get("rename_map") or {})

    tf_blob = ds_cfg.get("image_transforms") or {}
    tf_cfg = draccus.decode(ImageTransformsConfig, tf_blob) if tf_blob else ImageTransformsConfig()
    image_transforms = ImageTransforms(tf_cfg) if tf_cfg.enable else None

    policy_cfg = PreTrainedConfig.from_pretrained(str(model_dir))
    policy_cfg.pretrained_path = model_dir
    policy_cfg.device = device

    meta = LeRobotDatasetMetadata(repo_id, root=str(dataset_root), revision=ds_cfg.get("revision"))
    delta_timestamps = resolve_delta_timestamps(policy_cfg, meta)
    # `make_dataset` overwrites the camera stats with ImageNet's when
    # `use_imagenet_stats` is set; the policy reads `ds_meta.stats`, so the
    # probe has to make the same substitution or normalization differs.
    if use_imagenet_stats:
        for key in meta.camera_keys:
            if key in meta.depth_keys:
                continue
            for stats_type, stats in IMAGENET_STATS.items():
                meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    policy = make_policy(cfg=policy_cfg, ds_meta=meta, rename_map=rename_map or None)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(model_dir),
        preprocessor_overrides=_device_override(model_dir / "policy_preprocessor.json", device),
        postprocessor_overrides=_device_override(model_dir / "policy_postprocessor.json", device),
    )

    action_delta = list(getattr(policy_cfg, "action_delta_indices", None) or [0])
    if 0 not in action_delta:
        raise RuntimeError(
            f"{policy_cfg.type}'s action_delta_indices {action_delta} do not contain 0; "
            "the probe cannot align a predicted chunk with the demo's future actions."
        )
    n_executed = int(getattr(policy_cfg, "n_action_steps", 0) or 0)
    if n_executed <= 0:
        raise RuntimeError(f"{policy_cfg.type} declares no n_action_steps; refusing to guess")

    return {
        "policy": policy,
        "policy_cfg": policy_cfg,
        "preprocessor": preprocessor,
        "postprocessor": postprocessor,
        "make_dataset": lambda eps: _dataset_with_deltas(
            LeRobotDataset,
            repo_id,
            dataset_root,
            eps,
            delta_timestamps,
            image_transforms,
            ds_cfg.get("revision"),
            video_backend,
            tolerance_s,
        ),
        "delta_timestamps": delta_timestamps,
        "action_zero_index": action_delta.index(0),
        "n_executed": n_executed,
        "policy_type": policy_cfg.type,
        "train_config": {
            "repo_id": repo_id,
            "tolerance_s": tolerance_s,
            "video_backend": video_backend,
            "use_imagenet_stats": use_imagenet_stats,
            "rename_map": rename_map,
            "image_transforms": tf_blob,
            "n_action_steps": n_executed,
            "action_delta_indices": action_delta,
            "steps_trained": train_cfg.get("steps"),
        },
        "stats": meta.stats,
        "camera_keys": list(meta.camera_keys),
        "action_names": list(meta.features["action"].get("names") or []),
        "fps": int(meta.fps),
    }


def _dataset_with_deltas(
    cls,
    repo_id: str,
    root: Path,
    episodes: list[int],
    delta_timestamps: dict[str, list[float]] | None,
    image_transforms: Any,
    revision: str | None,
    video_backend: str | None,
    tolerance_s: float,
):
    return cls(
        repo_id,
        root=str(root),
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=image_transforms,
        revision=revision,
        video_backend=video_backend,
        return_uint8=True,
        tolerance_s=tolerance_s,
    )


def predict_episode_chunks(
    runtime: dict[str, Any],
    episode: int,
    frames: list[int],
    *,
    batch_size: int,
    num_workers: int,
) -> dict[str, np.ndarray]:
    """Open-loop chunk prediction at `frames` of one episode.

    Nothing is executed and nothing is fed back: every observation is the
    demonstration's own, which is what makes an ABSENT verdict here
    unambiguously *supervision*-ABSENT (GRASP §0).
    """
    import torch
    from torch.utils.data import DataLoader, Subset

    ds = runtime["make_dataset"]([episode])
    n = len(ds)
    bad = [f for f in frames if f < 0 or f >= n]
    if bad:
        raise IndexError(f"episode {episode} has {n} frames; asked for {bad[:5]}")
    policy = runtime["policy"]
    pre, post = runtime["preprocessor"], runtime["postprocessor"]
    a0 = runtime["action_zero_index"]

    loader = DataLoader(
        Subset(ds, frames),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    if hasattr(policy, "reset"):
        policy.reset()

    def _run_chunk(batch_dict: dict[str, Any]) -> torch.Tensor:
        chunk = policy.predict_action_chunk(batch_dict)
        if not torch.is_tensor(chunk):
            chunk = torch.as_tensor(chunk)
        if chunk.ndim != 3:
            raise RuntimeError(
                f"{runtime['policy_type']}.predict_action_chunk returned shape "
                f"{tuple(chunk.shape)}; the probe expects (B, H, A)"
            )
        return chunk

    def _slice_item(batch_dict: dict[str, Any], i: int, b: int) -> dict[str, Any]:
        # Isolate one sample from a processed batch for retry. Only slices
        # entries whose leading dim matches the batch size; anything else
        # (a scalar, a shared config value) passes through unchanged.
        out: dict[str, Any] = {}
        for k, v in batch_dict.items():
            if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == b:
                out[k] = v[i : i + 1]
            elif isinstance(v, list) and len(v) == b:
                out[k] = v[i : i + 1]
            else:
                out[k] = v
        return out

    pred_chunks: list[np.ndarray] = []
    demo_futures: list[np.ndarray] = []
    demo_valid: list[np.ndarray] = []
    kept_frames: list[int] = []
    decode_failures: list[int] = []
    frame_cursor = 0
    with torch.no_grad():
        for batch in loader:
            raw_action = batch["action"].detach().cpu().numpy()
            raw_pad = (
                batch["action_is_pad"].detach().cpu().numpy()
                if "action_is_pad" in batch
                else np.zeros(raw_action.shape[:2], dtype=bool)
            )
            staged = dict(batch)
            # `lerobot_train.py:603-606` converts uint8 camera tensors to
            # float32 in [0, 1] *before* the preprocessor; the dataset is
            # built with `return_uint8=True` (datasets/factory.py:200) and no
            # processor step does it. Skipping this feeds the normalizer
            # uint8 and it casts the stats to uint8 — a loud RuntimeError
            # here, but a silently different normalization if it were not.
            for cam in runtime["camera_keys"]:
                if cam in staged and staged[cam].dtype == torch.uint8:
                    staged[cam] = staged[cam].to(dtype=torch.float32) / 255.0
            processed = pre(staged)
            batch_frames = frames[frame_cursor : frame_cursor + raw_action.shape[0]]
            frame_cursor += raw_action.shape[0]

            try:
                chunk = _run_chunk(processed)
                ok_idx = list(range(raw_action.shape[0]))
            except AssertionError as exc:
                # pi0_fast decodes actions autoregressively from FAST tokens
                # (`lerobot/policies/pi0_fast/modeling_pi0_fast.py:detokenize_actions`)
                # and asserts the generated sequence starts with
                # ['Action', ':'] before parsing it. On a checkpoint that has
                # not fully converged, decoding can drift into a different
                # part of the vocabulary (observed: raw `<loc####>` /
                # segmentation-style tokens with no 'Action:' prefix at all)
                # and that assert fires — a real per-frame decode failure of
                # this policy, not a probe defect (14_PROBE.md, pi0_fast
                # entry). One bad frame previously killed the whole batch and
                # the whole job (job 3990846). Retry the batch one sample at
                # a time so a single unparseable frame is excluded rather
                # than losing every frame in its batch.
                b = raw_action.shape[0]
                chunks: list[torch.Tensor] = []
                ok_idx = []
                for i in range(b):
                    try:
                        one = _run_chunk(_slice_item(processed, i, b))
                    except AssertionError:
                        decode_failures.append(batch_frames[i])
                        continue
                    chunks.append(one)
                    ok_idx.append(i)
                if not chunks:
                    continue
                chunk = torch.cat(chunks, dim=0)
                del exc

            b, h, a = chunk.shape
            flat = post(chunk.reshape(b * h, a))
            if not torch.is_tensor(flat):
                flat = torch.as_tensor(flat)
            chunk = flat.reshape(b, h, flat.shape[-1]).float().cpu().numpy()
            pred_chunks.append(chunk)
            demo_futures.append(raw_action[ok_idx, a0 : a0 + h])
            demo_valid.append(~raw_pad[ok_idx, a0 : a0 + h])
            kept_frames.extend(batch_frames[i] for i in ok_idx)

    if not pred_chunks:
        raise RuntimeError(
            f"{runtime['policy_type']} failed to decode every one of {len(frames)} probed "
            f"frames in episode {episode}; nothing to score."
        )
    pred = np.concatenate(pred_chunks, axis=0)
    demo = np.concatenate(demo_futures, axis=0)
    valid = np.concatenate(demo_valid, axis=0)
    if pred.shape[-1] != demo.shape[-1]:
        raise RuntimeError(
            f"{runtime['policy_type']} predicts {pred.shape[-1]} action dims but the corpus has "
            f"{demo.shape[-1]}. Refusing to compare — fix the postprocessor, do not slice."
        )
    if demo.shape[1] < pred.shape[1]:
        pred = pred[:, : demo.shape[1]]
    return {
        "pred": pred,
        "demo_future": demo,
        "demo_valid": valid,
        "frames": kept_frames,
        "decode_failures": decode_failures,
    }


# ---------------------------------------------------------------------------
# The probe itself.
# ---------------------------------------------------------------------------


def _frame_plan(
    n_frames: int,
    demo_close: int | None,
    stride: int,
    offsets: tuple[int, ...],
    anchor_period: int,
    max_frames: int | None,
) -> list[int]:
    """Which frames to probe: the sweep, plus every landmark it might miss.

    The sweep is what makes ABSENT and UNTIMED decidable (a close at frame 20
    is only visible if frame 20 is probed). The offset-grid and anchor frames
    are forced in so a `--stride > 1` run still scores the same cells.
    """
    frames = set(range(0, n_frames, max(1, stride)))
    frames.update(range(0, n_frames, max(1, anchor_period)))
    if demo_close is not None:
        for off in offsets:
            f = demo_close + off
            if 0 <= f < n_frames:
                frames.add(f)
    ordered = sorted(frames)
    if max_frames is not None and len(ordered) > max_frames:
        keep = set(ordered[:: max(1, len(ordered) // max_frames)])
        keep.update(range(0, n_frames, max(1, anchor_period)))
        if demo_close is not None:
            keep.update(
                demo_close + off for off in offsets if 0 <= demo_close + off < n_frames
            )
        ordered = sorted(keep)
    return ordered


def probe_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    """Run the whole probe and return the report blob that goes to JSON."""
    import time

    seed = int(getattr(args, "seed", DEFAULT_SEED))
    seed_record = seed_everything(seed)

    model_dir = resolve_model_dir(Path(args.checkpoint).resolve())
    train_cfg = _read_train_config(model_dir)
    dataset_root = Path(
        args.dataset or train_cfg["dataset"]["root"]
    ).resolve()

    if args.episodes:
        episodes = [int(x) for x in args.episodes.split(",")]
    else:
        episodes = read_split(dataset_root, args.split, args.episodes_file)
    if args.limit_episodes:
        episodes = episodes[: args.limit_episodes]

    # The instrument's calibration is derived from the WHOLE frozen held-out
    # split, always — never from whatever subset this invocation probes.
    # Otherwise `--limit-episodes` would move the thresholds and two runs of
    # the same probe would not be comparable (the smoke that surfaced this
    # derived a 0.138 rad pose tolerance from one episode against 0.619 from
    # all 22).
    calib_episodes = read_split(dataset_root, "heldout", args.episodes_file)
    demos = load_demo_arrays(dataset_root, sorted(set(episodes) | set(calib_episodes)))

    # --- thresholds, derived and then checked against the frozen values ----
    all_g = np.concatenate([demos[e]["action"][:, GRIPPER_ACTION_DIM] for e in calib_episodes])
    derived = derive_close_threshold(all_g)
    threshold = (
        float(args.close_threshold) if args.close_threshold is not None else derived["threshold"]
    )
    drift = abs(derived["threshold"] - FROZEN_CLOSE_THRESHOLD)
    if args.close_threshold is None and drift > CLOSE_THRESHOLD_DRIFT_TOL:
        raise RuntimeError(
            f"derived close threshold {derived['threshold']:.4f} drifted {drift:.4f} from the "
            f"frozen {FROZEN_CLOSE_THRESHOLD} (tol {CLOSE_THRESHOLD_DRIFT_TOL}). A moved "
            "threshold silently re-scores every checkpoint ever probed — investigate the corpus, "
            "or pass --close-threshold explicitly and say so in the report."
        )

    runtime = build_runtime(model_dir, dataset_root, args.device)
    fps = runtime["fps"]
    n_executed = runtime["n_executed"]
    timing_tol_frames = int(round(args.timing_tol_s * fps))
    anchor_period = max(1, int(round(args.anchor_period_s * fps)))
    offsets = tuple(int(x) for x in args.offsets.split(","))

    demo_closes = {
        e: first_close_index(demos[e]["action"][:, GRIPPER_ACTION_DIM], threshold)
        for e in demos
    }
    pose_derivation = derive_pose_tolerance(
        [demos[e]["state"] for e in calib_episodes],
        [demo_closes[e] for e in calib_episodes],
        timing_tol_frames,
    )
    pose_drift = abs(pose_derivation["tolerance_rad"] - FROZEN_POSE_TOL_RAD)
    # The pose derivation is a function of the timing tolerance, so the drift
    # alarm only means anything at the default tolerance. Changing
    # --timing-tol-s legitimately moves it and must not trip the alarm.
    default_timing = args.timing_tol_s == DEFAULT_TIMING_TOL_S
    if args.pose_tol_rad is None and default_timing and pose_drift > POSE_TOL_DRIFT_TOL:
        raise RuntimeError(
            f"derived pose tolerance {pose_derivation['tolerance_rad']:.4f} rad drifted "
            f"{pose_drift:.4f} from the frozen {FROZEN_POSE_TOL_RAD} "
            f"(tol {POSE_TOL_DRIFT_TOL}). Investigate the split or the timing tolerance "
            "before re-scoring anything; pass --pose-tol-rad explicitly and say so if this "
            "is deliberate."
        )
    pose_tol = (
        float(args.pose_tol_rad)
        if args.pose_tol_rad is not None
        else pose_derivation["tolerance_rad"]
    )

    action_names = runtime["action_names"]
    stats = runtime["stats"]
    action_std = np.asarray(stats["action"]["std"], dtype=np.float64).ravel()

    rows: list[dict[str, Any]] = []
    mse_num_raw: list[np.ndarray] = []
    rollout_all: list[np.ndarray] = []
    rollout_demo: list[np.ndarray] = []
    rollout_valid: list[np.ndarray] = []

    # Seeded again, HERE. Loading the weights consumes an unknown and
    # rung-dependent amount of the torch RNG stream (backbone init, a
    # processor rebuild, a dtype cast), so a seed set only at process start
    # would still hand each rung a different stream position and two runs
    # that differ only in what got loaded would diverge. Re-seeding on the
    # doorstep of the first forward pass is what makes `--seed 0` mean the
    # same thing twice.
    seed_record = seed_everything(seed)
    t_start = time.time()

    for e in episodes:
        demo_a = demos[e]["action"]
        demo_s = demos[e]["state"]
        n_frames = demo_a.shape[0]
        t_demo = demo_closes[e]
        requested_frames = _frame_plan(
            n_frames, t_demo, args.stride, offsets, anchor_period, args.max_frames_per_episode
        )
        t0 = time.time()
        out = predict_episode_chunks(
            runtime,
            e,
            requested_frames,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        pred, demo_future, valid = out["pred"], out["demo_future"], out["demo_valid"]
        # A policy can fail to decode a subset of the requested frames
        # (pi0_fast's autoregressive FAST-token decode occasionally emits a
        # sequence that does not parse as an action chunk — see
        # `predict_episode_chunks`). Every downstream computation in this
        # loop (`frame_pos`, the offset grid, M3 anchors, the gripper trace)
        # must walk the SURVIVING frames, in the same order as `pred`, or
        # they silently misalign. `decode_failures` is recorded on the row
        # so a report can never hide that frames were dropped.
        frames = out["frames"]
        decode_failures = out["decode_failures"]
        elapsed = time.time() - t0

        horizon = int(pred.shape[1])
        events = [
            chunk_close_event(pred[i, :, GRIPPER_ACTION_DIM], threshold, n_executed)
            for i in range(pred.shape[0])
        ]
        scan = scan_close_events(frames, events)

        t_pred = scan["predicted_close_frame"]
        pose_err = None
        if t_pred is not None and t_demo is not None:
            u = int(np.clip(t_pred, 0, n_frames - 1))
            pose_err = float(
                np.linalg.norm(demo_s[u, RIGHT_ARM_SLICE] - demo_s[t_demo, RIGHT_ARM_SLICE])
            )
        verdict, reason = classify_verdict(
            n_frames_any_close=scan["n_frames_any_close"],
            n_frames_executed_close=scan["n_frames_executed_close"],
            predicted_close_frame=t_pred,
            demo_close_frame=t_demo,
            pose_err_rad=pose_err,
            timing_tol_frames=timing_tol_frames,
            pose_tol_rad=pose_tol,
        )

        # M1 auxiliary: the §5.2 offset grid, scored demo-relative.
        frame_pos = {f: i for i, f in enumerate(frames)}
        policy_rates: dict[int, float | None] = {}
        demo_cells: dict[int, int | None] = {}
        for off in offsets:
            if t_demo is None:
                policy_rates[off] = demo_cells[off] = None
                continue
            f = t_demo + off
            i = frame_pos.get(f)
            policy_rates[off] = (
                None if i is None else float(events[i].executed_step is not None)
            )
            demo_cells[off] = demo_executed_close(
                demo_a[:, GRIPPER_ACTION_DIM], f, threshold, n_executed
            )
        grid = demo_relative_cells(
            offsets=offsets, policy_rates=policy_rates, demo_cells=demo_cells
        )

        # M2: step-0 of the chunk is the action for the probed frame itself.
        step0_pred = pred[:, 0, :]
        step0_demo = demo_future[:, 0, :]
        mse_raw = per_dim_mse(step0_pred, step0_demo)
        n_step0 = float(step0_pred.shape[0])
        # Carry (sum of squares, count) so the corpus-level MSE is the true
        # frame-weighted mean and not a mean of per-episode means.
        mse_num_raw.append(np.stack([mse_raw * n_step0, np.full_like(mse_raw, n_step0)]))

        # M3: anchors every --anchor-period-s.
        anchor_idx = [frame_pos[f] for f in range(0, n_frames, anchor_period) if f in frame_pos]
        if anchor_idx:
            rollout_all.append(pred[anchor_idx])
            rollout_demo.append(demo_future[anchor_idx])
            rollout_valid.append(valid[anchor_idx])
            ep_curve = rollout_error_curve(
                pred[anchor_idx], demo_future[anchor_idx], valid[anchor_idx], RIGHT_ARM_SLICE
            )
        else:
            ep_curve = {"mae": [], "rmse": [], "n_anchors": []}

        trace = None
        if not args.no_gripper_trace:
            trace = [
                {
                    "t": int(f),
                    # 6 dp, not 4: the stored trace must re-derive the stored
                    # verdict exactly. At 4 dp one frame of 11,652 in the
                    # validation run rounded onto the far side of the
                    # threshold (0.6048 vs 0.60483479) and re-derived a
                    # different `any_step`. Harmless there, but per-frame data
                    # that cannot reproduce the conclusion is not an audit
                    # trail.
                    "demo_g": round(float(demo_a[f, GRIPPER_ACTION_DIM]), 6),
                    "pred_g": [round(float(x), 6) for x in pred[i, :, GRIPPER_ACTION_DIM]],
                    "any_step": ev.any_step,
                    "exec_step": ev.executed_step,
                }
                for i, (f, ev) in enumerate(zip(frames, events, strict=True))
            ]

        rows.append(
            {
                "episode": int(e),
                "n_frames": int(n_frames),
                "n_probed_frames": len(frames),
                "n_requested_frames": len(requested_frames),
                "decode_failures": [int(x) for x in decode_failures],
                "demo_close_frame": t_demo,
                "demo_close_s": None if t_demo is None else round(t_demo / fps, 3),
                "demo_starts_closed": bool(
                    demo_a[0, GRIPPER_ACTION_DIM] < threshold
                ),
                "predicted_close_frame": t_pred,
                "timing_error_frames": None
                if (t_pred is None or t_demo is None)
                else int(t_pred - t_demo),
                "timing_error_s": None
                if (t_pred is None or t_demo is None)
                else round((t_pred - t_demo) / fps, 3),
                "pose_err_rad": None if pose_err is None else round(pose_err, 4),
                "verdict": verdict,
                "reason": reason,
                "close_scan": scan,
                "demo_relative_grid": grid,
                "per_dim_mse_raw": [float(x) for x in mse_raw],
                "rollout_right_arm": ep_curve,
                "inference_s": round(elapsed, 2),
                "gripper_trace": trace,
            }
        )
        fail_note = f"  DECODE_FAIL={len(decode_failures)}" if decode_failures else ""
        print(
            f"[probe] ep {e:>4}  {len(frames):>4} frames  {elapsed:6.1f}s  "
            f"demo_close={t_demo}  pred_close={t_pred}  {verdict}{fail_note}",
            flush=True,
        )

    # --- the null models, on the same episodes and the same geometry -------
    # A verdict with no null model is not a measurement: on this corpus the
    # gripper action is nearly recoverable from the gripper STATE, so a pure
    # copycat can score TIMED. Every report carries the bar it has to clear.
    nulls = null_model_verdicts(
        demos,
        demo_closes,
        episodes,
        threshold=threshold,
        horizon=horizon,
        n_executed=n_executed,
        timing_tol_frames=timing_tol_frames,
        pose_tol_rad=pose_tol,
    )

    # --- aggregates, each re-derivable from the rows above -----------------
    counts = aggregate_verdicts(rows)
    num = np.sum([m[0] for m in mse_num_raw], axis=0)
    den = np.sum([m[1] for m in mse_num_raw], axis=0)
    mse_raw_all = num / np.where(den > 0, den, 1.0)
    mse_norm_all = mse_raw_all / np.where(action_std > 0, action_std, 1.0) ** 2

    rollout = {}
    if rollout_all:
        pa = np.concatenate(rollout_all)
        da = np.concatenate(rollout_demo)
        va = np.concatenate(rollout_valid)
        rollout = {
            "right_arm": rollout_error_curve(pa, da, va, RIGHT_ARM_SLICE),
            "left_arm": rollout_error_curve(pa, da, va, LEFT_ARM_SLICE),
            "gripper": rollout_error_curve(
                pa, da, va, slice(GRIPPER_ACTION_DIM, GRIPPER_ACTION_DIM + 1)
            ),
            "all_dims": rollout_error_curve(pa, da, va, None),
        }

    grid_pass = sum(1 for r in rows if r["demo_relative_grid"]["pass"])

    return {
        "schema": "camelo.offline_probe/1",
        "checkpoint": str(model_dir),
        "policy_type": runtime["policy_type"],
        "dataset": str(dataset_root),
        "split": args.split if not args.episodes else "explicit",
        "episodes": [int(e) for e in episodes],
        "fps": fps,
        "anchor_period_s": args.anchor_period_s,
        "stride": args.stride,
        "seed": seed,
        "seed_record": seed_record,
        "wall_s": round(time.time() - t_start, 1),
        "thresholds": {
            "close_threshold": threshold,
            "close_threshold_source": "cli" if args.close_threshold is not None else "derived",
            "close_threshold_derivation": derived,
            "close_threshold_frozen": FROZEN_CLOSE_THRESHOLD,
            "close_threshold_drift": drift,
            "timing_tol_s": args.timing_tol_s,
            "timing_tol_frames": timing_tol_frames,
            "pose_tol_rad": pose_tol,
            "pose_tol_source": "cli" if args.pose_tol_rad is not None else "derived",
            "pose_tol_derivation": pose_derivation,
            "pose_tol_frozen": FROZEN_POSE_TOL_RAD,
            "pose_tol_drift": pose_drift,
            "pose_tol_drift_checked": default_timing,
            "calibration_episodes": [int(e) for e in calib_episodes],
            "n_executed_steps": n_executed,
            "offsets": list(offsets),
        },
        "training_recipe": runtime["train_config"],
        "action_names": action_names,
        "action_std": [float(x) for x in action_std],
        "episode_rows": rows,
        "summary": {
            "verdicts": counts,
            "n_episodes": len(rows),
            "demo_relative_pass": grid_pass,
            "null_models": nulls,
            "per_dim_mse_raw": [float(x) for x in mse_raw_all],
            "per_dim_mse_normalized": [float(x) for x in mse_norm_all],
            "rollout": rollout,
        },
    }


# ---------------------------------------------------------------------------
# Seeding and across-seed aggregation.
#
# Pure logic (torch is imported lazily inside `seed_everything`, exactly like
# the rest of camelo.policy) so all of it is unit-tested with neither torch
# nor lerobot installed — AGENTS.md hard rule 3.
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> dict[str, Any]:
    """Seed every RNG the probe's inference path can draw from.

    Returns a *record* of what was actually seeded, which goes into the
    report header. "The run was seeded" is a claim, not a measurement,
    unless the report says which generators took the seed: torch is imported
    lazily here (a CPU box or a torch-less unit-test env has none), and a
    node with no visible GPU has no CUDA generator to seed.

    Note what this does **not** cover: `numpy.random.default_rng()`
    generators construct their own entropy and ignore the global seed. The
    probe's own numpy use is deterministic arithmetic, so the only RNG that
    reaches a prediction is torch's — but a future caller that reaches for
    `default_rng` inside the inference path must seed it explicitly.
    """
    seed = int(seed)
    if not 0 <= seed < 2**32:
        raise ValueError(f"seed must fit an unsigned 32-bit int, got {seed}")
    record: dict[str, Any] = {
        "seed": seed,
        "python_random": True,
        "numpy_legacy_global": True,
        "torch": False,
        "torch_cuda_devices": 0,
    }
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except Exception:  # pragma: no cover - exercised only where torch exists
        return record
    torch.manual_seed(seed)
    record["torch"] = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        record["torch_cuda_devices"] = int(torch.cuda.device_count())
    return record


def parse_seeds(spec: str) -> list[int]:
    """`"0,1,2"` -> `[0, 1, 2]`. Duplicates are refused, not silently deduped.

    A repeated seed would produce two identical rows and shrink the measured
    spread toward zero, which is exactly the wrong direction for an error
    bar to be wrong in.
    """
    seeds = [int(x) for x in str(spec).replace(" ", "").split(",") if x != ""]
    if not seeds:
        raise ValueError(f"--seeds parsed to nothing from {spec!r}")
    if len(set(seeds)) != len(seeds):
        raise ValueError(
            f"--seeds {spec!r} repeats a seed; a repeated seed contributes an identical "
            "row and understates the across-seed spread"
        )
    for s in seeds:
        if not 0 <= s < 2**32:
            raise ValueError(f"seed must fit an unsigned 32-bit int, got {s}")
    return seeds


def _spread(values: list[float | None]) -> dict[str, Any]:
    """mean / min / max / half-range over the seeds, keeping every value.

    Half-range, not std: at 2-3 seeds a standard deviation is a fiction with
    a decimal point on it, while `mean ± (max-min)/2` is literally the
    interval the runs landed in. The per-seed `values` list stays attached so
    the aggregate can always be re-derived from the rows below it
    (AGENTS.md, "the summary lies, not the measurement").
    """
    vals = [v for v in values if v is not None]
    if not vals:
        return {"values": list(values), "mean": None, "min": None, "max": None,
                "half_range": None}
    lo, hi = float(min(vals)), float(max(vals))
    return {
        "values": [None if v is None else float(v) for v in values],
        "mean": float(np.mean(vals)),
        "min": lo,
        "max": hi,
        "half_range": (hi - lo) / 2.0,
    }


def _median_timing_error_frames(report: dict[str, Any]) -> float | None:
    """Median signed timing error over the episodes that produced one."""
    deltas = [
        r["timing_error_frames"]
        for r in report["episode_rows"]
        if r.get("timing_error_frames") is not None
    ]
    return float(np.median(deltas)) if deltas else None


def _m3_right_arm(report: dict[str, Any]) -> tuple[float | None, float | None]:
    """(step-1, last-step) right-arm MAE from M3's curve, or (None, None)."""
    mae = ((report.get("summary") or {}).get("rollout") or {}).get("right_arm", {}).get("mae")
    if not mae:
        return None, None
    return float(mae[0]), float(mae[-1])


def seed_metrics(report: dict[str, Any]) -> dict[str, Any]:
    """The per-seed headline row, all of it re-derived from `episode_rows`."""
    s = report["summary"]
    step1, last = _m3_right_arm(report)
    return {
        "seed": report.get("seed"),
        "n_episodes": s["n_episodes"],
        "TIMED": s["verdicts"]["TIMED"],
        "MISPLACED": s["verdicts"]["MISPLACED"],
        "UNTIMED": s["verdicts"]["UNTIMED"],
        "ABSENT": s["verdicts"]["ABSENT"],
        "demo_relative_pass": s["demo_relative_pass"],
        "timing_error_median_frames": _median_timing_error_frames(report),
        "m3_right_arm_mae_step1": step1,
        "m3_right_arm_mae_last": last,
        "wall_s": report.get("wall_s"),
    }


#: The metrics the aggregate carries a mean ± half-range for.
AGGREGATE_METRICS = (
    "TIMED",
    "MISPLACED",
    "UNTIMED",
    "ABSENT",
    "demo_relative_pass",
    "timing_error_median_frames",
    "m3_right_arm_mae_step1",
    "m3_right_arm_mae_last",
)


def aggregate_seed_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine one report per seed into a mean ± half-range summary.

    Refuses to aggregate reports that are not the same measurement: a
    different checkpoint, a different episode list or a repeated seed would
    each produce a confident wrong error bar rather than an error.

    The **per-episode verdict agreement** is the part that matters most: an
    aggregate PASS of `12 ± 1` says nothing about whether the same twelve
    episodes passed each time, and §9.9 measured 19 of 22 episodes moving
    their predicted close frame while the headline moved by 2.
    """
    if not reports:
        raise ValueError("cannot aggregate zero reports")
    seeds = [r.get("seed") for r in reports]
    if any(s is None for s in seeds):
        raise ValueError("every report must carry a `seed`; re-run with --seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"reports repeat a seed ({seeds}); that understates the spread")
    ckpts = {r["checkpoint"] for r in reports}
    if len(ckpts) != 1:
        raise ValueError(f"reports are not of one checkpoint: {sorted(ckpts)}")
    eps = [tuple(r["episodes"]) for r in reports]
    if len(set(eps)) != 1:
        raise ValueError("reports do not share an episode list; this is not a comparison")

    per_seed = [seed_metrics(r) for r in reports]
    metrics = {
        key: _spread([m[key] for m in per_seed]) for key in AGGREGATE_METRICS
    }

    episodes = list(eps[0])
    by_ep: list[dict[str, Any]] = []
    for i, e in enumerate(episodes):
        verdicts = [r["episode_rows"][i]["verdict"] for r in reports]
        closes = [r["episode_rows"][i]["predicted_close_frame"] for r in reports]
        known = [c for c in closes if c is not None]
        by_ep.append({
            "episode": int(e),
            "verdicts": dict(zip(seeds, verdicts, strict=True)),
            "verdict_unanimous": len(set(verdicts)) == 1,
            "predicted_close_frame": dict(zip(seeds, closes, strict=True)),
            "close_frame_unanimous": len(set(closes)) == 1,
            "close_frame_spread_frames": (
                int(max(known) - min(known)) if len(known) == len(closes) and known else None
            ),
        })

    n_verdict_agree = sum(1 for r in by_ep if r["verdict_unanimous"])
    n_close_agree = sum(1 for r in by_ep if r["close_frame_unanimous"])
    spreads = [r["close_frame_spread_frames"] for r in by_ep
               if r["close_frame_spread_frames"] is not None]
    n = len(by_ep)
    agreement = {
        "n_episodes": n,
        "n_verdict_unanimous": n_verdict_agree,
        "frac_verdict_unanimous": (n_verdict_agree / n) if n else None,
        "n_close_frame_unanimous": n_close_agree,
        "frac_close_frame_unanimous": (n_close_agree / n) if n else None,
        "max_close_frame_spread_frames": max(spreads) if spreads else None,
        "episodes": by_ep,
        "verdict_disagreements": [r for r in by_ep if not r["verdict_unanimous"]],
    }

    bitwise = (
        n_close_agree == n
        and all(
            m["half_range"] in (0.0, None)
            for m in metrics.values()
        )
    )
    return {
        "schema": "camelo.offline_probe.aggregate/1",
        "checkpoint": reports[0]["checkpoint"],
        "policy_type": reports[0].get("policy_type"),
        "dataset": reports[0].get("dataset"),
        "split": reports[0].get("split"),
        "fps": reports[0].get("fps"),
        "stride": reports[0].get("stride"),
        "seeds": list(seeds),
        "n_seeds": len(seeds),
        "n_episodes": n,
        "per_seed": per_seed,
        "metrics": metrics,
        "episode_agreement": agreement,
        "identical_across_seeds": bool(bitwise),
    }


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------


def _fmt(x: Any, spec: str = "") -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return "—"
    return format(x, spec) if spec else str(x)


def render_markdown(report: dict[str, Any]) -> str:
    """Per-episode table first, aggregates after. Never the other way round."""
    th = report["thresholds"]
    s = report["summary"]
    lines: list[str] = []
    a = lines.append
    a(f"# Offline probe — `{Path(report['checkpoint']).parent.name}` ({report['policy_type']})")
    a("")
    a(f"- checkpoint: `{report['checkpoint']}`")
    a(f"- dataset: `{report['dataset']}` · split `{report['split']}` · "
      f"{len(report['episodes'])} episodes · {report['fps']} Hz")
    a(f"- executed window: first **{th['n_executed_steps']}** chunk steps "
      f"(`n_action_steps`) · wall {report['wall_s']} s")
    sr = report.get("seed_record") or {}
    if report.get("seed") is not None:
        cuda = sr.get("torch_cuda_devices") or 0
        a(f"- **seed {report['seed']}** — python/numpy"
          f"{'/torch' if sr.get('torch') else ' (no torch)'}"
          f"{f'/cuda×{cuda}' if cuda else ''} re-seeded immediately before the first "
          "forward pass. Every rung but ACT has a stochastic action head: this number "
          "is one seeded draw, not the checkpoint's true score. Read it beside the "
          "across-seed aggregate (14_PROBE.md §10), never alone.")
    else:
        a("- ⚠ **unseeded run** (pre-`--seed` report): for a stochastic head this is a "
          "single unreproducible draw — 14_PROBE.md §9.9.")
    a(f"- close threshold **{th['close_threshold']:.4f}** ({th['close_threshold_source']}; "
      f"demo modes {th['close_threshold_derivation']['centre_closed']:.4f} / "
      f"{th['close_threshold_derivation']['centre_open']:.4f}; frozen "
      f"{th['close_threshold_frozen']}, drift {th['close_threshold_drift']:.4f})")
    a(f"- timing tolerance **±{th['timing_tol_s']} s** (±{th['timing_tol_frames']} frames) · "
      f"pose tolerance **{th['pose_tol_rad']:.3f} rad** ({th['pose_tol_source']}; "
      f"p90 of the demos' own ±{th['timing_tol_frames']}-frame right-arm travel; "
      f"frozen {th['pose_tol_frozen']}, drift {th['pose_tol_drift']:.4f})")
    a(f"- both thresholds are calibrated on the **whole** frozen held-out split "
      f"({len(th['calibration_episodes'])} episodes), not on whatever subset this run probed")
    a("")
    a("## M1 — gripper-close timing verdict (per episode; the episode is the unit)")
    a("")
    a("| ep | frames | probed | demo close | pred close | Δt (s) | pose err (rad) | "
      "any-close % | exec-close % | min pred g | demo-rel | verdict |")
    a("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|:--|")
    for r in report["episode_rows"]:
        sc = r["close_scan"]
        a(
            f"| {r['episode']} | {r['n_frames']} | {r['n_probed_frames']} "
            f"| {_fmt(r['demo_close_frame'])} | {_fmt(r['predicted_close_frame'])} "
            f"| {_fmt(r['timing_error_s'], '+.2f')} | {_fmt(r['pose_err_rad'], '.3f')} "
            f"| {100 * sc['frac_frames_any_close']:.1f} "
            f"| {100 * sc['frac_frames_executed_close']:.1f} "
            f"| {_fmt(sc['min_predicted_gripper'], '.3f')} "
            f"| {'PASS' if r['demo_relative_grid']['pass'] else 'fail'} "
            f"| **{r['verdict']}** |"
        )
    a("")
    degenerate = [r["episode"] for r in report["episode_rows"] if r["demo_starts_closed"]]
    if degenerate:
        a(f"> Episodes {degenerate} begin with the gripper already commanded closed "
          "(`demo_close_frame = 0`): there is no open→close transition to time against, "
          "so their timing cell is degenerate. They are kept in the table and counted, "
          "and flagged here rather than dropped.")
        a("")
    decode_fail_eps = {
        r["episode"]: len(r["decode_failures"])
        for r in report["episode_rows"]
        if r.get("decode_failures")
    }
    if decode_fail_eps:
        n_total = sum(decode_fail_eps.values())
        a(f"> ⚠ **{n_total} probed frame(s) across {len(decode_fail_eps)} episode(s) "
          f"failed to decode and were EXCLUDED, not treated as no-close:** "
          f"{decode_fail_eps}. This policy's action head raised while parsing its own "
          "generated tokens into an action chunk (14_PROBE.md documents which rung and "
          "why); the frame is dropped from every metric — M1's frame sweep, M2, M3 — "
          "rather than counted as a non-close, which would understate `any-close %`. "
          "`n_probed_frames` in this table is the count that SURVIVED; the row's "
          "`decode_failures` list in the JSON names the dropped frames.")
        a("")
    a("### Verdict counts")
    a("")
    a("| verdict | episodes |")
    a("|:--|---:|")
    for v in VERDICTS:
        a(f"| {v} | {s['verdicts'][v]} |")
    a(f"| **total** | **{s['n_episodes']}** |")
    a("")
    a(f"Demo-relative PASS (GRASP §0 cell-for-cell, near ≥ {NEAR_RATE_MIN} where the demo "
      f"closes, far ≤ {FAR_RATE_MAX}): **{s['demo_relative_pass']} / {s['n_episodes']}** episodes.")
    a("")
    a("### The bar: three null models, same episodes, same geometry")
    a("")
    a("| model | TIMED | MISPLACED | UNTIMED | ABSENT | median Δt (frames) |")
    a("|:--|---:|---:|---:|---:|---:|")
    nm = s.get("null_models") or {}
    labels = {
        "state_hold": "state-hold copier (`chunk[k] = state[14]`)",
        "action_hold": "action-hold (`chunk[k] = action[t,14]`)",
        "oracle": "oracle (`chunk[k] = action[t+k,14]`)",
    }
    for kind, label in labels.items():
        if kind not in nm:
            continue
        v = nm[kind]["verdicts"]
        a(f"| {label} | {v['TIMED']} | {v['MISPLACED']} | {v['UNTIMED']} | {v['ABSENT']} "
          f"| {_fmt(nm[kind]['median_timing_error_frames'], '+.1f')} |")
    a(f"| **this checkpoint** | **{s['verdicts']['TIMED']}** | {s['verdicts']['MISPLACED']} "
      f"| {s['verdicts']['UNTIMED']} | {s['verdicts']['ABSENT']} | see the table above |")
    a("")
    a("⚠ **Read the verdict counts against the state-hold row, never against 0.** On this "
      "corpus the commanded gripper action is largely recoverable from the measured gripper "
      "*state* the policy is handed, so a pure copycat (F-99's diagnosis) can score TIMED "
      "without having learned anything. A TIMED count at or below the state-hold row is "
      "**not evidence of a learned close**; what discriminates is the demo-relative grid, "
      "the timing-error distribution, and M3.")
    a("")
    a("### Demo-relative offset grid (executed-window close rate, episode mean)")
    a("")
    offs = th["offsets"]
    a("| offset (frames) | " + " | ".join(str(o) for o in offs) + " |")
    a("|:--|" + "---:|" * len(offs))
    for label, key in (("policy", "policy_rate"), ("demo", "demo_cell")):
        vals = []
        for i in range(len(offs)):
            xs = [
                r["demo_relative_grid"]["cells"][i][key]
                for r in report["episode_rows"]
                if r["demo_relative_grid"]["cells"][i][key] is not None
            ]
            vals.append(f"{np.mean(xs):.3f}" if xs else "—")
        a(f"| {label} | " + " | ".join(vals) + " |")
    a("")
    a("## M2 — per-dim action MSE on held-out frames (tiebreak, never the ranking)")
    a("")
    a("| # | dim | raw MSE | normalized MSE |")
    a("|---:|:--|---:|---:|")
    for i, name in enumerate(report["action_names"]):
        a(f"| {i} | `{name}` | {s['per_dim_mse_raw'][i]:.6g} | "
          f"{s['per_dim_mse_normalized'][i]:.6g} |")
    a("")
    a("Raw is in the dim's own units (rad² for joints, fraction² for the gripper); "
      "normalized divides the residual by the corpus per-dim std first. The right arm "
      "(dims 7–13) and the gripper (dim 14) are the ones that matter — the left arm is "
      "alive but off-task (00 §5.2).")
    a("")
    a("## M3 — open-loop chunk rollout error (per horizon step)")
    a("")
    roll = s.get("rollout") or {}
    if roll and roll["right_arm"]["mae"]:
        h = len(roll["right_arm"]["mae"])
        a(f"Anchors every {report['anchor_period_s']} s; step 1 is backed by "
          f"{roll['right_arm']['n_anchors'][0]} anchor frames across "
          f"{report['summary']['n_episodes']} episodes; horizon {h} steps.")
        a("")
        a("| step | t (s) | right-arm MAE (rad) | right-arm RMSE | gripper MAE "
          "| all-dims MAE | n |")
        a("|---:|---:|---:|---:|---:|---:|---:|")
        for k in range(h):
            a(f"| {k + 1} | {(k + 1) / report['fps']:.2f} "
              f"| {_fmt(roll['right_arm']['mae'][k], '.4f')} "
              f"| {_fmt(roll['right_arm']['rmse'][k], '.4f')} "
              f"| {_fmt(roll['gripper']['mae'][k], '.4f')} "
              f"| {_fmt(roll['all_dims']['mae'][k], '.4f')} "
              f"| {roll['right_arm']['n_anchors'][k]} |")
    else:
        a("_no anchors probed_")
    a("")
    a("## What this probe cannot do")
    a("")
    a("It feeds fresh, in-distribution frames and asks about the gripper channel. It "
      "**never** asks whether the policy can navigate to the object (00 §5.3, GRASP §0). "
      "Read every TIMED as *\"the gripper channel is timed\"*, never as "
      "*\"this policy works\"*. Every ABSENT measured here is supervision-ABSENT, because "
      "the observations are the demonstration's own.")
    a("")
    a("Full per-frame data — including the predicted gripper vector of every chunk — is in "
      "the JSON beside this file. A headline that cannot be re-derived from those rows is a "
      "claim, not a measurement.")
    return "\n".join(lines) + "\n"


def render_aggregate_markdown(agg: dict[str, Any]) -> str:
    """The across-seed report: per-seed rows first, the mean ± range after.

    Same rule as `render_markdown` — the rows that the aggregate is computed
    from are printed above it, so every headline can be re-derived by hand.
    """
    lines: list[str] = []
    a = lines.append
    seeds = agg["seeds"]
    a(f"# Offline probe, across seeds — `{Path(agg['checkpoint']).parent.name}` "
      f"({agg['policy_type']})")
    a("")
    a(f"- checkpoint: `{agg['checkpoint']}`")
    a(f"- dataset: `{agg['dataset']}` · split `{agg['split']}` · "
      f"{agg['n_episodes']} episodes · stride {agg['stride']}")
    a(f"- seeds: **{', '.join(str(x) for x in seeds)}** ({agg['n_seeds']} runs of the "
      "identical checkpoint on the identical frames — only the RNG seed differs)")
    a("")
    a("## Per-seed rows (the aggregate below is computed from these)")
    a("")
    a("| seed | TIMED | MISPLACED | UNTIMED | ABSENT | demo-rel PASS | "
      "timing median (frames) | M3 right-arm MAE step 1 | last | wall (s) |")
    a("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in agg["per_seed"]:
        a(f"| {m['seed']} | {m['TIMED']} | {m['MISPLACED']} | {m['UNTIMED']} "
          f"| {m['ABSENT']} | {m['demo_relative_pass']} / {m['n_episodes']} "
          f"| {_fmt(m['timing_error_median_frames'], '+.1f')} "
          f"| {_fmt(m['m3_right_arm_mae_step1'], '.4f')} "
          f"| {_fmt(m['m3_right_arm_mae_last'], '.4f')} "
          f"| {_fmt(m['wall_s'])} |")
    a("")
    a("## Mean ± half-range across seeds")
    a("")
    a("| metric | mean | range | half-range |")
    a("|:--|---:|---:|---:|")
    pretty = {
        "TIMED": "TIMED episodes",
        "MISPLACED": "MISPLACED episodes",
        "UNTIMED": "UNTIMED episodes",
        "ABSENT": "ABSENT episodes",
        "demo_relative_pass": "demo-relative PASS",
        "timing_error_median_frames": "timing error, median (frames)",
        "m3_right_arm_mae_step1": "M3 right-arm MAE, step 1 (rad)",
        "m3_right_arm_mae_last": "M3 right-arm MAE, last step (rad)",
    }
    for key in AGGREGATE_METRICS:
        m = agg["metrics"][key]
        spec = ".4f" if key.startswith("m3_") else ".2f"
        a(f"| {pretty[key]} | {_fmt(m['mean'], spec)} "
          f"| [{_fmt(m['min'], spec)}, {_fmt(m['max'], spec)}] "
          f"| ±{_fmt(m['half_range'], spec)} |")
    a("")
    a("Half-range, not standard deviation: at this many seeds a std is a fiction with a "
      "decimal point on it, while `mean ± (max−min)/2` is literally the interval the runs "
      "landed in. It is a **lower bound** on the seed noise — two or three draws cannot "
      "find the tails.")
    a("")
    a("## Per-episode agreement across seeds")
    a("")
    ag = agg["episode_agreement"]
    a(f"- **verdict** identical across all seeds in **{ag['n_verdict_unanimous']} / "
      f"{ag['n_episodes']}** episodes "
      f"({_fmt(100 * (ag['frac_verdict_unanimous'] or 0), '.1f')} %)")
    a(f"- **predicted close frame** identical in **{ag['n_close_frame_unanimous']} / "
      f"{ag['n_episodes']}** episodes; largest spread "
      f"{_fmt(ag['max_close_frame_spread_frames'])} frames")
    a("")
    if agg["identical_across_seeds"]:
        a("✅ **Bit-identical across every seed** — every episode agreed on both verdict "
          "and predicted close frame, and every aggregate metric has a zero range. This "
          "head is deterministic at inference; a single run of it is the answer, not a draw.")
    else:
        a("⚠ **Not deterministic.** The headline moved with nothing but the seed. Quote "
          "this checkpoint as a range, never as a point estimate, and pin the seed on the "
          "rig (15_RIG_WINDOW_RUNBOOK.md §1/§2).")
    a("")
    if ag["verdict_disagreements"]:
        a("### Episodes whose verdict changed with the seed")
        a("")
        a("| ep | " + " | ".join(f"seed {x}" for x in seeds) + " | close frame per seed |")
        a("|---:|" + "|".join([":--"] * len(seeds)) + "|:--|")
        for r in ag["verdict_disagreements"]:
            vs = " | ".join(str(r["verdicts"][x]) for x in seeds)
            cf = ", ".join(f"{x}:{_fmt(r['predicted_close_frame'][x])}" for x in seeds)
            a(f"| {r['episode']} | {vs} | {cf} |")
        a("")
        a("These are the episodes where the ±1 s timing tolerance happens to sit between "
          "two draws. They are the whole of the headline's seed noise — an aggregate PASS "
          "of `12 ± 1` says nothing about *which* twelve passed, and this table does.")
        a("")
    a("Full per-seed data — every episode row, every predicted gripper vector — is in the "
      "per-seed JSON files beside this one, plus the machine-readable aggregate in "
      "`*_aggregate.json`. A headline that cannot be re-derived from those rows is a "
      "claim, not a measurement.")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="camelo.eval.offline_probe",
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True, type=Path,
                   help="checkpoint dir, its pretrained_model/, or a run's checkpoints/<step>")
    p.add_argument("--dataset", type=Path, default=None,
                   help="corpus root (default: the root the checkpoint's train_config.json names)")
    p.add_argument("--split", default="heldout", choices=("heldout", "train", "all"),
                   help="which frozen split to probe (default: heldout)")
    p.add_argument("--episodes-file", type=Path, default=None,
                   help="splits JSON (default: <dataset>/meta/splits_camelo.json)")
    p.add_argument("--episodes", default=None, help="explicit comma-separated episode ids")
    p.add_argument("--limit-episodes", type=int, default=None, help="smoke: probe only the first N")
    p.add_argument("--out", required=True, type=Path,
                   help="output path; <out>.json and <out>.md are written")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--stride", type=int, default=1,
                   help="probe every Nth frame (default 1; offset-grid and anchor frames are "
                        "always probed regardless)")
    p.add_argument("--max-frames-per-episode", type=int, default=None)
    p.add_argument("--anchor-period-s", type=float, default=2.0, help="M3 anchor spacing")
    p.add_argument("--timing-tol-s", type=float, default=DEFAULT_TIMING_TOL_S)
    p.add_argument("--close-threshold", type=float, default=None,
                   help="override the derived close threshold (say so in the report if you do)")
    p.add_argument("--pose-tol-rad", type=float, default=None,
                   help="override the derived right-arm pose tolerance")
    p.add_argument("--offsets", default=",".join(str(o) for o in DEFAULT_OFFSETS))
    p.add_argument("--no-gripper-trace", action="store_true",
                   help="omit the per-frame predicted gripper vectors from the JSON")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"RNG seed for python/numpy/torch(+cuda), re-applied immediately "
                        f"before the first forward pass (default {DEFAULT_SEED}). Every "
                        f"rung but ACT samples at inference, so an unseeded run is one "
                        f"unreproducible draw (14_PROBE.md §9.9)")
    p.add_argument("--seeds", default=None,
                   help="comma-separated seeds; runs the WHOLE probe once per seed and "
                        "writes <out>_seed<N>.{json,md} per seed plus <out>_aggregate.{md,json} "
                        "with mean ± half-range and the per-episode verdict agreement. "
                        "Overrides --seed")
    return p


def _write_report(report: dict[str, Any], out: Path) -> Path:
    """Write `<out>.json` + `<out>.md`; returns the JSON path."""
    json_path = out if out.suffix == ".json" else out.with_suffix(".json")
    md_path = json_path.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as fh:
        json.dump(report, fh, indent=1)
    with open(md_path, "w") as fh:
        fh.write(render_markdown(report))
    counts = report["summary"]["verdicts"]
    print(f"[probe] wrote {json_path} and {md_path}")
    print("[probe] " + "  ".join(f"{v}={counts[v]}" for v in VERDICTS))
    return json_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    stem = out.with_suffix("") if out.suffix == ".json" else out

    if not args.seeds:
        _write_report(probe_checkpoint(args), out)
        return 0

    # --seeds: the whole probe, once per seed, in ONE process. Same caveat as
    # slurm/offline_probe.slurm's "one process per checkpoint" — the rungs
    # that patch lerobot at load time are patched once here and every seed
    # sees the identical runtime, which is what makes the seeds the only
    # difference between the runs.
    seeds = parse_seeds(args.seeds)
    reports = []
    for i, s in enumerate(seeds, start=1):
        print(f"[probe] === seed {s} ({i}/{len(seeds)}) ===", flush=True)
        args.seed = s
        report = probe_checkpoint(args)
        _write_report(report, Path(f"{stem}_seed{s}"))
        reports.append(report)

    agg = aggregate_seed_reports(reports)
    agg_json = Path(f"{stem}_aggregate.json")
    agg_md = Path(f"{stem}_aggregate.md")
    with open(agg_json, "w") as fh:
        json.dump(agg, fh, indent=1)
    with open(agg_md, "w") as fh:
        fh.write(render_aggregate_markdown(agg))
    m = agg["metrics"]["demo_relative_pass"]
    print(f"[probe] wrote {agg_json} and {agg_md}")
    print(f"[probe] demo-relative PASS {m['mean']:.2f} ± {m['half_range']:.2f} "
          f"over seeds {seeds}; verdict unanimous on "
          f"{agg['episode_agreement']['n_verdict_unanimous']}/{agg['n_episodes']} episodes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
