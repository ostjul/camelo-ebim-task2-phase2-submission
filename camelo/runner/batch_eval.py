"""Batch evaluation: N scored episodes -> results.csv + summary.json."""

from __future__ import annotations

import csv
import json
import logging
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from camelo.runner.recenter import StartPoseNotReached

if TYPE_CHECKING:  # episode_runner imports ROS; keep this module offline-importable
    from camelo.runner.episode_runner import EpisodeRunner

log = logging.getLogger(__name__)


def summarize_rows(rows: list[dict]) -> dict:
    """Aggregate episode rows into the summary block.

    F-44 decision: `orientation_case == "no_target_bbox"` means the eval
    camera lost the target between resets — the episode measures the eval
    stack, not the policy. Such episodes are EXCLUDED from mean_iou and the
    orientation count and reported separately, so a flaky eval camera can
    never inflate or deflate a baseline number silently.

    The `clamp_*` / `stale_*` block promotes the executor counters
    `run_rollout` already returns into the summary, because a low IoU has
    two very different causes that the score alone cannot separate: the
    policy asked for the wrong pose, or the executor could not follow the
    pose it asked for. A high `clamp_fraction` means the 0.05 rad/tick
    limiter (the ONLY rate limiter in the command path — F-53) was
    saturated, so the arm lagged its own target; `stale_chunks` means the
    chunk ran out before the replan landed. Read both BEFORE concluding
    anything about the checkpoint.

    **Never read a clamp fraction without `mean_ticks` beside it** (F-88):
    `run_rollout` calls `backend.infer()` synchronously inside the control
    loop, so a slow policy publishes fewer commands per sim second and
    each one covers more ground — the clamp then saturates on latency
    rather than on anything the policy asked for. The same pi0.5
    checkpoint measured 90 % clamped at 632 ticks/episode on a degraded
    rig and 52-66 % at ~2620 ticks/episode after a sim restart, with no
    change to the policy at all. Clamp fraction is therefore partly a
    harness metric; `mean_control_hz_sim` is what makes two runs comparable.
    The unit is ticks per SIM second and the name says so on purpose: the
    wall-time figure is ~6.5x smaller and swapping them inverts every
    conclusion drawn from it (F-91).

    The grasp/pad block answers the protocol's §0 sim intermediate metrics
    1 and 2, and exists because scored IoU is floored at 0.0 (`c18c4b9`)
    and therefore ranks nothing: the `graspgate_n2` run reported
    `mean_iou: 0.0` for both episodes while its rows carried the gate's
    distances the whole time — the measurement was there, the summary threw
    it away (AGENTS.md, "the summary lies, not the measurement"). Three
    rules keep the block honest:

    * **A quantity that was never measured reads `None`, never `0.0`.**
      `mean_pad_disp_max_mm == 0.0` means "watched, and the pad did not
      move"; `None` means nobody watched. Reporting the second as the first
      is the same class of error as the floored IoU.
    * **Counts are counts of EPISODES.** The `_episodes` suffix is load
      bearing: the same quantities exist per tick in the rows
      (`loop_grasp_env_ticks_inside`), and a count whose denominator is
      unstated is not re-derivable.
    * **Instrument settings are carried up, never averaged.** Two envelopes
      live in this repo — 40 mm in `outputs/probes/grasp_pose_envelope.json`
      and 45 mm in `grasp_gate.DEFAULT_MAX_DIST_M` — and their mean, 42.5 mm,
      is a threshold no episode ever used. Disagreement reports `None` and
      sets `grasp_env_settings_consistent: False`.

    `pad_lift_max_mm` is retired as evidence (GRASP_EXPERIMENT_PROTOCOL.md
    §0; pad_metrics.py module header): a shoved pad's centroid rises the
    same way a carried one's does, so a peak alone cannot separate a grasp
    from a drag. The `pad_lift_sustained_*` / `pad_lifted_*` keys below
    replace it with height the pad HELD for a full `pad_lift_window_s`;
    `mean_pad_lift_max_mm` stays in this dict only so old and new
    summary.json files share columns, never to be quoted as a result.

    Means skip absent/None entries, and `grasp_env_observed_episodes` /
    `pad_observed_episodes` publish the denominators they were taken over,
    so a partially observed run cannot average over a smaller sample
    invisibly. Every field is re-derivable from results.csv.
    """
    degraded = [r for r in rows if r.get("orientation_case") == "no_target_bbox"]
    valid = [r for r in rows if r.get("orientation_case") != "no_target_bbox"]
    scored = [r["iou"] for r in valid if isinstance(r.get("iou"), (int, float))]

    # Rollout timing is a harness metric, so degraded episodes still count.
    def _values(key: str) -> list[float]:
        return [r[key] for r in rows if isinstance(r.get(key), (int, float))]

    def _mean(key: str) -> float | None:
        values = _values(key)
        return sum(values) / len(values) if values else None

    def _max(key: str) -> float | None:
        values = _values(key)
        return max(values) if values else None

    # Per episode first, then averaged: a single long episode must not
    # dominate the fraction the way a pooled clamped/ticks ratio would.
    fractions = [
        r["loop_clamped_ticks"] / r["loop_ticks"]
        for r in rows
        if isinstance(r.get("loop_clamped_ticks"), (int, float))
        and isinstance(r.get("loop_ticks"), (int, float))
        and r["loop_ticks"]
    ]

    # Commands per SIM second — the rate the robot actually experiences.
    # The loop paces itself on wall time but the rollout is bounded by sim
    # time, and the sim runs 6.5-9x slower than wall under load.
    rates = [
        r["loop_ticks"] / r["loop_sim_seconds"]
        for r in rows
        if isinstance(r.get("loop_ticks"), (int, float))
        and isinstance(r.get("loop_sim_seconds"), (int, float))
        and r["loop_sim_seconds"]
    ]

    # --- §0 sim intermediate metrics: grasp attempt, then pad motion ----
    # These keys arrive with the `loop_` prefix from GraspObserver /
    # PadTracker and are ABSENT from every run recorded before P1, so
    # absent, None and 0.0 must stay three different statements.
    def _number(value: object) -> float | None:
        """A measured number, or None.

        Bools are not measurements (and `isinstance(True, int)` is True), a
        CSV round-trip turns every cell into a string, and a NaN sentinel —
        `GraspGate` stores `float("nan")` for its unlatched readings — would
        silently poison a mean rather than being skipped like the None it
        stands for.
        """
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                return None
        if isinstance(value, (int, float)):
            return float(value) if math.isfinite(value) else None
        return None

    def _flag(value: object) -> bool | None:
        """True/False from a real bool or from the string results.csv keeps.

        `outputs/eval/graspgate_n2/results.csv` stores `False` as the four
        characters "False", which is truthy: a summary that read the rows
        back from disk would count every episode as a grasp.
        """
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            text = value.strip().lower()
            if text in ("true", "1"):
                return True
            if text in ("false", "0"):
                return False
        return None

    def _episodes_where(key: str) -> int:
        """Counts EPISODES, not ticks — which is what the field names say."""
        return sum(1 for r in rows if _flag(r.get(key)) is True)

    def _numbers(key: str, source: list[dict]) -> list[float]:
        return [v for v in (_number(r.get(key)) for r in source) if v is not None]

    def _mean_of(key: str, source: list[dict]) -> float | None:
        values = _numbers(key, source)
        return sum(values) / len(values) if values else None

    def _max_of(key: str, source: list[dict]) -> float | None:
        values = _numbers(key, source)
        return max(values) if values else None

    def _min_of(key: str, source: list[dict]) -> float | None:
        values = _numbers(key, source)
        return min(values) if values else None

    # An episode whose pad pose never arrived measured nothing: its 0.0 mm
    # is not "the pad did not move", and averaging it in would pull every
    # displacement toward zero exactly like the floored IoU did.
    def _pad_observed(row: dict) -> bool:
        seen = _flag(row.get("loop_pad_baseline_seen"))
        if seen is not None:
            return seen
        return bool(_number(row.get("loop_pad_observed_ticks")))

    pad_rows = [r for r in rows if _pad_observed(r)]
    # Likewise for the envelope: with no pad pose the observer records no
    # tick, so `ticks_inside == 0` there is an absence, not a zero.
    env_rows = [r for r in rows if (_number(r.get("loop_grasp_env_observed_ticks")) or 0) > 0]

    def _setting(key: str) -> float | None:
        values = set(_numbers(key, rows))
        return values.pop() if len(values) == 1 else None

    settings_consistent = all(
        len(set(_numbers(key, rows))) <= 1
        for key in ("loop_grasp_env_max_dist_m", "loop_grasp_env_max_angle_deg")
    )
    pad_lift_settings_consistent = all(
        len(set(_numbers(key, rows))) <= 1
        for key in ("loop_pad_lift_window_s", "loop_pad_lift_eps_mm")
    )

    return {
        "episodes": len(rows),
        "scored": len(scored),
        "mean_iou": sum(scored) / len(scored) if scored else None,
        "orientation_correct": len([r for r in valid if r.get("orientation_correct")]),
        "eval_degraded_no_target_bbox": len(degraded),
        "mean_rollout_wall_s": _mean("loop_wall_seconds"),
        "mean_rollout_sim_s": _mean("loop_sim_seconds"),
        "mean_ticks": _mean("loop_ticks"),
        "mean_control_hz_sim": sum(rates) / len(rates) if rates else None,
        "mean_clamp_fraction": sum(fractions) / len(fractions) if fractions else None,
        "max_clamp_fraction": max(fractions) if fractions else None,
        "max_requested_delta": _max("loop_max_requested_delta"),
        "mean_stale_chunks": _mean("loop_stale_chunks"),
        "mean_inferences": _mean("loop_inferences"),
        "mean_infer_s": _mean("loop_infer_mean_s"),
        # §0 metric 1: a grasp attempt at the right place.
        "grasp_env_observed_episodes": len(env_rows),
        "grasp_env_entered_episodes": _episodes_where("loop_grasp_env_entered"),
        "grasp_close_in_env_episodes": _episodes_where("loop_grasp_close_in_env"),
        "grasp_close_any_episodes": _episodes_where("loop_grasp_close_any"),
        "min_grasp_env_min_dist_m": _min_of("loop_grasp_env_min_dist_m", env_rows),
        "mean_grasp_env_min_dist_m": _mean_of("loop_grasp_env_min_dist_m", env_rows),
        "mean_grasp_env_ticks_inside": _mean_of("loop_grasp_env_ticks_inside", env_rows),
        # §0 metric 2: did it move the thing.
        "mean_pad_disp_max_mm": _mean_of("loop_pad_disp_max_mm", pad_rows),
        "max_pad_disp_max_mm": _max_of("loop_pad_disp_max_mm", pad_rows),
        "mean_pad_disp_final_mm": _mean_of("loop_pad_disp_final_mm", pad_rows),
        "mean_pad_lift_max_mm": _mean_of("loop_pad_lift_max_mm", pad_rows),
        "mean_pad_disp_toward_target_mm": _mean_of("loop_pad_disp_toward_target_mm", pad_rows),
        "pad_observed_episodes": len(pad_rows),
        # P4: sustained height above the liner plane, replacing the retired
        # pad_lift_max_mm peak (docstring above). measured_episodes is the
        # denominator for lifted_episodes: a pad seen for less than
        # pad_lift_window_s has no verdict, so it must not count as "not
        # lifted".
        "mean_pad_lift_sustained_mm": _mean_of("loop_pad_lift_sustained_mm", pad_rows),
        "max_pad_lift_sustained_mm": _max_of("loop_pad_lift_sustained_mm", pad_rows),
        "pad_lifted_episodes": _episodes_where("loop_pad_lifted"),
        "pad_lift_measured_episodes": len(_numbers("loop_pad_lift_sustained_mm", pad_rows)),
        # The instrument, carried up so a distance can be read against the
        # envelope it was judged with. Never averaged: see the docstring.
        "grasp_env_max_dist_m": _setting("loop_grasp_env_max_dist_m"),
        "grasp_env_max_angle_deg": _setting("loop_grasp_env_max_angle_deg"),
        "grasp_env_settings_consistent": settings_consistent,
        "pad_lift_window_s": _setting("loop_pad_lift_window_s"),
        "pad_lift_eps_mm": _setting("loop_pad_lift_eps_mm"),
        "pad_lift_settings_consistent": pad_lift_settings_consistent,
    }


def format_grasp_diagnostics(summary: dict) -> str:
    """The §0 metric-1/2 line: what a 0.0 IoU hides.

    Empty when no episode carried the instruments at all, because
    "closed inside the envelope in 0 of 2 episodes" and "nobody was
    measuring" are opposite findings and must not print the same string —
    that ambiguity IS the F-99 trap. `grasp_env_observed_episodes` and
    `pad_observed_episodes` are the denominators, so every number on this
    line can be recounted from results.csv.
    """
    env_episodes = summary.get("grasp_env_observed_episodes") or 0
    pad_episodes = summary.get("pad_observed_episodes") or 0
    if not env_episodes and not pad_episodes:
        return ""
    parts = []
    if env_episodes:
        parts.append(
            f"close in envelope {summary.get('grasp_close_in_env_episodes', 0)}"
            f"/{env_episodes} episodes"
        )
        # Separately, so "never closed" is distinguishable from "closed in
        # the wrong place" — the MISPLACED vs ABSENT split of §0.
        parts.append(f"close anywhere {summary.get('grasp_close_any_episodes', 0)}")
        parts.append(f"entered {summary.get('grasp_env_entered_episodes', 0)}")
        closest = summary.get("min_grasp_env_min_dist_m")
        if closest is not None:
            parts.append(f"closest {closest * 1000:.0f} mm")
        envelope = summary.get("grasp_env_max_dist_m")
        angle = summary.get("grasp_env_max_angle_deg")
        if envelope is not None and angle is not None:
            # A distance without the envelope it was judged against is not
            # a result: this repo holds both a 40 mm and a 45 mm gate.
            parts.append(f"envelope {envelope * 1000:.0f} mm/{angle:.0f} deg")
        elif not summary.get("grasp_env_settings_consistent", True):
            parts.append("envelope INCONSISTENT across episodes")
    if pad_episodes:
        disp = summary.get("mean_pad_disp_max_mm")
        pad = "pad displacement unmeasured" if disp is None else f"pad moved {disp:.1f} mm max"
        toward = summary.get("mean_pad_disp_toward_target_mm")
        if toward is not None:
            pad += f" ({toward:+.1f} mm toward target)"
        parts.append(f"{pad} over {pad_episodes} episodes")
        # P4 sustained lift, only once a full pad_lift_window_s was ever
        # seen — "0 of N lifted" must not print when 0 of N were judged.
        lift_episodes = summary.get("pad_lift_measured_episodes") or 0
        if lift_episodes:
            lifted = f"lifted {summary.get('pad_lifted_episodes', 0)}/{lift_episodes} episodes"
            window = summary.get("pad_lift_window_s")
            eps = summary.get("pad_lift_eps_mm")
            if window is not None and eps is not None:
                detail = f"sustained >= {eps:.0f} mm for {window:.1f} s"
                best = summary.get("max_pad_lift_sustained_mm")
                if best is not None:
                    detail += f"; best {best:.1f} mm"
                lifted += f" ({detail})"
            elif not summary.get("pad_lift_settings_consistent", True):
                lifted += " (lift settings INCONSISTENT across episodes)"
            parts.append(lifted)
    return "grasp: " + ", ".join(parts)


def format_diagnostics(summary: dict) -> str:
    """Executor health, then the grasp/pad signal, for the console. Empty
    when a run carries neither (e.g. every episode errored before the
    rollout)."""
    lines = []
    clamp = summary.get("mean_clamp_fraction")
    if clamp is not None:
        lines.append(_format_executor(summary, clamp))
    grasp = format_grasp_diagnostics(summary)
    if grasp:
        lines.append(grasp)
    return "\n".join(lines)


def _format_executor(summary: dict, clamp: float) -> str:
    parts = [f"clamped {clamp:.1%} of ticks (worst episode {summary['max_clamp_fraction']:.1%})"]
    # Always adjacent to the clamp fraction — it is what makes it mean
    # anything across runs (F-88).
    ticks = summary.get("mean_ticks")
    rate = summary.get("mean_control_hz_sim")
    if ticks is not None and rate is not None:
        parts.append(f"{ticks:.0f} ticks/episode at {rate:.1f} Hz sim")
    delta = summary.get("max_requested_delta")
    if delta is not None:
        parts.append(f"max requested delta {delta:.4f} rad/tick")
    stale = summary.get("mean_stale_chunks")
    if stale is not None:
        parts.append(f"stale chunks {stale:.1f}/episode")
    infer = summary.get("mean_infer_s")
    if infer is not None:
        parts.append(f"infer {infer * 1000:.0f} ms")
    return "executor: " + ", ".join(parts)


# F-53: the executor clamp is the only rate limiter in the whole command
# path (the sim bridge applies raw targets). Sweeping it DOWN is a
# legitimate experiment; sweeping it up is how a policy error becomes a
# collision, so the sweep refuses.
MAX_DELTA_CEILING = 0.05


def parse_sweep_variants(replan_steps: str, max_deltas: str) -> list[dict]:
    """Cartesian product of the two swept executor settings, in the order
    given, labelled with only what actually varies."""
    replans = [int(v) for v in replan_steps.split(",") if v.strip()]
    deltas = [float(v) for v in max_deltas.split(",") if v.strip()]
    if not replans or not deltas:
        raise SystemExit("--replan-steps-sweep and --max-delta-sweep need a value each")
    for delta in deltas:
        if delta > MAX_DELTA_CEILING:
            raise SystemExit(
                f"--max-delta {delta} exceeds the frozen {MAX_DELTA_CEILING} rad/tick "
                "limit (F-53: the executor clamp is the only rate limiter in the "
                "command path) — sweep downwards only"
            )
    variants = []
    for replan in replans:
        for delta in deltas:
            label = f"replan{replan}"
            if len(deltas) > 1:
                label += f"_delta{delta:g}"
            variants.append({"label": label, "replan_steps": replan, "max_delta": delta})
    return variants


# Grasp/pad ahead of IoU on purpose: §0 ranks on the intermediate metrics
# because IoU is floored at 0.0 and cannot separate two failing policies.
SWEEP_COLUMNS = (
    ("variant", "variant", "{}"),
    ("scored", "scored", "{}"),
    ("grasp_env_entered_episodes", "entered", "{}"),
    ("grasp_close_in_env_episodes", "close@env", "{}"),
    ("mean_pad_disp_max_mm", "pad mm", "{:.1f}"),
    ("mean_iou", "mean IoU", "{:.3f}"),
    ("orientation_correct", "orient", "{}"),
    ("mean_control_hz_sim", "Hz sim", "{:.1f}"),
    ("mean_clamp_fraction", "clamped", "{:.1%}"),
    ("max_requested_delta", "max req delta", "{:.4f}"),
    ("mean_stale_chunks", "stale", "{:.1f}"),
    ("mean_infer_s", "infer s", "{:.3f}"),
)


# A count is 0 both when the instrument watched and saw nothing and when
# no instrument ran, and only one of those is a result. A column listed
# here prints "-" unless its denominator says something was measured.
SWEEP_COLUMN_DENOMINATORS = {
    "grasp_env_entered_episodes": "grasp_env_observed_episodes",
    "grasp_close_in_env_episodes": "grasp_env_observed_episodes",
}


def sweep_table(results: list[tuple[str, dict]]) -> str:
    """Fixed-width comparison of several summaries, one row per variant."""
    rows = []
    for variant, summary in results:
        cells = []
        for key, _, fmt in SWEEP_COLUMNS:
            value = variant if key == "variant" else summary.get(key)
            denominator = SWEEP_COLUMN_DENOMINATORS.get(key)
            if denominator is not None and not summary.get(denominator):
                value = None
            cells.append("-" if value is None else fmt.format(value))
        rows.append(cells)

    headers = [header for _, header, _ in SWEEP_COLUMNS]
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
              for i, h in enumerate(headers)]
    lines = ["  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True)).rstrip()]
    lines.append("  ".join("-" * w for w in widths))
    for row in rows:
        lines.append("  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)).rstrip())
    return "\n".join(lines)


def run_batch(
    runner: EpisodeRunner,
    episodes: int,
    rollout_s: float,
    output_dir: Path,
    rate_hz: float = 20.0,
    run_name: str | None = None,
    episode_setup: Callable[[int], dict] | None = None,
) -> dict:
    """``episode_setup(index)``, when given, reconfigures the runner for that
    episode (swap the heuristic reference, jitter the approach goal) before
    ``run_episode``, and its return value is merged into the row as
    provenance columns for results.csv.
    """
    run_name = run_name or time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "results.csv"

    rows = []
    for index in range(episodes):
        fatal = None
        extra: dict = {}
        try:
            extra = episode_setup(index) if episode_setup else {}
            # artifact_dir: each episode's eval frames land in
            # <run_dir>/episode_<n>/ next to results.csv (F-31).
            row = runner.run_episode(index, rollout_s, rate_hz, artifact_dir=run_dir)
            row = {**extra, **row}  # runner keys win on collision
        except StartPoseNotReached as exc:
            # NOT one bad episode: every episode of the batch starts from the
            # same reset, so a start pose that is unreachable now is
            # unreachable for the rest of the run. Three runs finished and
            # were reported as a comparison of start poses none of them ever
            # reached. Keep the ledger written so far, then abort.
            log.error("episode %d: %s", index, exc)
            row = {**extra, "episode": index, "iou": None, "error": str(exc)}
            fatal = exc
        except Exception as exc:
            # With the traceback: a bare message ("inhomogeneous shape (3,)")
            # once cost a 9-minute episode to locate.
            log.error("episode %d failed: %s", index, exc, exc_info=True)
            row = {**extra, "episode": index, "iou": None, "error": str(exc)}
        rows.append(row)
        fields = sorted({key for r in rows for key in r})
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        if fatal is not None:
            raise fatal

    summary = {"run": run_name, "task_instruction": runner.task, "results_csv": str(csv_path)}
    summary.update(summarize_rows(rows))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("batch done: %s", json.dumps(summary, indent=2))
    return summary
