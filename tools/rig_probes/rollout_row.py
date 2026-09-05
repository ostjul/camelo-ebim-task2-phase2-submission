#!/usr/bin/env python3
"""Append one scored row to the T6 rollout ledger (protocol 15 §2.4-2.5).

Reads an archived rollout's own artifacts — its ``t6_act_*.log`` (the
``rollout stats: {...}`` / ``real-arm session: {...}`` python-literal dict
lines that ``scripts/run_policy.py`` prints), its ``t6_act_*.csv`` joint
trace, and ``outputs/rig/munich_2026-09-01/t6_slot_sequence.csv`` — plus the
operator's own judgement calls (picked / placed / the video-timestamped
contact frame) that nothing in the log can supply, and derives one row of
``outputs/rig/munich_2026-09-01/rollouts.csv`` (38 columns, header already
present). Nothing here re-runs the rollout; it only re-derives the row from
what was already recorded, per AGENTS.md ("the summary lies, not the
measurement") — every field is printed before it is appended.

    python3 tools/rig_probes/rollout_row.py \\
        --run-dir outputs/rig/munich_2026-09-01/t6_act_20260903_121558 \\
        --seq-index 10 --picked 1 --placed 1 --contact-frame 148 \\
        --video outputs/rig/munich_2026-09-01/t6_overlays/121558.mp4 \\
        --notes "clean grasp" --operator julian \\
        --policy act --checkpoint-step 100000 --block B2

``--dry-run`` prints the row without appending it.

Fields that cannot be derived (a missing ``--task``/``task=...`` in the log,
a missing ``slot_grasp_poses.json`` or slot, a missing contact frame) are
left blank rather than guessed, matching every other "blank if unknown"
column in the schema (docs/realdata/15_RIG_WINDOW_RUNBOOK.md §2.5).
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

#: The rollouts.csv column order — must match the header already written to
#: outputs/rig/munich_2026-09-01/rollouts.csv byte-for-byte. Checked at
#: append time (`_check_header`) so a drifted schema fails loudly instead of
#: silently writing a misaligned row.
HEADER = [
    "ts_iso",
    "rollout_id",
    "policy",
    "checkpoint_step",
    "run_job_id",
    "block",
    "slot",
    "slot_x_norm",
    "seq_index",
    "task_caption_sha1",
    "state_layout",
    "action_layout",
    "chunk_size",
    "n_executed_steps",
    "rate_hz",
    "max_delta",
    "temporal_ensemble",
    "num_inference_steps",
    "rng_seed",
    "start_pose_ok",
    "contact_frame",
    "close_cmd_frame",
    "timing_error_frames",
    "pose_at_close_rad_json",
    "pose_err_rad",
    "verdict",
    "verdict_reason",
    "pad_picked",
    "pad_placed",
    "task_success",
    "clamped_pct",
    "max_requested_delta",
    "stale_chunks",
    "ticks_per_s",
    "rollout_s",
    "video_path",
    "operator",
    "notes",
]

#: ACT trains at chunk_size 21 (docs/realdata/15 §1.6); the executed window
#: is --replan-steps, read from the log when present, else this default.
CHUNK_SIZE = 21
DEFAULT_N_EXECUTED_STEPS = 10
RATE_HZ = 20
MAX_DELTA = 0.05
STATE_LAYOUT = "s27a15"
ACTION_LAYOUT = "s27a15"
#: ACT's plain-regression head is deterministic (15 §1.2) — the sole rung
#: exempt from a pinned per-rollout RNG draw; it logs 0 for the record.
ACT_RNG_SEED = 0

#: Reuse the offline probe's pose tolerance so MISPLACED(pose) means the
#: same thing on both sides (camelo/eval/offline_probe.py FROZEN_POSE_TOL_RAD
#: docstring: "produces 0.619 at a 1.0 s tolerance"; 15 §2.4).
POSE_TOL_RAD = 0.619
#: 1.0 s at 20 Hz (offline_probe.py DEFAULT_TIMING_TOL_S / 15 §2.4).
TIMING_TOL_FRAMES = 20
#: The demo-close / policy-close threshold on the gripper channel
#: (camelo/eval/offline_probe.py FROZEN_CLOSE_THRESHOLD).
CLOSE_THRESHOLD = 0.6

_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
_ROLLOUT_STATS_RE = re.compile(r"^rollout stats: (\{.*\})\s*$", re.MULTILINE)
_REAL_ARM_SESSION_RE = re.compile(r"^real-arm session: (\{.*\})\s*$", re.MULTILINE)
_REPLAN_STEPS_RE = re.compile(r"--replan-steps[ =]+(\d+)")
_TASK_FLAG_RE = re.compile(r"""--task[ =]+(?:"([^"]+)"|'([^']+)')""")
_TASK_RESET_RE = re.compile(r"""\btask=(?:"([^"]+)"|'([^']+)')""")


class RolloutRowError(SystemExit):
    """Raised (as SystemExit) for anything a human must fix before re-running."""


def find_run_files(run_dir: Path) -> tuple[Path, Path]:
    """Return (log_path, csv_path) inside ``run_dir``, refusing ambiguity."""
    logs = sorted(run_dir.glob("*.log"))
    csvs = sorted(run_dir.glob("*.csv"))
    if len(logs) != 1:
        raise RolloutRowError(
            f"{run_dir}: expected exactly one *.log, found {[p.name for p in logs]}"
        )
    if len(csvs) != 1:
        raise RolloutRowError(
            f"{run_dir}: expected exactly one *.csv, found {[p.name for p in csvs]}"
        )
    return logs[0], csvs[0]


def extract_first_timestamp(log_text: str) -> str:
    """The log's first ``YYYY-MM-DD HH:MM:SS,mmm`` timestamp, as ISO-8601."""
    m = _TS_RE.search(log_text)
    if m is None:
        raise RolloutRowError("log has no timestamp line — cannot derive ts_iso")
    dt = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f")
    return dt.isoformat()


def extract_dict_line(log_text: str, pattern: re.Pattern[str], label: str) -> dict:
    """Parse a ``label: {...}`` python-literal-dict line (ast, not json)."""
    m = pattern.search(log_text)
    if m is None:
        raise RolloutRowError(f"log has no {label!r} line")
    return ast.literal_eval(m.group(1))


def extract_task_caption(log_text: str) -> str | None:
    """The ``--task`` value, or a ``task='...'`` reset line; None if absent.

    Neither is printed by the current run_policy.py/real_arms.py — the
    archived T6 logs measured on 2026-09-01..03 carry no task text at all —
    so a caller relying on this for an old log gets None and an empty
    ``task_caption_sha1``, not a guessed caption (a wrong guess would be
    worse than a blank cell: F-68's whole point is that the caption is
    byte-exact and never retyped).
    """
    for pattern in (_TASK_FLAG_RE, _TASK_RESET_RE):
        m = pattern.search(log_text)
        if m is not None:
            return m.group(1) if m.group(1) is not None else m.group(2)
    return None


def extract_replan_steps(log_text: str, default: int = DEFAULT_N_EXECUTED_STEPS) -> int:
    """The ``--replan-steps`` value from the log's argv line, else ``default``."""
    m = _REPLAN_STEPS_RE.search(log_text)
    return int(m.group(1)) if m is not None else default


def read_rollout_csv(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def find_close_cmd_frame(rows: list[dict]) -> int | None:
    """First row's ``tick`` where ``cmd_grip_right`` crosses below 0.6."""
    for row in rows:
        raw = row.get("cmd_grip_right", "")
        if raw == "":
            continue
        if float(raw) < CLOSE_THRESHOLD:
            return int(row["tick"])
    return None


def pose_at_frame(rows: list[dict], frame: int) -> list[float] | None:
    """The 7 ``meas_right_j1..7`` values at the row whose ``tick == frame``."""
    for row in rows:
        if int(row["tick"]) == frame:
            return [float(row[f"meas_right_j{j}"]) for j in range(1, 8)]
    return None


def load_slot_sequence(path: Path) -> dict[int, dict]:
    """``{rollout_idx: {"slot": ..., "x_norm": ...}}`` from t6_slot_sequence.csv.

    The file carries a leading ``# generator: ...`` comment line before the
    real header — skip any line starting with ``#``.
    """
    with path.open(newline="") as f:
        lines = [line for line in f if not line.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    return {int(row["rollout_idx"]): row for row in reader}


def load_slot_grasp_poses(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def pose_err_rad(pose: list[float], mean_pose: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(pose, mean_pose, strict=True)))


def compute_verdict(
    close_cmd_frame: int | None,
    timing_error_frames: int | None,
    pose_err: float | None,
    *,
    pose_tol_rad: float = POSE_TOL_RAD,
    timing_tol_frames: int = TIMING_TOL_FRAMES,
) -> tuple[str, str]:
    """The rig-side mechanism verdict (15 §2.4): ABSENT / MISPLACED / TIMED / UNTIMED.

    Order is load-bearing (mirrors camelo/eval/offline_probe.py's
    ``classify_verdict``, simplified for a single rig attempt with no
    chunk-placement sweep): ABSENT first (nothing to score), then pose
    (a close at the wrong configuration is worth flagging even if timing
    also drifted), then timing, else UNTIMED. ``verdict_reason`` mirrors the
    probe's: "pose" (not fixable by re-timing) vs "chunk_placement" (is).
    """
    if close_cmd_frame is None:
        return "ABSENT", "none"
    if pose_err is not None and pose_err > pose_tol_rad:
        return "MISPLACED", "pose"
    if timing_error_frames is not None and abs(timing_error_frames) <= timing_tol_frames:
        return "TIMED", "none"
    return "UNTIMED", "chunk_placement"


def _check_header(csv_path: Path) -> None:
    if not csv_path.exists():
        return
    with csv_path.open(newline="") as f:
        first_line = f.readline().rstrip("\n")
    if first_line and first_line.split(",") != HEADER:
        raise RolloutRowError(
            f"{csv_path}: on-disk header does not match this tool's HEADER — "
            "the schema has drifted; fix one side before appending "
            f"(disk: {first_line!r})"
        )


def build_row(args: argparse.Namespace) -> dict:
    log_path, csv_path = find_run_files(args.run_dir)
    log_text = log_path.read_text()
    rows = read_rollout_csv(csv_path)

    rollout_stats = extract_dict_line(log_text, _ROLLOUT_STATS_RE, "rollout stats")
    real_arm_session = extract_dict_line(log_text, _REAL_ARM_SESSION_RE, "real-arm session")

    task_caption = extract_task_caption(log_text)
    if task_caption is None:
        print(
            "warning: no --task/task=... found in the log; task_caption_sha1 left blank",
            file=sys.stderr,
        )
        task_caption_sha1 = ""
    else:
        task_caption_sha1 = hashlib.sha1(task_caption.encode("utf-8")).hexdigest()

    seq_row = load_slot_sequence(args.slot_sequence_csv).get(args.seq_index)
    if seq_row is None:
        raise RolloutRowError(f"{args.slot_sequence_csv}: no rollout_idx == {args.seq_index}")
    slot = seq_row["slot"]
    slot_x_norm = seq_row["x_norm"]

    close_cmd_frame = find_close_cmd_frame(rows)
    contact_frame = args.contact_frame
    timing_error_frames = (
        close_cmd_frame - contact_frame
        if close_cmd_frame is not None and contact_frame is not None
        else None
    )

    pose_at_close = pose_at_frame(rows, close_cmd_frame) if close_cmd_frame is not None else None
    pose_at_close_rad_json = "" if pose_at_close is None else json.dumps(pose_at_close)

    pose_err: float | None = None
    if pose_at_close is not None and args.slot_poses_json.exists():
        slot_poses = load_slot_grasp_poses(args.slot_poses_json)
        slot_entry = slot_poses.get(slot)
        if slot_entry is not None:
            pose_err = pose_err_rad(pose_at_close, slot_entry["mean_right_arm_rad"])

    verdict, verdict_reason = compute_verdict(close_cmd_frame, timing_error_frames, pose_err)

    ticks = rollout_stats.get("ticks")
    clamped_ticks = rollout_stats.get("clamped_ticks")
    sim_seconds = rollout_stats.get("sim_seconds")
    clamped_pct = 100.0 * clamped_ticks / ticks if ticks else ""
    ticks_per_s = ticks / sim_seconds if ticks is not None and sim_seconds else ""

    start_pose_max_err = real_arm_session.get("start_pose_max_err_rad")
    start_pose_ok = "" if start_pose_max_err is None else (start_pose_max_err <= 0.10)

    task_success = 1 if (args.picked and args.placed) else 0

    row = {
        "ts_iso": extract_first_timestamp(log_text),
        "rollout_id": args.run_dir.name,
        "policy": args.policy,
        "checkpoint_step": args.checkpoint_step or "",
        "run_job_id": "",
        "block": args.block or "",
        "slot": slot,
        "slot_x_norm": slot_x_norm,
        "seq_index": args.seq_index,
        "task_caption_sha1": task_caption_sha1,
        "state_layout": STATE_LAYOUT,
        "action_layout": ACTION_LAYOUT,
        "chunk_size": CHUNK_SIZE,
        "n_executed_steps": extract_replan_steps(log_text),
        "rate_hz": RATE_HZ,
        "max_delta": MAX_DELTA,
        "temporal_ensemble": "none",
        "num_inference_steps": "n/a",
        "rng_seed": ACT_RNG_SEED,
        "start_pose_ok": start_pose_ok,
        "contact_frame": contact_frame if contact_frame is not None else "",
        "close_cmd_frame": close_cmd_frame if close_cmd_frame is not None else "",
        "timing_error_frames": timing_error_frames if timing_error_frames is not None else "",
        "pose_at_close_rad_json": pose_at_close_rad_json,
        "pose_err_rad": "" if pose_err is None else pose_err,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "pad_picked": args.picked,
        "pad_placed": args.placed,
        "task_success": task_success,
        "clamped_pct": clamped_pct,
        "max_requested_delta": rollout_stats.get("max_requested_delta", ""),
        "stale_chunks": rollout_stats.get("stale_chunks", ""),
        "ticks_per_s": ticks_per_s,
        "rollout_s": sim_seconds if sim_seconds is not None else "",
        "video_path": args.video,
        "operator": args.operator or "",
        "notes": args.notes,
    }
    missing = set(row) - set(HEADER)
    if missing:  # pragma: no cover — a coding error, not a data error
        raise RolloutRowError(f"internal: row has columns not in HEADER: {missing}")
    return row


def append_row(csv_path: Path, row: dict) -> None:
    _check_header(csv_path)
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _contact_frame_type(value: str) -> int | None:
    return None if value.strip().lower() == "none" else int(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seq-index", type=int, required=True)
    parser.add_argument("--picked", type=int, choices=(0, 1), required=True)
    parser.add_argument("--placed", type=int, choices=(0, 1), required=True)
    parser.add_argument(
        "--contact-frame", type=_contact_frame_type, required=True, metavar="F|none"
    )
    parser.add_argument("--video", required=True, help="path to the rollout video")
    parser.add_argument("--notes", required=True)
    parser.add_argument("--operator", default="")
    parser.add_argument("--policy", default="act")
    parser.add_argument("--checkpoint-step", default="")
    parser.add_argument("--block", default="")
    parser.add_argument(
        "--rollouts-csv",
        type=Path,
        default=Path("outputs/rig/munich_2026-09-01/rollouts.csv"),
    )
    parser.add_argument(
        "--slot-sequence-csv",
        type=Path,
        default=Path("outputs/rig/munich_2026-09-01/t6_slot_sequence.csv"),
    )
    parser.add_argument(
        "--slot-poses-json",
        type=Path,
        default=Path("outputs/rig/munich_2026-09-01/slot_grasp_poses.json"),
    )
    parser.add_argument("--dry-run", action="store_true", help="print the row but do not append it")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    row = build_row(args)
    for key in HEADER:
        print(f"{key}={row[key]!r}")
    if args.dry_run:
        print("(dry run — nothing appended)")
        return 0
    append_row(args.rollouts_csv, row)
    print(f"appended to {args.rollouts_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
