"""Per-tick CSV of commanded vs measured arm joints, and the T5 verdict.

docs/realdata/16 §5 T5 asks for one thing: *"measured joints track the
recorded ones — report the per-joint max error and clamped_pct"*. Until
now the runner wrote no measured-joint trace at all on the real rig —
``TcpTrajectoryLog`` looks like it should serve, but it drops any row with
a non-finite base/spine, and on the Munich station both are always NaN, so
its CSV comes out empty exactly where T5 needs it.

So this log carries only what the rig actually publishes: the 14 commanded
arm targets, the 14 measured arm joints, both gripper channels either way,
and the per-tick control telemetry (clamp, inference, publish gap). No FK,
no base, no spine — nothing that can silently void a row.

Columns (fixed order, see ``COLUMNS``)::

    t, t_sim, tick, phase,
    cmd_left_j1..j7, cmd_right_j1..j7,
    meas_left_j1..j7, meas_right_j1..j7,
    cmd_grip_left, cmd_grip_right, meas_grip_left, meas_grip_right,
    clamped, infer_ms, max_publish_gap_s, state_age_s, gripper_latched

``t`` is monotonic seconds since the log opened, which the runner does
immediately before the first control tick of the rollout — so ``t`` is
"seconds since the rollout started" on the one clock that cannot be
rebased by a scene reset. ``t_sim`` is the same tick's ``Obs.t_sim`` (the
clock the executor interpolates its chunk on, and the key every other
per-tick CSV in this repo is written against — without it this file cannot
be joined to them).

``phase`` is written by the caller and the summary aggregates **only**
``phase == "rollout"`` rows. Today the runner opens the log after the
approach and closes it at the end of the rollout, so that is every row it
writes; the filter exists so that an approach, hold or teardown row added
later cannot silently enter the T5 max-error number. The hold and teardown
phases live in `camelo.runner.real_arms.RealArmSession`, on either side of
`run_rollout` and therefore outside this file's lifetime.

**Where the commanded values come from**: the ``Command`` the executor
returned, read at the publish site *after* any grasp-gate override — i.e.
the arm targets after `ChunkExecutor`'s per-tick delta clamp, in the ROBOT
frame, and the gripper open fractions that actually went out. That is
deliberately the executor's output rather than
`CommandPublisher.publish_arms`' wire values, because the wire is not
always comparable to a measured joint: ``--arm-command-frame gello``
publishes ``g0 + dir*(q - q0)`` and ``--arms left`` republishes a held
right arm's frozen pose. Under those flags the CSV states what the policy
asked the arm to do, which is what "do the measured joints track it?"
means; the wire values are the publisher's business and its own telemetry
reports them.

``state_age_s`` is the U-27 liveness column: the worst WALL age, at that
tick, of the arm/gripper `JointState` streams the ``meas_*`` columns are
read from (`camelo.control.state_age.max_state_age`). It is what makes a
dead stream visible to a POST-HOC reader of this file — on 2026-09-02 the
rig's right `ros2_control_node` died 0.5 s after activation and the runner
recorded 30 s of "measured" values that were one frozen sample repeated
(R-64). Empty when nothing measured it (sim, or a collector with no arm
streams); it is never ``0.0`` by omission.

A tick on which the executor had nothing to say (no chunk yet, or a chunk
invalidated by a clock rebase) still gets a row — one row per control tick
— with every commanded field and ``clamped`` left EMPTY. An empty field is
never ``0.0``: a missing command must not read as "commanded zero", and a
tick that never reached the clamp must not read as "not clamped".

Every row is flushed as it is written, so an operator's Ctrl+C keeps the
rows collected so far (the sine-probe lesson: the rows had been collected
and were lost).
"""

from __future__ import annotations

import csv
import logging
import math
import time
from pathlib import Path

log = logging.getLogger(__name__)

#: The phase whose rows the T5 summary is computed over.
ROLLOUT_PHASE = "rollout"

N_JOINTS = 7
SIDES = ("left", "right")


def _joint_columns(prefix: str) -> tuple[str, ...]:
    return tuple(
        f"{prefix}_{side}_j{j + 1}" for side in SIDES for j in range(N_JOINTS)
    )


CMD_JOINT_COLUMNS = _joint_columns("cmd")
MEAS_JOINT_COLUMNS = _joint_columns("meas")
COLUMNS: tuple[str, ...] = (
    "t",
    "t_sim",
    "tick",
    "phase",
    *CMD_JOINT_COLUMNS,
    *MEAS_JOINT_COLUMNS,
    "cmd_grip_left",
    "cmd_grip_right",
    "meas_grip_left",
    "meas_grip_right",
    "clamped",
    "infer_ms",
    "max_publish_gap_s",
    "state_age_s",
    "gripper_latched",
)


def _floats(values, n: int, what: str) -> list[float]:
    """``values`` as exactly ``n`` python floats (never numpy scalars).

    The conversion is load-bearing, not cosmetic: a numpy float32 written
    through ``str()`` does not round-trip, so the summary computed here and
    a recomputation from the CSV would disagree in the last digits — and
    the whole point of the file is that the printed verdict is
    re-derivable from the rows.
    """
    out = [float(v) for v in values]
    if len(out) != n:
        raise ValueError(f"{what} must have {n} values, got {len(out)}")
    return out


class JointTrackingLog:
    """Append-only CSV of commanded vs measured arm joints, one row per tick."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.n = 0
        self.n_rollout = 0
        self.n_commanded = 0  # rollout rows that carried a command
        self.n_executor_ticks = 0  # rollout rows where the clamp was evaluated
        self.n_clamped = 0
        # Per-joint max |measured - commanded| over the rollout phase, and
        # the tick each maximum was seen on. Kept per item, never as a
        # single headline: the block printed at the end has to be checkable
        # against the rows underneath it.
        self._max_err: dict[str, float] = {c: 0.0 for c in CMD_JOINT_COLUMNS}
        self._max_err_tick: dict[str, int | None] = {c: None for c in CMD_JOINT_COLUMNS}
        self._max_grip_err: dict[str, float] = {"left": 0.0, "right": 0.0}
        self._t0 = time.monotonic()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=list(COLUMNS))
        self._w.writeheader()
        self._f.flush()
        log.info("joint trace -> %s", self.path)

    # -- writing -----------------------------------------------------------
    def record(
        self,
        tick: int,
        phase: str,
        t_sim: float,
        measured_arms,
        measured_grippers,
        commanded_arms=None,
        commanded_grippers=None,
        clamped: bool | None = None,
        infer_s: float | None = None,
        max_publish_gap_s: float | None = None,
        state_age_s: float | None = None,
        gripper_latched: bool = False,
    ) -> None:
        """One control tick. ``commanded_*``/``clamped`` None = nothing was sent.

        ``measured_arms`` is (14,) — left j1..7 then right j1..7, i.e.
        ``state[S_LEFT_ARM]`` concatenated with ``state[S_RIGHT_ARM]`` —
        and ``commanded_arms`` is the same layout from the executor's
        ``Command``. Grippers are (2,), left then right, as canonical open
        fractions (1.0 open).

        ``gripper_latched`` (default False; the trailing ``gripper_latched``
        column, appended after ``state_age_s``) is 1 when
        `camelo.control.gripper_latch.GripperLatch` forced this tick's
        ``cmd_grip_right``, 0 otherwise — including every tick of a run with
        ``--gripper-latch`` absent, so an old reader that ignores the new
        column still sees exactly the file it always did.
        """
        if self._f is None:
            return
        measured = _floats(measured_arms, 2 * N_JOINTS, "measured_arms")
        meas_grip = _floats(measured_grippers, 2, "measured_grippers")
        row: dict[str, object] = {
            "t": time.monotonic() - self._t0,
            "t_sim": float(t_sim),
            "tick": int(tick),
            "phase": phase,
            "clamped": "" if clamped is None else int(bool(clamped)),
            "infer_ms": "" if infer_s is None else float(infer_s) * 1000.0,
            "max_publish_gap_s": (
                "" if max_publish_gap_s is None else float(max_publish_gap_s)
            ),
            "state_age_s": "" if state_age_s is None else float(state_age_s),
            "gripper_latched": int(bool(gripper_latched)),
        }
        for column, value in zip(MEAS_JOINT_COLUMNS, measured, strict=True):
            row[column] = value
        row["meas_grip_left"], row["meas_grip_right"] = meas_grip

        if commanded_arms is None:
            for column in CMD_JOINT_COLUMNS:
                row[column] = ""
        else:
            commanded = _floats(commanded_arms, 2 * N_JOINTS, "commanded_arms")
            for column, value in zip(CMD_JOINT_COLUMNS, commanded, strict=True):
                row[column] = value
        if commanded_grippers is None:
            row["cmd_grip_left"] = row["cmd_grip_right"] = ""
        else:
            cmd_grip = _floats(commanded_grippers, 2, "commanded_grippers")
            row["cmd_grip_left"], row["cmd_grip_right"] = cmd_grip

        self._w.writerow(row)
        self._f.flush()  # a Ctrl+C must not cost the rows already collected
        self.n += 1
        self._accumulate(row, tick)

    def _accumulate(self, row: dict, tick: int) -> None:
        """Fold one written row into the rollout-phase aggregates.

        Reads the row dict, i.e. exactly the values that went into the
        file, so nothing here can drift from what a reader recomputes.
        """
        if row["phase"] != ROLLOUT_PHASE:
            return
        self.n_rollout += 1
        if row["clamped"] != "":
            self.n_executor_ticks += 1
            self.n_clamped += int(row["clamped"])
        if row[CMD_JOINT_COLUMNS[0]] == "":
            return
        self.n_commanded += 1
        for cmd_col, meas_col in zip(CMD_JOINT_COLUMNS, MEAS_JOINT_COLUMNS, strict=True):
            cmd, meas = row[cmd_col], row[meas_col]
            if not (math.isfinite(cmd) and math.isfinite(meas)):
                continue
            err = abs(meas - cmd)
            if err > self._max_err[cmd_col]:
                self._max_err[cmd_col] = err
                self._max_err_tick[cmd_col] = tick
        for side in SIDES:
            cmd, meas = row[f"cmd_grip_{side}"], row[f"meas_grip_{side}"]
            if cmd == "" or not (math.isfinite(cmd) and math.isfinite(meas)):
                continue
            self._max_grip_err[side] = max(self._max_grip_err[side], abs(meas - cmd))

    def close(self) -> None:
        if self._f is None:
            return
        self._f.flush()
        self._f.close()
        self._f = None
        log.info("joint trace: %d rows -> %s", self.n, self.path)

    # -- the T5 verdict ----------------------------------------------------
    def summary(self) -> dict:
        """Per-joint max |measured − commanded| and clamped_pct, rollout only."""
        out: dict = {
            "joint_trace_csv": str(self.path),
            "rows": self.n,
            "rollout_rows": self.n_rollout,
            "commanded_rows": self.n_commanded,
            "executor_ticks": self.n_executor_ticks,
            "clamped_ticks": self.n_clamped,
            "clamped_pct": (
                100.0 * self.n_clamped / self.n_executor_ticks
                if self.n_executor_ticks
                else None
            ),
            "max_grip_err": dict(self._max_grip_err),
        }
        for side in SIDES:
            columns = [c for c in CMD_JOINT_COLUMNS if c.startswith(f"cmd_{side}_")]
            errors = [self._max_err[c] for c in columns]
            out[f"max_err_{side}_rad"] = errors
            if self.n_commanded:
                worst = max(range(N_JOINTS), key=lambda j: errors[j])
                out[f"worst_joint_{side}"] = worst + 1  # 1-based, as j1..j7
                out[f"worst_err_{side}_rad"] = errors[worst]
                out[f"worst_tick_{side}"] = self._max_err_tick[columns[worst]]
            else:
                out[f"worst_joint_{side}"] = None
                out[f"worst_err_{side}_rad"] = None
                out[f"worst_tick_{side}"] = None
        return out

    def format_summary(self) -> str:
        """The T5 block: one labelled report, re-derivable from the CSV."""
        s = self.summary()
        pct = s["clamped_pct"]
        lines = [
            "=== joint tracking (T5, docs/realdata/16 §5) ===",
            f"csv: {s['joint_trace_csv']}  rows={s['rows']} "
            f"(phase={ROLLOUT_PHASE}: {s['rollout_rows']}, "
            f"commanded: {s['commanded_rows']})",
            "clamped_pct: "
            + ("n/a (no executor tick)" if pct is None else f"{pct:.2f}%")
            + f"  ({s['clamped_ticks']}/{s['executor_ticks']} executor ticks clamped)",
            "per-joint max |measured - commanded| [rad], "
            f"phase={ROLLOUT_PHASE} only:",
        ]
        for side in SIDES:
            errors = s[f"max_err_{side}_rad"]
            per_joint = "  ".join(
                f"j{j + 1}={errors[j]:.5f}" for j in range(N_JOINTS)
            )
            worst = s[f"worst_joint_{side}"]
            if worst is None:
                tail = "worst n/a (no commanded tick)"
            else:
                err = s[f"worst_err_{side}_rad"]
                tail = (
                    f"worst j{worst} = {err:.5f} rad ({math.degrees(err):.3f} deg) "
                    f"at tick {s[f'worst_tick_{side}']}"
                )
            lines.append(f"  {side:<5} {per_joint}")
            lines.append(f"  {side:<5} {tail}")
        grip = s["max_grip_err"]
        lines.append(
            "gripper max |measured - commanded| [open fraction]: "
            f"left {grip['left']:.4f}  right {grip['right']:.4f}"
        )
        lines.append(
            "commanded = the executor's clamped ROBOT-frame targets read at "
            "the publish site. Under --arm-command-frame gello the WIRE "
            "carries g0 + dir*(q - q0) instead, and a side left out of "
            "--arms is republished frozen — in both cases the wire differs "
            "from these targets, and only these are comparable to a "
            "measured joint. An arm the run does not drive will show a "
            "large error here because it was never followed."
        )
        return "\n".join(lines)
