#!/usr/bin/env python3
"""Score a T5 replay / dry-run joint trace (run_policy --joint-csv) from its rows.

usage: score_t5_trace.py <joint.csv> [ep_actions.npy]

Prints per-joint max |measured - commanded| over the rollout phase, the worst
joint, clamped_pct, max_publish_gap_s and, with the npy, per-joint max
|measured - recorded| and |commanded - recorded| at the same tick (20 Hz,
frame 0 = tick 1). Everything is recomputed from the rows (docs/realdata/16
R-55/R-61); the runner's printed summary is not trusted.
"""
import csv
import sys

import numpy as np

SIDES = ("left", "right")


def joints(rows, prefix, side):
    return np.array([[float(r[f"{prefix}_{side}_j{j}"]) for j in range(1, 8)] for r in rows])


path = sys.argv[1]
rows = list(csv.DictReader(open(path)))
print(f"rows={len(rows)}  columns={len(rows[0])}")
phases = sorted({r["phase"] for r in rows})
print("phases:", phases, {p: sum(1 for r in rows if r["phase"] == p) for p in phases})
roll = [r for r in rows if r["phase"] == "rollout" and r["cmd_left_j1"] != ""]
if not roll:
    roll = [r for r in rows if r["cmd_left_j1"] != ""]
    print("NOTE: no 'rollout' rows with commands; using all commanded rows")
print(f"commanded rows: {len(roll)}  t={float(roll[0]['t']):.2f}..{float(roll[-1]['t']):.2f} s")
cmd = {s: joints(roll, "cmd", s) for s in SIDES}
meas = {s: joints(roll, "meas", s) for s in SIDES}
for s in SIDES:
    err = np.abs(meas[s] - cmd[s])
    span = np.round(cmd[s].max(0) - cmd[s].min(0), 3).tolist()
    print(f"{s}: max|meas-cmd| per joint = {np.round(err.max(0), 4).tolist()}"
          f"  worst j{int(err.max(0).argmax()) + 1}  mean {err.mean():.4f}  cmd range {span}")
clamped = [float(r["clamped"]) for r in roll if r["clamped"] != ""]
if clamped:
    c = np.array(clamped)
    print(f"clamped ticks: {int(c.sum())}/{len(c)} = {100 * c.mean():.1f} %")
gaps = [float(r["max_publish_gap_s"]) for r in rows if r["max_publish_gap_s"] != ""]
if gaps:
    print(f"max_publish_gap_s (worst so far, last row): {max(gaps):.3f}")
ages = [(i, float(r["state_age_s"]))
        for i, r in enumerate(rows) if r.get("state_age_s", "") != ""]
if ages:
    worst_i, worst = max(ages, key=lambda p: p[1])
    print(f"joint-state age (U-27): n={len(ages)} median {np.median([a for _, a in ages]):.3f} s"
          f" max {worst:.3f} s at row {worst_i + 1} (t={rows[worst_i]['t']}) — a max that grows"
          " without bound means the arm's publisher died and the meas_* columns froze")
inf = [float(r["infer_ms"]) for r in rows if r["infer_ms"] != ""]
if inf:
    print(f"inference: n={len(inf)} median {np.median(inf):.1f} ms max {max(inf):.1f} ms")
grip = [(float(r["cmd_grip_right"]), float(r["meas_grip_right"]))
        for r in roll if r["cmd_grip_right"] != ""]
if grip:
    ga = np.array(grip)
    print(f"right gripper cmd range [{ga[:, 0].min():.2f},{ga[:, 0].max():.2f}]"
          f" meas range [{ga[:, 1].min():.2f},{ga[:, 1].max():.2f}]")
if len(sys.argv) > 2:
    a = np.load(sys.argv[2])
    n = min(len(roll), len(a))
    rec = {"left": a[:n, 3:10], "right": a[:n, 10:17]}
    for s in SIDES:
        e = np.abs(meas[s][:n] - rec[s])
        ec = np.abs(cmd[s][:n] - rec[s])
        worst = int(e.max(0).argmax()) + 1
        print(f"{s} vs RECORDED (first {n} frames): max|meas-rec| {np.round(e.max(0), 4).tolist()}"
              f" worst j{worst}; max|cmd-rec| {np.round(ec.max(0), 4).tolist()}")
