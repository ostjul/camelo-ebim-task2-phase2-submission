# Contract cheat sheet (fr3duo_mobile, Task 2 / Task 1 Isaac)

Single source of truth in code: `camelo/contracts.py` (mirrored from the
benchmark; `verify_topics()` + `tests/test_contract_drift.py` keep it
honest). This page is the human-readable summary.

## Commanding the robot (what we publish)

| Topic | Type | Payload |
|---|---|---|
| `/bridge/left_joint_commands` | `sensor_msgs/JointState` | `left_fr3v2_joint1..7`, absolute rad |
| `/bridge/right_joint_commands` | `sensor_msgs/JointState` | `right_fr3v2_joint1..7`, absolute rad |
| `/bridge/left_robotiq_joint_commands` | `sensor_msgs/JointState` | name `left_robotiq_opening`, 0..1 (1 = open)\* |
| `/bridge/right_robotiq_joint_commands` | `sensor_msgs/JointState` | name `right_robotiq_opening`, 0..1\* |
| `/pedal/state` | `std_msgs/String` | token, republish > 1 Hz (1 s watchdog) |

\* **Gripper polarity — MEASURED on the DGX, 2026-08-06** (DGX_FINDINGS.md
F-18): with the task2 default `REPUBLISHER_GRIPPER_INVERT=true`, the wire
value on `/bridge/*_robotiq_joint_commands` is a **CLOSE fraction** —
`1.0` = CLOSED, `0.0` = OPEN (`driver = wire × 0.8 rad`). The canonical
20-dim action channel keeps the opposite sense (`1.0` = open, matching
`gripper_open_fraction`), so the command publisher **inverts at the wire**
(`contracts.gripper_wire_value`, polarity resolved by
`contracts.gripper_command_invert()` from the stack's env). Every
dataset-critical script runs `verify_gripper_polarity()` — a live
open/close/read-back assertion — before touching the robot.

Pedal tokens: `FWD`/`BACK` (±x), `A`/`B` (±y strafe), `A+C`/`B+C` (±yaw),
`NONE`. Fixed speeds: helper-stack defaults 0.5 m/s / 1.2 rad/s (task2).
A continuous `(vx, vy, wz)` is quantized by `camelo/control/base_quantizer.py`.

**Spine — SET on the DGX, 2026-08-07: `0.5000 m` COMMANDED.** Raised from
an initial 0.3800 m to match the reference Task 2 demos published on the
Hub, which work at 0.500 m for 96.8 % of their frames (F-58) — same robot,
same task, so matching it keeps our episodes geometrically comparable with
theirs and avoids fighting a working height someone else already tuned.
The drive does not reach its target: measured settled ~15 mm low at the old
0.38 setpoint (0.3652 m) — steady-state droop, confirmed by reading both
topics at once (F-57); expect a similar offset at 0.50. The
SOP is defined on the **commanded** value because that is what the recorder
writes into `action[19]` (`contracts.gripper`/`action_from_snapshot` reads
the applied command, falling back to measured only when absent).
`state[28]` carries the measured value, so **`action[19]` and `state[28]`
legitimately differ by the droop** — that is not a bug, and any check
comparing one against the other must allow for it. NOT commandable over
ROS; the only control is the Isaac Sim window's **Up/Down arrows**
(`isaacsim_fr3duo_teleop_bridge_core.py:666`, which prints
`franka_spine_vertical_joint: target=<m>` on every change and is
unavailable in a headless session). SOP: this one height for **every demo
and every rollout**, so `action[19]` / `state[28]` stay constant and
policies learn to ignore them. Units are **metres**, not radians.

**Task 2 approach zone — FROZEN 2026-08-10.** Numbers and stage list live
in `camelo/control/approach.py` (`TASK2_APPROACH_WAYPOINTS`,
`build_stages` / `STAGES`) and arm/spine poses in `camelo/contracts.py`
(`TASK2_SPAWN_XY_YAW`, `TASK2_APPROACH_XY_YAW`, `TASK2_ARM_READY_*`,
`SPINE_SOP_*`). Disable the whole gate with `--skip-approach`. Default
stages: spine → navigate → place_arms → **finegrained_start_position** →
start_pose → done. The fine trim (`PulseSettleTrim`, super-fine
0.002 m / 0.002 rad) is engage/coast/brake after the arms are out, then a
0.15 s in-zone stopped dwell before `start_pose` — drop it with
`ApproachController(finegrained_start_position=False)`
(the pose-based call site in `camelo/cli.py` `make_approach_from_args`).
Opt-in ``--approach perception`` swaps only the **navigate** stage for
head-camera localisation (`camelo/control/approach_perception_based.py`,
vision in `camelo/control/perception/`): yaw until **three or more** table
AABB corners are detected — line segments for geometry, with a relative
brightness cue (bright, unsaturated *versus this frame's floor*) vetoing
the grout crossings that otherwise look identical to table corners — then
walk the planner's waypoints on the perception estimate. The table world
pose is a **known landmark**, so only the ego pose is solved, through the
projection matrix and with no previous pose as input. A pose is published
only if its corners reproject within the gate, the tabletop outline it
predicts lies on detected segments, every corner it places in open view
was actually detected, and the table falls inside the implied field of
view. Two correspondences are never enough: they fit any assignment
exactly, which is how the earlier version drew a table onto empty floor.
Between accepted solves the estimate is dead-reckoned from
the commanded twist. Waypoints come from a `WaypointPlanner`; the default
returns `TASK2_APPROACH_WAYPOINTS` unchanged. Perception walks a denser
route (`PERCEPTION_APPROACH_WAYPOINTS`: west at y=2.95, overshoot to
x=1.80, then in onto the goal at (2.10, 3.05)) and drops vias already
behind the lock pose so search does not drive back into the east wall.
Transit is 10 % closer to the table than y=3.02; the penultimate via is
30 cm past the goal on −X.
Every via is yaw −90° (table-normal) within ±5°.
Dumps are head + BEV with
``gt=`` / ``est=`` (``(?, ?, ?)`` until located) plus ``resid=``/``n=``
and ``err=`` / ``body=`` / ``dgt=``.
Topic odom is display-only; once located ``start_pose`` sees the
estimate. Lost corners dead-reckon; only a border sliver that misses the
box drops the lock (prior stays frozen). Perception
keeps ``finegrained_start_position``: after
``place_arms`` it aligns 1–3 detected tabletop edges to the expected
start-pose table and pulses on that live ``est``. Navigate opens with
5 sim-s of body-right (`B`) then 1 sim-s of body-rear (`BACK`) before
yaw-search. Pose-based waypoints remain the default.

Changing it mid-dataset splits the data into geometrically inconsistent
halves — different arm reach and a different head-camera viewpoint for the
same task — and cannot be fixed without re-recording. If a task ever needs
lift motion, the fix is a ~20-line `/bridge/spine_command` subscriber in
the benchmark fork (deliberately deferred, DATA_COLLECTION.md §Spine SOP).

**Base motion quantum — MEASURED 2026-08-12 (F-93), A10G, `make
sim-scene-table HEADLESS=1 SCENE_ARGS="--render-hz 30"`.** Probe:
`tools/dgx_probes/base_motion_quantum.py --reset` (logs under
`outputs/probes/base_motion_quantum_clean.log`). Pedal speeds the helper
defaults (0.5 m/s / 1.2 rad/s). Medians over clean (non-timeout) trials;
settle = release→`|v|` under approach's stop gates.

| burst (wall) | token class | median \|Δxy\| | median \|Δyaw\| | median settle |
|---|---|---|---|---|
| 50 ms (≈ one approach tick) | FWD/BACK/A/B | 0.5–1.2 mm | ~0° | ~72 ms |
| 50 ms | A+C / B+C | 0.2–0.4 mm | ~0.01° | ~72 ms |
| 300 ms (≈ one render step wall) | FWD/BACK/A/B | 3.3–4.1 mm | ~0° | ~72–100 ms |
| 300 ms | A+C / B+C | 0.6–0.8 mm | 0.39–0.49° | ~71–74 ms |

Full-speed arithmetic for one render step (`PEDAL_* × 1/30`) would be
16.7 mm / 2.29° — measured short/medium bursts land far below that because
the swerve ramps wheel speed only after steering aligns
(`_steering_alignment_scale` in the benchmark bridge). Occasional trials
"catch" and run away (timeouts at 49–80 mm / multi-second settle) — the
same bang-bang hold that approach uses continuously.

Re-steer (second burst after a different axis), 300 ms: `FWD→A+C` median
primary yaw 1.01° but **off-axis XY scrub 28 mm**; `A→A+C` off-axis ~5 mm.
50 ms re-steer is noise (~0.2 mm).

Implications for approach: `_STOP_S` ≈ **0.08 s** (was guessed 0.05);
super-fine 0.002 m / 0.002 rad is the `PulseSettleTrim` gate — inside the
gate never fire or reverse; outside, linear catch is braked (same-axis
reverse) and yaw reverse only when opening (`err` and `wz` opposite signs;
`err` is goal−yaw), otherwise coast-to-stop; trim after `place_arms`; do
not timeout-accept a pose outside the gate (`--approach-timeout` is the
episode backstop).

**Task 2 arm ready poses — MEASURED 2026-08-10 from the reference demos.**
`TASK2_ARM_READY_LEFT/RIGHT` are medians over
`ext_hermanprawiro_task2_fixpos_v1` (F-58), whose 22 episodes park the base
at exactly `TASK2_APPROACH_XY_YAW`, so the joint values transfer directly.
Both put the flange ~0.19 m over the table top (z = 0.75 m): left 0.927 m,
right 0.945 m at the SOP spine. Cross-episode spread is ≤ 1.0° per joint on
the left and ≤ 11° on the right. Re-derive with
`tools/ready_pose_from_dataset.py`; check any candidate pose's geometry with
`tools/arm_pose_fk.py` (URDF forward kinematics, offline — it reproduces the
demos' measured flange height to within 7 mm).

The trap this replaced: the benchmark's own `ARM_READY_POSE` is the
**spine-down start pose**. The demos begin there with the spine at 0 and the
operator walks the arms down while the spine rises under them, holding the
flange near 0.94 m throughout. Command that same pose with the spine already
at the SOP and the arms sit ~0.6 m above the table, reaching over nothing —
which is exactly what happened before this was measured.

Two subtleties, both of which make naive statistics wrong:
- The left arm's **flange** parks after ~7 s and holds within 5 mm for the
  rest of the episode, but its **commanded joints keep drifting** (up to 35°
  on j2/j5), with measured tracking commanded to under 1°. That is RMPflow
  reshuffling a redundant arm around a held Cartesian target. Read the pose
  at the parking instant; averaging over the tail mixes in the drift.
- The right arm has no quiet staging window in every episode — several
  operators start reaching before the spine ramp finishes. Only episodes
  still staging at ramp-end are used; including the rest widens the spread
  from ~8° to over 60°. Episode 19 (regrasp, TRAINING.md) is dropped.

**Stage-2 base pose — MEASURED 2026-08-09 (F-69): `(2.100, 3.051,
yaw −90.0°)`.** Frame-0 `state[31:34]` of every episode in
`ext_hermanprawiro_task2_fixpos_v1` — identical in all 22 at that
precision; the H100's finer read shows a 0.4 mm y-split between two
episode groups (3.0505 vs 3.0509), so set no tolerance tighter than
that. 1.102 m from the table centre. An earlier geometry-derived guess
(2.05, 2.6) stood 0.45 m too close and visibly sprawled the arm. Launch
via `make sim-scene-table` (`TABLE_ROBOT_*`); the spawn transform IS the
reset pose, so the placement survives every episode reset — F-72
verified base and arms restore exactly.

**Task 2 instruction string — RE-FROZEN 2026-08-10 (F-77):
`Pick up the blue thermal pad and place it on the red target RAM board.`**

Code copy: `contracts.TASK2_INSTRUCTION`; `camelo/cli.py DEFAULT_TASK`
reads it, so a bare `make eval` sends it. Colour-grounded phrasing,
adopted to sit closer to the zero-shot headline runs.

**Two constants now, deliberately:**

| Constant | Meaning |
|---|---|
| `RECORDER_DEFAULT_CAPTION` | `Pick up the thermal pad and place it on the target RAM board.` — a **mirrored benchmark fact**: the caption any recording carries unless `--single_task` overrides it (`record_task2.py`). Drift-alarmed. Not ours to choose. |
| `TASK2_INSTRUCTION` | the caption **we** train and evaluate with. Ours to choose. |

They were one constant until F-77. While they differ, **every corpus
arrives captioned with the recorder string and must be re-captioned
before training** (`dataset_tools.modify_tasks`), or the fine-tune learns
one string while eval sends the other — the F-68 skew from the recording
side. `camelo.train.train` warns on exactly that mismatch, and
`ext_fixpos200_*` were re-captioned at build time.

F-68 originally found three strings circulating (the recorder default;
the old repo default "…liner side up."; the zero-shot E1/E2 colour
phrasing `pick up the blue thermal pad and place it on the red target`).
Note the current string is a **hybrid** — colour-grounded like the
zero-shot prompt but retaining "RAM board" — so it is not literally the
zero-shot phrasing either; a zero-shot-vs-fine-tuned comparison still
varies prompt unless the zero-shot rung is re-run at this string.
Fine-tuned checkpoints are locked to their training caption while
pretrained VLAs can adapt. **Checkpoints trained before 2026-08-10 are
locked to `RECORDER_DEFAULT_CAPTION`** and must be evaluated with
`--task` set to it explicitly. Zero-shot prompt variants are fine as
labelled sensitivity rows, never as the headline number.

**Bridge safety on the sim side: THERE IS NONE — corrected 2026-08-07
(F-53).** This file previously claimed "command smoothing alpha 0.08, max
step 0.008 rad @ 60 Hz (~0.48 rad/s slew)". Those numbers exist only as
constants nobody reads (`task1_isaacsim/scripts/isaac_bridge_constants.py:136-137`),
and the embodiment actually in use explicitly disables all three
(`assets/embodiments/fr3duo_mobile/embodiment_config.yaml:23-26`:
`command_smoothing_alpha: 1.0`, `max_position_step_rad: 0.0`,
`position_deadband_rad: 0.0`). The task2 bridge core contains **zero**
occurrences of smoothing/slew/deadband; `apply_commands` writes the raw
target straight into an `ArticulationAction`.

Consequence: **the chunk executor's per-tick delta clamp (0.05 rad @ 20 Hz)
is the ONLY rate limiter in the whole command path**, not a conservative
inner bound on a sim-side envelope. Do not weaken it on the assumption that
the bridge will catch anything, and do not cite a 0.48 rad/s slew when
budgeting motion.

**Chunk execution under asynchronous inference (`--async-inference`,
docs/realdata/16 U-29).** Synchronously, `backend.infer` runs *inside* the
control loop, so a remote round trip is dead time: measured on the rig
2026-09-02, a 0.45 s median RTT against a 0.4 s replan interval left **90
ticks in 30 s — 3 Hz, not 20**. With the flag on (default for `--backend
remote` on `--world real`; off for local and sim, which stay bit-identical)
`camelo.control.async_inference.AsyncInference` runs the call on one worker
thread, **one request in flight at a time, never queued**, and the loop
polls each tick. Four semantics follow, and none of them weakens the clamp
above. (1) A chunk is installed at **its own `t0`** — the sim time of the
observation that produced it — so the executor's `(t_sim - t0)/dt` indexing
starts it at the step the round trip already consumed rather than replaying
stale steps: `chunk_arrival_index_max` reports that step (≈ RTT/dt, plus up
to one tick because a reply is noticed at the next poll). (2) The next
request goes out `--replan-steps` after that `t0`, **never after the
arrival** — so when the trigger (8 steps) is shorter than the round trip (9)
the request leaves on the arrival tick and the real cadence is the RTT
itself; the outgoing chunk is then replaced around index `2 x RTT/dt`, i.e.
**~19 of ACT's 21**, about one step of margin. (3) When a chunk runs out
first, the loop **holds the last commanded target** and counts
`starved_ticks` — it never extrapolates, and never lets `step` keep ramping
toward a final row the policy stopped vouching for; held ticks publish the
same arm positions with the base twist zeroed (repeating a position is
standing still, repeating a velocity is not). (4) An exception in the worker
is re-raised in the control loop at the next poll, so `TimeoutError` /
`ConnectionClosed` / a server error still end the rollout through the
guarded `finally` and deactivate the arms. The U-23 stale-image guard is
evaluated at the **submission** point, on the observation being submitted.
`stale_chunks` is unchanged: a clock rebase still invalidates the chunk
inside `step`.

**Where an arriving chunk resumes — `--chunk-splice` (docs/realdata/16
U-31).** Semantics (1) above has a cost the first real rollout paid in
full. Installing a chunk at its wall-clock index presumes the arm
*executed* the elapsed steps, and behind the `--max-delta` clamp (plus the
companion's own 0.5 rad/s slew cap) it did not — so the spliced-in target
sits where the policy expected to be, not where the arm is. Measured
2026-09-03 (`t6_act_20260903_093858`, R-67/R-68): **all 70 of 70 chunk
arrivals demanded more than the clamp allows** — max |Δcmd| = 0.0500 rad
on every one — and the arm wiggled at the ~2.2 Hz chunk cadence instead of
descending. `--chunk-splice` chooses the hand-over: `index` is the old
behaviour and stays the default for every synchronous and sim run (byte
for byte); `nearest` — **the default whenever inference is asynchronous** —
rewinds playback to the point on the chunk closest to the MEASURED arm
pose (L2 over the 14 arm dims, the same ones `step` clamps, resolved to a
*fractional* index, because rounding to whole rows would leave up to half
a chunk step of jump and a chunk step is several times the clamp here),
stored as a per-chunk index offset so playback still advances at real
time; `offset` keeps the wall index and instead carries
`current_cmd − chunk[wall_idx]` as an additive arm correction decaying
linearly to zero over `--splice-ramp-ticks` (default 8) command ticks.
Grippers and the base twist are never corrected — offsetting a VELOCITY is
continued motion nobody asked for. Two invariants hold across all three:
`needs_replan` is judged on the **wall** index, never the spliced one (a
rewind does not make the observation behind the chunk any fresher, and
judging it on playback would walk the replacement index past the horizon
into starvation), while `exhausted` follows the **spliced** index, so a
chunk shorter than its offset still runs out and holds exactly as before.
Telemetry, per rollout and on each `policy chunk #N` line
(`splice=-6.0 jump=0.012`): `arrival_jump_rad_mean` / `_max` — the
pre-clamp discontinuity the arm is asked to swallow at each arrival, so
the number stays directly comparable to `--max-delta` and across policies
— and `splice_shift_mean` / `_max`, the chosen index minus the wall index
(negative = rewound), with `spliced_arrivals` and the raw sums beside them
so every aggregate can be re-derived. **Unmeasured on the rig.**

**The command leash — `--leash-rad` (docs/realdata/16 U-32).** The clamp
above bounds how fast the arm command may MOVE; nothing bounded how far it
could stand from the pose the arm actually holds, and behind the
companion's 0.5 rad/s slew — *half* the clamp — it drifted ahead until
every hand-over was spliced against a target no joint occupied.
`--leash-rad` clips the arm command to `measured ± leash_rad` per joint
**after** the clamp (**0.15 rad on `--world real`**, about 0.3 s of that
slew; OFF in sim, where the bridge applies our targets with no slew limit
of its own and every result on record stays byte-identical; `0` is the
explicit opt-out on real). Grippers and the base twist are never leashed,
for the same reason they are never spliced. Two things follow. `nearest`
anchors its search at that same MEASURED pose rather than at the last
command — the chunk's row 0 *is* the pose the policy observed, so the
matching row sits at or just before the wall index — bounded forward to
`wall_idx + 2` rows, with near-ties (within 1e-3 rad) resolved toward the
wall index instead of toward an arbitrary row, because a hovering policy's
rows are all within float noise of each other. And the leash is a
*following* limit: an arm that genuinely cannot move now stops the command
advancing rather than winding up ahead of it, so a high `leash_active_pct`
is a statement about the CONTROLLER, not about the policy. Measured
2026-09-03 (`t6_act_20260903_101739`) with neither in place: every splice
shift positive (mean **+11 of 21**, max +15) and **413 of 587 ticks held**.
Telemetry: `leash_active_ticks` / `leash_active_pct`, `cmd_lead_rad_max`
(recorded whether the leash is on or off — with it off it is the diagnostic
that says how far the command had run ahead), `leash_rad` beside them so
`0.0` cannot be read as "never needed", `lead=` on each `policy chunk #N`
line and a `leash telemetry:` line at the end of the run. **Unmeasured on
the rig.**

## Observing (what we subscribe)

| Topic | Type | Notes |
|---|---|---|
| `/isaac/clock` | `rosgraph_msgs/Clock` | sim time; **never `/clock`** (a leaked USD publisher sits at 0.0 there); rebases to ~0 on scene reset |
| `/isaac/joint_states_full` | `JointState` | full articulation, positions+velocities |
| `/isaac/applied_joint_commands` | `JointState` | post-arbitration targets = the recorded action source |
| `/isaac/odom` | `nav_msgs/Odometry` | world pose + body twist |
| `/isaac/cmd_vel_applied` | `geometry_msgs/Twist` | applied base twist |
| `/isaac/{left,right}_ee_pose` | `PoseStamped` | `*_fr3v2_link8`, world |
| `/isaac/head_camera/image_raw` | `Image` 1280x720 | **BEST_EFFORT QoS mandatory** |
| `/isaac/head_camera/camera_info` | `CameraInfo` | live pinhole `K`; **BEST_EFFORT**. Sim USD default is ~60° HFOV square pixels, **not** the yaml ZED Mini 90°×60° |
| `/isaac/{left,right}_wrist_camera/image_raw` | `Image` 848x480 | encodings vary (`rgb8`/`bgr8`), row stride matters — use `contracts.image_msg_to_array` |
| `/isaac/task2/scene_reset` / `_request` | `String` | reset ack / trigger (needs sim `--record`) |

All of the above require the sim launched with `--record`.

## Real robot (TMR station)

Same 20-dim action / 37-dim state. Topic **names** are the hardware-verified
list in `record_bag.bash` (LABS manifest + 2026-08-23/30 corrections), not
`/isaac/*` + `/bridge/*`. Select with `--world real` on `launch_policy.sh`
/ `run_policy.py` (or `CAMELO_WORLD=real` / `make … WORLD=real`).

| Role | Topic | Type |
|---|---|---|
| left/right arm state | `/left\|right/franka_robot_state_broadcaster/measured_joint_states` | `JointState` (`fr3_joint1..7`) |
| gripper state | `/left\|right/gripper/joint_states` | `JointState`, one joint `robotiq_85_left_knuckle_joint`, **RAW radians** (0.0 open, 0.7929 closed = `contracts.ROBOTIQ_CLOSED_RAD`), ~65 Hz |
| external wrench | `/left\|right/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame` | `geometry_msgs/WrenchStamped`, BEST_EFFORT — **12 of the s27a15 policy's 27 state dims** |
| spine state | `/spine/joint_states` | `JointState` |
| odom | `/swerve_drive_controller/odom` | `Odometry` |
| applied base twist | `/swerve_drive_controller/cmd_vel_out` | `TwistStamped` (assumed to mirror the command type — confirm with `ros2 topic type` on site) |
| head camera | `/head_camera/zed_node/rgb/color/rect/image` | `Image` BEST_EFFORT, ZED **1280x720** |
| left/right wrist | `/wrist_camera_{left,right}/camera/color/image_rect_raw` | `Image` BEST_EFFORT, D405 **640x480** |
| arm command (GELLO stand-in) | `/left\|right/gello/joint_states` | `JointState` (**`left\|right_fr3v2_joint1..7`**) |
| gripper command | `/left\|right/gripper/gripper_client/target_gripper_width_percent` | **`std_msgs/Float32`**, **0..1 OPEN FRACTION** (1.0 = open, 0.0 = closed) — the name is a misnomer, **MEASURED 2026-09-02**, see below |
| base command | `/swerve_drive_controller/cmd_vel` | **`TwistStamped`**, a fresh `header.stamp` per publish from the node clock: the SwerveDriveController ages each message by that field and substitutes **zeros for anything older than 0.5 s** (`cmd_vel_timeout`) — an unstamped `Twist` or `ros2 topic pub` never moves the base. Clamps **0.1 m/s / 0.1 rad/s**, ramps 0.1 m/s²; publish at 20 Hz, zeros on exit (source: the site's `base_nudge.py`; `contracts.REAL_BASE_*`). Message type follows the site bridge — **confirm with `ros2 topic type` on site** |
| spine command | `/spine/target_height` | (held; not a policy output) |

**Measured on the rig 2026-09-01** — three facts that each fail SILENTLY,
frozen here because every one of them looks like "the robot ignored us":

* **Arm command joint names are prefixed.** The measured topics carry
  unprefixed `fr3_joint1..7` (the side is the namespace), but the
  companion's `joint_impedance_controller` validates every incoming GELLO
  JointState against its own joint list (`validateGelloJointState_`), which
  is `left_fr3v2_joint1..7` / `right_fr3v2_joint1..7` — the same names the
  sim contract uses. Any other name is rejected and `switch_controller`
  answers `ok=False`, with nothing logged on the publishing side.
  `camelo.contracts` therefore publishes `LEFT_JOINTS`/`RIGHT_JOINTS` and
  keeps `REAL_ARM_JOINTS` for the *measured* names only.
* **The gripper command is `std_msgs/msg/Float32`,** not `Float64`. The
  rig's `robotiq_gripper_client.py` subscribes `Float32`, and the station's
  own GELLO publisher publishes `Float32`. A `Float64` publisher is simply
  never matched — no error, no motion.
* **Camera resolutions differ from sim.** The wrists are RealSense **D405
  at 640x480** (not the sim's 848x480); the head ZED is 1280x720. Only the
  camera *keys* are shared between the two worlds, never the shapes.
  **MEASURED 2026-09-02: the site's `start_cameras.bash` (`pixi run
  cameras`) has launched the ZED at `resolution:=VGA` since commit `646ecda`
  (2026-08-31) — the head topic now reads **672x376**, not 1280x720, on this
  rig (`docs/realdata/16` R-42/R-44). The contract shape stays **1280x720**
  — that is the s27a15 corpus / training resolution, not a description of
  whatever the launcher happens to emit — so the launcher must be fixed to
  match, not the contract. A live shape-drift alarm on real (`check_obs`
  exit 3, `run_policy` refusal unless `--allow-camera-shape-mismatch`) was
  added 2026-09-02 so a VGA frame can no longer flow silently into a policy
  trained on 1280x720. **Same day, same finding, both wrists**: T3
  (`t3_checkobs_130618`) measured `wrist_left`/`wrist_right` at **848x480**,
  not the contracted 640x480 — the launcher's own
  `depth_module.color_profile:='640x480x30'` argument is evidently not
  taking effect, so the RealSense falls back to its default color profile
  (848x480, `docs/realdata/16` R-45/U-21). The contract shape stays
  **640x480** (the corpus resolution); the launcher must be fixed to match,
  same as the head camera.

No `/isaac/clock` — `t_sim` is ROS time (wall). No EE-pose / scene-reset /
object-pose / pad-points topics, so `state[0:14]` is NaN; the 16-dim model
proprio (arms + grippers) is unchanged. Stop GELLO teleop before a policy
run or the two publishers fight on the gello topics.

### The gripper command unit — MEASURED 2026-09-02 (T1(c))

**`target_gripper_width_percent` is a 0..1 OPEN FRACTION, not a 0..100
percent.** The unit was *measured* — read off the rig's own driver stack —
not inferred from the topic name, which is wrong. `contracts.
gripper_width_percent()` keeps the misnomer as its name and emits 0..1.

* **The only subscriber** is `franka_gripper_manager/
  robotiq_gripper_client.py:27-35` (companion Jetson). It dedups —
  `if abs(target - self.last_width) < 0.02: return` — then sends
  `goal.command.position = 1 - gripper_position` with `max_effort = 1.0`.
  A wire value above 1 therefore becomes a NEGATIVE position.
* **One `GripperCommand` action goal per accepted message**, and a new goal
  **preempts** the one in flight. At 20 Hz every tick clears the 0.02 dedup,
  so the driver never finishes a motion: T1(c)'s companion log holds 8011
  `GripperCommand_Result(position=0.0, reached_goal=False)` lines, all of
  them preempt-cancelled *default* results — not feedback, and not evidence
  about position.
* **`GripperCommand.position` is knuckle radians**, 0 open to **0.7929**
  closed (`ROBOTIQ_CLOSED_RAD`; `robotiq_description/urdf/
  2f_85.ros2_control.xacro:18`, `robotiq_driver/src/
  hardware_interface.cpp:276,297-299`, register = `pos/0.7929*227+3` clamped
  to [0,255]). Nothing in `gripper_action_controller` clamps the goal.
  Feedback comes back on `/left|right/gripper/joint_states` at ~65 Hz.
* **Corroboration.** The station's own `gello_publisher.py` publishes
  `gello_hardware.process_gripper_position()` — documented "percentage
  (0-1)", clamped to [0,1] — at 30 Hz unconditionally, and the corpus column
  `…target_gripper_width_percent_value` spans **0.2025..1.0**
  (docs/realdata/01_DATASET_ANALYSIS.md:491). Nothing above 1 was ever
  recorded.
* **What the old 0..100 reading cost.** Publishing 100.0 / 30.0 mapped to
  `1 - 100 = -99 rad` / `-29 rad`, both clamped to register 0 = **fully
  open** — so the gripper sat open at *both* ends of the T1(c) sine while
  every wire value looked plausible. R-34 in docs/realdata/16 records the
  symptom ("grippers do not close at all"); this is the cause.

Note that `contracts.GRIPPER_CLOSED_RAD` (0.8) is a *different* constant: it
mirrors the sim recorder and is the frozen D7 corpus denominator the
checkpoints were trained with. The real driver's 0.7929 lives in
`ROBOTIQ_CLOSED_RAD` and must never be substituted for it.

#### Verified live 2026-09-02 (t1c_115741)

**The mapping above is now confirmed on the rig, not just derived.** Native
pixi probe `t1c_115741`, 11:58–12:01 CEST (`--arms right --gripper-span 0.7
--hold-s 40 --seconds 150`, arms **not** activated, deployed commit
`f049584` with `gripper_width_percent` now emitting 0..1), ran to completion
(189 rows, 149 wiggle rows, CSV + summary written, 0 faults): the grippers
**follow**. Per-second sample rows (grip cmd → measured open-fraction L / R,
our feedback = 1 − knuckle_rad/0.8): 0.99→1.00/1.00; 0.88→0.89/0.89;
0.68→0.66/0.65; 0.46→0.38/0.38; 0.33→0.18/0.19; 0.31→0.14/0.14;
0.43→0.25/0.25; 0.63→0.47/0.48; 0.84→0.76/0.75; 0.98→0.95/0.95 — a second
cycle matched within 0.02. Over the full run: measured range L/R 0.13..1.00
for cmd 0.30..1.00; **corr(cmd, measured) = 0.991 at lag 0 on both sides**
(0.875 at 1 s lag — the gripper follows within the 1 s row cadence).
Gripper feedback rate `gjs_hz` = **100.0 Hz, both sides, every row** (R-19
in doc 16 now carries the live rate; the bag's ~65 Hz stands as the bag's
own reading). Companion log for the full run: **+1782** accepted goals over
150 s (span 0.7 → wire moves ≈0.14/s → ≈7 goals/s per side against the 0.02
dedup — an order of magnitude above the old 0..100 stream's per-tick flood,
R-34/R-36, but expected for this span, see the acceptance-formula note in
doc 16 R-36/T4), `reached_goal=True` 651, result positions non-zero (0.066,
0.084, 0.105, 0.133 rad …). Operator: "grippers are also closing".

**The map.** The driver commands knuckle position = `1 − x` rad (`x` = wire
value, `robotiq_gripper_client.py:34`), clipped to the driver's 0.7929; our
feedback reads `open = 1 − rad/0.8`, so at steady state
`measured_open ≈ (x − 0.2)/0.8` — confirmed over the full run,
**mean |measured − (x − 0.2)/0.8| = 0.037 (L) / 0.036 (R)** (x=0.31 → 0.1375
predicted, 0.14 measured; x=0.88 → 0.85 predicted, 0.89 measured; mid-swing
rows lag more because the gripper is still travelling).

**Frozen facts:**

1. Command 1.0 = open (0 rad); command 0.2 = fully closed (0.8 rad →
   clipped to 0.7929, register 255). Commands below 0.2 are all "closed".
2. This is exactly the corpus command range 0.2025..1.0
   (`01_DATASET_ANALYSIS.md:491`) — the s27a15 policy output goes on the
   wire verbatim, and 0.2 from the policy means "close".
3. T4 item (3) of doc 16 ("open_fraction = 0.2 → it closes") is confirmed
   by mechanism and by these rows. T4 items (1) and (2) still need the live
   closed-on-object number: U-5 is only partially answered — an empty
   close reached measured open 0.14 (0.69 rad) at cmd 0.31, but a cmd ≤ 0.2
   close was not commanded today.
4. The command unit and the state unit are **not** the same scale (command
   `x` vs state `(x − 0.2)/0.8`). Any code that compares a commanded open
   fraction with the measured one (start-pose verification, the parity
   tool, `verify_gripper_polarity`) must go through this map — flagged as
   U-15 in doc 16.

### The `joint_impedance_controller` contract — MEASURED 2026-09-01

The companion controller on the arm command topic. Every line here changes
what the runner must do, and none of them raises on the publishing side:

* it **ingests while inactive**, and `on_activate` **REFUSES** unless a
  valid GELLO sample arrived within `max_gello_liveness_gap` = **2.0 s**;
* at that instant it captures `q0` (measured robot pose) and `g0` (the
  command it was holding) and then maps
  **`q_goal = q0 + dir*(g - g0)`** with
  `gello_joint_directions = [-1,-1,1,1,1,1,-1]`, rate-limited at 0.5 rad/s;
* it **rejects** samples stamped more than **0.5 s** in the past by ITS clock;
* if no valid sample arrives for **2.0 s while ACTIVE** it calls
  `rclcpp::shutdown()` — **the whole arm launch dies**.

Activation and deactivation go through
`/left|right/controller_manager/{list_controllers,switch_controller}`
(`controller_manager_msgs`). The station's `activate_arms.py` waits for
**both** sides' services to be discovered **before** switching either: a new
DDS participant appearing mid-session is a discovery burst that can fault a
live FCI loop. For the same reason every client this repo creates lives on
the runner's **own** node — never a second participant, never
`ros2 service call`.

Four consequences, all implemented (`camelo/ros/arm_activation.py`,
`camelo/runner/real_arms.py`, `camelo/ros/command_publisher.py`):

1. **Hold before activate.** After the first complete observation the runner
   publishes the **measured** pose at the tick rate — a zero-delta GELLO
   stand-in — until `list_controllers` reports the controller `active` on
   every `--arms` side. This satisfies the liveness gate AND makes
   `g0 == q0`, which is the only way the command frame below is knowable
   from this side. `on_activate` fires inside the controller, so the runner
   learns of it up to one `list_controllers` poll (0.5 s) later: the
   reference is therefore **bounded, not verified**. `CommandPublisher`
   keeps every arm publish of the last 2 s and, at capture, measures both
   the per-joint **spread** (max−min) of what was published inside
   `poll + 0.1 s` and the **delta** between the newest publish and the
   measured pose. Above **0.01 rad** either one means `(q0, g0)` are not one
   pose: a WARNING in the `robot` frame (absolute targets — the reference
   changes nothing on the wire), and a **refusal** in the `gello` frame,
   where a wrong reference silently reverses joints 1, 2 and 7. Both numbers
   land in the session stats (`activation_ref_spread_rad`,
   `activation_q0_g0_drift_rad`).
2. **Keep-alive.** `CommandPublisher` republishes the last arm + gripper
   command with a **fresh stamp** whenever nothing was published for
   `1/--keepalive-hz` (default 10 Hz), so a remote-inference stall never
   reaches 2.0 s. **Never on sim.** It runs on its **own daemon thread**,
   never a ROS timer: on the session executor it shares a pool with the
   camera drain callbacks and was measured at ~5 Hz under three cameras
   (docs/realdata/16 U-28). The quiet-check and the publish are taken under
   one lock — as is every ordinary arm publish, on BOTH sides, each message
   stamped from its own clock read in the same call (a `--arms right` run
   still publishes the frozen left arm, and it is re-stamped every tick).
   Per rollout the runner reports `keepalive_republished`,
   `max_publish_gap_s` (the worst gap between two arm publishes — the margin
   actually left on the 2.0 s fault), `keepalive_max_interval_s` (the
   watchdog's OWN worst period, which the gap above cannot show because the
   repeats are what keep it small) and `activation_to_first_cmd_s`; and
   `run_policy.py` prints all four when the rollout ends.
   The real arm command publishers use **queue depth 1**, not
   `DEFAULT_DEPTH`: KEEP_LAST(10) at the 20 Hz tick is exactly 0.5 s of
   writer history — the reject limit above — so a reader that stalls for a
   moment is handed a backlog whose oldest sample is already too old
   (MEASURED 2026-09-02 17:41, U-28). Depth is not part of QoS
   compatibility, so this cannot unmatch the companion; sim is unchanged.
   Before the activation switch the runner also refuses to proceed on a
   stream slower than `--min-activation-publish-hz` (8 Hz), and pays the
   backend's first (slow) inference **before** the switch rather than after
   it — both U-28.
3. **Deactivate, confirm, then stop.** On every exit path (exceptions
   included) the runner deactivates while still publishing, waits up to 10 s
   for `list_controllers` to confirm, and only then `safe_stop()`s.
4. **`--arm-command-frame`.** `robot` (default) publishes the policy's joint
   targets as-is; `gello` pre-inverts the controller's map, publishing
   `g = g0 + dir*(q_target - q0)`. Offline the corpus's 15-dim action is in
   the ROBOT frame (slope +1 vs measured state on all 14 joints), so
   `robot` is the default — **T1b on the rig decides** whether the
   controller-side transform must be undone, which is why it is a switch and
   not a constant. Joints 1, 2 and 7 are where the two differ.
   `--arm-command-frame` default `robot` — **MEASURED 2026-09-02 T1b
   (docs/realdata/16 R-39):** joint 1 in phase with an identity publish
   (`t1bj1_120702`, corr(cmd, meas) = +0.922, max |meas| 0.0173/0.02 rad) —
   the controller-side transform is NOT undone by the publisher; `robot`
   is confirmed, not assumed.

### Real-robot run flags

| Flag | Default | What it does |
|---|---|---|
| `--start-pose file:<json>` | required on real (unless `--no-start-pose-check`) | **Verify-only**: `recenter_arms()` commands NOTHING on real. `benchmark`/`task2` are SIM poses and are **always** refused here — `--no-start-pose-check` waives the check, not the sim-pose semantics. Lift the file from the teleop home pose / corpus frame 0 |
| `--start-pose-tol` | `0.10` rad | max per-joint error before the run refuses to start |
| `--no-start-pose-check` | off | log the per-joint errors and run anyway; with `--start-pose` OMITTED it also lifts the requirement to pass one. It never makes a SIM pose acceptable |
| `--arms left,right` | both | which arms this run drives; an unselected arm is **held at its measured pose** for the whole run |
| `--wait-for-activation` | **on for real** | hold + poll `list_controllers` until active on every selected arm |
| `--activate-arms` | off | do the switch here, activate_arms.py's way (all four services first, then both arms, best-effort then strict) |
| `--deactivate-on-exit` | on | deactivate + confirm before the stream stops |
| `--keepalive-hz` | `10.0` | republish rate of the last command; `0` disables. Runs on its **own daemon thread**, not a ROS timer — on the executor it was starved to ~5 Hz by the camera drain callbacks (docs/realdata/16 U-28). Reports `keepalive_max_interval_s` |
| `--min-activation-publish-hz` | `8.0` | REFUSE to switch `joint_impedance_controller` on when the command stream published fewer than this many samples/s over the activation window; raises `PublishCadenceTooLow` **before** the switch, so nothing is switched. The measured cadence is logged either way. `0` disables (docs/realdata/16 U-28) |
| `--arm-command-frame` | `robot` | `robot` \| `gello` (see above) |
| `--gello-joint-directions` | `-1,-1,1,1,1,1,-1` | the measured `dir`; 7 values of ±1 |
| `--no-wrench` | off | DECLARE that this rig has no wrench: zeros + `wrench_declared_absent`, never zeros by omission |
| `--async-inference` | **ON for `--backend remote` on real**, off for local and in sim | run `backend.infer` on its own thread so the 20 Hz loop never blocks on the round trip (docs/realdata/16 U-29: 0.45 s RTT → 90 ticks in 30 s before this). Semantics — arrival index, hold-on-starvation, exception surfacing — in "Chunk execution under asynchronous inference" above. Telemetry: `ticks` (now ≈ rate × sim seconds), `inferences`, `starved_ticks`, `dropped_requests` (0 unless an observation was thrown away), `chunk_arrival_index_max`, `infer_median_s` / `infer_max_s` |
| `--wire-image-size WxH` / `--wire-jpeg-quality N` | off (native resolution) / `90` — byte-identical to every request on record | `--backend remote` ONLY (a local run encodes nothing and the flags are refused there). Resize each camera on the CLIENT, before the JPEG, with the same op the server's own preprocessor would apply — `torchvision.transforms.v2.Resize` on the uint8 frame where the client has torch, PIL bilinear otherwise (measured max pixel difference **1/255**, mean 0.0005, on `outputs/rig/t5/ep163_frames`; the server's `v2.Resize` is then the identity on an already-correct frame). Spelled `WxH` for every camera or `head=WxH,wrist_right=WxH` per camera; a camera outside `--cameras` is refused rather than ignored. Pass the size the CHECKPOINT declares — the client cannot read its `train_config.json`. Measured payload for the rig's three cameras: **~116 kB** full-resolution, **~95 kB** for `--cameras head,wrist_right`, **~17 kB** adding `--wire-image-size 224x224` — the VLA-JEPA combination, whose 7-row chunk (0.35 s) is shorter than the ~0.24 s transport it replaces. Caveat: the resize is faithful, but the JPEG now lands at 224x224 instead of at native resolution, where the server's downscale used to average its artifacts away — measured mean 1.2 / max 36-42 counts against the full-resolution pipeline at the same quality, mean 0.82 / max 24 at `--wire-jpeg-quality 95` (25 kB). Telemetry: `wire_bytes_per_request`, `wire_encode_s`, and one startup line per camera (native -> wire shape, bytes before -> after). **Unmeasured on the rig** |
| `--chunk-splice {index,nearest,offset}` | **`nearest` when inference is asynchronous**, `index` otherwise (i.e. every synchronous and sim run, byte-identical) | where an arriving chunk resumes playback. `index` = the wall-clock index `(t_sim - t0)/dt`, which on the rig cost a maximal clamped jump at **70 of 70** arrivals (docs/realdata/16 R-68). `nearest` rewinds to the fractional index whose arm target is closest to the MEASURED arm pose (U-32; bounded to `wall_idx + 2`, near-ties resolved toward the wall index); `offset` keeps the wall index and ramps the difference off instead. Grippers/base never corrected. Semantics and the two index invariants in "Where an arriving chunk resumes" above. Telemetry: `arrival_jump_rad_mean`/`_max`, `splice_shift_mean`/`_max`, `spliced_arrivals`, and `splice=`/`jump=` on each `policy chunk #N` line. **Unmeasured on the rig** |
| `--chunk-time-base {observation,arrival}` | `observation` (today's behaviour, byte-identical everywhere) | what sim time an arriving SYNCHRONOUS chunk's `t0` is stamped with. `observation` = the sim time of the observation `backend.infer` was handed, so a slow inference leaves the wall index already `infer_s/dt` steps in when the chunk installs and `step` skips that many rows of an arm that never moved. `arrival` (plan-then-execute, docs/realdata/16 T6) re-samples the collector right after `backend.infer` returns and stamps `t0` there instead, so playback starts at row 0 and `needs_replan`/`exhausted`/the clamp/the leash/the chunk dump's `wall_idx`/`spliced_idx` all count from the arrival. Also accepted under `--async-inference`, where it simply re-bases the installed chunk to the arrival tick (harmless, but zeroes `chunk_arrival_index_max` — the flag is meant for the synchronous loop) |
| `--leash-rad` | **0.15 rad on real**, OFF in sim; `0` disables | how far the arm COMMAND may stand from the MEASURED joint pose, per joint, **after** the `--max-delta` clamp (docs/realdata/16 U-32). `--max-delta` bounds the command's speed, not its distance from the arm, and behind the companion's 0.5 rad/s slew the command ran ahead until `--chunk-splice nearest` spliced every arrival near the chunk's end: **413 of 587 ticks held** on 2026-09-03. Arms only — grippers and the base are never leashed. Semantics in "The command leash" above. Telemetry: `leash_active_ticks`, `leash_active_pct`, `cmd_lead_rad_max`, `leash_rad`, and `lead=` on each `policy chunk #N` line. **Unmeasured on the rig** |
| `--splice-ramp-ticks` | `8` (= 0.4 s at `--rate 20`) | `--chunk-splice offset` only: COMMAND ticks — not chunk steps — over which the hand-over correction decays linearly to zero. `0` disables the correction, making `offset` behave as `index` |
| `--max-image-age-s` | **`0.5` s on real**, off in sim | stale-image guard (docs/realdata/16 U-23): at every INFERENCE (the ~3.75 Hz replan cadence, not the 20 Hz tick) each camera's newest frame must have arrived within this many WALL seconds, or the rollout raises `StaleImageError` — deactivate, log, non-zero exit. `ObsCollector` never clears a cached frame, so without it a camera that dies mid-rollout feeds the policy the same image forever. `0` disables |
| `--max-state-age-s` | **`0.2` s on real**, off in sim | stale-JOINT-STATE guard (docs/realdata/16 U-27): on EVERY control tick — and once before activation, in `RealArmSession.prepare` — each arm's and each gripper's newest `JointState` must have arrived within this many WALL seconds, or the rollout raises `StaleStateError` — deactivate, log, non-zero exit. `ObsCollector` never clears a cached joint position, so without it a controller that shuts its arm launch down mid-rollout (R-64) leaves the policy — and `--joint-csv`'s `meas_*` columns — reading one frozen pose for the rest of the run. Both DRIVEN and HELD arms are watched; the spine is not (it has never come up on this rig). Joint states arrive at ~1 kHz per arm, so 0.2 s is ~200 missed messages, inside the companion's own 0.5 s `max_gello_message_age`. `0` disables |
| `--chunk-dump PATH` | off | diagnostic (`camelo.runner.chunk_dump`): archive every chunk that reaches `ChunkExecutor.set_chunk`, sync and async alike, to ONE `.npz` — chunk number, t0, sim time/wall/spliced index at arrival, the `jump`/`lead` the `policy chunk #N` line already prints, the observation state (`obs_state_at_inference` when reachable, else `obs_state_at_arrival` under `--async-inference` — the worker never returns the submitted observation, so arrival is the closest stand-in and the field name says which), the executor's last commanded arm vector, and the full `(H, ACTION_DIM)` chunk, plus a JSON sidecar (layout + this run's args) embedded in the same file. Off by default: byte-identical to every run on record. Never raises into the control loop — a write failure is logged, not thrown |

### `s27a15` observation and chunk plumbing

On `--world real` `ObsCollector` fills `Obs.rig` with the NAMED groups the
adapter expects — `left_arm(7)`, `right_arm(7)`, `right_gripper_rad` (the
**RAW** knuckle angle; the adapter owns `1 - clip(rad,0,0.8)/0.8`),
`left_wrench(6)`, `right_wrench(6)` — and returns **no observation at all**
until every one of them has arrived. The 37-dim `state` is still built
beside it, unchanged. A missing group must raise in the adapter, so it is
never zero-filled here; an exactly-zero wrench is real data on this rig
(4,359 of 121,828 corpus frames), which is why the guard is structural.

The policy's chunk is 15-wide and the executor eats canonical-20 rows, so
`camelo/policy/adapters/s27a15.py` (`chunk_for_executor`) widens it once, on
the way in: arms straight
across, **base 0**, **spine NaN (hold)**, left gripper = the measured
opening, right gripper = the model's own open fraction (the driver's units
are applied by `CommandPublisher`, so no conversion happens here). Both
mismatches raise — a 15-wide chunk under `canonical20`, a 20-wide chunk
under `s27a15` — because the widths differ but the meanings do not
overlap.

Franka `measured_joint_states` is BEST_EFFORT (sensor-data). A RELIABLE
subscriber is silently unmatched (`Last incompatible policy: RELIABILITY`)
and never sees the arm joints — real-world obs subscriptions use
`SENSOR_QOS` for that reason. Images stay BEST_EFFORT as in sim.

## Canonical vectors (= the recorded LeRobot dataset schema)

**action (20, float32)**: `[0:3]` base vx,vy,wz · `[3:10]` left arm ·
`[10:17]` right arm · `[17]`/`[18]` gripper open-fractions · `[19]` spine
height.

**observation.state (37, float32)**: `[0:7]`/`[7:14]` left/right EE pose ·
`[14:21]`/`[21:28]` arm joints · `[28]` spine · `[29:31]` grippers ·
`[31:34]` base odom x,y,yaw · `[34:37]` base vel.

**Model-facing contract** (`ai_data_contract.yaml`, embodiment
`franka_fr3_duo`): action = **16-dim** `[left_arm(7), left_gripper,
right_arm(7), right_gripper]`, ABSOLUTE joint rad, **16-step horizon**;
video keys `head, wrist_left, wrist_right` (that order); proprio =
flatten of the same 16 state keys. Base and spine are not model outputs —
`contracts.model_to_canonical_actions()` maps 16 -> 20 (base 0, spine
hold).

## Scoring (Task 2)

`ros2 service call /isaac/eval_camera/evaluate std_srvs/srv/Trigger '{}'`
(eval stack up first) -> JSON in `${ISAAC_DOCKER_ROOT:-~/docker/ebim-challenge}/eval-task2/evaluate/`:
`iou_thermalpad_vs_target_current`, `is_orientation_correct`,
`orientation_case`. Task score = Pick Success x Orientation x IoU. This
in-repo scorer is a development facilitator; official scoring is the
competition page's.

## Grasp envelope (Task 2)

**MEASURED on the DGX, 2026-08-16** (DGX_FINDINGS.md F-98) and **frozen at
45 mm / 7 deg**, 2026-08-25. `tools/dgx_probes/grasp_pose_envelope.py` FKs
the right TCP at each episode's own grasp frame across 22 demos: TCP-to-pad
distance min 27.9 / median 33.1 / **max 36.8 mm**, orientation vs the median
demo grasp max **6.96 deg**.

The shipped threshold is **45 mm**, not the 36.8 mm the demos reach, because
a learned rollout was observed at **41.8 mm** — the case the gate exists to
catch. That number comes from a *rollout*, not from the demo set, so the
generator cannot derive it: its `suggested_gate.max_dist_mm` is
`ceil(36.77/5)*5 = 40`, which **excludes the 41.8 mm approach**. The gate
therefore takes its thresholds from `camelo/control/grasp_gate.py`'s
`DEFAULT_MAX_DIST_M` / `DEFAULT_MAX_ANGLE_DEG` and only its *reference
attitude* from the envelope JSON (`grasp_gate.envelope_kwargs`); a payload
whose `suggested_gate` disagrees is announced on stderr, never silently
honoured.

Until 2026-08-25 the code did the opposite and every `--grasp-gate` run
fired at 40 mm while documenting 45. Both the gate and the passive detector
(`GraspObserver`) must be built from the same envelope — see
`observer_from_envelope` and `--grasp-envelope` — or a gate-on vs gate-off
comparison measures its two arms at different bars, in millimetres *or* in
degrees.

**`--gripper-latch CLOSE_BELOW[:HOLD_S[:RELEASE_ABOVE]]`** (default off,
`camelo/control/gripper_latch.py`) is a separate, simpler rig-side lever for
the sibling gripper bug: the ACT checkpoint's channel copies its own state
and re-opens to ~0.45 after a full close (measured
`t6a_a03_150950`, 25.8-28.8 s closed then 0.44-0.7 for 90 s). Unlike the
grasp gate it needs no pad pose — it only watches the policy's own
right-gripper command — so it works where `pad_observed_ticks: 0` (the real
rig). Below `CLOSE_BELOW` (0.35) it holds the published value at
`min(policy, 0.20)` for `HOLD_S` (30 s) and thereafter until the policy
commands above `RELEASE_ABOVE` (0.9) for 1.0 s continuously. Left gripper
untouched. `--gripper-latch-near POSE_SPEC:RADIUS_RAD` (default off) further
gates the ENGAGE transition on the measured right arm being within
`RADIUS_RAD` (L2, rad, 7 joints) of a reference pose (`file:PATH`,
`slot:xNNN` from `outputs/rig/munich_2026-09-01/slot_grasp_poses.json`, or
`list:a,b,c,d,e,f,g`) — a04 (`t6a_a04_154452`) showed the plain threshold
firing on the checkpoint's pre-shape dip 0.22 rad from the grasp pose.

## Task 1 Isaac (later)

Same robot, same `/bridge/*` command contract. Differences: cameras are
JPEG `CompressedImage` @ 10 Hz, proprio comes from
`/isaac/data_contract/*` (19-dim action, no spine), no LeRobot recorder,
no local scorer (ManipulationNet client, currently on hold upstream).
Upstream "full run completable" is pending (benchmark issue #15) — we
stay on Task 2 until that lands.
