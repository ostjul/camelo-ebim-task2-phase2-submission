"""Shared CLI plumbing for the policy-facing scripts."""

from __future__ import annotations

import argparse
import logging
import math

from camelo.contracts import TASK2_INSTRUCTION, WORLD_REAL, default_world
from camelo.control.arm_stream import MIN_ACTIVATION_PUBLISH_HZ
from camelo.control.chunk_executor import DEFAULT_LEASH_RAD
from camelo.control.image_age import DEFAULT_MAX_IMAGE_AGE_S
from camelo.control.state_age import DEFAULT_MAX_STATE_AGE_S

# numpy-only sibling (no torch / lerobot / ROS), so the choices below come
# from the one place the conventions are defined rather than being retyped.
# `s27a15.LAYOUT` / `s27a15.CANONICAL_SPACE` are the two action-space
# declarations, and they live beside the chunk seam that dispatches on them.
from camelo.policy.adapters import s27a15
from camelo.runner.real_arms import START_POSE_TOL_RAD

# Defaults for the real-robot publisher flags. Imported from the module that
# implements them rather than retyped — `camelo/ros/command_publisher.py`
# needs rclpy, so only these three names are lifted, at import time, from a
# constants-only block at the top of that file.
ARM_COMMAND_FRAMES = ("robot", "gello")
DEFAULT_GELLO_JOINT_DIRECTIONS = (-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0)
DEFAULT_KEEPALIVE_HZ = 10.0

log = logging.getLogger(__name__)

DEFAULT_TASK = {
    "task2": TASK2_INSTRUCTION,  # frozen, F-68 — see docs/CONTRACTS.md
    "task1": "Route the cable across the fixture board and plug it in.",
}


def add_policy_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--world",
        choices=("sim", "real"),
        default=default_world(),
        help="topic contract: sim = /isaac/* + /bridge/* (benchmark); "
        "real = TMR station topics from record_bag.bash. Also CAMELO_WORLD. "
        "real skips the sim approach gate and the benchmark drift check",
    )
    parser.add_argument(
        "--adapter",
        default="dummy",
        help="dummy | replay (open-loop demo chunks from task2_fixpos_200) | "
        "heuristic (demo-sampled trajectory + right-arm residual IK, on by "
        "default) | molmoact2 | pi05droid (droid8 arm-splitter rungs, F-58) | "
        "gr00t (N1.7-DROID, lerobot-native) | gr00t-base | xvla | pi0 | "
        "smolvla | lerobot:<repo_or_path> | gr00t-isaac (default: dummy)",
    )
    parser.add_argument("--checkpoint", default=None, help="override the adapter's checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--gr00t-embodiment",
        default=None,
        help="GR00T embodiment head — the main zero-shot lever (F-41). "
        "Default: per-checkpoint (DROID -> its single oxe_droid head; 3B base "
        "-> the dual-arm xdof head); gr00t-isaac defaults to new_embodiment",
    )
    parser.add_argument(
        "--action-layout",
        choices=["model16", "canonical", "s27a15", "droid8_right", "droid8_right_delta"],
        default=None,
        help="Declared meaning of the checkpoint's action vector (F-45: never "
        "inferred from width). 'canonical' ONLY for checkpoints trained on our "
        "recorded 20-dim contract; 's27a15' = the Munich real-robot 15-dim "
        "vector (pass it for --state-layout too); droid8_* = single-instance "
        "arm-splitter (right arm, F-58; _delta for joint-delta checkpoints); "
        "default: preset declaration, else model16",
    )
    parser.add_argument(
        "--camera-map",
        default=None,
        help="droid8 layouts only: override checkpoint-image-key -> camera, "
        "comma list (keys may be the last dot-segment), e.g. "
        "exterior_2_left=wrist_left. Default: per-checkpoint map "
        "(camelo/policy/adapters/droid8.py); unmapped features refuse to run",
    )
    parser.add_argument(
        "--state-layout",
        choices=["model16", "canonical", "s27a15"],
        default=None,
        help="Declared meaning of the checkpoint's state slot (F-63: never "
        "inferred from width). 'canonical' ONLY for checkpoints we fine-tuned "
        "on our recorded 37-dim contract — pass it together with "
        "--action-layout canonical; 's27a15' = the Munich real-robot 27-dim "
        "vector, and must be paired with --action-layout s27a15; default: "
        "model16, the coercion foreign pretrained checkpoints need",
    )
    parser.add_argument(
        "--gripper-command",
        choices=sorted(s27a15.GRIPPER_COMMAND_CONVENTIONS),
        default=None,
        help="s27a15 only: units the REAL gripper driver expects, from the "
        "policy's open fraction (1.0 = open). Default 'knuckle_rad' "
        "(rad = 0.8 * (1 - open)) — the sim-side inverse and the exact inverse "
        "of the transform the corpus was built with. VERIFY IT LIVE before any "
        "rollout (15_RIG_WINDOW_RUNBOOK.md §1.4): a flip reads as ABSENT, or as "
        "a perfectly-timed close that never grips",
    )
    parser.add_argument(
        "--allow-caption-drift",
        action="store_true",
        help="s27a15 only: permit a --task other than the corpus caption. Off "
        "by default because VLA-JEPA conditions on it and a retyped caption is "
        "silent train/eval skew (F-68)",
    )
    parser.add_argument(
        "--chunk-dt",
        type=float,
        default=None,
        help="seconds between chunk steps at the checkpoint's control rate "
        "(W5 rate-matched rungs). Default: auto — LeRobotAdapter derives it "
        "from the training dataset's meta/info.json fps via train_config.json; "
        "falls back to 1/30 when that is not resolvable. Pass explicitly to "
        "override (e.g. a checkpoint moved off the box its train_config.json "
        "points at)",
    )
    parser.add_argument("--backend", choices=["local", "remote"], default="local")
    parser.add_argument("--server", default="ws://localhost:8765", help="remote backend URL")
    parser.add_argument(
        "--dummy-obs",
        action="store_true",
        help="feed the policy a FIXED, fake observation (all-zero joints/wrench, "
        "mid-grey images matching --action-layout/--state-layout) instead of the "
        "collector's real one — everything else (real measured state for the "
        "start-pose check, the executor's max-delta clamp, keepalive, "
        "deactivate-on-exit) still runs on REAL sensor data, so the robot WILL "
        "move on whatever the checkpoint predicts from that fake input. For "
        "verifying the command-publish path end to end before real sensors are "
        "wired in — see scripts/run_dummy_client.py for the no-robot version",
    )
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK["task2"],
        help="language instruction handed to the policy (freeze it: train == eval)",
    )
    parser.add_argument("--rate", type=float, default=20.0, help="command tick rate [Hz]")
    parser.add_argument(
        "--replan-steps",
        type=int,
        default=8,
        help="request a fresh chunk after this many consumed steps (chunk "
        "steps are 1/30 sim s apart by default, so 8 = 0.27 sim s of open "
        "loop). Counted from the current chunk's OWN t0 — the sim time of "
        "the observation that produced it — never from when it arrived, so "
        "under --async-inference the request leaves replan_steps*dt after "
        "that observation — or on the arrival tick, whichever is later, "
        "since one request is in flight at a time — and the reply lands "
        "round_trip/dt steps after that. A chunk is therefore replaced "
        "around index max(replan_steps, round_trip/dt) + round_trip/dt, "
        "which must stay inside the chunk horizon or the chunk runs out "
        "first and the loop holds (starved_ticks). On the rig that is "
        "max(8, 9) + 9 = 18 of ACT's 21, so lowering this below 9 buys "
        "nothing: the round trip, not this flag, sets the cadence there",
    )
    parser.add_argument(
        "--async-inference",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="run policy inference on its own thread so the control loop "
        "never blocks on it. ON by default for --backend remote on --world "
        "real, OFF everywhere else (a local backend blocks its own process "
        "either way, and sim results stay bit-identical). Measured "
        "2026-09-02 (docs/realdata/16 U-29): a 0.45 s round trip to an EC2 "
        "ACT server left the synchronous loop running 90 ticks in 30 s — 3 "
        "Hz instead of 20. Asynchronous, the loop keeps stepping the "
        "current chunk while the request is on the wire, starts an arriving "
        "chunk at the index the round trip already consumed, and HOLDS the "
        "last commanded target if a chunk runs out first (starved_ticks); "
        "it never extrapolates",
    )
    parser.add_argument("--max-delta", type=float, default=0.05, help="rad per tick per joint")
    parser.add_argument(
        "--chunk-splice",
        choices=["index", "nearest", "offset"],
        default=None,
        help="where an arriving chunk resumes playback (docs/realdata/16 "
        "U-31). Default: auto — `nearest` when inference is asynchronous, "
        "`index` otherwise, so every synchronous and sim result stays "
        "byte-identical. `index` plays the chunk at its wall-clock index "
        "(t_sim - t0)/dt, which presumes the arm executed the elapsed "
        "steps at full speed; behind --max-delta (and the companion's own "
        "0.5 rad/s slew cap) it did not, so on the rig 2026-09-03 all 70 "
        "of 70 chunk arrivals demanded MORE than the clamp allows and the "
        "arm wiggled at the chunk cadence instead of descending (R-68). "
        "`nearest` rewinds to the row whose arm target is closest (L2 over "
        "the 14 arm dims) to the MEASURED arm pose — the chunk is anchored "
        "at the pose the policy observed, so that row sits at or just "
        "before the wall index, and the search is bounded to wall_idx + 2 "
        "with near-ties resolved toward the wall index (U-32) — and plays "
        "on from there at real time. `offset` keeps the wall index but adds "
        "(current_cmd - chunk[wall_idx]) to the arm targets, decaying to "
        "zero over --splice-ramp-ticks. Grippers and the base are never "
        "corrected. Telemetry: arrival_jump_rad_mean/max, "
        "splice_shift_mean/max",
    )
    parser.add_argument(
        "--leash-rad",
        type=float,
        default=None,
        help="U-32 command leash: how far the arm COMMAND may stand from "
        "the MEASURED joint pose, per joint, after the --max-delta clamp. "
        f"Default: {DEFAULT_LEASH_RAD} rad on --world real (about 0.3 s of "
        "the companion's 0.5 rad/s slew, which is half our clamp), OFF in "
        "sim so every sim result stays byte-identical; 0 disables it. "
        "--max-delta bounds how fast the command may MOVE, not how far it "
        "may drift from the pose the arm actually holds, and a command that "
        "has run ahead makes every chunk hand-over splice against a target "
        "no joint occupies (rig 2026-09-03: |cmd-measured| max 0.28 rad "
        "while the loop held for 413 of 587 ticks). Grippers and the base "
        "are "
        "never leashed. Telemetry: leash_active_ticks, leash_active_pct, "
        "cmd_lead_rad_max, and lead= on each policy chunk line",
    )
    parser.add_argument(
        "--chunk-time-base",
        choices=["observation", "arrival"],
        default=None,
        help="what sim time an arriving SYNCHRONOUS chunk's t0 is stamped "
        "with. Default: 'observation' — today's behaviour, byte-identical "
        "everywhere: t0 is the sim time of the observation backend.infer "
        "was handed, so a slow (real) inference leaves the wall index "
        "already infer_s/dt steps in the moment the chunk installs, and "
        "the executor's next step() skips that many rows of an arm that "
        "never moved. 'arrival' is plan-then-execute (docs/realdata/16 "
        "T6): re-sample the collector right after backend.infer returns "
        "and stamp t0 there instead, so playback starts at row 0 exactly "
        "where the arm is and advances one row per tick — needs_replan, "
        "exhausted, the clamp, the leash, and the chunk dump's "
        "wall_idx/spliced_idx all then count from the arrival. Also "
        "applies under --async-inference, where it simply re-bases the "
        "installed chunk to the arrival tick (harmless, but it zeroes "
        "chunk_arrival_index_max and is not what it is for — this flag is "
        "meant for the synchronous real-robot loop)",
    )
    parser.add_argument(
        "--splice-ramp-ticks",
        type=int,
        default=8,
        help="--chunk-splice offset only: COMMAND ticks (not chunk steps) "
        "over which the hand-over correction decays to zero. 8 at --rate "
        "20 is 0.4 s. 0 disables the correction",
    )
    parser.add_argument(
        "--chunk-dump",
        default=None,
        help="scripts/run_policy.py only, OFF by default: archive every "
        "policy chunk that reaches the executor (sync and async inference "
        "alike) to ONE .npz at PATH — chunk number, t0, sim time/wall/"
        "spliced index at arrival, the jump/lead the 'policy chunk #N' log "
        "line already prints, the observation state the chunk was computed "
        "from (or, under --async-inference, the measured state at arrival, "
        "since the worker never returns the submitted observation — named "
        "obs_state_at_inference vs obs_state_at_arrival so a reader is "
        "never left guessing which), the executor's last commanded arm "
        "vector, and the full (H, ACTION_DIM) chunk, plus a JSON sidecar "
        "(layout names + this run's args) embedded in the same file. "
        "Trivial memory (a rollout replans a few times a second); never "
        "raises into the control loop — a write failure is logged, not "
        "thrown",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="denoising steps of a flow-matching head (pi0/pi0.5, default 10). "
        "Inference-time only. Model-side knob: with --backend remote it "
        "belongs on serve_policy.py, not on the client",
    )
    parser.add_argument(
        "--temporal-ensemble",
        type=float,
        default=None,
        dest="temporal_ensemble",
        help="OFF by default (plan W4/protocol B1): route ACT through "
        "lerobot's ACTTemporalEnsembler with this coeff (0.01 is ACT's own "
        "default). Forces n_action_steps=1; refuses on a non-ACT checkpoint",
    )
    parser.add_argument(
        "--cameras",
        default="head,wrist_left,wrist_right",
        help="comma list of cameras (default: all three). Each camera runs "
        "in its own subscriber subprocess, so all three reach the wire rate "
        "(F-30 decision: all cameras, minimal overhead); subsets remain a "
        "debugging lever. E.g. --cameras head",
    )
    add_wire_image_args(parser)
    parser.add_argument(
        "--correct-poses",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="replay/heuristic only: play the full 20-dim demo; IK the right "
        "arm so world TCP matches GT given the live-vs-demo base/spine "
        "offset. Adapter defaults apply when neither flag is given: replay "
        "off (plain open-loop), heuristic ON (--no-correct-poses disables)",
    )
    parser.add_argument(
        "--index-mode",
        choices=["time", "progress"],
        default=None,
        help="heuristic only: 'time' (default) indexes the reference by the "
        "sim clock; 'progress' advances by nearest-TCP progress (monotone, "
        "rate-capped), so a perturbed start or a disturbance re-converges "
        "onto the reference instead of racing time. Needs --correct-poses "
        "(the heuristic default)",
    )
    parser.add_argument(
        "--skip-approach",
        action="store_true",
        help="skip the closed-loop spine→navigate gate and start manipulation "
        "immediately (smoke tests when already at the desk)",
    )
    parser.add_argument(
        "--approach",
        choices=["pose", "perception"],
        default="pose",
        help="desk approach navigator: pose (default, world-odom waypoints) or "
        "perception (locate against the table from head-camera edges and a "
        "surface cue, then walk "
        "the waypoints on the estimated ego). Ignored with --skip-approach",
    )
    parser.add_argument(
        "--approach-dump",
        default=None,
        help="directory for annotated head-camera frames during perception "
        "navigate (ignored unless --approach perception)",
    )
    parser.add_argument(
        "--approach-start-xy-yaw",
        default=None,
        metavar="X,Y,YAW",
        help="perception approach only (ignored unless --approach perception): "
        "'x,y,yaw' comma floats, world metres and RADIANS, e.g. "
        "'4.4,2.6,-1.5708' for the Task 2 sim spawn (yaw -90 deg). Seeds the "
        "Kalman pose filter at t=0 (design doc S1.2) so plan+PID have a fused "
        "pose from the first navigate tick instead of spinning blind until "
        "the table is seen. Deliberately ROUGH -- it is a filter seed, not a "
        "solve() prior (S1.1). RADIANS, not degrees: C.TASK2_SPAWN_XY_YAW "
        "itself stores radians, and a degrees flag would need a silent "
        "conversion at exactly the seam that must stay exact. Default: "
        "C.TASK2_SPAWN_XY_YAW",
    )
    parser.add_argument(
        "--approach-vision-hz",
        type=float,
        default=0.0,
        help="perception approach only: cap head-camera detect/solve to this "
        "many frames per second (0 = every new frame). The controller already "
        "skips a frame whose stamp it has processed; this additionally "
        "throttles a fast camera (rig ZED ~20 Hz) so vision (~40-50 ms per "
        "real frame) cannot eat the whole control tick. Suggested rig value: 4",
    )
    parser.add_argument(
        "--approach-profile",
        default=None,
        metavar="YAML",
        help="perception approach only: site profile — camera model used when "
        "no CameraInfo is published (rig ZED-M: self-calibrated k1, scaled "
        "output matrix) and the tabletop mask threshold. "
        "configs/rig/perception_munich.yaml for the Munich rig.",
    )
    parser.add_argument(
        "--approach-timeout",
        type=float,
        default=130.0,
        help="max sim seconds for spine settle + navigate + place_arms "
        "(default: 130.0)",
    )
    parser.add_argument(
        "--approach-only",
        action="store_true",
        help="run the base approach, stop the base, and exit — no arm activation, "
        "no policy rollout. The safe way to test --approach perception on the rig.",
    )
    parser.add_argument(
        "--start-pose",
        # None, not "benchmark": on --world real an EXPLICIT sim pose is
        # refused even with --no-start-pose-check, so the two have to be
        # tellable apart. Unset behaves as 'benchmark' everywhere in sim.
        default=None,
        help="where recenter_arms() parks the arms before a rollout: "
        "'benchmark' (ARM_READY_POSE — the F-86 default when unset, and "
        "spine-DOWN: the trimmed corpora contain ZERO frames within 0.5 rad "
        "of it), "
        "'task2' (TASK2_ARM_READY_LEFT/RIGHT, measured from the demos, F-58), "
        "or 'file:<path>' for a JSON {'left': [7], 'right': [7]} lifted from a "
        "real demo frame. See docs/research/PI05_STABILITY_INVESTIGATION.md. "
        "On --world real only 'file:<path>' is accepted (verify-only)",
    )
    parser.add_argument(
        "--start-pose-tol",
        type=float,
        default=START_POSE_TOL_RAD,
        help="real robot only: max per-joint |measured - --start-pose| [rad] "
        f"accepted before the rollout (default {START_POSE_TOL_RAD}). The "
        "arms are never COMMANDED on real — the pose is verified and the run "
        "refuses to start from anywhere else",
    )
    parser.add_argument(
        "--no-start-pose-check",
        dest="start_pose_check",
        action="store_false",
        help="real robot only: log the per-joint start-pose errors but run "
        "anyway. Without --start-pose at all this also lifts the requirement "
        "to pass one",
    )
    parser.add_argument(
        "--arms",
        default="left,right",
        help="real robot only: which arms this run drives (comma list). An "
        "unselected arm is HELD at its measured pose for the whole run — the "
        "policy's prediction for it never reaches the wire",
    )
    parser.add_argument(
        "--wait-for-activation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="real robot only, ON by default: hold the measured pose and poll "
        "/<arm>/controller_manager/list_controllers until "
        "joint_impedance_controller is active on every --arms side, before the "
        "policy speaks. Service clients are created on the runner's OWN node "
        "(a second DDS participant mid-session can fault a live FCI loop)",
    )
    parser.add_argument(
        "--activate-arms",
        action="store_true",
        help="real robot only, OFF by default: perform the switch from here "
        "the way the station's activate_arms.py does — wait for ALL four "
        "controller_manager services first, then switch both arms "
        "(best-effort, then strict). Off means an operator runs "
        "activate_arms.py while this process holds the pose",
    )
    parser.add_argument(
        "--deactivate-on-exit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="real robot only, ON by default: on every exit path deactivate "
        "joint_impedance_controller while still publishing, confirm via "
        "list_controllers (10 s), and only then stop commanding. Dropping the "
        "stream on an ACTIVE controller takes the whole arm launch down",
    )
    parser.add_argument(
        "--keepalive-hz",
        type=float,
        default=DEFAULT_KEEPALIVE_HZ,
        help="real robot only: republish the last arm + gripper command at "
        f"this rate when nothing else was published (default "
        f"{DEFAULT_KEEPALIVE_HZ}). The controller shuts its arm launch down "
        "after 2.0 s without a sample, so a remote-inference stall must not "
        "read as a dead publisher. 0 disables. Never active on sim",
    )
    parser.add_argument(
        "--min-activation-publish-hz",
        type=float,
        default=MIN_ACTIVATION_PUBLISH_HZ,
        help="real robot only: REFUSE to switch joint_impedance_controller on "
        "when the command stream published fewer than this many samples per "
        f"second over the activation window (default {MIN_ACTIVATION_PUBLISH_HZ}). "
        "The measured cadence is logged either way. On 2026-09-02 the runner "
        "activated onto a 6.7 Hz stream (camera workers starving the "
        "executor) and the controller rejected the first sample as 1.076 s "
        "old, then took the whole arm launch down (docs/realdata/16 U-28). "
        "0 disables the gate; nothing is switched when it fires",
    )
    parser.add_argument(
        "--arm-command-frame",
        choices=ARM_COMMAND_FRAMES,
        default="robot",
        help="real robot only: 'robot' (default) publishes the policy's joint "
        "targets as-is; 'gello' pre-inverts the controller's own map, "
        "g = g0 + dir*(q_target - q0), with (q0, g0) captured at activation. "
        "The corpus action frame is the ROBOT frame offline (slope +1 on all "
        "14 joints); default 'robot', measured 2026-09-02 T1b (docs/realdata/16 "
        "R-39: joint 1 in phase with an identity publish, corr +0.922) — kept "
        "as a switch, not a constant, in case a different rig disagrees",
    )
    parser.add_argument(
        "--gello-joint-directions",
        default=",".join(str(int(v)) for v in DEFAULT_GELLO_JOINT_DIRECTIONS),
        help="real robot only: the companion's gello_joint_directions, 7 "
        "values of +1/-1, used by --arm-command-frame gello. MEASURED "
        "2026-09-01; change it only against a re-measurement",
    )
    parser.add_argument(
        "--no-wrench",
        dest="wrench",
        action="store_false",
        help="real robot only: DECLARE that this rig publishes no external "
        "wrench. 12 of the s27a15 policy's 27 state dims are wrench, so this "
        "sends zeros in their place and marks the observation off-contract "
        "(wrench_declared_absent). Without it a missing wrench simply blocks "
        "the observation instead of being silently zero-filled",
    )
    parser.add_argument(
        "--allow-camera-shape-mismatch",
        action="store_true",
        help="real robot only, OFF by default: proceed even when a camera's "
        "measured frame shape disagrees with the contract's declared shape "
        "(docs/realdata/16 R-42). This runs the policy on frames it was not "
        "trained on -- use it only once you have confirmed that is what you "
        "want, not as a way to silence the check",
    )
    parser.add_argument(
        "--max-image-age-s",
        type=float,
        default=None,
        help="real robot only, ON by default at "
        f"{DEFAULT_MAX_IMAGE_AGE_S} s: refuse to keep running when the newest "
        "frame from any camera is older than this at the moment the policy "
        "consumes it (docs/realdata/16 U-23). Nothing else notices a camera "
        "dying mid-rollout -- the collector never clears a cached frame, so "
        "the policy would be fed the same image forever. The bar is the "
        "~3.75 Hz replan cadence, not the 20 Hz control tick. 0 disables the "
        "guard; ignored on --world sim",
    )
    parser.add_argument(
        "--max-state-age-s",
        type=float,
        default=None,
        help="real robot only, ON by default at "
        f"{DEFAULT_MAX_STATE_AGE_S} s: refuse to keep running when the newest "
        "JointState from either arm (or either gripper) is older than this "
        "(docs/realdata/16 U-27). Checked on EVERY control tick, and once "
        "before activation. Nothing else notices an arm's publisher dying "
        "mid-rollout -- the collector never clears a cached joint position, "
        "so the policy and the --joint-csv 'measured' columns would be fed "
        "the same frozen pose forever. Joint states arrive at ~1 kHz, so "
        "0.2 s is ~200 missed messages. 0 disables the guard; ignored on "
        "--world sim",
    )
    parser.add_argument(
        "--joint-csv",
        default=None,
        help="scripts/run_policy.py only, OFF by default: write one row per "
        "rollout control tick of commanded vs measured arm joints (14 each), "
        "both gripper channels, the per-tick clamp, infer_ms and "
        "max_publish_gap_s, then print the per-joint max |measured - "
        "commanded| and clamped_pct. This is the T5 exit criterion "
        "(docs/realdata/16 §5) and the only measured-joint trace that "
        "survives the real rig — tcp_trajectory.csv drops every row whose "
        "base/spine is non-finite, and on the station both always are",
    )
    parser.add_argument(
        "--grasp-gate",
        default=None,
        help="ABLATION (F-97): close the right gripper geometrically, because "
        "the checkpoints never predict the close (0/1120 chunks in the "
        "executed window). Takes the envelope JSON from "
        "tools/dgx_probes/grasp_pose_envelope.py. Fires inside the demos' own "
        "measured grasp envelope (45 mm / 7 deg) using the LIVE pad pose. "
        "Any score produced this way measures the ARM, not the policy — say so",
    )
    parser.add_argument(
        "--gripper-latch",
        default=None,
        metavar="CLOSE_BELOW[:HOLD_S[:RELEASE_ABOVE]]",
        help="OFF by default (byte-identical passthrough): once the policy's "
        "RIGHT gripper command drops below CLOSE_BELOW (default 0.35), hold "
        "the published command at min(policy, 0.20) for at least HOLD_S sim "
        "seconds (default 30.0, i.e. effectively the rest of a 30 s rollout) "
        "and thereafter until the policy commands above RELEASE_ABOVE "
        "(default 0.9) for 1.0 s continuously. A rig-side lever for the "
        "ACT checkpoint's gripper channel, which copies its own state and "
        "re-opens to ~0.45 after a full close (docs/realdata/"
        "15_RIG_WINDOW_RUNBOOK.md §3); needs no pad pose, unlike "
        "--grasp-gate. Left gripper is never touched. Example: "
        "--gripper-latch 0.35:30:0.9, or bare --gripper-latch 0.35 for the "
        "other two defaults",
    )
    parser.add_argument(
        "--gripper-latch-near",
        default=None,
        metavar="POSE_SPEC:RADIUS_RAD",
        help="Requires --gripper-latch. OFF by default (today's "
        "threshold-only behaviour): gate the latch's ENGAGE on joint-space "
        "proximity, because the ACT checkpoint's gripper command hovers at "
        "0.4-0.6 for many seconds during the approach pre-shape and a "
        "threshold-only latch fires while the arm is still far from the "
        "grasp pose (a04: fired at 9.7 s, 0.22 rad out). The latch may only "
        "start holding while the measured RIGHT arm (7 joints) is within "
        "RADIUS_RAD (L2, rad) of POSE_SPEC; hold/release once engaged are "
        "unaffected. POSE_SPEC is 'file:PATH' (JSON with key \"right\", 7 "
        "floats), 'slot:xNNN' (mean_right_arm_rad from "
        "outputs/rig/munich_2026-09-01/slot_grasp_poses.json), or "
        "'list:a,b,c,d,e,f,g' (inline). Example: "
        "--gripper-latch-near slot:x048:0.25",
    )
    parser.add_argument(
        "--grasp-envelope",
        default=None,
        help="MEASURE ONLY: configure the passive close-inside-envelope "
        "detector from this envelope JSON WITHOUT enabling the gate. C1 runs "
        "gate-on vs gate-off as two batches; the gate-on arm's observer "
        "inherits the gate's envelope, so without this the gate-off arm falls "
        "back to no reference attitude and scores 'inside' on distance alone "
        "— the two arms would then measure differently and C1 would not be "
        "one-variable. Pass the SAME file to both arms. Ignored (the gate's "
        "own envelope wins) when --grasp-gate is given",
    )
    parser.add_argument(
        "--grasp-gate-mode",
        choices=["entry", "dwell", "geom", "geom+dwell"],
        default="entry",
        help="C1b gate variant, only meaningful with --grasp-gate: 'entry' "
        "is the plain C1 gate (fires the first tick inside the 45 mm/7 deg "
        "ball -- also what a gate with dwell 0 and no box reduces to); "
        "'dwell' adds --grasp-gate-dwell-s of holding still before firing; "
        "'geom' replaces the ball with the demos' measured finger-frame box "
        "(tcp_frame_mm in the envelope JSON); 'geom+dwell' requires both. "
        "Default 'entry' keeps existing runs byte-for-byte unchanged",
    )
    parser.add_argument(
        "--grasp-gate-dwell-s",
        type=float,
        default=0.0,
        help="sim seconds the fire condition must hold continuously before "
        "the gate closes (modes 'dwell'/'geom+dwell' only); guards against "
        "a fly-through tick that satisfies the geometry for one sample "
        "(measured at 343 mm/s). Default 0.0 = fire on the first ok tick",
    )
    parser.add_argument(
        "--grasp-gate-box-margin-mm",
        default="0,0,0",
        help="'x,y,z' mm widening the measured finger box on each side "
        "(modes 'geom'/'geom+dwell' only) -- the box in tcp_frame_mm is "
        "measured at grasp frames only, so a margin of 0 can reject a "
        "demonstrator's own approach a tick either side of it. Default "
        "'0,0,0' (no margin)",
    )


def make_grasp_gate_from_args(args):
    """The F-97 geometric gripper gate, or None to let the policy own it.

    Default is None: the honest configuration is the policy driving its own
    gripper channel, even though it never closes it.

    mode="entry" with dwell_s=0.0 (the CLI default) returns the plain
    `from_envelope(...)` gate unchanged -- byte-for-byte the same object
    every prior run built -- so passing only --grasp-gate cannot change
    behaviour under this patch. Any other mode, or a nonzero dwell, builds
    the C1b `ConditionedGraspGate` instead.
    """
    spec = getattr(args, "grasp_gate", None)
    if not spec:
        return None
    import json
    from pathlib import Path

    from camelo.control.grasp_gate import conditioned_from_envelope, from_envelope

    path = Path(spec)
    if not path.is_file():
        raise SystemExit(
            f"--grasp-gate envelope not found: {path} — generate it with "
            "tools/dgx_probes/grasp_pose_envelope.py"
        )
    payload = json.loads(path.read_text())
    mode = getattr(args, "grasp_gate_mode", "entry") or "entry"
    dwell_s = float(getattr(args, "grasp_gate_dwell_s", 0.0) or 0.0)
    margin_spec = getattr(args, "grasp_gate_box_margin_mm", "0,0,0") or "0,0,0"
    try:
        box_margin_mm = tuple(float(v) for v in margin_spec.split(","))
    except ValueError:
        raise SystemExit(
            f"--grasp-gate-box-margin-mm must be 'x,y,z' floats, got {margin_spec!r}"
        ) from None
    if len(box_margin_mm) != 3:
        raise SystemExit(
            f"--grasp-gate-box-margin-mm must be 'x,y,z' (3 values), got {margin_spec!r}"
        )
    if mode == "entry" and dwell_s == 0.0:
        gate = from_envelope(payload)
    else:
        gate = conditioned_from_envelope(
            payload, mode=mode, dwell_s=dwell_s, box_margin_mm=box_margin_mm
        )
    # The run log must state the bar the gate actually fires on.
    print(
        f"[camelo] grasp gate: mode={mode} dwell_s={dwell_s:.3f} "
        f"box_margin_mm={box_margin_mm}"
    )
    return gate


def make_grasp_observer_from_args(args, grasp_gate=None):
    """The passive P1 detector, on the same envelope the gate would use.

    Precedence is deliberate: a live gate wins, because the instrument must
    report the bar the actuator actually fired on. Only when there is no
    gate does `--grasp-envelope` apply. Returning None lets the runner build
    its default, which is the distance-only fallback.
    """
    if grasp_gate is not None:
        return None  # the runner copies the gate's own envelope
    spec = getattr(args, "grasp_envelope", None)
    if not spec:
        return None
    import json
    from pathlib import Path

    from camelo.control.grasp_gate import observer_from_envelope

    path = Path(spec)
    if not path.is_file():
        raise SystemExit(
            f"--grasp-envelope not found: {path} — generate it with "
            "tools/dgx_probes/grasp_pose_envelope.py"
        )
    return observer_from_envelope(json.loads(path.read_text()))


def make_gripper_latch_from_args(args):
    """The rig-side gripper-close latch, or None to let the policy own it.

    Default is None (the flag not passed): the honest configuration, same as
    `make_grasp_gate_from_args`, is the policy driving its own gripper
    channel unassisted. Unlike the grasp gate this needs no pad pose and no
    envelope JSON -- it only ever looks at the gripper command itself -- so
    there is nothing here to read from disk, unless ``--gripper-latch-near``
    also names a pose file or slot.
    """
    spec = getattr(args, "gripper_latch", None)
    near_spec = getattr(args, "gripper_latch_near", None)
    if not spec:
        if near_spec:
            raise SystemExit("--gripper-latch-near requires --gripper-latch")
        return None
    from camelo.control.gripper_latch import (
        GripperLatch,
        parse_gripper_latch_near_spec,
        parse_gripper_latch_spec,
        resolve_gripper_latch_near_pose,
    )

    try:
        close_below, hold_s, release_above = parse_gripper_latch_spec(spec)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    near_pose = None
    near_radius_rad = None
    if near_spec:
        try:
            pose_spec, near_radius_rad = parse_gripper_latch_near_spec(near_spec)
            near_pose = resolve_gripper_latch_near_pose(pose_spec)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        print(
            f"[camelo] gripper latch near-gate: radius_rad={near_radius_rad:.3f} "
            f"pose={list(near_pose)}"
        )

    print(
        f"[camelo] gripper latch: close_below={close_below:.3f} "
        f"hold_s={hold_s:.3f} release_above={release_above:.3f}"
    )
    return GripperLatch(
        close_below=close_below,
        hold_s=hold_s,
        release_above=release_above,
        near_pose=near_pose,
        near_radius_rad=near_radius_rad,
    )


def make_start_pose_from_args(args):
    """(left, right) 7-dim arm poses, or None for the default.

    Kept free of any lerobot import: eval_batch.py runs in the ROS container,
    which has no lerobot (F-74). A demo-derived pose therefore arrives as a
    JSON file produced offline by tools/dgx_probes/demo_start_pose.py, not by
    reading a LeRobotDataset here.

    On ``--world real`` the meaning changes from "park the arms here" to
    "assert the arms are already here" (nothing is commanded), and the two
    built-in SIM poses are refused outright: `benchmark` is `ARM_READY_POSE`,
    `task2` is measured from the sim demos, and either one driven into two
    Franka arms is a large unplanned motion, not a default.

    `--no-start-pose-check` waives the **check**, never the sim-pose
    semantics: an explicit `--start-pose benchmark`/`task2` is refused with
    or without it, because the operator asked for a pose that does not
    describe this robot and silently answering "no pose at all" would run
    the rollout from wherever the arms happen to be while the command line
    says otherwise. Only an OMITTED `--start-pose` (the flag's `None`
    default) plus `--no-start-pose-check` means "no pose, deliberately".
    """
    spec = getattr(args, "start_pose", None)
    real = getattr(args, "world", "sim") == "real"
    if real:
        if spec is None:
            if not getattr(args, "start_pose_check", True):
                return None  # --no-start-pose-check: deliberately unverified
            raise SystemExit(
                "--world real needs --start-pose file:<path> — a JSON "
                "{\"left\": [7], \"right\": [7]} lifted from the rig's teleop "
                "home pose (or frame 0 of the corpus episode you are "
                "reproducing). The flag is verify-only on real: nothing is "
                "commanded. Pass --no-start-pose-check to run without any "
                "start-pose check at all"
            )
        if spec in ("benchmark", "task2"):
            raise SystemExit(
                f"--start-pose {spec!r} is a SIM pose and --world real refuses "
                "it: 'benchmark' is the benchmark's ARM_READY_POSE and 'task2' "
                "was measured from the sim demos. On the real robot the flag is "
                "verify-only, so pass 'file:<path>' to a JSON "
                "{\"left\": [7], \"right\": [7]} lifted from the rig's teleop "
                "home pose (or frame 0 of the corpus episode you are "
                "reproducing). --no-start-pose-check does NOT make this pose "
                "acceptable — it waives the check, not the sim-pose semantics; "
                "omit --start-pose entirely to run unverified"
            )
    if spec is None or spec == "benchmark":
        return None
    if spec == "task2":
        from camelo import contracts as C

        return (C.TASK2_ARM_READY_LEFT, C.TASK2_ARM_READY_RIGHT)
    if spec.startswith("file:"):
        import json
        from pathlib import Path

        path = Path(spec[len("file:") :])
        if not path.is_file():
            raise SystemExit(f"--start-pose file not found: {path}")
        data = json.loads(path.read_text())
        try:
            left, right = data["left"], data["right"]
        except (KeyError, TypeError) as exc:
            raise SystemExit(f"{path}: expected keys 'left' and 'right' — {exc}") from exc
        if len(left) != 7 or len(right) != 7:
            raise SystemExit(
                f"{path}: left/right must be 7 joints, got {len(left)}/{len(right)}"
            )
        return (left, right)
    raise SystemExit(
        f"--start-pose must be 'benchmark', 'task2' or 'file:<path>', got {spec!r}"
    )


def topics_from_args(args):
    """TopicMap for --world / CAMELO_WORLD (default sim)."""
    from camelo.contracts import topics_for

    return topics_for(getattr(args, "world", None))


def arms_from_args(args) -> tuple:
    """`--arms` -> the arms this run drives, in a stable order."""
    spec = getattr(args, "arms", "left,right") or ""
    arms = [a.strip().lower() for a in spec.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ("left", "right")]
    if unknown or not arms:
        raise SystemExit(f"--arms must be a non-empty subset of left,right; got {spec!r}")
    return tuple(a for a in ("left", "right") if a in arms)


def gello_directions_from_args(args) -> tuple:
    """`--gello-joint-directions` -> 7 signs, or refuse.

    Only +1 / -1 are accepted. The controller's map is
    ``q_goal = q0 + dir*(g - g0)``; any other magnitude would silently
    RESCALE every joint delta, which looks like a sluggish or twitchy arm
    rather than like a wrong flag.
    """
    spec = getattr(args, "gello_joint_directions", None)
    if not spec:
        return tuple(DEFAULT_GELLO_JOINT_DIRECTIONS)
    try:
        values = [float(v) for v in str(spec).split(",") if v.strip()]
    except ValueError:
        raise SystemExit(
            f"--gello-joint-directions must be 7 comma-separated +1/-1, got {spec!r}"
        ) from None
    if len(values) != 7 or any(v not in (1.0, -1.0) for v in values):
        raise SystemExit(
            "--gello-joint-directions must be exactly 7 values, each +1 or -1 "
            f"(MEASURED default {','.join(str(int(v)) for v in DEFAULT_GELLO_JOINT_DIRECTIONS)}); "
            f"got {spec!r}"
        )
    return tuple(values)


def action_space_from_args(args, backend=None) -> str:
    """Which action vector the executor will be handed — DECLARED, not measured.

    Locally the adapter knows (`PolicyAdapter.action_space`); across the
    remote split the client never builds one, so `--action-layout` is the
    declaration. When both exist they must agree: a client that thinks it is
    receiving the sim's 20-dim contract while the server serves the rig's 15
    is F-45 with a live robot attached, and `s27a15.chunk_for_executor`
    only catches the half of that where the widths differ.
    """
    declared = s27a15.LAYOUT if "s27a15" in (
        getattr(args, "action_layout", None), getattr(args, "state_layout", None)
    ) else s27a15.CANONICAL_SPACE
    adapter = getattr(backend, "adapter", None)
    from_adapter = getattr(adapter, "action_space", None)
    if from_adapter is not None and from_adapter != declared:
        raise SystemExit(
            f"the adapter emits action_space={from_adapter!r} but the flags "
            f"declare {declared!r} — pass --action-layout s27a15 --state-layout "
            "s27a15 together, or neither"
        )
    return declared


def approach_start_xy_yaw_from_args(args):
    """'--approach-start-xy-yaw' -> (x, y, yaw) in metres/RADIANS, or None.

    None lets ``PerceptionApproachController`` fall back to its own default
    (``C.TASK2_SPAWN_XY_YAW``, the Task 2 sim spawn) — this flag only
    overrides it. Not validated as a real pose (no bounds check): it is a
    rough Kalman seed, not a `solve()` prior (design doc S1.1/S1.2), so a
    wrong-but-parseable value is a bad rollout, not a contract violation.
    """
    spec = getattr(args, "approach_start_xy_yaw", None)
    if spec is None:
        return None
    try:
        values = tuple(float(v) for v in str(spec).split(","))
    except ValueError:
        raise SystemExit(
            "--approach-start-xy-yaw must be 'x,y,yaw' floats (metres, "
            f"RADIANS), got {spec!r}"
        ) from None
    if len(values) != 3:
        raise SystemExit(
            f"--approach-start-xy-yaw must be exactly 'x,y,yaw' (3 values), got {spec!r}"
        )
    if not all(math.isfinite(v) for v in values):
        # 'nan'/'inf' parse as floats and would otherwise seed the filter,
        # surfacing much later as a ValueError inside the planner's
        # arc-length resampling. Fail here, where the message is useful.
        raise SystemExit(
            f"--approach-start-xy-yaw must be finite numbers, got {spec!r}"
        )
    return values


def make_approach_from_args(args):
    """None when --skip-approach; on --world real only the perception approach
    with a rig profile drives the base (the pose-based controller needs the
    sim's map frame). Else the Task 2 desk-zone controller."""
    if getattr(args, "skip_approach", False):
        return None
    dump_dir = getattr(args, "approach_dump", None)
    start_xy_yaw_spec = getattr(args, "approach_start_xy_yaw", None)
    mode = getattr(args, "approach", "pose")
    real = getattr(args, "world", "sim") == "real"
    if real and mode != "perception":
        log.info(
            "--world real: no base approach (the pose-based controller is sim-only); "
            "pass --approach perception --approach-profile <yaml> to drive the base"
        )
        return None
    if real and not getattr(args, "approach_profile", None):
        raise SystemExit(
            "--world real --approach perception needs --approach-profile "
            "(configs/rig/perception_munich.yaml) with a `rig:` section"
        )
    if mode == "perception":
        from camelo.control.approach_perception_based import PerceptionApproachController

        vision_hz = float(getattr(args, "approach_vision_hz", 0.0) or 0.0)
        profile = None
        profile_path = getattr(args, "approach_profile", None)
        if profile_path:
            from camelo.control.perception.profile import load_profile

            profile = load_profile(profile_path)
            if real and profile.rig is None:
                raise SystemExit(f"{profile_path}: no `rig:` section — not a real-robot profile")
        return PerceptionApproachController(
            dump_dir=dump_dir,
            start_xy_yaw=approach_start_xy_yaw_from_args(args),
            vision_min_period_s=(1.0 / vision_hz) if vision_hz > 0.0 else 0.0,
            profile=profile,
        )
    if dump_dir:
        log.warning("--approach-dump is ignored unless --approach perception")
    if start_xy_yaw_spec:
        log.warning("--approach-start-xy-yaw is ignored unless --approach perception")
    from camelo.control.approach import ApproachController

    return ApproachController(finegrained_start_position=True)


def make_adapter_from_args(args):
    from camelo.policy.adapters import make_adapter

    kwargs = {}
    if args.adapter.startswith("gr00t-isaac"):
        kwargs["embodiment_tag"] = args.gr00t_embodiment or "new_embodiment"
    elif args.gr00t_embodiment and args.adapter.startswith(("gr00t", "lerobot")):
        kwargs["embodiment_tag"] = args.gr00t_embodiment
    if getattr(args, "action_layout", None) and not args.adapter.startswith(
        ("dummy", "replay", "heuristic", "gr00t-isaac")
    ):
        kwargs["action_layout"] = args.action_layout
    if getattr(args, "camera_map", None) and not args.adapter.startswith(
        ("dummy", "replay", "heuristic", "gr00t-isaac")
    ):
        entries = [item.split("=", 1) for item in args.camera_map.split(",") if item.strip()]
        bad = [item for item in entries if len(item) != 2]
        if bad:
            raise SystemExit(f"--camera-map entries must be key=camera, got {bad}")
        kwargs["camera_map"] = dict(entries)
    if getattr(args, "state_layout", None) and not args.adapter.startswith(
        ("dummy", "replay", "heuristic", "gr00t-isaac")
    ):
        kwargs["state_layout"] = args.state_layout
    if getattr(args, "chunk_dt", None) is not None and not args.adapter.startswith(
        ("dummy", "replay", "heuristic", "gr00t-isaac")
    ):
        kwargs["chunk_dt"] = args.chunk_dt
    real_layout = "s27a15" in (
        getattr(args, "action_layout", None),
        getattr(args, "state_layout", None),
    )
    if getattr(args, "gripper_command", None):
        if not real_layout:
            raise SystemExit(
                "--gripper-command is an s27a15 flag (the real gripper driver's "
                "units); the sim path publishes through "
                "camelo.contracts.gripper_wire_value"
            )
        kwargs["gripper_command"] = args.gripper_command
    if getattr(args, "allow_caption_drift", False):
        if not real_layout:
            raise SystemExit("--allow-caption-drift is an s27a15 flag")
        kwargs["allow_caption_drift"] = True
    if getattr(args, "num_inference_steps", None) is not None:
        if args.adapter.startswith(("dummy", "heuristic", "gr00t-isaac")):
            raise SystemExit(
                f"--num-inference-steps is a lerobot flow-matching knob; "
                f"adapter {args.adapter!r} has no such setting"
            )
        kwargs["num_inference_steps"] = args.num_inference_steps
    if getattr(args, "temporal_ensemble", None) is not None and not args.adapter.startswith(
        ("dummy", "replay", "heuristic", "gr00t-isaac")
    ):
        kwargs["temporal_ensemble_coeff"] = args.temporal_ensemble
    if getattr(args, "correct_poses", None) is not None and args.adapter.startswith(
        ("replay", "heuristic")
    ):
        kwargs["correct_poses"] = args.correct_poses
    if getattr(args, "index_mode", None) is not None:
        # Refuse rather than ignore (F-12 class: never a silent no-op) —
        # only the heuristic adapter has a progress follower.
        if not args.adapter.startswith("heuristic"):
            raise SystemExit(
                f"--index-mode is a heuristic-adapter knob; adapter "
                f"{args.adapter!r} has no such setting"
            )
        kwargs["index_mode"] = args.index_mode
    return make_adapter(args.adapter, checkpoint=args.checkpoint, device=args.device, **kwargs)


def add_wire_image_args(parser: argparse.ArgumentParser) -> None:
    """`--wire-image-size` / `--wire-jpeg-quality` — the REMOTE-path levers.

    Shared verbatim by `scripts/run_policy.py` (through `add_policy_args`)
    and `scripts/run_dummy_client.py`, which builds its own parser: the
    wire probe has to be able to send exactly what a rollout sends, and two
    hand-copied help texts are two chances to drift.
    """
    from camelo.policy.wire import JPEG_QUALITY

    parser.add_argument(
        "--wire-image-size",
        default=None,
        metavar="WxH|NAME=WxH,...",
        help="REMOTE backend only: resize each camera to this size BEFORE "
        "the JPEG encode, with the same torchvision Resize the checkpoint's "
        "own preprocessor applies on the server (bilinear + antialias on "
        "the uint8 frame; PIL bilinear when the client has no torch, which "
        "agrees to 1/255). Halves-and-then-some the transport half of the "
        "round trip: on the Munich rig the three full-resolution JPEGs are "
        "~116 kB and ~0.24 s of a ~0.30 s RTT, longer than VLA-JEPA's own "
        "7-row chunk. Spelled WxH (e.g. 224x224 for every camera) or "
        "per-camera (head=224x224,wrist_right=224x224); the two may be "
        "mixed. Default: off, native resolution, byte-identical to every "
        "run on record. Pass the size the CHECKPOINT declares — nothing "
        "here can read its train_config.json from the client side, and a "
        "smaller one is silently re-resized by the server from a frame "
        "that has already lost the pixels",
    )
    parser.add_argument(
        "--wire-jpeg-quality",
        type=int,
        default=JPEG_QUALITY,
        help="REMOTE backend only: JPEG quality of the images on the wire "
        f"(default {JPEG_QUALITY}, today's value — unchanged unless passed)",
    )


def wire_images_from_args(args, cameras=None):
    """The `WireImageSpec` this run sends with, or None = today's encoding.

    Returns None for the untouched defaults so `RemoteBackend` keeps its own
    `DEFAULT_WIRE_IMAGES` and nothing about an existing run changes.
    """
    from camelo.policy.wire import JPEG_QUALITY, WireImageSpec, parse_wire_image_size

    size_text = getattr(args, "wire_image_size", None)
    quality = int(getattr(args, "wire_jpeg_quality", JPEG_QUALITY))
    if not 1 <= quality <= 100:
        raise SystemExit(f"--wire-jpeg-quality must be in 1..100, got {quality}")
    if size_text is None and quality == JPEG_QUALITY:
        return None
    try:
        spec = (
            WireImageSpec()
            if size_text is None
            else parse_wire_image_size(size_text, cameras=cameras)
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    spec.quality = quality
    return spec


def make_backend_from_args(args):
    from camelo.policy.backend import LocalBackend, RemoteBackend

    if args.backend == "remote":
        # The model lives in the server process, so a client-side value would
        # be accepted and then quietly ignored — the sweep would "compare"
        # two identical runs. Refuse here (F-12 class: never a silent no-op).
        if getattr(args, "num_inference_steps", None) is not None:
            raise SystemExit(
                "--num-inference-steps has no effect with --backend remote: "
                "the policy runs in the server. Pass it to serve_policy.py "
                "and restart the server instead."
            )
        return RemoteBackend(
            args.server, wire_images=wire_images_from_args(args, cameras_from_args(args))
        )
    # Same refusal shape as --num-inference-steps above: the local backend
    # never encodes anything, so a --wire-* flag here would be accepted and
    # then quietly do nothing (F-12 class: never a silent no-op).
    if wire_images_from_args(args) is not None:
        raise SystemExit(
            "--wire-image-size/--wire-jpeg-quality condition the images on "
            "the WIRE; --backend local hands the adapter the frames "
            "directly and encodes nothing. Drop them, or use --backend remote."
        )
    return LocalBackend(make_adapter_from_args(args))


def make_executor_from_args(args):
    from camelo.control.chunk_executor import ChunkExecutor

    return ChunkExecutor(
        max_delta_per_tick=args.max_delta,
        replan_after_steps=args.replan_steps,
        chunk_splice=chunk_splice_from_args(args),
        splice_ramp_ticks=getattr(args, "splice_ramp_ticks", 8),
        leash_rad=leash_rad_from_args(args),
    )


def leash_rad_from_args(args) -> float | None:
    """The U-32 command leash in rad, or None = leash OFF.

    Real world only, and on by default there (`DEFAULT_LEASH_RAD` = 0.15
    rad, about 0.3 s of the companion's 0.5 rad/s slew cap). In sim the
    bridge applies our targets with no slew limit of its own, so the
    command cannot run away from the measured pose the way it does behind
    the real controller, and leaving the leash off keeps every sim result
    already on record byte-identical. `--leash-rad 0` is the explicit
    opt-out on real; `--leash-rad <x>` turns it on anywhere.
    """
    value = getattr(args, "leash_rad", None)
    if value is None:
        value = DEFAULT_LEASH_RAD if getattr(args, "world", None) == WORLD_REAL else 0.0
    value = float(value)
    return value if value > 0.0 else None


def chunk_splice_from_args(args) -> str:
    """Where an arriving chunk resumes playback (docs/realdata/16 U-31).

    Auto-resolved to `nearest` exactly where the hand-over gap exists — a
    chunk that lands SOME ticks after the observation it was computed
    from, i.e. asynchronous inference — and to `index` everywhere else.
    Synchronously the loop is blocked on the reply, so the chunk arrives
    at its own `t0` and the wall index is 0: there is nothing to splice
    and every sim result stays byte-identical. `--chunk-splice` overrides
    in either direction.
    """
    value = getattr(args, "chunk_splice", None)
    if value is not None:
        return str(value)
    return "nearest" if async_inference_from_args(args) else "index"


def chunk_time_base_from_args(args) -> str:
    """What sim time an arriving synchronous chunk's t0 is stamped with.

    Unlike `chunk_splice_from_args` / `async_inference_from_args`, there is
    no conditional default here: 'observation' is right everywhere until an
    operator explicitly asks for plan-then-execute, so the sentinel just
    resolves to it rather than branching on --world/--backend.
    """
    value = getattr(args, "chunk_time_base", None)
    return "observation" if value is None else str(value)


def async_inference_from_args(args) -> bool:
    """Whether ``backend.infer`` runs off the control thread.

    ON by default exactly where it changes something: a REMOTE backend on
    the REAL robot, where the round trip is a network round trip and the
    loop it blocks is driving a live arm (docs/realdata/16 U-29, measured
    0.45 s median -> 3 Hz effective). A LOCAL backend would still hold the
    GIL through its own forward pass, and in SIM the clock is ours to
    starve, so both stay synchronous and byte-identical to every result
    already on record. ``--async-inference`` / ``--no-async-inference``
    override in either direction.
    """
    value = getattr(args, "async_inference", None)
    if value is not None:
        return bool(value)
    return (
        getattr(args, "backend", None) == "remote"
        and getattr(args, "world", None) == WORLD_REAL
    )


def cameras_from_args(args) -> list[str]:
    from camelo import contracts as C

    keys = [key.strip() for key in args.cameras.split(",") if key.strip()]
    unknown = [key for key in keys if key not in C.CAMERA_KEYS]
    if unknown:
        raise SystemExit(f"unknown cameras {unknown}; valid: {C.CAMERA_KEYS}")
    return keys


def max_image_age_from_args(args) -> float | None:
    """The U-23 stale-image threshold in wall seconds, or None = guard OFF.

    Real world only, and on by default there (`DEFAULT_MAX_IMAGE_AGE_S`): a
    dead camera on the rig is a frozen frame the policy cannot see through
    (docs/realdata/16 U-23), while in sim the cameras and the clock stop
    together, so an age threshold would fire on every paused sim rather than
    on a fault. `--max-image-age-s 0` is the explicit opt-out on real.
    """
    if getattr(args, "world", None) != WORLD_REAL:
        return None
    value = getattr(args, "max_image_age_s", None)
    value = DEFAULT_MAX_IMAGE_AGE_S if value is None else float(value)
    return value if value > 0.0 else None


def max_state_age_from_args(args) -> float | None:
    """The U-27 stale-joint-state threshold in wall seconds, None = guard OFF.

    Real world only, and on by default there (`DEFAULT_MAX_STATE_AGE_S`): a
    controller that shuts its arm launch down on the rig leaves the
    collector's last measured pose in place forever (docs/realdata/16 U-27,
    R-64), while in sim the joint stream and the clock stop together, so an
    age threshold would fire on every paused sim rather than on a fault.
    `--max-state-age-s 0` is the explicit opt-out on real.
    """
    if getattr(args, "world", None) != WORLD_REAL:
        return None
    value = getattr(args, "max_state_age_s", None)
    value = DEFAULT_MAX_STATE_AGE_S if value is None else float(value)
    return value if value > 0.0 else None


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
