# EBiM Task 2 Real-Robot Submission — Team Camelo

Team **Camelo**'s submission for **EBiM Task 2** — *"Pick up the thermal pad and place it on the target RAM board"* — run on the **real robot** at the Munich rig: a mobile dual-Franka-FR3 "TMR" base with a ZED-M head camera and two RealSense D405 wrist cameras. This repository is the runnable source behind the submission; the served policy also ships as a prebuilt GPU container image, `ghcr.io/ostjul/camelo-ebim-task2-phase2-submission:v0.1.0`.

## 1. What this is

The submission has four parts. An **ACT** policy, trained on the Munich demonstration corpus in the rig's own `s27a15` layout (27-dim state, 15-dim action, 20 Hz control, head 1280x720 + two wrists 640x480), does the manipulation; it is served remotely and driven by an **executor** that runs natively — not containerized — on the station laptop; a **new perception-based base approach** repositions the mobile base before a rollout using odometry plus an occasional head-camera table fix; and a set of **rig helper files** make the whole thing reproducible session to session.

- **The policy checkpoint.** ACT, `s27a15` layout, published on Hugging Face as [`ostjul/camelo-ebim-task2-act-s27a15`](https://huggingface.co/ostjul/camelo-ebim-task2-act-s27a15), served by `scripts/serve_policy.py` in a GPU container.
- **The executor.** `camelo`'s ROS-side runner (`scripts/run_policy.py`), run natively in the station's pixi Humble environment and pip-installed from this repository's source — a container participant on the station's DDS graph is banned while its own nodes run (rule C-3, §3), so this is not optional.
- **The perception-based base approach.** `--approach perception --approach-profile configs/rig/perception_munich.yaml` — drives the base from an operator-supplied rough start pose to the parked pose using relative odometry, corrected by head-camera table localisation. It ships **off by default and offline-validated only**: the team had no further access to the rig after it was written, so the table model and the parked goal pose in the profile are left `null` for the operator to measure (§8), and the reference run (§7) had the base parked by hand.
- **Helper files.** Site-script snapshots, reference poses/frames and scoring tools that bring the rig up and check a rollout the same way every session (§11).

**Read §10 before running anything on hardware:** no scored grasp has succeeded on this rig yet. The approach-and-pose half of the task is solved; gripper-close timing is not.

## Quick start

Three terminals. On the cloud GPU VM (§6):

```bash
docker compose up policy-server        # or the docker run form in §6
```

On the station (ebimHP), a tunnel to it, then the executor in the pixi shell (§7):

```bash
ssh -N -L 8767:127.0.0.1:8767 <user>@<gpu-vm>
```

```bash
cd ~/teleoperation/station && pixi shell && cd ~/camelo/camelo-ebim
python -u scripts/run_dummy_client.py --server ws://127.0.0.1:8767 --rate 2 --seconds 20   # expect chunk=(21, 15)
```

then the a05 rollout line in §7, after the bring-up and alignment steps there (§7, §9). The base approach (§8) is a separate, optional step before the rollout that needs two operator measurements first; the reference run had the base placed by hand.

## 2. Repository layout

```
.
├── README.md                     # this file
├── Dockerfile                    # FROM ghcr.io/ostjul/camelo-ebim-task2-phase2-submission:v0.1.0 — pins the released server image
├── compose.yaml                  # one service: policy-server (GPU, ws://…:8767)
├── LICENSE                       # Apache-2.0
├── pyproject.toml
├── Makefile                      # `make test`, `make dds-profile`, `make check-obs`, …
├── camelo/                       # policy bridge, ROS I/O, control, training code
│   ├── contracts.py                        # sim + real topic/vector definitions
│   ├── policy/adapters/s27a15.py           # the Munich rig's 27-state/15-action layout
│   ├── policy/server.py                    # the policy websocket server
│   └── control/
│       ├── approach.py                     # pose-based approach (sim default)
│       ├── approach_perception_based.py    # the NEW perception-based approach FSM
│       ├── perception/                     # detect / localize / geometry / planner / profile
│       └── perception_based_navigation.md  # design doc for the approach (§8)
├── scripts/
│   ├── run_policy.py              # the rollout / approach runner — this IS the executor
│   ├── serve_policy.py            # the ACT policy server entry point
│   └── capture_head.py / stream_head.py / read_spine.py / run_dummy_client.py / check_obs.py
├── configs/
│   ├── realdata/act/              # ACT training recipe: train.yaml, launch.sh, smoke.sh, README.md
│   └── rig/perception_munich.yaml # the Munich rig's perception + base-drive profile
├── tools/
│   ├── rig_probes/                # rig probes and scoring tools (table in §11)
│   └── rig_parity_check.py        # the parity gate — run before trusting any rollout
├── docs/
│   ├── CONTRACTS.md                        # topic + contract cheat sheet ("Real robot (TMR station)")
│   ├── setup/SETUP.md                      # environment setup, incl. the cv2/PIL core-dep note
│   └── realdata/
│       ├── 16_RIG_TEST_PROTOCOL.md         # rig facts, T1–T6 test definitions, rules C-1..C-16
│       ├── 16c_T6_CHEATSHEET.md            # terminal-by-terminal bring-up (§7 source)
│       ├── 16d_T6_LEARNINGS.md             # what's proven, what failed, next steps (§10 source)
│       └── site_scripts/                   # verbatim snapshots of the companion + station scripts
└── outputs/rig/
    ├── t5/                          # reference poses and frames
    │   ├── start_pose_s27a15_ep163.json / start_pose_s27a15_ep009.json
    │   ├── ep163_frame0_head.png    # reference head frame for the alignment overlay
    │   └── ep163_frames/            # per-frame PNGs — also the real-frame perception regression set
    └── munich_2026-09-01/
        ├── slot_grasp_poses.json    # per-slot demo grasp poses (used by --gripper-latch-near)
        ├── runs.csv                 # one row per bring-up / probe session (T1–T6)
        └── rollouts.csv             # one row per scored rollout (protocol §2)
```

`docs/realdata/site_scripts/` is a snapshot for inspection; the rig's own copies on the companion and on ebimHP are the source of truth (see each subdirectory's own `README.md`/`MD5SUMS.txt`).

## 3. The rig

A mobile dual-Franka-FR3 ("TMR") base, swerve-driven (`swerve_drive_controller`), with a ZED-M head camera and two RealSense D405 wrist cameras. Three compute nodes:

| Box | Role |
|---|---|
| **companion** (Jetson AGX Orin, Ubuntu 22.04, ROS 2 Humble) | arm controllers (`joint_impedance_controller` per arm), gripper clients, and the base controller (`swerve_drive_controller`) — brought up by `start_upper.bash` (arms+grippers) and `start_base.bash` (base, in isolation; the combined `start_robot.bash` is unusable, §7) |
| **ebimHP** (control laptop, Ubuntu 24.04) | native station camera nodes (ZED + both D405s), a pixi-managed RoboStack **Humble** environment, and the camelo executor running inside that shell |
| **GPU box** — a cloud GPU instance (the organisers run an A100/H100 VM on Google Cloud) | the ACT policy server (`scripts/serve_policy.py` in the released container), reached from ebimHP through an SSH tunnel so the executor always talks to `ws://127.0.0.1:8767` (§6) |

**DDS rules.** No camelo **container** participant may exist on ebimHP while any station ROS node runs — a container joining that graph correlates with two whole-station kernel OOM events (rule **C-3**); a native pixi participant does not have this problem, which is why the executor runs natively. Native runs must also use camelo's own rendered Fast DDS profile (`make dds-profile`), not the site's — the site's profile has no socket buffer sizes, and a default 208 KB receive buffer silently drops most of every fragmented camera frame under a foreign participant (rule **C-14**).

## 4. The `s27a15` contract

27-dim state in, 15-dim action out, 20 Hz, defined in `camelo/policy/adapters/s27a15.py` and mirrored in `docs/CONTRACTS.md`. It is **not** a re-slicing of the sim contract — the only thing shared is that arm-joint dims 0–13 name the same fourteen joints in the same order, as absolute radians.

**`observation.state` (27, float32):**

| dims | field |
|---|---|
| `[0:7]` | left arm, measured joint positions (rad) |
| `[7:14]` | right arm, measured joint positions (rad) |
| `[14]` | right gripper, open fraction (1.0 = open; converted from the raw knuckle angle) |
| `[15:21]` | left external wrench (force xyz, torque xyz), stiffness frame |
| `[21:27]` | right external wrench, same |

**`action` (15, float32), absolute joint targets in radians:**

| dims | field |
|---|---|
| `[0:7]` | left arm target |
| `[7:14]` | right arm target |
| `[14]` | right gripper target, open fraction (1.0 = open) |

**Cameras**, corpus key order `head, wrist_left, wrist_right`: head (ZED, `/head_camera/zed_node/rgb/color/rect/image`) at **1280x720**; both wrists (RealSense D405) at **640x480**. There is no base or left-gripper channel in the action; the executor supplies a zero base twist and a held left-gripper value when the 15-dim chunk is widened to the sim-derived executor's 20-dim slew clamp (`s27a15.rig_chunk_to_canonical`).

**Topics** (`docs/CONTRACTS.md`, "Real robot (TMR station)"):

| Role | Topic | Type |
|---|---|---|
| left/right arm state | `/left\|right/franka_robot_state_broadcaster/measured_joint_states` | `JointState`, BEST_EFFORT |
| gripper state | `/left\|right/gripper/joint_states` | `JointState`, one joint `robotiq_85_left_knuckle_joint`, raw radians (0.0 open, 0.7929 closed) |
| external wrench | `/left\|right/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame` | `WrenchStamped`, BEST_EFFORT |
| spine state | `/spine/joint_states` | `JointState` |
| odom | `/swerve_drive_controller/odom` | `Odometry` |
| applied base twist | `/swerve_drive_controller/cmd_vel_out` | `TwistStamped` |
| head camera | `/head_camera/zed_node/rgb/color/rect/image` | `Image`, BEST_EFFORT, 1280x720 |
| left/right wrist camera | `/wrist_camera_{left,right}/camera/color/image_rect_raw` | `Image`, BEST_EFFORT, 640x480 |
| arm command (GELLO stand-in) | `/left\|right/gello/joint_states` | `JointState`, names `left\|right_fr3v2_joint1..7` |
| gripper command | `/left\|right/gripper/gripper_client/target_gripper_width_percent` | `std_msgs/Float32`, **0..1 open fraction** (misnomer — measured, not assumed) |
| **base command** | `/swerve_drive_controller/cmd_vel` | **`geometry_msgs/TwistStamped`** — a fresh `header.stamp` per publish; the controller zeros anything older than 0.5 s (`cmd_vel_timeout`), so an unstamped `Twist` or a bare `ros2 topic pub` never moves the base. Clamped to 0.1 m/s / 0.1 rad/s, ramped at 0.1 m/s²; publish at 20 Hz, zero on exit |
| spine command | `/spine/target_height` | held; not a policy output |

`tools/rig_probes/base_wire_check.sh` (§8) re-confirms the base command's message type before the first base command of a session. The type, the stamp watchdog and the clamps above are taken from the site's own `base_nudge.py` and `start_base.bash` (shipped verbatim, §11), not from a live `ros2 topic type` by this team — camelo's publisher has never been matched against the running controller (§8).

## 5. The ACT checkpoint

**Hugging Face:** [`ostjul/camelo-ebim-task2-act-s27a15`](https://huggingface.co/ostjul/camelo-ebim-task2-act-s27a15).

**Training recipe** (`configs/realdata/act/train.yaml`, `configs/realdata/act/README.md`): ACT, from scratch (ResNet18/ImageNet backbone, ~80M trainable params, no PEFT) — the only checkpoint family in this program with a measured *vision-timed* gripper close, so it carries no sim-to-real embodiment gap. `chunk_size=21`, `n_action_steps=10` (1.05 s / 0.5 s at this corpus's 20 Hz — time-matched from the sim corpus's 30 fps `32/16`); native camera resolution, no forced crop/letterbox; `MEAN_STD` normalization (ACTConfig default); batch 8, AdamW lr 1e-5, 100,000 steps, `save_freq=8333`, 1×H100. Of the corpus's 217 episodes / 121,828 frames, `eval_split: 0.1` holds back the last 20 of the 195 train-list episodes, leaving **175 episodes / 99,654 frames** actually trained on (100,000 steps × batch 8 ÷ 99,654 = 8.03 epochs). Launched 2026-08-31, smoke run PASSed, full run measured 0.090 s/step (≈2.5 h pure training). **The served checkpoint is step 100,000** — the exact one driven in every T6 rollout below.

**Offline evaluation.** `tools/rig_parity_check.py` is the gate before any rollout is trusted: it drives the checkpoint through the exact adapter the rig runs, fed from a held-out episode's recorded frames as if they were live, and diffs the result against (A) a live re-run of the training pipeline over every frame/chunk-step/all 15 action dims, and (B) a stored probe JSON:

```bash
.venv/bin/python tools/rig_parity_check.py \
    --probe outputs/probes/act_3983190_100000.json \
    --checkpoint outputs/runs/train_..._3983190/checkpoints/100000/pretrained_model \
    --dataset outputs/datasets/ebim_task2_realrobotdata/task2_munich_s27a15_hires \
    --episode 9 --state-layout s27a15 --action-layout s27a15
```

(The corpus itself is not part of this repository, so the parity gate is reproducible only with access to it; the served checkpoint's own probe output ships as `probe_report.md` in the Hub repo.)

**Held-out probe of the served checkpoint** (22 held-out episodes, 20 Hz, `probe_report.md` on the Hub): gripper-close timing **TIMED 21/22**, UNTIMED 1, MISPLACED 0, ABSENT 0 — the only trained rung whose raw TIMED beats the state-copying null (20/22); **demo-relative PASS 11/22**, the highest of the ten rungs at the handoff, identical on two pinned seeds (ACT's head is deterministic; the other rungs sample at inference). Timing-error median 0.0 frames, IQR [−9.5, +4.0]; right-arm MAE over the 21-step chunk 0.033 → 0.074 rad. These are offline numbers against demonstrations, not task success.

**On-rig evaluation.** Protocol 15 §2's scored block (15 rollouts, ≥5 per slot at x≈0.48/0.51/0.61) was started once as **block B2** and stopped after 3 rollouts — all **ABSENT** by mechanism verdict — on an executor line since superseded. A better line found later the same window (async inference, observation-time chunk base, offset splice — §7, §10) reached the demonstration's own grasp pose and closed on the pad once in an unscored probe, but no scored rollout has yet used it. **See §10 before treating any number here as a task-success rate — there isn't one yet.**

## 6. Policy server

The submission image (`ghcr.io/ostjul/camelo-ebim-task2-phase2-submission:v0.1.0`) serves the checkpoint straight from the Hub:

```bash
docker compose up policy-server
```

equivalent to:

```bash
docker run --rm --gpus all --network host \
  -e CAMELO_MODE=serve \
  -e CAMELO_POLICY=lerobot \
  -e CAMELO_CHECKPOINT=ostjul/camelo-ebim-task2-act-s27a15 \
  -e CAMELO_EXTRA_ARGS="--action-layout s27a15 --state-layout s27a15 --host 0.0.0.0 --port 8767 --seed 0" \
  -e HF_HOME=/hf \
  -v ~/.cache/huggingface:/hf \
  ghcr.io/ostjul/camelo-ebim-task2-phase2-submission:v0.1.0
```

`CAMELO_MODE=serve` routes the entrypoint to `scripts/serve_policy.py`; `--seed 0` pins random/numpy/torch state at server start and on every reset. The checkpoint is fetched into the mounted Hub cache on first start — bake it into the image instead for an offline box.

**Native alternative** (no docker; `pip install -e ".[policy]"` already done):

```bash
python scripts/serve_policy.py --adapter lerobot \
  --checkpoint ostjul/camelo-ebim-task2-act-s27a15 \
  --action-layout s27a15 --state-layout s27a15 \
  --host 0.0.0.0 --port 8767 --seed 0
```

`--checkpoint` accepts a Hub repo id directly (lerobot's own `PreTrainedConfig.from_pretrained` resolves it) — no manual download step. Port 8767 is the ACT convention used throughout this repo's rig sessions.

**What the image contains.** `ros:jazzy-ros-base` (Ubuntu 24.04, Python 3.12) with this repository installed editable, **torch 2.11.0+cu128 / torchvision 0.26.0+cu128** from the PyTorch `cu128` index, and **lerobot 0.6.1** (`pip install -e ".[policy]" lerobot==0.6.1`; the build's own import check prints the resolved versions). The checkpoint was trained with the same torch 2.11.0 and lerobot 0.6.1, so the served code path is the training one. The CUDA 12.8 wheels were chosen over PyPI's default CUDA 13 build deliberately: they run on any **R570-or-newer** NVIDIA driver (CUDA 13 wheels need R580+), which covers the current GCP Deep Learning VM images either way. The image is `linux/amd64` only and was cross-built under qemu on an aarch64 box; the team has no x86 GPU host, so the GPU path was not exercised on this exact image before release — the native serving line above is the fallback if the container fails to see the GPU (`nvidia-smi` inside `docker run --gpus all … nvidia-smi` is the first thing to check).

**On a cloud GPU VM** (the organisers' setup: an A100/H100 instance on Google Cloud). Any image with the NVIDIA container toolkit works (Deep Learning VM images have it); `docker compose` needs the toolkit for the GPU reservation in `compose.yaml`, or use the `docker run --gpus all` form above. Keep port 8767 closed to the internet: the executor speaks plain `ws://`, so reach the server through an SSH tunnel from the station (`ssh -L 8767:127.0.0.1:8767 <user>@<gpu-vm>`, §7) and leave the T6 command line at `--server ws://127.0.0.1:8767`. The rollout was developed against a round trip of roughly half a second; the asynchronous inference mode with the observation-time chunk base tolerates the extra hop, and `--max-image-age-s 0.5` remains the safety gate. Bandwidth: three cameras at 20 Hz, JPEG-encoded on the wire, is on the order of 5 MB/s upstream from the station — check it once with the dummy client (§7) before a scored run. Do **not** enable the wire resize/quality flags: ACT was trained at native resolution.

## 7. Executor on the rig

The executor is `scripts/run_policy.py`, run **natively** inside the station's pixi Humble shell — never as a container (rule C-3, §3).

**Get the code onto the station** — this repository at the released tag:

```bash
git clone https://github.com/ostjul/camelo-ebim-task2-phase2-submission.git ~/camelo/camelo-ebim
cd ~/camelo/camelo-ebim && git checkout v0.1.0
```

Then, on the station, a pixi shell and an editable install. `--no-deps` keeps pip from fighting the conda-pinned `numpy==1.26` or the OpenCV that `ros-humble-cv-bridge` already provides; the import line afterwards is the check — add anything it reports missing with `pixi add` (`py-opencv`, `pillow`, `pyyaml`, `msgpack-python`, `websockets`). `opencv-python-headless` and `pillow` are core dependencies of this repo (the perception approach). This recipe follows the rig sessions' pixi setup; the `--no-deps` install itself is the recommended form, not a transcript of a measured run.

```bash
cd ~/teleoperation/station && pixi shell
source configs/tmr_laptop_env.sh
cd ~/camelo/camelo-ebim && pip install -e . --no-deps
python -c "import numpy, yaml, msgpack, websockets, PIL, cv2; print('deps ok')"
```

**Camelo's own DDS profile**, not the site's (rule C-14), then confirm the environment and the perception approach's two core deps:

```bash
make dds-profile PY=python
export FASTRTPS_DEFAULT_PROFILES_FILE=$PWD/outputs/rig/fastdds_camelo.xml
python -c "import sys, os; print(sys.executable); print('PROFILE=', os.environ.get('FASTRTPS_DEFAULT_PROFILES_FILE'), 'DOMAIN=', os.environ.get('ROS_DOMAIN_ID'), 'RMW=', os.environ.get('RMW_IMPLEMENTATION'))"
python -c "import cv2, PIL"
```

Must print the pixi python, the camelo profile, `DOMAIN= 0`, `RMW= rmw_fastrtps_cpp`.

**Cameras**, then verify the wrists negotiated the corpus resolution (a regression here silently drops to 848x480):

```bash
bash ~/teleoperation/station/start_cameras.bash
grep -o "stream_type: Color.*Width: [0-9]*, Height: [0-9]*" $(ls -t ~/cameras_*.log | head -1) | sort | uniq -c
```

expect `2 × ... Width: 640, Height: 480`.

**Arms, on the companion** (`ssh companion`), homed before any activation:

```bash
python3 ~/home_arms.py --file ~/t5_ep163_home_pose.yaml
```

**Base/table alignment** (full procedure in §9), then a camera ingest check:

```bash
python -u scripts/capture_head.py > outputs/rig/live_head_$(date +%H%M%S).log 2>&1; echo "exit=$?"
tools/rig_probes/overlay_latest.sh
python -u scripts/check_obs.py --world real --seconds 10 > outputs/rig/t3_checkobs_$(date +%H%M%S).log 2>&1; echo "exit=$?"
```

`exit=1` with the wrists reading 640x480 (not 848) is the healthy result — the exit code alone is not a pass/fail gate here (see the doc for why).

**Tunnel to the GPU box**, then a dummy client immediately before every rollout:

```bash
ssh -N -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
  -L 8767:127.0.0.1:8767 <user>@<gpu-vm>
python -u scripts/run_dummy_client.py --server ws://127.0.0.1:8767 --rate 2 --seconds 20 2>&1 | tee outputs/rig/t6_dummy_$(date +%H%M%S).log
```

Every request must answer `chunk=(21, 15)` with no reconnects.

**The rollout** — the best working run on this rig to date (`a05`, 2026-09-03): base parked by hand at the verified pose (§9) with **no base approach**, right arm only, async inference, observation-time chunk base, offset splice, replan every 12 steps, a gripper latch that holds the close once the policy commands it (`0.3:20:0.9` = latch when the commanded gripper drops below 0.3, hold the close for 20 s, release only above 0.9 — `CLOSE_BELOW[:HOLD_S[:RELEASE_ABOVE]]`), 120 s. Video on, E-stop in hand:

```bash
R=a05; TS=$(date +%H%M%S); python -u scripts/run_policy.py --world real --backend remote --server ws://127.0.0.1:8767 --action-layout s27a15 --state-layout s27a15 --task "Pick up the thermal pad and place it on the target RAM board" --arms right --start-pose file:outputs/rig/t5/start_pose_s27a15_ep163.json --start-pose-tol 0.10 --activate-arms --wait-for-activation --keepalive-hz 10 --arm-command-frame robot --rate 20 --async-inference --chunk-time-base observation --chunk-splice offset --splice-ramp-ticks 20 --replan-steps 12 --max-delta 0.04 --max-image-age-s 0.5 --gripper-latch 0.3:20:0.9 --seconds 120 --chunk-dump outputs/rig/t6a_${R}_$TS.npz --joint-csv outputs/rig/t6a_${R}_$TS.csv > outputs/rig/t6a_${R}_$TS.log 2>&1; echo "exit=$?"
```

Set `R` to the run label you are recording. On `--world real` no approach runs unless `--approach perception --approach-profile …` is passed (§8): this line expects the base already at the parked pose. The earlier cheat-sheet variant (`--replan-steps 8`, no latch, 30 s) is the regression-check line, not the grasp attempt; a proximity-gated latch (`--gripper-latch-near slot:xNNN:0.25`, `xNNN` from `slot_grasp_poses.json`) exists but was not part of the a05 run.

**Activation and hold, by contract** (`docs/CONTRACTS.md`): the companion's `joint_impedance_controller` refuses to activate unless a valid command arrived within the last 2.0 s, and shuts the whole arm launch down if 2.0 s pass with none once active. The executor therefore (1) **holds before activating** — publishes the measured pose (a zero-delta stand-in) until the controller reports active, making the activation delta `g0 == q0`; (2) **keeps alive** — republishes the last command with a fresh stamp whenever nothing has gone out for `1/--keepalive-hz` (default 10 Hz); (3) **deactivates, confirms, then stops** on every exit path, including exceptions; and (4) uses `--arm-command-frame robot` (the default, measured not assumed — the corpus's own action is already in the robot frame).

**After every rollout**, from a second companion login:

```bash
ps -eo pid,etime,args | awk '/ros2_control_n[o]de/' | wc -l; grep -c -iE 'FATAL|reflex|communication_constraints|died' ~/start_upper.log; grep -c 'Rejecting GELLO' ~/start_upper.log; ps -eo pid,etime,args | awk '/robotiq_gripper_cli[e]nt/' | wc -l    # want 4, 0, 0, 2
```

## 8. Base approach

`--approach perception` replaces only the **navigate** stage of the episode's approach FSM (pose-based `ApproachController` stays the sim default). It fuses three channels in a Kalman filter — never a raw vision loop, never blocked on one: an operator-supplied **start pose** (`--approach-start-xy-yaw`) seeds it at t=0; **relative odometry** (`S_BASE_ODOM`) integrates forward from that seed every tick; **perception** — a tabletop rectangle recovered from head-camera edges and projected through a known table AABB — corrects the estimate whenever it locks, at a per-measurement gain set from the fit's own evidence rather than a flat constant. The controller and planner always drive on the **fused** estimate, never a raw reading; the filter's odometry-based mirror rejection is what catches the AABB's 180° pose ambiguity (`solve()` is gates-only and will happily publish a mirrored lock — nothing vetoes it before the filter, and this is measured: with the prior removed from `solve()`, a round-trip test at a mirrored pose reproduces the mirror).

**The profile file**, `configs/rig/perception_munich.yaml`, is what makes this rig-specific:

| Field | What it holds |
|---|---|
| `camera` | self-calibrated ZED-M model (no `CameraInfo` on this rig): `k1 = -0.19` at the nominal 90° HFOV straightens the workbench's edges from 25 px of bow to < 1 px RMS (fit from 7 real frames); `undistort_scale: 0.85` keeps near table corners in frame. Line straightness can't fix focal length — replace with a factory/checkerboard calibration before trusting metric distances |
| `surface_v_abs: 165` | absolute brightness threshold (not the sim's floor-relative one) separating the white tabletop from the white partitions behind it with zero wall leakage, while keeping the robot's own near-white shell out of the near edge |
| `ignore_regions` | two image strips (both lower corners) masking the base's own shell, which otherwise bridges into the tabletop blob |
| `rig:` | `base_only: true` (navigate → done, no spine/arm stage), `continuous_twist: true` (`TwistStamped`, no pedal quantizer), `max_linear_mps`/`max_angular_radps: 0.1` (mirrors `contracts.REAL_BASE_*`), `kp_linear: 1.0`/`kp_yaw: 1.5`, `goal_xy_m: 0.03`/`goal_yaw_rad: 0.035` arrival tolerance, `settle_s: 1.0` |
| `walls: none` | no wall map on this rig |
| `spine_m: 0.434` | the corpus's own recorded setpoint — read it live (`scripts/read_spine.py`) before trusting it |

**Three fields are `null` and must be measured at the rig by the operator — the controller refuses to construct while `goal_xy_yaw` is `null`.** The team's access to the rig ended before this component existed, so these values were never taken; the reference rollout (§7) parked the base by hand and did not use §8 at all. Everything is expressed in the **table frame**: origin at the tabletop centre, x along the long side (the pad row), +y toward the side the robot parks on, z up, yaw 0 for the table itself.

| Field | How to get it |
|---|---|
| `table: {origin_xy: [0.0, 0.0], size_xy: [L, W], height_m: H, yaw: 0.0}` | tape measure, metres: `L` the long side (along the pad row), `W` the short side, `H` floor to tabletop. `origin_xy` stays `[0, 0]` — the frame is defined at the table centre |
| `goal_xy_yaw: [x, y, yaw]` | park the base at the T6 pose that passes the §9 overlay + pad-centroid check, then measure from the table centre to the base's `base_link` origin (the swerve controller's odom child frame — the chassis centre; `base_wire_check.sh` prints the frame ids): `x` along the row, `y` toward the robot (positive). `yaw` is where the base's forward axis points in that frame: a base squarely facing the table from the +y side has `yaw = -1.5708` (−90°, the same convention as the sim goal `TASK2_APPROACH_XY_YAW`) |
| `start_xy_yaw: [x, y, yaw]` | optional — a typical rough starting pose in the same frame as a default seed; `--approach-start-xy-yaw` overrides it per run regardless |

There is no command-line override for `table` or `goal_xy_yaw`: edit the profile (or a copy of it and pass that path to `--approach-profile`).

**Where the base contract comes from.** Everything camelo assumes about the base wire is taken from the site's own scripts, shipped verbatim in `docs/realdata/site_scripts/companion/`: `base_nudge.py` publishes `geometry_msgs/TwistStamped` on `/swerve_drive_controller/cmd_vel`, stamped from the node clock at 20 Hz, because the controller ages every command against its own clock and drops anything older than `cmd_vel_timeout` (0.5 s) — which is why a bare `ros2 topic pub` never moves this base — with 0.1 m/s / 0.1 rad/s ceilings and 0.3 s of zeros on exit; `start_base.bash` launches `swerve_drive_controller` alone (`franka_bringup tmrv0_2.launch.py`), on domain 0 over UDP-only Fast DDS, publishes odometry on `/swerve_drive_controller/odom`, and documents the 0.1 m/s² ramp. `camelo/ros/command_publisher.py` reproduces that contract (`contracts.REAL_BASE_*`, `TopicMap.base_cmd_stamped`), **but it has never been matched against the live controller by this team** — hence the two checks below, which exist precisely because nobody has run them yet.

**Operator checks, before the base is ever commanded.** From the pixi shell with `start_base.bash` running on the companion:

```bash
tools/rig_probes/base_wire_check.sh
```

confirms the message **type** on `/swerve_drive_controller/cmd_vel` and `cmd_vel_out` really is `TwistStamped` (a mismatch means the publisher never matches and the base silently never moves — flip `TopicMap.base_cmd_stamped` if so), the odom rate/frame ids, and that nothing publishes `camera_info`. Then, standing clear:

```bash
python3 tools/rig_probes/base_twist_probe.py                  # 2 s forward + back at 0.1 m/s
python3 tools/rig_probes/base_twist_probe.py --x 0.05 --s 3    # slower, longer
python3 tools/rig_probes/base_twist_probe.py --yaw 0.1 --no-return
```

This moves the base through camelo's own `CommandPublisher` — the exact path the approach will use — and prints the odometry delta over the move (expect the commanded distance within a few cm); it refuses if nothing subscribes to the command topic.

**Running it**, once `table`/`goal_xy_yaw` are measured and written into the profile:

```bash
python -u scripts/run_policy.py --world real --adapter dummy --approach-only \
  --approach perception --approach-profile configs/rig/perception_munich.yaml \
  --approach-start-xy-yaw <x,y,yaw> --approach-vision-hz 4 \
  --approach-dump outputs/rig/approach_munich_$(date +%H%M%S)
```

`--approach-only` parks the base, sends zeros on the base wire and exits — no arm activation, no hold phase, no policy (the `dummy` adapter would only hold the measured pose anyway). Drop it, and add the T6 arm flags, to continue into a rollout from the parked pose.

**This invocation has never run on the rig, and the team cannot run it before submission** — rig access ended before this component existed, so `table`/`goal_xy_yaw` are `null` in the shipped profile, the two checks above have never been executed, and no run log exists. What *is* measured: the camera/threshold calibration (fitted on real head frames recorded on this rig, §12), the base contract in the site's own scripts, and the controller logic (closed-loop against a swerve-plant model, §12). Driving the real base to a measured goal is not. Treat §8 as an optional extra at the operator's discretion; the submitted configuration is §7 with the base parked by hand and verified per §9.

**Dead-reckoning fallback and verification gate.** Vision locks are sparse (~1.6 Hz) against a ~50 Hz control loop, and localization is never gated on a lock existing: between locks — and if none is ever acquired — the fused pose is simply relative odometry propagated from the start seed, which a wrong seed biases until (unless) vision corrects it. Once the base settles, re-run the same alignment check used for every T6 rollout (§9) — head-camera overlay against the corpus's frame 0, plus the per-slot pad-centroid check — before trusting the scene for a policy rollout; the overlay's mean-pixel metric alone is known to miss a shifted/reordered pad row.

**Limitations** (`camelo/control/perception_based_navigation.md`): one known landmark/one room; the start pose is operator-supplied and unverified against a true spawn; the AABB's 180° symmetry can pass a mirrored lock through `solve()` itself; vision (~1.6 Hz) is sparse against ~50 Hz control, so drift between locks shows only in the GT-vs-fused dump; detection is heuristic (brightness/Hough), not learned; there is no live obstacle sensing and no wall map on this rig; before the first `CameraInfo` (never published here) projection falls back to a generic 60° intrinsic; and every threshold and default is tuned for this one spawn/table/cubicle.

## 9. Alignment and scoring procedure

**Base/table alignment**, every session before the first rollout and after any push of the base or table:

```bash
python -u scripts/capture_head.py > outputs/rig/live_head_$(date +%H%M%S).log 2>&1; echo "exit=$?"
tools/rig_probes/overlay_latest.sh
```

`overlay_latest.sh` overlays the newest capture against the corpus's own frame 0, printing the mean absolute pixel difference (≈30 = placed, ≈95 = roughly a metre off) plus a side-by-side and a 50/50 blend. A continuous version streams one head frame per second on the rig (`python -u scripts/stream_head.py --hz 1`) into a live-updating blend on the operator's machine (`tools/rig_probes/live_overlay.py`).

**The mean-pixel-difference overlay alone cannot see a shifted or reordered pad row** — a real finding (16 R-77): the whole-frame average hid a pad row at the wrong spacing and slot while still scoring "placed." The per-slot centroid check must run alongside it, not instead:

```bash
tools/rig_probes/pad_centroid.sh --slot xNNN
```

expects the red pad's centroid within ±12 px of that slot's pixel value (`x048`/`x051`/`x061` = 614/653/781 px at 1280x720).

**Recording a rollout's result.** `outputs/rig/munich_2026-09-01/rollouts.csv` holds one 38-column row per scored rollout (policy, checkpoint step, block, slot, verdict, `pad_picked`/`pad_placed`/`task_success`, `clamped_pct`, pose error, video path, notes); `runs.csv` holds one row per bring-up/probe session instead. Both are derived, never hand-typed, from a rollout's archived artifacts plus the operator's own judgement calls (picked/placed/contact frame):

```bash
python3 tools/rig_probes/rollout_row.py \
    --run-dir outputs/rig/munich_2026-09-01/<run-dir> \
    --seq-index <N> --picked {0,1} --placed {0,1} --contact-frame {F|none} \
    --video <path-or-description> --operator <name> \
    --policy act --checkpoint-step 100000 --block <block-id> --dry-run
```

Drop `--dry-run` once the row looks right. `score_t5_trace.py` recomputes a joint trace's own numbers rather than trusting the runner's printed summary, writing `score.txt`: row/column counts, phases present, the commanded-row time range, per-joint max `|measured − commanded|` (and the worst joint), `clamped_pct`, and — given the recorded action `.npy` too — per-joint max error against it at the same tick:

```bash
python3 tools/rig_probes/score_t5_trace.py outputs/rig/munich_2026-09-01/<run-dir>/<trace>.csv > outputs/rig/munich_2026-09-01/<run-dir>/score.txt
```

## 10. Results and known limitations

**What is proven.** The closed-loop remote-policy pipeline works on the robot end to end: the 20 Hz control loop with async inference, keep-alive and self-deactivation runs cleanly (activation → first command in 0.56 s, 0 starved/dropped/stale chunks across five 30 s regression rollouts), the parity gate passes on the serving box, the gripper's wire units are correct end to end, and base/table placement is reproducible by head-camera overlay. Three site confounds — wrist camera resolution, the arm controller's velocity cap, wrist auto-exposure vs. the recording session — were found, fixed, and each independently confirmed **not** to be the reason the policy wasn't grasping once closed.

**Every executor variant tried was measured, not guessed at**: wall-clock index splice, nearest-pose splice, leash-to-measured, synchronous plan-then-execute, and — best so far — async inference with an observation-time chunk base and an offset splice. That combination reached the demonstration's own grasp pose (L2 ≈ 0.08–0.09 rad at 24–48 s into a 120 s run) and, once, closed on the pad — the gripper command reached the demo's own closed value with the pad between the fingers — but the close ramps over ≈3 s (the demos close in ≈1 s), long enough for the arm to drift off the pad first.

**No scored grasp has succeeded on this rig.** Across the twelve closed-loop rollouts run in the development window that produced this checkpoint there is exactly one accidental full task sequence (approach, close, transport) that did not reproduce, and zero picks. Scored block B2 (protocol §2, target 15 rollouts) ran three before the operator stopped it — all **ABSENT** by mechanism verdict — on an executor line since superseded by the offset-splice line above; it needs restarting on that line, most likely with a proximity-gated gripper latch, before a task-success rate can honestly be reported. A threshold-only latch, tried once, either fires early (on the pre-shape dip) or never fires; a proximity-gated variant exists in code but is untested on hardware.

The reference configuration is the `a05` line in §7 (base parked by hand, no approach). **Honest one-line summary:** the approach-and-pose problem is solved; gripper-close timing is not — read this checkpoint as an approach/pose demonstration, not a grasp-success rate, until block B2 is re-run and a `pad_placed=1` row exists in `rollouts.csv`.

**The base approach is offline-validated only.** Its unit and closed-loop logic are tested against real captured head frames and a swerve-plant model (§12), and its two wire checks (§8) are written but have never been run: the team's rig access ended before the component existed, so `table`/`goal_xy_yaw` are `null` and it has never driven this base. The base contract it implements is taken from the site's own `base_nudge.py`/`start_base.bash` (§8, §11).

## 11. Helper files

| Path | Purpose | Source of truth |
|---|---|---|
| `docs/realdata/site_scripts/companion/start_upper.bash` | Arms + grippers bring-up (`joint_impedance_controller` per arm, gripper clients); log `~/start_upper.log` | companion `~` (snapshot here for inspection) |
| `docs/realdata/site_scripts/companion/start_base.bash` | Base-only bring-up (`swerve_drive_controller`, no arms/spine); log `~/start_base.log` | companion `~` |
| `docs/realdata/site_scripts/companion/base_nudge.py` | Reference `TwistStamped` base command (why `ros2 topic pub` never moves this base) | companion `~` |
| `docs/realdata/site_scripts/companion/home_arms.py` | PTP-homes both arms from a YAML pose file (`t5_ep163_home_pose.yaml`, `t5_ep009_home_pose.yaml`) | companion `~` |
| `docs/realdata/site_scripts/ebimhp_station/start_cameras.bash` | Head (ZED-M) + both wrist (D405) camera launch, incl. the site's resolution/exposure fix | station `~/teleoperation/station` git checkout |
| `docs/realdata/site_scripts/ebimhp_station/configs/d405_color_640x480.yml` | Wrist camera config: `depth_module.color_profile: 640x480x30`, fixed exposure 5000, auto-exposure off | station git checkout |
| `docs/realdata/site_scripts/ebimhp_station/configs/sync_robot_clock.sh` | Companion↔station clock sync (`--check` always safe; a real correction needs arms down) | station git checkout |
| `docs/realdata/site_scripts/ebimhp_station/configs/tmr_laptop_env.sh` | Renders the site's own Fast DDS profile and shared launcher env | station git checkout |
| `tools/rig_probes/base_wire_check.sh` | Operator check (never yet run on the rig): base command/applied-twist message type, odom rate/frame ids, absence of `camera_info` | this repo |
| `tools/rig_probes/base_twist_probe.py` | Moves the base through camelo's own `CommandPublisher`, prints the odometry delta over the move (never yet run on the rig) | this repo |
| `tools/rig_probes/overlay_head.py` / `overlay_latest.sh` | Base/table alignment: live head frame vs. corpus frame 0, side-by-side + blend + mean `|diff|` | this repo |
| `tools/rig_probes/live_overlay.py` | Continuous streaming version of the same overlay | this repo |
| `tools/rig_probes/pad_centroid.py` / `pad_centroid.sh` | Per-slot red/teal pad centroid check against frozen slot pixel positions | this repo |
| `tools/rig_probes/rollout_row.py` | Derives one `rollouts.csv` row from an archived rollout's own log/CSV/slot data + operator judgement | this repo |
| `tools/rig_probes/score_t5_trace.py` | Recomputes per-joint tracking error, `clamped_pct`, writes `score.txt` from a joint trace | this repo |
| `tools/rig_probes/record_rig_bag.sh` / `play_rig_bag.sh` / `rig_bag_topics.py` | Records/replays a small rosbag of camelo's own real-world topic set (derived from `camelo/contracts.py`, never hand-duplicated) | this repo |
| `tools/rig_parity_check.py` | The parity gate: adapter-under-test vs. the training pipeline, on real recorded frames | this repo |
| `outputs/rig/t5/start_pose_s27a15_ep163.json` / `..._ep009.json` | Frozen per-episode start poses for `--start-pose file:...` | generated offline from the corpus, copied to the rig |
| `outputs/rig/t5/ep163_frame0_head.png` | Reference head frame for the alignment overlay | generated offline from the corpus |
| `outputs/rig/t5/ep163_frames/` | Per-frame PNGs; doubles as the real-frame regression set for the perception detector (§12) | generated offline from the corpus |
| `outputs/rig/munich_2026-09-01/slot_grasp_poses.json` | Per-slot mean demo grasp pose, used by `--gripper-latch-near slot:xNNN:radius` | derived offline from the corpus |
| `outputs/rig/munich_2026-09-01/runs.csv` / `rollouts.csv` | Session log / scored-rollout ledger (§9) | this repo, appended on the rig |

## 12. Reproducing offline

```bash
make test
```

runs the full pytest suite (`pytest tests/ -q`), including the perception approach's own unit and regression tests. Two are worth running alone:

```bash
pytest tests/test_perception_profile.py -q       # real-frame perception test
pytest tests/test_approach_rig_profile.py -q     # closed-loop rig test
```

The first loads `configs/rig/perception_munich.yaml` and detects the tabletop in real Munich head frames tracked under `outputs/rig/t5/ep163_frames/` (it skips itself, rather than failing, if that frame is missing). The second drives `PerceptionApproachController` in rig/base-only mode against a plant modelling the swerve controller's own 0.1 m/s / 0.1 rad/s clamp and 0.1 m/s² ramp, checking that the base actually **arrives**, not just that a command was issued.

And the parity gate against a real checkpoint and dataset — the exact command is in §5.

## 13. Licence / contact

Apache-2.0 (`LICENSE`). Team **Camelo** — point of contact on the submission issue.

---

Release `v0.1.0` — 2026-09-05 (build `2e37879`).
