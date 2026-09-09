# EBiM Task 2 Real-Robot Submission — Team Camelo

Team **Camelo**'s submission for **EBiM Task 2** — *"Pick up the thermal pad
and place it on the target RAM board"* — run on the real robot at the Munich
rig: a mobile dual-Franka-FR3 "TMR" base, ZED-M head camera, two RealSense
D405 wrist cameras. Four parts: an **ACT** policy checkpoint served from
Hugging Face (`ostjul/camelo-ebim-task2-act-s27a15`); an **executor** that
runs natively (not containerized) on the station; an optional
**perception-based base approach** — off by default, offline-validated only,
never driven this base; and a set of **rig helper files**. **Read Status
before running anything on hardware: no scored grasp has succeeded on this
rig yet.**

Image: `ghcr.io/ostjul/camelo-ebim-task2-phase2-submission:v0.1.0` · Release `v0.1.0` — 2026-09-09 (build `3e76de4`).

One block per step: **terminal → command → one sentence.**

## Terminal legend

**M** = your Mac/workstation, **A** = companion (`ssh companion` from the
station, arm/gripper/base stack), **C** = second companion login, **W** =
station watcher, **T** = station tunnel, **P** = station pixi shell
(everything camelo), **K** = station camera launcher, **G** = the GPU box.

## Policy server (G)

**G — bring the server up (compose form).**
```bash
docker compose up policy-server
```
One service, `ghcr.io/ostjul/camelo-ebim-task2-phase2-submission:v0.1.0`, serving `ostjul/camelo-ebim-task2-act-s27a15` over `ws://0.0.0.0:8767`.

**G — equivalent `docker run` form.**
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
Same server without compose; the checkpoint id is a Hub repo id, no manual download step needed.

**G — pre-fill the Hub cache (optional, for an offline box).**
```bash
hf download ostjul/camelo-ebim-task2-act-s27a15
```
Run this once before an offline session so the server does not fetch on first start.

Confirm it is listening from the station: see "Tunnel and dummy client"
below — every request must answer `chunk=(21, 15)` with no reconnects.

## Install on the station (P, once)

**P — clone the released tag.**
```bash
git clone https://github.com/ostjul/camelo-ebim-task2-phase2-submission.git ~/camelo/camelo-ebim
cd ~/camelo/camelo-ebim && git checkout v0.1.0
```
Gets this repository onto the station at the exact released commit.

**P — pixi shell, editable install, dep check.**
```bash
cd ~/teleoperation/station && pixi shell
source configs/tmr_laptop_env.sh
cd ~/camelo/camelo-ebim && pip install -e . --no-deps
python -c "import numpy, yaml, msgpack, websockets, PIL, cv2; print('deps ok')"
```
`--no-deps` keeps pip from fighting the conda-pinned numpy/OpenCV; add anything the import line reports missing with `pixi add`.

**P — camelo's own DDS profile, then verify the environment.**
```bash
make dds-profile PY=python
export FASTRTPS_DEFAULT_PROFILES_FILE=$PWD/outputs/rig/fastdds_camelo.xml
python -c "import sys, os; print(sys.executable); print('PROFILE=', os.environ.get('FASTRTPS_DEFAULT_PROFILES_FILE'), 'DOMAIN=', os.environ.get('ROS_DOMAIN_ID'), 'RMW=', os.environ.get('RMW_IMPLEMENTATION'))"
```
Must print the pixi python, the camelo profile, `DOMAIN= 0`, `RMW= rmw_fastrtps_cpp`; never the site's own profile (a foreign participant's default socket buffer silently drops fragmented camera frames).

## Phase 0 bring-up (A/C/K)

**A — clock (arms DOWN first).**
```bash
cd ~/teleoperation/station && ./configs/sync_robot_clock.sh && ./configs/sync_robot_clock.sh --check
```
Companion clock must be within 0.05 s of the station; redo every ~30 min with the arms down.

**A — bring-up, detached.**
```bash
ssh companion
TMR_WS=~/ros2_ws setsid nohup bash ~/start_upper.bash --restart > ~/start_upper.log 2>&1 < /dev/null &  tail -f ~/start_upper.log
```
Wait ~40 s, then the four numbers below must read `4 0 2 1`.

**C — the four numbers.**
```bash
ps -eo pid,etime,args | awk '/ros2_control_n[o]de/' | wc -l; grep -c -iE 'FATAL|reflex|communication_constraints|died' ~/start_upper.log; ps -eo pid,etime,args | awk '/robotiq_gripper_cli[e]nt/' | wc -l; ps -eo pid,args | awk '/start_upp[e]r/' | wc -l    # want 4, 0, 2, 1
```
Controllers, fault lines, gripper clients, launcher.

**K — cameras.**
```bash
bash ~/teleoperation/station/start_cameras.bash
```
Starts the ZED head camera and both D405 wrists.

**K — verify wrist width (every session).**
```bash
grep -o "stream_type: Color.*Width: [0-9]*, Height: [0-9]*" $(ls -t ~/cameras_*.log | head -1) | sort | uniq -c
```
Expect `2 × ... Width: 640, Height: 480`; an `848` means the wrist launch regressed.

## Watcher (W, plain shell on the station)

**W — pick the PIDs once.**
```bash
TS=$(date +%Y%m%d_%H%M%S); ps -eo pid,args | awk '/gello_pub[l]isher|realsense2_camera_n[o]de|zed_open_captur[e]|pedal_state_pub[l]isher|mode_mana[g]er|spine_brid[g]e|spine_state_pub[l]isher|mobile_base_state_brid[g]e|labs_pedal_brid[g]e/' | tee ~/camelo/camelo-ebim/outputs/rig/t2_pids_$TS.txt; PIDS=$(awk '{print $1}' ~/camelo/camelo-ebim/outputs/rig/t2_pids_$TS.txt | paste -sd,); echo "$PIDS"
```
Lists the station's camera/bridge processes; the echo must print a comma list of PIDs.

**W — the loop (leave it running).**
```bash
while :; do { date +%T; ps -o pid,rss,etime,cmd -p "$PIDS"; free -m | awk '/^Mem:/{print "MEM used="$3" free="$4}'; echo; } >> ~/camelo/camelo-ebim/outputs/rig/t2_rss_$TS.log; sleep 5; done
```
Every watched RSS must stay flat (±50 MB); abort the session at +1 GB on any PID.

## Arms and scene (A/P/M)

**A — home both arms (E-stop in hand).**
```bash
python3 ~/home_arms.py --file ~/t5_ep163_home_pose.yaml
```
Wait for `left: at home.` and `right: at home.` (or "already at home").

**P — head capture for the base/table overlay.**
```bash
python -u scripts/capture_head.py > outputs/rig/live_head_$(date +%H%M%S).log 2>&1; echo "exit=$?"
```
Repeat after every push of the base or table.

**M — overlay the newest capture against the corpus's frame 0.**
```bash
tools/rig_probes/overlay_latest.sh
```
Prints the mean absolute pixel difference (≈30 = placed, ≈95 = roughly a metre off) plus a side-by-side and a blend.

**Scene and pad row.** Red pad leftmost, three teal pads to its right, ending
near the fixture. **The mean-`|diff|` overlay alone cannot see a
shifted/reordered pad row** — check the per-slot centroid too before trusting
a "placed" overlay score:
```bash
tools/rig_probes/pad_centroid.sh --slot xNNN
```
`xNNN` = this rollout's slot; expect the red-pad centroid within ±12 px of that slot's pixel value (`x048`/`x051`/`x061` = 614/653/781 px).

**P — spine height vs. the corpus.**
```bash
python -u outputs/rig/read_spine.py
```
Prints `spine <live> mm vs corpus 434 mm: <diff> -> OK|MISMATCH`; jog the spine until OK, then recapture the head view.

**P — cameras at the wire rate and shape.**
```bash
python -u scripts/check_obs.py --world real --seconds 10 > outputs/rig/t3_checkobs_$(date +%H%M%S).log 2>&1; echo "exit=$?"
```
`exit=1` with wrists at 640×480 is the healthy result; `exit=3` means the wrists are still 848 wide.

## Tunnel and dummy client (T/P)

**T — tunnel to the GPU box (plain shell, leave it open).**
```bash
ssh -N -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
  -L 8767:127.0.0.1:8767 <user>@<gpu-vm>
```
Prints nothing while healthy; if it ever exits, restart it before the next rollout.

**P — dummy client, immediately before EVERY rollout.**
```bash
python -u scripts/run_dummy_client.py --server ws://127.0.0.1:8767 --rate 2 --seconds 20 2>&1 | tee outputs/rig/t6_dummy_$(date +%H%M%S).log
```
Every request must answer `chunk=(21, 15)` with no reconnects, else the tunnel or the server is down.

## Rollout (P)

**P — the reference rollout (`a05`): base parked by hand, right arm only,
async inference, observation-time chunk base, offset splice, replan every 12
steps, gripper latch, 120 s. Video on, E-stop in hand.**
```bash
R=a05; TS=$(date +%H%M%S); python -u scripts/run_policy.py --world real --backend remote --server ws://127.0.0.1:8767 --action-layout s27a15 --state-layout s27a15 --task "Pick up the thermal pad and place it on the target RAM board" --arms right --start-pose file:outputs/rig/t5/start_pose_s27a15_ep163.json --start-pose-tol 0.10 --activate-arms --wait-for-activation --keepalive-hz 10 --arm-command-frame robot --rate 20 --async-inference --chunk-time-base observation --chunk-splice offset --splice-ramp-ticks 20 --replan-steps 12 --max-delta 0.04 --max-image-age-s 0.5 --gripper-latch 0.3:20:0.9 --seconds 120 --chunk-dump outputs/rig/t6a_${R}_$TS.npz --joint-csv outputs/rig/t6a_${R}_$TS.csv > outputs/rig/t6a_${R}_$TS.log 2>&1; echo "exit=$?"
```
Set `R` to the run label being recorded; expect it to reach the demonstration's own grasp pose, not a confirmed pick — no approach runs on `--world real` unless `--approach perception …` is passed (see below).

**C — after every rollout.**
```bash
ps -eo pid,etime,args | awk '/ros2_control_n[o]de/' | wc -l; grep -c -iE 'FATAL|reflex|communication_constraints|died' ~/start_upper.log; grep -c 'Rejecting GELLO' ~/start_upper.log; ps -eo pid,etime,args | awk '/robotiq_gripper_cli[e]nt/' | wc -l    # want 4, 0, 0, 2
```
Then paste the tail of the rollout log for scoring.

**P — pass table for the rollout (grep the log rather than eyeballing it).**
```bash
grep 'policy chunk' outputs/rig/t6a_${R}_$TS.log | tail -20
```
```bash
grep -E 'rollout stats:|leash telemetry:' outputs/rig/t6a_${R}_$TS.log
```
| field | where | pass |
| --- | --- | --- |
| `splice=` | each `policy chunk` line | 0 … −9, never ≥ +2 on many chunks |
| `jump=` | each `policy chunk` line | ≪ 0.05 |
| `lead=` | each `policy chunk` line | ≤ 0.15 |
| `starved_ticks` | `rollout stats:` | ≈ 0 (of ~587) |
| `leash_active_pct` | `leash telemetry:` | report it; high = the controller can't follow, not the policy |
| `cmd_lead_rad_max` | `rollout stats:` / `leash telemetry:` | ≤ 0.15 |
| `arrival_jump_rad_max` | `rollout stats:` | ≪ 0.05 |
| `clamped_pct` | `rollout stats:` | low |
| visual | video | no wiggle during the hover |

**M/P — score and record the row, `--dry-run` first.**
```bash
python3 tools/rig_probes/rollout_row.py \
    --run-dir outputs/rig/munich_2026-09-01/<run-dir> \
    --seq-index <N> --picked {0,1} --placed {0,1} --contact-frame {F|none} \
    --video <path-or-description> --operator <name> \
    --policy act --checkpoint-step 100000 --block <block-id> --dry-run
```
```bash
python3 tools/rig_probes/score_t5_trace.py outputs/rig/munich_2026-09-01/<run-dir>/<trace>.csv > outputs/rig/munich_2026-09-01/<run-dir>/score.txt
```
Drop `--dry-run` once the printed row looks right; `score_t5_trace.py` recomputes the joint trace's own numbers rather than trusting the runner's printed summary.

## Optional: base approach (P)

`--approach perception` is off by default and offline-validated only — it has
never driven this base. Two fields in `configs/rig/perception_munich.yaml`
are `null` and must be measured by the operator before it will construct:

| Field | How to get it |
| --- | --- |
| `table: {origin_xy: [0.0, 0.0], size_xy: [L, W], height_m: H, yaw: 0.0}` | tape measure, metres: `L` the long side (along the pad row), `W` the short side, `H` floor to tabletop |
| `goal_xy_yaw: [x, y, yaw]` | park the base at the verified pose, then measure from the table centre to `base_link`; a base squarely facing the table has `yaw = -1.5708` |

**P — `--approach-only` test, once both fields are filled in.**
```bash
python -u scripts/run_policy.py --world real --adapter dummy --approach-only \
  --approach perception --approach-profile configs/rig/perception_munich.yaml \
  --approach-start-xy-yaw <x,y,yaw> --approach-vision-hz 4 \
  --approach-dump outputs/rig/approach_munich_$(date +%H%M%S)
```
Parks the base and exits — no arm activation, no policy; drop `--approach-only` to continue into a rollout from the parked pose. This exact invocation has never run on the rig.

## Wind-down (A)

Ctrl+C in **W**, **K**, **T**; then in **A**:
```bash
pkill -INT -f start_upper.bash; sleep 8; ps -eo pid,etime,args | awk '/robotiq_gripper_cli[e]nt|ros2_control_n[o]de|ros2 laun[c]h|start_upp[e]r/' | cut -c1-100    # want nothing; kill any leftover PID by hand, then: exit
```
Arms down last, after the cameras and the tunnel.

## Status

- The closed-loop remote-policy pipeline works end to end: activation to first command in 0.56 s, 0 starved/dropped/stale chunks across five 30 s regression rollouts.
- The adapter-vs-training parity gate passed on the serving box (internal tooling, not shipped here); gripper wire units are correct end to end; base/table placement is reproducible by head-camera overlay.
- Three site confounds (wrist camera resolution, arm velocity cap, wrist auto-exposure) were found, fixed, and each confirmed not to be why the policy wasn't grasping.
- The best executor line (async inference, observation-time chunk base, offset splice) reached the demonstration's own grasp pose (L2 ≈ 0.08–0.09 rad at 24–48 s into a 120 s run).
- That line closed on the pad once in an unscored probe, but the close ramps over ≈3 s against the demos' ≈1 s — long enough for the arm to drift off the pad first.
- **No scored grasp has succeeded on this rig.** Zero picks across the twelve closed-loop rollouts run in the development window.
- Scored block B2 (target 15 rollouts) stopped after 3 — all ABSENT by mechanism verdict — on an executor line since superseded; it needs restarting on the offset-splice line, most likely with a proximity-gated gripper latch.
- The reference configuration is the `a05` line above: base parked by hand, no approach.
- The base approach is offline-validated only: its two wire checks have never been run, `table`/`goal_xy_yaw` are `null`, and it has never driven this base.
- Honest summary: the approach-and-pose problem is solved; gripper-close timing is not.

## Repository layout

Only what the station executor needs. The model lives on Hugging Face, the
policy server and its torch/lerobot stack live inside the image, and the
training code, tests and internal protocol docs stay in the private repo.

```
.
├── README.md                    # this file: the operator cheat sheet
├── Dockerfile                   # FROM ghcr.io/ostjul/camelo-ebim-task2-phase2-submission:v0.1.0 — pins the released policy-server image
├── compose.yaml                 # one service: policy-server (GPU, ws://…:8767)
├── Makefile                     # make dds-profile, make check-obs
├── pyproject.toml / LICENSE     # pip install -e . on the station; Apache-2.0
├── camelo/                      # the executor: contracts, control (approach, perception,
│                                #   chunk executor, gripper latch), policy client (remote
│                                #   backend, s27a15 layout), ros I/O, runner
├── scripts/                     # run_policy.py (the executor), run_dummy_client.py,
│                                #   check_obs.py, capture_head.py, stream_head.py,
│                                #   read_spine.py, render_dds_profile.py
├── configs/rig/perception_munich.yaml   # camera model + tabletop thresholds + rig drive
├── tools/rig_probes/            # alignment, scoring, bag and base-wire probes (table below)
└── outputs/rig/                 # reference poses and frames, slot grasp poses, ledgers (table below)
```

## Helper files

| Path | What it is |
| --- | --- |
| `tools/rig_probes/base_wire_check.sh` | Operator check (never yet run on the rig): base command/applied-twist message type, odom rate/frame ids |
| `tools/rig_probes/base_twist_probe.py` | Moves the base through camelo's own `CommandPublisher`, prints the odometry delta (never yet run on the rig) |
| `tools/rig_probes/overlay_head.py` / `overlay_latest.sh` | Base/table alignment: live head frame vs. corpus frame 0, side-by-side + blend + mean `|diff|` |
| `tools/rig_probes/live_overlay.py` | Continuous streaming version of the same overlay |
| `tools/rig_probes/pad_centroid.py` / `pad_centroid.sh` | Per-slot red/teal pad centroid check against frozen slot pixel positions |
| `tools/rig_probes/rollout_row.py` | Derives one `rollouts.csv` row from an archived rollout's log/CSV/slot data + operator judgement |
| `tools/rig_probes/score_t5_trace.py` | Recomputes per-joint tracking error, `clamped_pct`, writes `score.txt` from a joint trace |
| `tools/rig_probes/record_rig_bag.sh` / `play_rig_bag.sh` / `rig_bag_topics.py` | Records/replays a small rosbag of camelo's own real-world topic set |
| `outputs/rig/t5/start_pose_s27a15_ep163.json` / `..._ep009.json` | Frozen per-episode start poses for `--start-pose file:...` |
| `outputs/rig/t5/ep163_frame0_head.png` | Reference head frame for the alignment overlay |
| `outputs/rig/t5/ep163_frames/` | Per-frame PNGs; doubles as the real-frame regression set for the perception detector |
| `outputs/rig/munich_2026-09-01/slot_grasp_poses.json` | Per-slot mean demo grasp pose, used by `--gripper-latch-near slot:xNNN:radius` |
| `outputs/rig/munich_2026-09-01/runs.csv` / `rollouts.csv` | Session log / scored-rollout ledger |

## Licence / citation

Apache-2.0 (`LICENSE`). Team **Camelo** — point of contact on the submission issue.
