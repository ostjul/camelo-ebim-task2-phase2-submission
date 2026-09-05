# T6 cheat sheet — 2026-09-03

One block per step: **terminal → command → one sentence.** Details and gates
are in [16 §3.2](16_RIG_TEST_PROTOCOL.md). Terminals: **M** = Mac,
**A** = companion (`ssh companion` from ebimHP, stack), **C** = second
companion login, **W** = ebimHP watcher, **T** = ebimHP tunnel, **P** = ebimHP
pixi shell (everything camelo), **K** = ebimHP camera launcher.

## Phase 0 (done today: clock synced, `4 0 2 1`, cameras up)

**A — clock (arms DOWN first).**
```bash
cd ~/teleoperation/station && ./configs/sync_robot_clock.sh && ./configs/sync_robot_clock.sh --check
```
Companion clock must be within 0.05 s of ebimHP; redo every ~30 min with the arms down.

**A — bring-up, detached.**
```bash
ssh companion
TMR_WS=~/ros2_ws setsid nohup bash ~/start_upper.bash --restart > ~/start_upper.log 2>&1 < /dev/null &  tail -f ~/start_upper.log
```
Wait ~40 s, then the four numbers must read `4 0 2 1`.

**C — the four numbers.**
```bash
ps -eo pid,etime,args | awk '/ros2_control_n[o]de/' | wc -l; grep -c -iE 'FATAL|reflex|communication_constraints|died' ~/start_upper.log; ps -eo pid,etime,args | awk '/robotiq_gripper_cli[e]nt/' | wc -l; ps -eo pid,args | awk '/start_upp[e]r/' | wc -l    # want 4, 0, 2, 1
```
Controllers, fault lines, gripper clients, launcher.

**K — cameras.**
```bash
bash ~/teleoperation/station/start_cameras.bash
```
```bash
cd ~/teleoperation/station && pixi run cameras
```

Wrists reach 640×480 via `config_file:=/home/ebim/teleoperation/station/configs/d405_color_640x480.yml` on both wrist launches (U-21 CLOSED 2026-09-03, R-73; `enable_depth:=true` does not move colour — do not use it). Start with `2>&1 | tee ~/cameras_$(date +%H%M%S).log` so the width check below has a log to grep.

**K — verify wrist width (do this every session, before T6-4).**
```bash
grep -o "stream_type: Color.*Width: [0-9]*, Height: [0-9]*" $(ls -t ~/cameras_*.log | head -1) | sort | uniq -c
```
Expect `2 × ... Width: 640, Height: 480`; a `848` in the count means the wrist launch args regressed.

## Watcher (W, plain shell on ebimHP)

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

## Deploy (M)

**M — rsync from a clean tree (`git status --short` empty).**
```bash
cd ~/workspace/camelo-ebim && git rev-parse HEAD > .deployed-commit && rsync -av --delete --exclude .git --exclude outputs --exclude .venv --exclude __pycache__ --exclude '*.pyc' --exclude .claude ./ ebim@192.168.0.5:camelo/camelo-ebim/ && ssh ebim@192.168.0.5 'cat ~/camelo/camelo-ebim/.deployed-commit'
```
The printed SHA must equal `git rev-parse HEAD`.

## Pixi environment (P, once per shell, in this order)

```bash
cd ~/teleoperation/station && pixi shell
```
```bash
source configs/tmr_laptop_env.sh
```
```bash
cd ~/camelo/camelo-ebim && make dds-profile PY=python
```
```bash
export FASTRTPS_DEFAULT_PROFILES_FILE=$PWD/outputs/rig/fastdds_camelo.xml
```
```bash
python -c "import sys, os; print(sys.executable); print('PROFILE=', os.environ.get('FASTRTPS_DEFAULT_PROFILES_FILE'), 'DOMAIN=', os.environ.get('ROS_DOMAIN_ID'), 'RMW=', os.environ.get('RMW_IMPLEMENTATION'))"
```
Must print the pixi python, the camelo profile, `DOMAIN= 0`, `RMW= rmw_fastrtps_cpp`.
Then `python -c "import cv2, PIL"` — the perception approach (`--approach perception`)
needs both (core deps since PR #38); the station's `ros-humble-cv-bridge` normally
brings OpenCV, otherwise `pixi add py-opencv pillow` (numpy stays 1.26).

## Arms and scene

**A — home both arms to episode 163 frame 0 (E-stop in hand).**
```bash
python3 ~/home_arms.py --file ~/t5_ep163_home_pose.yaml
```
Wait for `left: at home.` and `right: at home.` (or "already at home").

**P — head capture for the base/table overlay.**
```bash
python -u scripts/capture_head.py > outputs/rig/live_head_$(date +%H%M%S).log 2>&1; echo "exit=$?"
```
Writes into a per-capture directory under `outputs/rig/` (not a single `live_head_<HHMMSS>.png` file — harmless, `tools/rig_probes/overlay_latest.sh` finds the newest capture anywhere under `outputs/rig`, commit `ec52a90`); repeat after every push of the base or table.

**M — overlay the newest capture on episode 163's frame 0.**
```bash
tools/rig_probes/overlay_latest.sh
```
Fetches the capture, prints `mean|diff|` (≈30 = placed, ≈95 = far off) and opens the side-by-side and the blend.

**Scene:** red pad leftmost at x ≈ 0.48 of the head image, three teal pads to its right at x ≈ 0.545 / 0.60 / 0.655, ending near the fixture. **The mean-`|diff|` overlay check does not catch a shifted/reordered pad row** (R-77) — also check the pad-row placement itself (episode 163 frame 0: RED then three teals at x ≈ 615/697/772/839 px, 97 px spacing) before trusting a "placed" overlay score.

### Alignment Visualization (optional, once the two steps above work)

Continuous version of the manual capture/overlay pair above: the rig streams
one head frame per second to a fixed file, and the Mac shows it live-blended
against the reference in one auto-refreshing window, so you can watch
alignment settle while nudging the base/table instead of re-running
`overlay_latest.sh` after every push.

**One-time, M — passwordless SSH to the rig.** `live_overlay.py` polls in a
loop with `BatchMode=yes` (it can't pause for a password prompt), so a
missing key fails every poll with a bare `scp: Connection closed` rather than
prompting:
```bash
ssh-copy-id ebim@192.168.0.5   # skip if `ssh ebim@192.168.0.5 echo ok` already needs no password
```
`ssh-copy-id` isn't guaranteed on a stock Mac (unlike Linux, it doesn't always
ship with macOS's OpenSSH). If the shell says `command not found`:
```bash
cat ~/.ssh/id_ed25519.pub | ssh ebim@192.168.0.5 'mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys'
```
(swap `id_ed25519.pub` for whatever `ls ~/.ssh/*.pub` shows if there's no ed25519 key).

**P — second pixi shell (repeat the Pixi environment steps above in a new terminal; keep the original P free).**
```bash
python -u scripts/stream_head.py --hz 1
```
Overwrites `outputs/rig/t5/live_head_latest.png` in place about once a
second. Leave it running only while actively placing the scene, then Ctrl+C
it before `check_obs`, the dummy client, or the rollout — it's an extra
full-rate camera decode competing on the station, on top of the one those
steps already pay for.

**M — live view.**
```bash
tools/rig_probes/live_overlay.py                                   # vs episode 163
tools/rig_probes/live_overlay.py outputs/rig/t5/ep009_frame0_head.png
```
Polls the streamed frame over plain `scp` (`RIG_HOST` overrides the default
host, same as `overlay_latest.sh`) and redraws the blend + `mean|diff|` in
the window title roughly once a second; a dropped poll just leaves the last
good frame up with the scp error in the title instead of closing the window.
Close the window or Ctrl+C to stop. If the rig-side checkout isn't at the
default `~/camelo/camelo-ebim` (e.g. a dev checkout like
`camelo/camelo-ebim-viz`), point at it with `RIG_REMOTE_DIR` or
`--remote-dir`. Needs `matplotlib` on the Mac's python3 in addition to the
numpy/PIL `overlay_latest.sh` already needs.

### Spine height and camera checks

**P — spine height vs the corpus (434 mm on every recorded frame).**
```bash
python -u outputs/rig/read_spine.py
```
Prints `spine <live> mm vs corpus 434 mm: <diff> -> OK|MISMATCH`; jog the spine until OK, then recapture the head view.

**P — cameras at the wire rate and shape.**
```bash
python -u scripts/check_obs.py --world real --seconds 10 > outputs/rig/t3_checkobs_$(date +%H%M%S).log 2>&1; echo "exit=$?"
```
`exit=3` means the wrists are still 848 wide (pipeline runs need `--allow-camera-shape-mismatch`); `exit=1` means shapes are on contract. Wrists are now 640×480 through `configs/d405_color_640x480.yml` (U-21 CLOSED, R-73), so a healthy session should exit 1 — `--allow-camera-shape-mismatch` is dropped from the T6-6 command below; only add it back if this step still exits 3.

## Recording a debug bag

Small rosbag of camelo's own real-world contract (`tools/rig_probes/rig_bag_topics.py`,
derived from `camelo/contracts.py` — never a hand-maintained duplicate), for replaying
off-rig later. Not `record_bag.bash` — that's the LABS dataset recorder and takes the
docker/`gello-humble` path, which is a camelo container participant next to live station
nodes (banned by C-3, the R-22/R-23 OOM). This one runs **native pixi only** and refuses
otherwise.

**P — check topics before recording.**
```bash
tools/rig_probes/record_rig_bag.sh --check
```
Every topic should show `ok`; spine/base topics are expected `MISSING` on the arms-only `start_upper.bash` stack.

**P — record a small bag.**
```bash
tools/rig_probes/record_rig_bag.sh --seconds 10
```
~1.4 GB for all three cameras at 10 s (`--cameras head` or `--cameras none` to shrink it). Refuses to run outside the pixi shell, without camelo's DDS profile (C-14 — the site profile drops the wrists to 10–15 Hz), or off domain 0. Prints the bag path and `ros2 bag info` on exit; zero-message topics there are the ones to explain before trusting the bag.

**M — copy it off and replay locally.**
```bash
rsync -av --info=progress2 ebim@192.168.0.5:camelo/camelo-ebim/outputs/rig/bags/<bag> outputs/rig/bags/
tools/rig_probes/play_rig_bag.sh outputs/rig/bags/<bag>
```
Second terminal, against the replayed stream:
```bash
./scripts/ros_compose.sh run --rm camelo-ros python3 scripts/check_obs.py --world real --seconds 10
```
Loops by default (`--once` to play through once and stop) — it keeps playing
past `check_obs`'s own window, so there's no need to remember a flag for that.
Unset `FASTRTPS_DEFAULT_PROFILES_FILE` first if the shell still carries the
rig's profile — off-rig it whitelists an interface this box doesn't have, and
playback and `check_obs` silently never discover each other.

**M — visualize in Foxglove.** Opening the `.mcap` file directly (drag it into
Foxglove Studio, or File → Open Local File) needs nothing below — it's a static
file, Foxglove reads mcap natively. Below is for watching the **live replay**
(looping by default, above) update in real time instead.

**One-time, M — enable Docker Desktop host networking.** Docker Desktop on a
Mac runs containers in a Linux VM. `network_mode: host` (below, required so
the bridge shares the replay container's DDS traffic) then binds inside that
VM by default, not to the real Mac — so `ws://localhost:PORT` refuses the
connection from Foxglove Studio even with the bridge log saying it's
listening. Docker Desktop → Settings → Resources → Network → **Enable host
networking** closes that gap. VERIFIED end-to-end on a Mac: Foxglove
connected and streamed replayed topics live.

Default port is **8768**, not the foxglove_bridge default of 8765 — 8765 is
already the default port for this repo's own remote *policy* server
(`camelo/policy/server.py`, `scripts/serve_policy.py --port 8765`), and on at
least one Mac it was also squatted by an unrelated local tool (Cursor) bound
to `127.0.0.1:8765`. Because host networking binds straight to the machine,
either one silently steals the connection instead of erroring — Foxglove
reports nothing wrong, it's just talking to the wrong server. Check first
with `lsof -iTCP:8768 -sTCP:LISTEN -n -P` (swap the port if you pick a
different one) and expect only `com.docker...` in the output.

Third terminal, alongside the `play_rig_bag.sh` one — a bridge container on
the same DDS graph (`network_mode: host`, same as the player, so no `-p` is
needed or even effective: compose's own message is "Published ports are
discarded when using host network mode" — host mode already binds the port
straight to the machine):
```bash
./scripts/ros_compose.sh run --rm camelo-ros bash -lc \
  'apt-get update -qq && apt-get install -y -qq ros-jazzy-foxglove-bridge && \
   source /opt/ros/jazzy/setup.bash && \
   ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8768'
```
Wait for `Server listening on port 8768`, then in Foxglove Studio: **Open
connection → Foxglove WebSocket → `ws://localhost:8768`**. (Or from a
terminal: `open "foxglove://open?ds=foxglove-websocket&ds.url=ws%3A%2F%2Flocalhost%3A8768"`.)
The apt install isn't baked into the image (~10 s each time); worth adding to
`docker/Dockerfile.ros` if this becomes routine.

## Remote policy

**T — tunnel (plain shell, leave it open).** Port shown is ACT (8767, the
default below); swap both `8767`s for `8766` to drive VLA-JEPA instead — see
"Servers" further down for which is which.
```bash
ssh -N -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -L 8767:127.0.0.1:8767 -i ~/pci-sim.pem ubuntu@54.224.145.114
```
Prints nothing while healthy; if it ever exits, restart it before the next rollout.

**P — dummy client, immediately before EVERY rollout.**
```bash
python -u scripts/run_dummy_client.py --server ws://127.0.0.1:8767 --rate 2 --seconds 20 2>&1 | tee outputs/rig/t6_dummy_$(date +%H%M%S).log
```
Every request answered with `chunk=(21, 15)` and no reconnects, else the tunnel or the server is down.

**P — T6-6 (async, observation-time base, offset splice — current default, R-83), the rollout (video on, E-stop in hand, right arm only, 30 s).**
```bash
TS=$(date +%H%M%S); python -u scripts/run_policy.py --world real --backend remote --server ws://127.0.0.1:8767 --action-layout s27a15 --state-layout s27a15 --task "Pick up the thermal pad and place it on the target RAM board" --arms right --start-pose file:outputs/rig/t5/start_pose_s27a15_ep163.json --start-pose-tol 0.10 --activate-arms --wait-for-activation --keepalive-hz 10 --arm-command-frame robot --rate 20 --async-inference --chunk-time-base observation --chunk-splice offset --splice-ramp-ticks 20 --replan-steps 8 --max-delta 0.04 --max-image-age-s 0.5 --seconds 30 --chunk-dump outputs/rig/t6_chunks_$TS.npz --joint-csv outputs/rig/t6_act_$TS.csv > outputs/rig/t6_act_$TS.log 2>&1; echo "exit=$?"
```
Port **8767 is ACT** — the port depends on which server is up (see "Servers" below; VLA-JEPA is 8766); match `-L` on the tunnel to whichever you're driving. The offset splice bleeds the hand-over discontinuity off as an additive correction over `--splice-ramp-ticks` command ticks instead of an index jump, and the observation-time base stops re-committing to rows the arm has already passed. R-83 (14:51–14:58) reached L2 0.08 rad to the demo's grasp pose on a 120 s repeat (`--seconds 120`, `t6a_a02_145428`) — the best executor line measured on this rig on either checkpoint; U-36 is answered (R-82: an async loop approaches the pad, the synchronous line below does not — the sync stop-and-go itself is off-distribution for this checkpoint, not the wrist-exposure fix). Confirm the start pose first (`--start-pose-tol 0.10` will refuse to run off-pose by more than that — R-83's `t6a_a01_145207` hit exactly this refusal after a prior rollout left the arm un-rehomed; re-home rather than widening the tolerance). **Use `--seconds 120`, not 30, for any run meant as a grasp attempt** — R-85 (a03) only reached the grasp pose and closed at 25.8–48 s, well past a 30 s window.

Optional: append `--gripper-latch 0.5:30:0.9 --gripper-latch-near slot:xNNN:0.25` (`xNNN` = this rollout's slot) to hold the close once the policy commands it and gate the engage on proximity to that slot's demo grasp pose, so the pre-shape dip does not trigger it early — R-86 (a04/a05) showed a threshold-only latch either fires early or never fires; the proximity gate is untested on the rig (a06 planned).

**P — T6-6 (synchronous plan-then-execute, superseded 2026-09-03 — do NOT use for ACT: R-80).**
```bash
TS=$(date +%H%M%S); python -u scripts/run_policy.py --world real --backend remote --server ws://127.0.0.1:8767 --action-layout s27a15 --state-layout s27a15 --task "Pick up the thermal pad and place it on the target RAM board" --arms right --start-pose file:outputs/rig/t5/start_pose_s27a15_ep163.json --start-pose-tol 0.10 --activate-arms --wait-for-activation --keepalive-hz 10 --arm-command-frame robot --rate 20 --no-async-inference --chunk-time-base arrival --chunk-splice index --replan-steps 10 --max-delta 0.05 --max-image-age-s 0.5 --seconds 30 --chunk-dump outputs/rig/t6_chunks_$TS.npz --joint-csv outputs/rig/t6_act_$TS.csv > outputs/rig/t6_act_$TS.log 2>&1; echo "exit=$?"
```
Synchronous observe→infer→execute-10-rows-from-row-0 loop (commit `23ee7e9`): R-78 (12:15:58) produced one full task sequence — approach, close, transport — but R-80's three repeats of this exact line then wiggled and never descended, and R-82 confirmed the mechanism is the sync stop-and-go itself, not a confound (U-36 ANSWERED). Kept here only as a historical alternative, not for scoring ACT.

`--allow-camera-shape-mismatch` is dropped now that the wrists are 640×480 (U-21 CLOSED, R-73); add it back only if the T6-4 `check_obs` step still exits 3. If either command aborts with `STALE JOINT STATE` while the four numbers still read `4 0 2 1`, rerun with `--max-state-age-s 0.5` added.

Both commands above get the U-32 leash for free: `--leash-rad` defaults to 0.15 on `--world real`; pass `--leash-rad 0` to opt out. High `leash_active_pct` on the offset-splice line is expected (the command leads the arm by up to 0.15 rad continuously — the arm tracking with lag, U-26) and is not itself a fault.

## Servers (EC2, two policies up in parallel as of 2026-09-03)

Both served from **our** clone `~/workspace/camelo-ebim-jo` (commit `0f32375`),
with `--seed 0` (pins random/numpy/torch at start and on every reset, protocol
15 §2.4), checkpoints bind-mounted **read-only** from the colleague's
checkout — do not write into that mount.

**VLA-JEPA @30000, port 8766** (R-81; transport fix in flight, U-40):
```bash
docker run --rm --gpus all -p 8766:8766 -v ~/workspace/camelo-ebim-jo:/workspace -v <colleague-checkout>:/ckpt:ro camelo-ebim/policy:latest python scripts/serve_policy.py --adapter vla_jepa --checkpoint /ckpt/vla_jepa/checkpoint-30000 --action-layout s27a15 --state-layout s27a15 --port 8766 --seed 0
```

**ACT @100000, port 8767** (default line above; R-83):
```bash
docker run --rm --gpus all -p 8767:8767 -v ~/workspace/camelo-ebim-jo:/workspace -v <colleague-checkout>:/ckpt:ro camelo-ebim/policy:latest python scripts/serve_policy.py --adapter lerobot --checkpoint /ckpt/act/checkpoint-100000 --action-layout s27a15 --state-layout s27a15 --port 8767 --seed 0
```

The tunnel (T6-2 below) and the dummy client (T6-3) must forward/dial
whichever port matches the server you intend to drive this session — `-L
8766:127.0.0.1:8766` for VLA-JEPA, `-L 8767:127.0.0.1:8767` for ACT — `ss -ltn`
only tells you a local listener exists, not which server is behind it (U-30).

**P — ALTERNATIVE to the policy: open-loop replay of episode 9 (validated twice on 2026-09-02, R-55/R-61; same runner, no server; E-stop in hand).**
First home both arms to episode 9's frame 0 on **A** (`python3 ~/home_arms.py --file ~/t5_ep009_home_pose.yaml`) and place the scene as episode 9 (red pad in the 4th slot; overlay with `tools/rig_probes/overlay_latest.sh outputs/rig/t5/ep009_frame0_head.png`).
```bash
TS=$(date +%H%M%S); python -u scripts/run_policy.py --world real --backend local --adapter replay --checkpoint outputs/rig/t5/ep009_actions.npy --arms right --start-pose file:outputs/rig/t5/start_pose_s27a15_ep009.json --start-pose-tol 0.10 --activate-arms --wait-for-activation --keepalive-hz 10 --arm-command-frame robot --rate 20 --max-delta 0.05 --replan-steps 8 --cameras '' --seconds 22 --joint-csv outputs/rig/t5_replay_ep9_$TS.csv > outputs/rig/t5_replay_ep9_$TS.log 2>&1; echo "exit=$?"
```
Replays the recorded right-arm actions of episode 9 (22 s); `dt=0.050` on the first chunk line, `clamped_pct` ≈ 6–10 %, and the gripper grasps the red pad if the scene matches. For episode 163 instead: swap `ep009` → `ep163`, `t5_ep009_home_pose.yaml` → `t5_ep163_home_pose.yaml`, `--seconds 26` (never run yet).

**C — after every rollout.**
```bash
ps -eo pid,etime,args | awk '/ros2_control_n[o]de/' | wc -l; grep -c -iE 'FATAL|reflex|communication_constraints|died' ~/start_upper.log; grep -c 'Rejecting GELLO' ~/start_upper.log; ps -eo pid,etime,args | awk '/robotiq_gripper_cli[e]nt/' | wc -l    # want 4, 0, 0, 2
```
Then paste the tail of the rollout log to Claude for scoring.

**P — pass table for the next rollout (first with U-32; grep the log rather than eyeballing it).**
```bash
grep 'policy chunk' outputs/rig/t6_act_$TS.log | tail -20
```
```bash
grep -E 'rollout stats:|leash telemetry:' outputs/rig/t6_act_$TS.log
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

## Scored block (protocol 15 §2), as run 2026-09-03 for block B2

Everything above (phase 0, watcher, deploy, pixi, cameras) is done once per
session, not per rollout. The slot sequence is frozen at T+0
(`outputs/rig/munich_2026-09-01/t6_slot_sequence.csv`, `rollout_idx` → slot)
and never re-randomized mid-block.

**Once, at the start of the block:**

- **A — home** both arms to episode 163 frame 0 (`python3 ~/home_arms.py
  --file ~/t5_ep163_home_pose.yaml`), as in "Arms and scene" above.
- **A — re-open the gripper** (ebimHP-local one-liner, not checked into this
  repo, same class of site-only script as `home_arms.py`) and confirm the
  measured open fraction reads ≥ 0.99 before the first rollout — R-79 caught
  a residual 0.856 after a prior grasp attempt that silently biased that
  session's wrench/gripper state (U-39); R-80's stack restart + re-open got
  a clean 1.00 at r00.

**Per rollout (repeat for every `seq_index` in the frozen sequence):**

1. **A — place the pad** at this rollout's slot (`t6_slot_sequence.csv`,
   `rollout_idx` → `slot`/`x_norm`: x048 ≈ 0.48, x051 ≈ 0.51, x061 ≈ 0.61).
2. **P — capture + check the placement**, per-slot centroid, not just the
   mean-`|diff|` overlay (R-77 showed the overlay alone misses a
   shifted/reordered row):
   ```bash
   python -u scripts/capture_head.py > outputs/rig/live_head_$(date +%H%M%S).log 2>&1; echo "exit=$?"
   ```
   ```bash
   tools/rig_probes/pad_centroid.sh --slot xNNN
   ```
   `xNNN` = this rollout's slot; expect the red-pad centroid within ±12 px of
   that slot's pixel value (x048/x051/x061 = 614/653/781 px, R-77/pad_centroid.py).
3. **P — the rollout**, same line as T6-6's default above (async,
   observation-time base, offset splice, R-83; U-37's fixed wrist exposure,
   `--leash-rad` default 0.15), output files tagged `t6s_r<NN>` so the run
   directory carries the rollout number:
   ```bash
   R=r00; TS=$(date +%H%M%S); python -u scripts/run_policy.py --world real --backend remote --server ws://127.0.0.1:8767 --action-layout s27a15 --state-layout s27a15 --task "Pick up the thermal pad and place it on the target RAM board" --arms right --start-pose file:outputs/rig/t5/start_pose_s27a15_ep163.json --start-pose-tol 0.10 --activate-arms --wait-for-activation --keepalive-hz 10 --arm-command-frame robot --rate 20 --async-inference --chunk-time-base observation --chunk-splice offset --splice-ramp-ticks 20 --replan-steps 8 --max-delta 0.04 --max-image-age-s 0.5 --seconds 30 --joint-csv outputs/rig/munich_2026-09-01/t6s_${R}_${TS}/t6s_${R}_${TS}.csv > outputs/rig/munich_2026-09-01/t6s_${R}_${TS}/t6s_${R}_${TS}.log 2>&1; echo "exit=$?"
   ```
   Video on, E-stop in hand, throughout. (R-80's block B2 ran on the
   synchronous line, superseded 2026-09-03 — see R-82/R-83 and U-36; restart
   the block on this line before judging the checkpoint.)
4. **C — after-rollout check** (same 4-number gate as every other rollout,
   "after every rollout" above), then **report line**: the operator's plain
   verbal account of the rollout (e.g. R-80's "wiggles and does not even go
   down") — paste it to Claude alongside the log tail for scoring; do not
   editorialize it into a verdict yourself first.
5. **M/P — score and record the row**, `--dry-run` first, matching the
   `--seq-index` to `t6_slot_sequence.csv`'s `rollout_idx`:
   ```bash
   python3 tools/rig_probes/rollout_row.py --run-dir outputs/rig/munich_2026-09-01/t6s_${R}_${TS} --seq-index <N> --picked {0,1} --placed {0,1} --contact-frame {F|none} --video "phone (operator)" --notes "R=${R}: <operator report>" --operator julian --policy act --checkpoint-step 100000 --block B2 --dry-run
   ```
   Drop `--dry-run` once the printed row looks right, then:
   ```bash
   python3 tools/rig_probes/score_t5_trace.py outputs/rig/munich_2026-09-01/t6s_${R}_${TS}/t6s_${R}_${TS}.csv > outputs/rig/munich_2026-09-01/t6s_${R}_${TS}/score.txt
   ```

**Stop the block early** (as R-80 did after r02) if the mechanism is
visibly not going to close — three wiggling, non-descending rollouts is
enough to not burn the remaining twelve at ~2–3 min each; a `--picked 0
--placed 0` row is still a valid, recorded rollout, not a discarded one.

## Wind-down

Ctrl+C in **W**, **K**, **T**; then in **A**:
```bash
pkill -INT -f start_upper.bash; sleep 8; ps -eo pid,etime,args | awk '/robotiq_gripper_cli[e]nt|ros2_control_n[o]de|ros2 laun[c]h|start_upp[e]r/' | cut -c1-100    # want nothing; kill any leftover PID by hand, then: exit
```
Arms down last, after the cameras and the tunnel.
