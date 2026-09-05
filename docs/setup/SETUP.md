# Setup

## Hardware reality: one box at a time

We have EITHER the DGX (Spark/GB10) OR the H100 available — not both
simultaneously (an A10/A10G can also run the sim path; see §6):

| Available today | What you can do | What you cannot |
|---|---|---|
| **DGX Spark (GB10, aarch64)** | Everything sim-in-the-loop: sim + helper + eval stacks (`make sim-up helpers-up eval-stack-up`), recording, policy inference (`--backend local` or server on `ws://localhost:8765`). GR00T N1.7 officially runs here (~8–10 Hz PyTorch per NVIDIA's model card; TensorRT ~10 Hz); X-VLA (0.9B) and smolVLA comfortably. Small fine-tunes (ACT, X-VLA, LoRA) are feasible on the 128 GB unified memory, just slower than the H100. | Isaac Sim livestreaming (aarch64 — use the browser controller on :8090 via SSH tunnel between policy runs); big full-parameter fine-tunes are slow. |
| **H100 (x86)** | Training days: `lerobot-train` for all rungs (ACT → X-VLA → GR00T → pi0), offline dataset work, offline adapter tests, policy-server soak tests. | **The benchmark sim cannot run here** — Isaac Sim requires an RTX-class GPU with RT cores; H100 has none. No sim-in-the-loop eval, no recording. |
| **A10 / A10G (~24 GB)** | Sim + GR00T + optional WebRTC (`scripts/launch_*.sh a10`). | Fat fine-tunes; room+livestream+GR00T without watching VRAM. |

Practical rhythm: record + evaluate on DGX (or A10) days; merge datasets,
train, and prep checkpoints on H100 days; re-evaluate on the next sim day.
Same-box policy runs isolate the model on localhost (`launch_policy.sh`);
two machines (GPU + robot/sim, no local GPU) is §7.

GELLO recording additionally requires the teleop rig hardware (attached to
the DGX Spark); for pipeline testing, junk episodes are recorded with the
dummy policy as teacher (docs/DATA_COLLECTION.md).

## 1. Checkouts

```bash
# side by side; the code resolves the benchmark as a sibling by default
~/work/ebim-benchmark      # the benchmark (fork), assets downloaded per its README
~/work/camelo-ebim         # this repo
~/work/teleoperation       # EBiM-Benchmark/teleoperation (GELLO/pedal publishers, GB10 only)
```

Non-sibling layout: `export EBIM_BENCHMARK_ROOT=/path/to/ebim-benchmark`.

## 2. ROS side — container-first (the DGX has no host ROS)

The DGX Spark has **no ROS 2 underlay on the host** (the benchmark runs all
its ROS inside containers), so the `camelo-ros` container is the default
path (DGX_FINDINGS.md F-01):

```bash
cd camelo-ebim
./scripts/ros_compose.sh build camelo-ros
./scripts/ros_compose.sh run --rm camelo-ros make check-obs
# the compose service bind-mounts the live source — edits on the host apply
# without a rebuild
```

Native-venv alternative, only on a box that DOES have a ROS 2 Jazzy
underlay (rclpy cannot come from PyPI):

```bash
source /opt/ros/jazzy/setup.bash
uv venv --system-site-packages .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
# Since PR #38 `opencv-python-headless` and `pillow` are CORE dependencies
# (the perception approach). Re-run the line above in any venv or container
# created before 2026-09-05, and rebuild `camelo-ros` / `camelo-ros-humble`
# (both Dockerfiles now import cv2 + PIL at build time as the check). The
# ROS layer itself never imports them at module scope (tests/test_layering.py).
cp .env.example .env                  # then: set -a; source .env; set +a
```

DDS settings must match the benchmark stack on every process:
`ROS_DOMAIN_ID`, `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`,
`FASTDDS_BUILTIN_TRANSPORTS=UDPv4`. Containers need `network_mode: host`
**and** `ipc: host` (or keep the UDPv4 forcing) — with a private /dev/shm,
FastDDS discovery succeeds but zero samples arrive.

## 3. Policy/training environment (either box)

```bash
cd camelo-ebim
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[policy,train]"
# lerobot >= 0.6 covers GR00T N1.7 (policy type groot) and X-VLA natively.
# Only the optional gr00t-isaac adapter (TensorRT path) needs the
# Isaac-GR00T repo — docker/Dockerfile.policy bakes it in.
```

Or build the image: `docker compose -f docker/compose.yaml build camelo-policy`.

## 4. Launch rules for policy runs (the benchmark side)

Bring-up wrappers: `make sim-up` (Isaac Sim container, backgrounds) then
`make sim-scene` (**foreground — owns its terminal, run it in tmux**;
`SCENE=room` default, `SCENE=barebone` for the LFS-free scene), plus
`make helpers-up`, `make eval-stack-up`, `make stack-status`. Before the
first run: benchmark assets downloaded per its README **and `git lfs pull`
in the benchmark checkout** (the room scene's `robot_room.usd` is an LFS
object — a 133-byte pointer file crashes the scene load).

Camera-rate reality on the DGX Spark (measured, DGX_FINDINGS.md F-15):
the four RGB streams publish at ~1.6–2.3 Hz with the room scene under
`--record`, render-throughput-bound. Expect stale frames across control
ticks; levers are `--robot-camera-frame-skip`, recording fewer cameras,
and disabling the eval camera's semantic-segmentation graph between
scored runs.

These are the non-negotiables; violating them fails silently:

- Sim: `bash $EBIM_BENCHMARK_ROOT/task2_isaacsim/scripts/run_isaacsim_teleop.sh --scene room --no-browser -- --record`
  - **`--record` on**: publishes clock/cameras/recording topics and enables the scene-reset hook the episode runner needs.
  - **No keyboard-teleop flags** (`--arm-keyboard-teleop` etc.): with them, the bridge ignores all ROS arm/gripper commands.
  - **`--no-browser`**: the browser controller continuously streams its slider pose and fights the policy (`run_policy` warns if it detects one).
- Helper stack: default position-controller mode; do not pass `--with-*-teleop` adapter flags for policy runs.
- Eval stack (only for `eval_batch`): `make eval-stack-up` — it runs the benchmark's one-time `setup.sh` first (F-34: skipping it lets docker create the artifact dir as root and the container crash-loops on `PermissionError: /output/evaluate`).
- **One task's helper stack at a time** — task1/task2 stacks bind identical topics and both want port 8090.
- GB10 is headless (aarch64 cannot livestream Isaac Sim): use the browser controller via `ssh -L 8090:localhost:8090 gb10` for eyeballing, and turn it off again for policy runs.

## 5. Smoke test order (each step gates the next; single box unless noted)

```bash
make test                        # offline: contracts, executor, quantizer, wire
make sim-up                      # Isaac Sim container (backgrounds)
make sim-scene                   # task2 scene — foreground, keep its tmux pane
make helpers-up stack-status     # helper containers alive?
make check-obs                   # M1.1: all streams alive, finite 37-dim state
                                 #   (cameras ~2 Hz on the Spark is expected)
make sine                        # M1.2: arms wiggle, grippers cycle
make run-policy                  # M1.6: dummy adapter holds pose
# junk episode via dummy teacher + replay (auto polarity check) -> DATA_COLLECTION.md
make serve ADAPTER=gr00t &       # M1.7 (same box, process isolation)
make run-policy BACKEND=remote SERVER=ws://localhost:8765
make eval-stack-up               # includes the benchmark's one-time setup.sh
make eval ADAPTER=dummy N=3      # M1.8: loop works end-to-end, scored CSV
                                 #   + eval frames in outputs/eval/<run>/episode_<n>/
make eval ADAPTER=gr00t N=5      # zero-shot baseline (then xvla, pi0)
```

## 6. Two-script launch (DGX default / `a10`)

```bash
# terminal 1 (tmux) — sim + helpers
./scripts/launch_sim.sh            # DGX
./scripts/launch_sim.sh a10        # A10 headless (room; SCENE=barebone if no LFS)
./scripts/launch_sim.sh a10 webrtc # A10 + WebRTC livestream (room)

# terminal 2 — GR00T server + driver (30 sim-s rollout by default)
./scripts/launch_policy.sh         # DGX (sim topics)
./scripts/launch_policy.sh a10     # A10
./scripts/launch_policy.sh --world real   # TMR station topics (record_bag.bash)

# scored eval (needs `make eval-stack-up` on terminal 1 first):
# one episode, default max rollout (120 sim-s), prints mean IoU + writes summary.json
./scripts/launch_policy.sh eval
./scripts/launch_policy.sh a10 eval
N=1 ./scripts/launch_policy.sh a10 eval   # quick smoke
# open-loop demo chunks (ep 89 is bundled; no HF_TOKEN). Other REPLAY_EP=N still need HF_TOKEN.
ADAPTER=replay ./scripts/launch_policy.sh a10
CORRECT_POSES=True N=20 ADAPTER=replay ./scripts/launch_policy.sh a10 eval
# local pi0.5 / ACT fine-tune (LeRobotAdapter). CKPT may be a run dir or the
# pretrained_model dir; a run dir resolves to the latest checkpoints/*/pretrained_model.
# Layout flags are required for a local dir (F-45/F-63) — derive with
# `python scripts/eval_recipe.py <ckpt>` if unsure.
ADAPTER=lerobot \
  CKPT=pi05_state_route_3way/pi05_ft_lora_v2_untrimmed_20260819_194751 \
  ACTION_LAYOUT=canonical STATE_LAYOUT=model16 \
  ./scripts/launch_policy.sh a10 eval
```

WebRTC (A10 only): SG inbound **TCP 49100** + **UDP 47998** from your IP,
then Isaac Sim WebRTC Streaming Client → plain IP from
`launch_sim.sh a10 webrtc` (not `IP:port`). Client docs:
https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/manual_livestream_clients.html

Gray viewport almost always means signaling worked (TCP) but media did not
(UDP). Confirm the SG rule is **Custom UDP** 47998 (not TCP). On the
instance, `ss -ulnp | grep 47998` should show a listener; if it does not,
Kit bound random UDP ports instead — open UDP `7000-7500` from your IP as a
temporary check (streamsdk 7.x on Isaac 5.1 has been seen to ignore the
pinned 47998 host port).

One-time on A10: accept https://huggingface.co/nvidia/Cosmos-Reason2-2B
(first GR00T infer), then
`docker compose -f docker/compose.yaml build camelo-ros camelo-policy`
(same as DGX/H100). Watch `nvidia-smi` — room+livestream+GR00T is tight on 24 GB.

## 7. Remote inference (GPU box vs robot / sim box)

Do not forward DDS. Same split as `launch_policy.sh a10`: GPU runs
`serve_policy.py` (`camelo-policy`); robot/sim box runs
`run_policy --backend remote` (`camelo-ros`). Sim publishes `/bridge/*`;
real (`--world real`) publishes the TMR station topics in `record_bag.bash`.
Images travel as JPEG on `ws://…:8765`.

**GPU** — checkpoint, `--action-layout`, `--state-layout`,
`--num-inference-steps` live here. Path is a Hub id or
`checkpoints/*/pretrained_model`. No TLS — tunnel or VPN, don't publish
`:8765`.

```bash
docker run --rm --network host --gpus all \
  -e HF_TOKEN -e HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}" \
  -v "$PWD:/workspace/camelo-ebim" \
  -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
  -w /workspace/camelo-ebim \
  camelo-ebim/policy:latest \
  python -u scripts/serve_policy.py --adapter lerobot \
    --checkpoint <pretrained_model> \
    --action-layout canonical --state-layout model16
```

**Robot / sim** — host-network `camelo-ros`. Sim needs an `ebim-benchmark`
checkout (`verify_topics()`). Real robot uses the TMR station topics in
`record_bag.bash`, not `/isaac/*` + `/bridge/*`. Images are still BEST_EFFORT
raw `Image`. Real `target_gripper_width_percent` is an OPENING (1.0 = open,
0..1 open fraction — the topic name is a misnomer, measured 2026-09-02 T1(c))
unless `CAMELO_GRIPPER_INVERT` overrides. Stop GELLO teleop first.

```bash
ssh -N -L 8765:127.0.0.1:8765 user@gpu-host   # or SERVER=ws://<gpu-vpn>:8765

# sim (default)
./scripts/ros_compose.sh run --rm camelo-ros make check-obs
./scripts/ros_compose.sh run --rm camelo-ros \
  python3 -u scripts/run_policy.py --backend remote \
    --server ws://127.0.0.1:8765 --seconds 180

# real robot
./scripts/launch_policy.sh --world real
# or client only:
./scripts/ros_compose.sh run --rm -e CAMELO_WORLD=real camelo-ros \
  python3 -u scripts/run_policy.py --backend remote --world real \
    --server ws://127.0.0.1:8765 --seconds 180
```

### 7.1 The real-robot arm choreography

`--world real` is not just a different topic map: the companion's
`joint_impedance_controller` refuses to activate without a live command
stream and **shuts the whole arm launch down** if the stream stops for 2.0 s
while it is active (the full contract is in
[docs/CONTRACTS.md](../CONTRACTS.md) "Real robot"). So a policy run has a
fixed order, and `run_policy.py --world real` performs all of it:

1. **verify** — the arms are compared against `--start-pose file:<json>`
   and **nothing is commanded**. `recenter_arms()` does not move a real
   robot; `benchmark` / `task2` are SIM poses and are refused **with or
   without** `--no-start-pose-check` (that flag waives the check, not the
   sim-pose semantics — omit `--start-pose` entirely to run unverified).
   Off-pose by more than `--start-pose-tol` (0.10 rad) aborts before
   anything publishes.
2. **hold** — the MEASURED pose is published at `--rate` (a zero-delta GELLO
   stand-in). This is what `on_activate` needs, and it makes `g0 == q0`.
3. **activate** — either the operator runs the station's `activate_arms.py`
   while the hold continues (default), or `--activate-arms` does the switch
   from the runner's own node. Either way the run waits until
   `list_controllers` says `joint_impedance_controller` is `active` on every
   `--arms` side, and logs the captured `q0` / `g0`. The controller's own
   `on_activate` can fire a whole poll (0.5 s) before that, so the capture
   also checks that the published stream and the arm both stood STILL across
   that window (spread and delta, tol 0.01 rad). Over tolerance it warns in
   `--arm-command-frame robot` and **refuses** in `gello`, where a wrong
   `(q0, g0)` reverses joints 1, 2 and 7 with a perfectly valid message.
4. **policy** — chunks flow; the keep-alive timer (`--keepalive-hz`, 10 Hz)
   repeats the last command with a fresh stamp whenever inference is slow,
   so a remote stall never reaches the 2.0 s liveness gap.
5. **deactivate** — on every exit path, including exceptions: deactivate
   while still publishing, confirm via `list_controllers` (10 s), and only
   then stop commanding. The run ends with a `publish telemetry:` line —
   `keepalive_republished` and `max_publish_gap_s`, the worst gap between
   two arm publishes in THIS rollout. That gap is the margin left on the
   controller's 2.0 s fault, and it is the number T6 asks for.

`scripts/eval_batch.py` refuses `--world real` outright, before any ROS
session is created: it has none of the choreography above, and its scene
reset and gripper-polarity gate both command sim-only poses.

The full invocation (GPU box already serving the `s27a15` checkpoint):

```bash
python3 -u scripts/run_policy.py \
  --world real --backend remote --server ws://127.0.0.1:8765 \
  --action-layout s27a15 --state-layout s27a15 \
  --start-pose file:outputs/rig/home_pose.json --start-pose-tol 0.10 \
  --arms left,right --wait-for-activation \
  --arm-command-frame robot --gello-joint-directions -1,-1,1,1,1,1,-1 \
  --keepalive-hz 10 --deactivate-on-exit \
  --rate 20 --replan-steps 8 --max-delta 0.05 --seconds 120
```

Add `--activate-arms` to switch the controllers from here instead of running
`activate_arms.py`; `--arms left` drives one arm and holds the other at its
measured pose for the whole run; `--no-wrench` DECLARES a rig without the
external-wrench topics (zeros plus `wrench_declared_absent`, never zeros by
omission — 12 of the policy's 27 state dims are wrench).

Before any of this, the arm-safe rung is
`python3 -u scripts/sine_probe_real.py --hold-s 10 --joints 6,7
--amplitude 0.03`: it publishes the start pose with **zero** offset for
`--hold-s` seconds (activate during that window), then wiggles only the
`--joints` selection. Add `--activate-arms --hold-s 12` to have the probe do
the switching itself, on its own node, through the same `ArmControllers` path
the runner uses (`--hold-s` has to be able to contain the whole activation —
`--activation-timeout-s` + 2 s — or the run is refused before anything is
published), and deactivate on every exit path, Ctrl+C included, before it
stops publishing. T1b — does the controller-side joint-direction
transform need undoing? — is `--joints 1 --amplitude 0.02`, because joint 1's
`dir` is −1 and a wrong frame moves the arm the other way within one cycle.

`camelo-ros` does not reserve a GPU. Dummy, heuristic, perception, and
`--backend remote` do not need one. `CAMELO_ROS_GPU=0` forces that even if
`nvidia-smi` is present but CDI is empty.

### 7.2 Humble image (rig) — `camelo-ebim/ros-humble`

**Which image.** Default is `camelo-ros` (Jazzy). Switch to
`camelo-ros-humble` on the Munich rig, where the Jazzy image's Fast DDS
2.14.6 / `rmw_fastrtps_cpp` 8.4.4 talks to a Humble companion and station
(Fast DDS 2.6.10): it *receives* topics fine, but its SERVICE calls to
`/{left,right}/controller_manager/{list_controllers,switch_controller}`
never get a reply — discovery works, request/reply does not (measured
2026-09-02). That breaks `camelo/ros/arm_activation.py`, so `--activate-arms`
and `--wait-for-activation` cannot work from the Jazzy image. The station's
own Humble teleop nodes have never been discovered from it either
(docs/realdata/16 R-24), and version skew is the leading hypothesis for the
station OOM (R-22/23). `ros-humble` puts both sides on the 2.6.x line.

| | `camelo-ros` (Jazzy) | `camelo-ros-humble` |
|---|---|---|
| sim (`/isaac/*`, `/bridge/*`), replay, `eval_batch` | yes | no |
| `--backend local`, `tools/rig_parity_check.py` | yes (needs torch) | **no** — image has no torch/lerobot |
| `--world real` + `--backend remote` | yes | yes |
| `--activate-arms` against the Humble companion | blocked (no service reply) | the reason this image exists |

Covered by the Humble image: `scripts/sine_probe_real.py` (incl.
`--activate-arms`), `scripts/check_obs.py --world real` (camera workers
decode `Image` with plain numpy — no cv_bridge), and
`scripts/run_policy.py --world real --backend remote`. Anything needing
torch stays on `camelo-ros`; the policy itself runs on the GPU box behind
`serve_policy.py`, exactly as in §7.

**Build + smoke test** (on the robot box):

```bash
./scripts/ros_compose.sh build camelo-ros-humble
# imports only, no DDS traffic — --network none proves it never touches the graph.
# Through the entrypoint (it sources /opt/ros/humble/setup.bash); NEVER
# --entrypoint python3, which skips the ROS underlay and fails on `import rclpy`.
docker run --rm --network none camelo-ebim/ros-humble:latest \
  python3 -c 'import rclpy, controller_manager_msgs.srv'
```

**Run** — same env sourcing as the Jazzy path, same flags:

```bash
set -a; source ~/teleoperation/station/configs/tmr_laptop_env.sh; set +a
./scripts/ros_compose.sh run --rm -e CAMELO_WORLD=real camelo-ros-humble \
  python3 -u scripts/check_obs.py --world real
./scripts/ros_compose.sh run --rm -e CAMELO_WORLD=real camelo-ros-humble \
  python3 -u scripts/run_policy.py --world real --backend remote \
    --server ws://127.0.0.1:8765 --seconds 120
```

The compose service mirrors `camelo-ros` (`network_mode: host`, `ipc: host`,
`/tmp:/tmp` + `FASTRTPS_DEFAULT_PROFILES_FILE` passthrough for the station's
whitelisted profile, repo bind mount) and adds `pid: host`
(docs/realdata/16b set-up A hardening). It reserves no GPU and is absent from
`compose.gpu.yaml`, so `ros_compose.sh` cannot attach a device to it.

### 7.3 Socket buffers — camelo's own Fast DDS profile (rig)

**The measurement first** (station, 2026-09-02 14:25, read-only):

| Reading | Value | What it rules in / out |
|---|---|---|
| `nstat -az` `UdpRcvbufErrors` | 46,761,273 | drops at the **socket receive queue** |
| `nstat -az` `UdpInErrors` | 46,761,273 — identical | every UDP error IS a receive-buffer overrun |
| `nstat -az` `IpReasmFails` | 0 | not IP fragmentation/reassembly |
| `sysctl net.core.rmem_default` | 212992 (208 KiB) | the queue every participant gets |
| `sysctl net.core.rmem_max` | 2147483647 (2 GiB) | the kernel ceiling is **not** the limit |
| `/tmp/tmr_fastdds_laptop_1000.xml` | no `receiveBufferSize`/`sendBufferSize` | nothing raises it off the default |

(Counters are cumulative since the 2026-08-29 boot. `nstat -a` prints
absolute values without rewriting nstat's history file, so reading them is
safe mid-session.)

An 848×480×3 `Image` sample is ~1.2 MB, delivered as ~20 datagrams of
`maxMessageSize` (65500 B) in one burst. A 208 KiB queue holds ~3 of them, so
the burst overflows and the **whole sample** is lost — and camelo's image
readers are BEST_EFFORT (AGENTS.md hard rule 5), so unlike a RELIABLE pair
there is no retransmission to recover it. That is exactly what
`camelo/ros/camera_workers.py` reports: `recv` at 10–15 msgs/s on 30 fps
wrist topics whose minimum stamp gap is 1/30 s. The samples are lost **below**
our callback; no Python-side change can help.

The site's profile is rendered per session by
`~/teleoperation/station/configs/tmr_laptop_env.sh` and is not ours to edit,
so camelo derives its **own** copy: the same `interfaceWhiteList`, the same
`maxMessageSize`, the same participant profile, plus two socket-buffer lines
on the UDPv4 transport. `make dds-profile` prints the diff so you can check
that nothing else moved.

**Two lines, native (pixi shell on the station):**

```bash
make dds-profile        # reads $DDS_SRC (default /tmp/tmr_fastdds_laptop_$(id -u).xml)
export FASTRTPS_DEFAULT_PROFILES_FILE=$PWD/outputs/rig/fastdds_camelo.xml
```

Order matters: `set -a; source ~/teleoperation/station/configs/tmr_laptop_env.sh;
set +a` renders the source file **and** points
`FASTRTPS_DEFAULT_PROFILES_FILE` at it, so source the station env first and
export ours second, or the station's buffer-less profile wins.

**Container** — `docker/compose.yaml` already passes
`FASTRTPS_DEFAULT_PROFILES_FILE` through and bind-mounts the repo at
`/workspace/camelo-ebim`, so the derived file is visible inside at the repo
path (no `/tmp` copy needed):

```bash
make dds-profile
./scripts/ros_compose.sh run --rm -e CAMELO_WORLD=real \
  -e FASTRTPS_DEFAULT_PROFILES_FILE=/workspace/camelo-ebim/outputs/rig/fastdds_camelo.xml \
  camelo-ros-humble python3 -u scripts/check_obs.py --world real
```

Overrides: `make dds-profile DDS_SRC=… DDS_DST=… DDS_RX_MB=16 DDS_TX_MB=4`.
Nothing else changes behaviour — `check-obs`, `sine-probe-real` and
`run_policy` read the env var, never the file, so a rendered profile that is
not exported is inert.

**Verification — reproduce the failing measurement, not a proxy** (AGENTS.md):

```bash
nstat -az UdpRcvbufErrors            # before
… run check_obs --world real …
nstat -az UdpRcvbufErrors            # after
```

Two exit criteria, both required:

1. the delta across the run is **0** (it grew by millions before — 46.7 M
   since boot), and
2. the workers' `recv` reaches the wire rate (~30/s on the 30 fps wrist
   topics), not 10–15/s.

A zero delta with `recv` still at 10–15/s means the drop moved somewhere else
(a starved callback, a topic that never matched) — do not call it fixed.
`recv` at the wire rate with a non-zero delta means the buffer is still too
small for the burst; raise `DDS_RX_MB`.

**No `sysctl` is needed here**: `net.core.rmem_max` is already 2 GiB, well
above the 16 MiB we request, and `rmem_default` is irrelevant because Fast DDS
sets `SO_RCVBUF` explicitly. On a box where `rmem_max` is **smaller** than the
requested `receiveBufferSize`, the kernel silently clamps the socket to
`rmem_max` and the drops continue against a profile that reads correctly —
check `sysctl net.core.rmem_max` before blaming the profile. Raising it
(`sudo sysctl -w net.core.rmem_max=16777216`, persisted in
`/etc/sysctl.d/`) is a host-wide change and therefore the **operator's**
call: no camelo target, script or container does it.
