# Daily drivers. ROS targets need a sourced ROS 2 Jazzy environment and the
# task2 sim + helper stack up (docs/setup/SETUP.md; `make sim-up helpers-up`).
# Variables:
#   ADAPTER=pi0 BACKEND=remote SERVER=ws://h100:8765 CKPT=... N=5 SECONDS=120
#   WORLD=real  — TMR station topics (record_bag.bash); default sim
PY ?= python3
EBIM_BENCHMARK_ROOT ?= ../ebim-benchmark
ADAPTER ?= dummy
BACKEND ?= local
SERVER ?= ws://localhost:8765
TASK ?=
N ?= 5
SECONDS ?= 120
CKPT ?=
DATA ?=
CONFIG ?= configs/act.yaml
WORLD ?= sim

CKPT_ARG = $(if $(CKPT),--checkpoint $(CKPT),)
TASK_ARG = $(if $(TASK),--task "$(TASK)",)
WORLD_ARG = $(if $(filter real,$(WORLD)),--world real,)

.PHONY: test lint sim-up sim-scene sim-scene-table helpers-up helpers-down \
	eval-stack-setup eval-stack-up \
	preflight preflight-data \
	stack-status check-obs sine sine-probe-real replay replay-reference run-policy serve \
	dummy-client eval \
	dds-profile \
	eval-sweep record train merge prepare-replay prepare-heuristic \
	augment collect-augmented \
	stage train-smoke train-slurm train-slurm-2gpu train-slurm-4gpu train-smoke-ddp eval-recipe

test:
	$(PY) -m pytest tests/ -q

lint:
	ruff check camelo scripts tests

# --- benchmark stack bring-up (single box with docker, e.g. the DGX) --------
# SCENE=room needs `git lfs pull` in the benchmark checkout (robot_room.usd
# is an LFS object); SCENE=barebone has plain assets.
# On an x86 RTX box (A10/A10G EC2) set LIVESTREAM=1 and PUBLIC_IP=<eip>
# for the WebRTC viewer; leave both unset on the DGX (aarch64 cannot livestream).
# Headless EC2 without a viewer: HEADLESS=1 (LIVESTREAM already implies headless).
SCENE ?= room
# Frozen spine SOP, COMMANDED value (CONTRACTS.md, F-50/F-57/F-58). Equal
# min/max pins the spine and re-establishes it after every scene reset;
# an unpinned scene starts at 0.0 — geometry no dataset contains.
SPINE ?= 0.50
# Robot spawn override. Empty = the scene's own task preset (task2 spawns at
# 4.4, 2.6, yaw -90). Because the spawn transform IS the reset pose, an
# override here survives every episode reset — driving there manually would
# not. TABLE_* is the stage-2 pose used by sim-scene-table (below).
ROBOT_X ?=
ROBOT_Y ?=
ROBOT_YAW ?=
# The stage-2 base pose, taken from the reference demos rather than derived:
# every one of the 22 episodes in ext_hermanprawiro_task2_fixpos_v1 starts at
# base (2.100, 3.051, yaw -90 deg), bit-identical, 1.102 m from the table
# centre. An earlier geometry-derived guess (2.05, 2.6) stood 0.45 m too
# close, which forces the elbow to swing wide to reach the table and leaves
# the arm visibly sprawled. Measured from state[31:34] at frame 0.
TABLE_ROBOT_X ?= 2.10
TABLE_ROBOT_Y ?= 3.05
TABLE_ROBOT_YAW ?= -90
# Passthrough hooks so both scene targets stay usable for teleop/recording:
#   SIM_FLAGS   -> run_isaacsim_teleop.sh flags (e.g. --with-gello-teleop)
#   SCENE_ARGS  -> scene_room.py flags after the -- (e.g. --render-hz 30)
SIM_FLAGS ?=
SCENE_ARGS ?=

# A10/x86 livestream knobs (F-60): LIVESTREAM=1 + PUBLIC_IP=<eip> for the
# WebRTC viewer, HEADLESS=1 for headless EC2. Unset on the DGX (aarch64
# cannot livestream). LIVESTREAM already implies headless.
LIVESTREAM ?= 0
HEADLESS ?= 0
PUBLIC_IP ?=

ROBOT_ARGS = $(if $(ROBOT_X),--robot-x $(ROBOT_X),) \
	$(if $(ROBOT_Y),--robot-y $(ROBOT_Y),) \
	$(if $(ROBOT_YAW),--robot-yaw $(ROBOT_YAW),)

# One recipe, two entry points — sim-scene keeps the scene preset,
# sim-scene-table overrides ROBOT_* to put the robot at the table.
LS_ARGS = $(if $(filter 1,$(LIVESTREAM)),--livestream,$(if $(filter 1,$(HEADLESS)),--headless,))

define launch_scene
@if [ "$(LIVESTREAM)" = "1" ] && [ -z "$(PUBLIC_IP)" ]; then \
	echo "sim-scene: LIVESTREAM=1 requires PUBLIC_IP=<this host's public IP>" >&2; \
	exit 2; fi
@echo "launch: spine=$(SPINE) robot=[$(strip $(ROBOT_ARGS))] scene_args=[$(SCENE_ARGS)]"
@test "$(SPINE)" = "0.50" || echo "  WARNING: spine $(SPINE) is off the 0.50 SOP \
(CONTRACTS.md, F-57/F-58). A stray SPINE in the environment overrides the \
Makefile default — 'unset SPINE', or pass SPINE=0.50 on the command line."
cd $(EBIM_BENCHMARK_ROOT) && PUBLIC_IP="$(PUBLIC_IP)" \
	bash task2_isaacsim/scripts/run_isaacsim_teleop.sh \
	--scene $(SCENE) --no-browser $(LS_ARGS) $(SIM_FLAGS) -- --record \
	--spine-keyboard-min $(SPINE) --spine-keyboard-max $(SPINE) \
	$(ROBOT_ARGS) $(SCENE_ARGS)
endef

sim-up:               ## start the Isaac Sim container (backgrounds, returns)
	@# F-49: a container created before the current boot has a stale
	@# $$XAUTHORITY mountpoint baked into its rootfs and fails to start
	@# after a reboot with a confusing "not a directory" error — detect
	@# the exited pre-boot container and recreate it.
	cd $(EBIM_BENCHMARK_ROOT) && FLAG= && \
	CID=$$(docker ps -aq --filter name=isaac-sim --filter status=exited | head -1) && \
	if [ -n "$$CID" ]; then \
		BOOT=$$(date -d "$$(uptime -s)" +%s 2>/dev/null || echo 0); \
		CREATED=$$(date -d "$$(docker inspect -f '{{.Created}}' $$CID)" +%s 2>/dev/null || echo 0); \
		if [ "$$BOOT" -gt 0 ] && [ "$$CREATED" -gt 0 ] && [ "$$CREATED" -lt "$$BOOT" ]; then \
			echo "sim-up: isaac-sim container predates this boot — recreating (F-49)"; \
			FLAG=--force-recreate; \
		fi; \
	fi && \
	docker compose --env-file docker/.env.base \
		-f docker/docker-compose.yaml --profile isaac-sim-5.1.0 up -d $$FLAG

sim-scene: sim-up     ## launch the task2 scene — FOREGROUND, owns this terminal (tmux it)
	$(launch_scene)

# Stage 2 only (manipulation). The task2 spawn preset stands ~2.4 m from the
# table, far outside reach, so a stage-2 policy or demo can never touch the
# pad from it. This target overrides the spawn to the reference demos' own
# base pose (see TABLE_ROBOT_* above) — not a derived guess. Because the
# spawn transform IS the reset pose, it survives every episode reset.
sim-scene-table: ROBOT_X := $(TABLE_ROBOT_X)
sim-scene-table: ROBOT_Y := $(TABLE_ROBOT_Y)
sim-scene-table: ROBOT_YAW := $(TABLE_ROBOT_YAW)
sim-scene-table: sim-up  ## same scene with the robot AT the table (stage 2) — FOREGROUND
	$(launch_scene)

# NOTE: sim-scene/sim-scene-table already bring the helpers up themselves and
# forward --no-browser. Running helpers-up AFTER them re-ups with no arguments,
# and browser_controller is opt-OUT — so it restarts the browser controller,
# whose slider pose fights the policy (SETUP.md §4). For any policy or eval
# run keep the default; teleop flows that want the browser pass HELPER_FLAGS=.
HELPER_FLAGS ?= --no-browser

# Without task2_isaacsim/.env, compose defaults REPUBLISHER_GRIPPER_INVERT to
# false while camelo reads .env.example (true) — polarity check fails with a
# full inverted swing. Seed .env from the example once if missing.
helpers-up:           ## republisher + position controller (no teleop; browser off by default)
	@test -f $(EBIM_BENCHMARK_ROOT)/task2_isaacsim/.env || \
		cp $(EBIM_BENCHMARK_ROOT)/task2_isaacsim/.env.example \
		   $(EBIM_BENCHMARK_ROOT)/task2_isaacsim/.env
	cd $(EBIM_BENCHMARK_ROOT) && bash task2_isaacsim/scripts/run_helper_containers.sh up $(HELPER_FLAGS)

helpers-down:
	cd $(EBIM_BENCHMARK_ROOT) && bash task2_isaacsim/scripts/run_helper_containers.sh down

# setup.sh is the benchmark's own one-time step (idempotent): it creates the
# artifact dir owned by YOUR uid before docker can auto-create it as root,
# which crash-loops the container with PermissionError (F-34).
eval-stack-setup:
	cd $(EBIM_BENCHMARK_ROOT) && bash scripts/evaluation/task2/setup.sh

eval-stack-up: eval-stack-setup  ## task2 eval service (Trigger -> IoU JSON)
	cd $(EBIM_BENCHMARK_ROOT) && bash scripts/evaluation/task2/run.sh up

stack-status:
	@docker ps --format '{{.Names}}\t{{.Status}}' | grep -E 'isaac|task2|eval' || \
		echo "no benchmark containers running"
	@# F-89: a long-lived isaac-sim container can lose its GPU while the HOST
	@# GPU stays perfectly healthy — cameras go to 0 Hz, the sim logs Vulkan
	@# ERROR_INITIALIZATION_FAILED or "CUDA error 100: cudaErrorNoDevice", and
	@# host nvidia-smi looks fine, which sends you chasing the wrong thing.
	@# Relaunching the scene CANNOT fix it: run_isaacsim_teleop.sh docker-execs
	@# into the same broken container. Only a container restart clears it.
	@CID=$$(docker ps -q --filter name=isaac-sim | head -1); \
	if [ -n "$$CID" ]; then \
		if docker exec $$CID nvidia-smi -L >/dev/null 2>&1; then \
			echo "isaac-sim GPU:\tok (NVML responds inside the container)"; \
		else \
			echo "isaac-sim GPU:\tDEGRADED — NVML dead inside the container (F-89)."; \
			echo "\t\tThe host GPU is fine and the RUNNING sim keeps rendering: it"; \
			echo "\t\tholds a GPU context opened before this broke. Only NEW processes"; \
			echo "\t\tin the container fail. Cameras at 0 Hz means it has progressed"; \
			echo "\t\tto collapse — check 'make preflight'."; \
			echo "\t\tA long run started now risks collapsing mid-batch, which scores"; \
			echo "\t\tepisodes against a FROZEN camera frame rather than failing."; \
			echo "\t\tfix: docker restart $$(docker ps --filter name=isaac-sim --format '{{.Names}}' | head -1)"; \
			echo "\t\tthen relaunch the scene. A scene relaunch ALONE will not work."; \
		fi; \
	fi

# Rig only, and OPT-IN: this writes a file and changes nothing by itself.
# Neither check-obs nor sine-probe-real below reads DDS_DST — the profile
# takes effect only once the operator exports FASTRTPS_DEFAULT_PROFILES_FILE
# (native) or passes -e (container), so the default behaviour of every target
# here is exactly what it was. Reason, usage and the nstat verification:
# docs/setup/SETUP.md §7.3.
DDS_SRC ?= /tmp/tmr_fastdds_laptop_$(shell id -u).xml
DDS_DST ?= outputs/rig/fastdds_camelo.xml
DDS_RX_MB ?= 16
DDS_TX_MB ?= 4
dds-profile:          ## rig — camelo's own Fast DDS profile (big socket buffers) from the station's
	$(PY) scripts/render_dds_profile.py --src $(DDS_SRC) --dst $(DDS_DST) \
		--rx-mb $(DDS_RX_MB) --tx-mb $(DDS_TX_MB)

check-obs:            ## M1.1 — verify observation reception
	$(PY) scripts/check_obs.py --seconds 10 $(WORLD_ARG)

preflight:            ## before every recording block — cameras + clock + spine SOP
	$(PY) scripts/preflight.py --seconds 10 $(WORLD_ARG)

preflight-data:       ## after recording — spine/NaN check on DATA=<dataset dir>
	$(PY) scripts/preflight.py --dataset $(DATA)

sine:                 ## M1.2 — verify the command path (arms wiggle in sim)
	$(PY) scripts/sine_actor.py --seconds 20 $(WORLD_ARG)

# Always --world real (it is the point), so no WORLD_ARG here. PROBE_ARGS
# carries the script's own guards (--arms/--amplitude/--no-gripper/--csv).
# LOG IT WITH `| tee -i`, never a bare `| tee` (MEASURED 2026-09-02, T1(a)):
# Ctrl+C hits the whole foreground group, plain tee dies first and the next
# print raises BrokenPipeError. `-i` makes tee ignore SIGINT; the alternative
# is `> outputs/rig/t1a.log 2>&1` plus `tail -f` in another pane.
PROBE_SECONDS ?= 120
PROBE_ARGS ?= --arms right
sine-probe-real:      ## M1.2 real robot: wiggle right arm j6/j7 ±0.03 rad, measured vs commanded (log with `| tee -i`, not `| tee`)
	$(PY) -u scripts/sine_probe_real.py --seconds $(PROBE_SECONDS) \
		--amplitude 0.03 $(PROBE_ARGS)

replay:               ## M1.5 — replay a recorded episode (DATA=<dataset dir> EP=0)
	$(PY) scripts/replay_episode.py --dataset $(DATA) --episode $(or $(EP),0) $(REPLAY_ARGS)

# The harness reference (F-82): replay the dataset's own actions from the
# frame whose recorded spine matches the live pinned one, starting from the
# recorded pose, and film it. Scored 0.586 IoU at 1.7% clamped ticks — the
# number a policy is compared against, not 1.0.
replay-reference:     ## M1.5 — reference replay w/ start pose + video (DATA=… EP=0)
	$(MAKE) replay DATA=$(DATA) EP=$(or $(EP),0) REPLAY_ARGS="--reset \
		--start-frame auto --goto-start \
		--video outputs/videos/replay_ep$(or $(EP),0)_eval_camera.mp4 $(REPLAY_ARGS)"

prepare-replay:       ## download task2_fixpos_200 actions for ADAPTER=replay (EP=89 bundled)
	$(PY) scripts/prepare_replay_actions.py --episode $(or $(EP),89)

prepare-heuristic:    ## single-episode download for ADAPTER=heuristic (EP=89 bundled)
	$(PY) scripts/prepare_heuristic_actions.py --episode $(or $(EP),89)

# RUN_ARGS passes extra flags through to run_policy.py — e.g.
# RUN_ARGS="--approach perception" for perception-based desk navigation
# (camelo/cli.py --approach), mirroring EVAL_ARGS below.
RUN_ARGS ?=

run-policy:           ## M1.6/7 — drive the sim with a policy, no scoring
	$(PY) scripts/run_policy.py --adapter $(ADAPTER) --backend $(BACKEND) \
		--server $(SERVER) $(CKPT_ARG) $(TASK_ARG) $(WORLD_ARG) --seconds $(SECONDS) $(RUN_ARGS)

serve:                ## M1.7 — policy server near the GPU (H100)
	$(PY) scripts/serve_policy.py --adapter $(ADAPTER) $(CKPT_ARG) $(TASK_ARG)

dummy-client:         ## intermediate step — ROS node, no sensors, fixed obs on a timer
	$(PY) scripts/run_dummy_client.py --server $(SERVER) $(TASK_ARG) \
		--rate $(or $(RATE),1) --seconds $(SECONDS)

# EVAL_ARGS passes extra flags through to eval_batch.py — a checkpoint
# trained here needs --state-layout/--action-layout restated (F-63/F-45)
# and make cannot forward unknown options by itself.
EVAL_ARGS ?=

eval:                 ## M1.8 — N scored episodes -> outputs/eval/<run>/results.csv
	$(PY) scripts/eval_batch.py --adapter $(ADAPTER) --backend $(BACKEND) \
		--server $(SERVER) $(CKPT_ARG) $(TASK_ARG) --episodes $(N) --rollout-s $(SECONDS) \
		$(WORLD_ARG) $(EVAL_ARGS)

# Executor settings to compare, one block of N episodes each. Same session,
# same server, same checkpoint — only the executor changes.
# UPWARDS by default (F-88): backend.infer() blocks the control loop, so
# replanning MORE often costs control ticks and raises clamping. Chunks are
# 50 steps, so 32 still leaves headroom before one goes stale.
REPLAN ?= 8,16,32

eval-sweep:           ## M1.8 — same checkpoint under REPLAN=8,4,2 -> outputs/eval/<run>/sweep.json
	$(PY) scripts/eval_sweep.py --adapter $(ADAPTER) --backend $(BACKEND) \
		--server $(SERVER) $(CKPT_ARG) $(TASK_ARG) --episodes $(N) --rollout-s $(SECONDS) \
		--replan-steps-sweep $(REPLAN) $(WORLD_ARG) $(EVAL_ARGS)

# AUGMENT_ARGS carries generate_augmented_trajectories.py's own flags
# (--source/--per-family/--families/--seed/--scale/--out/…) — no defaults
# magic here, unlike ADAPTER/N/SECONDS above, since a source episode has no
# sane repo-wide default.
augment:              ## data augmentation — validated heuristic variants -> manifest.json
	$(PY) scripts/generate_augmented_trajectories.py $(AUGMENT_ARGS)

MANIFEST ?= data/heuristic/task2_fixpos_200/aug_v1/manifest.json

collect-augmented:    ## play MANIFEST's augmented variants in sim, one per episode
	$(PY) scripts/collect_augmented.py --manifest $(MANIFEST) --rollout-s $(SECONDS) $(EVAL_ARGS)

record:               ## M2 — pointer to the benchmark recorder workflow
	@echo "Recording uses the benchmark's own recorder (docs/runbooks/DATA_COLLECTION.md):"
	@echo "  1. sim:      bash \$$EBIM_BENCHMARK_ROOT/task2_isaacsim/scripts/run_isaacsim_teleop.sh --scene room --no-browser -- --record"
	@echo "  2. helpers:  see docs/runbooks/DATA_COLLECTION.md (GELLO or policy teacher)"
	@echo "  3. recorder: bash \$$EBIM_BENCHMARK_ROOT/task2_isaacsim/scripts/run_recorder.sh"

merge:                ## M3.0 — merge dataset versions (SOURCES="a b" OUT=... [DROP="name=19"])
	$(PY) -m camelo.train.merge_datasets --sources $(SOURCES) --output $(OUT) $(if $(DROP),--drop $(DROP),)

train:                ## M3.1 — lerobot-train wrapper (CONFIG=... DATA=...)
	$(PY) -m camelo.train.train --config $(CONFIG) --data $(DATA)

# --- SLURM cluster (tiger3, x86 H100) -------------------------------------
# Additive: every target above is unchanged and still runs on the DGX.
# These three only make sense where sbatch exists — docs/setup/TIGER3_H100.md.
# GPUS/CPUS/TIME are sbatch-side because the QOS is derived from walltime:
# short jobs get scheduling priority, so keep TIME honest rather than
# padded.
# R-1 (docs/realdata/13_LAUNCH_LOG.md): the two 4xH100 long runs died of
# host OOM at 256G (128 standing dataloader workers + a checkpoint/eval
# collision). 512G measured ~5x headroom against the same shape and is the
# default the moment GPUS=4, but stays overridable (`make train-slurm
# GPUS=4 MEM=256G` still gets 256G) since `?=` only fires when MEM is unset.
GPUS ?= 1
CPUS ?= 16
MEM  ?= $(if $(filter 4,$(GPUS)),512G,128G)
TIME ?= 04:00:00
EXTRA ?=
LAUNCHER ?=
# The smoke length prescribed by DGX_CHECKLIST.md §4. ~48 s of the 10-min
# gpu-test window at the measured 6.3 steps/s, so there is no reason to
# shorten it.
SMOKE_STEPS ?= 300

eval-recipe:          ## print the exact `make eval` line for RUN=<run or ckpt dir>
	$(PY) scripts/eval_recipe.py $(RUN) $(if $(N),--episodes $(N),)

stage:                ## LOGIN NODE — cache datasets/checkpoints for offline jobs
	$(PY) scripts/stage_assets.py $(if $(DATASETS),--datasets $(DATASETS),) \
		$(if $(MODELS),--models $(MODELS),) $(if $(RUNG),--rung $(RUNG),)

# sbatch --export is itself a COMMA-separated list, so a value containing a
# comma (--dataset.episodes=[0,1,2]) is silently chopped into bogus extra
# variables. Export into sbatch's own environment and let --export=ALL carry
# it through instead.
train-smoke:          ## 10-min gpu-test confirmation run — do this before any real run
	@mkdir -p outputs/slurm
	CONFIG='$(CONFIG)' DATA='$(DATA)' \
	EXTRA='--steps=$(SMOKE_STEPS) --save_freq=$(SMOKE_STEPS) --log_freq=50 $(EXTRA)' \
	sbatch --export=ALL --qos=gpu-test --time=00:10:00 \
		--gres=gpu:1 --cpus-per-task=8 --mem=64G --job-name=camelo-smoke \
		slurm/train.slurm

train-slurm:          ## full training run (GPUS/CPUS/MEM/TIME/EXTRA/LAUNCHER)
	@mkdir -p outputs/slurm
	CONFIG='$(CONFIG)' DATA='$(DATA)' EXTRA='$(EXTRA)' LAUNCHER='$(LAUNCHER)' \
	sbatch --export=ALL --time=$(TIME) --gres=gpu:$(GPUS) \
		--cpus-per-task=$(CPUS) --mem=$(MEM) \
		slurm/train.slurm

# --- Single-node DDP launch templates (docs/realdata/08_MODEL_SELECTION_AND_RIG_SCHEDULE.md §4.1 A0/§6) ---
# `lerobot-train` is an accelerate program: multi-GPU is a launcher prefix,
# not a flag, per docs/setup/TIGER3_H100.md "Multi-GPU" section and
# `accelerate launch --help` (--num_processes). One SLURM task
# (--ntasks=1, the default above) runs `accelerate launch --num_processes=N`,
# which forks N data-parallel processes on this one node's N GPUs --
# `--gres=gpu:N` is the only per-GPU resource request needed, never
# `--ntasks=N`. CPUS below follows the node facts in 07 §1 (12 nodes x
# 4 H100 80GB, 112 cores, ~1 TiB RAM each): 16 cores per GPU (`GPUS=4
# CPUS=64` scales down exactly to `GPUS=2 CPUS=32`). MEM is 128G/GPU
# (08 §6's original recipe) EXCEPT at GPUS=4, where R-1 (13_LAUNCH_LOG.md)
# raised it to 512G after two 256G host-OOM kills — `train-slurm`'s own
# GPUS-conditional default carries that through here too. slurm/train.slurm
# resolves `--num_workers` itself (yaml/EXTRA win over a CPUS/NUM_PROCESSES
# fallback — see the comment at its top), so these targets do not need to
# compute worker counts.
# `--multi_gpu` is LOAD-BEARING, not decoration (added 2026-08-31 after the
# 2-GPU rehearsal, job 3983176, died in 11 s with
#   ImportError: DeepSpeed is not installed => run `pip3 install deepspeed`).
# `accelerate launch` reads $HF_HOME/accelerate/default_config.yaml when no
# distributed flag is given, and the one on this account is a leftover from an
# unrelated project: `distributed_type: DEEPSPEED`, pointing at a ZeRO-2 JSON
# in a different workspace. accelerate/commands/launch.py:1215-1225 only
# adopts that file's distributed_type when NONE of --multi_gpu/--cpu/--tpu/
# --use_deepspeed/--use_fsdp was passed; passing --multi_gpu skips the block
# entirely, leaving use_deepspeed False so :1397 dispatches to
# multi_gpu_launcher. Explicitly asking for DDP is also simply more honest
# than inheriting whatever a global config happens to say.
train-slurm-2gpu:     ## full 2-GPU DDP run, single node (CONFIG/DATA/TIME/EXTRA)
	$(MAKE) train-slurm GPUS=2 CPUS=32 MEM=128G \
		LAUNCHER="accelerate launch --multi_gpu --num_processes=2"

train-slurm-4gpu:     ## full 4-GPU DDP run, single node (CONFIG/DATA/TIME/EXTRA)
	$(MAKE) train-slurm GPUS=4 CPUS=64 MEM=512G \
		LAUNCHER="accelerate launch --multi_gpu --num_processes=4"

# gpu-test QOS is a 60-minute window, not the 10-minute figure the older
# docs assumed (08 §0, live sacctmgr/sbatch --test-only measurement,
# 2026-08-31) -- 00:55:00 leaves a safety margin under the QOS boundary. A
# 300-step smoke fits comfortably (measured single-GPU: ~48s of a 10-min
# window at 6.3 steps/s) and also rehearses the DDP launch path itself
# before any long run is queued on it (08 §4.1 item S3/S5).
train-smoke-ddp:      ## 55-min gpu-test DDP rehearsal — GPUS=2 or GPUS=4
	@case '$(GPUS)' in 2|4) ;; *) \
		echo "train-smoke-ddp needs GPUS=2 or GPUS=4 (got '$(GPUS)')" >&2; exit 1 ;; \
	esac
	@mkdir -p outputs/slurm
	CONFIG='$(CONFIG)' DATA='$(DATA)' \
	EXTRA='--steps=$(SMOKE_STEPS) --save_freq=$(SMOKE_STEPS) --log_freq=50 $(EXTRA)' \
	LAUNCHER='accelerate launch --multi_gpu --num_processes=$(GPUS)' \
	sbatch --export=ALL --qos=gpu-test --time=00:55:00 \
		--gres=gpu:$(GPUS) --cpus-per-task=$$(( 16 * $(GPUS) )) \
		--mem=$$(( 64 * $(GPUS) ))G --job-name=camelo-smoke-ddp$(GPUS) \
		slurm/train.slurm
