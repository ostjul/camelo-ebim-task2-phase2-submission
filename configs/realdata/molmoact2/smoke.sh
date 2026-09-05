#!/bin/bash
# 300-step, 4xH100 gpu-test smoke -- settles offline load with no DNS
# (catches the transitive MolmoAct2-DROID pull failing hard on a compute
# node), per-GPU VRAM (the 4-GPU VRAM estimate is an extrapolated fit,
# "do not book a long run on it" -- read the real peak here), and updt_s.
# Submit this EARLY: the 4-GPU queue wait is volatile (measured between
# ~0.9h and 7.2h, decaying 6x in 31 minutes during one review pass) --
# submitting first lets the wait overlap other Phase-A work.
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
# 2026-08-31, after the 2-GPU rehearsal (3983334): walltime raised 00:30:00
# -> 00:55:00 and log_freq 50 -> 25. The rehearsal did NOT log a single
# step line inside its 30-minute window -- 2xH100, global batch 32, first
# backward at +3m25s and no step:50 after 15 more minutes, with
# `nvidia-smi` sampled three times through `srun --overlap` showing
# **0 % GPU utilisation on both devices** while the two rank processes sat
# at ~420 % CPU each. This rung is CPU-preprocessing-bound, not
# GPU-bound, and its step rate is the single number the 4-GPU long run has
# to be sized against. 00:55:00 is the documented gpu-test budget (see the
# Makefile's train-smoke-ddp comment); log_freq=25 means a rate lands after
# 25 steps instead of 50, so even a smoke that hits the wall yields the
# measurement.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source configs/realdata/_lib.sh

EPISODES="$(train_episodes_flag)"
mkdir -p outputs/slurm

CONFIG='configs/realdata/molmoact2/train.yaml' DATA="$DATA" \
EXTRA="--steps=300 --save_freq=300 --log_freq=25 --eval_steps=0 $EPISODES" \
LAUNCHER='accelerate launch --multi_gpu --num_processes=4 --num_cpu_threads_per_process=4' \
sbatch --export=ALL --qos=gpu-test --time=00:55:00 \
    --gres=gpu:4 --cpus-per-task=64 --mem=256G --job-name=camelo-smoke-molmoact2-4gpu \
    slurm/train.slurm
