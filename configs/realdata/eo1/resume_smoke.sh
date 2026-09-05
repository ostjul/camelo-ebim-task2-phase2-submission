#!/bin/bash
# R-1 smoke: prove the EO-1 resume MECHANISM before trusting the 16 h run.
#
# WHAT THIS PROVES (all three of the gate's questions):
#   1. it resumes AT STEP 7500, not from the base checkpoint — the silent
#      failure mode this rung is exposed to, because EO-1's train.yaml carries
#      `policy.path` and lerobot checks that BEFORE `resume` (see
#      slurm/resume.slurm's header)
#   2. loss continues at the pre-crash level (~0.030 at step 15000 of job
#      3986556; ~0.026-0.030 across its last three log lines)
#   3. a checkpoint save completes — the operation that was interrupted
#
# WHY 2 GPUs AND NOT 4. Not a shortcut: at 01:20 on 09-01 no node in the gpu
# partition had four free GPUs (another user holds a ~20-job gpu:4 queue), so
# `sbatch --start` put a 4-GPU gpu-test smoke at **07:52** — six hours out,
# which would push the 16 h resume past the evening T0 for no added
# information. A 2-GPU job schedules immediately. The three questions above
# are all rank-count-independent; the one thing 2 GPUs cannot prove is that
# the 4-rank host footprint fits 512G, and THAT is measured instead by
# slurm/resume.slurm's MEMWATCH sampler — here at 2 ranks, so the per-rank
# number can be scaled and checked, and again live in the long run itself.
#
# It reproduces the FAILING MEASUREMENT rather than a proxy (AGENTS.md): with
# eval_steps=150 and save_freq=300 the run ends at step 7800 with an eval pass
# and a 22 GB checkpoint save on the SAME step — exactly the coincidence that
# OOM-killed rank 0 at step 15000.
#
# MEM=256G at 2 ranks is the SAME per-rank budget as the long run's 512G at 4,
# so the sampler's reading scales directly.
#
# OUT_DIR is a throwaway dir, so the smoke reads the real 007500 checkpoint
# but can never write a short-run checkpoint into the real run tree.
# Delete it afterwards: the save is 22 GB and the fileset is at 2 % free.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

mkdir -p outputs/slurm
RUN_DIR='outputs/runs/train_20260831_144338_3986556' \
GPUS=2 WORKERS=8 PREFETCH=2 \
OUT_DIR='outputs/runs/resume_smoke_eo1_%j' \
EXTRA='--steps=7800 --eval_steps=150 --save_freq=300 --log_freq=25' \
sbatch --export=ALL --qos=gpu-test --time=00:30:00 \
    --gres=gpu:2 --cpus-per-task=32 --mem=256G \
    --job-name=camelo-resume-smoke-eo1 \
    slurm/resume.slurm
