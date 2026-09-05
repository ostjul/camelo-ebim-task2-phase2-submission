#!/bin/bash
# R-2 smoke: prove the MolmoAct2 resume MECHANISM before trusting the 16 h run.
# Read configs/realdata/eo1/resume_smoke.sh's header for the full rationale,
# including why this runs at 2 GPUs rather than 4 (no node had four free GPUs
# at submission time; a 4-GPU gpu-test slot was six hours out).
#
# Pre-crash level to match: loss 0.774 at the last log line of job 3986557
# (19:21:15, step ~7000), with discrete_ce_loss 0.766 / action_flow_loss 0.007.
#
# 200 steps rather than EO-1's 300: this rung pays ~11 min of startup for
# 43.6 GB of weights, so 200 x ~2.3 s keeps the whole thing inside 30 min.
#
# eval_steps=100 / save_freq=200 puts an eval pass and the 14 GB save on the
# same final step 6450 — and the eval pass is exactly where rank 1 was killed
# (lerobot_train.py:659, `for eval_batch in eval_dataloader`).
#
# TRAIN_MODULE: this rung CANNOT resume through the stock `lerobot-train`.
# lerobot 0.6.1 pushes a hard-coded `normalizer_processor` override into the
# saved pipeline whenever `policy.pretrained_path` is set — which resume is
# what sets — and MolmoAct2's pipeline names that step
# `molmoact2_masked_normalizer`, so it dies before step 1 with a KeyError.
# Found by the R-2 smoke (job 3989706), fixed in camelo/train/resume_train.py;
# full mechanism in that module's docstring and 13_LAUNCH_LOG.md §8.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

mkdir -p outputs/slurm
RUN_DIR='outputs/runs/train_20260831_144615_3986557' \
TRAIN_MODULE='camelo.train.resume_train' \
GPUS=2 WORKERS=8 PREFETCH=2 \
OUT_DIR='outputs/runs/resume_smoke_molmoact2_%j' \
EXTRA='--steps=6450 --eval_steps=100 --save_freq=200 --log_freq=25' \
sbatch --export=ALL --qos=gpu-test --time=00:30:00 \
    --gres=gpu:2 --cpus-per-task=32 --mem=256G \
    --job-name=camelo-resume-smoke-molmoact2 \
    slurm/resume.slurm
