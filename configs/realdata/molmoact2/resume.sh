#!/bin/bash
# R-2: RESUME the MolmoAct2 run that job 3986557 lost to a host OOM.
#
# WHAT KILLED IT (docs/realdata/13_LAUNCH_LOG.md "R-2"): the SAME host-OOM
# class as EO-1, not a per-model fault. SLURM `Detected 2 oom_kill events in
# StepId=3986557.0`, rank 1 SIGKILLed at 19:21:20; sacct MaxRSS 255.97G
# against ReqMem 256G. Ranks 2 and 3 died first with
#   RuntimeError: DataLoader worker ... killed by signal: Bus error.
#   It is possible that dataloader's workers are out of shared memory.
# raised at lerobot_train.py:659, `for eval_batch in eval_dataloader` — i.e.
# inside the step-7000 EVAL pass. /dev/shm is charged to the job cgroup, so
# "out of shared memory" and the OOM kill are one cause, not two.
#
# WHY EVAL IS THE TRANSIENT THAT TIPS IT: lerobot builds a SECOND DataLoader
# for the held-out split (lerobot_train.py:528) with the same num_workers and
# the same persistent_workers=True. Once the first eval pass runs, each rank
# holds 16 train + 16 eval workers for the rest of the run — 4 x 32 = 128
# `spawn`ed processes, none of them sharing library pages with the parent.
#
# THE FIX is identical to EO-1's (configs/realdata/eo1/resume.sh):
#   MEM 256G -> 512G (node RealMemory ~1002 GiB, MaxMemPerNode=UNLIMITED)
#   --num_workers 16 -> 8/rank, --prefetch_factor 4 -> 2  => 64 workers, not
#     128. This rung has less dataloader slack than EO-1 (data_s 0.157 vs
#     updt_s 2.11, ~7 % of the step) but still ~2x headroom at 8 workers;
#     the smoke measures data_s rather than assuming it.
#   eval_steps is left at 1000: against save_freq=6250 it already collides
#     only at 25000, the final step.
#
# RESUMES FROM 006250, which is COMPLETE (14 GB, weights + training_state).
#
# BUDGET. 25,000 - 6,250 = 18,750 steps at the measured 2.26 s/step wall
# = 11.8 h, plus ~11 min startup (43.6 GB of weights) and 19 capped evals.
# 16:00:00 requested, well under the 24 h cliff.
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
GPUS=4 WORKERS=8 PREFETCH=2 \
sbatch --export=ALL --time=16:00:00 \
    --gres=gpu:4 --cpus-per-task=64 --mem=512G \
    --job-name=camelo-resume-molmoact2 \
    slurm/resume.slurm
