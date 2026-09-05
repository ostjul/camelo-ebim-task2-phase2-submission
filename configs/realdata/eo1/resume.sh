#!/bin/bash
# R-1: RESUME the EO-1 run that job 3986556 lost to a host OOM.
#
# WHAT KILLED IT (docs/realdata/13_LAUNCH_LOG.md "R-1"): SLURM
# `Detected 1 oom_kill event in StepId=3986556.0`, rank 0 SIGKILLed at
# 22:46:21 *inside* save_checkpoint at step 15000. sacct: MaxRSS 255.97G
# against ReqMem 256G. The standing footprint was 4 ranks x (16 train + 16
# eval) = 128 `spawn`ed dataloader workers, and step 15000 is a multiple of
# BOTH eval_steps=2500 and save_freq=7500, so the eval pass and rank 0
# materialising 22 GB of model+AdamW state on the host landed on one step.
#
# RESUMES FROM 007500, NOT 015000. The OOM landed mid-save, so `015000/`
# holds only a 7.6 KB pretrained_model/config.json — no weights, no
# training_state — and `checkpoints/last` still (correctly) points at
# 007500. Nothing is deleted; the stub is simply skipped.
#
# THE FIX, both halves measured rather than reasoned (see the R-1 smoke):
#   MEM 256G -> 512G. The node is RealMemory=1026099 MB (~1002 GiB) and
#     partition `gpu` sets MaxMemPerNode=UNLIMITED, so 512G is allowed and
#     still leaves ~490 GiB for the CPU jobs that share these nodes.
#   --num_workers 16 -> 8 per rank and --prefetch_factor 4 -> 2, which takes
#     the standing worker count from 128 to 64 and halves the in-flight
#     batches. Free on throughput: this rung's data_s was 0.011 s against
#     updt_s 1.874 (0.6 % of the step) — it is compute-bound with enormous
#     dataloader slack.
#   --eval_steps 2500 -> 2000, so eval and save stop colliding. At 7500/2500
#     they coincided at EVERY save (15000/22500/30000); at 7500/2000 only at
#     30000, the final step, where a failure still leaves 22500 on disk.
#
# BUDGET. 30,000 - 7,500 = 22,500 steps at the measured 1.88 s/step = 11.8 h,
# plus ~6 min startup and 12 capped evals. 16:00:00 requested (~15 % margin),
# well under the 24 h cliff.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

mkdir -p outputs/slurm
RUN_DIR='outputs/runs/train_20260831_144338_3986556' \
GPUS=4 WORKERS=8 PREFETCH=2 \
EXTRA='--eval_steps=2000' \
sbatch --export=ALL --time=16:00:00 \
    --gres=gpu:4 --cpus-per-task=64 --mem=512G \
    --job-name=camelo-resume-eo1 \
    slurm/resume.slurm
