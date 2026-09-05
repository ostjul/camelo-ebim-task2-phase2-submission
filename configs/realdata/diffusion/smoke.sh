#!/bin/bash
# 300-step gpu-test smoke -- checks --policy.horizon=24 parses and is
# accepted (must be a multiple of 8), VRAM with three independent ResNet18
# encoders, updt_s. 00:30:00, not `make train-smoke`'s hardcoded 00:10:00.
#
# Two attempts failed here, both before step 1 (full diagnosis and fix in
# train.yaml's IMAGE HANDLING block):
#   3983178  ValueError: `observation.images.wrist_left` does not match
#            `observation.images.head` -- the CONFIG-level shape check
#            (DiffusionConfig.validate_features).
#   3983235  RuntimeError: stack expects each tensor to be equal size, but
#            got [8,2,3,720,1280] ... and [8,2,3,480,640] -- the RUNTIME
#            torch.stack in DiffusionPolicy.forward (modeling_diffusion.py:170).
#            Closing the config check does NOT close this one; they need
#            different mechanisms.
# This smoke now has to clear both at once.
#
# No forced eval pass, on purpose: the config sets `eval_steps: 0`, because
# lerobot builds the eval dataset WITHOUT dataset.image_transforms
# (datasets/factory.py:209), so an eval pass would re-raise the runtime stack
# error at step 1000 -- exactly the class of bug a 300-step smoke cannot see,
# and the reason that deviation is written down in train.yaml rather than
# discovered 15 minutes into the long run.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source configs/realdata/_lib.sh

EPISODES="$(train_episodes_flag)"
mkdir -p outputs/slurm

CONFIG='configs/realdata/diffusion/train.yaml' DATA="$DATA" \
EXTRA="--steps=300 --save_freq=300 --log_freq=50 $EPISODES" \
sbatch --export=ALL --qos=gpu-test --time=00:30:00 \
    --gres=gpu:1 --cpus-per-task=8 --mem=64G --job-name=camelo-smoke-diffusion \
    slurm/train.slurm
