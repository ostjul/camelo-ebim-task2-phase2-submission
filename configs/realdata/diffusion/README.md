# Diffusion Policy — first wave

**Purpose.** From-scratch control arm #2: no OOD question, and the only architecture that can represent the corpus's multimodality (58/238 episodes have a re-grasp; pad sits in ≥3 slots). Zero prior measurement of any kind.
**Init:** none — from scratch.
**State/action:** 27/15 canonical.
**Gripper:** state dim 14 open fraction, action dim 14 width fraction, no flip.
**Normalization:** MIN_MAX (STATE+ACTION, class default) — sensitive to the corpus's 4.98% all-zero right-wrench dropout (ep15: 81.8% zeros). Sanity-check q01/q99 before trusting a run.
**Cameras / image op:** native resolution, no crop; three separate ResNet18 encoders (`use_separate_rgb_encoder_per_camera`, class default).
**Horizon:** `horizon=24` (not `chunk_size` — no such field; nearest multiple of 8, the U-Net downsampling factor), `n_action_steps=12`, `n_obs_steps=2`.
**Trainable:** everything, from scratch.
**Batch/steps/LR:** batch 8, 40,000 steps (3.21 epochs — do not port the sim corpus's 200k), AdamW lr 1e-4 cosine (preset). `save_freq=3333`.
**GPUs/walltime:** 1×H100, 08:00:00 requested. Est. 3.3–6.7h (ESTIMATE, unmeasured) — trust the smoke.
**Launch:** `configs/realdata/diffusion/launch.sh`
**Smoke:** `configs/realdata/diffusion/smoke.sh` (300 steps, `gpu-test`, 00:30:00)
**Expect / selection probe:** M1 gripper-close timing on held-out episodes. No behavioral prior exists — rank purely on the probe.

---

**Launched 2026-08-31** (40,000 steps x batch 8 / 99,654 frames = **3.21 epochs**) — smokes 3983178 FAIL / 3983235 FAIL / **3983369 PASS** -> long run **3983411**, 08:00:00. Two independent camera-shape defects had to be fixed (see train.yaml and 13_LAUNCH_LOG.md D-1/D-2); the rung now runs on 240x320 frames and with `eval_steps: 0`. Measured 0.083 s/step, mem_gb 5.98, 293,923,439 trainable, loss 1.056->0.239: 40,000 steps is ~1 h.

_Epoch figures on this page are `steps x batch / 99,654` -- the frames the trainer actually sees. The corpus has 121,828 frames in 217 episodes; 195 are on the train list, and lerobot's `dataset.eval_split: 0.1` then holds back the last 20 of those, leaving **175 episodes / 99,654 frames** to train on (10,522 in the eval split). Earlier epoch numbers on this page were against a larger count._
