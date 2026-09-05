# ACT — first wave

**Purpose.** From-scratch control arm: no OOD-embodiment question, and the only checkpoint family with a measured vision-timed gripper close (0.958 near/0.000 far — on a *different* checkpoint than the vision-timing sweep itself; see 08 §2.3 reviewer note).
**Init:** none — from scratch (ResNet18/ImageNet via `TORCH_HOME`, staged with `make stage RUNG=act`).
**State/action:** 27/15 canonical; names in the derived corpus's `meta/info.json`.
**Gripper:** state dim 14 open fraction, action dim 14 width fraction (1.0=open), no flip.
**Normalization:** MEAN_STD (ACTConfig default).
**Cameras / image op:** native resolution, no crop/letterbox/resize (U11 — nothing to be OOD against).
**Horizon:** `chunk_size=21`, `n_action_steps=10` (1.05s/0.5s at 20Hz, time-matched from the sim corpus's 30fps `32/16`).
**Trainable:** everything, ~80M params, no PEFT.
**Batch/steps/LR:** batch 8, 100,000 steps (8.03 epochs), AdamW lr 1e-5 (preset). `save_freq=8333`.
**GPUs/walltime:** 1×H100, 12:00:00 requested. Est. 4.4h floor (M, sim corpus) — real corpus's AV1/720p frames expected slower, trust the smoke.
**Launch:** `configs/realdata/act/launch.sh`
**Smoke:** `configs/realdata/act/smoke.sh` (300 steps, `gpu-test`, 00:30:00)
**Expect / selection probe:** M1 gripper-close timing (TIMED/MISPLACED/UNTIMED/ABSENT) on held-out episodes. Watch for degradation under image staleness — that's the signature that makes a close vision- rather than proprioception-timed.

---

**Launched 2026-08-31** (100,000 steps x batch 8 / 99,654 frames = **8.03 epochs**) — smoke **3983177 PASS** (updt_s 0.088, data_s 0.002, mem_gb 11.4, 51,587,471 trainable, loss 15.72->3.30) -> long run **3983190**, 12:00:00, RUNNING. Measured 0.090 s/step: 100,000 steps is ~2.5 h of training, ~3.7 h with the eval passes.

_Epoch figures on this page are `steps x batch / 99,654` -- the frames the trainer actually sees. The corpus has 121,828 frames in 217 episodes; 195 are on the train list, and lerobot's `dataset.eval_split: 0.1` then holds back the last 20 of those, leaving **175 episodes / 99,654 frames** to train on (10,522 in the eval split). Earlier epoch numbers on this page were against a larger count._
