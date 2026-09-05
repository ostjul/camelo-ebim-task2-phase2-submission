# SmolVLA — opportunistic

**Purpose.** Cheap and fast, but far OOD from SmolVLA's SO-100/SO-101 single-arm hobby-servo pretraining on every axis, no internal baseline of any kind. Launch only if a GPU slot is idle.
**Init:** `outputs/checkpoints/smolvla_base_state27` (already rebuilt in this repo — metadata-only relabel of `lerobot/smolvla_base` fixing the F-77 stale 6-dim state declaration the raw hub checkpoint carries).
**State/action:** 27/15, padded to 32 internally.
**Gripper:** state dim 14 open fraction, action dim 14 width fraction, no flip.
**Normalization:** MEAN_STD (class default).
**Cameras / image op:** `rename_map` to generic camera1/2/3 slots (positional — no semantic names); `resize_imgs_with_padding=(512,512)` letterbox unchanged (U11).
**Horizon:** `chunk_size=n_action_steps=50` (2.5s).
**Trainable:** expert only (`train_expert_only=true`, `freeze_vision_encoder=true`, class defaults). `load_vlm_weights=true` **must be set explicitly** — default (False) is for training the expert from scratch.
**Batch/steps/LR:** batch 16, 40,000 steps (6.42 epochs), AdamW lr 1e-4 cosine (preset). `save_freq=3333`.
**GPUs/walltime:** 1×H100, 08:00:00 requested. Est. 2.1h floor (M-derived, sim corpus, 85 samples/s @ batch 16) — trust the smoke.
**Launch:** `configs/realdata/smolvla/launch.sh`
**Smoke:** `configs/realdata/smolvla/smoke.sh` (300 steps, `gpu-test`, 00:30:00)
**Expect / selection probe:** M1 gripper-close timing on held-out episodes. Worst executor-ceiling ratio in the program when previously zero-shot'd (34–35×) — treat any positive signal as noteworthy, not expected.

---

**Launched 2026-08-31** (40,000 steps x batch 16 / 99,654 frames = **6.42 epochs**) — smoke **3983184 PASS** (updt_s 0.157, data_s 0.008, mem_gb 6.1, 99,880,992 trainable, loss 1.258->0.219) -> long run **3983370**, 08:00:00. `scheduler_decay_steps: 40000` added so it matches `steps` -- the default 30,000 would have run the last 25% of the job at the 2.5e-6 LR floor (13_LAUNCH_LOG.md S-1).

_Epoch figures on this page are `steps x batch / 99,654` -- the frames the trainer actually sees. The corpus has 121,828 frames in 217 episodes; 195 are on the train list, and lerobot's `dataset.eval_split: 0.1` then holds back the last 20 of those, leaving **175 episodes / 99,654 frames** to train on (10,522 in the eval split). Earlier epoch numbers on this page were against a larger count._
