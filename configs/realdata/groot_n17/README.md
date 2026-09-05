# GR00T N1.7 — second wave, step-cut to the 24h cliff

**Purpose.** Best dimensional fit (132-dim slots take 27/15 unmodified); only rung with no gripper-polarity question (`new_embodiment` = fresh projectors); only rung with a falsifiable prediction (first TIMED close from GR00T on this cluster, not a task solve).
**Init:** local dir of `nvidia/GR00T-N1.7-3B` (must be a real directory, not the hub id — `huggingface-cli download ... --local-dir /scratch/gpfs/FHEIDE/JOST/models/GR00T-N1.7-3B`).
**State/action:** 27/15, zero-padded internally to 132/132.
**Gripper:** no flip, no binarize (fresh embodiment, nothing pretrained to match).
**Normalization:** GR00T's own internal min-max pack (`normalization_mapping=IDENTITY` by design, lerobot's normalizer bypassed).
**Cameras / image op:** dataset-side `Resize([256,256])` (uniformity for `np.stack` only) + GR00T's own shipped squash (`letter_box_transform=False`, U11). No `rename_map` — would raise on the `base_model_path` route.
**Horizon:** `chunk_size=40` (2.0s), `n_action_steps=8` (0.4s).
**Trainable:** `tune_projector=tune_diffusion_model=tune_vlln=true`, `tune_llm=tune_visual=false`, `use_relative_actions=false`.
**Batch/steps/LR:** batch 32, **30,000 steps (9.63 epochs)** — raised from the 20,000 provisional cut once smoke 3983181 measured 0.907 s/step against the 1.5–3.0 s/step ESTIMATE band; see train.yaml's re-derivation. AdamW lr 1e-4 cosine (preset). `save_freq=2500`.
**GPUs/walltime:** 1×H100, 22:00:00 requested. Est. ≤16.7h at worst-case 3.0 s/step (ESTIMATE, unmeasured).
**Launch:** `configs/realdata/groot_n17/launch.sh`
**Smoke:** `configs/realdata/groot_n17/smoke.sh` (300 steps, `gpu-test`, 00:30:00) — load-bearing, sets the final step count.
**Expect / selection probe:** M1 gripper-close timing on held-out episodes. Prior sim IoU 0.0, never trained successfully on this cluster before — bar is "does it train and produce a TIMED-shaped close at all."

---

**Launched 2026-08-31** (30,000 steps x batch 32 / 99,654 frames = **9.63 epochs**) — smoke **3983181 PASS** (updt_s 0.394, data_s 0.513 -> **0.907 s/step**, mem_gb 36.2, loss 0.741->0.086) -> long run **3983337**, 22:00:00. The measured rate is 1.7-3.3x faster than the 1.5-3.0 s/step ESTIMATE the 20,000-step cut was made against, so **steps raised to 30,000** (the protocol's original figure) -- 7.6 h of training, 8.8 h with the 15% margin, 40% of the request. `eval_steps: 0` (13_LAUNCH_LOG.md D-2).

_Epoch figures on this page are `steps x batch / 99,654` -- the frames the trainer actually sees. The corpus has 121,828 frames in 217 episodes; 195 are on the train list, and lerobot's `dataset.eval_split: 0.1` then holds back the last 20 of those, leaving **175 episodes / 99,654 frames** to train on (10,522 in the eval split). Earlier epoch numbers on this page were against a larger count._
