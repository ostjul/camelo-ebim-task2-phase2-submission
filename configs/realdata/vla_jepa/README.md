# VLA-JEPA — second wave

**Purpose.** Closest real-robot hardware match on the ladder (FR3 + Robotiq 2F-85 + 3 RealSense D435 + 100 demos/3 pick-place tasks, verified against the paper). Launched in Phase A, evaluated second half of the rig window.
**Init:** `lerobot/VLA-JEPA-Pretrain` (Qwen3-VL-2B + frozen V-JEPA2 ViT-L, ~3.1B params, ~7.5GB staged).
**State/action:** 27/15 via `reinit_modules` (re-inits the projections; rest keeps pretrained weights). Written as a JSON-array **string**, not a YAML list — a real list flattens to Python repr and draccus rejects it (caught by the parse-check).
**Open risk:** released checkpoint declares only 2 image inputs, no wrist camera — this config passes our native 3 keys through with no `rename_map`; verify at smoke time it doesn't silently drop a camera or error.
**Gripper:** state dim 14 open fraction, action dim 14 width fraction, no flip.
**Normalization:** policy defaults (MEAN_STD state, MIN_MAX action, IDENTITY visual).
**Horizon:** `chunk_size=n_action_steps=7` (0.35s at 20Hz) — tight replan; rig inference must return well under 0.35s.
**Trainable:** whole network except what `reinit_modules` excludes from the pretrained load.
**Batch/steps/LR:** batch 16 (ESTIMATE), 30,000 steps (4.82 epochs), AdamW lr 1e-4 cosine (preset), `scheduler_decay_steps=30000` (class default, matches). `save_freq=2500`.
**GPUs/walltime:** 1×H100, 16:00:00 requested. Est. 7–13h (ESTIMATE, unmeasured).
**Launch:** `configs/realdata/vla_jepa/launch.sh`
**Smoke:** `configs/realdata/vla_jepa/smoke.sh` (300 steps, `gpu-test`, 00:30:00)
**Expect / selection probe:** M1 gripper-close timing on held-out episodes. Card says "No evaluation results have been provided for this policy yet" — the lerobot port itself is unproven.

---

**Launched 2026-08-31** (30,000 steps x batch 16 / 99,654 frames = **4.82 epochs**) — smokes 3983180 FAIL / 3983302 FAIL / **3983402** -- four defects fixed, all verified by loading the real checkpoint on CPU: the missing `action_decoder` reinit prefix, the exterior_1/2_left camera keys (this rung runs on **two** cameras, wrist_left dropped), the gripper binarizer aimed at left-arm joint 6, and the camera-stack shape. See train.yaml and 13_LAUNCH_LOG.md V-1..V-3.

_Epoch figures on this page are `steps x batch / 99,654` -- the frames the trainer actually sees. The corpus has 121,828 frames in 217 episodes; 195 are on the train list, and lerobot's `dataset.eval_split: 0.1` then holds back the last 20 of those, leaving **175 episodes / 99,654 frames** to train on (10,522 in the eval split). Earlier epoch numbers on this page were against a larger count._
