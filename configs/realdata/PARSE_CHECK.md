# Parse check — `configs/realdata/*/train.yaml` against installed lerobot 0.6.1

Verifies every flag name/value in every config is accepted by the real
`TrainPipelineConfig`/policy dataclasses, **without training, without
SLURM/GPU, and without the derived corpus existing** — `task2_munich_s27a15`
has not been built yet (a sibling agent owns that). Run on the login node.

## Method

For each `configs/realdata/<policy>/train.yaml`:

1. Load the YAML with `yaml.safe_load`, then apply the exact same injections
   `camelo/train/train.py:main()` applies before invoking `lerobot-train`
   (`output_dir`/`job_name` defaults, `dataset.repo_id`/`dataset.root` from
   `--data`, `wandb.enable=false`, `policy.push_to_hub` default) — this is
   not a re-implementation, it imports and calls
   `camelo.train.train.flatten()` directly.
2. Append `--dataset.episodes=[0,1,2,3,4]` in place of the real
   `meta/train_episodes.txt` (the launch scripts read the real file at
   submission time — see `<policy>/launch.sh` / `_lib.sh`; the corpus and
   that file don't exist yet, so this is the documented mock).
3. Build `sys.argv` from the resulting flags and call the **real**
   two-step lerobot entry sequence, not a shortcut:
   `draccus.parse(TrainPipelineConfig, ...)` followed by `cfg.validate()`
   — mirroring `lerobot.scripts.lerobot_train.train()` (`@parser.wrap()`,
   which calls `cfg.validate()` immediately after parsing). Skipping
   `validate()` looked like it worked at first but silently left
   `cfg.policy=None` for every `--policy.path`-based config, because
   `--policy.path` is resolved by `TrainPipelineConfig._resolve_pretrained_from_cli()`
   (called from `validate()`), not by `draccus.parse()` alone — this is a
   real trap in lerobot 0.6.1's CLI plumbing, not something specific to
   this repo, and the first version of this check missed it.
4. Ran with `HF_HOME=/scratch/gpfs/FHEIDE/JOST/.cache/huggingface`,
   `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1` —
   the same env `slurm/train.slurm` exports — so any `--policy.path`
   resolution reads from the real HF cache exactly as a submitted job
   would, and fails the same way if a checkpoint isn't actually staged.

Harness: `/tmp/parsecheck_scratch/run_all.py` (not committed — a login-node
scratch script; reproduce by pointing it at this repo and running with the
env above).

## Result: 11/11 PASS (after fixing 2 real bugs the check itself found)

| config | class resolved | key fields confirmed |
|---|---|---|
| `act/train.yaml` | `ACTConfig` | `chunk_size=21 n_action_steps=10`, no `pretrained_path` |
| `diffusion/train.yaml` | `DiffusionConfig` | `horizon=24 n_action_steps=12 n_obs_steps=2` |
| `pi05_lora_15k/train.yaml` | `PI05Config` | `pretrained_path=lerobot/pi05_base`, `chunk_size=n_action_steps=50`, `scheduler_decay_steps=15000`, `dtype=bfloat16`, `gradient_checkpointing=True`, peft `r=32 alpha=64` targets resolved |
| `pi05_lora_30k/train.yaml` | `PI05Config` | same, `scheduler_decay_steps=30000` |
| `vla_jepa/train.yaml` | `VLAJEPAConfig` | `pretrained_path=lerobot/VLA-JEPA-Pretrain`, `chunk_size=n_action_steps=7`, `reinit_modules=['model.action_model.action_encoder', 'model.action_model.state_encoder']` (real `list[str]`, decoded from the JSON-string form — see bug #2 below) |
| `groot_n17/train.yaml` | `GrootConfig` | `base_model_path=/scratch/gpfs/FHEIDE/JOST/models/GR00T-N1.7-3B`, `chunk_size=40 n_action_steps=8`, `max_steps=20000` (== `steps`), `use_relative_actions=False`, `dataset.image_transforms.tfs={'resize': ...(256,256)}` |
| `eo1/train.yaml` | `EO1Config` | `vlm_base=IPEC-COMMUNITY/EO-1-3B` (not `pretrained_path` — see bug #1 below), `chunk_size=n_action_steps=8` |
| `molmoact2/train.yaml` | `MolmoAct2Config` | `checkpoint_path=allenai/MolmoAct2`, `chunk_size=n_action_steps=20`, `image_keys=[head,wrist_left,wrist_right]`, `setup_type`/`control_mode` set, `enable_lora_vlm=True`, `scheduler_decay_steps=25000`, `model_dtype=bfloat16` |
| `molmoact2_ddp2_rehearsal/train.yaml` | `MolmoAct2Config` | same fields, `steps=300 batch_size=16` |
| `pi0_fast/train.yaml` | `PI0FastConfig` | `pretrained_path=outputs/models/pi0fast-base-lerobot-tokenizer`, `chunk_size=n_action_steps=50`, `scheduler_decay_steps=25000`, peft `r=32 alpha=64` targets resolved |
| `smolvla/train.yaml` | `SmolVLAConfig` | `pretrained_path=outputs/checkpoints/smolvla_base_state27`, `load_vlm_weights=True`, `chunk_size=n_action_steps=50` |

All 11 also confirmed: `tolerance_s=1.01` **top-level** (not
`dataset.tolerance_s`, which does not exist as a field — matches
`docs/realdata/00_PROTOCOL.md` D2), `dataset.video_backend=pyav`,
`dataset.eval_split=0.1`, `dataset.episodes` decoded to a real
`list[int]`, `policy.push_to_hub=False`, and the injected mock
`--dataset.episodes` round-tripping correctly through `sbatch`'s
comma-splitting concern (confirmed by inspection of `slurm/train.slurm`'s
own header comment and `_lib.sh` — the bracketed list travels inside
`EXTRA` via `--export=ALL`, never through a literal `--export=VAR=...`).

## Two real bugs found and fixed by this check (not by the docs)

**1. `eo1/train.yaml` — `--policy.path=IPEC-COMMUNITY/EO-1-3B` does not
work.** Every research doc (00/03/06/08) describes EO-1's init the same
way every other `--policy.path`-based policy is described, and that is
what this task originally wrote. The parse-check caught it:

```
ValueError: Missing 'type' field in config.json of IPEC-COMMUNITY/EO-1-3B
```

Investigated further (not just patched around): `IPEC-COMMUNITY/EO-1-3B`'s
`config.json` is a raw `transformers`-style config
(`model_type="eo1"`, `auto_map` → `EO1VisionFlowMatchingModel`), **not** a
lerobot-native checkpoint — same shape as the GR00T/MolmoAct2 "raw HF
repo" trap, just undocumented for EO-1. Manually patching in a `"type":
"eo1"` key still fails —
`draccus.DecodingError: The fields action_chunk_size, ... (33 total) are
not valid for EO1Config` — the field *names* are completely different
from `EO1Config`, not just missing a tag. `--policy.path` is structurally
the wrong mechanism for this checkpoint in lerobot 0.6.1.

Fix: `EO1Config`'s only pretrained-weight-bearing field is `vlm_base`
(default `"Qwen/Qwen2.5-VL-3B-Instruct"`; `EO1Config.__post_init__` calls
`Qwen2_5_VLConfig.from_pretrained(self.vlm_base)`). Verified directly:

```python
>>> from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
>>> Qwen2_5_VLConfig.from_pretrained('IPEC-COMMUNITY/EO-1-3B')  # succeeds, model_type='qwen2_5_vl'
```

`transformers`' own `from_pretrained` tolerates the extra EO-1-specific
fields where draccus's strict dataclass decoder does not. Changed the
config to `policy.type: eo1` + `policy.vlm_base: IPEC-COMMUNITY/EO-1-3B`
(no `path`). This warm-starts from EO-1's own further-pretrained backbone
config/weights rather than plain Qwen2.5-VL — **but this was verified only
at the config layer**, not confirmed to actually pull EO-1's fine-tuned
weight *values* (vs. just its architecture hyperparameters) at model-load
time; flagged in `eo1/README.md` as unverified, to be confirmed at smoke
time (S7).

**2. `vla_jepa/train.yaml` — a YAML list for `reinit_modules` does not
parse.** `camelo/train/train.py`'s `flatten()` stringifies a Python list
value with `str()` (Python repr, single-quoted), which draccus's
`list[str]` decoder rejects:

```
draccus.utils.DecodingError: `reinit_modules`: Could not decode the value
into any of the given types:
    list[str]: The given value='[\'model.action_model.action_encoder\', ...]' is not of a valid input
```

Fix: write it as a JSON-array **string** in the YAML (`reinit_modules:
'["a", "b"]'`), the same convention `groot_n17/train.yaml`'s
`dataset.image_transforms.tfs` already uses for the identical reason —
confirmed this decodes to a real `list[str]` (`['model.action_model.action_encoder',
'model.action_model.state_encoder']`) after the fix.

## What this check does *not* verify

- That a `--policy.path`/`base_model_path`/`checkpoint_path` checkpoint's
  **weights** actually load and are shape-compatible beyond its
  `config.json` (`draccus.parse` + `cfg.validate()` only resolves and
  decodes the *config*; no `PreTrainedPolicy.from_pretrained()` /
  `make_policy()` call was made, no state dict was touched).
- Anything that depends on the real dataset existing (`dataset_to_policy_features`,
  camera-key consistency, `check_state_width`, the A6 gripper-mask sidecar
  logic in `camelo/train/train.py`) — all gated behind
  `outputs/datasets/ebim_task2_realrobotdata/task2_munich_s27a15` existing,
  which it does not yet.
- GR00T's `infer_groot_model_version(base_model_path)` — harmless on a
  not-yet-staged local path (returns `None`, no error), but not exercised
  against the real staged snapshot either.
- VLA-JEPA's camera-count question (checkpoint declares 2 image inputs,
  our corpus has 3) — `validate_features()` only requires "at least one",
  so this doesn't surface as a parse-time error either way; see
  `vla_jepa/README.md`.
