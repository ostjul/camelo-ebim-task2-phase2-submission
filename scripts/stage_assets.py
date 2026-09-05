#!/usr/bin/env python3
"""Pull datasets + base checkpoints into the local HF cache — LOGIN NODE ONLY.

    python scripts/stage_assets.py --datasets hermanprawiro/task2_fixpos_v1
    python scripts/stage_assets.py --models lerobot/pi0_base nvidia/GR00T-N1.7-3B

On a SLURM cluster whose compute nodes have no route to the internet
(tiger3: DNS itself fails — docs/setup/TIGER3_H100.md), every weight and every
dataset has to be resident in the cache *before* a job starts. A
fine-tune that reaches for the hub mid-job does not fall back to the
cache; it dies several minutes in, after the queue wait.

This is the counterpart to the `HF_HUB_OFFLINE=1` the job scripts export:
staging puts the bytes there, the env var proves nothing tried to fetch
more. Both are needed — offline mode over a cold cache fails just as hard
as no offline mode over no network, only with a less obvious error.

Idempotent: already-cached repos re-verify and return immediately, so
re-running before every submission is the cheap habit.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Datasets and checkpoints the training ladder needs (docs/runbooks/TRAINING.md).
# Not staged by default — pulling every rung is ~40 GB — but named here so
# `--all-models` and the docs cannot drift apart.
# Each VLA rung needs its base checkpoint AND the separate repos its
# processor pipeline reaches for — a language tokenizer, a vision backbone.
# Those loads happen at MODEL CONSTRUCTION, long after the "download"
# looks done, so every entry below cost one failed job to discover
# (docs/setup/TIGER3_H100.md). Staging is per-rung for exactly this reason.
LADDER_MODELS = {
    "act": [],  # trained from scratch, no base checkpoint
    "xvla": ["lerobot/xvla-base", "facebook/bart-large"],
    # GR00T pulls its backbone separately, and that backbone is gated.
    "groot": ["nvidia/GR00T-N1.7-3B", "nvidia/Cosmos-Reason2-2B"],
    "pi0": ["lerobot/pi0_base", "google/paligemma-3b-pt-224"],
    # pi0_fast's own text tokenizer defaults to google/paligemma-3b-pt-224
    # (configuration_pi0_fast.py: text_tokenizer_name) — already staged by
    # the pi0/droid8 rungs above, so it is not repeated here.
    #
    # `lerobot/pi0fast-base`'s SAVED preprocessor pipeline
    # (policy_preprocessor.json, baked at export time) pins
    # action_tokenizer_name to the ORIGINAL upstream repo
    # 'physical-intelligence/fast', while the checkpoint's OWN config.json
    # already declares lerobot's mirror 'lerobot/fast-action-tokenizer' as
    # the default — the two drifted apart at export. Only the preprocessor
    # pipeline's value is actually used at load time when `--policy.path`
    # is set: `make_pre_post_processors` loads the SAVED
    # policy_preprocessor.json straight off disk in that case rather than
    # rebuilding it from `PI0FastConfig.action_tokenizer_name`
    # (lerobot/policies/factory.py: the `if pretrained_path:` branch calls
    # `PolicyProcessorPipeline.from_pretrained(...)` and never reaches
    # `make_pi0_fast_pre_post_processors`, which is the only place that
    # config field is read) — so job 3962341 loaded the MODEL fine and then
    # died reaching for 'physical-intelligence/fast' under HF_HUB_OFFLINE=1,
    # and no `--policy.*` config override can fix it: lerobot's trainer
    # builds no CLI-reachable hook into that load at all
    # (`TrainPipelineConfig` has no `preprocessor_overrides` field; measured
    # directly against this transformers version, see git history for the
    # offline repro).
    #
    # 'physical-intelligence/fast' is NOT staged as the fix: its repo lays
    # tokenizer files at the ROOT, but this transformers version's
    # AutoProcessor always probes them under a 'bpe_tokenizer/' subfolder
    # (matching lerobot/fast-action-tokenizer's OWN layout) — a genuine
    # repo/loader mismatch, not a caching gap. Confirmed it fails even
    # ONLINE (`ValueError: Couldn't instantiate the backend tokenizer...`),
    # so no amount of staging or stubbing fixes it. The tokenizer content
    # is byte-identical between the two repos (diffed tokenizer.json), so
    # the real fix is `derive_pi0fast_base` below: build a LOCAL checkpoint
    # directory that symlinks every file back into the pristine HF cache
    # except policy_preprocessor.json, which gets the one-field rewrite.
    # `configs/pi0fast_ft.yaml` points `policy.path` at that directory
    # instead of the hub id — the HF cache itself is never mutated.
    "pi0fast": ["lerobot/pi0fast-base", "lerobot/fast-action-tokenizer"],
    "smolvla": ["lerobot/smolvla_base", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"],
    # MolmoAct2 is ~43.6 GB on disk, not the 21.8 GB its card advertises.
    # pi05_droid_jointpos loads PaliGemma as its tokenizer at MODEL LOAD
    # time — a missing gate acceptance fails the job, not the download.
    "droid8": [
        "lerobot/MolmoAct2-DROID-LeRobot",
        "DAVIAN-Robotics/pi05_droid_jointpos",
        "google/paligemma-3b-pt-224",
    ],
    # docs/realdata/08_MODEL_SELECTION_AND_RIG_SCHEDULE.md §4.1 item A4 / 00
    # §3 item 2 rungs, not staged by any rung above. checkpoint_path defaults
    # to 'allenai/MolmoAct2' itself (configuration_molmoact2.py:37) — read
    # directly (no torch, no weights) and confirmed [2026-08-31]: its
    # config.json carries NO second `checkpoint_path`/repo reference, unlike
    # the sibling DROID checkpoint the "droid8" rung above stages (F-71) —
    # so the base does NOT transitively pull MolmoAct2-DROID. The FAST
    # tokenizer is needed because `action_mode` defaults to "both"
    # (configuration_molmoact2.py:45), which uses the discrete head.
    "molmoact2": ["allenai/MolmoAct2", "allenai/MolmoAct2-FAST-Tokenizer"],
    # VLA-JEPA-Pretrain's two backbones (configuration_vla_jepa.py:42-43):
    # Qwen3-VL-2B-Instruct was already cached by an earlier rung's pull, but
    # listed explicitly here so this rung alone is self-sufficient on a
    # fresh cache.
    "vlajepa": [
        "lerobot/VLA-JEPA-Pretrain",
        "Qwen/Qwen3-VL-2B-Instruct",
        "facebook/vjepa2-vitl-fpc64-256",
    ],
    # EO-1: stage `lerobot/eo1-base`, NOT `IPEC-COMMUNITY/EO-1-3B`.
    #
    # IPEC-COMMUNITY/EO-1-3B is the ORIGINAL EO-1 codebase's native HF
    # checkpoint (config.json: model_type "eo1", auto_map ->
    # configuration_eo1.EO1VisionFlowMatchingConfig). Nothing in lerobot
    # 0.6.1 can load its weights: it names every backbone tensor
    # `vlm_backbone.model.layers.*` / `vlm_backbone.visual.*` while
    # `Qwen2_5_VLForConditionalGeneration` expects `model.language_model.*` /
    # `model.visual.*`, so pointing `vlm_base` at it binds **0 of 836**
    # tensors and `from_pretrained` returns a random 3.77 B model without
    # raising (F-81; docs/realdata/13_LAUNCH_LOG.md §4 E-1).
    #
    # `lerobot/eo1-base` is upstream's own lossless key-layout conversion of
    # that exact revision (fe0a015d…) into lerobot's `eo1` module tree, and
    # it also carries EO-1's extended tokenizer (the 7 `<|state_pad|>` /
    # `<|action_pad|>` / … tokens at 151665-151671 that Qwen's own vocab
    # stops short of). It is loaded through `--policy.path`, not `vlm_base`,
    # so both the Qwen backbone AND EO-1's flow-matching action head come
    # from the checkpoint. `derive_eo1_base` below turns it into a directory
    # lerobot 0.6.1 can actually consume.
    "eo1": ["lerobot/eo1-base"],
}

# Repos whose terms must be accepted in a browser before any token can
# fetch them. They 403 rather than 404, and on a cluster whose compute
# nodes have no network there is no recovery once the job has started —
# so staging checks them up front and says which page to open.
MANUAL_GATED = {
    "google/paligemma-3b-pt-224": "pi05_droid_jointpos's tokenizer",
    "nvidia/Cosmos-Reason2-2B": "GR00T N1.7's backbone",
}

# Some `trust_remote_code=True` processors probe an optional per-subfolder
# file that the repo simply never shipped (e.g. a nested tokenizer's own
# config.json). Online, transformers treats the resulting 404 as
# informative and falls back gracefully; HF_HUB_OFFLINE=1 has never made
# that request before, so it cannot tell "confirmed absent" from "network
# is down" and refuses to guess — the exact probe that is harmless online
# is fatal offline. Measured: PI0Fast's own tokenizer load,
# `AutoProcessor.from_pretrained('lerobot/fast-action-tokenizer',
# trust_remote_code=True)`, raises LocalEntryNotFoundError under
# HF_HUB_OFFLINE=1 on `bpe_tokenizer/config.json`, which the repo never
# ships (confirmed via the online call, which 404s on it and moves on).
# Materializing an empty JSON stub at the exact probed path gives offline
# mode a real, already-"downloaded" file to find — the harness's own
# config-loading code then falls through to `tokenizer_config.json`'s
# explicit `tokenizer_class`, exactly as it does online after the 404.
POST_STAGE_STUBS: dict[str, list[str]] = {
    "lerobot/fast-action-tokenizer": ["bpe_tokenizer/config.json"],
}


def write_post_stage_stubs(repo_id: str, snapshot_dir: Path) -> None:
    for rel in POST_STAGE_STUBS.get(repo_id, []):
        target = snapshot_dir / rel
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}\n")
        print(f"  -> stubbed {target} (offline-probe workaround, see POST_STAGE_STUBS)")


# Where the pi0_fast derived base checkpoint lives — NOT under the HF
# cache. See `derive_pi0fast_base` for why: the cache stays a byte-for-byte,
# re-downloadable mirror of what huggingface.co serves, and this directory
# (created by `make stage RUNG=pi0fast`, not shipped in the repo) carries
# the one file that actually needs to differ from upstream.
PI0FAST_DERIVED_BASE = Path("outputs/models/pi0fast-base-lerobot-tokenizer")


def derive_pi0fast_base(dst: Path = PI0FAST_DERIVED_BASE) -> None:
    """Build a LOCAL checkpoint directory: `lerobot/pi0fast-base` with its
    `policy_preprocessor.json` action tokenizer repointed to the mirror that
    actually loads offline (see the long comment on LADDER_MODELS["pi0fast"]).

    An earlier version of this function rewrote the HF cache's OWN copy of
    `policy_preprocessor.json` in place. That worked, but left the cache no
    longer a faithful mirror of the upstream repo — a diff against
    huggingface.co would show a local edit nothing in the repo history
    explains, and there is no CLI-reachable config override that could do
    this instead (`make_pre_post_processors` loads the saved pipeline
    straight off disk whenever `--policy.path` is set — see the
    LADDER_MODELS["pi0fast"] comment — and lerobot's trainer builds no hook
    for user-supplied processor overrides at all).

    So this derives a SEPARATE checkout instead: every file except the one
    that needs to change is a symlink back into the pristine cache blob (no
    extra copy of the ~11 GB model.safetensors), and only
    `policy_preprocessor.json` is a real, rewritten file. The HF cache
    itself is never touched. `configs/pi0fast_ft.yaml` points `policy.path`
    at this directory instead of the hub id.

    Idempotent — safe to call on every `make stage`.
    """
    import json

    from huggingface_hub import snapshot_download

    src = Path(
        snapshot_download(repo_id="lerobot/pi0fast-base", repo_type="model", local_files_only=True)
    )
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if not item.is_file() or item.name == "policy_preprocessor.json":
            continue
        link = dst / item.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(item.resolve())

    data = json.loads((src / "policy_preprocessor.json").resolve().read_text())
    for step in data.get("steps", []):
        if step.get("registry_name") != "action_tokenizer_processor":
            continue
        cfg = step.get("config", {})
        if cfg.get("action_tokenizer_name") == "physical-intelligence/fast":
            cfg["action_tokenizer_name"] = "lerobot/fast-action-tokenizer"
    (dst / "policy_preprocessor.json").write_text(json.dumps(data, indent=2) + "\n")

    (dst / "README.md").write_text(
        "# pi0fast-base, lerobot tokenizer mirror\n\n"
        "Derived from the cached `lerobot/pi0fast-base` snapshot by "
        "`scripts/stage_assets.py:derive_pi0fast_base` (re-run any time via "
        "`make stage RUNG=pi0fast`). Every file here is a symlink back into "
        "the pristine HF cache EXCEPT `policy_preprocessor.json`, which is a "
        "real file with one field changed:\n\n"
        "    action_tokenizer_processor.action_tokenizer_name:\n"
        "      physical-intelligence/fast -> lerobot/fast-action-tokenizer\n\n"
        "Why: the upstream export pins the action tokenizer to "
        "`physical-intelligence/fast`, whose tokenizer files sit at the repo "
        "root, while this transformers version always probes them under a "
        "`bpe_tokenizer/` subfolder — the load fails even ONLINE "
        "(`ValueError: Couldn't instantiate the backend tokenizer...`), so no "
        "amount of staging fixes it. `lerobot/fast-action-tokenizer` carries "
        "byte-identical tokenizer content (diffed `tokenizer.json`), laid out "
        "in the subfolder this loader expects, and is what the checkpoint's "
        "OWN `config.json` already names as `action_tokenizer_name` — so this "
        "resolves a drift between two files in the SAME checkpoint export, "
        "not a substitution.\n\n"
        "The HF cache is left untouched: `models--lerobot--pi0fast-base` "
        "stays a byte-for-byte mirror of what huggingface.co serves.\n"
    )
    print(
        f"  -> derived {dst} "
        "(policy_preprocessor.json action_tokenizer_name -> lerobot/fast-action-tokenizer)"
    )


# Where the EO-1 derived base checkpoint lives. Same rationale as
# PI0FAST_DERIVED_BASE: the HF cache stays a byte-for-byte mirror of what
# huggingface.co serves and every file that must differ is a real file here.
EO1_DERIVED_BASE = Path("outputs/checkpoints/eo1_3b_lerobot")

# The corpus the EO-1 preprocessor pipeline is baked against — the same root
# `configs/realdata/_lib.sh` names. See `derive_eo1_base` for why a *dataset*
# path is an input to staging a *checkpoint*.
EO1_CORPUS_DEFAULT = Path(
    "outputs/datasets/ebim_task2_realrobotdata/task2_munich_s27a15_hires"
)

# EO1Config in lerobot 0.6.1 does not have these fields; `lerobot/eo1-base`
# was exported by a later lerobot, and draccus REJECTS unknown keys
# (`DecodingError: The fields ... are not valid for EO1Config`), so the
# config.json cannot be used as shipped. None of the six changes behaviour
# under 0.6.1 — there is no language-recipe path, no text loss and no PEFT
# recipe in this version to configure — so dropping them is a no-op, not a
# silent downgrade. Recorded here, not inline, so the list is auditable.
EO1_UNSUPPORTED_CONFIG_KEYS = (
    "flow_loss_weight",     # 1.0 = the only value 0.6.1's flow head implements
    "text_loss_weight",     # 0.01; 0.6.1 computes no text loss at all
    "tokenizer_max_length",  # 1000; 0.6.1 does not truncate
    "use_language_recipe",  # false
    "recipe",               # the language-recipe message template
    "recipe_path",          # null
)


def derive_eo1_base(
    dst: Path = EO1_DERIVED_BASE, dataset_root: Path = EO1_CORPUS_DEFAULT
) -> None:
    """Build a LOCAL EO-1 checkpoint directory lerobot 0.6.1 can load.

    `lerobot/eo1-base` is upstream's lossless key-layout conversion of
    `IPEC-COMMUNITY/EO-1-3B` (revision fe0a015d…) — 836 tensors already named
    the way `EO1Policy`'s module tree wants them, plus EO-1's extended
    tokenizer. It cannot be used as shipped for two reasons, both measured
    rather than read:

    1. **Its `config.json` was written by a later lerobot.** Six fields
       (`EO1_UNSUPPORTED_CONFIG_KEYS`) do not exist on 0.6.1's `EO1Config`
       and draccus rejects unknown keys outright. This directory carries a
       real `config.json` with exactly those six removed, `vlm_base` and
       `pretrained_path` repointed at itself (so nothing reaches the hub on
       an offline compute node), and `use_fast_processor: true`.

       That last one is not cosmetic. `accelerator.prepare(dataloader)` hands
       the training loop batches that are ALREADY on the GPU, and the PIL
       image-processor backend calls `Tensor.numpy()` on them —
       `TypeError: can't convert cuda:0 device type tensor to numpy`, which
       is what killed smoke job 3983182 before step 1. The torchvision
       backend keeps the tensor a tensor and only `.to(device)`s it.

    2. **It ships no processor pipeline.** With `--policy.path` set,
       `make_pre_post_processors` loads `policy_preprocessor.json` straight
       off disk and never rebuilds it from the config (the same load path
       documented at length on `LADDER_MODELS["pi0fast"]`), so a checkpoint
       without one raises `ProcessorMigrationError`. This function generates
       it with lerobot's own `make_eo1_pre_post_processors`.

    Which is why a DATASET path is an input here: `EO1ConversationTemplateStep`
    stores the visual input-feature keys in that JSON and derives its image
    list from them, and `lerobot_train` overrides **no** field of that step.
    The generated pipeline therefore names this corpus's three cameras, and
    the artifact is corpus-specific — pass `dataset_root` to rebuild it for
    another corpus. (The normalization stats it also writes are overridden
    from `dataset.meta.stats` at train time, so those are not a hidden input.)

    Idempotent — safe to call on every `make stage`.
    """
    import json

    from huggingface_hub import snapshot_download

    src = Path(
        snapshot_download(repo_id="lerobot/eo1-base", repo_type="model", local_files_only=True)
    )
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if not item.is_file() or item.name == "config.json":
            continue
        link = dst / item.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(item.resolve())

    cfg = json.loads((src / "config.json").resolve().read_text())
    for key in EO1_UNSUPPORTED_CONFIG_KEYS:
        cfg.pop(key, None)
    resolved = str(dst.resolve())
    cfg["vlm_base"] = resolved
    cfg["pretrained_path"] = resolved
    cfg["repo_id"] = None
    cfg["push_to_hub"] = False
    cfg["use_fast_processor"] = True
    cfg["chunk_size"] = 8
    cfg["n_action_steps"] = 8
    (dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"  -> derived {dst} (config.json: dropped {', '.join(EO1_UNSUPPORTED_CONFIG_KEYS)})")

    dataset_root = Path(dataset_root)
    if not (dataset_root / "meta" / "info.json").is_file():
        print(
            f"  !! {dataset_root} not found — processor pipeline NOT generated.\n"
            "     Training would raise ProcessorMigrationError. Re-run with the\n"
            "     corpus present:  python scripts/stage_assets.py --rung eo1"
        )
        return

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.eo1.processor_eo1 import make_eo1_pre_post_processors
    from lerobot.utils.feature_utils import dataset_to_policy_features

    policy_cfg = PreTrainedConfig.from_pretrained(str(dst))
    meta = LeRobotDatasetMetadata(str(dataset_root), root=str(dataset_root))
    features = dataset_to_policy_features(meta.features)
    policy_cfg.output_features = {
        k: v for k, v in features.items() if v.type is FeatureType.ACTION
    }
    policy_cfg.input_features = {
        k: v for k, v in features.items() if v.type is not FeatureType.ACTION
    }
    pre, post = make_eo1_pre_post_processors(policy_cfg, meta.stats)
    pre.save_pretrained(str(dst))
    post.save_pretrained(str(dst))
    cameras = sorted(
        k for k, v in policy_cfg.input_features.items() if v.type is FeatureType.VISUAL
    )
    print(
        f"  -> processor pipeline written for {dataset_root.name}: "
        f"{len(cameras)} cameras {cameras}"
    )

    (dst / "CAMELO_README.md").write_text(
        "# EO-1 base, lerobot 0.6.1 loadable\n\n"
        "Derived from the cached `lerobot/eo1-base` snapshot by\n"
        "`scripts/stage_assets.py:derive_eo1_base` (`make stage RUNG=eo1`).\n"
        "Every file is a symlink back into the pristine HF cache EXCEPT\n"
        "`config.json` and the generated `policy_pre/postprocessor*` files.\n\n"
        f"- `config.json`: dropped {', '.join(EO1_UNSUPPORTED_CONFIG_KEYS)} "
        "(fields a later lerobot added; draccus rejects unknown keys), "
        "repointed `vlm_base`/`pretrained_path` here, set "
        "`use_fast_processor: true` (the PIL backend calls `.numpy()` on the "
        "GPU-resident batch accelerate hands the loop — job 3983182), and set "
        "`chunk_size`/`n_action_steps` to 8.\n"
        f"- processor pipeline: generated against `{dataset_root}` — it names "
        f"that corpus's cameras {cameras}, so this artifact is CORPUS-SPECIFIC.\n\n"
        "The upstream weights are untouched: 836/836 tensors bind with 0 "
        "missing and 0 unexpected keys.\n"
    )


# Vision backbones come from torch.hub, NOT the HF hub — so HF_HUB_OFFLINE
# says nothing about them and a cold torch cache kills the job with a DNS
# error instead of a cache miss. ACT and diffusion both default to
# resnet18/IMAGENET1K_V1; stage it whenever those rungs are in play.
LADDER_BACKBONES = {
    "act": ["ResNet18_Weights.IMAGENET1K_V1"],
    "diffusion": ["ResNet18_Weights.IMAGENET1K_V1"],
}


def stage_backbone(spec: str) -> None:
    """Warm the torch.hub cache for a torchvision weights enum, e.g.
    ``ResNet18_Weights.IMAGENET1K_V1``."""
    import torchvision.models as tvm

    enum_name, member = spec.split(".", 1)
    weights = getattr(getattr(tvm, enum_name), member)
    print(f"staging backbone {spec} ...", flush=True)
    weights.get_state_dict(progress=False)
    print(f"  -> {weights.url}")


def _human(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GiB"


def stage(repo_id: str, repo_type: str, local_dir: Path | None) -> int:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError

    kwargs = {"repo_id": repo_id, "repo_type": repo_type, "max_workers": 8}
    if local_dir is not None:
        kwargs["local_dir"] = str(local_dir / repo_id.split("/")[-1])
    print(f"staging {repo_type} {repo_id} ...", flush=True)
    try:
        path = Path(snapshot_download(**kwargs))
    except GatedRepoError:
        why = MANUAL_GATED.get(repo_id, "gated")
        print(f"  !! GATED ({why}) — accept the terms, then re-run:")
        print(f"     https://huggingface.co/{repo_id}")
        return 1
    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    print(f"  -> {path}  ({_human(size)})")
    if repo_type == "model":
        write_post_stage_stubs(repo_id, path)
        if repo_id == "lerobot/pi0fast-base":
            derive_pi0fast_base()
        if repo_id == "lerobot/eo1-base":
            derive_eo1_base()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=[], help="hub dataset repo ids")
    parser.add_argument("--models", nargs="*", default=[], help="hub model repo ids")
    parser.add_argument(
        "--rung",
        nargs="*",
        default=[],
        choices=sorted(LADDER_MODELS),
        help="stage every base checkpoint AND vision backbone a ladder rung "
        "needs (docs/runbooks/TRAINING.md)",
    )
    parser.add_argument(
        "--backbones",
        nargs="*",
        default=[],
        help="torchvision weights enums, e.g. ResNet18_Weights.IMAGENET1K_V1",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="materialise datasets here as a plain directory too (lerobot-train "
        "wants a --dataset.root, not a cache path)",
    )
    args = parser.parse_args()

    # Staging is the one step that MUST have network. Failing here with a
    # clear message beats a job dying on a compute node 20 minutes later.
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        print("error: HF_HUB_OFFLINE=1 — staging needs network; run me on the login node")
        return 2

    blocked = 0
    for repo in args.models + [m for r in args.rung for m in LADDER_MODELS[r]]:
        blocked += stage(repo, "model", None)
    for repo in args.datasets:
        blocked += stage(repo, "dataset", args.dataset_dir)
    for spec in args.backbones + [b for r in args.rung for b in LADDER_BACKBONES.get(r, [])]:
        stage_backbone(spec)

    if blocked:
        print(f"\n{blocked} repo(s) still gated — jobs using them WILL fail on a compute node.")
        return 3

    if not (args.models or args.datasets or args.rung or args.backbones):
        parser.print_help()
        return 1
    print("\nstaged. Jobs may now run with HF_HUB_OFFLINE=1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
