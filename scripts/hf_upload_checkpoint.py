#!/usr/bin/env python3
"""Publish a lerobot checkpoint to a Hugging Face model repo with our card.

    python scripts/hf_upload_checkpoint.py --checkpoint /ckpt/act/checkpoint-100000 \\
        [--repo-id ostjul/camelo-ebim-task2-act-s27a15] [--private] [--dry-run]

Validates the ``pretrained_model`` layout first (config.json,
model.safetensors, train_config.json; the pre/post-processor JSONs are
warned about when absent, and every ``state_file`` they reference — the
normaliser statistics ``*_processor.safetensors`` — must be present),
renders ``submission/rig/MODEL_CARD.md`` and uploads every regular file of
the directory (lerobot's own layout; hidden and ``._*`` files skipped) plus
any ``--extra`` file at the repo root. The checkpoint directory is never
written to (it is a read-only mount on the serving box). Needs a token:
``hf auth login`` or ``HF_TOKEN``; ``--dry-run`` needs neither.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_REPO_ID = "ostjul/camelo-ebim-task2-act-s27a15"
SOURCE_URL = "https://github.com/ostjul/camelo-ebim-task2-phase2-submission"
REQUIRED = ("config.json", "model.safetensors", "train_config.json")
OPTIONAL = ("policy_preprocessor.json", "policy_postprocessor.json")
ROOT = Path(__file__).resolve().parent.parent
CARD_TEMPLATE = ROOT / "submission" / "rig" / "MODEL_CARD.md"


def resolve_model_dir(checkpoint: Path) -> Path:
    """Accept the pretrained_model dir, its parent (checkpoints/NNN), or a run dir."""
    for cand in (checkpoint, checkpoint / "pretrained_model"):
        if (cand / "config.json").is_file():
            return cand
    raise SystemExit(f"{checkpoint}: no config.json here or in pretrained_model/")


def validate(model_dir: Path) -> dict:
    missing = [name for name in REQUIRED if not (model_dir / name).is_file()]
    if missing:
        raise SystemExit(f"{model_dir}: missing {missing} — not a complete lerobot checkpoint")
    config = json.loads((model_dir / "config.json").read_text())
    if config.get("type") != "act":
        raise SystemExit(f"{model_dir}: config.json type is {config.get('type')!r}, expected 'act'")
    for name in OPTIONAL:
        if not (model_dir / name).is_file():
            print(f"warning: {name} absent (older lerobot checkpoints do not have it)")
            continue
        # The processor pipelines load their statistics from sidecar files;
        # a repo without them builds a policy with no normaliser (F-81 family).
        for step in json.loads((model_dir / name).read_text()).get("steps", []):
            state_file = step.get("state_file")
            if state_file and not (model_dir / state_file).is_file():
                raise SystemExit(f"{model_dir}: {name} references missing {state_file}")
    return config


def describe(config: dict) -> list[str]:
    lines = []
    for key in ("input_features", "output_features"):
        feats = config.get(key) or {}
        for name, spec in feats.items():
            shape = spec.get("shape") if isinstance(spec, dict) else spec
            lines.append(f"  {key}: {name} {shape}")
    for key in ("chunk_size", "n_action_steps"):
        if key in config:
            lines.append(f"  {key}: {config[key]}")
    return lines


def files_to_upload(model_dir: Path) -> list[str]:
    """Every regular file of the checkpoint dir (lerobot's layout), no hidden/AppleDouble."""
    return sorted(
        p.name
        for p in model_dir.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.name != "README.md"
    )


def render_card(model_dir: Path, repo_id: str, files: list[str]) -> str:
    size_mb = sum((model_dir / f).stat().st_size for f in files) / 1e6
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip() or "unknown"
    except OSError:
        sha = "unknown"
    listing = "\n".join(
        f"- `{f}` ({(model_dir / f).stat().st_size / 1e6:.1f} MB)" for f in files
    )
    text = CARD_TEMPLATE.read_text()
    for key, value in (
        ("@REPO_ID@", repo_id),
        ("@SOURCE_URL@", SOURCE_URL),
        ("@FILES@", listing),
        ("@SIZE_MB@", f"{size_mb:.0f}"),
        ("@DATE@", _dt.date.today().isoformat()),
        ("@GIT_SHA@", sha),
    ):
        text = text.replace(key, value)
    keys = ("REPO_ID", "SOURCE_URL", "FILES", "SIZE_MB", "DATE", "GIT_SHA")
    left = [k for k in keys if f"@{k}@" in text]
    if left:
        raise SystemExit(f"model card has unrendered placeholders: {left}")
    return text


def upload(
    model_dir: Path,
    repo_id: str,
    private: bool,
    files: list[str],
    card: str,
    api=None,
    extra: tuple[Path, ...] = (),
):
    if api is None:
        from huggingface_hub import HfApi, get_token

        # HfApi() resolves the cached login lazily; ask the resolver, not the object.
        token = os.environ.get("HF_TOKEN") or get_token()
        if not token:
            raise SystemExit("no Hugging Face token: run `hf auth login` or set HF_TOKEN")
        api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=list(files),
        commit_message="Upload ACT s27a15 checkpoint (camelo)",
    )
    with tempfile.TemporaryDirectory() as tmp:
        card_path = Path(tmp) / "README.md"
        card_path.write_text(card)
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Model card",
        )
    for path in extra:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add {path.name}",
        )
    return f"https://huggingface.co/{repo_id}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument(
        "--private", action="store_true", help="create the repo private (default: public)"
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="validate + render the card, upload nothing"
    )
    ap.add_argument(
        "--extra", type=Path, action="append", default=[],
        help="extra file to upload at the repo root (e.g. the offline probe report)",
    )
    args = ap.parse_args(argv)

    model_dir = resolve_model_dir(args.checkpoint.expanduser())
    config = validate(model_dir)
    files = files_to_upload(model_dir)
    extra = tuple(p.expanduser() for p in args.extra)
    for p in extra:
        if not p.is_file():
            raise SystemExit(f"--extra {p}: not a file")
    card = render_card(model_dir, args.repo_id, files)
    print(f"checkpoint: {model_dir}")
    print("\n".join(describe(config)))
    print(f"files: {files}")
    if extra:
        print(f"extra: {[str(p) for p in extra]}")
    print(f"repo: {args.repo_id} ({'private' if args.private else 'PUBLIC'})")
    if args.dry_run:
        print("--- model card ---")
        print(card)
        print("dry-run: nothing uploaded")
        return 0
    url = upload(model_dir, args.repo_id, args.private, files, card, extra=extra)
    print(f"uploaded: {url}")
    print(
        "serve: python scripts/serve_policy.py --adapter lerobot "
        f"--checkpoint {args.repo_id} --action-layout s27a15 --state-layout s27a15 "
        "--port 8767 --seed 0"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
