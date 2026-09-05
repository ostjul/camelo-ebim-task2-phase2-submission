#!/usr/bin/env python3
"""Download action columns for the open-loop replay policy (no videos).

    python scripts/prepare_replay_actions.py
    python scripts/prepare_replay_actions.py --episode 3

Pulls only ``meta/**`` + ``data/**`` from the hardcoded Task 2 fixpos demos
(``hermanprawiro/task2_fixpos_200``), extracts one episode's 20-dim action
column into ``data/replay/…/epNNN_actions.npy`` and the matching world
TCP GT into ``epNNN_gt_traj.npz``, and leaves the rest of the stack alone.
Episode 89 is bundled in git (no hub fetch). Other episodes need
``HF_TOKEN`` / ``huggingface-cli login``. The replay adapter loads that npy
(and the npz when ``--correct-poses``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camelo.policy.adapters.replay import (
    DEFAULT_DATASET_DIR,
    DEFAULT_EPISODE,
    DEFAULT_REPO_ID,
    default_actions_path,
    default_gt_traj_path,
)


def download_tabular(repo_id: str, local_dir: Path) -> Path:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"downloading {repo_id} (meta+data only) → {local_dir}", flush=True)
    try:
        path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
            allow_patterns=["meta/**", "data/**"],
        )
    except GatedRepoError:
        print(
            f"error: {repo_id} is gated — accept the terms, then re-run:\n"
            f"  https://huggingface.co/datasets/{repo_id}",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    return Path(path)


def _load_episode_table(dataset_dir: Path, episode: int, columns: list[str]):
    import pandas as pd

    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise SystemExit(f"missing {info_path} — download incomplete?")
    info = json.loads(info_path.read_text())
    fps = float(info.get("fps", 30))
    files = sorted((dataset_dir / "data").rglob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet under {dataset_dir}/data")
    frames: list = []
    for path in files:
        df = pd.read_parquet(path, columns=columns)
        part = df[df.episode_index == episode]
        if len(part):
            frames.append(part)
    if not frames:
        raise SystemExit(f"episode {episode} not found in {dataset_dir}")
    ep = pd.concat(frames, ignore_index=True).sort_values("frame_index")
    return ep.reset_index(drop=True), fps, info


def extract_episode_actions(dataset_dir: Path, episode: int) -> tuple:
    """Return ((T, 20) float32, fps) for one episode — parquet only."""
    import numpy as np

    ep, fps, info = _load_episode_table(
        dataset_dir, episode, ["episode_index", "frame_index", "action"]
    )
    action_feat = info.get("features", {}).get("action", {})
    shape = action_feat.get("shape")
    if shape != [20]:
        print(f"warning: expected action shape [20], got {shape}", flush=True)
    actions = np.stack(
        [np.asarray(row, dtype=np.float32) for row in ep["action"].to_numpy()]
    )
    if actions.ndim != 2 or actions.shape[1] != 20:
        raise SystemExit(f"bad action array shape {actions.shape}")
    return actions, fps


def extract_episode_gt(dataset_dir: Path, episode: int) -> dict:
    """FK world TCP from measured state for one episode."""
    import numpy as np

    from camelo import contracts as C
    from camelo.control.gt_traj import compute_gt_traj

    ep, fps, _info = _load_episode_table(
        dataset_dir,
        episode,
        ["episode_index", "frame_index", "observation.state"],
    )
    state = np.stack(
        [np.asarray(row, dtype=np.float64) for row in ep["observation.state"]]
    )
    if state.shape[1] != C.STATE_DIM:
        raise SystemExit(f"expected state dim {C.STATE_DIM}, got {state.shape}")
    t = ep["frame_index"].to_numpy(dtype=np.float64) / fps
    return compute_gt_traj(
        t=t,
        base_xy_yaw=state[:, C.S_BASE_ODOM],
        spine=state[:, C.S_SPINE],
        arm_q=state[:, C.S_RIGHT_ARM],
        recorded_link8=state[:, C.S_RIGHT_EE],
        episode=episode,
        fps=fps,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--episode", type=int, default=DEFAULT_EPISODE)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="where to materialise meta+data (default: outputs/datasets/…)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="override the actions npy path (default: data/replay/…/epNNN_actions.npy)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="reuse an existing --dataset-dir (no hub fetch)",
    )
    args = parser.parse_args()

    if not args.skip_download:
        download_tabular(args.repo_id, args.dataset_dir)
    elif not (args.dataset_dir / "meta" / "info.json").is_file():
        raise SystemExit(f"--skip-download set but {args.dataset_dir} has no meta/info.json")

    actions, fps = extract_episode_actions(args.dataset_dir, args.episode)
    out = args.out or default_actions_path(args.episode)
    out.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np

    np.save(out, actions)
    from camelo.control.gt_traj import write_gt_traj

    gt_arrays = extract_episode_gt(args.dataset_dir, args.episode)
    gt_path = default_gt_traj_path(args.episode)
    if args.out is not None:
        gt_path = Path(str(out).replace("_actions.npy", "_gt_traj.npz"))
    write_gt_traj(gt_path, gt_arrays)
    meta = {
        "repo_id": args.repo_id,
        "episode": int(args.episode),
        "fps": fps,
        "frames": int(actions.shape[0]),
        "action_dim": int(actions.shape[1]),
        "dataset_dir": str(args.dataset_dir),
        "actions_path": str(out),
        "gt_traj_path": str(gt_path),
    }
    meta_path = out.with_name(out.name.replace("_actions.npy", "_meta.json"))
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"wrote {out}  ({actions.shape[0]} frames @ {fps:g} fps, "
        f"{actions.shape[0] / fps:.1f} sim-s)  meta={meta_path}",
        flush=True,
    )
    print(f"wrote {gt_path}  T={len(gt_arrays['t'])}  TCP is the IK target", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
