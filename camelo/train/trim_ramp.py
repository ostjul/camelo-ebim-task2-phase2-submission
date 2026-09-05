"""Drop each episode's spine-ramp frames — per episode, never a fixed count.

    python -m camelo.train.trim_ramp \
        --source outputs/datasets/ext_fixpos200_train_v1 \
        --output outputs/datasets/ext_fixpos200_trim_v1

Recordings start with the spine at 0.000 and ramp to their plateau over
~5-7 s. Eval never reproduces that — the scene pins the spine at launch
and the rollout begins after it settles — so those leading frames show arm
heights the policy cannot meet at inference. Worse, the 16-dim model
proprio has no spine channel at all, so during the ramp the policy sees
arms moving for a reason it has no state to explain.

The cut is measured per episode against that episode's OWN plateau
(`scripts/analyze_ramp_trim.py` is the read-only report and the same
logic). This corpus deliberately spans plateau heights 0.45-0.55 as
augmentation, so "reaches 0.50" would be the wrong test, and a single
frame count would over-trim the fast episodes and under-trim the slow.

**No video re-encode.** LeRobot v3 keeps each episode's video as a
timestamp range into a shared mp4, so trimming leading frames is a
metadata edit: advance `videos/<key>/from_timestamp`, drop the parquet
rows, re-index. That matters here because tiger3's ffmpeg cannot start at
all (missing libfftw3), so a re-encode is not available even as a
fallback.

Alignment between the dropped rows and the advanced video offset is the
one thing that can silently go wrong, so `--verify` decodes frames from
both datasets and compares them pixel-wise.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from camelo import contracts as C  # noqa: E402

SETTLE_TOL_M = 0.005
SETTLE_HOLD = 15
MIN_MARGIN_S = 3.0


def settle_frame(spine: np.ndarray) -> int:
    """First frame from which the spine holds its own plateau."""
    plateau = float(np.median(spine[len(spine) // 2 :]))
    close = np.abs(spine - plateau) < SETTLE_TOL_M
    settled, run = len(spine), 0
    for i in range(len(spine) - 1, -1, -1):
        if close[i]:
            run += 1
            if run >= SETTLE_HOLD:
                settled = i
        else:
            run = 0
            if settled < len(spine):
                break
    return settled


def plan_cuts(frame, fps: float) -> dict[int, int]:
    """episode -> frames to drop, refusing any cut that crowds the grasp."""
    cuts, tight = {}, []
    for episode, group in frame.groupby("episode_index"):
        state = np.stack(group["observation.state"].to_numpy())
        action = np.stack(group["action"].to_numpy())
        cut = settle_frame(state[:, C.S_SPINE])
        grip = action[:, C.A_RIGHT_GRIP]
        moved = np.where(np.abs(grip - grip[0]) > 0.05)[0]
        if len(moved) and (moved[0] - cut) / fps < MIN_MARGIN_S:
            tight.append(int(episode))
        cuts[int(episode)] = int(cut)
    if tight:
        raise RuntimeError(
            f"refusing to trim: episodes {tight[:10]} would be cut within "
            f"{MIN_MARGIN_S}s of their first gripper motion (F-84)"
        )
    return cuts


def trim(source: Path, output: Path) -> dict[int, int]:
    import pandas as pd

    if not (source / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"not a LeRobot dataset: {source}")
    if output.exists():
        raise FileExistsError(f"output already exists: {output} (trims are not resumable)")

    info = json.loads((source / "meta" / "info.json").read_text())
    fps = float(info["fps"])

    data_files = sorted((source / "data").glob("chunk-*/*.parquet"))
    frame = pd.concat([pd.read_parquet(f) for f in data_files], ignore_index=True)
    cuts = plan_cuts(frame, fps)

    kept = []
    for episode, group in frame.groupby("episode_index"):
        kept.append(group.iloc[cuts[int(episode)] :])
    out_frame = pd.concat(kept, ignore_index=True)

    # Re-index: frame_index/timestamp restart per episode, index is global.
    out_frame["frame_index"] = out_frame.groupby("episode_index").cumcount()
    out_frame["timestamp"] = out_frame["frame_index"] / fps
    out_frame["index"] = np.arange(len(out_frame), dtype=np.int64)

    shutil.copytree(source, output, symlinks=True)
    for stale in (output / "data").glob("chunk-*/*.parquet"):
        stale.unlink()
    target = output / "data" / "chunk-000" / "file-000.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    out_frame.to_parquet(target, index=False)

    # Episode metadata: lengths, global row ranges, and the video offsets —
    # advancing from_timestamp by exactly the dropped duration is what keeps
    # frames aligned with rows without touching a single video file.
    meta_files = sorted((output / "meta" / "episodes").glob("chunk-*/*.parquet"))
    episodes = pd.concat([pd.read_parquet(f) for f in meta_files], ignore_index=True)
    lengths = out_frame.groupby("episode_index").size()
    cursor = 0
    for row in episodes.index:
        episode = int(episodes.at[row, "episode_index"])
        cut = cuts.get(episode, 0)
        length = int(lengths.get(episode, 0))
        episodes.at[row, "length"] = length
        episodes.at[row, "dataset_from_index"] = cursor
        episodes.at[row, "dataset_to_index"] = cursor + length
        cursor += length
        for column in episodes.columns:
            if column.endswith("/from_timestamp"):
                episodes.at[row, column] = float(episodes.at[row, column]) + cut / fps
    for stale in meta_files:
        stale.unlink()
    first = output / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    first.parent.mkdir(parents=True, exist_ok=True)
    episodes.to_parquet(first, index=False)
    episodes_meta = [c for c in episodes.columns if c.startswith("meta/episodes/")]
    if episodes_meta:
        episodes[episodes_meta] = 0
        episodes.to_parquet(first, index=False)

    info["total_frames"] = int(len(out_frame))
    (output / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    from lerobot.datasets.dataset_tools import recompute_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    recompute_stats(LeRobotDataset(repo_id=output.name, root=output))

    from camelo.train.dataset_stats import repair_degenerate_quantiles, verify

    repair_degenerate_quantiles(output)
    verify(output)

    dropped = sum(cuts.values())
    (output / "camelo_provenance.json").write_text(
        json.dumps(
            {
                "source": str(source.resolve()),
                "transform": "dropped each episode's spine-ramp frames (per-episode settle "
                "detection); video files untouched, from_timestamp advanced instead",
                "frames_dropped": dropped,
                "cuts": cuts,
            },
            indent=2,
        )
    )
    print(f"trimmed {dropped} frames ({dropped / len(frame) * 100:.1f} %) -> {output}")
    return cuts


def verify(source: Path, output: Path, cuts: dict[int, int], episodes: int = 3) -> int:
    """Decode from both datasets and compare — the alignment check."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    before = LeRobotDataset(repo_id=source.name, root=source)
    after = LeRobotDataset(repo_id=output.name, root=output)
    camera = f"observation.images.{C.CAMERA_KEYS[0]}"
    bad = 0
    for episode in sorted(cuts)[:episodes]:
        cut = cuts[episode]
        b_from = int(before.meta.episodes[episode]["dataset_from_index"])
        a_from = int(after.meta.episodes[episode]["dataset_from_index"])
        b, a = before[b_from + cut], after[a_from]
        state_ok = np.allclose(b["observation.state"].numpy(), a["observation.state"].numpy())
        bi = b[camera].numpy().astype(np.float32)
        ai = a[camera].numpy().astype(np.float32)
        pix = float(np.abs(bi - ai).max())
        ok = state_ok and pix < 1e-3
        print(f"  ep {episode:>3} cut {cut:>3}: state match {state_ok}, max pixel diff {pix:.5f} "
              f"-> {'OK' if ok else 'MISALIGNED'}")
        bad += 0 if ok else 1
    if bad:
        print(f"FAIL: {bad} episode(s) misaligned — video offset does not match dropped rows")
        return 1
    print("ok: trimmed frames line up with the source, video and state alike")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    cuts = trim(args.source, args.output)
    return verify(args.source, args.output, cuts) if args.verify else 0


if __name__ == "__main__":
    sys.exit(main())
