"""A5: subsample each episode's frame rate by a fixed factor.

    python -m camelo.train.rate_subsample \
        --source outputs/datasets/ext_fixpos200_model16_v1 \
        --output outputs/datasets/ext_fixpos200_model16_7hz5_v1 \
        --factor 4

W5's research question is whether training a policy at the rate the rig
actually delivers (the Spark's measured render floor, P3) makes the
resulting staleness in-distribution rather than a deployment-time surprise.
That means the training corpus itself has to be relabeled at the target
rate, not just subsampled at inference — this is the builder for it.

**Keep frame 0, K, 2K, ... of each episode** (`--factor K`), relabel the
kept frame at index i as `frame_index = i`, `timestamp = i / new_fps`, and
set `meta/info.json`'s `fps` to `source_fps / K` (7.5 Hz for K=4, 2.0 Hz
for K=15 against a 30 fps source). A partial trailing run of fewer than K
frames at an episode's end is simply not a multiple of K and is dropped,
never padded.

**No video re-encode — the same trick `trim_ramp.py` uses, one level up.**
LeRobot decodes a frame by adding the row's own `timestamp` to the
episode's `videos/<key>/from_timestamp` and seeking that instant in the
untouched shared mp4 (`dataset_reader._query_videos`). Relabeling the kept
frame at index i as `i / new_fps` lands EXACTLY on `K*i / source_fps` — the
original video timestamp of the frame that was kept — because
`new_fps = source_fps / K` makes `i / new_fps == K*i / source_fps`
identically (e.g. `i / 7.5 == 4*i / 30`). So the video files, and every
`from_timestamp`/`to_timestamp` in `meta/episodes`, are copied through
byte-for-byte: only which rows exist and what `frame_index`/`timestamp`
they carry changes. This is not a stylistic echo of `trim_ramp` — tiger3's
ffmpeg cannot even start here (missing `libfftw3.so.3`, the same finding
`trim_ramp.py`'s docstring notes), so the `LeRobotDataset.create` /
`add_frame` / `save_episode` encode path is not available regardless of
preference.

Keeps every episode, every task, every feature — this builder only
changes which frames exist and their fps/frame_index/timestamp labels.

**A5 is rate ONLY.** The idle-frame filter that the A5 slot in
`docs/research/GRASP_EXPERIMENT_PLAN.md` also names is deliberately NOT
applied here: W5 asks whether matching the deployment rate alone fixes
staleness, and folding an idle-frame filter into the same corpus would
confound that answer with a second, uncontrolled variable.

`--verify` decodes a few frames from both datasets per episode and
compares them pixel-wise and state-wise — the alignment check that the
`i / new_fps == K*i / source_fps` identity above actually lands on the
frame it claims to, the same role `trim_ramp.verify` plays for the
contiguous-prefix case.
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


def subsample_frame(frame, factor: int, new_fps: float):
    """Pure pandas: keep every `factor`-th row of each episode (by
    `frame_index` order) and relabel `frame_index`/`timestamp`/`index`.

    No lerobot and no disk I/O — this is the part a test can exercise
    directly against a hand-built `DataFrame`. Kept rows are frame_index
    0, K, 2K, ... of each episode; a trailing partial run of fewer than K
    frames is simply not a multiple of K and is dropped, never padded.
    """
    if factor < 1:
        raise ValueError(f"--factor must be a positive integer, got {factor}")
    sorted_frame = frame.sort_values(["episode_index", "frame_index"])
    rank = sorted_frame.groupby("episode_index").cumcount()
    out = sorted_frame[rank % factor == 0].reset_index(drop=True)
    # timestamp = i / new_fps lands on the ORIGINAL video's K*i / source_fps
    # (module docstring) — that identity is what makes skipping the
    # re-encode correct, not just convenient.
    out["frame_index"] = out.groupby("episode_index").cumcount()
    out["timestamp"] = out["frame_index"] / new_fps
    out["index"] = np.arange(len(out), dtype=np.int64)
    return out


def provenance_payload(
    source: Path,
    factor: int,
    source_fps: float,
    new_fps: float,
    frames_in: int,
    frames_out: int,
    source_provenance: dict | None,
) -> dict:
    """What lands in the output's `camelo_provenance.json`.

    `source_provenance` nests the source's own `camelo_provenance.json`
    forward when it had one — every builder in this package overwrites
    the file wholesale, so nesting is the only way an earlier build's
    lineage (e.g. `convert_model_state`) survives this one, the same
    convention `obs_mask.provenance_payload` uses.
    """
    return {
        "source": str(Path(source).resolve()),
        "transform": "camelo.train.rate_subsample",
        "factor": factor,
        "source_fps": source_fps,
        "fps": new_fps,
        "frames_in": frames_in,
        "frames_out": frames_out,
        "note": "keeps frame 0, K, 2K, ... of every episode (K = factor) and relabels "
        "frame_index/timestamp at new_fps = source_fps / factor; video files and "
        "from_timestamp/to_timestamp are untouched (no re-encode — see module "
        "docstring). This is A5, RATE ONLY: the idle-frame filter that the A5 slot "
        "in docs/research/GRASP_EXPERIMENT_PLAN.md also names is deliberately not "
        "applied here.",
        "source_provenance": source_provenance,
    }


def subsample(source: Path, output: Path, factor: int) -> dict:
    import pandas as pd

    if factor < 1:
        raise ValueError(f"--factor must be a positive integer, got {factor}")
    if not (source / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"not a LeRobot dataset: {source}")
    if output.exists():
        raise FileExistsError(
            f"output already exists: {output} (rate-subsample runs are not resumable)"
        )

    info = json.loads((source / "meta" / "info.json").read_text())
    source_fps = float(info["fps"])
    new_fps = source_fps / factor

    data_files = sorted((source / "data").glob("chunk-*/*.parquet"))
    frame = pd.concat([pd.read_parquet(f) for f in data_files], ignore_index=True)
    frames_in = len(frame)

    out_frame = subsample_frame(frame, factor, new_fps)

    shutil.copytree(source, output, symlinks=True)
    for stale in (output / "data").glob("chunk-*/*.parquet"):
        stale.unlink()
    target = output / "data" / "chunk-000" / "file-000.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    out_frame.to_parquet(target, index=False)

    # Episode metadata: only lengths and global row ranges shrink. The
    # video from_timestamp/to_timestamp columns are NOT touched — the
    # video files are untouched too, and from_timestamp already marks
    # where each episode's frame 0 sits in the shared mp4.
    meta_files = sorted((output / "meta" / "episodes").glob("chunk-*/*.parquet"))
    episodes = pd.concat([pd.read_parquet(f) for f in meta_files], ignore_index=True)
    lengths = out_frame.groupby("episode_index").size()
    cursor = 0
    for row in episodes.index:
        episode = int(episodes.at[row, "episode_index"])
        length = int(lengths.get(episode, 0))
        episodes.at[row, "length"] = length
        episodes.at[row, "dataset_from_index"] = cursor
        episodes.at[row, "dataset_to_index"] = cursor + length
        cursor += length
    for stale in meta_files:
        stale.unlink()
    first = output / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    first.parent.mkdir(parents=True, exist_ok=True)
    episodes.to_parquet(first, index=False)
    episodes_meta = [c for c in episodes.columns if c.startswith("meta/episodes/")]
    if episodes_meta:
        episodes[episodes_meta] = 0
        episodes.to_parquet(first, index=False)

    info["fps"] = new_fps
    info["total_frames"] = int(len(out_frame))
    (output / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    from lerobot.datasets.dataset_tools import recompute_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    recompute_stats(LeRobotDataset(repo_id=output.name, root=output))

    from camelo.train.dataset_stats import repair_degenerate_quantiles, verify as verify_stats

    repair_degenerate_quantiles(output)
    verify_stats(output)

    source_provenance_path = source / "camelo_provenance.json"
    source_provenance = (
        json.loads(source_provenance_path.read_text())
        if source_provenance_path.is_file()
        else None
    )
    frames_out = int(len(out_frame))
    provenance = provenance_payload(
        source, factor, source_fps, new_fps, frames_in, frames_out, source_provenance
    )
    (output / "camelo_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        f"kept {frames_out}/{frames_in} frames ({frames_out / frames_in * 100:.1f} %) "
        f"at {new_fps} fps -> {output}"
    )
    return {
        "frames_in": frames_in,
        "frames_out": frames_out,
        "source_fps": source_fps,
        "fps": new_fps,
    }


def verify(source: Path, output: Path, factor: int, episodes: int = 3) -> int:
    """Decode from both datasets and compare — the alignment check.

    Confirms the `i / new_fps == K*i / source_fps` identity the module
    docstring relies on actually retrieves the frame it claims to, the
    same role `trim_ramp.verify` plays for the contiguous-prefix case.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    before = LeRobotDataset(repo_id=source.name, root=source)
    after = LeRobotDataset(repo_id=output.name, root=output)
    camera = f"observation.images.{C.CAMERA_KEYS[0]}"
    bad, checked = 0, 0
    for episode in range(min(episodes, after.meta.total_episodes)):
        b_from = int(before.meta.episodes[episode]["dataset_from_index"])
        a_from = int(after.meta.episodes[episode]["dataset_from_index"])
        length = int(after.meta.episodes[episode]["length"])
        if length == 0:
            continue
        for i in sorted({0, length // 2, length - 1}):
            b, a = before[b_from + i * factor], after[a_from + i]
            state_ok = np.allclose(b["observation.state"].numpy(), a["observation.state"].numpy())
            bi = b[camera].numpy().astype(np.float32)
            ai = a[camera].numpy().astype(np.float32)
            pix = float(np.abs(bi - ai).max())
            ok = state_ok and pix < 1e-3
            print(
                f"  ep {episode:>3} kept-frame {i:>4} (orig {i * factor:>4}): "
                f"state match {state_ok}, max pixel diff {pix:.5f} -> "
                f"{'OK' if ok else 'MISALIGNED'}"
            )
            checked += 1
            bad += 0 if ok else 1
    if bad:
        print(f"FAIL: {bad}/{checked} sampled frame(s) misaligned")
        return 1
    print(f"ok: {checked} sampled frames line up with the source, video and state alike")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factor", type=int, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    subsample(args.source, args.output, args.factor)
    return verify(args.source, args.output, args.factor) if args.verify else 0


if __name__ == "__main__":
    sys.exit(main())
