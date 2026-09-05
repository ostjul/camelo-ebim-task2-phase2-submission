"""Prepare and convert an externally-produced v2.1 LeRobot dataset to v3.0.

    python -m camelo.train.convert_v21_to_v30 --root outputs/datasets/ext_.../some_v21_dataset

Written against `task2_munich` (MCAP -> LeRobot v2.1, "labs-mcap-converter",
see the dataset's own README). Its v2.1 export doesn't meet the assumptions
lerobot's own `convert_dataset_v21_to_v30.py` makes about a v2.1 dataset, so
running that script directly on data from this pipeline fails partway
through, or worse, "succeeds" and hands back an unloadable dataset. Every
failure below was found by actually running the full chain end to end
(convert -> load -> index a frame -> recompute_stats), not by reading the
converter's source, and each one is a plain crash with no ambiguity once you
hit it -- this script exists to hit it once, here, instead of per dataset.

Five fixes, applied to the v2.1 root before handing it to lerobot's own
converter:

1. **`meta/episodes_stats.jsonl` had one line for N episodes** (episode 0
   only, empty stats). The stock converter indexes into it positionally
   (`episodes_stats_vals[i] for i in range(num_episodes)`) and throws
   `IndexError` on episode 1. We regenerate it with real per-episode stats
   for every numeric feature via lerobot's own `compute_episode_stats` --
   though this mostly doesn't matter for state/action, since
   `recompute_stats()` (called in postprocessing below, same as
   `trim_ramp`/`rate_subsample`/`convert_model_state`/`obs_mask`) recomputes
   those from scratch anyway. Video features get shape-correct placeholder
   stats: `recompute_stats` has an explicit `# TODO: enable image and video
   stats re-computation` and never touches them, and `dataset_stats.verify()`
   only ever checks `observation.state`/`action` -- a real per-frame video
   stat has no consumer downstream regardless.

2. **Some `shape: [1]` state columns are stored as length-1 arrays.**
   lerobot's `get_hf_features_from_features` maps `shape == (1,)` to a bare
   scalar `Value` (a special case distinct from the `Sequence` branch used
   for shape (2,)+), but if the source pipeline wrote them as length-1
   arrays instead, the conversion step doesn't care -- it just pandas-concats
   columns through -- but the FIRST `LeRobotDataset(...)` on the "converted"
   result dies with `TypeError: Couldn't cast array of type list<element:
   float> to float`. We unwrap every array-valued `shape==[1]` column here.

3. **`index` (the running frame index across the whole dataset, distinct
   from per-episode `frame_index`) doesn't exist.** Real LeRobot recordings
   always carry it; `DatasetReader.get_item` reads it unconditionally
   (`item["index"].item()`), so every single-frame access throws
   `AttributeError: 'NoneType' object has no attribute 'item'` the moment
   you index into the "converted" dataset. We add it as a running count in
   the same file order the converter concatenates in, so after conversion
   it comes out 0..total_frames-1 contiguous.

4. **`task_index` doesn't exist either**, same failure mode as `index`. We
   derive it per episode from `meta/tasks.jsonl` + `meta/episodes.jsonl`.

5. **`episode_index` has gaps.** The source pipeline drops episodes that
   fail its own sync check (see the dataset's `info.json` ->
   `conversion_failures`) but never renumbers the survivors -- data/video
   files go straight from episode 3 to 5, skipping 4, and more such gaps.
   The stock converter reassigns data/video files dense 0..N-1 indices
   purely by sorted-glob position, but reads `meta/episodes.jsonl` (and our
   regenerated episodes_stats.jsonl) keyed by the OLD sparse episode_index,
   so its own internal consistency check trips: `ValueError: Number of
   episodes is not the same ({4, 5})`. This one only showed up at full
   (238-episode) scale on `task2_munich` -- a 2-episode smoke test with no
   gap can't catch it. We renumber `episode_index` to dense 0..N-1 by sorted
   file position (matching exactly how the converter itself renumbers
   data/video files), in the parquet AND in `meta/episodes.jsonl`.

**Not fixed here, and not fixable by a conversion script: per-camera frame
rate.** On `task2_munich` the three cameras run at genuinely different real
capture rates (head ~10 Hz, wrists ~30 Hz) against the state's 20 Hz grid,
and `info.json`'s declared `video.fps` for the head camera (22) does not
match its real rate (~10, confirmed by decoding). `LeRobotDataset`'s default
`tolerance_s=1e-4` (0.1 ms) is far tighter than any of these cameras'
real frame spacing, so under default settings ~97% of randomly sampled
frames raise `FrameTimestampError`. `tolerance_s=0.15` cleared every sampled
frame in testing. None of this repo's other dataset tools (`merge_datasets`,
`convert_model_state`, `trim_ramp`, `rate_subsample`) override the default,
so **anything that loads a dataset built this way needs
`LeRobotDataset(..., tolerance_s=...)` set explicitly** -- this is a
training-time call about acceptable image staleness, not something a
conversion script should silently paper over.

This script runs `convert_dataset_v21_to_v30.convert_dataset(...)` with
`--video-file-size-in-mb=1`, forcing one video file per episode per camera.
Without it, the stock converter's default multi-episode batching tries to
ffmpeg-concat episodes with different per-episode time_base (AV1, one
episode's average_rate is e.g. 981082/98157, the next 954043/95392) and
crashes muxing (`av.error.ValueError: Invalid argument`). `=1` is proven
safe here (no two consecutive same-camera episodes on `task2_munich` sum
under 1 MB), not just a low-risk guess -- verified against every camera's
real file sizes before use, not assumed from the number alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def stack_column(df, key: str) -> np.ndarray:
    return np.stack([np.atleast_1d(np.asarray(v)) for v in df[key].to_numpy()])


def unwrap_scalar_columns(df, features: dict) -> tuple:
    """Array-valued shape==[1] columns -> bare scalars (lerobot's own convention)."""
    fixed = []
    for key, ft in features.items():
        if key not in df.columns or tuple(ft["shape"]) != (1,) or ft["dtype"] == "string":
            continue
        sample = df[key].iloc[0]
        if hasattr(sample, "__len__"):
            df[key] = df[key].apply(lambda v: np.asarray(v).reshape(-1)[0])
            fixed.append(key)
    return df, fixed


def _video_placeholder_stats(channels: int, count: int) -> dict:
    shape = (channels, 1, 1)
    return {
        "min": np.zeros(shape, dtype=np.float32),
        "max": np.ones(shape, dtype=np.float32),
        "mean": np.full(shape, 0.5, dtype=np.float32),
        "std": np.full(shape, 0.25, dtype=np.float32),
        "count": np.array([count]),
        "q01": np.zeros(shape, dtype=np.float32),
        "q10": np.full(shape, 0.1, dtype=np.float32),
        "q50": np.full(shape, 0.5, dtype=np.float32),
        "q90": np.full(shape, 0.9, dtype=np.float32),
        "q99": np.ones(shape, dtype=np.float32),
    }


def prepare(root: Path) -> None:
    """Mutate a v2.1 dataset in place so lerobot's own converter can process it.

    Safe to re-run: already-fixed columns and an already-dense episode_index
    are detected and left alone.
    """
    import pandas as pd
    from lerobot.datasets.compute_stats import compute_episode_stats

    info = json.loads((root / "meta" / "info.json").read_text())
    features = info["features"]
    video_keys = [k for k, v in features.items() if v["dtype"] in ("video", "image")]

    task_to_index = {}
    for line in (root / "meta" / "tasks.jsonl").read_text().splitlines():
        rec = json.loads(line)
        task_to_index[rec["task"]] = rec["task_index"]
    episode_tasks = {}
    for line in (root / "meta" / "episodes.jsonl").read_text().splitlines():
        rec = json.loads(line)
        episode_tasks[rec["episode_index"]] = rec["tasks"][0]

    ep_paths = sorted((root / "data").glob("*/*.parquet"))
    if not ep_paths:
        raise FileNotFoundError(f"no per-episode parquet files under {root / 'data'}")

    known_bookkeeping = {"index", "task_index"}
    all_fixed = set()
    out_lines = []
    running_index = 0
    old_to_new = {}
    for new_ep_idx, ep_path in enumerate(ep_paths):
        df = pd.read_parquet(ep_path)
        df, fixed = unwrap_scalar_columns(df, features)

        non_video_keys = {k for k in features if k not in video_keys}
        missing = non_video_keys - set(df.columns) - known_bookkeeping
        if missing:
            raise ValueError(
                f"{ep_path}: unexpected missing columns {sorted(missing)}"
                " -- investigate, don't guess"
            )

        if "task_index" not in df.columns:
            old_ep_idx_probe = int(df["episode_index"].iloc[0])
            task_idx = task_to_index[episode_tasks[old_ep_idx_probe]]
            df["task_index"] = np.full(len(df), task_idx, dtype=np.int64)
            fixed = fixed + ["task_index (added)"]

        old_ep_idx = int(df["episode_index"].iloc[0])
        old_to_new[old_ep_idx] = new_ep_idx
        if old_ep_idx != new_ep_idx:
            df["episode_index"] = new_ep_idx
            fixed = fixed + [f"episode_index ({old_ep_idx}->{new_ep_idx})"]

        if "index" not in df.columns:
            df["index"] = np.arange(running_index, running_index + len(df), dtype=np.int64)
            fixed = fixed + ["index (added)"]
        running_index += len(df)

        if fixed:
            all_fixed.update(fixed)
            df.to_parquet(ep_path, index=False)

        ep_idx = int(df["episode_index"].iloc[0])
        numeric_keys = [k for k in features if k not in video_keys and k in df.columns]
        episode_data = {k: stack_column(df, k) for k in numeric_keys}
        ep_stats = compute_episode_stats(episode_data, features)
        for key in video_keys:
            ep_stats[key] = _video_placeholder_stats(features[key]["shape"][-1], len(df))

        jsonable_stats = {
            k: {sk: np.asarray(sv).tolist() for sk, sv in v.items()} for k, v in ep_stats.items()
        }
        out_lines.append(json.dumps({"episode_index": ep_idx, "stats": jsonable_stats}))

    out_path = root / "meta" / "episodes_stats.jsonl"
    out_path.write_text("\n".join(out_lines) + "\n")
    print(f"prepare: wrote {len(out_lines)} per-episode stats entries to {out_path}")
    non_renumber_fixes = sorted(x for x in all_fixed if not x.startswith("episode_index ("))
    print(f"prepare: columns/values fixed: {non_renumber_fixes}")

    episodes_path = root / "meta" / "episodes.jsonl"
    ep_lines = episodes_path.read_text().splitlines()
    renumbered = 0
    new_ep_lines = []
    for line in ep_lines:
        rec = json.loads(line)
        old_idx = rec["episode_index"]
        if old_idx not in old_to_new:
            raise ValueError(
                f"episodes.jsonl has episode_index {old_idx} with no matching data/video file"
            )
        new_idx = old_to_new[old_idx]
        renumbered += new_idx != old_idx
        rec["episode_index"] = new_idx
        new_ep_lines.append((new_idx, json.dumps(rec)))
    new_ep_lines.sort(key=lambda t: t[0])
    episodes_path.write_text("\n".join(line for _, line in new_ep_lines) + "\n")
    print(
        f"prepare: renumbered {renumbered}/{len(ep_lines)} episodes.jsonl entries "
        f"to dense 0..{len(ep_lines) - 1}"
    )


def convert(root: Path, repo_id: str, video_file_size_in_mb: int = 1) -> None:
    from lerobot.scripts.convert_dataset_v21_to_v30 import convert_dataset

    convert_dataset(
        repo_id=repo_id,
        root=str(root),
        push_to_hub=False,
        video_file_size_in_mb=video_file_size_in_mb,
    )


def postprocess(root: Path) -> None:
    from lerobot.datasets.dataset_tools import recompute_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    recompute_stats(LeRobotDataset(repo_id=root.name, root=root))

    from camelo.train.dataset_stats import repair_degenerate_quantiles, verify

    repair_degenerate_quantiles(root)
    verify(root)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root", type=Path, required=True, help="v2.1 dataset root; converted in place to v3.0"
    )
    parser.add_argument(
        "--repo-id", default=None, help="label only (push-to-hub always off); defaults to root.name"
    )
    parser.add_argument(
        "--video-file-size-in-mb",
        type=int,
        default=1,
        help="see module docstring point 5 before raising this",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="only run the in-place v2.1 fixes, skip lerobot's converter and postprocessing",
    )
    args = parser.parse_args()

    info = json.loads((args.root / "meta" / "info.json").read_text())
    if info.get("codebase_version") == "v3.0":
        print(f"{args.root} is already v3.0, nothing to do")
        return 0

    prepare(args.root)
    if args.skip_convert:
        return 0

    convert(args.root, args.repo_id or args.root.name, args.video_file_size_in_mb)
    postprocess(args.root)
    print(f"{args.root}: converted to v3.0 and verified (recompute_stats + repair + verify)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
