"""Build the derived real-robot training corpus from `task2_munich`.

`docs/realdata/00_PROTOCOL.md` §1 step 1.3. The source corpus is a
LeRobot v3.0 dataset that **cannot be trained on in place** — three
defects, none of them fixable with a launch flag:

1. **Nine STATE features, 84 dims.** `lerobot/utils/feature_utils.py`
   classifies anything whose key starts with `observation` as
   `FeatureType.STATE`, so the eight grouped
   `observation.state.<group>` columns are handed to every policy
   *alongside* the real `observation.state`. Deleting the keys from
   `meta/info.json` alone is not enough: the reader passes the declared
   schema into `Dataset.from_parquet(..., features=...)` and HF datasets
   raises `CastError: Couldn't cast … because column names don't match`.
   **The parquet has to be rewritten.**
2. **Wrong `q01`/`q99`.** The shipped `meta/stats.json` quantiles are a
   count-weighted mean of *per-episode* 5,000-bin histogram estimates,
   which puts 54.5 % of state values outside `[-1, 1]` under the
   QUANTILES path and `max|z| = 41.79`. They are recomputed here as true
   global quantiles over all frames.
3. **Two head-camera resolutions.** Episodes 0–20 are 336×188, the rest
   1280×720, and `default_collate` raises on a batch spanning both. The
   21 low-res head videos are re-encoded to 1280×720; every other video
   file is **symlinked**, so the derived tree costs ~0.2 GB instead of
   the 3.4 GB a naive copy would.

The canonical widths are 27 state / 15 action (`00 §0.2`, decisions
D5/D6). The right gripper's knuckle angle becomes an **open fraction**
`1 - clip(rad, 0, GRIPPER_CLOSED_RAD) / GRIPPER_CLOSED_RAD` (D7) and the
dim is renamed to carry the substring `gripper`, which MolmoAct2 uses to
find gripper dims by name. The action gripper channel keeps its recorded
polarity (1.0 = open, D8) and is only clamped into `[0, 1]`.

Build, then smoke::

    .venv/bin/python -m camelo.train.build_real_corpus \
        --source outputs/datasets/ebim_task2_realrobotdata/task2_munich \
        --output outputs/datasets/ebim_task2_realrobotdata/task2_munich_s27a15

    .venv/bin/python -m camelo.train.build_real_corpus --verify-only \
        --output outputs/datasets/ebim_task2_realrobotdata/task2_munich_s27a15

The build ends with the same gates `--verify-only` re-runs: G3
(normalization), G1 (one STATE / one ACTION / three VISUAL feature) and
G-load (decode every episode tail plus N random frames). Pass
`--skip-smoke` to stop after G3/G1.

Nothing here needs a GPU, SLURM or the network. Only the gates touch
lerobot, and only inside the function that needs it (AGENTS.md layering
rule 3); `av` and `PIL` are imported inside the re-encoder and the QC
writer for the same reason.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from camelo import contracts as C  # noqa: E402

# --- the canonical vectors (00 §0.2 — copy these, do not re-derive) --------

#: Source `observation.state` indices, in output order: left 7 joints,
#: right 7 joints, right gripper knuckle, left wrench 6, right wrench 6.
STATE_KEEP_S27: tuple[int, ...] = (
    *range(0, 7),
    *range(21, 28),
    28,
    *range(15, 21),
    *range(36, 42),
)
#: Source `action` indices, in output order: left 7, right 7, right gripper.
ACTION_KEEP_A15: tuple[int, ...] = (*range(0, 7), *range(8, 16))

STATE_PRESETS = {"s27": STATE_KEEP_S27, "r27": STATE_KEEP_S27}
ACTION_PRESETS = {"a15": ACTION_KEEP_A15}

#: Output dim of the gripper channel in both vectors — the parity that makes
#: `use_relative_actions` (positional, by index) meaningful.
STATE_GRIPPER_DIM = 14
ACTION_GRIPPER_DIM = 14

#: MolmoAct2 finds gripper dims by the substring "gripper" in the feature
#: name, so this rename is load-bearing, not cosmetic (03 §1.5).
STATE_GRIPPER_NAME = "franka_robot_right_gripper_open_fraction"

HEAD_KEY = "observation.images.head"
#: Head videos recorded at 336×188 instead of 1280×720 (01 §1.5).
LOWRES_EPISODES: tuple[int, ...] = tuple(range(21))
HEAD_TARGET_WIDTH = 1280
HEAD_TARGET_HEIGHT = 720

#: G3 (00 §1.3). The `max|z|` clause is the load-bearing half — under true
#: global q01/q99 every dim is ~2 % outside [-1, 1] by construction, so the
#: per-dim clause can only catch a builder bug (wrong axis, wrong rows).
G3_MAX_FRAC_OUTSIDE = 0.025
G3_MAX_ABS_Z = 10.0

#: 00 §5.1 — 24 held-out episodes (~10 %), stratified on three axes.
HELDOUT_SIZE = 24
#: `zero_close` is the fourth stratum the 00 §5.1 final-review note asks for:
#: the 7 candidate no-op recordings are stratified separately rather than
#: folded into "single-close" (7/238 * 24 = 0.7 -> 1).
HELDOUT_TARGETS = {"valid": 15, "invalid": 9, "hires": 22, "lowres": 2, "regrasp": 6,
                   "zero_close": 1}

STAT_KEYS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")
QUANTILE_KEYS = ("q01", "q10", "q50", "q90", "q99")
QUANTILE_PERCENTS = (1.0, 10.0, 50.0, 90.0, 99.0)


# --- pure functions (no disk, no lerobot — these are what tests exercise) --


def select_dims(values: np.ndarray, keep) -> np.ndarray:
    """`values[:, keep]` with the order of `keep` preserved."""
    keep = list(keep)
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(f"expected a (frames, dims) array, got shape {values.shape}")
    if max(keep) >= values.shape[1]:
        raise ValueError(f"index {max(keep)} out of range for width {values.shape[1]}")
    return values[:, keep].copy()


def select_names(names, keep) -> list[str]:
    """The same selection applied to a feature's `names` list."""
    names = list(names)
    return [names[i] for i in keep]


def gripper_open_fraction(radians, closed_rad: float = C.GRIPPER_CLOSED_RAD) -> np.ndarray:
    """Knuckle angle (rad) -> open fraction, 1.0 = fully open (D7).

    `closed_rad` is `camelo.contracts.GRIPPER_CLOSED_RAD` (0.8), the sim
    constant, so the real and sim gripper channels stay mergeable. The
    measured maximum knuckle angle in this corpus is 0.7929 rad, so the
    closed end lands at 0.0089 rather than exactly 0 — a ~1 % dead band,
    and the price of that mergeability.
    """
    if closed_rad <= 0:
        raise ValueError(f"closed_rad must be positive, got {closed_rad}")
    return 1.0 - np.clip(np.asarray(radians, dtype=np.float64), 0.0, closed_rad) / closed_rad


def apply_state_transform(state: np.ndarray, closed_rad: float = C.GRIPPER_CLOSED_RAD):
    """Select the 27 canonical state dims and convert the gripper dim."""
    out = select_dims(state, STATE_KEEP_S27).astype(np.float32)
    out[:, STATE_GRIPPER_DIM] = gripper_open_fraction(out[:, STATE_GRIPPER_DIM], closed_rad)
    return out


def apply_action_transform(action: np.ndarray):
    """Select the 15 canonical action dims and clamp the gripper to [0, 1].

    No polarity flip (D8): the recorded channel is already 1.0 = open. The
    clamp exists because a single frame at 1.0001 raises `MolmoAct2 action
    gripper values are not under [-1,1]` and kills the job (03 §2.2).
    """
    out = select_dims(action, ACTION_KEEP_A15).astype(np.float32)
    out[:, ACTION_GRIPPER_DIM] = np.clip(out[:, ACTION_GRIPPER_DIM], 0.0, 1.0)
    return out


def global_stats(values: np.ndarray) -> dict:
    """Exact mean/std/min/max and **true global** quantiles over all rows.

    The shipped stats are a count-weighted mean of per-episode histogram
    quantile *estimates* (`datasets/compute_stats.py`), which is the defect
    D9 exists to repair — so this deliberately does not go through any
    lerobot aggregation path.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    quantiles = np.percentile(values, QUANTILE_PERCENTS, axis=0)
    entry = {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(values.shape[0])],
    }
    for key, row in zip(QUANTILE_KEYS, quantiles, strict=True):
        entry[key] = row.tolist()
    return entry


def quantile_report(values: np.ndarray, entry: dict) -> dict:
    """G3 under `NormalizationMode.QUANTILES`: `2(x-q01)/(q99-q01) - 1`."""
    values = np.asarray(values, dtype=np.float64)
    q01 = np.asarray(entry["q01"], dtype=np.float64)
    q99 = np.asarray(entry["q99"], dtype=np.float64)
    denom = np.where(q99 - q01 == 0, 1e-8, q99 - q01)
    z = 2.0 * (values - q01) / denom - 1.0
    outside = (np.abs(z) > 1.0).mean(axis=0)
    return {
        "frac_outside_per_dim": outside.tolist(),
        "max_frac_outside": float(outside.max()),
        "worst_dim_outside": int(np.argmax(outside)),
        "max_abs_z_per_dim": np.abs(z).max(axis=0).tolist(),
        "max_abs_z": float(np.abs(z).max()),
        "worst_dim_z": int(np.argmax(np.abs(z).max(axis=0))),
    }


def mean_std_report(values: np.ndarray, entry: dict) -> dict:
    """The same check under `NormalizationMode.MEAN_STD`: `(x-mean)/std`."""
    values = np.asarray(values, dtype=np.float64)
    mean = np.asarray(entry["mean"], dtype=np.float64)
    std = np.asarray(entry["std"], dtype=np.float64)
    degenerate = [int(i) for i in np.where(std <= 0)[0]]
    safe = np.where(std <= 0, 1e-8, std)
    z = (values - mean) / safe
    return {
        "max_abs_z": float(np.abs(z).max()),
        "worst_dim": int(np.argmax(np.abs(z).max(axis=0))),
        "zero_std_dims": degenerate,
        "nan_or_inf": bool(not np.isfinite(z).all()),
    }


def episode_table(
    episode_index: np.ndarray,
    action: np.ndarray,
    validity: np.ndarray,
    source_ids: list[int] | None = None,
) -> list[dict]:
    """Per-episode facts the split and the subsets are stratified on.

    `grip_closes` counts `action[15]` crossing below 0.6 (01 §5.3); a
    re-grasp episode is one with >= 2 closes. Validity is constant within
    every episode and that is asserted, not assumed.

    `episode` is the id **in this corpus**; `source_episode` is the id in
    `task2_munich`. They differ whenever rows were dropped, and every
    stratum question (`hires`) is answered against the *source* id,
    because that is what the 336x188 head videos are keyed to.
    """
    episode_index = np.asarray(episode_index)
    action = np.asarray(action)
    validity = np.asarray(validity)
    rows = []
    for episode in np.unique(episode_index):
        mask = episode_index == episode
        values = validity[mask]
        if values.min() != values.max():
            raise ValueError(f"episode {episode}: annotation.human.validity is not constant")
        grip = action[mask, ACTION_GRIPPER_DIM] if action.shape[1] == 15 else action[mask, 15]
        closes = int(((grip[:-1] > 0.6) & (grip[1:] <= 0.6)).sum())
        source = int(episode) if source_ids is None else int(source_ids[int(episode)])
        rows.append(
            {
                "episode": int(episode),
                "source_episode": source,
                "frames": int(mask.sum()),
                "valid": int(values[0]),
                "hires": source not in LOWRES_EPISODES,
                "grip_closes": closes,
                "regrasp": closes >= 2,
            }
        )
    return rows


def heldout_targets(rows: list[dict], size: int) -> dict:
    """Proportional stratum targets for a held-out set of `size` episodes.

    Rounding each stratum's share to the nearest whole episode. A stratum
    that covers every episode or none of them (e.g. `hires` once the
    low-res episodes have been dropped) carries no information and is
    **omitted**, rather than being asked for as a constraint that is
    trivially satisfied or impossible.

    On the 238-episode corpus this reproduces 00 §5.1's hand-written
    targets exactly (15/9, 22/2, 6/18, and 1 zero-close).
    """
    total = len(rows)
    counts = {
        "valid": sum(1 for r in rows if r["valid"] == 1),
        "hires": sum(1 for r in rows if r["hires"]),
        "regrasp": sum(1 for r in rows if r["regrasp"]),
        "zero_close": sum(1 for r in rows if r["grip_closes"] == 0),
    }
    targets: dict[str, int] = {}
    for name, count in counts.items():
        if count in (0, total):
            continue
        targets[name] = int(round(size * count / total))
    if "valid" in targets:
        targets["invalid"] = size - targets["valid"]
    if "hires" in targets:
        targets["lowres"] = size - targets["hires"]
    return targets


def subset_lists(rows: list[dict], slow_camera: tuple[int, ...] = (55, 80, 145)) -> dict:
    """ALL / VALID / HIRES / CLEAN episode lists with their frame counts."""
    by_episode = {row["episode"]: row for row in rows}

    def pack(name: str, episodes: list[int], definition: str) -> dict:
        return {
            "definition": definition,
            "episodes": sorted(episodes),
            "n_episodes": len(episodes),
            "n_frames": int(sum(by_episode[e]["frames"] for e in episodes)),
        }

    all_eps = [r["episode"] for r in rows]
    valid = [r["episode"] for r in rows if r["valid"] == 1]
    hires = [r["episode"] for r in rows if r["hires"]]
    # 55/80/145 are SOURCE episode ids (01's slow-camera list), so they are
    # matched against source_episode, not against this corpus's numbering.
    slow = {r["episode"] for r in rows if r.get("source_episode", r["episode"]) in slow_camera}
    clean = [e for e in valid if e in set(hires) and e not in slow]
    valid_hires = [e for e in valid if e in set(hires)]
    return {
        "ALL": pack("ALL", all_eps, "every episode"),
        "VALID": pack("VALID", valid, "annotation.human.validity == 1"),
        "HIRES": pack("HIRES", hires, "head camera recorded at 1280x720 (episodes 21-237)"),
        "CLEAN": pack(
            "CLEAN",
            clean,
            f"VALID and HIRES and not in {list(slow_camera)} (slow camera) — read as "
            "*conservative*, not *known-good* (00 §1.4)",
        ),
        "VALID_HIRES": pack("VALID_HIRES", valid_hires, "VALID and HIRES, no slow-camera drop"),
    }


def stratified_heldout(
    rows: list[dict],
    size: int = HELDOUT_SIZE,
    seed: int = 0,
    targets: dict | None = None,
    max_tries: int = 500_000,
) -> dict:
    """A frozen held-out set that hits every stratum marginal **exactly**.

    By episode, never by frame (00 §5.1) — frames within an episode share
    a pose, a scene and a lighting condition, and treating them as
    independent is the pseudo-replication that flipped a t from -0.44 to
    -3.17 (TRAINING.md §M3).

    Rejection sampling rather than proportional allocation: the
    marginals (validity, head resolution, re-grasp, zero-close) are not
    independent here, so allocating per cell and rounding does not land on
    the target counts. Drawing until they all match is exact,
    deterministic in `seed`, and trivially checkable by a test.

    Any stratum absent from `targets` is simply not constrained, which is
    how a corpus that has already dropped its low-res episodes stops
    asking for a `hires` count it cannot vary.
    """
    targets = dict(targets or HELDOUT_TARGETS)
    episodes = np.array([r["episode"] for r in rows])
    valid = np.array([r["valid"] == 1 for r in rows])
    hires = np.array([r["hires"] for r in rows])
    regrasp = np.array([r["regrasp"] for r in rows])
    zero_close = np.array([r["grip_closes"] == 0 for r in rows])
    rng = np.random.default_rng(seed)
    for attempt in range(max_tries):
        pick = rng.permutation(len(rows))[:size]
        drawn = {
            "valid": int(valid[pick].sum()),
            "hires": int(hires[pick].sum()),
            "regrasp": int(regrasp[pick].sum()),
            "zero_close": int(zero_close[pick].sum()),
        }
        if any(drawn[name] != want for name, want in targets.items() if name in drawn):
            continue
        heldout = sorted(int(e) for e in episodes[pick])
        train = sorted(int(e) for e in episodes if int(e) not in set(heldout))
        by_episode = {r["episode"]: r for r in rows}
        return {
            "seed": seed,
            "tries": attempt + 1,
            "heldout": heldout,
            "train": train,
            "n_heldout": len(heldout),
            "n_train": len(train),
            "heldout_frames": int(sum(by_episode[e]["frames"] for e in heldout)),
            "train_frames": int(sum(by_episode[e]["frames"] for e in train)),
            "strata": {
                "valid": int(valid[pick].sum()),
                "invalid": int((~valid[pick]).sum()),
                "hires": int(hires[pick].sum()),
                "lowres": int((~hires[pick]).sum()),
                "regrasp": int(regrasp[pick].sum()),
                "not_regrasp": int((~regrasp[pick]).sum()),
                "zero_close": int(sum(by_episode[e]["grip_closes"] == 0 for e in heldout)),
            },
            "targets": targets,
        }
    raise RuntimeError(
        f"no {size}-episode subset matched the stratum targets {targets} in {max_tries} draws"
    )


def episodes_flag(episodes: list[int]) -> str:
    """The exact `--dataset.episodes` value a launch command needs.

    draccus parses the `DatasetConfig.episodes: list[int] | None` field
    from a bracketed, comma-separated literal. `sbatch --export` is itself
    comma-separated and will chop this, which is why the Makefile passes it
    as a single quoted argument (TIGER3_H100.md).
    """
    return "[" + ",".join(str(int(e)) for e in episodes) + "]"


# --- parquet / disk work ---------------------------------------------------


def _list_column(values: np.ndarray) -> pa.ListArray:
    """A `list<element: float>` column, matching the source's own type."""
    values = np.ascontiguousarray(values, dtype=np.float32)
    frames, width = values.shape
    offsets = pa.array(np.arange(0, (frames + 1) * width, width, dtype=np.int32))
    return pa.ListArray.from_arrays(offsets, pa.array(values.reshape(-1), type=pa.float32()))


def _load_source(source: Path):
    info = json.loads((source / "meta" / "info.json").read_text())
    data_files = sorted((source / "data").glob("chunk-*/*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"no data parquet under {source / 'data'}")
    if len(data_files) > 1:
        raise NotImplementedError(
            f"{source} has {len(data_files)} data files; this builder assumes one "
            "(task2_munich has exactly one) — extend it deliberately, do not guess"
        )
    return info, data_files[0]


def renumber_episodes(source_episode_index: np.ndarray, keep: tuple[int, ...] | None):
    """Map a kept subset of episodes onto a 0-based contiguous numbering.

    **This renumbering is not optional when rows are dropped.**
    `dataset_reader.get_item` reads `ep_idx` out of the row's own
    `episode_index` *value* and then does `self._meta.episodes[ep_idx]` —
    a **positional** index into the episodes table
    (`dataset_reader.py:320, :219, :275`; `dataset_metadata.get_video_file_path`
    does the same). A corpus that kept the source ids 21..237 would read
    the wrong episode's row for every frame, and would `IndexError` past
    216. So kept episode `k` becomes new id `rank(k)`.

    The video files keep their **source** names: the path comes from the
    episodes row's `videos/<key>/file_index`, which is carried through
    unchanged, so new episode 0 still points at `file-021.mp4`.

    Returns `(row_mask, new_episode_index, source_ids)` where `source_ids[i]`
    is the original episode id of new episode `i`.
    """
    source_episode_index = np.asarray(source_episode_index)
    present = np.unique(source_episode_index)
    source_ids = [int(e) for e in present] if keep is None else [
        int(e) for e in present if int(e) in set(keep)
    ]
    if not source_ids:
        raise ValueError("the keep set selects no episodes")
    remap = {old: new for new, old in enumerate(source_ids)}
    row_mask = np.isin(source_episode_index, source_ids)
    new_episode_index = np.array(
        [remap[int(e)] for e in source_episode_index[row_mask]], dtype=np.int64
    )
    return row_mask, new_episode_index, source_ids


def rewrite_parquet(
    source_file: Path,
    output_file: Path,
    closed_rad: float,
    keep_episodes: tuple[int, ...] | None = None,
) -> dict:
    """Drop the 8 sub-columns, select/reorder state and action, transform.

    Column order follows the source minus the dropped keys; the HF reader
    compares the column *set* against the declared features, not the order.

    When `keep_episodes` is given the rows of every other episode are
    dropped and `episode_index` / `index` are renumbered so the corpus is
    self-consistent (see `renumber_episodes`).
    """
    table = pq.read_table(source_file)
    source_episode_index = np.asarray(table.column("episode_index"))
    row_mask, new_episode_index, source_ids = renumber_episodes(
        source_episode_index, keep_episodes
    )
    if not row_mask.all():
        table = table.filter(pa.array(row_mask))

    state = np.stack(table.column("observation.state").to_numpy(zero_copy_only=False))
    action = np.stack(table.column("action").to_numpy(zero_copy_only=False))
    new_state = apply_state_transform(state, closed_rad)
    new_action = apply_action_transform(action)

    dropped = [c for c in table.column_names if c.startswith("observation.state.")]
    columns, names = [], []
    for name in table.column_names:
        if name in dropped:
            continue
        if name == "observation.state":
            columns.append(_list_column(new_state))
        elif name == "action":
            columns.append(_list_column(new_action))
        elif name == "episode_index":
            columns.append(pa.array(new_episode_index, type=pa.int64()))
        elif name == "index":
            # The global row number. It must stay 0..N-1 and agree with the
            # episodes table's dataset_from_index/dataset_to_index, which
            # _query_hf_dataset and _get_query_indices both rely on.
            columns.append(pa.array(np.arange(table.num_rows, dtype=np.int64), type=pa.int64()))
        else:
            columns.append(table.column(name).combine_chunks())
        names.append(name)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_arrays(columns, names=names), output_file)

    return {
        "rows": int(table.num_rows),
        "dropped_columns": dropped,
        "columns_in": len(table.column_names),
        "columns_out": len(names),
        "state": new_state,
        "action": new_action,
        "episode_index": new_episode_index,
        "source_ids": source_ids,
        "validity": np.asarray(table.column("annotation.human.validity")),
        "source_action": action,
        "source_state": state,
    }


def rewrite_info(
    info: dict,
    dropped: list[str],
    state_names,
    action_names,
    rows: int,
    episodes: int | None = None,
) -> dict:
    info = json.loads(json.dumps(info))
    for key in dropped:
        info["features"].pop(key, None)
    info["features"]["observation.state"] = {
        "dtype": "float32",
        "shape": [len(state_names)],
        "names": list(state_names),
        "fps": info["fps"],
    }
    info["features"]["action"] = {
        "dtype": "float32",
        "shape": [len(action_names)],
        "names": list(action_names),
        "fps": info["fps"],
    }
    info["total_frames"] = int(rows)
    if episodes is not None:
        info["total_episodes"] = int(episodes)
        # lerobot's own convention; nothing in the 0.6.1 read path consults it.
        info["splits"] = {"train": f"0:{int(episodes)}"}
    return info


def rewrite_episode_stats(
    source_file: Path,
    output_file: Path,
    episode_index: np.ndarray,
    state: np.ndarray,
    action: np.ndarray,
    dropped: list[str],
    source_ids: list[int] | None = None,
) -> int:
    """Per-episode `meta/episodes` stats, consistent with the new widths.

    `LeRobotDatasetMetadata` reads this parquet for `dataset_from_index`,
    `length` and the per-episode `stats/<feature>/<stat>` columns; leaving
    42-wide state stats beside a 27-wide feature is exactly the kind of
    silent disagreement F-83/F-85 are about.

    When `source_ids` is given, only those source episodes are kept, their
    rows are re-ordered to the new 0-based numbering, and
    `episode_index` / `dataset_from_index` / `dataset_to_index` are
    rewritten to match the filtered data parquet. `videos/<key>/file_index`
    is deliberately **not** touched — that is what keeps the derived tree
    pointing at the source's own video filenames.
    """
    table = pq.read_table(source_file)
    source_episodes = np.asarray(table.column("episode_index"))
    if source_ids is not None:
        position = {int(e): i for i, e in enumerate(source_episodes)}
        table = table.take(pa.array([position[e] for e in source_ids], type=pa.int64()))

    n_episodes = table.num_rows
    replacements: dict[str, pa.Array] = {}
    for feature, values in (("observation.state", state), ("action", action)):
        per_stat: dict[str, list] = {key: [] for key in STAT_KEYS}
        for episode in range(n_episodes):
            block = values[episode_index == episode]
            entry = global_stats(block)
            for key in STAT_KEYS:
                per_stat[key].append(entry[key])
        for key in STAT_KEYS:
            arrow_type = pa.int64() if key == "count" else pa.float64()
            replacements[f"stats/{feature}/{key}"] = pa.array(
                per_stat[key], type=pa.list_(arrow_type)
            )

    lengths = np.array([int((episode_index == e).sum()) for e in range(n_episodes)])
    boundaries = np.concatenate([[0], np.cumsum(lengths)])
    replacements["episode_index"] = pa.array(np.arange(n_episodes, dtype=np.int64))
    replacements["length"] = pa.array(lengths.astype(np.int64))
    replacements["dataset_from_index"] = pa.array(boundaries[:-1].astype(np.int64))
    replacements["dataset_to_index"] = pa.array(boundaries[1:].astype(np.int64))

    columns, names = [], []
    for name in table.column_names:
        if any(name.startswith(f"stats/{key}/") for key in dropped):
            continue
        columns.append(replacements.get(name, table.column(name).combine_chunks()))
        names.append(name)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_arrays(columns, names=names), output_file)
    return len(names)


def reencode_head_video(source: Path, output: Path, width: int, height: int, codec: str) -> int:
    """Re-encode one head video, preserving every presentation timestamp.

    LeRobot decodes a frame by seeking `from_timestamp + row.timestamp` in
    the file, so the PTS grid is the alignment contract. Copying `pts` and
    `time_base` through verbatim keeps the derived video frame-for-frame
    addressable the way the source was.
    """
    import av  # lazy: keeps the pure functions importable without PyAV

    container_in = av.open(str(source))
    stream_in = container_in.streams.video[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    container_out = av.open(str(output), "w")
    stream_out = container_out.add_stream(codec, rate=stream_in.average_rate)
    stream_out.width, stream_out.height = width, height
    stream_out.pix_fmt = "yuv420p"
    stream_out.time_base = stream_in.time_base
    stream_out.codec_context.time_base = stream_in.time_base
    stream_out.options = {"crf": "23"} | ({"preset": "8"} if codec == "libsvtav1" else {})
    frames = 0
    for frame in container_in.decode(stream_in):
        resized = frame.reformat(width=width, height=height, format="yuv420p")
        resized.pts, resized.time_base = frame.pts, stream_in.time_base
        for packet in stream_out.encode(resized):
            container_out.mux(packet)
        frames += 1
    for packet in stream_out.encode():
        container_out.mux(packet)
    container_out.close()
    container_in.close()
    return frames


def link_videos(
    source: Path,
    output: Path,
    reencode: dict[str, set[int]],
    codec: str,
    keep_episodes: tuple[int, ...] | None = None,
) -> dict:
    """Symlink the whole `videos/` tree, then overwrite what we re-encode.

    Symlinks are what makes the derived corpus ~0.2 GB instead of 3.4 GB.
    Nothing in this builder writes into `videos/` except the re-encoded
    head files, which land as real files beside the links.
    """
    summary = {"symlinked": 0, "reencoded": 0, "reencoded_frames": 0, "codec": codec}
    for source_file in sorted((source / "videos").glob("*/chunk-*/*.mp4")):
        relative = source_file.relative_to(source)
        target = output / relative
        video_key = relative.parts[1]
        episode = int(source_file.stem.split("-")[-1])
        if keep_episodes is not None and episode not in set(keep_episodes):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if episode in reencode.get(video_key, set()):
            frames = reencode_head_video(
                source_file, target, HEAD_TARGET_WIDTH, HEAD_TARGET_HEIGHT, codec
            )
            summary["reencoded"] += 1
            summary["reencoded_frames"] += frames
        else:
            target.symlink_to(source_file.resolve())
            summary["symlinked"] += 1
    return summary


def write_qc_pngs(
    source: Path, output: Path, episodes: tuple[int, ...], reference: int
) -> list[str]:
    """G5: side-by-side original vs re-encoded head frames, for the user.

    U6 gates the re-encode on a human looking at these. The reference
    frame from a natively-1280×720 episode is there so "does the upscale
    look reasonable" can be answered against what the rest of the corpus
    actually looks like, not against the 336×188 original alone.
    """
    import av
    from PIL import Image

    qc_dir = output / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    def grab(path: Path, index: int):
        container = av.open(str(path))
        stream = container.streams.video[0]
        picked = None
        for i, frame in enumerate(container.decode(stream)):
            if i == index:
                picked = frame.to_ndarray(format="rgb24")
                break
        container.close()
        if picked is None:
            raise RuntimeError(f"{path} has no frame {index}")
        return picked

    written = []
    for episode in episodes:
        name = f"chunk-000/file-{episode:03d}.mp4"
        original = grab(source / "videos" / HEAD_KEY / name, 60)
        derived = grab(output / "videos" / HEAD_KEY / name, 60)
        left = Image.fromarray(original).resize(
            (HEAD_TARGET_WIDTH, HEAD_TARGET_HEIGHT), Image.NEAREST
        )
        right = Image.fromarray(derived)
        canvas = Image.new("RGB", (HEAD_TARGET_WIDTH * 2 + 16, HEAD_TARGET_HEIGHT), "black")
        canvas.paste(left, (0, 0))
        canvas.paste(right, (HEAD_TARGET_WIDTH + 16, 0))
        path = qc_dir / f"head_ep{episode:03d}_orig336x188_vs_reencoded1280x720.png"
        canvas.save(path)
        written.append(str(path))

    native = grab(source / "videos" / HEAD_KEY / f"chunk-000/file-{reference:03d}.mp4", 60)
    path = qc_dir / f"head_ep{reference:03d}_native1280x720_reference.png"
    Image.fromarray(native).save(path)
    written.append(str(path))

    # The field-of-view contact sheet. The re-encode makes the *tensor*
    # uniform; it cannot make the *scene* uniform, and eps 0-20 were shot
    # with a visibly narrower framing than 21-237. The user needs to see
    # that before deciding U6, because it is an argument for --drop-lowres
    # that resolution alone does not make.
    tile = (640, 360)
    grid = ((0, 10, 20), (reference, 100, 237))
    canvas = Image.new("RGB", (tile[0] * 3, tile[1] * 2), "black")
    for row, episodes_in_row in enumerate(grid):
        for column, episode in enumerate(episodes_in_row):
            frame = grab(source / "videos" / HEAD_KEY / f"chunk-000/file-{episode:03d}.mp4", 60)
            canvas.paste(Image.fromarray(frame).resize(tile), (column * tile[0], row * tile[1]))
    path = qc_dir / "head_fov_lowres_eps000_010_020_vs_hires_eps021_100_237.png"
    canvas.save(path)
    written.append(str(path))
    return written


# --- gates -----------------------------------------------------------------


def summarize_gates(gates: dict[str, dict]) -> dict:
    """Combine gate result dicts into one overall pass/fail.

    `gates` maps a gate name ("G3", "G1", "G-load", …) to that gate's
    result, which must carry a boolean `"pass"` key (every `gate_*`
    function above returns one). A gate that was not executed — e.g.
    G-load under `--skip-smoke` — must simply be absent from `gates`,
    never present with a placeholder value: this function has no way to
    tell "skipped" from "ran and passed" apart from omission, so an empty
    `gates` dict is reported as an overall PASS with nothing failed. The
    caller is responsible for reporting what was skipped separately (see
    `main`'s `--skip-smoke` handling).
    """
    failed = [name for name, result in gates.items() if not result["pass"]]
    return {"pass": not failed, "failed": failed}


def print_gates_summary(gates: dict[str, dict], *, skipped: tuple[str, ...] = ()) -> bool:
    """Print the single final `GATES: PASS|FAIL` line; return the overall pass/fail.

    `skipped` names gates that were deliberately not run (e.g. G-load
    under `--skip-smoke`) — printed on their own line so a FAIL summary
    never gets misread as "everything ran and something failed" when in
    fact fewer gates ran than usual.
    """
    if skipped:
        print(f"GATES: skipped {', '.join(skipped)}")
    summary = summarize_gates(gates)
    if summary["pass"]:
        print("GATES: PASS")
    else:
        print(f"GATES: FAIL ({', '.join(summary['failed'])})")
    return summary["pass"]


def gate_g3(root: Path, state: np.ndarray | None = None, action: np.ndarray | None = None) -> dict:
    """G3: normalization sanity under QUANTILES, plus a MEAN_STD check."""
    stats = json.loads((root / "meta" / "stats.json").read_text())
    if state is None or action is None:
        table = pq.read_table(sorted((root / "data").glob("chunk-*/*.parquet"))[0])
        state = np.stack(table.column("observation.state").to_numpy(zero_copy_only=False))
        action = np.stack(table.column("action").to_numpy(zero_copy_only=False))

    result = {"pass": True, "features": {}}
    for feature, values in (("observation.state", state), ("action", action)):
        entry = stats[feature]
        quantiles = quantile_report(values, entry)
        mean_std = mean_std_report(values, entry)
        ok = (
            quantiles["max_frac_outside"] <= G3_MAX_FRAC_OUTSIDE
            and quantiles["max_abs_z"] <= G3_MAX_ABS_Z
            and not mean_std["nan_or_inf"]
        )
        result["pass"] &= ok
        result["features"][feature] = {"quantiles": quantiles, "mean_std": mean_std, "pass": ok}
        print(f"G3 {feature} ({values.shape[1]} dims, {values.shape[0]} frames)")
        print("  QUANTILES  per-dim % of frames outside [-1,1]:")
        for dim, frac in enumerate(quantiles["frac_outside_per_dim"]):
            marker = "  <-- worst" if dim == quantiles["worst_dim_outside"] else ""
            print(
                f"    dim {dim:>2}  {100 * frac:6.3f} %   "
                f"max|z| {quantiles['max_abs_z_per_dim'][dim]:8.4f}{marker}"
            )
        print(
            f"  QUANTILES  worst per-dim outside {100 * quantiles['max_frac_outside']:.3f} % "
            f"(dim {quantiles['worst_dim_outside']}, limit "
            f"{100 * G3_MAX_FRAC_OUTSIDE:.1f} %); max|z| {quantiles['max_abs_z']:.4f} "
            f"(dim {quantiles['worst_dim_z']}, limit {G3_MAX_ABS_Z:.0f})"
        )
        print(
            f"  MEAN_STD   max|z| {mean_std['max_abs_z']:.4f} (dim {mean_std['worst_dim']}); "
            f"zero-std dims {mean_std['zero_std_dims']}; "
            f"non-finite {mean_std['nan_or_inf']}"
        )
        print(f"  => {'PASS' if ok else 'FAIL'}")
    print(f"G3 overall: {'PASS' if result['pass'] else 'FAIL'}")
    return result


def gate_policy_features(root: Path) -> dict:
    """G1: what `dataset_to_policy_features` actually hands a policy."""
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.utils.feature_utils import dataset_to_policy_features

    meta = LeRobotDatasetMetadata(repo_id=Path(root).name, root=Path(root))
    features = dataset_to_policy_features(meta.features)
    by_type: dict[str, list[str]] = {}
    for key, feature in features.items():
        by_type.setdefault(feature.type.name, []).append(key)
    for kind, keys in sorted(by_type.items()):
        shapes = {k: tuple(features[k].shape) for k in sorted(keys)}
        print(f"G1 {kind}: {len(keys)} feature(s) {shapes}")
    state = by_type.get("STATE", [])
    action = by_type.get("ACTION", [])
    ok = (
        state == ["observation.state"]
        and action == ["action"]
        and tuple(features["observation.state"].shape) == (len(STATE_KEEP_S27),)
        and tuple(features["action"].shape) == (len(ACTION_KEEP_A15),)
        and len(by_type.get("VISUAL", [])) == 3
    )
    print(f"G1 overall: {'PASS' if ok else 'FAIL'}")
    return {"pass": bool(ok), "by_type": by_type}


def gate_smoke(
    root: Path,
    tolerance_s: float = 1.01,
    video_backend: str = "pyav",
    n_random: int = 200,
    windows: tuple[int, ...] = (0, 50, 40),
    seed: int = 0,
) -> dict:
    """G-load: decode every episode tail plus N random frames, all cameras.

    The tail set is `to-1, to-2, to-3, to-4` (the task's "last 4") union
    `to-5, to-10` (00 §1.5) — every one of the 799 frames that fail at
    `tolerance_s=0.15` lives in that window, so this gate exercises the
    mechanism it exists to catch rather than passing vacuously.
    """
    import gc
    import time

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root)
    results: dict = {"tolerance_s": tolerance_s, "video_backend": video_backend, "passes": []}
    rng = np.random.default_rng(seed)
    overall_ok = True

    for window in windows:
        delta = None
        if window:
            fps = json.loads((root / "meta" / "info.json").read_text())["fps"]
            delta = {"action": [i / fps for i in range(window)]}
        dataset = LeRobotDataset(
            repo_id=root.name,
            root=root,
            tolerance_s=tolerance_s,
            video_backend=video_backend,
            delta_timestamps=delta,
        )
        indices: list[int] = []
        for episode in range(dataset.meta.total_episodes):
            start = int(dataset.meta.episodes[episode]["dataset_from_index"])
            stop = int(dataset.meta.episodes[episode]["dataset_to_index"])
            for back in (1, 2, 3, 4, 5, 10):
                if stop - back >= start:
                    indices.append(stop - back)
        tails = len(indices)
        indices += [int(i) for i in rng.integers(0, len(dataset), size=n_random)]

        errors, shapes, widths = [], {}, {}
        started = time.perf_counter()
        for index in indices:
            try:
                item = dataset[index]
            except Exception as exc:  # noqa: BLE001 — the gate is "did anything raise"
                errors.append((index, f"{type(exc).__name__}: {exc}"))
                continue
            for key in ("observation.images.head", "observation.images.wrist_left",
                        "observation.images.wrist_right"):
                shapes.setdefault(key, set()).add(tuple(item[key].shape))
            state_shape = tuple(item["observation.state"].shape)
            widths.setdefault("observation.state", set()).add(state_shape)
            widths.setdefault("action", set()).add(tuple(item["action"].shape))
        elapsed = time.perf_counter() - started

        head_shapes = shapes.get("observation.images.head", set())
        ok = (
            not errors
            and len(head_shapes) == 1
            and widths["observation.state"] == {(len(STATE_KEEP_S27),)}
        )
        overall_ok &= ok
        label = "bare" if not window else f"{window}-step action window"
        print(
            f"G-load [{label}]: {len(indices)} frames ({tails} episode tails from "
            f"{dataset.meta.total_episodes} episodes + {n_random} random), "
            f"{len(errors)} error(s), {elapsed:.1f} s = {len(indices) / elapsed:.1f} reads/s"
        )
        for key in sorted(shapes):
            print(f"    {key}: {sorted(shapes[key])}")
        for key in sorted(widths):
            print(f"    {key}: {sorted(widths[key])}")
        for index, message in errors[:10]:
            print(f"    ERROR at {index}: {message}")
        print(f"  => {'PASS' if ok else 'FAIL'}", flush=True)
        # Drop the dataset before building the next one: each pass holds a
        # memory-mapped HF dataset plus decoder state, and the login node is
        # shared.
        del dataset
        gc.collect()
        results["passes"].append(
            {
                "window": window,
                "frames": len(indices),
                "tail_frames": tails,
                "errors": len(errors),
                "error_sample": errors[:10],
                "seconds": elapsed,
                "reads_per_s": len(indices) / elapsed,
                "shapes": {k: sorted(map(list, v)) for k, v in shapes.items()},
                "widths": {k: sorted(map(list, v)) for k, v in widths.items()},
                "pass": bool(ok),
            }
        )
    results["pass"] = bool(overall_ok)
    print(f"G-load overall: {'PASS' if overall_ok else 'FAIL'}")
    return results


# --- build -----------------------------------------------------------------


def provenance_payload(source: Path, args, summary: dict, source_provenance) -> dict:
    return {
        "source": str(Path(source).resolve()),
        "transform": "camelo.train.build_real_corpus",
        "state_preset": args.state_preset,
        "action_preset": args.action_preset,
        "state_indices": list(STATE_KEEP_S27),
        "action_indices": list(ACTION_KEEP_A15),
        "gripper": {
            "state_dim": STATE_GRIPPER_DIM,
            "state_name": STATE_GRIPPER_NAME,
            "transform": "1 - clip(rad, 0, GRIPPER_CLOSED_RAD) / GRIPPER_CLOSED_RAD",
            "closed_rad": C.GRIPPER_CLOSED_RAD,
            "action_dim": ACTION_GRIPPER_DIM,
            "action_transform": "clip(x, 0, 1) — no polarity flip, 1.0 = open (D8)",
        },
        "dropped_columns": summary["dropped_columns"],
        "episodes_kept": summary["source_ids"],
        "episode_numbering": (
            "episode_index is 0-based and contiguous in THIS corpus; episodes_kept[i] is the "
            "id of episode i in task2_munich. Renumbering is mandatory when rows are dropped "
            "(dataset_reader.get_item indexes meta.episodes POSITIONALLY by the row's "
            "episode_index value). Video files keep their SOURCE names, addressed via the "
            "episodes table's videos/<key>/file_index."
        ),
        "head_reencode": summary["videos"],
        "stats_method": "true global np.percentile over all frames (D9); "
        "per-episode meta/episodes stats recomputed exactly",
        "splits": summary["splits"],
        "subsets": {k: v["n_episodes"] for k, v in summary["subsets"].items()},
        "source_provenance": source_provenance,
    }


def build(args) -> dict:
    source, output = Path(args.source), Path(args.output)
    if output.exists() and not args.force:
        raise FileExistsError(f"output already exists: {output} (pass --force to rebuild)")
    if output.exists():
        shutil.rmtree(output)

    info, source_data = _load_source(source)
    print(f"source {source} — {info['total_episodes']} episodes / {info['total_frames']} frames")

    keep_episodes: tuple[int, ...] | None = None
    if args.drop_lowres:
        keep_episodes = tuple(
            e for e in range(int(info["total_episodes"])) if e not in set(LOWRES_EPISODES)
        )
        print(
            f"--drop-lowres: DROPPING source episodes {sorted(LOWRES_EPISODES)} outright — "
            f"{len(keep_episodes)} episodes kept and renumbered 0..{len(keep_episodes) - 1}"
        )

    print("rewriting the data parquet (dropping the 8 observation.state.<group> sub-columns)…")
    rewritten = rewrite_parquet(
        source_data,
        output / "data" / "chunk-000" / "file-000.parquet",
        C.GRIPPER_CLOSED_RAD,
        keep_episodes,
    )
    state, action = rewritten["state"], rewritten["action"]
    size_in = source_data.stat().st_size / 1024**2
    size_out = (output / "data" / "chunk-000" / "file-000.parquet").stat().st_size / 1024**2
    print(
        f"  {rewritten['columns_in']} -> {rewritten['columns_out']} columns, "
        f"{size_in:.1f} -> {size_out:.1f} MiB, {rewritten['rows']} rows"
    )
    print(f"  dropped: {rewritten['dropped_columns']}")

    state_names = select_names(info["features"]["observation.state"]["names"], STATE_KEEP_S27)
    state_names[STATE_GRIPPER_DIM] = STATE_GRIPPER_NAME
    action_names = select_names(info["features"]["action"]["names"], ACTION_KEEP_A15)
    print("state names (27):")
    for dim, name in enumerate(state_names):
        print(f"  {dim:>2}  {name}")
    print("action names (15):")
    for dim, name in enumerate(action_names):
        print(f"  {dim:>2}  {name}")
    if "gripper" not in state_names[STATE_GRIPPER_DIM]:
        raise RuntimeError("state gripper dim name must contain 'gripper' (MolmoAct2, 03 §1.5)")
    if "gripper" not in action_names[ACTION_GRIPPER_DIM]:
        raise RuntimeError("action gripper dim name must contain 'gripper'")

    # tasks.parquet and any other meta file rides through unchanged.
    (output / "meta").mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "meta" / "tasks.parquet", output / "meta" / "tasks.parquet")

    print("rewriting meta/episodes stats…")
    columns = rewrite_episode_stats(
        sorted((source / "meta" / "episodes").glob("chunk-*/*.parquet"))[0],
        output / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        rewritten["episode_index"],
        state,
        action,
        rewritten["dropped_columns"],
        rewritten["source_ids"] if keep_episodes is not None else None,
    )
    print(f"  meta/episodes: {columns} columns, {len(rewritten['source_ids'])} rows")

    print("recomputing meta/stats.json (true global quantiles)…")
    stats = json.loads((source / "meta" / "stats.json").read_text())
    for key in rewritten["dropped_columns"]:
        stats.pop(key, None)
    table = pq.read_table(output / "data" / "chunk-000" / "file-000.parquet")
    for name in table.column_names:
        if name in ("observation.state", "action"):
            continue
        if name not in stats:
            continue
        values = np.asarray(table.column(name).to_numpy(zero_copy_only=False))
        if values.dtype == object:
            values = np.stack(values)
        stats[name] = global_stats(values.astype(np.float64))
    stats["observation.state"] = global_stats(state)
    stats["action"] = global_stats(action)
    (output / "meta" / "stats.json").write_text(json.dumps(stats, indent=4))

    reencode: dict[str, set[int]] = {}
    if not args.drop_lowres:
        reencode[HEAD_KEY] = set(LOWRES_EPISODES)
    print(f"linking videos/ (re-encoding {len(reencode.get(HEAD_KEY, ()))} head files)…")
    videos = link_videos(source, output, reencode, args.head_codec, keep_episodes)
    videos["reencoded_episodes"] = sorted(reencode.get(HEAD_KEY, ()))
    videos["target"] = f"{HEAD_TARGET_WIDTH}x{HEAD_TARGET_HEIGHT}"
    print(
        f"  {videos['symlinked']} symlinked, {videos['reencoded']} re-encoded "
        f"({videos['reencoded_frames']} frames, {args.head_codec})"
    )

    rows = episode_table(
        rewritten["episode_index"],
        rewritten["source_action"],
        rewritten["validity"],
        rewritten["source_ids"],
    )
    subsets = subset_lists(rows)
    for name, entry in subsets.items():
        print(
            f"  subset {name:<12} {entry['n_episodes']:>3} episodes / "
            f"{entry['n_frames']:>7} frames"
        )
    heldout_size = args.heldout_size
    if args.drop_lowres and heldout_size == HELDOUT_SIZE:
        # Hold out the same ~10 % of a smaller corpus, not the same count.
        heldout_size = int(round(HELDOUT_SIZE * len(rows) / int(info["total_episodes"])))
    targets = heldout_targets(rows, heldout_size)
    split = stratified_heldout(rows, size=heldout_size, seed=args.seed, targets=targets)
    split["source_heldout"] = [rows[e]["source_episode"] for e in split["heldout"]]
    split["episode_ids_are"] = (
        "indices in THIS corpus; source_heldout gives the same episodes' ids in task2_munich"
    )
    print(
        f"  held-out {split['n_heldout']} episodes ({split['heldout_frames']} frames), "
        f"train {split['n_train']} ({split['train_frames']} frames), strata {split['strata']}"
    )

    train_episodes = list(split["train"])
    subsets["V1_TRAIN"] = {
        "definition": "everything in this corpus minus the frozen held-out set"
        + (" (this corpus is HIRES: eps 0-20 are not in it)" if args.drop_lowres else ""),
        "episodes": train_episodes,
        "n_episodes": len(train_episodes),
        "n_frames": int(sum(r["frames"] for r in rows if r["episode"] in set(train_episodes))),
    }
    (output / "meta" / "subsets.json").write_text(json.dumps(subsets, indent=2) + "\n")
    (output / "meta" / "splits_camelo.json").write_text(json.dumps(split, indent=2) + "\n")
    (output / "meta" / "episode_table.json").write_text(json.dumps(rows, indent=2) + "\n")
    (output / "meta" / "train_episodes.txt").write_text(episodes_flag(train_episodes) + "\n")
    print(f"  meta/train_episodes.txt: --dataset.episodes={episodes_flag(train_episodes)[:60]}…")

    # `info.json` carries ONLY the keys `DatasetInfo` knows about: `from_dict`
    # (datasets/utils.py:180-192) warns on every unknown key, on every single
    # load. The build's own metadata lives in `camelo_provenance.json` and the
    # `meta/*_camelo`/`meta/subsets` files instead. `splits` is left as lerobot
    # writes it ({"train": "0:N"}): nothing in the read path ever consults it
    # (verified in 0.6.1), so the real split mechanism is --dataset.episodes.
    new_info = rewrite_info(
        info,
        rewritten["dropped_columns"],
        state_names,
        action_names,
        rewritten["rows"],
        len(rewritten["source_ids"]),
    )
    (output / "meta" / "info.json").write_text(json.dumps(new_info, indent=4))

    summary = {
        "dropped_columns": rewritten["dropped_columns"],
        "source_ids": rewritten["source_ids"],
        "videos": videos,
        "splits": split,
        "subsets": subsets,
    }
    source_provenance_path = source / "camelo_provenance.json"
    source_provenance = (
        json.loads(source_provenance_path.read_text())
        if source_provenance_path.is_file()
        else None
    )
    (output / "camelo_provenance.json").write_text(
        json.dumps(provenance_payload(source, args, summary, source_provenance), indent=2) + "\n"
    )

    if not args.drop_lowres and not args.no_qc:
        print("writing G5 QC PNGs…")
        for path in write_qc_pngs(source, output, args.qc_episodes, args.qc_reference):
            print(f"  {path}")

    print()
    gates = {"G3": gate_g3(output, state, action)}
    print()
    gates["G1"] = gate_policy_features(output)
    return {"summary": summary, "gates": gates}


def _windows(args) -> tuple[int, ...]:
    return tuple(int(w) for w in str(args.windows).split(",") if w.strip() != "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-preset", default="s27", choices=sorted(STATE_PRESETS))
    parser.add_argument("--action-preset", default="a15", choices=sorted(ACTION_PRESETS))
    parser.add_argument("--head-codec", default="libsvtav1", choices=["libsvtav1", "libx264"])
    parser.add_argument(
        "--drop-lowres",
        action="store_true",
        help="U6 fallback: do not re-encode episodes 0-20; exclude them from the train list",
    )
    parser.add_argument("--no-qc", action="store_true", help="skip the G5 side-by-side PNGs")
    parser.add_argument("--qc-episodes", type=int, nargs="+", default=(0, 7, 15))
    parser.add_argument("--qc-reference", type=int, default=21)
    parser.add_argument("--heldout-size", type=int, default=HELDOUT_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="re-run G3, G1 and G-load against an existing --output "
        "(G3/G1 only under --skip-smoke)",
    )
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--tolerance-s", type=float, default=1.01)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--n", type=int, default=200, help="random frames in the load smoke")
    parser.add_argument(
        "--windows",
        default="0,50,40",
        help="comma-separated action delta_timestamps window lengths for the load smoke "
        "(0 = no window). The login node kills long-running processes, so run them "
        "one at a time if a full three-pass smoke is cut short.",
    )
    args = parser.parse_args()

    if args.verify_only:
        gates = {"G3": gate_g3(Path(args.output))}
        print()
        gates["G1"] = gate_policy_features(Path(args.output))
        if args.skip_smoke:
            ok = print_gates_summary(gates, skipped=("G-load",))
            return 0 if ok else 1
        print()
        gates["G-load"] = gate_smoke(
            Path(args.output), args.tolerance_s, args.video_backend, args.n, _windows(args)
        )
        ok = print_gates_summary(gates)
        return 0 if ok else 1

    if args.source is None:
        parser.error("--source is required unless --verify-only")
    result = build(args)
    gates = dict(result["gates"])  # G3, G1 — from the build itself, not re-run
    if args.skip_smoke:
        ok = print_gates_summary(gates, skipped=("G-load",))
        return 0 if ok else 1
    print()
    gates["G-load"] = gate_smoke(
        Path(args.output), args.tolerance_s, args.video_backend, args.n, _windows(args)
    )
    ok = print_gates_summary(gates)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
