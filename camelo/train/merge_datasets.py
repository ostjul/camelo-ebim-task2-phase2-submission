"""Merge recorder output versions into one training dataset.

The benchmark recorder writes a fresh ``task2_thermalpad_vN`` directory per
launch; training wants one dataset. Uses lerobot's aggregation utility.

    python -m camelo.train.merge_datasets \
        --sources <benchmark>/task2_isaacsim/dataset/task2_thermalpad_v1 \
                  <benchmark>/task2_isaacsim/dataset/task2_thermalpad_v2 \
        --output outputs/datasets/task2_merged_v1

Per-source episode drops (e.g. fixpos episode 19's double grasp,
TRAINING.md) filter through lerobot's ``delete_episodes`` before
aggregation:

    ... --sources outputs/datasets/ext_hermanprawiro_task2_fixpos_v1 \
        --output outputs/datasets/ext_fixpos_clean_v1 \
        --drop ext_hermanprawiro_task2_fixpos_v1=19

Success labels live in each source's task2_extras/episodes_task2.jsonl;
filter whole-episode success there the same way if needed (aggregation
itself keeps everything).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

_LEROBOT_HINT = (
    "lerobot with dataset tooling support is required "
    "(pip install 'camelo-ebim[train]'); if an import path moved in "
    "your lerobot version, adjust camelo/train/merge_datasets.py."
)


def parse_drop_specs(specs: list[str] | None) -> dict[str, list[int]]:
    """``<source_dir_name>=<ep>[,<ep>...]`` specs -> {name: sorted unique indices}."""
    drops: dict[str, set[int]] = {}
    for spec in specs or []:
        name, sep, eps = spec.partition("=")
        if not sep or not name or not eps:
            raise ValueError(f"--drop spec must be <source_name>=<ep>[,<ep>...], got {spec!r}")
        try:
            indices = {int(e) for e in eps.split(",")}
        except ValueError as exc:
            raise ValueError(f"non-integer episode index in --drop spec {spec!r}") from exc
        if any(i < 0 for i in indices):
            raise ValueError(f"negative episode index in --drop spec {spec!r}")
        drops.setdefault(name, set()).update(indices)
    return {name: sorted(eps) for name, eps in drops.items()}


def merge(sources: list[Path], output: Path, drops: dict[str, list[int]] | None = None) -> None:
    drops = drops or {}
    for src in sources:
        if not (src / "meta" / "info.json").is_file():
            raise FileNotFoundError(f"not a LeRobot dataset: {src}")
    source_names = {src.name for src in sources}
    unknown = sorted(set(drops) - source_names)
    if unknown:
        raise ValueError(
            f"--drop names {unknown} match no --sources dir name {sorted(source_names)}"
        )
    original_sources = list(sources)
    if output.exists():  # lerobot's create() uses exist_ok=False deep inside
        raise FileExistsError(
            f"output already exists: {output} — remove it or pick a new --output "
            "(merges are not resumable)"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        from lerobot.datasets.aggregate import aggregate_datasets
    except ImportError as exc:
        raise RuntimeError(_LEROBOT_HINT) from exc

    filter_root = output.parent / f"{output.name}.tmp_filter"
    if drops:
        try:
            from lerobot.datasets.dataset_tools import delete_episodes
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError(_LEROBOT_HINT) from exc
        if filter_root.exists():
            shutil.rmtree(filter_root)
        filtered = []
        for src in sources:
            if src.name not in drops:
                filtered.append(src)
                continue
            dst = filter_root / src.name
            dataset = LeRobotDataset(repo_id=src.name, root=src)
            delete_episodes(dataset, drops[src.name], output_dir=dst, repo_id=src.name)
            print(f"dropped episodes {drops[src.name]} from {src.name}")
            filtered.append(dst)
        sources = filtered

    aggregate_datasets(
        repo_ids=[src.name for src in sources],
        aggr_repo_id=output.name,
        roots=[src for src in sources],
        aggr_root=output,
    )
    if filter_root.exists():
        shutil.rmtree(filter_root)

    provenance = {
        "sources": [str(src.resolve()) for src in original_sources],
        "dropped_episodes": drops,
        "note": "merged by camelo.train.merge_datasets; success labels per "
        "source in <source>/task2_extras/episodes_task2.jsonl",
    }
    (output / "camelo_provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"merged {len(sources)} datasets -> {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--drop",
        nargs="+",
        default=None,
        metavar="NAME=EP[,EP...]",
        help="drop episodes from a source before merging, keyed by its dir name",
    )
    args = parser.parse_args()
    merge(args.sources, args.output, parse_drop_specs(args.drop))
    return 0


if __name__ == "__main__":
    sys.exit(main())
