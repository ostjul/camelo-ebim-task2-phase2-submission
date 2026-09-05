#!/usr/bin/env python3
"""Repair a v3 dataset whose episode metadata points at files that do not exist.

    python scripts/repair_episode_meta.py outputs/datasets/ext_hermanprawiro_task2_fixpos_200

`hermanprawiro/task2_fixpos_200` ships `meta/episodes/chunk-000/file-000.parquet`
containing all 200 episode rows, but 90 of those rows (episodes 17-199)
carry `meta/episodes/file_index = 1`, pointing at a `file-001.parquet`
that was never uploaded. Every lerobot tool that reads per-episode
metadata — `delete_episodes`, and therefore our `make merge --drop` —
follows that pointer and dies with a bare FileNotFoundError naming a path
the dataset never had (F-71).

The frame data and the videos are complete and self-consistent; only the
pointer column is wrong. So the repair is to rewrite each row's
chunk/file index to the file the row is *actually* stored in — verifiable
locally, no re-download, no data touched.

Idempotent: a dataset whose pointers already resolve is left alone, so
this is safe to re-run after re-fetching the corpus.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CHUNK_COL = "meta/episodes/chunk_index"
FILE_COL = "meta/episodes/file_index"


def repair(root: Path, dry_run: bool = False) -> int:
    import pandas as pd

    meta_dir = root / "meta" / "episodes"
    if not meta_dir.is_dir():
        print(f"error: no meta/episodes under {root}", file=sys.stderr)
        return 2

    files = sorted(meta_dir.glob("chunk-*/file-*.parquet"))
    if not files:
        print(f"error: no episode metadata parquet under {meta_dir}", file=sys.stderr)
        return 2

    repaired = 0
    for path in files:
        frame = pd.read_parquet(path)
        if CHUNK_COL not in frame.columns or FILE_COL not in frame.columns:
            continue
        # The row lives in THIS file, whatever it claims.
        chunk_idx = int(path.parent.name.split("-")[1])
        file_idx = int(path.stem.split("-")[1])
        wrong = (frame[CHUNK_COL] != chunk_idx) | (frame[FILE_COL] != file_idx)
        if not wrong.any():
            continue
        # Only rewrite pointers that do not resolve — a legitimately
        # multi-file dataset must be left alone.
        dangling = []
        for _, row in frame[wrong].iterrows():
            chunk_dir = meta_dir / f"chunk-{int(row[CHUNK_COL]):03d}"
            target = chunk_dir / f"file-{int(row[FILE_COL]):03d}.parquet"
            if not target.is_file():
                dangling.append(target)
        if not dangling:
            continue
        print(f"{path}: {int(wrong.sum())} rows point at missing files, e.g. {dangling[0]}")
        if dry_run:
            continue
        frame.loc[wrong, CHUNK_COL] = chunk_idx
        frame.loc[wrong, FILE_COL] = file_idx
        frame.to_parquet(path, index=False)
        repaired += int(wrong.sum())

    if repaired:
        note = root / "camelo_repair.json"
        note.write_text(
            json.dumps(
                {
                    "repaired_rows": repaired,
                    "columns": [CHUNK_COL, FILE_COL],
                    "why": "published metadata pointed at episode-metadata files that were "
                    "never uploaded; all rows are present in the files that exist (F-71). "
                    "Frame data and videos untouched.",
                },
                indent=2,
            )
        )
        print(f"repaired {repaired} rows -> {note}")
    else:
        print("nothing to repair — every episode-metadata pointer resolves")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return repair(args.root, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
