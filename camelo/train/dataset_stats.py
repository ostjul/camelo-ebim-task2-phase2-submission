"""Post-`recompute_stats` repairs and checks for a derived dataset.

Two failure modes, both silent, both found the hard way (F-83, F-85).

**Missing stats.** lerobot returns any key it has no stats for UNCHANGED
(`processor/normalize_processor.py:329-331`), so a dataset that lost its
`observation.state` stats trains unnormalized without a warning.

**Degenerate quantiles.** `NormalizationMode.QUANTILES` divides by
`q99 - q01` and guards only the exactly-zero case with an epsilon
(`:396-398`). That guard is right for a genuinely constant column, where
the numerator is also zero. It is catastrophic for a column that is
constant for >99 % of frames but not all: q01 and q99 collapse onto the
same value while real outliers remain, so those frames normalize to
`2 * (x - q01) / 1e-8`, i.e. ~1e7.

Task 2's spine action is exactly that shape. Trimming the ramp removed
the 0 -> 0.5 sweep that used to spread the column, leaving ~99 % of frames
at each episode's plateau; because most episodes plateau near 0.50, both
quantiles landed on 0.499191 while the deliberately augmented 0.45/0.55
episodes stayed. pi0.5 — the only rung whose norm_map is QUANTILES rather
than MEAN_STD — then trained against action targets of magnitude 1e7 and
sat at loss 2.6e11 from step 200 while SLURM reported COMPLETED, exit 0.

Repair widens such dims to min/max, which is what a quantile range is
approximating anyway. Genuinely constant columns are left alone: their
epsilon path is harmless.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# A column counts as varying (and so its collapsed quantiles as degenerate)
# when its raw span exceeds this. Below it, epsilon division is harmless.
SPAN_TOL = 1e-6


def repair_degenerate_quantiles(root: Path) -> dict[str, list[int]]:
    """Widen q01/q99 to min/max wherever they collapsed on a varying column.

    Returns {feature: [dim, ...]} for whatever was repaired."""
    stats_path = Path(root) / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    repaired: dict[str, list[int]] = {}

    for feature, entry in stats.items():
        if not isinstance(entry, dict) or not {"q01", "q99", "min", "max"} <= set(entry):
            continue
        q01 = np.asarray(entry["q01"], dtype=np.float64)
        q99 = np.asarray(entry["q99"], dtype=np.float64)
        low = np.asarray(entry["min"], dtype=np.float64)
        high = np.asarray(entry["max"], dtype=np.float64)
        if q01.ndim == 0:
            continue
        bad = np.where((q99 - q01 == 0) & (high - low > SPAN_TOL))[0]
        if not len(bad):
            continue
        q01[bad] = low[bad]
        q99[bad] = high[bad]
        entry["q01"] = q01.tolist()
        entry["q99"] = q99.tolist()
        repaired[feature] = [int(i) for i in bad]

    if repaired:
        stats_path.write_text(json.dumps(stats, indent=4))
        for feature, dims in repaired.items():
            print(f"stats: widened collapsed q01/q99 to min/max on {feature} dims {dims} (F-85)")
    return repaired


def worst_normalized_magnitude(root: Path, feature: str) -> float:
    """Largest |value| the QUANTILES path would produce for `feature`."""
    import glob

    import pandas as pd

    stats = json.loads((Path(root) / "meta" / "stats.json").read_text()).get(feature)
    if not stats or "q01" not in stats:
        return float("nan")
    q01 = np.asarray(stats["q01"], dtype=np.float64)
    q99 = np.asarray(stats["q99"], dtype=np.float64)
    denom = np.where(q99 - q01 == 0, 1e-8, q99 - q01)
    files = sorted(glob.glob(str(Path(root) / "data" / "chunk-*" / "*.parquet")))
    frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    values = np.stack(frame[feature].to_numpy()).astype(np.float64)
    return float(np.abs(2.0 * (values - q01) / denom - 1.0).max())


def verify(root: Path, features: tuple[str, ...] = ("observation.state", "action")) -> None:
    """Raise unless every feature has usable stats AND sane quantile scaling."""
    stats = json.loads((Path(root) / "meta" / "stats.json").read_text())
    for feature in features:
        entry = stats.get(feature) or {}
        missing = {"min", "max", "mean", "std", "q01", "q99"} - set(entry)
        if missing:
            raise RuntimeError(
                f"{root}: {feature} is missing {sorted(missing)} — a dataset without stats "
                "trains UNNORMALIZED and nothing warns (F-83)"
            )
        worst = worst_normalized_magnitude(root, feature)
        if worst > 100.0:
            raise RuntimeError(
                f"{root}: {feature} normalizes to magnitude {worst:.3e} under QUANTILES — "
                "collapsed q01/q99 on a varying column (F-85). Repair before training."
            )
        print(f"stats: {feature} ok (|normalized| max {worst:.3f})")
