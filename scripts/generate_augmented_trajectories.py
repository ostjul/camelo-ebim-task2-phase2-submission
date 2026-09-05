#!/usr/bin/env python3
"""Generate validated augmented heuristic references from prepared demos.

    python scripts/prepare_heuristic_actions.py --episode 89   # source prep
    python scripts/generate_augmented_trajectories.py \
        --source data/heuristic/task2_fixpos_200/ep089_actions.npy \
        --families start_offset,approach,disturbance,transfer,timewarp \
        --per-family 10 --seed 1 \
        --out data/heuristic/task2_fixpos_200/aug_v1

Offline and numpy-only (no ROS, no sim, no GPU): every variant is edited
in world-TCP space, re-solved with the same right-arm IK the heuristic
adapter runs online, and judged by `camelo.control.augment.validate`
BEFORE it costs any sim time. Only PASSING variants are written; failures
are printed with their reasons and counted in the manifest, so a
mis-tuned magnitude shows up as a bad yield rather than as a silent
short manifest.

Output layout under --out: one `<name>_{actions.npy,gt_traj.npz,meta.json}`
triple per passing variant (loadable by the unmodified adapter via
`--adapter heuristic:<actions_path>`), plus `manifest.json` listing them
for `scripts/collect_augmented.py`.

`splice` needs at least two --source episodes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camelo.control import augment


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="prepared epNNN_actions.npy (repeatable; the sibling _gt_traj.npz "
        "is found next to it). Run scripts/prepare_heuristic_actions.py first",
    )
    parser.add_argument(
        "--families",
        default="start_offset,approach,disturbance,transfer,timewarp",
        help=f"comma list from {', '.join(augment.FAMILIES)} "
        "(default: every single-source family; add 'splice' with 2+ sources)",
    )
    parser.add_argument("--per-family", type=int, default=10, help="variants drawn per family")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="magnitude multiplier on every family's sampled offsets/speeds "
        "(1.0 = the calibrated defaults)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/heuristic/task2_fixpos_200/aug_v1"),
        help="output directory for variant triples + manifest.json",
    )
    args = parser.parse_args()

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    unknown = [f for f in families if f not in augment.FAMILIES]
    if unknown:
        raise SystemExit(f"unknown families {unknown}; valid: {augment.FAMILIES}")

    refs = [augment.load_reference(path) for path in args.source]
    if "splice" in families and len(refs) < 2:
        raise SystemExit("family 'splice' needs at least two --source episodes")

    manifest_variants: list[dict] = []
    stats: dict[str, dict] = {}
    for family in families:
        t0 = time.monotonic()
        variants = augment.generate(
            refs, family, count=args.per_family, seed=args.seed, scale=args.scale
        )
        kept = 0
        for v in variants:
            if not v.report.passed:
                print(f"[reject] {v.name}: {'; '.join(v.report.reasons)}")
                continue
            paths = augment.write_variant(
                args.out,
                v.name,
                v.ref,
                {
                    "family": v.family,
                    "sources": v.sources,
                    "params": v.params,
                    "validation": v.report.as_dict(),
                },
            )
            manifest_variants.append(
                {
                    "name": v.name,
                    "family": v.family,
                    "sources": v.sources,
                    "actions": paths["actions"],
                    "gt": paths["gt"],
                    "meta": paths["meta"],
                }
            )
            kept += 1
        stats[family] = {"drawn": len(variants), "kept": kept}
        print(
            f"[{family}] kept {kept}/{len(variants)} "
            f"({time.monotonic() - t0:.1f}s)"
        )

    manifest = {
        "generator_version": augment.GENERATOR_VERSION,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "scale": args.scale,
        "sources": args.source,
        "families": stats,
        "variants": manifest_variants,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    total = len(manifest_variants)
    print(f"manifest: {manifest_path} ({total} variants)")
    if total == 0:
        print("ERROR: no variant passed validation — nothing to collect", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
