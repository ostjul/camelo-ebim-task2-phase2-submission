#!/usr/bin/env bash
# Fetch the newest head capture from ebimHP and check the red pad against a slot.
#   tools/rig_probes/pad_centroid.sh --slot x051
set -euo pipefail
cd "$(dirname "$0")/../.."
HOST="${RIG_HOST:-ebim@192.168.0.5}"
OUT="outputs/rig/munich_2026-09-01/t6_overlays"; mkdir -p "$OUT"
REMOTE=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" \
  'find ~/camelo/camelo-ebim/outputs/rig -type f -name "live_head_*.png" -printf "%T@ %p\n" | sort -rn | head -1 | cut -d" " -f2-')
NAME=$(basename "$REMOTE" .png)
scp -q "$HOST:$REMOTE" "$OUT/$NAME.png"
python3 tools/rig_probes/pad_centroid.py "$OUT/$NAME.png" "$@"
