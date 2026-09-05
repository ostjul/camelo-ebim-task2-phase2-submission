# Shared helpers for configs/realdata/<policy>/launch.sh and smoke.sh.
# Sourced, not executed: `source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"`.
#
# All real-data runs share one corpus (built by a sibling agent, see
# docs/realdata/00_PROTOCOL.md and camelo/train/build_real_corpus.py):
#   outputs/datasets/ebim_task2_realrobotdata/task2_munich_s27a15_hires/
# with the frozen train-episode list at meta/train_episodes.txt inside it
# (the held-out split lives in meta/splits_camelo.json — never train on it).
#
# 2026-08-31 (launch): switched from task2_munich_s27a15 (238 eps) to the
# _hires variant (217 eps, renumbered 0-216, 121,828 frames, 195 train /
# 22 held out) -- the --drop-lowres build, which removes the 21 native
# 336x188 head episodes instead of re-encoding them (00_PROTOCOL.md D4 /
# G5).  This is the ONE place the corpus root is named; every launch.sh
# and smoke.sh sources it, so nothing else needs editing.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="${DATA:-$REPO_ROOT/outputs/datasets/ebim_task2_realrobotdata/task2_munich_s27a15_hires}"

# Emits --dataset.episodes=[0,1,2,...] read from $DATA/meta/train_episodes.txt.
# Format-agnostic (one-per-line or comma-separated): pulls every integer out
# of the file, so it survives whichever convention the corpus builder used.
# sbatch --export is itself comma-separated (see slurm/train.slurm's own
# header comment) -- that only bites a literal --export=VAR=a,b,c on the
# sbatch command line. Here the bracketed list travels inside EXTRA, which
# the Makefile passes through the job's exported environment
# (--export=ALL), never through --export's own comma-split parser, so it
# survives intact.
train_episodes_flag() {
    local f="$DATA/meta/train_episodes.txt"
    if [ ! -f "$f" ]; then
        echo "error: $f not found -- has the derived corpus been built yet?" \
             "(DATA=$DATA, override with DATA=<path> before calling this script)" >&2
        exit 2
    fi
    local ids
    ids=$(grep -oE '[0-9]+' "$f" | paste -sd, -)
    if [ -z "$ids" ]; then
        echo "error: $f contained no episode indices" >&2
        exit 2
    fi
    echo "--dataset.episodes=[$ids]"
}
