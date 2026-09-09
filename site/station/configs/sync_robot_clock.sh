#!/usr/bin/env bash
# Sync the TMR companion's clock to this laptop.
#
# WHY THIS EXISTS
# ---------------
# The GELLO publisher and the pedal/base bridges run on the LAPTOP and stamp their
# messages with the LAPTOP clock. Two consumers on the robot age those stamps against the
# ROBOT clock:
#
#   * SwerveDriveController: cmd_vel_timeout 0.5 s. Skew > 0.5 s and the base silently
#     never moves - no error is logged anywhere.
#   * JointImpedanceController: rejects GELLO samples stamped in the future beyond
#     future_timestamp_tolerance, so arm teleop refuses to activate.
#
# The companion has NO outbound internet, so its chronyd has no reachable NTP source and
# never converges. It still disciplines the clock frequency, which means it also fights a
# manual `date -s` - a plain set only partially takes. This script stops chronyd for the
# duration of the correction, and verifies afterwards instead of assuming.
#
# HOW IT CORRECTS
# ---------------
# The new time is computed ON THE ROBOT from the robot's own clock plus a measured delta:
#
#     ssh companion 'date -s "$(date -d "+<delta> seconds")"'
#
# NOT by sending a laptop timestamp. A laptop timestamp goes stale while ssh connects and
# while you type the sudo password - which is exactly how an earlier attempt left the clock
# 4 s out. With a delta, the elapsed time does not matter.
#
# The skew measurement itself uses a multiplexed ssh connection: a cold connection costs
# ~0.3 s round-trip, and the midpoint estimate then carries up to +-0.15 s of error, which
# is larger than the tolerance we are trying to hit. Reusing one connection drops the
# round-trip to ~15 ms.
#
# USAGE
#   ./configs/sync_robot_clock.sh            # measure, correct if needed, verify
#   ./configs/sync_robot_clock.sh --check    # measure only, change nothing
#   ./configs/sync_robot_clock.sh --tolerance 0.1
#
# sudo on the robot prompts for a password once per correction round. sudo caches
# credentials PER TTY, so a separate `sudo -v` beforehand does not help - each ssh gets a
# fresh tty. That is why the whole correction runs inside a single `ssh -t`.
set -uo pipefail

HOST="${TMR_HOST:-companion}"
TOLERANCE=0.05      # seconds; below this we leave the clock alone
SAMPLES=7           # skew samples per round; the median is used
MAX_ROUNDS=3
CHECK_ONLY=0

# Unix socket paths are capped at ~108 bytes, and a long path fails in a way that looks
# like the remote command returning nothing. Keep this short.
CTL="/tmp/.tmrclock.$$"

while [ $# -gt 0 ]; do
    case "$1" in
        --check) CHECK_ONLY=1; shift ;;
        --tolerance) TOLERANCE="$2"; shift 2 ;;
        --host) HOST="$2"; shift 2 ;;
        -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

cleanup() { ssh -S "$CTL" -O exit "$HOST" 2>/dev/null; }
trap cleanup EXIT

echo "Robot clock sync  (host: $HOST, tolerance: ${TOLERANCE}s)"

if ! ssh -M -S "$CTL" -o ControlPersist=120 -o ConnectTimeout=10 -fN "$HOST" 2>/dev/null; then
    echo "ERROR: cannot open an ssh connection to $HOST" >&2
    exit 1
fi

# Median skew over several samples. Robust to a single slow round-trip.
measure_skew() {
    local samples=() i l1 r l2
    for ((i = 0; i < SAMPLES; i++)); do
        l1=$(date +%s.%N)
        r=$(ssh -S "$CTL" "$HOST" 'date +%s.%N' 2>/dev/null)
        l2=$(date +%s.%N)
        [ -z "$r" ] && continue
        samples+=("$(python3 -c "print(f'{$r-($l1+$l2)/2:.6f}')")")
    done
    [ ${#samples[@]} -eq 0 ] && return 1
    printf '%s\n' "${samples[@]}" | python3 -c "
import sys, statistics
v = [float(x) for x in sys.stdin if x.strip()]
print(f'{statistics.median(v):.6f}')
"
}

report() {
    local skew="$1"
    python3 - "$skew" "$TOLERANCE" <<'PY'
import sys
skew, tol = float(sys.argv[1]), float(sys.argv[2])
where = "behind" if skew < 0 else "ahead of"
print(f"  robot is {abs(skew):.3f}s {where} the laptop", end="")
print("   OK" if abs(skew) < tol else "   NEEDS CORRECTION")
PY
}

skew=$(measure_skew) || { echo "ERROR: could not measure skew" >&2; exit 1; }
report "$skew"

within() { python3 -c "import sys; sys.exit(0 if abs(float('$1')) < float('$TOLERANCE') else 1)"; }

if within "$skew"; then
    exit 0
fi

if [ "$CHECK_ONLY" = 1 ]; then
    echo "  (--check: no changes made)"
    exit 1
fi

for ((round = 1; round <= MAX_ROUNDS; round++)); do
    # Correction the ROBOT must apply to its own clock to match the laptop.
    # NOT an f-string: "$skew" expands to a single-quoted literal, and reusing single
    # quotes inside a single-quoted f-string is a SyntaxError before Python 3.12
    # (PEP 701 relaxed it). This ran as Python 3.11 and produced an EMPTY delta, so the
    # correction silently became `date -d " seconds"` and the clock never moved - while
    # the script still reported "NEEDS CORRECTION" each round and asked for sudo again.
    delta=$(python3 -c "print('%+.6f' % -float('$skew'))")
    if [ -z "$delta" ]; then
        echo "ERROR: could not compute the correction delta from skew='$skew'" >&2
        exit 1
    fi
    echo
    echo "Round $round: applying ${delta}s on $HOST (sudo password may be requested)"

    # One ssh -t, so the single sudo authentication covers every step. chronyd is stopped
    # first because it disciplines the clock and will otherwise partially undo the step;
    # it is restarted afterwards so the robot is left as we found it.
    # Preferred path: a narrow NOPASSWD helper installed by
    # /etc/sudoers.d/tmr-clock. No password, no tty, so bringup can correct skew
    # unattended. It validates its own argument because it runs as root.
    if ssh -o BatchMode=yes "$HOST" "sudo -n /usr/local/sbin/tmr-set-clock $delta" >/dev/null 2>&1; then
        :
    else
        # Fallback: the original interactive path. sudo caches credentials PER TTY, so a
        # separate `sudo -v` does not help - the whole correction runs inside one `ssh -t`.
        ssh -t "$HOST" "sudo bash -c '
            systemctl stop chrony 2>/dev/null || systemctl stop chronyd 2>/dev/null || true
            date -s \"\$(date -d \"${delta} seconds\" --rfc-3339=ns)\" >/dev/null
            hwclock -w 2>/dev/null || true
            systemctl start chrony 2>/dev/null || systemctl start chronyd 2>/dev/null || true
        '" </dev/tty || { echo "ERROR: correction command failed" >&2; exit 1; }
    fi

    sleep 2
    skew=$(measure_skew) || { echo "ERROR: could not re-measure skew" >&2; exit 1; }
    report "$skew"
    within "$skew" && { echo; echo "Clock synced. Restart any running teleop nodes - a step"; \
                        echo "change in the clock upsets ROS timers."; exit 0; }
done

echo
echo "WARNING: still outside tolerance after $MAX_ROUNDS rounds." >&2
echo "Something on the robot is actively disciplining the clock. Check:" >&2
echo "  ssh $HOST 'chronyc tracking; chronyc sources'" >&2
exit 1
