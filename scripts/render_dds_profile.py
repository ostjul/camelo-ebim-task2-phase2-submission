#!/usr/bin/env python3
"""Derive camelo's Fast DDS profile from the station's rendered one.

    python3 scripts/render_dds_profile.py \
        --src /tmp/tmr_fastdds_laptop_1000.xml \
        --dst outputs/rig/fastdds_camelo.xml

Prints the destination path and a unified diff of what changed, so the
operator can see with their own eyes that the `interfaceWhiteList`,
`maxMessageSize` and the participant profile came across untouched and only
the two socket-buffer lines are new. Reads the source, writes the
destination, touches nothing else — no site file is modified.

Why: the station drops ~1.2 MB `Image` samples in the kernel's 208 KiB
socket receive queue (`UdpRcvbufErrors` == `UdpInErrors`, `IpReasmFails 0`,
measured 2026-09-02); the rendered profile sets no `receiveBufferSize`. Full
reasoning in `camelo/ros/dds_profile.py` and docs/setup/SETUP.md §7.3.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camelo.ros.dds_profile import (  # noqa: E402
    DEFAULT_RECEIVE_BUFFER_BYTES,
    DEFAULT_SEND_BUFFER_BYTES,
    DdsProfileError,
    render_camelo_profile,
    render_profile_text,
)

_MB = 1024 * 1024


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Copy the station's rendered Fast DDS profile, adding "
        "socket buffers big enough for whole camera samples.",
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("/tmp/tmr_fastdds_laptop_1000.xml"),
        help="the station's rendered profile (read-only; default: %(default)s)",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("outputs/rig/fastdds_camelo.xml"),
        help="where to write camelo's copy (default: %(default)s)",
    )
    parser.add_argument(
        "--rx-mb",
        type=float,
        default=DEFAULT_RECEIVE_BUFFER_BYTES / _MB,
        help="receiveBufferSize in MiB (default: %(default)s)",
    )
    parser.add_argument(
        "--tx-mb",
        type=float,
        default=DEFAULT_SEND_BUFFER_BYTES / _MB,
        help="sendBufferSize in MiB (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    receive_bytes = int(round(args.rx_mb * _MB))
    send_bytes = int(round(args.tx_mb * _MB))

    try:
        before = args.src.read_text()
    except OSError as exc:
        print(f"render_dds_profile: cannot read {args.src}: {exc}", file=sys.stderr)
        return 2
    try:
        after, changes = render_profile_text(before, receive_bytes, send_bytes)
        dst = render_camelo_profile(args.src, args.dst, receive_bytes, send_bytes)
    except DdsProfileError as exc:
        print(f"render_dds_profile: refusing to render: {exc}", file=sys.stderr)
        return 2

    print(f"src {args.src}")
    print(f"dst {dst}")
    for change in changes:
        verb = "replaced in" if change.replaced else "inserted into"
        print(
            f"{verb} transport_descriptor {change.transport_id!r}: "
            f"sendBufferSize={change.send_buffer_bytes} "
            f"({change.send_buffer_bytes / _MB:g} MiB), "
            f"receiveBufferSize={change.receive_buffer_bytes} "
            f"({change.receive_buffer_bytes / _MB:g} MiB)"
        )
    diff = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=str(args.src),
        tofile=str(dst),
        n=2,
    )
    sys.stdout.writelines(diff)
    print(
        "everything else is byte-identical (interfaceWhiteList, maxMessageSize, "
        "useBuiltinTransports, participant profile, discovery)"
    )
    print(f"use it with:  export FASTRTPS_DEFAULT_PROFILES_FILE={dst.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
