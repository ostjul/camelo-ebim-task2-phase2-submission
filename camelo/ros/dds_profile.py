"""Derive camelo's own Fast DDS profile from the station's rendered one.

Why this module exists (MEASURED on the station, 2026-09-02 14:25, read-only):

* `nstat -az` reported `UdpRcvbufErrors` 46,761,273 == `UdpInErrors`
  (cumulative since the 2026-08-29 boot) with `IpReasmFails 0` — every loss
  is a datagram dropped at the **socket receive queue**, not a failed IP
  reassembly.
* `net.core.rmem_default = 212992` (208 KiB) and the station's rendered
  profile (`/tmp/tmr_fastdds_laptop_<uid>.xml`, docs/realdata/16 R-04) sets
  **no** `receiveBufferSize`/`sendBufferSize` on its UDPv4 transport
  descriptor, so every participant on the box gets that 208 KiB queue.
* An 848x480x3 `Image` sample is ~1.2 MB — ~20 datagrams of `maxMessageSize`
  (65500 B) delivered as one burst. A 208 KiB queue holds ~3 of them, so the
  burst overflows and the **whole sample** is lost: the BEST_EFFORT reader
  path camelo uses (AGENTS.md hard rule 5) has no retransmission to fall back
  on, unlike a RELIABLE writer/reader pair.
* Symptom above our code: `camelo/ros/camera_workers.py` counts `recv`
  (samples handed up by DDS) at 10–15 msgs/s on 30 fps wrist topics whose
  minimum stamp gap is 1/30 s — the drops happen BELOW the callback, so no
  amount of Python-side work can recover them.

The fix is a bigger socket receive buffer, and the honest way to get one is a
profile of **our own**: the site file is rendered per session by
`~/teleoperation/station/configs/tmr_laptop_env.sh` and is not ours to edit
(AGENTS.md hard rule 1 in spirit — we join the site's graph, we do not modify
it). So this module copies the rendered file byte-for-byte and inserts two
elements per UDPv4 transport descriptor. Everything else — the
`interfaceWhiteList`, `maxMessageSize`, `useBuiltinTransports=false`, the
participant profile, discovery — is carried across untouched, because each of
those is load-bearing at the site and a "helpful" second change would make a
failed run un-attributable.

The kernel side needs nothing on this station: `net.core.rmem_max` is already
2147483647, far above the 16 MiB we ask for. On a box where `rmem_max` is
smaller than `receive_buffer_bytes`, the kernel silently clamps the
`setsockopt(SO_RCVBUF)` and the drops come back with a profile that looks
correct — raising it is a `sysctl` and therefore the **operator's** call, not
something this module or any camelo process does. See docs/setup/SETUP.md
§7.3.

Stdlib only, no rclpy: `make dds-profile` runs on the host, outside the
container and outside any ROS environment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: 16 MiB — ~13 whole 1.2 MB Image samples of headroom on the receive queue.
DEFAULT_RECEIVE_BUFFER_BYTES = 16 * 1024 * 1024
#: 4 MiB — camelo publishes joint commands, not images; this is slack, not need.
DEFAULT_SEND_BUFFER_BYTES = 4 * 1024 * 1024

_DESCRIPTOR_RE = re.compile(
    r"<transport_descriptor\b[^>]*>.*?</transport_descriptor>", re.DOTALL
)
_TYPE_RE = re.compile(r"<type>\s*([^<]*?)\s*</type>")
_ID_RE = re.compile(r"<transport_id>\s*([^<]*?)\s*</transport_id>")
#: `<maxMessageSize>` alone on its own line — the anchor we insert after.
_MAX_MSG_LINE_RE = re.compile(
    r"^([ \t]*)<maxMessageSize>[^<]*</maxMessageSize>[ \t]*\r?$", re.MULTILINE
)
_MAX_MSG_ANY_RE = re.compile(r"<maxMessageSize\b")
_BUFFER_ANY_RE = re.compile(r"<(sendBufferSize|receiveBufferSize)\b[^>]*>")
#: An existing buffer element alone on its own line — replaceable in place.
_BUFFER_LINE_RE = re.compile(
    r"^[ \t]*<(sendBufferSize|receiveBufferSize)>[^<]*</\1>[ \t]*\r?\n",
    re.MULTILINE,
)


class DdsProfileError(ValueError):
    """The source profile is not the shape docs/realdata/16 R-04 describes.

    Raised instead of guessing: a Fast DDS profile that silently fails to
    parse is applied as "no profile at all", which on this station means no
    `interfaceWhiteList` either — a much worse failure than refusing here.
    """


@dataclass(frozen=True)
class TransportChange:
    """One UDPv4 transport descriptor we rewrote, for the CLI's summary."""

    transport_id: str
    send_buffer_bytes: int
    receive_buffer_bytes: int
    #: True when the source already carried buffer elements we replaced.
    replaced: bool


def render_profile_text(
    text: str,
    receive_buffer_bytes: int = DEFAULT_RECEIVE_BUFFER_BYTES,
    send_buffer_bytes: int = DEFAULT_SEND_BUFFER_BYTES,
) -> tuple[str, list[TransportChange]]:
    """Return `(rendered_xml, changes)` for the profile in `text`.

    Pure string surgery, deliberately: `xml.etree` would round-trip the file
    through its own serializer and lose the site's comments, attribute
    quoting and indentation, making a `diff` against the rendered original
    unreadable — and that diff is the only way an operator can check we
    changed nothing but the buffers.
    """
    for name, value in (
        ("receive_buffer_bytes", receive_buffer_bytes),
        ("send_buffer_bytes", send_buffer_bytes),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise DdsProfileError(f"{name} must be a positive int, got {value!r}")

    eol = "\r\n" if "\r\n" in text else "\n"
    out: list[str] = []
    changes: list[TransportChange] = []
    cursor = 0
    for match in _DESCRIPTOR_RE.finditer(text):
        block = match.group(0)
        type_match = _TYPE_RE.search(block)
        if type_match is None or type_match.group(1) != "UDPv4":
            continue
        id_match = _ID_RE.search(block)
        transport_id = id_match.group(1) if id_match else "<unnamed>"
        rewritten, replaced = _rewrite_descriptor(
            block, transport_id, receive_buffer_bytes, send_buffer_bytes, eol
        )
        out.append(text[cursor : match.start()])
        out.append(rewritten)
        cursor = match.end()
        changes.append(
            TransportChange(
                transport_id=transport_id,
                send_buffer_bytes=send_buffer_bytes,
                receive_buffer_bytes=receive_buffer_bytes,
                replaced=replaced,
            )
        )
    if not changes:
        raise DdsProfileError(
            "no UDPv4 <transport_descriptor> found — this is not the station's "
            "rendered profile (docs/realdata/16 R-04 expects exactly one, with "
            "<type>UDPv4</type> and an <interfaceWhiteList>)"
        )
    out.append(text[cursor:])
    return "".join(out), changes


def _rewrite_descriptor(
    block: str,
    transport_id: str,
    receive_buffer_bytes: int,
    send_buffer_bytes: int,
    eol: str,
) -> tuple[str, bool]:
    """Insert (or replace) the two buffer elements in one descriptor block."""
    inline = len(_BUFFER_ANY_RE.findall(block))
    on_own_line = len(_BUFFER_LINE_RE.findall(block))
    if inline != on_own_line:
        raise DdsProfileError(
            f"transport_descriptor {transport_id!r} already carries a "
            "sendBufferSize/receiveBufferSize that is not alone on its line; "
            "replacing it would be ambiguous — edit the source by hand"
        )
    replaced = inline > 0
    if replaced:
        block = _BUFFER_LINE_RE.sub("", block)

    anchors = _MAX_MSG_LINE_RE.findall(block)
    total = len(_MAX_MSG_ANY_RE.findall(block))
    if total != 1 or len(anchors) != 1:
        raise DdsProfileError(
            f"transport_descriptor {transport_id!r} must contain exactly one "
            "<maxMessageSize> element, alone on its own line (found "
            f"{total} element(s), {len(anchors)} usable as an anchor); "
            "R-04 expects <maxMessageSize>65500</maxMessageSize>"
        )
    anchor = _MAX_MSG_LINE_RE.search(block)
    assert anchor is not None  # guarded by the count check above
    indent = anchor.group(1)
    newline = block.find("\n", anchor.end())
    insert_at = len(block) if newline == -1 else newline + 1
    prefix = "" if newline != -1 else eol
    addition = (
        f"{prefix}"
        f"{indent}<sendBufferSize>{send_buffer_bytes}</sendBufferSize>{eol}"
        f"{indent}<receiveBufferSize>{receive_buffer_bytes}</receiveBufferSize>{eol}"
    )
    return block[:insert_at] + addition + block[insert_at:], replaced


def render_camelo_profile(
    src: Path,
    dst: Path,
    receive_buffer_bytes: int = DEFAULT_RECEIVE_BUFFER_BYTES,
    send_buffer_bytes: int = DEFAULT_SEND_BUFFER_BYTES,
) -> Path:
    """Write `src` to `dst` with big socket buffers on every UDPv4 transport.

    `src` is the station's rendered profile (`/tmp/tmr_fastdds_laptop_<uid>.xml`)
    and is only ever read. Returns `dst`.
    """
    src = Path(src)
    dst = Path(dst)
    try:
        text = src.read_text()
    except FileNotFoundError as exc:
        raise DdsProfileError(
            f"source profile {src} does not exist — source the station env first "
            "(`set -a; source ~/teleoperation/station/configs/tmr_laptop_env.sh; "
            "set +a`), which renders it"
        ) from exc
    rendered, _ = render_profile_text(text, receive_buffer_bytes, send_buffer_bytes)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(rendered)
    return dst
