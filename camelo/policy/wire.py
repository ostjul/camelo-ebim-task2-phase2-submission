"""msgpack wire format shared by RemoteBackend and the policy server.

Messages (msgpack maps, binary websocket frames):
  client -> server:  {"type": "reset", "task": str}
                     {"type": "infer", "t_sim": float, "state": f32 bytes,
                      "images": {key: jpeg bytes},
                      "rig": {group: f32 list} (optional — s27a15 only)}
  server -> client:  {"type": "ok", "seed": int (optional — echoes a
                      --seed pinned at this reset, absent when unseeded)}
                     {"type": "chunk", "t0": float, "dt": float, "horizon": int,
                      "actions": f32 bytes, "action_dim": int (optional; 20 when
                      absent, for pre-s27a15 peers), "infer_s": float (optional —
                      server-side model time, lets the client split RTT into
                      transport vs compute, F-40)}
                     {"type": "error", "message": str}

Images travel as JPEG (quality 90): ~3x100 KB per inference at 10 Hz is
comfortable on a LAN and keeps the GB10 client free of torch.

That default is a LAN assumption, and the Munich rig is not one: over the
ssh tunnel the three full-resolution JPEGs (720x1280 head + two 480x640
wrists, ~116 kB measured on `outputs/rig/t5/ep163_frames`) cost ~0.24 s of
the ~0.30-0.37 s round trip, which is longer than VLA-JEPA's own 7-row
chunk (0.35 s at 20 Hz). `WireImageSpec` is the opt-in lever for that: it
resizes each camera to the geometry the checkpoint's own preprocessor
would resize it to anyway, BEFORE the JPEG encode, and lets the quality
move. Default (`WireImageSpec()`) is byte-identical to every request ever
sent — no resize, quality 90 — so nothing on record changes.

**What the model actually sees does move a little**, and not because of
the resize. The resize itself agrees with the server's to 1/255
(`resize_for_wire`). What changes is WHERE the JPEG lands: today it is
applied at native resolution and the server's downscale then averages the
compression noise away, whereas a pre-resized frame carries its artifacts
at the model's own 224x224. MEASURED on `outputs/rig/t5/ep163_frames`
(head + wrist_right, frames 0/100/200), same quality 90: mean 1.2 counts,
max 36-42. `--wire-jpeg-quality 95` buys most of it back (mean 0.82, max
24) for 25 kB instead of 17 kB — still a quarter of the 95 kB the two
full-resolution cameras cost — and even quality 100 keeps a ~19-count
maximum, because the difference is the baseline's own noise, not the
shrunk frame's.

**Vector widths are declared, not assumed.** The sim contract is 37 in /
20 out; the Munich rig's `s27a15` layout is 27 in / 15 out, and PR #23's
GPU-box/robot-box split is exactly the deployment that needs it. So the
state width is checked against the set of layouts this repo speaks rather
than against `C.STATE_DIM` alone, and the chunk carries its own width —
a hardcoded 20 here silently reshapes a 15-dim chunk into garbage.
"""

from __future__ import annotations

import functools
import io
import logging
from dataclasses import dataclass, field

import msgpack
import numpy as np

from camelo import contracts as C
from camelo.policy.adapters import s27a15
from camelo.policy.base import ActionChunk, Obs

log = logging.getLogger(__name__)

JPEG_QUALITY = 90

#: State widths any adapter in this repo speaks: the sim's 37-dim recorder
#: vector, and the Munich rig's 27-dim vector. Anything else is a bug on the
#: sending side and must not be silently reshaped.
STATE_DIMS = (C.STATE_DIM, s27a15.STATE_DIM)
#: Chunk width when a peer predating `action_dim` sends none.
DEFAULT_ACTION_DIM = C.ACTION_DIM


def _encode_jpeg(image: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _decode_jpeg(data: bytes) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


# ---------------------------------------------------------------------------
# Client-side image conditioning (opt-in; the default below changes nothing)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=8)
def _torchvision_resize(height: int, width: int):
    """`torchvision.transforms.v2.Resize`, or None when torch is absent.

    Cached because building the transform per frame at 20 Hz is pointless,
    and because the import probe itself costs more than the resize.
    """
    try:
        from torchvision.transforms import v2
    except Exception:  # no torch on this box — the GB10 client's normal state
        return None
    return v2.Resize(size=[height, width])


def resize_for_wire(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """HxWx3 uint8 -> (H, W)x3 uint8, the SAME op the SERVER would apply.

    The server's `LeRobotAdapter._real_image_tensor` replays the
    checkpoint's own `dataset.image_transforms` Resize —
    `torchvision.transforms.v2.Resize(size=[H, W])`, i.e. bilinear with
    `antialias=True`, on the **uint8 CHW** tensor, before the /255 cast
    (`lerobot_generic.training_image_transforms`). So that is what this
    reproduces, and doing it here is only moving the same arithmetic to
    the other end of the tunnel: `v2.Resize` short-circuits to the identity
    when its input already has the requested shape, so a frame resized here
    passes through the server's transform untouched.

    Two implementations, in order of fidelity:

      * **torchvision, when it is installed** — the identical call, so the
        result is bit-identical to the server's (measured max |diff| = 0).
      * **PIL `Image.resize(..., Image.BILINEAR)`** otherwise, which is the
        realistic client path: the ROS-side box deliberately carries no
        torch (see this module's header). PIL's bilinear is area-weighted
        for downscales the same way `antialias=True` is, so the two agree
        closely but not exactly.

    MEASURED 2026-09-03, torch 2.14 / torchvision 0.29 / Pillow 12.3, on the
    corpus frames in `outputs/rig/t5/ep163_frames` (head 720x1280 and
    wrist_right 640x480, frames 0/100/200) resized to 224x224: PIL bilinear
    vs torchvision **max |diff| = 1/255, mean 0.0005**, and the 99.9th
    percentile of the per-pixel difference is 0. For contrast, PIL's
    *default* filter (BICUBIC — what the sim path's `_image_tensor` uses)
    lands at max 32/255 on the same frames, which is why this function does
    not just call `Image.resize(size)`.
    """
    height, width = int(size[0]), int(size[1])
    if image.shape[0] == height and image.shape[1] == width:
        return image
    transform = _torchvision_resize(height, width)
    if transform is not None:
        import torch

        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
        return transform(tensor).permute(1, 2, 0).numpy()
    from PIL import Image

    return np.asarray(
        Image.fromarray(image).resize((width, height), Image.BILINEAR), dtype=np.uint8
    )


@dataclass
class WireImageSpec:
    """How the REMOTE client conditions each camera before the JPEG encode.

    `WireImageSpec()` — no sizes, quality 90 — is exactly `_encode_jpeg`,
    so every request this repo has ever sent stays byte-identical. It only
    does something once an operator asks for it
    (`--wire-image-size` / `--wire-jpeg-quality`).

    `sizes` and `default_size` are **(H, W)**, numpy/torchvision order; the
    CLI's `WxH` spelling is converted by `parse_wire_image_size`. A camera
    with no entry and no `default_size` is sent at its native resolution.

    Sizing is the operator's declaration, not a checkpoint lookup: the
    client is on the other side of the tunnel from the weights and cannot
    read `train_config.json`. Pass the size the checkpoint's own Resize
    declares (VLA-JEPA: 224x224) — a *different* size is not caught here,
    it is silently re-resized by the server to whatever the checkpoint
    wants, from a frame that has already lost the pixels.
    """

    sizes: dict[str, tuple[int, int]] = field(default_factory=dict)
    default_size: tuple[int, int] | None = None
    quality: int = JPEG_QUALITY

    def is_identity(self) -> bool:
        """True when this spec encodes exactly what the default path does."""
        return not self.sizes and self.default_size is None and int(self.quality) == JPEG_QUALITY

    def size_for(self, camera: str) -> tuple[int, int] | None:
        return self.sizes.get(camera, self.default_size)

    def prepare(self, camera: str, image: np.ndarray) -> np.ndarray:
        """The array that goes into the JPEG encoder for `camera`."""
        target = self.size_for(camera)
        if target is None:
            return image
        return resize_for_wire(image, target)

    def encode(self, camera: str, image: np.ndarray) -> bytes:
        return _encode_jpeg(self.prepare(camera, image), quality=int(self.quality))

    def describe(self, images: dict[str, np.ndarray]) -> str:
        """One-shot startup line: per camera, native -> wire shape and bytes.

        The "before" number is today's wire form (native geometry, quality
        90), re-encoded here so the saving is measured rather than
        estimated. Called once per run, on an observation that is already
        being paid for (the pre-activation warm-up), so the extra encode
        costs nothing a rollout can feel.
        """
        rows = []
        for camera, image in sorted(images.items()):
            after = self.encode(camera, image)
            before = len(after) if self.is_identity() else len(_encode_jpeg(image))
            sent = self.prepare(camera, image)
            rows.append(
                f"{camera}: {image.shape[1]}x{image.shape[0]} -> "
                f"{sent.shape[1]}x{sent.shape[0]}, {before} -> {len(after)} B"
            )
        return f"wire images (jpeg q{int(self.quality)}): " + "; ".join(rows)


#: The historical wire behaviour: native resolution, quality 90.
DEFAULT_WIRE_IMAGES = WireImageSpec()


def parse_wire_image_size(text: str, cameras=None) -> WireImageSpec:
    """`"224x224"` or `"head=224x224,wrist_right=224x224"` -> a spec.

    Spelled **WxH** on the command line (the way an operator says
    "224x224" or "640x360"), stored (H, W). A bare `WxH` sets
    `default_size`, i.e. every camera; `name=WxH` entries set that camera
    only, and the two forms may be mixed (the bare one is the default for
    cameras with no entry of their own).

    `cameras` (optional) is the set of names a typo is checked against —
    an unknown camera here would otherwise be a flag that silently does
    nothing.
    """
    spec = WireImageSpec()
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        camera, _, size_text = item.rpartition("=")
        try:
            width_text, _, height_text = size_text.lower().partition("x")
            width, height = int(width_text), int(height_text)
        except ValueError:
            raise ValueError(
                f"--wire-image-size: {item!r} is not WxH or NAME=WxH "
                "(e.g. '224x224' or 'head=224x224,wrist_right=224x224')"
            ) from None
        if width <= 0 or height <= 0:
            raise ValueError(f"--wire-image-size: {item!r} must be positive")
        if not camera:
            spec.default_size = (height, width)
        else:
            if cameras is not None and camera not in cameras:
                raise ValueError(
                    f"--wire-image-size: unknown camera {camera!r}; valid: {sorted(cameras)}"
                )
            spec.sizes[camera] = (height, width)
    if spec.is_identity():
        raise ValueError(f"--wire-image-size: {text!r} declared no size")
    return spec


def _encode_rig_value(value):
    """One s27a15 rig group -> a msgpack-native scalar / bool / float list."""
    if isinstance(value, bool):
        return value
    array = np.asarray(value, dtype=np.float64)
    return float(array) if array.ndim == 0 else [float(x) for x in array.ravel()]


def pack_reset(task: str) -> bytes:
    return msgpack.packb({"type": "reset", "task": task})


def pack_obs(obs: Obs, images: WireImageSpec | None = None) -> bytes:
    """One inference request. `images` defaults to today's exact encoding."""
    spec = DEFAULT_WIRE_IMAGES if images is None else images
    payload = {
        "type": "infer",
        "t_sim": float(obs.t_sim),
        "state": np.asarray(obs.state, dtype=np.float32).tobytes(),
        "images": {k: spec.encode(k, v) for k, v in obs.images.items()},
    }
    if obs.image_t_sim:  # optional: per-image sim-time stamps (image_age)
        payload["image_t_sim"] = {k: float(v) for k, v in obs.image_t_sim.items()}
    if obs.rig is not None:
        # Named groups, so the far side packs the 27-dim vector with the
        # same code the local path uses (adapters/s27a15.py). Sent as plain
        # lists: msgpack has no ndarray type and the groups are 6-7 floats.
        payload["rig"] = {k: _encode_rig_value(v) for k, v in obs.rig.items()}
    return msgpack.packb(payload)


def unpack_obs(payload: dict) -> Obs:
    state = np.frombuffer(payload["state"], dtype=np.float32).copy()
    if state.shape not in [(d,) for d in STATE_DIMS]:
        raise ValueError(f"state must be one of {STATE_DIMS} dims, got {state.shape}")
    images = {k: _decode_jpeg(v) for k, v in payload["images"].items()}
    image_t_sim = {k: float(v) for k, v in (payload.get("image_t_sim") or {}).items()}
    return Obs(
        t_sim=payload["t_sim"],
        state=state,
        images=images,
        image_t_sim=image_t_sim,
        rig=payload.get("rig"),
    )


def pack_chunk(chunk: ActionChunk, infer_s: float | None = None) -> bytes:
    actions = np.asarray(chunk.actions, dtype=np.float32)
    payload = {
        "type": "chunk",
        "t0": float(chunk.t0),
        "dt": float(chunk.dt),
        "horizon": int(actions.shape[0]),
        "action_dim": int(actions.shape[1]),
        "actions": actions.tobytes(),
    }
    if infer_s is not None:
        payload["infer_s"] = float(infer_s)
    return msgpack.packb(payload)


def unpack_chunk(payload: dict) -> ActionChunk:
    actions = np.frombuffer(payload["actions"], dtype=np.float32).reshape(
        payload["horizon"], int(payload.get("action_dim", DEFAULT_ACTION_DIM))
    )
    return ActionChunk(t0=payload["t0"], actions=actions.copy(), dt=payload["dt"])


def pack_error(message: str) -> bytes:
    return msgpack.packb({"type": "error", "message": message})


def pack_ok(seed: int | None = None) -> bytes:
    payload = {"type": "ok"}
    if seed is not None:
        # Echoed so a client-side log can confirm the server actually
        # pinned the seed it was asked for. Unknown-key-tolerant peers
        # (RemoteBackend.reset() only checks payload["type"]) ignore it.
        payload["seed"] = int(seed)
    return msgpack.packb(payload)


def unpack(data: bytes) -> dict:
    return msgpack.unpackb(data)
