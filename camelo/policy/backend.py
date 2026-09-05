"""Local and remote policy backends behind one interface.

``run_policy.py --backend local`` runs the adapter (and model) in-process;
``--backend remote`` sends observations to a policy server (H100) over a
websocket. The rest of the stack cannot tell the difference.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

from camelo.policy.base import ActionChunk, Obs, PolicyAdapter
from camelo.policy.wire import (
    DEFAULT_WIRE_IMAGES,
    WireImageSpec,
    pack_obs,
    pack_reset,
    unpack,
    unpack_chunk,
)

log = logging.getLogger(__name__)


class PolicyBackend(ABC):
    @abstractmethod
    def reset(self, task: str) -> None: ...

    @abstractmethod
    def infer(self, obs: Obs) -> ActionChunk: ...

    def close(self) -> None:  # noqa: B027 — optional hook, most backends need nothing
        pass

    def stats(self) -> dict:
        """Backend-side timing merged into the rollout stats (F-40)."""
        return {}


class LocalBackend(PolicyBackend):
    def __init__(self, adapter: PolicyAdapter):
        self.adapter = adapter

    def reset(self, task: str) -> None:
        self.adapter.reset(task)

    def infer(self, obs: Obs) -> ActionChunk:
        actions = self.adapter.infer(obs)
        return ActionChunk(t0=obs.t_sim, actions=actions, dt=self.adapter.chunk_dt)


class RemoteBackend(PolicyBackend):
    """Synchronous websocket client (runs inside the ROS-side process)."""

    def __init__(
        self,
        url: str,
        connect_timeout: float = 10.0,
        recv_timeout: float = 15.0,
        wire_images: WireImageSpec | None = None,
    ):
        from websockets.sync.client import connect

        self.url = url
        # How each camera is conditioned before the JPEG encode. The default
        # is `WireImageSpec()` — native resolution, quality 90 — so a run
        # that passes nothing sends byte-identical requests to every run on
        # record. See wire.py's header for why the lever exists.
        self.wire_images = DEFAULT_WIRE_IMAGES if wire_images is None else wire_images
        self._wire_images_logged = False
        # Bounded reply wait. Without it a server that is alive on the socket
        # but wedged in inference (CUDA stall, a forward that never returns)
        # blocks the ROS-side 20 Hz loop forever: the keep-alive holds the
        # arm, but the rollout never ends and no guard can fire (review
        # 2026-09-02, docs/realdata/16 §0.4). TimeoutError escapes infer()
        # like StaleImageError does and the runner's guarded finally
        # deactivates the controllers.
        self.recv_timeout = recv_timeout
        self._ws = connect(url, open_timeout=connect_timeout, max_size=None)
        self.last_rtt: float | None = None
        self.rtts: list[float] = []  # infer requests only
        self.server_infer_s: list[float] = []
        # Per-infer request size and pack time. Same list discipline as
        # `rtts`: `warm_up_backend` trims the warm-up's own sample off all
        # of them, so a first-request outlier never becomes the run's mean.
        self.wire_bytes: list[int] = []
        self.encode_s: list[float] = []

    def _request(self, data: bytes) -> dict:
        t0 = time.monotonic()
        self._ws.send(data)
        reply = unpack(self._ws.recv(timeout=self.recv_timeout))
        self.last_rtt = time.monotonic() - t0
        if reply.get("type") == "error":
            raise RuntimeError(f"policy server error: {reply.get('message')}")
        return reply

    def reset(self, task: str) -> None:
        self._request(pack_reset(task))

    def _log_wire_images_once(self, obs: Obs) -> None:
        """What this run actually puts on the wire, once, at the first infer.

        On the real robot that first infer is the pre-activation warm-up
        (`camelo.runner.real_arms.warm_up_backend`), so the extra baseline
        encode inside `describe()` is paid where nothing is active and
        nothing can be hurt by it.
        """
        if self._wire_images_logged:
            return
        self._wire_images_logged = True
        if not obs.images:
            return
        try:
            log.info("%s", self.wire_images.describe(obs.images))
        except Exception as exc:  # a log line must never end a rollout
            log.warning("could not describe the wire images: %s", exc)

    def infer(self, obs: Obs) -> ActionChunk:
        self._log_wire_images_once(obs)
        t_encode = time.monotonic()
        data = pack_obs(obs, self.wire_images)
        self.encode_s.append(time.monotonic() - t_encode)
        self.wire_bytes.append(len(data))
        reply = self._request(data)
        self.rtts.append(self.last_rtt)
        if "infer_s" in reply:  # older servers omit it
            self.server_infer_s.append(float(reply["infer_s"]))
        chunk = unpack_chunk(reply)
        # The server stamps t0 with the obs sim time it received; trust it,
        # but fall back to the request's own timestamp for older servers.
        if chunk.t0 == 0.0 and obs.t_sim != 0.0:
            chunk.t0 = obs.t_sim
        return chunk

    def stats(self) -> dict:
        """Client RTT split into model compute vs transport (F-40) — the
        number the H100-split decision needs once a real model is loaded."""
        if not self.rtts:
            return {}
        out = {
            "rtt_mean_s": sum(self.rtts) / len(self.rtts),
            "rtt_max_s": max(self.rtts),
        }
        if self.server_infer_s:
            server_mean = sum(self.server_infer_s) / len(self.server_infer_s)
            out["server_infer_mean_s"] = server_mean
            out["transport_mean_s"] = out["rtt_mean_s"] - server_mean
        if self.wire_bytes:
            # The other half of `transport_mean_s`: how many bytes bought it.
            # Without this a --wire-image-size run and a full-resolution one
            # differ only in a timing number, and nothing on record says by
            # how much the payload moved.
            out["wire_bytes_per_request"] = sum(self.wire_bytes) / len(self.wire_bytes)
        if self.encode_s:
            out["wire_encode_s"] = sum(self.encode_s) / len(self.encode_s)
        return out

    def close(self) -> None:
        import contextlib

        with contextlib.suppress(Exception):
            self._ws.close()
