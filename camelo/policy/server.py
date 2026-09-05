"""Websocket policy server: runs the adapter + model near the GPU.

Start on the H100 (or GB10 for small models):

    python scripts/serve_policy.py --adapter pi0 --port 8765

One client at a time (the episode runner); protocol in wire.py. No ROS
imports anywhere on this side.
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback

from camelo.policy.base import ActionChunk, PolicyAdapter
from camelo.policy.wire import pack_chunk, pack_error, pack_ok, unpack, unpack_obs

log = logging.getLogger("camelo.policy.server")


def seed_everything(seed: int) -> None:
    """Pin random/numpy/torch RNG state (CPU + all CUDA devices).

    Called once at server start and again on every `reset` request
    (docs/realdata/15_RIG_WINDOW_RUNBOOK.md §1.2/§2.4): the same seed must
    put every rollout's first forward pass in the same RNG state, whatever
    stochastic head the adapter runs (e.g. flow-matching noise). torch is
    imported lazily and is optional — this module has no ROS imports and
    must stay importable with no torch installed (camelo/__init__.py
    layering), and a torch-free adapter (e.g. --adapter dummy) still gets
    `random`/`numpy` seeded.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


async def _handle(ws, adapter: PolicyAdapter, seed: int | None = None):
    log.info("client connected: %s", ws.remote_address)
    async for message in ws:
        try:
            payload = unpack(message)
            kind = payload.get("type")
            if kind == "reset":
                if seed is not None:
                    # Pinned BEFORE adapter.reset() so the reset call
                    # itself — and the rollout's first forward pass right
                    # after — see the same RNG state every time.
                    seed_everything(seed)
                    log.info("seed=%d pinned at reset", seed)
                adapter.reset(payload.get("task", ""))
                await ws.send(pack_ok(seed=seed))
            elif kind == "infer":
                obs = unpack_obs(payload)
                t0 = time.monotonic()
                actions = adapter.infer(obs)
                infer_s = time.monotonic() - t0
                log.info(
                    "infer t_sim=%.2f state=%s images=%s -> actions=%s infer_s=%.3f",
                    obs.t_sim,
                    tuple(obs.state.shape),
                    sorted(obs.images),
                    tuple(actions.shape),
                    infer_s,
                )
                # infer_s lets the client split RTT into transport vs model
                # compute — inseparable from the client side alone (F-40).
                await ws.send(
                    pack_chunk(
                        ActionChunk(t0=obs.t_sim, actions=actions, dt=adapter.chunk_dt),
                        infer_s=infer_s,
                    )
                )
            else:
                await ws.send(pack_error(f"unknown message type: {kind!r}"))
        except Exception as exc:  # keep serving after a bad request
            log.error("request failed: %s\n%s", exc, traceback.format_exc())
            await ws.send(pack_error(str(exc)))
    log.info("client disconnected")


async def serve(
    adapter: PolicyAdapter,
    host: str = "0.0.0.0",
    port: int = 8765,
    seed: int | None = None,
):
    import websockets

    if seed is not None:
        seed_everything(seed)
        log.info("seed=%d pinned at server start", seed)

    async with websockets.serve(
        lambda ws: _handle(ws, adapter, seed=seed), host, port, max_size=None
    ):
        log.info("policy server (%s) listening on ws://%s:%d", adapter.name, host, port)
        await asyncio.Future()


def main(
    adapter: PolicyAdapter,
    host: str = "0.0.0.0",
    port: int = 8765,
    seed: int | None = None,
):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    asyncio.run(serve(adapter, host, port, seed=seed))
