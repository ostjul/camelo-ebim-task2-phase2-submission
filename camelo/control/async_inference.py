"""Run policy inference off the control thread.

Measured on the rig 2026-09-02 (docs/realdata/16 U-29,
``t6_act_20260902_172921``): ``run_policy.py --world real --backend remote
--rate 20 --replan-steps 8`` against an ACT server on EC2 through an ssh
tunnel. The round trip is 0.33-0.53 s (median 0.45 s — three JPEGs ~140 kB
per request over a slow uplink; the same client with tiny grey images
measures 0.15 s), while the replan interval is 8 x 0.05 s = 0.4 s. Because
``RemoteBackend.infer`` is synchronous, the 20 Hz loop stopped dead for the
whole round trip on every replan: **90 control ticks in 30 s — 3 Hz, not
20** — and the executor advanced 90 steps instead of 600. The arm was
covered in between only by the publisher's keep-alive republish.

This helper moves the call to a single worker thread that owns the backend:

* ``submit(obs)`` hands off a snapshot and returns immediately. **One
  request is in flight at a time**; a second ``submit`` while one is out is
  dropped and counted (``dropped_requests``), never queued — a queued
  request would be answered with a chunk stamped at an observation the
  robot has already driven past. The rollout loop asks only when
  ``in_flight`` is False, so that counter reading 0 is the normal, healthy
  report: nothing was thrown away.
* ``poll()`` returns the finished chunk or ``None``, and **re-raises the
  worker's exception in the control thread**. That is the safety property
  the synchronous path had for free: ``TimeoutError`` (the backend's
  bounded recv), ``ConnectionClosed`` and a server-side error still escape
  the rollout, so the runner's guarded ``finally`` deactivates the
  controllers instead of leaving an active arm on a dead policy.

The thread is a daemon so a wedged request can never hold the process open;
``close()`` joins it briefly and gives up rather than blocking teardown,
because teardown is exactly where the arm is being deactivated.

Deliberately generic: no ``camelo.policy`` import (AGENTS.md hard rule 3 —
``camelo.control`` is the bottom layer). It calls ``backend.infer(obs)``
and hands back whatever that returns.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any

log = logging.getLogger(__name__)

_STOP = object()  # sentinel: the only non-Obs item the request queue carries


class AsyncInference:
    """A single worker thread owning ``backend``; one request in flight."""

    def __init__(self, backend: Any, name: str = "camelo-inference"):
        self.backend = backend
        self.dropped_requests = 0
        self.last_infer_s: float | None = None  # wall RTT of the newest reply
        self._requests: queue.Queue = queue.Queue()
        self._results: queue.Queue = queue.Queue()
        # Written by the CONTROL thread only (submit sets it, poll clears
        # it), so it needs no lock: the worker never touches it.
        self._in_flight = False
        self._thread = threading.Thread(target=self._work, name=name, daemon=True)
        self._thread.start()

    @property
    def in_flight(self) -> bool:
        """True while a request is out and its reply has not been polled."""
        return self._in_flight

    def submit(self, obs: Any) -> bool:
        """Hand ``obs`` to the worker. False = dropped (one already out).

        Dropped, never queued: a queued request would come back stamped at
        an observation the robot has already driven past. A caller that
        checks ``in_flight`` first — the rollout loop does — never sees a
        False, which is why ``dropped_requests`` reads 0 on a healthy run;
        it is the counter for a caller that asks anyway.
        """
        if self._in_flight:
            self.dropped_requests += 1
            return False
        self._in_flight = True
        self._requests.put(obs)
        return True

    def poll(self) -> Any | None:
        """The finished chunk, or None. Re-raises the worker's exception."""
        try:
            kind, payload, infer_s = self._results.get_nowait()
        except queue.Empty:
            return None
        self._in_flight = False
        if kind == "error":
            log.error("policy inference failed after %.3f s: %r", infer_s, payload)
            raise payload
        self.last_infer_s = infer_s
        return payload

    def close(self, timeout_s: float = 1.0) -> None:
        """Ask the worker to stop; never block teardown on a wedged call."""
        self._requests.put(_STOP)
        self._thread.join(timeout=timeout_s)
        if self._thread.is_alive():
            # Daemon: it dies with the process. Said out loud because the
            # only way to get here is a request still on the wire, which is
            # worth seeing beside the deactivation log.
            log.warning("inference worker still busy at close; leaving it to the daemon exit")

    def stats(self) -> dict:
        return {"dropped_requests": self.dropped_requests}

    def _work(self) -> None:
        while True:
            item = self._requests.get()
            if item is _STOP:
                return
            t0 = time.monotonic()
            try:
                chunk = self.backend.infer(item)
            except Exception as exc:  # surfaced in the control thread at poll()
                self._results.put(("error", exc, time.monotonic() - t0))
                return  # the rollout is over; nothing will be submitted again
            self._results.put(("chunk", chunk, time.monotonic() - t0))
