"""Per-camera subscriber subprocesses — one process, one subscription each.

rclpy image ingest is per-PROCESS GIL-bound: one process tops out around
~2 Hz for a single camera and ~0.8 Hz for three, while three separate
processes ingest the full ~5.6 Hz the wire carries (DGX_FINDINGS.md F-30,
decision: all cameras, minimal overhead). So each camera gets its own
spawned worker process with its own rclpy context and single image
subscription; decoded frames stream back over a latest-wins queue and the
main process drains them on a timer.

Workers inherit the DDS env (ROS_DOMAIN_ID / RMW / UDPv4 forcing) through
the environment. Lifecycle (F-32/F-32b): daemon=True only reaps workers on
a NORMAL interpreter exit, so each worker also arms PR_SET_PDEATHSIG and
polls getppid() to die with its parent after a SIGKILL/OOM/crash — an
orphan keeps a live image subscription and steals the very bandwidth this
module exists to protect. Callers must still close() the owning
ObsCollector on the normal path.

Teardown order (MEASURED on the rig 2026-09-02, native RoboStack Humble,
`scripts/check_obs.py --world real --seconds 10 --cameras head`): a
10.4 Hz observation run completed cleanly, but the worker printed a
traceback at exit — `RCLError: failed to initialize wait set: the given
context is not valid, either rcl_init() was not called or rcl_shutdown()
was called`, from `rclpy/executors.py spin_once`. The worker's spin loop
must exit on its own — on the stop flag (parent death, checked every
1.0 s), on KeyboardInterrupt/ExternalShutdownException, or on rclpy.ok()
going false — BEFORE anything destroys the node or shuts the context
down, never the reverse. `_camera_worker` therefore: (1) inits via
`camelo.ros.session.init_rclpy` (SignalHandlerOptions.NO) so rclpy never
owns SIGINT/SIGTERM itself and can't race this loop with its own
`rcl_shutdown()`; (2) spins with a bounded `rclpy.spin_once(node,
timeout_sec=...)` inside `while not stop and rclpy.ok():`, never
`rclpy.spin(node)`; (3) destroys the node and only then shuts rclpy down,
in that order, in a `finally`; (4) still treats an RCLError that lands
during that window (e.g. the SignalHandlerOptions fallback distro) as a
clean exit rather than a crash, by matching its message.

Per-frame cost (MEASURED on the rig 2026-09-02, docs/realdata/16 R-45).
Three things were making each worker pay for frames nobody could use, and
all three are fixed here:

* **One subscription per camera, not two.** This module used to subscribe
  TWICE to the same topic — SENSOR_QOS (BEST_EFFORT/VOLATILE) *and*
  SYSTEM_QOS (RELIABLE/VOLATILE) — so a publisher matching both delivered
  every frame twice and the worker decoded and pickled it twice. See
  ``_camera_worker`` for why one BEST_EFFORT reader is sufficient.
* **A pre-decode rate limit.** ``image_msg_to_array`` is the expensive
  step and the parent drains at ~18 Hz; decoding a 30 fps wire in full
  spends work on frames the parent will never see. ``on_image`` therefore
  skips the decode for frames arriving ahead of a send deadline that
  advances by ``MIN_SEND_INTERVAL_S`` (default 1/25 s: a 30 fps wire is
  decoded at <= 25 Hz, a 15 fps one is untouched). It is a deadline, not
  "time since the last send" — see ``on_image`` for why the obvious form
  would have delivered 15 Hz.
* **Full queue drops the NEW frame.** The old ``_put_latest`` evicted by
  ``get_nowait()``-ing the old frame out — on a ``multiprocessing.Queue``
  that makes the WORKER unpickle a whole image just to discard it, on the
  hot path, to make room. Dropping the newcomer costs nothing and the
  parent's drain is latest-wins anyway.

None of this measures the wire rate — that is what the ``recv`` counter
below is for. ``frames``/``sim Hz`` in ``check_obs`` count parent drain
ticks and the stamp-gap median aliases to the drain period, so before
these counters existed nothing inside our own participant could say how
many frames the camera actually put on the wire (AGENTS.md: "``ros2 topic
hz`` as the rate" measures the wire but not our ingest — this is the
missing other half).
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing
import os
import queue as queue_mod
import signal
import time
from typing import NamedTuple

log = logging.getLogger(__name__)

#: Minimum wall gap between two frames leaving a worker. 1/25 s: below the
#: 30 fps the rig's head camera carries (so its decode load drops ~17 %)
#: and comfortably above the ~18 Hz the parent actually drains, so the
#: parent never starves for a frame this limit withheld. Override per
#: instance (``CameraWorkers(..., min_send_interval_s=...)``) or process
#: (``CAMELO_CAMERA_MIN_SEND_INTERVAL_S``); 0 disables the limit.
MIN_SEND_INTERVAL_S = 1.0 / 25.0
MIN_SEND_INTERVAL_ENV = "CAMELO_CAMERA_MIN_SEND_INTERVAL_S"


class WireCounts(NamedTuple):
    """What one worker did with the frames DDS handed it, cumulative since
    the worker started. ``recv`` is the only number in this repo measured
    inside our own participant that reflects the publisher's rate; the
    others say what we chose to do with it.

    recv         image callbacks entered (frames delivered to this process)
    decoded      ``image_msg_to_array`` calls that produced an array
    shipped      frames put on the queue for the parent
    dropped_full queue was full — the NEW frame was discarded
    """

    recv: int
    decoded: int
    shipped: int
    dropped_full: int


class CameraFrame(NamedTuple):
    """One decoded frame plus the timing the staleness instrument needs
    (camelo.control.image_age): the message's own sim-time stamp and the
    wall time it was decoded. ``t_stamp`` is None for an unstamped message
    (header.stamp == 0), never 0.0 — an age against sim-s 0 is not a
    measurement.

    ``counts`` rides along so the wire counters reach the parent without a
    second queue or a second pickle: four ints beside a decoded image cost
    nothing. It is None for a frame built by anything that does not track
    them (tests, the eval recorder's fakes)."""

    array: object  # HxWx3 uint8 RGB numpy array
    t_stamp: float | None
    t_wall: float
    counts: WireCounts | None = None


def resolve_min_send_interval(explicit: float | None = None) -> float:
    """Per-instance value, else the env override, else the default.

    A malformed or negative env value is ignored with a warning rather
    than silently disabling the limit."""
    if explicit is not None:
        return max(0.0, float(explicit))
    raw = os.environ.get(MIN_SEND_INTERVAL_ENV)
    if raw is None or not raw.strip():
        return MIN_SEND_INTERVAL_S
    try:
        value = float(raw)
    except ValueError:
        log.warning("%s=%r is not a number — using %.4f s", MIN_SEND_INTERVAL_ENV, raw,
                    MIN_SEND_INTERVAL_S)
        return MIN_SEND_INTERVAL_S
    if value < 0.0:
        log.warning("%s=%r is negative — using %.4f s", MIN_SEND_INTERVAL_ENV, raw,
                    MIN_SEND_INTERVAL_S)
        return MIN_SEND_INTERVAL_S
    return value


def _put_or_drop(frames, item) -> bool:
    """Non-blocking put; on Full the NEW item is dropped. Returns whether
    it was shipped.

    Deliberately NOT the old evict-the-oldest behaviour: on a
    ``multiprocessing.Queue`` a worker-side ``get_nowait()`` makes this
    process unpickle a full image just to throw it away, so "make room"
    cost more than the frame it was making room for. The parent's drain is
    latest-wins (``CameraWorkers.poll``), so the freshest frame still wins
    — it is simply the next one rather than this one, PROVIDED the parent
    is still draining at all.

    That proviso is the trade-off this makes explicit: while the queue
    stays full, EVERY new frame is dropped, not just the one that lost the
    race — nothing here reaches back in to replace a stale item with a
    fresher one. So during a parent-side drain stall (a blocked
    ``poll()``, a busy control loop, anything that stops calling
    ``get_nowait()``), the frame the parent eventually sees can be as old
    as the whole stall, however long that runs. The evicted old behaviour
    bounded staleness to one wire period (~1 camera frame), at the cost of
    unpickling a frame on every send just to discard it; this trades that
    bound away for the cheaper steady-state path. ``dropped_full`` in
    ``WireCounts`` is the signal that this trade is being paid for at any
    given moment — a stall shows up there as a run of drops, not as a
    single lost frame."""
    try:
        frames.put_nowait(item)
        return True
    except queue_mod.Full:
        return False


def _stamp_seconds(msg) -> float | None:
    stamp = msg.header.stamp
    t = float(stamp.sec) + float(stamp.nanosec) * 1e-9
    return t if t > 0.0 else None


def _die_with_parent() -> None:
    """Ask the kernel to SIGTERM this process when its parent dies (F-32)."""
    with contextlib.suppress(Exception):  # non-glibc: the getppid poll covers it
        import ctypes

        PR_SET_PDEATHSIG = 1
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGTERM)


def _camera_worker(
    key: str,
    topic: str,
    frames: multiprocessing.Queue,
    min_send_interval_s: float | None = None,
) -> None:
    """Child process: one rclpy context, one image subscription, decode, ship."""
    _die_with_parent()
    # Frames are disposable — never let the queue's feeder thread block this
    # process's exit waiting on a parent that no longer drains (F-32b).
    frames.cancel_join_thread()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    import rclpy
    from rclpy.executors import ExternalShutdownException
    from sensor_msgs.msg import Image

    from camelo.contracts import image_msg_to_array
    from camelo.ros.qos import SENSOR_QOS
    from camelo.ros.session import init_rclpy

    try:
        from rclpy.exceptions import RCLError
    except ImportError:  # pragma: no cover - distro layout drift
        RCLError = Exception

    # Use the shared init helper (signal_handler_options=NO) so rclpy never
    # installs its own SIGINT/SIGTERM handler here. A bare rclpy.init() was
    # MEASURED on the rig 2026-09-02 to race this worker's own spin loop:
    # terminate() (or a terminal SIGINT reaching the whole process group)
    # let rclpy's handler call rcl_shutdown() on the context WHILE the loop
    # below was still spinning, and the next wait-set creation raised
    # "failed to initialize wait set: the given context is not valid".
    # With signal handling off, a plain SIGTERM just kills the process (no
    # Python runs, no traceback) and a plain SIGINT still raises
    # KeyboardInterrupt in this thread, which the loop below treats as a
    # clean stop — same story as session.py's SIGNAL_HANDLERS_OFF.
    init_rclpy(rclpy)
    node = rclpy.create_node(f"camelo_camera_{key}")
    parent_pid = os.getppid()
    decode_errors = 0
    stop = {"requested": False}
    min_interval = resolve_min_send_interval(min_send_interval_s)
    # Cumulative since this worker started, never reset: the 3 s report
    # below differences them for its per-interval rates, and the parent
    # differences them across whatever window it cares about. A counter
    # that resets cannot be differenced by two readers at once.
    counts = {"recv": 0, "decoded": 0, "shipped": 0, "dropped_full": 0}
    # Monotonic time the next frame is allowed to go out. A DEADLINE that
    # advances by min_interval, NOT "now + min_interval" measured from the
    # last send — the difference decides the delivered rate and the naive
    # form is a trap. With a 40 ms limit (25 Hz) and a 33.3 ms wire (30 fps),
    # "skip anything younger than 40 ms since the last send" passes every
    # SECOND frame and delivers 15 Hz — BELOW the ~18 Hz the parent drains,
    # so a limit meant to cut wasted decodes would have cut delivered frames
    # instead. Accumulating the deadline lets the sequence 33/67/100/133 ms
    # keep 5 of every 6 frames and averages the promised 25 Hz. The clamp
    # below stops an idle period from banking credit for a burst. None
    # until the first frame: nothing is owed before anything was sent.
    next_at = {"t": None}

    def on_image(msg):
        nonlocal decode_errors
        counts["recv"] += 1
        now = time.monotonic()
        if min_interval > 0.0 and next_at["t"] is not None and now < next_at["t"]:
            # Return BEFORE image_msg_to_array: the decode is the cost this
            # limit exists to avoid, so the check cannot live after it.
            return
        try:
            array = image_msg_to_array(msg)
        except Exception:
            decode_errors += 1
            if decode_errors <= 3:
                log.exception(
                    "camera %s decode failed topic=%s enc=%s %dx%d",
                    key, topic, getattr(msg, "encoding", "?"),
                    getattr(msg, "width", 0), getattr(msg, "height", 0),
                )
            return
        counts["decoded"] += 1
        # Counted optimistically so the snapshot the frame CARRIES includes
        # itself — a frame that says "shipped=N" is the Nth. Corrected below
        # if the queue turns out to be full (that frame is discarded anyway,
        # counters and all, so only this process's totals need the rollback).
        counts["shipped"] += 1
        frame = CameraFrame(array, _stamp_seconds(msg), now, WireCounts(**counts))
        if not _put_or_drop(frames, frame):
            counts["shipped"] -= 1
            counts["dropped_full"] += 1
        # Advance on the ATTEMPT, not only on a successful ship. The parent
        # drains slower than this limit allows (~18 Hz vs 25), so Full is a
        # normal event; advancing only on success would let every drop
        # license an immediate full-rate decode and the limit would leak
        # away exactly when the queue is busiest.
        base = now if next_at["t"] is None else next_at["t"]
        next_at["t"] = max(base + min_interval, now)

    def check_parent():
        # Flag the stop and let the loop below exit on its own next check —
        # never shut the context down from inside a callback the spin loop
        # is still driving. Calling rclpy.shutdown() here directly would be
        # a self-inflicted version of the exact race this function exists
        # to avoid: the loop's next wait-set creation would raise.
        if os.getppid() != parent_pid:
            stop["requested"] = True

    seen_pubs = {"n": -1}

    def report_publishers():
        infos = node.get_publishers_info_by_topic(topic)
        n = len(infos)
        if n == seen_pubs["n"]:
            return
        seen_pubs["n"] = n
        if not infos:
            log.warning("camera %s: no publishers on %s yet", key, topic)
            return
        for info in infos:
            qos = info.qos_profile
            log.info(
                "camera %s topic=%s type=%s reliability=%s durability=%s",
                key, topic, info.topic_type,
                qos.reliability, qos.durability,
            )

    last_report = {"t": time.monotonic(), "counts": dict(counts)}

    def report_wire():
        """Per-interval wire counters — the only measurement of the
        publisher's rate taken inside our own participant."""
        now = time.monotonic()
        elapsed = max(now - last_report["t"], 1e-6)
        prev = last_report["counts"]
        delta = {name: counts[name] - prev[name] for name in counts}
        last_report["t"] = now
        last_report["counts"] = dict(counts)
        log.info(
            "camera %s wire: recv=%.1f/s decoded=%.1f/s shipped=%.1f/s dropped_full=%d",
            key,
            delta["recv"] / elapsed,
            delta["decoded"] / elapsed,
            delta["shipped"] / elapsed,
            delta["dropped_full"],
        )

    def report():
        report_publishers()
        report_wire()

    # ONE subscription, BEST_EFFORT (SENSOR_QOS = BEST_EFFORT/VOLATILE).
    # There used to be a second, SYSTEM_QOS (RELIABLE/VOLATILE) one on the
    # same topic and callback, on the theory that a RELIABLE driver needed a
    # RELIABLE reader. It does not — DDS compatibility is one-way: a
    # BEST_EFFORT reader matches a RELIABLE writer (the writer merely
    # offers more than the reader requests), and a VOLATILE reader matches
    # a TRANSIENT_LOCAL writer. BOTH publisher shapes were measured on the
    # rig 2026-09-02 (docs/realdata/16 R-45) and both delivered to the
    # SENSOR_QOS reader. The only thing the duplicate reliably achieved was
    # matching twice wherever it matched at all, so every frame was
    # delivered, decoded and pickled twice. Note this was never the sim's
    # path: F-28/F-30's fix (bbcd4c7) subscribed ONCE with
    # qos_profile_sensor_data and reached ~wire rate on all three sim
    # cameras — the duplicate arrived later (2026-09-01, 36ab731) as a
    # real-rig defensive measure, so removing it is unconditional and no
    # world/topic-map branch is needed. `report_publishers` below still
    # prints each publisher's reliability/durability every 3 s, so a
    # genuine mismatch stays visible instead of becoming a silent no-op.
    node.create_subscription(Image, topic, on_image, SENSOR_QOS)
    node.create_timer(1.0, check_parent)
    node.create_timer(3.0, report)
    log.info(
        "camera worker %s subscribing to %s (min_send_interval=%.4fs)",
        key, topic, min_interval,
    )
    try:
        # Bounded spin_once, not rclpy.spin(node): the loop must be able to
        # notice the stop flag and exit on its own, BEFORE anything shuts
        # the context down — never the reverse.
        while not stop["requested"] and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=1.0)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # both are clean shutdowns — no traceback on success (F-33)
    except RCLError as exc:
        message = str(exc)
        if "context is not valid" in message or "rcl_shutdown" in message:
            # A shutdown landed between our last rclpy.ok() check and the
            # next wait-set creation (e.g. the SignalHandlerOptions
            # fallback path, where rclpy still owns SIGINT/SIGTERM). The
            # context is already gone either way — that is a normal exit,
            # not a bug, so log it quietly instead of a traceback.
            log.debug(
                "camera %s: spin loop saw the context shut down during "
                "teardown: %s", key, exc,
            )
        else:
            raise
    finally:
        with contextlib.suppress(Exception):
            node.destroy_node()
        with contextlib.suppress(Exception):
            rclpy.try_shutdown()


class CameraWorkers:
    """Owns one worker process + latest-wins queue per camera key."""

    #: Set by ``close()`` BEFORE it touches the workers, so no cross-process
    #: read can start once teardown has begun (see ``poll``/``close``). A
    #: class attribute, not just an ``__init__`` assignment, so an instance
    #: built without ``__init__`` (the unit tests do this to avoid spawning
    #: subprocesses) still reads False instead of raising.
    _closing = False

    def __init__(self, cameras: dict[str, str], min_send_interval_s: float | None = None):
        """cameras: {key: image_topic}.

        ``min_send_interval_s`` bounds each worker's DECODE rate (see
        ``resolve_min_send_interval``): None takes the env override or the
        1/25 s default, 0 disables the limit.
        """
        ctx = multiprocessing.get_context("spawn")  # never fork an rclpy parent
        self.queues: dict[str, multiprocessing.Queue] = {}
        self.processes = []
        self.min_send_interval_s = resolve_min_send_interval(min_send_interval_s)
        # Latest WireCounts seen per camera, refreshed by poll() from the
        # snapshot each frame carries. Empty until the first frame arrives.
        self.wire: dict[str, WireCounts] = {}
        # Cameras poll() has already logged a closed-queue teardown race
        # for, so a poller that keeps calling in during teardown gets one
        # debug line per camera instead of one per call.
        self._closed_queue_logged: set[str] = set()
        for key, topic in cameras.items():
            frames = ctx.Queue(maxsize=2)
            process = ctx.Process(
                target=_camera_worker,
                args=(key, topic, frames, self.min_send_interval_s),
                name=f"camelo-cam-{key}", daemon=True,
            )
            process.start()
            self.queues[key] = frames
            self.processes.append(process)
        if cameras:
            log.info("camera workers started: %s", cameras)

    def status(self) -> dict[str, str]:
        """Per-worker liveness for wait_for_obs diagnostics."""
        out = {}
        for process in self.processes:
            key = (process.name or "").removeprefix("camelo-cam-") or process.name
            if process.is_alive():
                out[key] = "alive"
            else:
                out[key] = f"dead exit={process.exitcode}"
        return out

    def poll(self) -> dict[str, CameraFrame]:
        """Latest frame per camera since the last poll (may be empty).

        A camera whose queue was closed out from under this call (``close()``
        racing a still-running poll during teardown) contributes nothing for
        that camera rather than raising: ``multiprocessing.Queue.get_nowait``
        can raise ``OSError``/``ValueError`` on a closed queue, or
        ``EOFError`` if the feeder pipe went away mid-read, depending on
        platform and how far teardown got — none of those mean a frame is
        available, so they are treated the same as ``queue.Empty`` here.

        Once ``close()`` has begun this returns empty WITHOUT touching a
        queue at all, and that guard is load-bearing rather than tidy.
        ``Queue.get_nowait()`` is not the non-blocking call its name
        promises: the "nowait" covers only the availability poll, and once
        bytes are visible it does a BLOCKING ``_recv_bytes()`` for the whole
        length-prefixed message. A worker SIGTERM'd mid-frame leaves a
        truncated message in the pipe, and because the PARENT holds that
        queue's own write fd for the life of the Queue the pipe never
        reaches EOF — so the read never returns and no exception is ever
        raised for the handler above to catch (MEASURED offline,
        ``tests/test_camera_workers_close_deadlock.py``). ``close()`` shuts
        that window from both ends: this flag stops a NEW read from
        starting, and dropping the parent's write end before terminating
        turns an already-blocked one into EOF.
        """
        if self._closing:
            return {}
        # __init__ sets this, but a few unit tests build a CameraWorkers via
        # __new__ (no subprocesses) and skip it — fall back rather than
        # raise on an attribute that is bookkeeping, not required state.
        closed_logged = self.__dict__.setdefault("_closed_queue_logged", set())
        latest: dict[str, CameraFrame] = {}
        for key, frames in self.queues.items():
            while True:
                try:
                    latest[key] = frames.get_nowait()
                except queue_mod.Empty:
                    break
                except (OSError, ValueError, EOFError) as exc:
                    if key not in closed_logged:
                        closed_logged.add(key)
                        log.debug(
                            "camera %s: queue closed during poll (%r) — "
                            "teardown race, treating as empty", key, exc,
                        )
                    break
        for key, frame in latest.items():
            wire = getattr(frame, "counts", None)
            if wire is not None:
                self.wire[key] = wire
        return latest

    def wire_counts(self) -> dict[str, WireCounts]:
        """Cumulative per-camera wire counters as of the last polled frame.

        They ride in on the frames themselves, so a camera that has shipped
        nothing since the workers started is simply absent — and a camera
        that stops shipping freezes at its last value rather than reading 0.
        Difference two calls to get a rate over a window (``check_obs``
        does); the absolute totals count from worker start, not from here.
        """
        return dict(self.wire)

    def close(self) -> None:
        """Terminate AND reap the workers — the process cannot exit while a
        worker lives (F-32b), so this must run on every normal path.

        Also close the queues: a live Queue feeder thread in the parent keeps
        the interpreter alive after workers are gone (same hang class as F-32b).

        The queue work happens BEFORE the workers are signalled, and that
        order is the fix for a hang MEASURED on the rig 2026-09-02
        (t3_checkobs_140915): ``check_obs`` printed its whole report and
        then sat in interpreter exit for 6+ minutes, the Ctrl+C traceback
        landing in ``concurrent/futures/thread.py _python_exit -> t.join()``.
        A camera frame is megabytes and a pipe holds ~64 KiB, so a worker is
        essentially always mid-message; ``terminate()`` then left a
        TRUNCATED length-prefixed message in the pipe. The parent's
        ``_drain_cameras`` callback — already in flight on the executor's
        ThreadPoolExecutor — was inside ``Queue.get_nowait()``, which blocks
        in ``_recv_bytes()`` once bytes are visible, and never saw EOF
        because THIS process still holds that queue's write fd. rclpy's
        ``MultiThreadedExecutor`` runs callbacks on a ThreadPoolExecutor
        whose threads are non-daemon, and ``_python_exit`` joins them
        unconditionally, so one wedged drain callback makes the interpreter
        unable to exit at all. Hence, per queue and before any signal:
        ``cancel_join_thread()`` so nothing in the parent ever joins a
        feeder, and drop the parent's own write end so the dying worker's
        truncated message ends at EOF (``poll()`` catches that) instead of
        blocking forever. ``_closing`` above stops any further read.
        """
        self._closing = True
        for frames in self.queues.values():
            with contextlib.suppress(Exception):
                frames.cancel_join_thread()
            # The parent never puts, so its copy of the write end exists
            # only to keep the pipe from ever reaching EOF. Drop it while
            # the worker still holds its own — closing it after the worker
            # died would be too late for a read already blocked mid-message.
            writer = getattr(frames, "_writer", None)
            if writer is not None:
                with contextlib.suppress(Exception):
                    writer.close()
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)
        self.processes = []
        for frames in self.queues.values():
            with contextlib.suppress(Exception):
                frames.close()
            with contextlib.suppress(Exception):
                frames.cancel_join_thread()
        self.queues = {}
