"""rclpy session boilerplate shared by all ROS entry points."""

from __future__ import annotations

import contextlib
import logging
import threading
import time

log = logging.getLogger(__name__)


#: Why this module refuses rclpy's own SIGINT handler. MEASURED on the rig
#: 2026-09-02 (T1(b), the both-arms sine run — NOT the T1b joint-1 direction
#: probe — on native RoboStack Humble on ebimHP): the operator pressed
#: Ctrl+C, and BEFORE any Python code saw a KeyboardInterrupt, rclpy's signal
#: handler had already called `rcl_shutdown()` on the context. The spin thread
#: died with "failed to initialize wait set: the given context is not valid",
#: and the caller's teardown then raised on its first publish ("publisher's
#: context is invalid"), skipping the rest of it — the CSV, the verdict, and
#: (with `--activate-arms`) the DEACTIVATION. That last one is not a lost
#: file: a controller left ACTIVE with a dead publisher takes the whole arm
#: launch down 2.0 s later. Every teardown this repo has — `RealArmSession.
#: shutdown()`, `sine_probe_real.py` — assumes it can still talk to ROS while
#: unwinding, and that assumption is only true if SIGINT arrives as a plain
#: KeyboardInterrupt in the main thread.
SIGNAL_HANDLERS_OFF = "no-signal-handlers"
SIGNAL_HANDLERS_DEFAULT = "default-signal-handlers"


def init_rclpy(rclpy=None) -> str:
    """`rclpy.init()` WITHOUT rclpy installing its own SIGINT handler.

    `signal_handler_options=SignalHandlerOptions.NO` exists in Humble and
    Jazzy; on anything that does not take it, this falls back to the default
    and says so LOUDLY rather than failing — but the teardown guarantees
    above are then only as good as the caller's own error handling.

    `rclpy` is injectable so the choice can be unit-tested with no ROS: the
    one thing that must never regress here is the keyword actually reaching
    `init`, and an import-only check would pass on a call that dropped it.
    """
    if rclpy is None:
        import rclpy
    try:
        from rclpy.signals import SignalHandlerOptions

        options = SignalHandlerOptions.NO
    except (ImportError, AttributeError):
        options = None
    if options is not None:
        try:
            rclpy.init(signal_handler_options=options)
            log.info(
                "rclpy initialised with signal_handler_options=NO: SIGINT is a "
                "plain KeyboardInterrupt and the context stays valid through "
                "the teardown (deactivate, safe_stop, write the evidence)"
            )
            return SIGNAL_HANDLERS_OFF
        except TypeError:
            log.warning(
                "this rclpy's init() does not take signal_handler_options; "
                "falling back to its own SIGINT handler"
            )
    log.warning(
        "rclpy owns SIGINT: Ctrl+C calls rcl_shutdown() BEFORE any Python "
        "teardown runs, so publishes and service calls in a finally block "
        "will raise on an invalid context (rig 2026-09-02, T1(b)). On the real "
        "robot that means a controller can stay ACTIVE with a dead publisher"
    )
    rclpy.init()
    return SIGNAL_HANDLERS_DEFAULT


def shutdown_executor(executor, timeout_sec: float = 2.0, grace_sec: float = 0.5) -> list[str]:
    """Shut an rclpy executor down and NAME whatever refuses to stop.

    `Executor.shutdown(timeout_sec)` bounds only rclpy's own work tracker.
    A `MultiThreadedExecutor` dispatches callbacks to a
    `concurrent.futures.ThreadPoolExecutor` whose worker threads are
    NON-daemon, and `concurrent.futures.thread._python_exit` joins every one
    of them at interpreter shutdown with NO timeout. So a callback wedged in
    a blocking call cannot be escaped by any timeout here or by
    `daemon=True` on the spin thread: it hangs interpreter exit itself,
    after the program has finished and printed everything.

    MEASURED on the rig 2026-09-02 (`t3_checkobs_140915`): check_obs printed
    its complete report and then sat 6+ minutes in exit; the Ctrl+C
    traceback was `threading._shutdown -> atexit_call -> _python_exit ->
    t.join()`. The wedge was a camera drain callback inside
    `multiprocessing.Queue.get_nowait()` (see `CameraWorkers.close`, fixed
    there). This function cannot unwedge such a thread — nothing can — so it
    does the two things that are possible: cancel work that has not started
    yet, and log which threads are still running, so the next occurrence is
    diagnosable from the log rather than from a bare Ctrl+C. Entry points
    must still `os._exit` after cleanup (see `scripts/run_policy.py`).

    Returns the names of the pool threads still alive after `grace_sec`.
    """
    ok = True
    with contextlib.suppress(Exception):
        ok = executor.shutdown(timeout_sec=timeout_sec) is not False
    if not ok:
        log.warning(
            "executor.shutdown(%.1fs) reported unfinished work — a callback "
            "is still running", timeout_sec,
        )
    pool = getattr(executor, "_executor", None)
    if pool is None:
        return []
    try:
        pool.shutdown(wait=False, cancel_futures=True)
    except TypeError:  # pragma: no cover - Python < 3.9 has no cancel_futures
        with contextlib.suppress(Exception):
            pool.shutdown(wait=False)
    except Exception:  # pragma: no cover - already-shutdown pool
        pass
    threads = list(getattr(pool, "_threads", ()) or ())
    deadline = time.monotonic() + max(0.0, grace_sec)
    for worker in threads:  # idle workers exit on the sentinel; give them a beat
        with contextlib.suppress(Exception):
            worker.join(timeout=max(0.0, deadline - time.monotonic()))
    stuck = [t.name for t in threads if t.is_alive()]
    if stuck:
        log.warning(
            "%d executor callback thread(s) still running after shutdown "
            "(%s) — they are NON-daemon, so interpreter exit will block on "
            "them; the entry point must os._exit", len(stuck), ", ".join(stuck),
        )
    return stuck


@contextlib.contextmanager
def ros_session(
    node_name: str,
    verify_contract: bool = True,
    num_threads: int = 6,
    world: str | None = None,
):
    """Init rclpy, spin a node in a background thread, clean up on exit.

    Spinning in the background keeps subscriptions, timers (pedal repeat),
    and service futures alive while the caller's thread blocks on policy
    inference. The executor is multi-threaded so timers (pedal repeat,
    camera drain) and service futures never queue behind a slow callback.
    Cameras are not subscribed in this process at all — they live in
    per-camera subscriber subprocesses (camera_workers.py, F-30).

    ``world`` selects the topic map (``sim`` / ``real``). Real-robot runs
    skip the benchmark drift check — there is no topics.yaml on the station.
    """
    import rclpy
    from rclpy.executors import MultiThreadedExecutor

    from camelo.contracts import topics_for

    topics = topics_for(world)
    if verify_contract and topics.verify_against_benchmark:
        from camelo.contracts import verify_topics

        verify_topics()  # fail loudly before touching the ROS graph

    init_rclpy(rclpy)
    node = rclpy.create_node(node_name)
    executor = MultiThreadedExecutor(num_threads=num_threads)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True, name="ros-spin")
    thread.start()
    try:
        yield node
    finally:
        # Teardown order matters (DGX_FINDINGS.md F-16): stop the spin, JOIN
        # the thread, and only then destroy the node — destroying it while
        # the executor thread still spins aborts with a core dump on exit.
        # rclpy.shutdown() has been observed to hang forever after a finished
        # rollout (same hang class as F-32b); bound it and let entry points
        # os._exit after cleanup.
        shutdown_executor(executor, timeout_sec=2.0)
        thread.join(timeout=3.0)
        with contextlib.suppress(Exception):
            executor.remove_node(node)
            node.destroy_node()

        def _shutdown():
            # Exactly once: with signal handlers off nothing else shuts the
            # context down, and `ok()` keeps this a no-op in the fallback
            # case where rclpy's own handler already did it.
            with contextlib.suppress(Exception):
                if rclpy.ok():
                    rclpy.shutdown()

        stopper = threading.Thread(target=_shutdown, daemon=True, name="rclpy-shutdown")
        stopper.start()
        stopper.join(timeout=2.0)
        if stopper.is_alive():
            log.warning("rclpy.shutdown() did not finish within 2s; abandoning")
