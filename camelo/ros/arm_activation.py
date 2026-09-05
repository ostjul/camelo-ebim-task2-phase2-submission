"""`joint_impedance_controller` activation, from the runner's OWN node.

MEASURED on the rig 2026-09-01. The companion's controller:

  * ingests the GELLO topic whether or not it is active;
  * **refuses** ``on_activate`` unless a valid sample arrived within
    ``max_gello_liveness_gap`` = 2.0 s, and captures ``q0`` (robot) and
    ``g0`` (command) at that instant;
  * maps ``q_goal = q0 + dir * (g - g0)``, rate-limited at 0.5 rad/s;
  * rejects samples stamped more than 0.5 s in the past by ITS clock;
  * calls ``rclcpp::shutdown()`` — the whole arm launch dies — if no valid
    sample arrives for 2.0 s while it is ACTIVE.

So activation is not a side task: the command stream has to be up BEFORE it,
and must not stop until after deactivation. That choreography lives in
`camelo/runner/real_arms.py`; this module is only the service plumbing.

**One participant.** The station's own `activate_arms.py` waits for BOTH
sides' services to be discovered before switching either, because a new DDS
participant appearing mid-session is a discovery burst that can fault a live
FCI loop. The same reasoning forbids shelling out to `ros2 service call` or
spinning up a second node here: every client below is created on the
runner's existing node, at startup, alongside the publisher whose stream the
controller is watching.

`controller_manager_msgs` is imported lazily so the module (and its response
parsing, which is where the interesting logic is) can be unit-tested with no
ROS installed.
"""

from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)

#: The controller this repo activates. Anything else on the arm is the
#: station's business.
CONTROLLER_NAME = "joint_impedance_controller"
ACTIVE_STATE = "active"
ARMS = ("left", "right")


def controller_state(response, controller: str = CONTROLLER_NAME) -> str | None:
    """State of ``controller`` in a ListControllers response, or None.

    None means "the controller manager does not list it at all", which is a
    different fault from "listed and inactive" — a typo in the name, or a
    controller_manager that has not loaded it, would otherwise look exactly
    like an operator who has not pressed the button yet.
    """
    for item in getattr(response, "controller", []) or []:
        if getattr(item, "name", None) == controller:
            return getattr(item, "state", None)
    return None


def parse_arm_states(responses: dict, controller: str = CONTROLLER_NAME) -> dict:
    """``{side: ListControllers response | None}`` -> ``{side: state | None}``."""
    return {
        side: (None if response is None else controller_state(response, controller))
        for side, response in responses.items()
    }


class ArmControllers:
    """Service clients for ``/<side>/controller_manager/*`` on ONE node."""

    def __init__(
        self,
        node,
        arms=ARMS,
        controller: str = CONTROLLER_NAME,
        call_timeout_s: float = 5.0,
    ):
        try:
            from controller_manager_msgs.srv import ListControllers, SwitchController
        except ModuleNotFoundError as exc:
            # MEASURED on the rig 2026-09-02 (t1r_120415, docs/realdata/16
            # R-38): this import is missing in BOTH the station's RoboStack
            # Humble pixi env and the camelo-ebim/ros:latest container, so
            # --activate-arms has never been runnable anywhere. Re-raise
            # with the fix so the failure is actionable instead of a bare
            # "No module named 'controller_manager_msgs'".
            from camelo.ros.real_imports import FIX_HINT

            raise ModuleNotFoundError(
                f"{exc} — required for arm activation (--activate-arms / "
                f"--wait-for-activation) on --world real. Fix: {FIX_HINT}"
            ) from exc

        self.node = node
        self.arms = tuple(arms)
        unknown = [a for a in self.arms if a not in ARMS]
        if unknown:
            raise ValueError(f"arms must be a subset of {ARMS}, got {unknown}")
        self.controller = controller
        self.call_timeout_s = call_timeout_s
        self._ListControllers = ListControllers
        self._SwitchController = SwitchController
        self._list = {
            side: node.create_client(
                ListControllers, f"/{side}/controller_manager/list_controllers"
            )
            for side in self.arms
        }
        self._switch = {
            side: node.create_client(
                SwitchController, f"/{side}/controller_manager/switch_controller"
            )
            for side in self.arms
        }

    # -- plumbing ----------------------------------------------------------
    def service_names(self) -> list[str]:
        return [
            f"/{side}/controller_manager/{kind}"
            for side in self.arms
            for kind in ("list_controllers", "switch_controller")
        ]

    def wait_for_services(self, timeout_s: float = 30.0) -> bool:
        """ALL selected arms' services, before any switch — activate_arms.py's
        rule: discovering a service mid-switch is the burst that faults a live
        FCI loop, so the waiting all happens first."""
        deadline = time.monotonic() + timeout_s
        for side in self.arms:
            for kind, clients in (("list", self._list), ("switch", self._switch)):
                remaining = max(deadline - time.monotonic(), 0.1)
                if not clients[side].wait_for_service(timeout_sec=remaining):
                    log.error(
                        "%s controller_manager %s_controller%s service not "
                        "discovered within %.0fs",
                        side, kind, "s" if kind == "list" else "", timeout_s,
                    )
                    return False
        log.info("controller_manager services up: %s", self.service_names())
        return True

    def _call(self, client, request):
        """Blocking call on a node that is spun by someone else's thread."""
        future = client.call_async(request)
        deadline = time.monotonic() + self.call_timeout_s
        while not future.done():
            if time.monotonic() > deadline:
                return None
            time.sleep(0.02)
        return future.result()

    # -- state -------------------------------------------------------------
    def states(self) -> dict:
        """``{side: state string | None}`` for the selected arms."""
        responses = {
            side: self._call(client, self._ListControllers.Request())
            for side, client in self._list.items()
        }
        return parse_arm_states(responses, self.controller)

    def active(self) -> dict:
        return {side: state == ACTIVE_STATE for side, state in self.states().items()}

    def all_active(self) -> bool:
        states = self.states()
        return bool(states) and all(s == ACTIVE_STATE for s in states.values())

    # -- switching ---------------------------------------------------------
    def _switch_request(self, activate: bool, strictness: int):
        request = self._SwitchController.Request()
        names = [self.controller]
        # ROS 2 renamed these fields (start/stop -> activate/deactivate) and
        # the station's distro is not this repo's to assume. Set whichever
        # the message actually has; a silently unset field would produce an
        # ok=True switch that switched nothing.
        pairs = (
            ("activate_controllers", "deactivate_controllers"),
            ("start_controllers", "stop_controllers"),
        )
        for on_field, off_field in pairs:
            if hasattr(request, on_field):
                setattr(request, on_field, names if activate else [])
                setattr(request, off_field, [] if activate else names)
                break
        else:
            raise RuntimeError(
                "SwitchController.Request has neither activate_controllers nor "
                "start_controllers — unknown controller_manager_msgs version"
            )
        request.strictness = int(strictness)
        return request

    def _strictness(self, name: str) -> int:
        request_cls = self._SwitchController.Request
        return int(getattr(request_cls, name, 1 if name == "BEST_EFFORT" else 2))

    def switch(self, activate: bool) -> bool:
        """Switch every selected arm, BEST_EFFORT first then STRICT.

        The station's `activate_arms.py` order, and it matters: BEST_EFFORT
        tolerates an arm that is already in the requested state (a second
        operator, a retry) where STRICT answers ok=False for it; STRICT is
        the retry that reports a genuine refusal instead of shrugging.
        """
        ok = True
        for strictness_name in ("BEST_EFFORT", "STRICT"):
            ok = True
            strictness = self._strictness(strictness_name)
            for side in self.arms:
                response = self._call(
                    self._switch[side], self._switch_request(activate, strictness)
                )
                if response is None:
                    log.error(
                        "%s switch_controller (%s, %s) did not answer within %.1fs",
                        side,
                        "activate" if activate else "deactivate",
                        strictness_name,
                        self.call_timeout_s,
                    )
                    ok = False
                    continue
                if not getattr(response, "ok", False):
                    log.warning(
                        "%s switch_controller (%s, %s) answered ok=False%s",
                        side,
                        "activate" if activate else "deactivate",
                        strictness_name,
                        " — is the GELLO stream live and are the joint names "
                        "left|right_fr3v2_jointN?" if activate else "",
                    )
                    ok = False
            if ok:
                log.info(
                    "switch_controller %s ok (%s) on %s",
                    "activate" if activate else "deactivate",
                    strictness_name,
                    list(self.arms),
                )
                return True
        return ok

    def wait_until(
        self, active: bool, timeout_s: float = 10.0, poll_s: float = 0.25, on_poll=None
    ) -> bool:
        """Poll list_controllers until every selected arm reaches ``active``.

        ``on_poll`` runs on every iteration — the command stream must keep
        flowing while we wait, or the controller shuts the arm launch down
        (2.0 s) exactly while we are asking it a question.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if on_poll is not None:
                on_poll()
            states = self.states()
            reached = states and all(
                (state == ACTIVE_STATE) is bool(active) for state in states.values()
            )
            if reached:
                return True
            time.sleep(poll_s)
        return False
