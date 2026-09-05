"""Continuous base twist -> discrete pedal token.

The bridge accepts only the pedal token vocabulary (one motion at a time at
fixed speed); a policy emitting continuous (vx, vy, wz) must be quantized.
Thresholds are relative to the helper stack's configured pedal speeds so a
recorded twist (which is itself token-generated) round-trips to the same
token; hysteresis avoids token flapping around the threshold.
"""

from __future__ import annotations

from camelo.contracts import PEDAL_ANGULAR_SPEED, PEDAL_LINEAR_SPEED

# axis -> (positive token, negative token)
_AXIS_TOKENS = {
    "x": ("FWD", "BACK"),
    "y": ("A", "B"),
    "w": ("A+C", "B+C"),
}
_TOKEN_AXIS = {tok: (axis, sign) for axis, (pos, neg) in _AXIS_TOKENS.items()
               for tok, sign in ((pos, 1.0), (neg, -1.0))}


class BaseQuantizer:
    def __init__(
        self,
        linear_speed: float = PEDAL_LINEAR_SPEED,
        angular_speed: float = PEDAL_ANGULAR_SPEED,
        engage: float = 0.5,
        release: float = 0.3,
    ):
        if release >= engage:
            raise ValueError("release threshold must be below engage threshold")
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.engage = engage
        self.release = release
        self._token = "NONE"

    def reset(self) -> None:
        self._token = "NONE"

    def quantize(self, vx: float, vy: float, wz: float) -> str:
        norm = {
            "x": vx / self.linear_speed,
            "y": vy / self.linear_speed,
            "w": wz / self.angular_speed,
        }
        axis, value = max(norm.items(), key=lambda kv: abs(kv[1]))

        # Hysteresis: hold the current token while its own axis stays above
        # the release threshold and no other axis clearly dominates it.
        if self._token != "NONE":
            held_axis, held_sign = _TOKEN_AXIS[self._token]
            held_value = norm[held_axis] * held_sign
            if held_value >= self.release and abs(value) <= max(held_value, self.engage):
                return self._token

        if abs(value) < self.engage:
            self._token = "NONE"
        else:
            pos, neg = _AXIS_TOKENS[axis]
            self._token = pos if value > 0 else neg
        return self._token
