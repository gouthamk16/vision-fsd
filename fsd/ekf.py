"""Constant turn-rate and velocity (CTRV) extended Kalman filter for one tracked object, global frame.

State x = (px, py, v, yaw, yaw_rate); a measurement is a detection's (px, py, yaw). Design:
docs/superpowers/specs/2026-06-09-ctrv-ekf-tracking-design.md.
"""

from __future__ import annotations

import math

import numpy as np

from fsd.ego_motion import wrap_angle

# Straight-line motion below this turn rate: v / yaw_rate blows up as it reaches zero.
_STRAIGHT_RATE = 1e-4


def ctrv_step(x: np.ndarray, dt: float) -> np.ndarray:
    px, py, v, yaw, rate = x
    if abs(rate) < _STRAIGHT_RATE:
        px, py = px + v * math.cos(yaw) * dt, py + v * math.sin(yaw) * dt
    else:
        px += v / rate * (math.sin(yaw + rate * dt) - math.sin(yaw))
        py += v / rate * (math.cos(yaw) - math.cos(yaw + rate * dt))
    return np.array([px, py, v, yaw + rate * dt, rate])


class CtrvEkf:
    """A new track starts still (v = 0, no turn) with wide uncertainty on both, so the first updates set them."""

    def __init__(self, px: float, py: float, yaw: float, pos_std: float = 0.5, yaw_std: float = 0.1,
                 accel_std: float = 2.0, yaw_accel_std: float = 0.5):
        self.x = np.array([px, py, 0.0, yaw, 0.0])
        self.P = np.diag([pos_std**2, pos_std**2, 10.0**2, yaw_std**2, 1.0**2])
        self.R = np.diag([pos_std**2, pos_std**2, yaw_std**2])
        self.noise = np.diag([accel_std**2, yaw_accel_std**2])

    def predict(self, dt: float) -> None:
        # A numerical Jacobian of ctrv_step. The step must reach past _STRAIGHT_RATE: inside it the straight-line
        # branch has no turn rate in it, so a smaller step would lose the position / turn-rate coupling exactly
        # when a track is new (rate 0). ctrv_step is smooth across the branch, so 1e-3 is still accurate.
        eps = 1e-3
        jac = np.stack([(ctrv_step(self.x + e, dt) - ctrv_step(self.x - e, dt)) / (2 * eps) for e in np.eye(5) * eps], axis=1)
        c, s = math.cos(self.x[3]), math.sin(self.x[3])
        # Random longitudinal and yaw acceleration over the step (the standard CTRV process noise).
        g = np.array([[0.5 * dt**2 * c, 0.0], [0.5 * dt**2 * s, 0.0], [dt, 0.0], [0.0, 0.5 * dt**2], [0.0, dt]])
        self.x = ctrv_step(self.x, dt)
        self.P = jac @ self.P @ jac.T + g @ self.noise @ g.T

    def update(self, z: np.ndarray) -> None:
        h = np.zeros((3, 5))
        h[0, 0] = h[1, 1] = h[2, 3] = 1.0
        residual = z - self.x[[0, 1, 3]]
        residual[2] = wrap_angle(residual[2])
        gain = self.P @ h.T @ np.linalg.inv(h @ self.P @ h.T + self.R)
        self.x = self.x + gain @ residual
        self.x[3] = wrap_angle(self.x[3])
        self.P = (np.eye(5) - gain @ h) @ self.P

    def forecast(self, horizons_s: tuple[float, ...]) -> np.ndarray:
        """(K, 2) global positions at each horizon from now, along the current turn."""
        return np.array([ctrv_step(self.x, t)[:2] for t in horizons_s]).reshape(-1, 2)
