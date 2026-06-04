"""Ground-truth object velocity and short-horizon prediction from nuScenes instance tokens.

GT-first oracle: matching objects across keyframes is free because every annotation
carries a stable ``instance_token``, so velocity is just a finite difference of the
same object's global position. Velocity is computed in the global frame (true ground
motion) and rotated into the current ego frame; future positions are a constant-velocity
extrapolation. This is the baseline a learned tracker is later validated against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fsd.data import SurroundFrame
from fsd.lidar_projection import quaternion_to_rotation_matrix, transform_points
from fsd.object_detection import Box3D


@dataclass(frozen=True)
class TrackedObject:
    box: Box3D
    track_id: str
    velocity_ego: np.ndarray   # (vx, vy) m/s, current ego frame
    speed_mps: float
    future_xy_ego: np.ndarray  # (K, 2) constant-velocity predicted centers, current ego frame


class GtVelocityTracker:
    """Finite-difference velocity over stable instance tokens, with constant-velocity prediction."""

    def __init__(self, horizons_s: tuple[float, ...] = (1.0, 2.0, 3.0), min_speed_mps: float = 0.5):
        self.horizons_s = horizons_s
        self.min_speed_mps = min_speed_mps
        self._prev: dict[str, tuple[np.ndarray, int]] = {}

    def reset(self) -> None:
        self._prev.clear()

    def update(self, frame: SurroundFrame, boxes: list[Box3D]) -> list[TrackedObject]:
        ego_pose = frame.ego_pose
        rot_ego = quaternion_to_rotation_matrix(ego_pose["rotation"])[:2, :2]
        ts = frame.timestamp_us

        tracked: list[TrackedObject] = []
        current: dict[str, tuple[np.ndarray, int]] = {}
        for box in boxes:
            token = box.instance_token
            # Recover the object's global xy (exact inverse of how center_ego was built).
            global_xy = transform_points(
                box.center_ego.reshape(1, 3), ego_pose["rotation"], ego_pose["translation"]
            )[0, :2]
            if token:
                current[token] = (global_xy, ts)

            v_ego = np.zeros(2)
            speed = 0.0
            prev = self._prev.get(token) if token else None
            if prev is not None:
                prev_xy, prev_ts = prev
                dt = (ts - prev_ts) / 1_000_000.0
                if dt > 0.0:
                    v_ego = rot_ego.T @ ((global_xy - prev_xy) / dt)
                    speed = float(np.hypot(v_ego[0], v_ego[1]))

            if speed >= self.min_speed_mps:
                future = np.stack([box.center_ego[:2] + v_ego * h for h in self.horizons_s])
            else:
                future = np.empty((0, 2))

            tracked.append(
                TrackedObject(
                    box=box,
                    track_id=token,
                    velocity_ego=v_ego,
                    speed_mps=speed,
                    future_xy_ego=future,
                )
            )

        self._prev = current
        return tracked
