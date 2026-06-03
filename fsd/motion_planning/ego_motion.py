from __future__ import annotations

import math
from typing import Any

from fsd.lidar_projection import quaternion_to_rotation_matrix
from fsd.motion_planning.state import EgoState


def wrap_angle(angle: float) -> float:
    """Map an angle in radians to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_pose(pose: dict[str, Any]) -> float:
    rot = quaternion_to_rotation_matrix(pose["rotation"])
    return math.atan2(rot[1, 0], rot[0, 0])


def estimate_ego_state(
    current_pose: dict[str, Any],
    current_timestamp_us: int,
    previous_pose: dict[str, Any] | None,
    previous_timestamp_us: int | None,
) -> EgoState:
    current_translation = current_pose["translation"]
    current_yaw = yaw_from_pose(current_pose)

    fallback = previous_pose is None or previous_timestamp_us is None
    if fallback:
        speed_mps = 0.0
        yaw_rate_rps = 0.0
    else:
        dt_s = (current_timestamp_us - previous_timestamp_us) / 1_000_000.0
        if dt_s <= 0.0:
            fallback = True
            speed_mps = 0.0
            yaw_rate_rps = 0.0
        else:
            previous_translation = previous_pose["translation"]
            dx = current_translation[0] - previous_translation[0]
            dy = current_translation[1] - previous_translation[1]
            speed_mps = math.hypot(dx, dy) / dt_s
            yaw_rate_rps = wrap_angle(current_yaw - yaw_from_pose(previous_pose)) / dt_s

    return EgoState(
        x_global=current_translation[0],
        y_global=current_translation[1],
        yaw_global=current_yaw,
        speed_mps=speed_mps,
        yaw_rate_rps=yaw_rate_rps,
        timestamp_us=current_timestamp_us,
        fallback_initial_speed=fallback,
    )
