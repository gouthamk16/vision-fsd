from __future__ import annotations

import math
from dataclasses import dataclass

from fsd.data import Pose

# Renault ZOE body (nuScenes ego vehicle) around the ego-frame origin, which
# nuScenes puts at the centre of the rear axle: 0.66 m rear overhang, 4.08 m
# long, 1.73 m wide. As ((x_min, x_max), (y_min, y_max)) in metres.
EGO_FOOTPRINT = ((-0.66, 3.43), (-0.87, 0.87))


@dataclass(frozen=True)
class EgoState:
    pose: Pose
    speed_mps: float
    yaw_rate_rps: float
    timestamp_us: int
    fallback_initial_speed: bool = False


def wrap_angle(angle: float) -> float:
    """Map an angle in radians to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def estimate_ego_state(
    pose: Pose,
    timestamp_us: int,
    previous_pose: Pose | None,
    previous_timestamp_us: int | None,
) -> EgoState:
    dt_s = 0.0 if previous_timestamp_us is None else (timestamp_us - previous_timestamp_us) / 1_000_000.0
    if previous_pose is None or dt_s <= 0.0:
        return EgoState(pose, 0.0, 0.0, timestamp_us, fallback_initial_speed=True)
    dx, dy = (pose.translation - previous_pose.translation)[:2]
    return EgoState(
        pose=pose,
        speed_mps=math.hypot(dx, dy) / dt_s,
        yaw_rate_rps=wrap_angle(pose.yaw - previous_pose.yaw) / dt_s,
        timestamp_us=timestamp_us,
    )
