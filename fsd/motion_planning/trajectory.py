from __future__ import annotations

import math
from collections.abc import Sequence

from fsd.motion_planning.ego_motion import wrap_angle
from fsd.motion_planning.state import EgoState, PlannedTrajectory, TrajectoryPoint


def time_values(horizon_s: float, dt_s: float) -> list[float]:
    if not math.isfinite(horizon_s) or not math.isfinite(dt_s):
        raise ValueError("horizon_s and dt_s must be finite")
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    if horizon_s < 0.0:
        raise ValueError("horizon_s must be non-negative")

    values = [0.0]
    step = 1
    while step * dt_s < horizon_s:
        value = round(step * dt_s, 10)
        if value > values[-1]:
            values.append(value)
        step += 1

    rounded_horizon = round(horizon_s, 10)
    if rounded_horizon > values[-1]:
        values.append(rounded_horizon)
    elif horizon_s > 0.0 and values[-1] == 0.0:
        values.append(horizon_s)
    return values


def make_emergency_stop(ego: EgoState, horizon_s: float, dt_s: float) -> PlannedTrajectory:
    times = time_values(horizon_s, dt_s)
    initial_speed_mps = 0.0 if horizon_s == 0.0 else ego.speed_mps
    x = 0.0
    yaw = 0.0
    points = [
        TrajectoryPoint(
            x=x,
            y=0.0,
            yaw=yaw,
            speed_mps=initial_speed_mps,
            t=0.0,
        )
    ]

    previous_t = 0.0
    previous_speed_mps = initial_speed_mps
    for t in times[1:]:
        progress = t / horizon_s
        speed_mps = ego.speed_mps * max(0.0, 1.0 - progress)
        dt = t - previous_t
        interval_speed_mps = (previous_speed_mps + speed_mps) / 2.0
        x += interval_speed_mps * dt
        points.append(
            TrajectoryPoint(
                x=x,
                y=0.0,
                yaw=yaw,
                speed_mps=speed_mps,
                t=t,
            )
        )
        previous_t = t
        previous_speed_mps = speed_mps

    return PlannedTrajectory(
        points=points,
        cost=0.0,
        is_emergency_stop=True,
        metadata={"curvature": 0.0, "target_speed_mps": 0.0},
    )


def trajectory_path_length(points: Sequence[TrajectoryPoint]) -> float:
    distance = 0.0
    for previous, current in zip(points, points[1:]):
        distance += math.hypot(current.x - previous.x, current.y - previous.y)
    return distance


def final_heading_error(points: Sequence[TrajectoryPoint]) -> float:
    if len(points) < 2:
        return 0.0
    return wrap_angle(points[-1].yaw - points[0].yaw)
