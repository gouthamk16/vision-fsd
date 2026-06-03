from __future__ import annotations

from dataclasses import dataclass

from fsd.motion_planning.state import PlannedTrajectory, PlannerWorld
from fsd.motion_planning.trajectory import trajectory_path_length


@dataclass(frozen=True)
class CostWeights:
    progress: float = 4.0
    lateral_offset: float = 1.2
    curvature: float = 2.0
    speed_error: float = 0.4
    lane_center: float = 0.7


def score_trajectory(
    trajectory: PlannedTrajectory,
    world: PlannerWorld,
    weights: CostWeights | None = None,
) -> float:
    weights = weights or CostWeights()
    progress = trajectory_path_length(trajectory.points)
    _, endpoint_y = trajectory.endpoint_xy
    lateral_offset = abs(endpoint_y)
    curvature = abs(float(trajectory.metadata.get("curvature", 0.0)))
    target_speed_mps = float(trajectory.metadata.get("target_speed_mps", world.ego.speed_mps))
    speed_error = abs(target_speed_mps - world.ego.speed_mps)

    cost = (
        -progress * weights.progress
        + lateral_offset * weights.lateral_offset
        + curvature * weights.curvature
        + speed_error * weights.speed_error
    )

    if world.lane_context is not None and world.lane_context.has_high_confidence(0.6):
        lane_endpoint_y = float(world.lane_context.centerline_xy[-1, 1])
        cost += abs(endpoint_y - lane_endpoint_y) * weights.lane_center

    return cost
