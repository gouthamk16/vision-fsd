from __future__ import annotations

from dataclasses import dataclass
import math

from fsd.motion_planning.occupancy import CollisionGrid
from fsd.motion_planning.state import PlannedTrajectory, TrajectoryPoint


@dataclass(frozen=True)
class TrajectoryValidatorConfig:
    vehicle_radius_m: float = 1.25


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reason: str


class TrajectoryValidator:
    def __init__(self, config: TrajectoryValidatorConfig | None = None) -> None:
        self.config = config or TrajectoryValidatorConfig()
        if not math.isfinite(self.config.vehicle_radius_m) or self.config.vehicle_radius_m < 0.0:
            raise ValueError("vehicle_radius_m must be finite and non-negative")

    def validate(
        self,
        trajectory: PlannedTrajectory,
        collision_grid: CollisionGrid,
    ) -> ValidationResult:
        if not trajectory.points:
            return ValidationResult(valid=False, reason="empty")

        previous_point = trajectory.points[0]
        if not _point_is_finite(previous_point):
            return ValidationResult(valid=False, reason="non_finite")
        if _footprint_blocked(previous_point, collision_grid, self.config.vehicle_radius_m):
            return ValidationResult(valid=False, reason="collision")

        for point in trajectory.points[1:]:
            if not _point_is_finite(point):
                return ValidationResult(valid=False, reason="non_finite")

            if point.t <= previous_point.t:
                return ValidationResult(valid=False, reason="non_increasing_time")

            if _segment_footprint_blocked(
                previous_point,
                point,
                collision_grid,
                self.config.vehicle_radius_m,
            ):
                return ValidationResult(valid=False, reason="collision")

            previous_point = point

        return ValidationResult(valid=True, reason="valid")


def _footprint_blocked(
    point: TrajectoryPoint,
    collision_grid: CollisionGrid,
    vehicle_radius_m: float,
) -> bool:
    return collision_grid.footprint_blocked(point.x, point.y, vehicle_radius_m)


def _segment_footprint_blocked(
    start: TrajectoryPoint,
    end: TrajectoryPoint,
    collision_grid: CollisionGrid,
    vehicle_radius_m: float,
) -> bool:
    spacing = _collision_sample_spacing(collision_grid.resolution, vehicle_radius_m)
    distance = math.hypot(end.x - start.x, end.y - start.y)
    steps = 1
    if spacing > 0.0 and distance > 0.0:
        steps = max(1, math.ceil(distance / spacing))

    for step in range(1, steps + 1):
        ratio = step / steps
        x = start.x + (end.x - start.x) * ratio
        y = start.y + (end.y - start.y) * ratio
        if collision_grid.footprint_blocked(x, y, vehicle_radius_m):
            return True
    return False


def _collision_sample_spacing(resolution: float, vehicle_radius_m: float) -> float:
    candidates = []
    if math.isfinite(resolution) and resolution > 0.0:
        candidates.append(resolution / 2.0)
    if math.isfinite(vehicle_radius_m) and vehicle_radius_m > 0.0:
        candidates.append(vehicle_radius_m / 2.0)
    if not candidates:
        return 0.0
    return min(candidates)


def _point_is_finite(point: TrajectoryPoint) -> bool:
    return (
        math.isfinite(point.x)
        and math.isfinite(point.y)
        and math.isfinite(point.yaw)
        and math.isfinite(point.speed_mps)
        and math.isfinite(point.t)
    )
