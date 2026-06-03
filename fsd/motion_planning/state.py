from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EgoState:
    x_global: float
    y_global: float
    yaw_global: float
    speed_mps: float
    yaw_rate_rps: float
    timestamp_us: int
    fallback_initial_speed: bool = False


@dataclass(frozen=True)
class LaneContext:
    centerline_xy: np.ndarray
    confidence: float

    def has_high_confidence(self, threshold: float) -> bool:
        return (
            self.confidence >= threshold
            and self.centerline_xy.ndim == 2
            and self.centerline_xy.shape[1] == 2
            and self.centerline_xy.shape[0] >= 2
        )


@dataclass(frozen=True)
class TrajectoryPoint:
    x: float
    y: float
    yaw: float
    speed_mps: float
    t: float


@dataclass(frozen=True)
class PlannedTrajectory:
    points: list[TrajectoryPoint]
    cost: float
    is_emergency_stop: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        if not self.points:
            return 0.0
        return self.points[-1].t - self.points[0].t

    @property
    def endpoint_xy(self) -> tuple[float, float]:
        if not self.points:
            return 0.0, 0.0
        endpoint = self.points[-1]
        return endpoint.x, endpoint.y


@dataclass(frozen=True)
class PlannerWorld:
    ego: EgoState
    collision_grid: np.ndarray
    occupancy_probability: np.ndarray
    height_range: np.ndarray
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    resolution: float
    lane_context: LaneContext | None = None


@dataclass(frozen=True)
class PlanningResult:
    selected: PlannedTrajectory
    candidates: list[PlannedTrajectory]
    valid_candidates: list[PlannedTrajectory]
    world: PlannerWorld
    reason: str
