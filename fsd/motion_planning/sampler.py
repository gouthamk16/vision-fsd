from __future__ import annotations

from dataclasses import dataclass
import math

from fsd.motion_planning.state import EgoState, PlannedTrajectory, TrajectoryPoint
from fsd.motion_planning.trajectory import time_values


@dataclass(frozen=True)
class LatticeSamplerConfig:
    horizon_s: float = 3.0
    dt_s: float = 0.25
    target_speeds_mps: tuple[float, ...] = (0.0, 2.5, 5.0, 7.5)
    curvatures: tuple[float, ...] = (-0.12, -0.06, -0.03, 0.0, 0.03, 0.06, 0.12)


def _step_constant_curvature_arc(
    x: float,
    y: float,
    yaw: float,
    distance: float,
    curvature: float,
) -> tuple[float, float, float]:
    delta_yaw = curvature * distance
    if abs(curvature) < 1e-12:
        return (
            x + distance * math.cos(yaw),
            y + distance * math.sin(yaw),
            yaw + delta_yaw,
        )

    new_yaw = yaw + delta_yaw
    return (
        x + (math.sin(new_yaw) - math.sin(yaw)) / curvature,
        y + (-math.cos(new_yaw) + math.cos(yaw)) / curvature,
        new_yaw,
    )


class LatticeSampler:
    def __init__(self, config: LatticeSamplerConfig | None = None) -> None:
        self.config = config or LatticeSamplerConfig()

    def sample(self, ego: EgoState) -> list[PlannedTrajectory]:
        trajectories = []
        for target_speed_mps in self.config.target_speeds_mps:
            for curvature in self.config.curvatures:
                trajectories.append(self._integrate(ego, target_speed_mps, curvature))
        return trajectories

    def _integrate(
        self,
        ego: EgoState,
        target_speed_mps: float,
        curvature: float,
    ) -> PlannedTrajectory:
        x = 0.0
        y = 0.0
        yaw = 0.0
        points = [
            TrajectoryPoint(
                x=x,
                y=y,
                yaw=yaw,
                speed_mps=ego.speed_mps,
                t=0.0,
            )
        ]

        previous_t = 0.0
        previous_speed_mps = ego.speed_mps
        for t in time_values(self.config.horizon_s, self.config.dt_s)[1:]:
            dt_s = t - previous_t
            progress = t / self.config.horizon_s if self.config.horizon_s > 0.0 else 1.0
            speed_mps = ego.speed_mps + (target_speed_mps - ego.speed_mps) * progress
            interval_speed_mps = (previous_speed_mps + speed_mps) / 2.0
            distance = interval_speed_mps * dt_s
            x, y, yaw = _step_constant_curvature_arc(x, y, yaw, distance, curvature)
            points.append(
                TrajectoryPoint(
                    x=x,
                    y=y,
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
            is_emergency_stop=False,
            metadata={
                "curvature": curvature,
                "target_speed_mps": target_speed_mps,
            },
        )
