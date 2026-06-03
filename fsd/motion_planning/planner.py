from __future__ import annotations

from dataclasses import dataclass, replace

from fsd.motion_planning.costs import CostWeights, score_trajectory
from fsd.motion_planning.occupancy import CollisionGrid
from fsd.motion_planning.sampler import LatticeSampler, LatticeSamplerConfig
from fsd.motion_planning.state import PlannedTrajectory, PlannerWorld, PlanningResult
from fsd.motion_planning.trajectory import make_emergency_stop
from fsd.motion_planning.validator import TrajectoryValidator, TrajectoryValidatorConfig


@dataclass(frozen=True)
class LocalPlannerConfig:
    horizon_s: float = 3.0
    dt_s: float = 0.25
    target_speeds_mps: tuple[float, ...] = (0.0, 2.5, 5.0, 7.5)
    curvatures: tuple[float, ...] = (-0.12, -0.06, -0.03, 0.0, 0.03, 0.06, 0.12)
    vehicle_radius_m: float = 1.25


class LocalPlanner:
    def __init__(
        self,
        config: LocalPlannerConfig | None = None,
        weights: CostWeights | None = None,
    ) -> None:
        self.config = config or LocalPlannerConfig()
        self.weights = weights or CostWeights()
        self.sampler = LatticeSampler(
            LatticeSamplerConfig(
                horizon_s=self.config.horizon_s,
                dt_s=self.config.dt_s,
                target_speeds_mps=self.config.target_speeds_mps,
                curvatures=self.config.curvatures,
            )
        )
        self.validator = TrajectoryValidator(
            TrajectoryValidatorConfig(vehicle_radius_m=self.config.vehicle_radius_m)
        )

    def plan(self, world: PlannerWorld) -> PlanningResult:
        collision_grid = CollisionGrid(
            blocked=world.collision_grid,
            x_range=world.x_range,
            y_range=world.y_range,
            resolution=world.resolution,
        )
        candidates = self.sampler.sample(world.ego)
        valid_candidates: list[PlannedTrajectory] = []

        for candidate in candidates:
            validation = self.validator.validate(candidate, collision_grid)
            if not validation.valid:
                continue

            cost = score_trajectory(candidate, world, self.weights)
            valid_candidates.append(replace(candidate, cost=cost))

        if not valid_candidates:
            emergency = make_emergency_stop(world.ego, self.config.horizon_s, self.config.dt_s)
            validation = self.validator.validate(emergency, collision_grid)
            emergency = replace(
                emergency,
                metadata={
                    **emergency.metadata,
                    "fallback_collision_free": 1.0 if validation.valid else 0.0,
                    "fallback_validation_reason": validation.reason,
                },
            )
            return PlanningResult(
                selected=emergency,
                candidates=candidates,
                valid_candidates=[],
                world=world,
                reason="no_valid_trajectory",
            )

        selected = min(valid_candidates, key=lambda trajectory: trajectory.cost)
        return PlanningResult(
            selected=selected,
            candidates=candidates,
            valid_candidates=valid_candidates,
            world=world,
            reason="selected",
        )
