from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fsd.bev_tensor import bev_tensor_from_lidar
from fsd.data import LidarFrame, SurroundFrame
from fsd.motion_planning.ego_motion import estimate_ego_state
from fsd.motion_planning.occupancy import build_collision_grid
from fsd.motion_planning.planner import LocalPlanner, LocalPlannerConfig
from fsd.motion_planning.state import PlannerWorld, PlanningResult
from fsd.occupancy import TemporalOccupancyMapper


@dataclass(frozen=True)
class OfflinePlanningRuntimeConfig:
    x_range: tuple[float, float] = (-50.0, 50.0)
    y_range: tuple[float, float] = (-50.0, 50.0)
    resolution: float = 0.25
    occupancy_threshold: float = 0.62
    height_threshold: float = 0.45


class OfflinePlanningRuntime:
    def __init__(
        self,
        config: OfflinePlanningRuntimeConfig | None = None,
        planner_config: LocalPlannerConfig | None = None,
    ) -> None:
        self.config = config or OfflinePlanningRuntimeConfig()
        self.mapper = TemporalOccupancyMapper(
            x_range=self.config.x_range,
            y_range=self.config.y_range,
            resolution=self.config.resolution,
        )
        self.planner = LocalPlanner(planner_config)
        self._previous_pose: dict[str, Any] | None = None
        self._previous_timestamp_us: int | None = None
        self._current_scene_token: str | None = None

    def reset(self) -> None:
        self.mapper.reset()
        self._previous_pose = None
        self._previous_timestamp_us = None
        self._current_scene_token = None

    def step(self, frame: SurroundFrame, lidar: LidarFrame) -> PlanningResult:
        if self._current_scene_token is None:
            self._current_scene_token = frame.scene_token
        elif frame.scene_token != self._current_scene_token:
            self.reset()
            self._current_scene_token = frame.scene_token

        occupancy_probability = self.mapper.step(lidar)
        height_tensor = bev_tensor_from_lidar(
            lidar,
            x_range=self.config.x_range,
            y_range=self.config.y_range,
            resolution=self.config.resolution,
        )
        collision_grid = build_collision_grid(
            occupancy_probability=occupancy_probability,
            height_range=height_tensor.height_range,
            x_range=self.config.x_range,
            y_range=self.config.y_range,
            resolution=self.config.resolution,
            occupancy_threshold=self.config.occupancy_threshold,
            height_threshold=self.config.height_threshold,
        )
        ego = estimate_ego_state(
            current_pose=lidar.ego_pose,
            current_timestamp_us=lidar.timestamp_us,
            previous_pose=self._previous_pose,
            previous_timestamp_us=self._previous_timestamp_us,
        )
        self._previous_pose = lidar.ego_pose
        self._previous_timestamp_us = lidar.timestamp_us

        world = PlannerWorld(
            ego=ego,
            collision_grid=collision_grid.blocked,
            occupancy_probability=occupancy_probability,
            height_range=height_tensor.height_range,
            x_range=self.config.x_range,
            y_range=self.config.y_range,
            resolution=self.config.resolution,
            lane_context=None,
        )
        return self.planner.plan(world)
