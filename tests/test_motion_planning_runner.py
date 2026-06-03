from pathlib import Path
from types import SimpleNamespace

import numpy as np

from fsd.data import LidarFrame, SurroundFrame
from fsd.motion_planning.runner import OfflinePlanningRuntime, OfflinePlanningRuntimeConfig
from fsd.motion_planning.state import PlannedTrajectory, PlanningResult


class FakeMapper:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def step(self, lidar: LidarFrame) -> np.ndarray:
        return np.zeros((2, 2), dtype=np.float32)


class CapturingPlanner:
    def __init__(self) -> None:
        self.worlds = []

    def plan(self, world):
        self.worlds.append(world)
        selected = PlannedTrajectory(points=[], cost=0.0, is_emergency_stop=False)
        return PlanningResult(
            selected=selected,
            candidates=[selected],
            valid_candidates=[selected],
            world=world,
            reason="selected",
        )


def make_frame(scene_token: str, sample_index: int, timestamp_us: int) -> SurroundFrame:
    return SurroundFrame(
        scene_token=scene_token,
        scene_name=scene_token,
        sample_token=f"{scene_token}-{sample_index}",
        sample_index=sample_index,
        timestamp_us=timestamp_us,
        cameras={},
    )


def make_lidar(x: float, timestamp_us: int) -> LidarFrame:
    return LidarFrame(
        channel="LIDAR_TOP",
        pointcloud_path=Path("unused.bin"),
        timestamp_us=timestamp_us,
        sample_data_token=f"lidar-{timestamp_us}",
        calibrated_sensor={},
        ego_pose={"translation": [x, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]},
    )


def test_offline_planning_runtime_resets_history_on_scene_change(monkeypatch):
    monkeypatch.setattr(
        "fsd.motion_planning.runner.bev_tensor_from_lidar",
        lambda *args, **kwargs: SimpleNamespace(height_range=np.zeros((2, 2), dtype=np.float32)),
    )

    runtime = OfflinePlanningRuntime(
        OfflinePlanningRuntimeConfig(
            x_range=(0.0, 2.0),
            y_range=(0.0, 2.0),
            resolution=1.0,
        )
    )
    mapper = FakeMapper()
    planner = CapturingPlanner()
    runtime.mapper = mapper
    runtime.planner = planner

    runtime.step(make_frame("scene-a", sample_index=0, timestamp_us=1_000_000), make_lidar(0.0, 1_000_000))
    runtime.step(make_frame("scene-a", sample_index=1, timestamp_us=2_000_000), make_lidar(2.0, 2_000_000))
    runtime.step(make_frame("scene-b", sample_index=0, timestamp_us=3_000_000), make_lidar(20.0, 3_000_000))

    assert mapper.reset_count == 1
    assert not planner.worlds[1].ego.fallback_initial_speed
    assert planner.worlds[1].ego.speed_mps == 2.0
    assert planner.worlds[2].ego.fallback_initial_speed
    assert planner.worlds[2].ego.speed_mps == 0.0
