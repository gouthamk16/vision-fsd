import pytest

from fsd import visualize
from fsd.motion_planning.runner import OfflinePlanningRuntimeConfig


class EmptySceneLoader:
    def __init__(self, dataroot=None):
        self.dataroot = dataroot

    def iter_scene_frames(self, **kwargs):
        return iter(())


class CapturingPlanningRuntime:
    configs = []

    def __init__(self, config=None):
        self.config = config
        self.configs.append(config)

    def reset(self):
        pass


@pytest.mark.parametrize("view", ["planner_bev", "planner_camera"])
def test_run_visualizer_passes_bev_resolution_to_planner_runtime(monkeypatch, view):
    CapturingPlanningRuntime.configs = []
    monkeypatch.setattr(visualize, "NuScenesSceneLoader", EmptySceneLoader)
    monkeypatch.setattr(visualize, "OfflinePlanningRuntime", CapturingPlanningRuntime)

    with pytest.raises(RuntimeError, match="No frames were rendered"):
        visualize.run_visualizer(view=view, bev_resolution=0.5)

    assert len(CapturingPlanningRuntime.configs) == 1
    config = CapturingPlanningRuntime.configs[0]
    assert isinstance(config, OfflinePlanningRuntimeConfig)
    assert config.resolution == 0.5
    assert config.x_range == (-50.0, 50.0)
    assert config.y_range == (-50.0, 50.0)
    assert config.occupancy_threshold == 0.62
    assert config.height_threshold == 0.45
