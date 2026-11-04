import numpy as np

from conftest import make_frame, yaw_pose
from fsd.nmp.config import NmpConfig
from fsd.nmp.planner import PlanResult
from fsd.nmp.render import render_planner_bev

CFG = NmpConfig()


def _result(plan: np.ndarray) -> PlanResult:
    return PlanResult(
        plan=plan,
        candidates=np.repeat(plan[None], 4, axis=0),
        costs=np.zeros(4, np.float32),
        cost_volume=np.random.default_rng(0).normal(size=(CFG.horizon, *CFG.grid.shape)).astype(np.float32),
        detections=[],
        forecasts=np.zeros((0, CFG.horizon + 1, 3), np.float32),
        v0=0.0,
    )


def test_render_stopped_ego_without_detections_map_or_log():
    image = render_planner_bev(make_frame(yaw_pose()), np.zeros((0, 3)), _result(np.zeros((CFG.horizon, 3), np.float32)), CFG.grid)
    assert image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3


def test_render_moving_ego_with_logged_future():
    plan = np.zeros((CFG.horizon, 3), np.float32)
    plan[:, 0] = 5.0 * CFG.times
    points = np.array([[10.0, 2.0, 1.0], [20.0, -3.0, 0.0]])
    image = render_planner_bev(make_frame(yaw_pose()), points, _result(plan), CFG.grid, logged=plan + [0.0, 0.5, 0.0])
    assert image.shape[0] > 600
