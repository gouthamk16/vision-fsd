from dataclasses import replace

import cv2
import numpy as np

from conftest import make_frame, yaw_pose
from fsd.data import CameraFrame, Pose
from fsd.nmp.config import NmpConfig
from fsd.nmp.planner import PlanResult
from fsd.nmp.render import _camera_pixels, render_planner_bev
from fsd.object_detection import Box3D
from fsd.tracking import TrackedObject

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
    image = render_planner_bev(make_frame(yaw_pose()), np.zeros((0, 3)), _result(np.zeros((CFG.horizon, 3), np.float32)), CFG.grid, tracked=[])
    assert image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3


def test_render_moving_ego_with_logged_future():
    plan = np.zeros((CFG.horizon, 3), np.float32)
    plan[:, 0] = 5.0 * CFG.times
    points = np.array([[10.0, 2.0, 1.0], [20.0, -3.0, 0.0]])
    image = render_planner_bev(make_frame(yaw_pose()), points, _result(plan), CFG.grid, logged=plan + [0.0, 0.5, 0.0], tracked=[])
    assert image.shape[0] > 600


def _front_camera(tmp_path) -> CameraFrame:
    """160x90 pinhole 1.7 m ahead of the rear axle, 1.5 m up, looking along ego x (camera z forward, y down)."""
    path = tmp_path / "front.jpg"
    cv2.imwrite(str(path), np.full((90, 160, 3), 128, np.uint8))
    to_ego = Pose(np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]), np.array([1.7, 0.0, 1.5]))
    return CameraFrame("CAM_FRONT", path, 0, np.array([[100.0, 0, 80], [0, 100.0, 45], [0, 0, 1]]), to_ego, yaw_pose())


def test_ground_point_ahead_projects_below_the_horizon_on_the_centre_column(tmp_path):
    frame = make_frame(yaw_pose())
    pixels = _camera_pixels(np.array([[10.0, 0.0], [-5.0, 0.0]]), frame, _front_camera(tmp_path))
    assert len(pixels) == 1
    assert abs(pixels[0, 0] - 80) < 1e-6 and pixels[0, 1] > 45


def test_render_with_front_camera(tmp_path):
    frame = replace(make_frame(yaw_pose()), cameras={"CAM_FRONT": _front_camera(tmp_path)})
    plan = np.zeros((CFG.horizon, 3), np.float32)
    plan[:, 0] = 5.0 * CFG.times
    image = render_planner_bev(frame, np.zeros((0, 3)), _result(plan), CFG.grid, logged=plan, tracked=[])
    assert image.dtype == np.uint8 and image.shape[2] == 3


def test_render_without_a_plan_while_history_fills():
    image = render_planner_bev(make_frame(yaw_pose()), np.zeros((0, 3)), None, CFG.grid, tracked=[])
    assert image.dtype == np.uint8 and image.shape[2] == 3


def test_tracked_vehicles_are_drawn_where_the_filter_puts_them():
    car = Box3D("car", np.array([12.0, 3.0, 0.8]), np.array([1.9, 4.6, 1.6]), 0.0, 0.9)
    moving = TrackedObject(car, "p0", np.array([5.0, 0.0]), 5.0, np.array([[12.0 + 5.0 * t, 3.0] for t in CFG.times]), car)
    plan = np.zeros((CFG.horizon, 3), np.float32)
    plain = render_planner_bev(make_frame(yaw_pose()), np.zeros((0, 3)), _result(plan), CFG.grid, tracked=[])
    tracked = render_planner_bev(make_frame(yaw_pose()), np.zeros((0, 3)), _result(plan), CFG.grid, tracked=[moving])
    assert not np.array_equal(plain, tracked)
