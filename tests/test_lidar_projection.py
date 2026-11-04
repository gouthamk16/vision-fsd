from pathlib import Path

import numpy as np

from conftest import yaw_pose
from fsd.data import CameraFrame, Pose
from fsd.lidar_projection import _off_ego_body, project_to_camera


def test_ego_body_returns_are_dropped():
    points = np.array([[1.0, 0.0, 1.5], [1.0, 0.0, 3.0], [8.0, 0.0, 0.5], [1.0, 2.0, 0.5]])
    assert _off_ego_body(points).tolist() == [False, True, True, True]


def test_project_to_camera_keeps_indices_of_visible_points():
    # Camera looking along ego +x: camera z = ego x, camera x = -ego y, camera y = -ego z.
    to_ego = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    camera = CameraFrame("CAM_FRONT", Path("x.jpg"), 0, np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]]),
                         calibration=Pose(to_ego, np.zeros(3)), ego_pose=yaw_pose())
    points = np.array([[10.0, 0.0, 0.0, 0.0], [-5.0, 0.0, 0.0, 0.0], [10.0, 100.0, 0.0, 0.0]], dtype=np.float32)
    uv, depth, kept = project_to_camera(points, yaw_pose(), camera, (100, 100, 3))
    assert kept.tolist() == [0]
    assert np.allclose(uv[0], [50.0, 50.0]) and np.isclose(depth[0], 10.0)
