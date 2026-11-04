from pathlib import Path

import numpy as np

from conftest import make_frame, yaw_pose
from fsd.bev import BevGrid
from fsd.data import LidarFrame, LidarSweep
from fsd.occupancy import TemporalOccupancyMapper

GRID = BevGrid((-20.0, 20.0), (-20.0, 20.0), 0.5)


def _lidar(pose):
    return LidarFrame((LidarSweep(Path("x.bin"), 0, yaw_pose(), pose),))


def _points(xyz):
    xyz = np.asarray(xyz, dtype=np.float32)
    return np.hstack([xyz, np.zeros((len(xyz), 1), np.float32)])


def _prob_at(mapper, prob, x, y):
    rows, cols, _ = mapper.grid.cells(np.array([[x, y]]))
    return prob[rows[0], cols[0]]


def test_ray_is_free_up_to_the_obstacle_and_unknown_behind_it():
    mapper = TemporalOccupancyMapper(GRID)
    pose = yaw_pose()
    ys = np.arange(-3.0, 3.0, 0.05)
    wall = [[10.0, y, 1.0] for y in ys]
    ground_beyond = [[15.0, y, 0.0] for y in ys]
    prob = mapper.step(make_frame(pose), _lidar(pose), _points(wall + ground_beyond))
    assert _prob_at(mapper, prob, 5.0, 0.0) < 0.5    # free: in front of the first obstacle
    assert _prob_at(mapper, prob, 10.0, 0.0) > 0.5   # hit
    assert _prob_at(mapper, prob, 13.0, 0.0) == 0.5  # shadowed, despite a ground return beyond
    assert _prob_at(mapper, prob, 0.0, 8.0) == 0.5   # no return in that direction


def test_grid_follows_ego_motion_and_resets_on_new_scene():
    mapper = TemporalOccupancyMapper(GRID, decay=1.0)
    pose = yaw_pose()
    mapper.step(make_frame(pose), _lidar(pose), _points([[10.0, 5.0, 1.0]]))
    moved = yaw_pose(2.0, 0.0)
    prob = mapper.step(make_frame(moved), _lidar(moved), _points(np.empty((0, 3))))
    assert _prob_at(mapper, prob, 8.0, 5.0) > 0.5    # same world cell, now 2 m closer
    prob = mapper.step(make_frame(moved, scene_token="other"), _lidar(moved), _points(np.empty((0, 3))))
    assert np.all(prob == 0.5)
