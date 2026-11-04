import math
from pathlib import Path

import numpy as np

from conftest import make_frame, yaw_pose
from fsd.data import LidarFrame, LidarSweep
from fsd.nmp.config import NmpConfig
from fsd.nmp.samples import CLASS_NAMES, NmpSample, _future_boxes, ego_state, load_sample, save_sample, voxelize

CFG = NmpConfig()


def test_voxelize_indexes_sweep_z_row_col():
    points = np.array([[70.3, 39.9, -1.9], [70.3, 39.9, -1.9], [80.0, 0.0, 0.0], [0.0, 0.0, 9.0]])
    sweep = np.array([1, 1, 0, 0])
    assert voxelize(points, sweep, CFG).tolist() == [((1 * 27 + 0) * 704 + 0) * 400 + 0]


def _lidar(poses_and_times):
    return LidarFrame(tuple(LidarSweep(Path("x.bin"), t, yaw_pose(), pose) for pose, t in poses_and_times))


def test_ego_state_stationary_is_zero_curvature():
    assert ego_state(_lidar([(yaw_pose(), 100_000), (yaw_pose(), 50_000), (yaw_pose(), 0)])) == (0.0, 0.0)


def test_ego_state_turning():
    v0, k0 = ego_state(_lidar([(yaw_pose(1.0, 0.0, 0.05), 100_000), (yaw_pose(0.5, 0.0, 0.025), 50_000), (yaw_pose(), 0)]))
    assert math.isclose(v0, 10.0, rel_tol=1e-6) and math.isclose(k0, 0.05, rel_tol=1e-6)


def _annotation(instance, x):
    return {"instance_token": instance, "translation": [x, 2.0, 0.5], "size": [1.9, 4.6, 1.5], "rotation": [1.0, 0.0, 0.0, 0.0], "num_lidar_pts": 5}


def test_box_missing_in_future_is_masked():
    frames = [make_frame(yaw_pose(2.0 * k, 0.0), 500_000 * k) for k in range(7)]
    annotations = [[_annotation("early", 10.0 + k)] if k <= 2 else [] for k in range(7)]
    annotations[4].append(_annotation("late", 30.0))
    categories = {"early": "vehicle.car", "late": "human.pedestrian.adult"}
    boxes, valid, cls = _future_boxes(frames, annotations, categories)
    rows = {CLASS_NAMES[c]: i for i, c in enumerate(cls)}
    assert valid[rows["car"]].tolist() == [True, True, True, False, False, False, False]
    assert valid[rows["pedestrian"]].tolist() == [False] * 4 + [True, False, False]
    # Every step is expressed in the step-0 ego frame.
    assert np.allclose(boxes[rows["car"], 2, :2], [12.0, 2.0])


def test_save_load_round_trip(tmp_path):
    sample = NmpSample(
        token="tok", scene="scene-0001",
        voxels=np.array([3, 70], np.uint32), map_masks=np.eye(4, dtype=np.uint8)[None].repeat(8, 0),
        ego_future=np.ones((6, 3), np.float32), v0=4.5, k0=-0.01,
        boxes=np.zeros((2, 7, 5), np.float32), box_valid=np.ones((2, 7), bool), box_class=np.array([1, 3], np.int8),
    )
    save_sample(sample, tmp_path / "s.npz")
    loaded = load_sample(tmp_path / "s.npz")
    for field in ("voxels", "map_masks", "ego_future", "boxes", "box_valid", "box_class"):
        assert np.array_equal(getattr(loaded, field), getattr(sample, field)) and getattr(loaded, field).dtype == getattr(sample, field).dtype
    assert (loaded.token, loaded.scene, loaded.v0, loaded.k0) == ("tok", "scene-0001", 4.5, -0.01)
