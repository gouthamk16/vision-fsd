import json
import math
from pathlib import Path

import numpy as np
import pytest

from conftest import make_frame, yaw_pose
from fsd.data import Pose, _group_by_nearest, quaternion_to_rotation_matrix
from fsd.nuscenes_index import iter_json_array


def test_quaternion_yaw_matches_pose_yaw():
    yaw = 0.7
    pose = Pose(quaternion_to_rotation_matrix([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]), np.zeros(3))
    assert math.isclose(pose.yaw, yaw)


def test_pose_apply_invert_round_trip():
    pose = yaw_pose(10.0, -3.0, 1.1)
    points = np.array([[1.0, 2.0, 0.5], [-4.0, 0.0, 1.0]])
    assert np.allclose(pose.invert(pose.apply(points)), points)


def test_relative_to_maps_between_ego_frames():
    previous, current = yaw_pose(0.0, 0.0, 0.0), yaw_pose(2.0, 0.0, math.pi / 2)
    # A point 1 m ahead of the current ego is at global (2, 1), i.e. (2, 1) in the previous frame.
    assert np.allclose(current.relative_to(previous).apply(np.array([[1.0, 0.0, 0.0]])), [[2.0, 1.0, 0.0]])


def test_iter_json_array_handles_any_layout_and_chunk_boundaries(tmp_json):
    records = [{"token": f"t{i}", "nested": {"a": [i, i + 1]}} for i in range(50)]
    for text in (json.dumps(records), json.dumps(records, indent=4)):
        assert list(iter_json_array(tmp_json(text), chunk_size=7)) == records
    assert list(iter_json_array(tmp_json("[ ]"))) == []


def test_group_by_nearest_drops_targets_without_a_close_partner():
    target = [{"timestamp": t} for t in (0, 100, 200)]
    other = [{"timestamp": t} for t in (5, 290)]
    groups = _group_by_nearest({"A": target, "B": other}, "A", tolerance_us=20)
    assert [g["B"]["timestamp"] for g in groups] == [5]


def _loader_with_samples(n: int):
    from types import SimpleNamespace

    from fsd.data import NuScenesSceneLoader

    tokens = [f"s{i}" for i in range(n)]
    samples = {t: {"token": t, "next": tokens[i + 1] if i + 1 < n else ""} for i, t in enumerate(tokens)}
    loader = NuScenesSceneLoader.__new__(NuScenesSceneLoader)
    loader.index = SimpleNamespace(samples=samples)
    return loader, {"first_sample_token": "s0"}


def test_samples_slices_start_and_max_frames():
    loader, scene = _loader_with_samples(3)
    assert [s["token"] for s in loader.samples(scene, 1, 1)] == ["s1"]
    assert loader.samples(scene, 2, 0) == []
    with pytest.raises(IndexError, match="beyond scene length 3"):
        loader.samples(scene, 3)



def _loader_with_sweeps(n: int, period_us: int):
    """LIDAR_TOP sweeps every `period_us`, ego at x = seconds since the first; every 10th is a keyframe."""
    from types import SimpleNamespace

    from fsd.data import LIDAR_TOP, NuScenesSceneLoader

    records = [
        {"token": f"l{i}", "timestamp": i * period_us, "ego_pose_token": f"p{i}", "calibrated_sensor_token": "c",
         "sample_token": f"s{i // 10}", "is_key_frame": i % 10 == 0, "filename": f"l{i}.bin"}
        for i in range(n)
    ]
    loader = NuScenesSceneLoader.__new__(NuScenesSceneLoader)
    loader.dataroot = Path("root")
    cameras = [  # CAM_FRONT at ~12 Hz
        {"timestamp": t, "ego_pose_token": "p0", "calibrated_sensor_token": "c", "is_key_frame": False, "filename": f"cam{t}.jpg"}
        for t in range(0, n * period_us, 83_000)
    ]
    loader.index = SimpleNamespace(
        sweeps={"scene": {LIDAR_TOP: records, "CAM_FRONT": cameras}},
        ego_poses={f"p{i}": {"translation": [i * period_us / 1e6, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]} for i in range(n)},
        calibrated_sensors={"c": {"translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0], "camera_intrinsic": np.eye(3).tolist()}},
    )
    loader.scenes = [{"token": "scene", "name": "scene-x", "location": "boston-seaport"}]
    loader._lidar_position = {r["token"]: i for i, r in enumerate(records)}
    return loader


def test_future_ego_poses_steps_in_time_and_stops_at_scene_end():
    loader = _loader_with_sweeps(n=33, period_us=50_000)  # 1.6 s of sweeps
    poses = loader.future_ego_poses(make_frame(yaw_pose(), timestamp_us=100_000, scene_token="scene"), steps=6)
    assert [round(float(p.translation[0]), 2) for p in poses] == [0.6, 1.1, 1.6]


def test_lidar_sweep_frames_every_stride_with_history():
    loader = _loader_with_sweeps(n=7, period_us=50_000)
    frames = list(loader.iter_lidar_sweep_frames(scene_name="scene-x", stride=2, lidar_sweeps=3))
    assert [f.timestamp_us for f, _ in frames] == [0, 100_000, 200_000, 300_000]
    assert [len(lidar.sweeps) for _, lidar in frames] == [1, 3, 3, 3]
    assert frames[2][0].ego_pose.translation[0] == 0.2 and frames[0][0].is_key_frame
    assert [f.cameras["CAM_FRONT"].timestamp_us for f, _ in frames] == [0, 83_000, 166_000, 332_000]
