import json
import math

import numpy as np

from conftest import yaw_pose
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
