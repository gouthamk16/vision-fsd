from pathlib import Path

import numpy as np

from fsd.data import CameraFrame, SurroundFrame
from fsd.object_detection import Box3D, box_bottom_corners_ego
from fsd.tracking import GtVelocityTracker


def _frame(ego_translation, ego_rotation, timestamp_us):
    pose = {"rotation": list(ego_rotation), "translation": list(ego_translation)}
    cam = CameraFrame("CAM_FRONT", Path("x.jpg"), timestamp_us, "sd", {}, pose)
    return SurroundFrame("scene", "scene-x", "sample", 0, timestamp_us, {"CAM_FRONT": cam})


def _box(center_ego, token):
    center_ego = np.asarray(center_ego, dtype=np.float64)
    size = np.array([1.9, 4.6, 1.7])
    return Box3D(
        sample_token="s",
        annotation_token="a",
        class_name="car",
        raw_category="vehicle.car",
        center_ego=center_ego,
        size=size,
        yaw_ego=0.0,
        corners_ego=box_bottom_corners_ego(center_ego, size, 0.0),
        num_lidar_pts=10,
        num_radar_pts=0,
        instance_token=token,
    )


def test_constant_velocity_forward():
    tracker = GtVelocityTracker(horizons_s=(1.0, 2.0, 3.0))
    identity = [1.0, 0.0, 0.0, 0.0]

    tracker.update(_frame([0.0, 0.0, 0.0], identity, 0), [_box([20.0, 2.0, 0.0], "obj")])
    out = tracker.update(_frame([1.0, 0.0, 0.0], identity, 500_000), [_box([21.5, 2.0, 0.0], "obj")])

    obj = out[0]
    # object moved 2.5 m global in 0.5 s -> 5 m/s along ego +x
    assert abs(obj.speed_mps - 5.0) < 1e-6
    assert np.allclose(obj.velocity_ego, [5.0, 0.0], atol=1e-6)
    assert obj.future_xy_ego.shape == (3, 2)
    assert np.allclose(obj.future_xy_ego[1], [31.5, 2.0], atol=1e-6)  # 2 s horizon


def test_velocity_rotates_into_ego_frame():
    # ego yaw +90 deg (faces global +y); a global +x mover reads as ego -y motion.
    tracker = GtVelocityTracker()
    yaw90 = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]

    tracker.update(_frame([0.0, 0.0, 0.0], yaw90, 0), [_box([10.0, 0.0, 0.0], "obj")])
    out = tracker.update(_frame([0.0, 0.0, 0.0], yaw90, 500_000), [_box([10.0, -2.5, 0.0], "obj")])

    assert np.allclose(out[0].velocity_ego, [0.0, -5.0], atol=1e-6)


def test_first_sighting_has_zero_velocity():
    tracker = GtVelocityTracker()
    out = tracker.update(_frame([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], 0), [_box([5.0, 0.0, 0.0], "obj")])
    assert out[0].speed_mps == 0.0
    assert out[0].future_xy_ego.shape == (0, 2)


def test_reset_clears_history():
    tracker = GtVelocityTracker()
    identity = [1.0, 0.0, 0.0, 0.0]
    tracker.update(_frame([0.0, 0.0, 0.0], identity, 0), [_box([20.0, 0.0, 0.0], "obj")])
    tracker.reset()
    out = tracker.update(_frame([0.0, 0.0, 0.0], identity, 500_000), [_box([22.5, 0.0, 0.0], "obj")])
    assert out[0].speed_mps == 0.0  # no carry-over across reset
