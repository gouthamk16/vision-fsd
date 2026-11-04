import math

import numpy as np

from conftest import make_frame, yaw_pose
from fsd.object_detection import Box3D
from fsd.tracking import GtVelocityTracker, PredictionVelocityTracker


def _box(center_ego, token="", class_name="car"):
    return Box3D(class_name, np.asarray(center_ego, dtype=np.float64), np.array([1.9, 4.6, 1.7]), 0.0, instance_token=token)


def test_constant_velocity_forward():
    tracker = GtVelocityTracker(horizons_s=(1.0, 2.0, 3.0))
    tracker.update(make_frame(yaw_pose(0.0, 0.0), 0), [_box([20.0, 2.0, 0.0], "obj")])
    obj = tracker.update(make_frame(yaw_pose(1.0, 0.0), 500_000), [_box([21.5, 2.0, 0.0], "obj")])[0]
    # Moved 2.5 m in the global frame in 0.5 s -> 5 m/s along ego +x.
    assert math.isclose(obj.speed_mps, 5.0)
    assert np.allclose(obj.velocity_ego, [5.0, 0.0])
    assert np.allclose(obj.future_xy_ego[1], [31.5, 2.0])  # 2 s horizon


def test_velocity_rotates_into_ego_frame():
    # Ego faces global +y; a global +x mover reads as ego -y motion.
    tracker = GtVelocityTracker()
    tracker.update(make_frame(yaw_pose(yaw=math.pi / 2), 0), [_box([10.0, 0.0, 0.0], "obj")])
    out = tracker.update(make_frame(yaw_pose(yaw=math.pi / 2), 500_000), [_box([10.0, -2.5, 0.0], "obj")])
    assert np.allclose(out[0].velocity_ego, [0.0, -5.0])


def test_first_sighting_has_zero_velocity():
    out = GtVelocityTracker().update(make_frame(yaw_pose(), 0), [_box([5.0, 0.0, 0.0], "obj")])
    assert out[0].speed_mps == 0.0
    assert out[0].future_xy_ego.shape == (0, 2)


def test_new_scene_clears_history():
    tracker = GtVelocityTracker()
    tracker.update(make_frame(yaw_pose(), 0), [_box([20.0, 0.0, 0.0], "obj")])
    out = tracker.update(make_frame(yaw_pose(), 500_000, scene_token="next"), [_box([22.5, 0.0, 0.0], "obj")])
    assert out[0].speed_mps == 0.0


def test_prediction_association_recovers_velocity():
    tracker = PredictionVelocityTracker()
    tracker.update(make_frame(yaw_pose(), 0), [_box([20.0, 2.0, 0.0]), _box([10.0, -3.0, 0.0])])
    out = tracker.update(make_frame(yaw_pose(), 500_000), [_box([22.5, 2.0, 0.0]), _box([10.5, -3.0, 0.0])])
    assert np.allclose(sorted(o.speed_mps for o in out), [1.0, 5.0])
    assert len({o.track_id for o in out}) == 2


def test_prediction_gate_rejects_teleport():
    tracker = PredictionVelocityTracker(gate_m=5.0)
    tracker.update(make_frame(yaw_pose(), 0), [_box([20.0, 0.0, 0.0])])
    out = tracker.update(make_frame(yaw_pose(), 500_000), [_box([40.0, 0.0, 0.0])])
    assert out[0].speed_mps == 0.0


def test_prediction_class_must_match():
    tracker = PredictionVelocityTracker()
    tracker.update(make_frame(yaw_pose(), 0), [_box([20.0, 0.0, 0.0], class_name="car")])
    out = tracker.update(make_frame(yaw_pose(), 500_000), [_box([21.0, 0.0, 0.0], class_name="truck")])
    assert out[0].speed_mps == 0.0
