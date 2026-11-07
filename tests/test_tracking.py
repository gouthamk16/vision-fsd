import math

import numpy as np

from conftest import make_frame, yaw_pose
from fsd.object_detection import Box3D
from fsd.tracking import GtVelocityTracker, PredictionVelocityTracker

STEP_US = 500_000


def _box(center_ego, token="", class_name="car", yaw=0.0):
    return Box3D(class_name, np.asarray(center_ego, dtype=np.float64), np.array([1.9, 4.6, 1.7]), yaw, instance_token=token)


def _frame(i, pose=None, scene="scene"):
    return make_frame(pose or yaw_pose(), i * STEP_US, scene)


def test_gt_track_converges_to_the_true_velocity():
    tracker = GtVelocityTracker(horizons_s=(1.0, 2.0, 3.0))
    for i in range(12):  # 5 m/s along global +x, ego parked at the origin
        obj = tracker.update(_frame(i), [_box([20.0 + 2.5 * i, 2.0, 0.0], "obj")])[0]
    assert np.allclose(obj.velocity_ego, [5.0, 0.0], atol=0.3)
    assert np.allclose(obj.future_xy_ego[1], obj.smoothed.center_ego[:2] + [10.0, 0.0], atol=0.6)


def test_velocity_rotates_into_the_ego_frame():
    # Ego faces global +y; a global +x mover reads as ego -y motion.
    tracker, pose = GtVelocityTracker(), yaw_pose(yaw=math.pi / 2)
    for i in range(12):
        obj = tracker.update(_frame(i, pose), [_box([10.0, -2.5 * i, 0.0], "obj", yaw=-math.pi / 2)])[0]
    assert np.allclose(obj.velocity_ego, [0.0, -5.0], atol=0.3)


def test_first_sighting_is_still():
    obj = GtVelocityTracker().update(_frame(0), [_box([5.0, 0.0, 0.0], "obj")])[0]
    assert obj.speed_mps == 0.0 and obj.future_xy_ego.shape == (0, 2)


def test_new_scene_starts_new_tracks():
    tracker = GtVelocityTracker()
    for i in range(6):
        tracker.update(_frame(i), [_box([20.0 + 2.5 * i, 0.0, 0.0], "obj")])
    assert tracker.update(_frame(6, scene="next"), [_box([35.0, 0.0, 0.0], "obj")])[0].speed_mps == 0.0


def test_prediction_tracks_keep_identity_and_converge():
    tracker = PredictionVelocityTracker()
    for i in range(12):
        out = tracker.update(_frame(i), [_box([20.0 + 2.5 * i, 2.0, 0.0]), _box([10.0 + 0.5 * i, -3.0, 0.0])])
        ids = ids if i else [o.track_id for o in out]
        assert [o.track_id for o in out] == ids
    assert np.allclose([o.speed_mps for o in out], [5.0, 1.0], atol=0.3)


def test_a_missed_detection_coasts_and_gates_on_the_prediction():
    tracker = PredictionVelocityTracker(gate_m=5.0)
    for i in range(8):  # 9 m/s: 4.5 m per frame, inside the gate
        first = tracker.update(_frame(i), [_box([4.5 * i, 0.0, 0.0])])[0].track_id
    tracker.update(_frame(8), [])  # missed: the track coasts
    # 9 m from its last detection, beyond the 5 m gate, but where the filter predicts it.
    assert tracker.update(_frame(9), [_box([4.5 * 9, 0.0, 0.0])])[0].track_id == first


def test_a_track_retires_after_max_age():
    tracker = PredictionVelocityTracker(max_age_s=0.5)
    first = tracker.update(_frame(0), [_box([20.0, 0.0, 0.0])])[0].track_id
    tracker.update(_frame(1), [])
    tracker.update(_frame(2), [])
    assert tracker.update(_frame(3), [_box([20.0, 0.0, 0.0])])[0].track_id != first


def test_class_must_match():
    tracker = PredictionVelocityTracker()
    first = tracker.update(_frame(0), [_box([20.0, 0.0, 0.0], class_name="car")])[0].track_id
    assert tracker.update(_frame(1), [_box([20.5, 0.0, 0.0], class_name="truck")])[0].track_id != first


def test_a_teleport_starts_a_new_track():
    tracker = PredictionVelocityTracker(gate_m=5.0)
    first = tracker.update(_frame(0), [_box([20.0, 0.0, 0.0])])[0].track_id
    assert tracker.update(_frame(1), [_box([40.0, 0.0, 0.0])])[0].track_id != first


def test_a_flipped_heading_is_folded_onto_the_track():
    # A box looks the same from either end, so a detector can report the heading reversed for a frame.
    tracker = PredictionVelocityTracker()
    for i in range(10):
        obj = tracker.update(_frame(i), [_box([20.0 + 2.5 * i, 0.0, 0.0], yaw=math.pi if i == 9 else 0.0)])[0]
    assert abs(obj.smoothed.yaw_ego) < math.radians(5.0) and obj.velocity_ego[0] > 4.0


def test_smoothed_boxes_jitter_less_than_detections():
    rng = np.random.default_rng(0)
    tracker, raw, smooth = PredictionVelocityTracker(), [], []
    for i in range(30):  # a parked car seen with 0.4 m of position noise
        centre = np.array([15.0, 3.0, 0.0]) + np.r_[rng.normal(0.0, 0.4, 2), 0.0]
        obj = tracker.update(_frame(i), [_box(centre)])[0]
        raw.append(centre[:2])
        smooth.append(obj.smoothed.center_ego[:2])
    # Across the heading the filter all but removes it; along it, the noise is partly read as speed changes.
    assert np.linalg.norm(np.std(smooth[10:], axis=0)) < 0.7 * np.linalg.norm(np.std(raw[10:], axis=0))
