import math

from conftest import yaw_pose
from fsd.ego_motion import estimate_ego_state, wrap_angle


def test_wrap_angle_maps_to_minus_pi_pi():
    assert math.isclose(wrap_angle(3.5), 3.5 - 2 * math.pi)
    assert math.isclose(wrap_angle(-3.5), 2 * math.pi - 3.5)


def test_first_frame_uses_zero_speed_fallback():
    ego = estimate_ego_state(yaw_pose(5.0, 7.0, 0.25), 1_000_000, None, None)
    assert ego.speed_mps == 0.0
    assert ego.yaw_rate_rps == 0.0
    assert ego.fallback_initial_speed


def test_speed_and_yaw_rate_from_pose_delta():
    ego = estimate_ego_state(yaw_pose(3.0, 4.0, 0.4), 2_000_000, yaw_pose(0.0, 0.0, 0.1), 1_000_000)
    assert math.isclose(ego.speed_mps, 5.0)
    assert math.isclose(ego.yaw_rate_rps, 0.3)
    assert not ego.fallback_initial_speed


def test_non_increasing_time_falls_back():
    ego = estimate_ego_state(yaw_pose(3.0, 4.0), 1_000_000, yaw_pose(), 1_000_000)
    assert ego.fallback_initial_speed and ego.speed_mps == 0.0
