import math

from fsd.motion_planning.ego_motion import estimate_ego_state, wrap_angle


def quat_from_yaw(yaw: float) -> list[float]:
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def pose(x: float, y: float, yaw: float) -> dict:
    return {"translation": [x, y, 0.0], "rotation": quat_from_yaw(yaw)}


def test_wrap_angle_maps_to_minus_pi_pi():
    assert math.isclose(wrap_angle(3.5), -2.7831853071795862)
    assert math.isclose(wrap_angle(-3.5), 2.7831853071795862)


def test_first_frame_uses_zero_speed_fallback():
    ego = estimate_ego_state(
        current_pose=pose(5.0, 7.0, 0.25),
        current_timestamp_us=1_000_000,
        previous_pose=None,
        previous_timestamp_us=None,
    )

    assert ego.x_global == 5.0
    assert ego.y_global == 7.0
    assert math.isclose(ego.yaw_global, 0.25)
    assert ego.speed_mps == 0.0
    assert ego.yaw_rate_rps == 0.0
    assert ego.fallback_initial_speed


def test_speed_and_yaw_rate_from_pose_delta():
    ego = estimate_ego_state(
        current_pose=pose(3.0, 4.0, 0.4),
        current_timestamp_us=2_000_000,
        previous_pose=pose(0.0, 0.0, 0.1),
        previous_timestamp_us=1_000_000,
    )

    assert math.isclose(ego.speed_mps, 5.0)
    assert math.isclose(ego.yaw_rate_rps, 0.3)
    assert not ego.fallback_initial_speed
