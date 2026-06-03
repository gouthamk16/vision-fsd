import math

import pytest

from fsd.motion_planning.sampler import LatticeSampler, LatticeSamplerConfig
from fsd.motion_planning.state import EgoState
from fsd.motion_planning.trajectory import make_emergency_stop, time_values


def ego(speed: float = 4.0) -> EgoState:
    return EgoState(
        x_global=0.0,
        y_global=0.0,
        yaw_global=0.0,
        speed_mps=speed,
        yaw_rate_rps=0.0,
        timestamp_us=1,
    )


def test_sampler_generates_timed_trajectories():
    sampler = LatticeSampler(
        LatticeSamplerConfig(
            horizon_s=1.0,
            dt_s=0.5,
            target_speeds_mps=(2.0, 4.0),
            curvatures=(-0.1, 0.0, 0.1),
        )
    )

    trajectories = sampler.sample(ego(speed=4.0))

    assert len(trajectories) == 6
    assert [p.t for p in trajectories[0].points] == [0.0, 0.5, 1.0]
    assert all(traj.points[0].x == 0.0 and traj.points[0].y == 0.0 for traj in trajectories)
    assert any(traj.metadata["curvature"] == 0.1 for traj in trajectories)


def test_straight_trajectory_moves_forward():
    sampler = LatticeSampler(
        LatticeSamplerConfig(
            horizon_s=1.0,
            dt_s=0.5,
            target_speeds_mps=(2.0,),
            curvatures=(0.0,),
        )
    )

    traj = sampler.sample(ego(speed=2.0))[0]

    assert traj.points[-1].x > 1.5
    assert math.isclose(traj.points[-1].y, 0.0, abs_tol=1e-6)
    assert math.isclose(traj.points[-1].yaw, 0.0, abs_tol=1e-6)


def test_straight_trajectory_uses_average_interval_speed():
    sampler = LatticeSampler(
        LatticeSamplerConfig(
            horizon_s=1.0,
            dt_s=0.5,
            target_speeds_mps=(10.0,),
            curvatures=(0.0,),
        )
    )

    traj = sampler.sample(ego(speed=0.0))[0]

    assert math.isclose(traj.points[-1].x, 5.0, abs_tol=1e-6)
    assert math.isclose(traj.points[-1].y, 0.0, abs_tol=1e-6)


def test_curved_trajectory_uses_constant_curvature_arc_step():
    sampler = LatticeSampler(
        LatticeSamplerConfig(
            horizon_s=1.0,
            dt_s=0.25,
            target_speeds_mps=(5.0,),
            curvatures=(0.1,),
        )
    )

    traj = sampler.sample(ego(speed=5.0))[0]
    distance = 5.0
    curvature = 0.1

    assert math.isclose(traj.points[-1].yaw, curvature * distance, abs_tol=1e-6)
    assert math.isclose(traj.points[-1].x, math.sin(curvature * distance) / curvature, abs_tol=1e-6)
    assert math.isclose(traj.points[-1].y, (1.0 - math.cos(curvature * distance)) / curvature, abs_tol=1e-6)


def test_emergency_stop_rolls_forward_while_braking():
    traj = make_emergency_stop(ego(speed=3.0), horizon_s=1.0, dt_s=0.5)

    assert traj.is_emergency_stop
    assert [p.t for p in traj.points] == [0.0, 0.5, 1.0]
    assert [p.speed_mps for p in traj.points] == [3.0, 1.5, 0.0]
    assert traj.points[0].x == 0.0
    assert traj.points[0].x < traj.points[1].x < traj.points[2].x
    assert math.isclose(traj.points[-1].x, 1.5, abs_tol=1e-6)
    assert all(math.isclose(p.y, 0.0, abs_tol=1e-6) for p in traj.points)
    assert all(math.isclose(p.yaw, 0.0, abs_tol=1e-6) for p in traj.points)


def test_time_values_rejects_non_finite_inputs():
    with pytest.raises(ValueError):
        time_values(float("nan"), 0.5)
    with pytest.raises(ValueError):
        time_values(float("inf"), 0.5)
    with pytest.raises(ValueError):
        time_values(1.0, float("nan"))


def test_time_values_does_not_duplicate_rounded_horizon():
    assert time_values(1.00000000001, 0.5) == [0.0, 0.5, 1.0]


def test_emergency_stop_zero_horizon_reports_stopped_state():
    traj = make_emergency_stop(ego(speed=3.0), horizon_s=0.0, dt_s=0.5)

    assert traj.is_emergency_stop
    assert len(traj.points) == 1
    assert traj.points[0].t == 0.0
    assert traj.points[0].speed_mps == 0.0
    assert traj.points[0].x == 0.0


def test_time_values_keeps_tiny_positive_horizon():
    horizon_s = 0.00000000001

    assert time_values(0.0, 0.5) == [0.0]
    assert time_values(horizon_s, 0.5) == [0.0, horizon_s]
