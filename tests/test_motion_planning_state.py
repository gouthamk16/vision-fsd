import numpy as np

from fsd.motion_planning.state import (
    EgoState,
    LaneContext,
    PlannedTrajectory,
    PlannerWorld,
    TrajectoryPoint,
)


def test_planned_trajectory_duration_and_endpoint():
    points = [
        TrajectoryPoint(x=0.0, y=0.0, yaw=0.0, speed_mps=2.0, t=0.0),
        TrajectoryPoint(x=1.0, y=0.0, yaw=0.0, speed_mps=2.0, t=0.5),
        TrajectoryPoint(x=2.0, y=0.0, yaw=0.0, speed_mps=2.0, t=1.0),
    ]
    traj = PlannedTrajectory(points=points, cost=3.5, is_emergency_stop=False)

    assert traj.duration_s == 1.0
    assert traj.endpoint_xy == (2.0, 0.0)


def test_planned_trajectory_empty_points_has_zero_duration_and_origin_endpoint():
    traj = PlannedTrajectory(points=[], cost=0.0, is_emergency_stop=True)

    assert traj.duration_s == 0.0
    assert traj.endpoint_xy == (0.0, 0.0)


def test_planner_world_accepts_missing_lane_context():
    ego = EgoState(
        x_global=10.0,
        y_global=20.0,
        yaw_global=0.2,
        speed_mps=4.0,
        yaw_rate_rps=0.01,
        timestamp_us=123,
        fallback_initial_speed=False,
    )
    collision = np.zeros((4, 5), dtype=bool)
    world = PlannerWorld(
        ego=ego,
        collision_grid=collision,
        occupancy_probability=np.full((4, 5), 0.5, dtype=np.float32),
        height_range=np.zeros((4, 5), dtype=np.float32),
        x_range=(-2.0, 2.0),
        y_range=(-2.5, 2.5),
        resolution=1.0,
        lane_context=None,
    )

    assert world.lane_context is None
    assert world.collision_grid.shape == (4, 5)


def test_lane_context_is_optional_confidence_container():
    lane = LaneContext(centerline_xy=np.array([[0.0, 0.0], [5.0, 0.2]]), confidence=0.7)

    assert lane.has_high_confidence(0.6)
    assert not lane.has_high_confidence(0.8)


def test_lane_context_rejects_1d_centerline():
    lane = LaneContext(centerline_xy=np.array([0.0, 1.0]), confidence=1.0)

    assert not lane.has_high_confidence(0.5)
