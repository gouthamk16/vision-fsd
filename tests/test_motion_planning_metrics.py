import numpy as np

from fsd.motion_planning.metrics import planning_metrics
from fsd.motion_planning.state import EgoState, PlannedTrajectory, PlannerWorld, PlanningResult, TrajectoryPoint


def test_planning_metrics_reports_counts_and_fallback_emergency_flags():
    selected = PlannedTrajectory(
        points=[
            TrajectoryPoint(x=0.0, y=0.0, yaw=0.0, speed_mps=2.0, t=0.0),
            TrajectoryPoint(x=3.0, y=4.0, yaw=0.0, speed_mps=0.0, t=1.0),
        ],
        cost=7.5,
        is_emergency_stop=True,
    )
    other = PlannedTrajectory(points=[], cost=9.0, is_emergency_stop=False)
    world = PlannerWorld(
        ego=EgoState(
            x_global=0.0,
            y_global=0.0,
            yaw_global=0.0,
            speed_mps=0.0,
            yaw_rate_rps=0.0,
            timestamp_us=1,
            fallback_initial_speed=True,
        ),
        collision_grid=np.zeros((2, 2), dtype=bool),
        occupancy_probability=np.zeros((2, 2), dtype=np.float32),
        height_range=np.zeros((2, 2), dtype=np.float32),
        x_range=(0.0, 1.0),
        y_range=(0.0, 1.0),
        resolution=0.5,
    )
    result = PlanningResult(
        selected=selected,
        candidates=[selected, other],
        valid_candidates=[selected],
        world=world,
        reason="no_valid_trajectory",
    )

    metrics = planning_metrics(result)

    assert metrics["candidate_count"] == 2.0
    assert metrics["valid_candidate_count"] == 1.0
    assert metrics["selected_cost"] == 7.5
    assert metrics["selected_path_length_m"] == 5.0
    assert metrics["emergency_stop"] == 1.0
    assert metrics["fallback_initial_speed"] == 1.0
