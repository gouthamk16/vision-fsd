import numpy as np

from fsd.motion_planning.costs import score_trajectory
from fsd.motion_planning.occupancy import build_collision_grid
from fsd.motion_planning.planner import LocalPlanner, LocalPlannerConfig
from fsd.motion_planning.state import EgoState, PlannedTrajectory, PlannerWorld, TrajectoryPoint


def world(blocked_column: int | None = None) -> PlannerWorld:
    prob = np.full((40, 40), 0.5, dtype=np.float32)
    height = np.zeros((40, 40), dtype=np.float32)
    if blocked_column is not None:
        prob[15:25, blocked_column] = 0.9
    grid = build_collision_grid(prob, height, (-10.0, 10.0), (-10.0, 10.0), 0.5)
    return PlannerWorld(
        ego=EgoState(
            x_global=0.0,
            y_global=0.0,
            yaw_global=0.0,
            speed_mps=4.0,
            yaw_rate_rps=0.0,
            timestamp_us=1,
        ),
        collision_grid=grid.blocked,
        occupancy_probability=prob,
        height_range=height,
        x_range=(-10.0, 10.0),
        y_range=(-10.0, 10.0),
        resolution=0.5,
        lane_context=None,
    )


def test_planner_selects_forward_trajectory_in_clear_space():
    planner = LocalPlanner(LocalPlannerConfig(horizon_s=1.0, dt_s=0.5))

    result = planner.plan(world())

    assert result.reason == "selected"
    assert not result.selected.is_emergency_stop
    assert result.selected.points[-1].x > 1.0
    assert len(result.valid_candidates) > 0


def test_planner_falls_back_to_emergency_stop_when_all_candidates_collide():
    w = world()
    w.collision_grid[:] = True
    planner = LocalPlanner(LocalPlannerConfig(horizon_s=1.0, dt_s=0.5))

    result = planner.plan(w)

    assert result.reason == "no_valid_trajectory"
    assert result.selected.is_emergency_stop
    assert result.selected.points[-1].speed_mps == 0.0


def simple_traj(curvature: float) -> PlannedTrajectory:
    return PlannedTrajectory(
        points=[
            TrajectoryPoint(0.0, 0.0, 0.0, 4.0, 0.0),
            TrajectoryPoint(2.0, 0.0, 0.0, 4.0, 1.0),
        ],
        cost=0.0,
        is_emergency_stop=False,
        metadata={"curvature": curvature, "target_speed_mps": 4.0},
    )


def test_curvature_cost_is_symmetric_for_left_and_right_turns():
    w = world()

    left = score_trajectory(simple_traj(-0.12), w)
    right = score_trajectory(simple_traj(0.12), w)

    assert left == right


def test_emergency_fallback_records_collision_validation_status():
    w = world()
    w.collision_grid[:] = True
    planner = LocalPlanner(LocalPlannerConfig(horizon_s=1.0, dt_s=0.5))

    result = planner.plan(w)

    assert result.reason == "no_valid_trajectory"
    assert result.selected.is_emergency_stop
    assert result.selected.metadata["fallback_collision_free"] == 0.0
    assert result.selected.metadata["fallback_validation_reason"] == "collision"
