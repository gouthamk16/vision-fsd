import numpy as np
import pytest

from fsd.motion_planning.occupancy import CollisionGrid
from fsd.motion_planning.state import PlannedTrajectory, TrajectoryPoint
from fsd.motion_planning.validator import TrajectoryValidator, TrajectoryValidatorConfig


def trajectory(points):
    return PlannedTrajectory(points=points, cost=0.0, is_emergency_stop=False)


def test_validator_accepts_clear_ordered_trajectory():
    grid = CollisionGrid(np.zeros((10, 10), dtype=bool), (-5.0, 5.0), (-5.0, 5.0), 1.0)
    validator = TrajectoryValidator(TrajectoryValidatorConfig(vehicle_radius_m=0.4))
    traj = trajectory(
        [
            TrajectoryPoint(0.0, 0.0, 0.0, 2.0, 0.0),
            TrajectoryPoint(1.0, 0.0, 0.0, 2.0, 0.5),
        ]
    )

    result = validator.validate(traj, grid)

    assert result.valid
    assert result.reason == "valid"


def test_validator_rejects_collision():
    blocked = np.zeros((10, 10), dtype=bool)
    blocked[4, 5] = True
    grid = CollisionGrid(blocked, (-5.0, 5.0), (-5.0, 5.0), 1.0)
    validator = TrajectoryValidator(TrajectoryValidatorConfig(vehicle_radius_m=0.4))
    traj = trajectory(
        [
            TrajectoryPoint(0.0, 0.0, 0.0, 2.0, 0.0),
            TrajectoryPoint(1.0, 0.0, 0.0, 2.0, 0.5),
        ]
    )

    result = validator.validate(traj, grid)

    assert not result.valid
    assert result.reason == "collision"


def test_validator_rejects_swept_collision_between_points():
    blocked = np.zeros((5, 5), dtype=bool)
    blocked[2, 2] = True
    grid = CollisionGrid(blocked, (0.0, 5.0), (0.0, 5.0), 1.0)
    validator = TrajectoryValidator(TrajectoryValidatorConfig(vehicle_radius_m=0.4))
    traj = trajectory(
        [
            TrajectoryPoint(1.4, 2.5, 0.0, 2.0, 0.0),
            TrajectoryPoint(3.6, 2.5, 0.0, 2.0, 1.0),
        ]
    )

    result = validator.validate(traj, grid)

    assert not result.valid
    assert result.reason == "collision"


def test_validator_rejects_non_increasing_time():
    grid = CollisionGrid(np.zeros((10, 10), dtype=bool), (-5.0, 5.0), (-5.0, 5.0), 1.0)
    validator = TrajectoryValidator()
    traj = trajectory(
        [
            TrajectoryPoint(0.0, 0.0, 0.0, 2.0, 0.5),
            TrajectoryPoint(1.0, 0.0, 0.0, 2.0, 0.5),
        ]
    )

    result = validator.validate(traj, grid)

    assert not result.valid
    assert result.reason == "non_increasing_time"


def test_validator_rejects_empty_trajectory():
    grid = CollisionGrid(np.zeros((10, 10), dtype=bool), (-5.0, 5.0), (-5.0, 5.0), 1.0)
    result = TrajectoryValidator().validate(trajectory([]), grid)

    assert not result.valid
    assert result.reason == "empty"


def test_validator_rejects_non_finite_point_values():
    grid = CollisionGrid(np.zeros((10, 10), dtype=bool), (-5.0, 5.0), (-5.0, 5.0), 1.0)
    traj = trajectory([TrajectoryPoint(float("nan"), 0.0, 0.0, 2.0, 0.0)])

    result = TrajectoryValidator().validate(traj, grid)

    assert not result.valid
    assert result.reason == "non_finite"


def test_validator_treats_out_of_bounds_as_collision():
    grid = CollisionGrid(np.zeros((10, 10), dtype=bool), (-5.0, 5.0), (-5.0, 5.0), 1.0)
    validator = TrajectoryValidator(TrajectoryValidatorConfig(vehicle_radius_m=0.4))
    traj = trajectory([TrajectoryPoint(10.0, 0.0, 0.0, 2.0, 0.0)])

    result = validator.validate(traj, grid)

    assert not result.valid
    assert result.reason == "collision"


def test_validator_rejects_invalid_vehicle_radius_config():
    with pytest.raises(ValueError):
        TrajectoryValidator(TrajectoryValidatorConfig(vehicle_radius_m=-0.1))
    with pytest.raises(ValueError):
        TrajectoryValidator(TrajectoryValidatorConfig(vehicle_radius_m=float("nan")))
