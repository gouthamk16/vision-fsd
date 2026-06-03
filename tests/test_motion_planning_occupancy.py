import numpy as np

from fsd.motion_planning.occupancy import CollisionGrid, build_collision_grid


def test_build_collision_grid_uses_occupancy_and_height():
    prob = np.array([[0.5, 0.7], [0.2, 0.4]], dtype=np.float32)
    height_range = np.array([[0.1, 0.1], [0.8, 0.2]], dtype=np.float32)

    grid = build_collision_grid(
        occupancy_probability=prob,
        height_range=height_range,
        x_range=(-1.0, 1.0),
        y_range=(-1.0, 1.0),
        resolution=1.0,
        occupancy_threshold=0.62,
        height_threshold=0.5,
    )

    assert grid.blocked.tolist() == [[False, True], [True, False]]


def test_xy_to_row_col_matches_existing_bev_orientation():
    grid = CollisionGrid(
        blocked=np.zeros((4, 4), dtype=bool),
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
        resolution=1.0,
    )

    assert grid.xy_to_row_col(1.5, 1.5) == (0, 0)
    assert grid.xy_to_row_col(-1.5, -1.5) == (3, 3)
    assert grid.xy_to_row_col(2.1, 0.0) is None


def test_xy_to_row_col_documents_upper_inclusive_lower_exclusive_bounds():
    grid = CollisionGrid(
        blocked=np.zeros((4, 4), dtype=bool),
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
        resolution=1.0,
    )

    assert grid.xy_to_row_col(2.0, 2.0) == (0, 0)
    assert grid.xy_to_row_col(-2.0, 0.0) is None
    assert grid.xy_to_row_col(0.0, -2.0) is None


def test_vehicle_footprint_checks_radius_cells():
    blocked = np.zeros((5, 5), dtype=bool)
    blocked[2, 2] = True
    grid = CollisionGrid(
        blocked=blocked,
        x_range=(-2.5, 2.5),
        y_range=(-2.5, 2.5),
        resolution=1.0,
    )

    assert grid.footprint_blocked(x=0.0, y=0.0, radius_m=0.4)
    assert grid.footprint_blocked(x=0.9, y=0.0, radius_m=1.0)
    assert not grid.footprint_blocked(x=2.0, y=2.0, radius_m=0.4)


def test_vehicle_footprint_extending_outside_grid_is_blocked():
    grid = CollisionGrid(
        blocked=np.zeros((5, 5), dtype=bool),
        x_range=(-2.5, 2.5),
        y_range=(-2.5, 2.5),
        resolution=1.0,
    )

    assert grid.footprint_blocked(x=2.0, y=0.0, radius_m=1.0)
    assert grid.footprint_blocked(x=0.0, y=2.0, radius_m=1.0)


def test_vehicle_footprint_uses_cell_area_intersection_not_square_window():
    blocked = np.zeros((5, 5), dtype=bool)
    blocked[1, 1] = True
    grid = CollisionGrid(
        blocked=blocked,
        x_range=(-2.5, 2.5),
        y_range=(-2.5, 2.5),
        resolution=1.0,
    )

    assert not grid.footprint_blocked(x=0.0, y=0.0, radius_m=0.4)
    assert grid.footprint_blocked(x=0.0, y=0.0, radius_m=1.0)
