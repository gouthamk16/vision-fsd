import pytest

from fsd.nmp.config import NmpConfig


def test_obstacle_dilation_must_be_a_whole_number_of_cells():
    with pytest.raises(ValueError, match="obstacle_dilation"):
        NmpConfig(resolution=0.3)  # 1.0 m would round to 3 cells = 0.9 m, which no longer covers the ego body


def test_obstacle_dilation_must_exceed_the_ego_half_width():
    with pytest.raises(ValueError, match="obstacle_dilation"):
        NmpConfig(obstacle_dilation=0.8)


def test_mirroring_needs_a_grid_symmetric_about_the_ego():
    with pytest.raises(ValueError, match="mirror"):
        NmpConfig(y_range=(-40.0, 30.0))
    NmpConfig(y_range=(-40.0, 30.0), mirror_p=0.0)
