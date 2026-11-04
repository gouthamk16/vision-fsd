import cv2
import numpy as np

from fsd.bev import BevGrid, resample


def test_forward_is_up_and_left_is_left():
    grid = BevGrid((-10.0, 10.0), (-10.0, 10.0), 1.0)
    (col_fwd, row_fwd), (col_left, row_left) = grid.to_pixels(np.array([[9.5, 0.0], [0.0, 9.5]]))
    assert row_fwd < 10 and col_fwd == 10
    assert col_left < 10 and row_left == 10


def test_cells_floor_points_past_the_edge_to_outside():
    # int() truncation used to fold points up to one cell past x_max into row 0.
    grid = BevGrid((-10.0, 10.0), (-10.0, 10.0), 1.0)
    rows, cols, inside = grid.cells(np.array([[10.4, 0.0], [9.6, 0.0], [0.0, -10.2]]))
    assert inside.tolist() == [False, True, False]
    assert rows[1] == 0


def test_resample_aligns_cells_across_resolutions():
    coarse = BevGrid((-10.0, 10.0), (-10.0, 10.0), 1.0)
    fine = coarse.with_resolution(0.25)
    raster = np.zeros(coarse.shape, np.float32)
    rows, cols, _ = coarse.cells(np.array([[5.5, -3.5]]))
    raster[rows[0], cols[0]] = 1.0
    up = resample(raster, coarse, fine, cv2.INTER_NEAREST)
    hot = np.argwhere(up > 0.5)
    centre = fine.to_pixels(np.array([[5.5, -3.5]]))[0][::-1]
    assert len(hot) == 16 and np.allclose(hot.mean(axis=0) + 0.5, centre)


def test_resample_by_extent_leaves_uncovered_cells_zero():
    small = BevGrid((-5.0, 5.0), (-5.0, 5.0), 1.0)
    big = BevGrid((-10.0, 10.0), (-10.0, 10.0), 1.0)
    out = resample(np.ones(small.shape, np.float32), small, big, cv2.INTER_NEAREST)
    assert out[10, 10] == 1.0 and out[0, 0] == 0.0 and out.sum() == 100
