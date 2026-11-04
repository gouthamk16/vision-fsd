import math

import numpy as np

from conftest import yaw_pose
from fsd.fusion_detect import _fit_footprint
from fsd.object_detection import Box3D, box_from_global


def test_footprint_order_and_heading():
    box = Box3D("car", np.array([10.0, 0.0, 0.0]), np.array([2.0, 4.0, 1.5]), math.pi / 2)
    fl, fr, rr, rl = box.footprint
    assert np.allclose(fl, [9.0, 2.0]) and np.allclose(fr, [11.0, 2.0])
    assert np.allclose(rr, [11.0, -2.0]) and np.allclose(rl, [9.0, -2.0])


def test_box_from_global_uses_the_reference_pose():
    record = {"translation": [12.0, 5.0, 1.0], "size": [2, 4, 1.5], "rotation": [math.cos(0.5), 0, 0, math.sin(0.5)]}
    box = box_from_global(record, yaw_pose(2.0, 5.0, 1.0), "car")
    assert np.allclose(box.center_ego[:2], [10 * math.cos(1.0), -10 * math.sin(1.0)])
    assert math.isclose(box.yaw_ego, 0.0, abs_tol=1e-9)


def test_rear_face_gives_forward_heading_and_pushes_centre_back():
    # Rear bumper of a car 10 m ahead: points spread across y, nearly constant x.
    y = np.linspace(-0.9, 0.9, 20)
    xy = np.stack([10.0 + 0.02 * np.sin(7 * y), y], axis=1)
    center, yaw = _fit_footprint(xy, width=1.9, length=4.6)
    assert abs(yaw) < 0.1
    assert math.isclose(center[0], 10.0 + 2.3, abs_tol=0.1)


def test_side_face_keeps_heading_along_the_visible_length():
    x = np.linspace(8.0, 12.4, 30)
    xy = np.stack([x, 3.0 + 0.02 * np.cos(5 * x)], axis=1)
    center, yaw = _fit_footprint(xy, width=1.9, length=4.6)
    assert abs(yaw) < 0.1
    assert math.isclose(center[1], 3.0 + 0.95, abs_tol=0.1)
