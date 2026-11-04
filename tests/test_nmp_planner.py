import math
from dataclasses import replace

import numpy as np
import torch

from fsd.nmp.config import NmpConfig
from fsd.nmp.evaluate import _match
from fsd.nmp.metrics import PlanMetrics, boxes_overlap, constant_velocity_plan, ego_footprint
from fsd.nmp.model import NmpOutput
from fsd.nmp.planner import decode_detections, select_plan
from fsd.nmp.samples import CLASS_NAMES
from fsd.object_detection import Box3D

CFG = replace(NmpConfig(), x_range=(-8.0, 8.0), y_range=(-4.8, 4.8))
T = CFG.horizon


def _square(cx, cy, half=1.0, yaw=0.0):
    corners = np.array([[half, half], [half, -half], [-half, -half], [-half, half]])
    c, s = math.cos(yaw), math.sin(yaw)
    return corners @ np.array([[c, -s], [s, c]]).T + [cx, cy]


def test_select_plan_picks_min_summed_cost():
    cost = torch.zeros(T, *CFG.grid.shape)
    cost[:, :, : CFG.grid.shape[1] // 2] = 5.0  # left half (y > 0) is expensive
    candidates = torch.zeros(2, T, 3)
    candidates[0, :, 1] = 2.0
    candidates[1, :, 1] = -2.0
    best, costs = select_plan(cost, candidates, CFG.grid)
    assert best == 1 and costs.tolist() == [30.0, 0.0]


def test_boxes_overlap_rotated_cases():
    assert boxes_overlap(_square(0, 0), _square(1.9, 0))
    assert not boxes_overlap(_square(0, 0), _square(2.1, 0))
    # A 45-degree square's corner reaches sqrt(2) ~ 1.41 m: overlaps at 2.3 m, clear at 2.5 m.
    assert boxes_overlap(_square(0, 0), _square(2.3, 0, yaw=math.pi / 4))
    assert not boxes_overlap(_square(0, 0), _square(2.5, 0, yaw=math.pi / 4))


def test_ego_footprint_follows_pose():
    fp = ego_footprint(np.array([10.0, 5.0, math.pi / 2]))
    assert np.allclose(fp.mean(axis=0), [10.0, 5.0 + 1.385], atol=1e-6)


def _boxes_at(step_xy: dict[int, tuple[float, float]], valid_steps: set[int]):
    boxes = np.zeros((1, T + 1, 5), np.float32)
    valid = np.zeros((1, T + 1), bool)
    for step, (x, y) in step_xy.items():
        boxes[0, step] = [x, y, 2.0, 4.0, 0.0]
        valid[0, step] = step in valid_steps
    return boxes, valid


def test_collision_rate_is_cumulative():
    plan = np.zeros((T, 3), np.float32)
    boxes, valid = _boxes_at({2: (1.0, 0.0)}, {2})
    metrics = PlanMetrics()
    metrics.add(plan, plan, boxes, valid, np.ones(CFG.grid.shape, bool), CFG)
    summary = metrics.summary()
    assert (summary["collision_1s"], summary["collision_2s"], summary["collision_3s"]) == (100.0, 100.0, 100.0)
    assert summary["l2_3s"] == 0.0 and summary["offroad_3s"] == 0.0


def test_missing_box_step_never_collides():
    plan = np.zeros((T, 3), np.float32)
    boxes, valid = _boxes_at({2: (1.0, 0.0)}, set())
    metrics = PlanMetrics()
    metrics.add(plan, plan, boxes, valid, np.ones(CFG.grid.shape, bool), CFG)
    assert metrics.summary()["collision_3s"] == 0.0


def test_cv_plan_goes_straight_at_v0():
    plan = constant_velocity_plan(4.0, CFG)
    assert np.allclose(plan[:, 0], 4.0 * CFG.times) and np.allclose(plan[:, 1:], 0.0)


def test_decode_detections_recovers_a_peak():
    h, w = CFG.grid.shape[0] // 4, CFG.grid.shape[1] // 4
    cls = torch.full((1, 1, h, w), -10.0)
    cls[0, 0, 5, 6] = 5.0
    reg = torch.zeros(1, (T + 1) * 6, h, w)
    reg[0, 3::6] = math.log(4.6)
    reg[0, 2::6] = math.log(1.9)
    reg[0, 5::6] = 1.0  # cos(yaw) = 1
    boxes, forecasts = decode_detections(NmpOutput(cls, reg, torch.zeros(1, T, *CFG.grid.shape)), CFG)
    assert len(boxes[0]) == 1 and forecasts[0].shape == (1, T + 1, 3)
    assert math.isclose(float(boxes[0][0].size[1]), 4.6, rel_tol=1e-5)


def test_match_ignores_vehicles_beyond_eval_range():
    car = CLASS_NAMES.index(CFG.det_classes[0])
    gt = np.zeros((2, T + 1, 5), np.float32)
    gt[0, :, 0], gt[1, :, 0] = 10.0, 60.0
    forecasts = np.zeros((2, T + 1, 3), np.float32)
    forecasts[0, :, 0], forecasts[1, :, 0] = 10.0, 60.0
    dets = [Box3D("car", np.array([x, 0.0, 0.8]), np.array([1.9, 4.6, 1.6]), 0.0, 0.9) for x in (10.0, 60.0)]
    scores, tps, _, n_gt = _match(dets, forecasts, gt, np.ones((2, T + 1), bool), np.array([car, car]), CFG)
    assert n_gt == 1 and scores == [0.9] and tps == [True]
