import math
from dataclasses import replace

import numpy as np
import torch
import torch.nn.functional as F

from fsd.nmp.config import NmpConfig
from fsd.nmp.losses import cell_centers, detection_loss, detection_targets, lookup_costs, planning_loss, violations
from fsd.nmp.model import REG_PER_STEP, NmpOutput
from fsd.nmp.samples import CLASS_NAMES

CFG = replace(NmpConfig(), x_range=(-8.0, 8.0), y_range=(-4.8, 4.8))  # 80 x 48 cells
H, W = CFG.grid.shape
T = CFG.horizon


def _traj(x, y=0.0, yaw=0.0):
    return torch.tensor([[x + 0.5 * k, y, yaw] for k in range(T)], dtype=torch.float32)


def _cells(traj):
    rows, cols, _ = CFG.grid.cells(traj[:, :2].numpy())
    return torch.from_numpy(rows), torch.from_numpy(cols)


def test_planning_loss_zero_when_gt_cheapest_by_margin():
    gt, neg = _traj(0.0), _traj(0.0, 2.0)
    cost = torch.full((1, T, H, W), 1000.0)
    r, c = _cells(gt)
    cost[0, torch.arange(T), r, c] = 0.0
    loss = planning_loss(cost, gt[None], torch.stack([gt, neg])[None], torch.zeros(1, 2, T), CFG.grid)
    assert float(loss) == 0.0


def test_planning_loss_positive_when_negative_cheaper():
    gt, neg = _traj(0.0), _traj(0.0, 2.0)
    cost = torch.zeros((1, T, H, W), requires_grad=True)
    with torch.no_grad():
        r, c = _cells(neg)
        cost[0, torch.arange(T), r, c] = -1000.0
    loss = planning_loss(cost, gt[None], neg[None, None], torch.zeros(1, 1, T), CFG.grid)
    loss.backward()
    assert float(loss) > 0 and torch.all(cost.grad[0, torch.arange(T), r, c] < 0)


def test_collision_adds_gamma_only_at_that_step():
    neg = _traj(0.0)
    obstacles = torch.zeros(1, T, H, W, dtype=torch.bool)
    body = neg[2, :2] + torch.tensor([1.385, 0.0])  # body centre ahead of the rear axle
    r, c = _cells(torch.cat([body, torch.zeros(1)])[None])
    obstacles[0, 2, r[0], c[0]] = True
    gamma = violations(neg[None, None], obstacles, torch.ones(1, H, W, dtype=torch.bool), CFG)
    assert gamma[0, 0].tolist() == [0.0, 0.0, 200.0, 0.0, 0.0, 0.0]


def test_collision_counts_an_overlap_at_the_ego_front():
    neg = _traj(0.0)
    obstacles = torch.zeros(1, T, H, W, dtype=torch.bool)
    front = neg[2, :2] + torch.tensor([2.9, 0.0])  # 1.5 m ahead of the body centre, still inside the 4.1 m body
    r, c = _cells(torch.cat([front, torch.zeros(1)])[None])
    obstacles[0, 2, r[0], c[0]] = True
    gamma = violations(neg[None, None], obstacles, torch.ones(1, H, W, dtype=torch.bool), CFG)
    assert gamma[0, 0].tolist() == [0.0, 0.0, 200.0, 0.0, 0.0, 0.0]


def test_offroad_adds_gamma_everywhere_off_the_drivable_area():
    gamma = violations(_traj(0.0)[None, None], torch.zeros(1, T, H, W, dtype=torch.bool), torch.zeros(1, H, W, dtype=torch.bool), CFG)
    assert torch.all(gamma == 100.0)


def test_offroad_not_counted_past_the_grid():
    far = _traj(20.0)  # the whole body lies beyond the 8 m grid edge, where the map can't judge
    gamma = violations(far[None, None], torch.zeros(1, T, H, W, dtype=torch.bool), torch.zeros(1, H, W, dtype=torch.bool), CFG)
    assert torch.all(gamma == 0.0)


def test_out_of_grid_points_clamp():
    cost = torch.arange(T * H * W, dtype=torch.float32).reshape(1, T, H, W)
    far = torch.tensor([[[500.0, 0.0, 0.0]] * T])
    picked = lookup_costs(cost, far[None], CFG.grid)
    assert torch.equal(picked[0, 0], cost[0, torch.arange(T), 0, W // 2])


def _boxes(valid_now: bool):
    # One car at (2, 1) heading 30 deg, moving +1 m in x per step.
    boxes = torch.zeros(1, 1, T + 1, 5)
    for k in range(T + 1):
        boxes[0, 0, k] = torch.tensor([2.0 + k, 1.0, 2.0, 4.6, math.radians(30)])
    valid = torch.full((1, 1, T + 1), valid_now)
    return boxes, valid, torch.tensor([[CLASS_NAMES.index("car")]]), torch.tensor([[12]])


def test_detection_loss_finite_without_vehicles():
    cls, reg, mask = detection_targets(*_boxes(False), CFG)
    assert cls.sum() == 0 and mask.sum() == 0
    out = NmpOutput(torch.zeros(1, 1, H // 4, W // 4), torch.zeros(1, (T + 1) * REG_PER_STEP, H // 4, W // 4), torch.zeros(1, T, H, W))
    loss = detection_loss(out, (cls, reg, mask))
    assert torch.isfinite(loss) and float(loss) > 0


def test_target_encoding_round_trip():
    cls, reg, mask = detection_targets(*_boxes(True), CFG)
    centers = cell_centers(CFG)
    i, j = map(int, torch.nonzero(cls[0, 0] == 1.0)[0])
    for k in (0, 3):
        dx, dy, logw, logl, s, c, s2, c2 = reg[0, REG_PER_STEP * k : REG_PER_STEP * (k + 1), i, j]
        assert mask[0, k, i, j]
        assert np.allclose([centers[i, j, 0] + 4 * dx, centers[i, j, 1] + 4 * dy], [2.0 + k, 1.0], atol=1e-4)
        assert math.isclose(float(torch.atan2(s, c)), math.radians(30), abs_tol=1e-4)
        assert math.isclose(float(torch.atan2(s2, c2)), math.radians(60), abs_tol=1e-4)
        assert math.isclose(float(torch.exp(logl)), 4.6, rel_tol=1e-5)


def test_targets_for_a_batch_without_any_boxes():
    cls, reg, mask = detection_targets(torch.zeros(2, 0, T + 1, 5), torch.zeros(2, 0, T + 1, dtype=torch.bool), torch.zeros(2, 0, dtype=torch.long), torch.zeros(2, 0, dtype=torch.long), CFG)
    assert cls.shape == (2, 1, H // 4, W // 4) and reg.shape == (2, (T + 1) * REG_PER_STEP, H // 4, W // 4) and not mask.any()


def test_targets_mask_only_the_step_a_vehicle_is_missing():
    boxes, valid, cls_idx, points = _boxes(True)
    valid[0, 0, 3] = False  # occluded or out of the scene at step 3 only
    cls, reg, mask = detection_targets(boxes, valid, cls_idx, points, CFG)
    i, j = map(int, torch.nonzero(cls[0, 0] == 1.0)[0])
    assert mask[0, :, i, j].tolist() == [True, True, True, False, True, True, True]
    assert not reg[0, 3 * REG_PER_STEP : 4 * REG_PER_STEP, i, j].any()


def test_targets_skip_a_vehicle_no_lidar_point_hit():
    boxes, valid, cls_idx, _ = _boxes(True)
    cls, reg, mask = detection_targets(boxes, valid, cls_idx, torch.tensor([[0]]), CFG)
    assert cls.sum() == 0 and not mask.any()


def _centre_cell(x, y):
    size = 4 * CFG.resolution
    return int((CFG.x_range[1] - x) // size), int((CFG.y_range[1] - y) // size)


def test_heatmap_peaks_once_at_the_box_centre_and_decays():
    cls, _, _ = detection_targets(*_boxes(True), CFG)
    heat = cls[0, 0]
    i, j = _centre_cell(2.0, 1.0)
    assert heat[i, j] == 1.0 and heat[i + 1, j] < 1.0 and heat[i + 2, j] < heat[i + 1, j]
    peaks = (heat > 0) & (heat == F.max_pool2d(heat[None, None], 3, stride=1, padding=1)[0, 0])
    assert int(peaks.sum()) == 1


def test_heatmap_keeps_two_nearby_cars_apart():
    boxes, valid, cls_idx, points = _boxes(True)
    second = boxes.clone()
    second[..., 1] -= 2.4  # a car in the next lane
    cls, _, _ = detection_targets(torch.cat([boxes, second], 1), valid.repeat(1, 2, 1), cls_idx.repeat(1, 2), points.repeat(1, 2), CFG)
    heat = cls[0, 0]
    peaks = (heat > 0) & (heat == F.max_pool2d(heat[None, None], 3, stride=1, padding=1)[0, 0])
    assert int((heat == 1.0).sum()) == 2 and int(peaks.sum()) == 2


def test_focal_loss_prefers_the_target_heatmap():
    targets = detection_targets(*_boxes(True), CFG)
    cls, reg, _ = targets
    p = cls.clamp(1e-4, 1 - 1e-4)
    cost = torch.zeros(1, T, H, W)
    matching = NmpOutput(torch.log(p) - torch.log1p(-p), reg, cost)
    flat = NmpOutput(torch.full_like(cls, -2.0), reg, cost)
    assert detection_loss(matching, targets) < detection_loss(flat, targets)
