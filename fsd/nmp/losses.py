"""Perception loss (dense detection + forecasting) and the paper's max-margin planning loss."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from fsd.bev import BevGrid
from fsd.ego_motion import EGO_FOOTPRINT
from fsd.nmp.config import NmpConfig
from fsd.nmp.model import REG_PER_STEP, NmpOutput
from fsd.nmp.samples import CLASS_NAMES

STRIDE = 4
OFFSET_SCALE_M = 4.0
# Trajectories track the rear axle (the ego-frame origin). Off-road is judged at the body centre.
BODY_CENTER_M = sum(EGO_FOOTPRINT[0]) / 2


def body_points(cfg: NmpConfig) -> np.ndarray:
    """Offsets along the ego axis whose discs of the obstacle dilation radius cover the footprint.

    A hit at any of them means the bodies overlap (conservative by up to ~0.5 m past the bumpers). On the grid the
    corners keep only ~0.05 m of slack at 1.0 m, so a clip of under one cell there can be missed; a larger dilation
    would close it at the cost of penalising side passes that clear by ~0.3 m.
    """
    (x0, x1), (_, half_width) = EGO_FOOTPRINT
    reach = math.sqrt(cfg.obstacle_dilation**2 - half_width**2)
    span = (x0 + reach, x1 - reach)
    return np.linspace(*span, math.ceil((span[1] - span[0]) / (2 * reach)) + 1)


def cell_centers(cfg: NmpConfig) -> torch.Tensor:
    """(h, w, 2) ego-frame (x, y) of the stride-4 detection cells."""
    h, w = cfg.grid.shape[0] // STRIDE, cfg.grid.shape[1] // STRIDE
    size = STRIDE * cfg.resolution
    rows, cols = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    return torch.stack([cfg.x_range[1] - (rows + 0.5) * size, cfg.y_range[1] - (cols + 0.5) * size], dim=-1)


def _gaussian_radius(length: torch.Tensor, width: torch.Tensor, overlap: float = 0.1) -> torch.Tensor:
    """CornerNet's radius (cells) within which a shifted box still overlaps the true one by `overlap` (as CenterPoint)."""
    roots = []
    for a, b, c in ((1.0, length + width, length * width * (1 - overlap) / (1 + overlap)),
                    (4.0, 2 * (length + width), (1 - overlap) * length * width),
                    (4 * overlap, -2 * overlap * (length + width), (overlap - 1) * length * width)):
        roots.append((b + torch.sqrt(b**2 - 4 * a * c)) / 2)
    return torch.minimum(torch.minimum(roots[0], roots[1]), roots[2])


def _heatmap(boxes: torch.Tensor, owner: torch.Tensor, cfg: NmpConfig) -> torch.Tensor:
    """(B, 1, h, w) CenterPoint target: 1 at each vehicle's centre cell, a Gaussian around it, max over vehicles.

    A single peak per vehicle is what keeps the decode from finding several (the old every-cell-inside target made
    a plateau with many). Radius at least 2 cells, as CenterPoint.
    """
    h, w = cfg.grid.shape[0] // STRIDE, cfg.grid.shape[1] // STRIDE
    size = STRIDE * cfg.resolution
    row = torch.floor((cfg.x_range[1] - boxes[..., 0, 0]) / size)                 # (B, K) centre cell
    col = torch.floor((cfg.y_range[1] - boxes[..., 0, 1]) / size)
    owner = owner & (row >= 0) & (row < h) & (col >= 0) & (col < w)
    radius = _gaussian_radius(boxes[..., 0, 3] / size, boxes[..., 0, 2] / size).floor().clamp(min=2)
    sigma = (2 * radius + 1) / 6
    dr = torch.arange(h, device=boxes.device)[None, None, :, None] - row[..., None, None]
    dc = torch.arange(w, device=boxes.device)[None, None, None, :] - col[..., None, None]
    near = (dr.abs() <= radius[..., None, None]) & (dc.abs() <= radius[..., None, None]) & owner[..., None, None]
    peaks = torch.exp(-(dr**2 + dc**2) / (2 * sigma[..., None, None] ** 2)) * near
    return peaks.amax(dim=1, keepdim=True)


def detection_targets(boxes: torch.Tensor, box_valid: torch.Tensor, box_class: torch.Tensor, box_points: torch.Tensor, cfg: NmpConfig):
    """Per-cell (heatmap, reg, reg_mask). Regression is supervised at every cell inside a step-0 vehicle box, not
    only at the centre as in CenterPoint: denser supervision for a small dataset, and the decode reads it at the peak.

    Vehicles no LiDAR return hit are left out, as the eval leaves them out: the input holds no evidence of them.
    """
    centers = cell_centers(cfg).to(boxes.device)                                   # (h, w, 2)
    if boxes.shape[1] == 0:
        h, w = centers.shape[:2]
        steps = boxes.shape[2]
        zeros = boxes.new_zeros
        return zeros((boxes.shape[0], 1, h, w)), zeros((boxes.shape[0], steps * REG_PER_STEP, h, w)), zeros((boxes.shape[0], steps, h, w), dtype=torch.bool)
    detect = torch.tensor([CLASS_NAMES.index(c) for c in cfg.det_classes], device=boxes.device)
    owner = box_valid[..., 0] & torch.isin(box_class, detect) & (box_points > 0)    # (B, K)
    rel = centers[None, None] - boxes[:, :, None, None, 0, :2]                      # (B, K, h, w, 2)
    cos, sin = torch.cos(boxes[..., 0, 4]), torch.sin(boxes[..., 0, 4])
    along = rel[..., 0] * cos[..., None, None] + rel[..., 1] * sin[..., None, None]
    across = -rel[..., 0] * sin[..., None, None] + rel[..., 1] * cos[..., None, None]
    inside = (along.abs() <= boxes[..., 0, 3, None, None] / 2) & (across.abs() <= boxes[..., 0, 2, None, None] / 2) & owner[..., None, None]
    positive = inside.any(dim=1)                                                    # (B, h, w)
    idx = inside.float().argmax(dim=1)                                              # owning box per cell
    picked = torch.gather(boxes, 1, idx.flatten(1)[..., None, None].expand(-1, -1, *boxes.shape[2:]))
    picked = picked.view(*idx.shape, *boxes.shape[2:])                              # (B, h, w, T+1, 5)
    valid = torch.gather(box_valid, 1, idx.flatten(1)[..., None].expand(-1, -1, box_valid.shape[2])).view(*idx.shape, -1)
    offset = (picked[..., :2] - centers[None, :, :, None]) / OFFSET_SCALE_M
    yaw = picked[..., 4:]
    # The doubled angle is the vehicle's axis, the same whichever way it faces; the full angle only picks the side.
    reg = torch.cat([offset, picked[..., 2:4].clamp(min=1e-3).log(), torch.sin(yaw), torch.cos(yaw), torch.sin(2 * yaw), torch.cos(2 * yaw)], dim=-1)
    reg = reg.permute(0, 3, 4, 1, 2).flatten(1, 2)                                  # (B, (T+1)*REG_PER_STEP, h, w)
    mask = (valid & positive[..., None]).permute(0, 3, 1, 2)                        # (B, T+1, h, w)
    return _heatmap(boxes, owner, cfg), reg * mask.repeat_interleave(REG_PER_STEP, dim=1), mask


def _focal(logits: torch.Tensor, heat: torch.Tensor) -> torch.Tensor:
    """CenterNet's penalty-reduced focal loss (alpha 2, beta 4), per centre cell."""
    peak = heat.eq(1.0).float()
    p = logits.sigmoid()
    positive = (1 - p) ** 2 * F.logsigmoid(logits) * peak
    negative = p**2 * F.logsigmoid(-logits) * (1 - heat) ** 4 * (1 - peak)
    return -(positive.sum() + negative.sum()) / peak.sum().clamp(min=1)


def detection_loss(out: NmpOutput, targets) -> torch.Tensor:
    cls, reg, mask = targets
    focal = _focal(out.cls_logits.float(), cls)
    entries = mask.repeat_interleave(REG_PER_STEP, dim=1)
    smooth = F.smooth_l1_loss(out.box_reg.float()[entries], reg[entries], reduction="sum") / entries.sum().clamp(min=1)
    return focal + smooth


def _flat_cells(xy: torch.Tensor, grid: BevGrid) -> torch.Tensor:
    """Row-major cell index of each (x, y), clamped to the grid edge."""
    h, w = grid.shape
    rows = torch.floor((grid.x_range[1] - xy[..., 0]) / grid.resolution).long().clamp(0, h - 1)
    cols = torch.floor((grid.y_range[1] - xy[..., 1]) / grid.resolution).long().clamp(0, w - 1)
    return rows * w + cols


def _on_grid(xy: torch.Tensor, grid: BevGrid) -> torch.Tensor:
    """1 where (x, y) falls in a grid cell (BevGrid.cells' convention), else 0: past the edge the map can't judge."""
    inside = (xy[..., 0] <= grid.x_range[1]) & (xy[..., 0] > grid.x_range[0]) & (xy[..., 1] <= grid.y_range[1]) & (xy[..., 1] > grid.y_range[0])
    return inside.float()


def _gather_steps(volume: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """volume (B, T, H, W), cells (B, N, T) -> (B, N, T) value of step t at each cell."""
    b, t = volume.shape[:2]
    step = torch.arange(t, device=cells.device) * (volume.shape[2] * volume.shape[3])
    return volume.reshape(b, -1).gather(1, (cells + step).flatten(1)).view(cells.shape)


def lookup_costs(cost: torch.Tensor, traj: torch.Tensor, grid: BevGrid) -> torch.Tensor:
    return _gather_steps(cost, _flat_cells(traj[..., :2], grid))


def violations(traj: torch.Tensor, obstacles: torch.Tensor, drivable: torch.Tensor, cfg: NmpConfig) -> torch.Tensor:
    """Rule-violation margin gamma (B, N, T): collision with a future GT box, or leaving the road."""
    heading = torch.stack([torch.cos(traj[..., 2]), torch.sin(traj[..., 2])], dim=-1)
    offsets = torch.as_tensor(body_points(cfg), dtype=traj.dtype, device=traj.device)
    points = _flat_cells(traj[..., None, :2] + offsets[:, None] * heading[..., None, :], cfg.grid)  # (B, N, T, P)
    b, n, t, p = points.shape
    hits = _gather_steps(obstacles.float(), points.permute(0, 1, 3, 2).reshape(b, n * p, t)).view(b, n, p, t)
    collide = hits.amax(dim=2)
    centre = traj[..., :2] + BODY_CENTER_M * heading
    cells = _flat_cells(centre, cfg.grid)
    off_road = 1.0 - drivable.float().flatten(1).gather(1, cells.flatten(1)).view(cells.shape)
    return cfg.gamma_collision * collide + cfg.gamma_offroad * off_road * _on_grid(centre, cfg.grid)


def planning_loss(cost: torch.Tensor, gt: torch.Tensor, negatives: torch.Tensor, gamma: torch.Tensor, grid: BevGrid) -> torch.Tensor:
    """Paper eq. 6: max over negatives of sum_t ReLU(c_gt - c_neg + d + gamma), batch mean."""
    c_gt = lookup_costs(cost, gt[:, None], grid)
    c_neg = lookup_costs(cost, negatives, grid)
    margin = torch.linalg.norm(gt[:, None, :, :2] - negatives[..., :2], dim=-1)
    return F.relu(c_gt - c_neg + margin + gamma).sum(dim=-1).max(dim=1).values.mean()
