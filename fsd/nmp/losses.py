"""Perception loss (dense detection + forecasting) and the paper's max-margin planning loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

from fsd.bev import BevGrid
from fsd.ego_motion import EGO_FOOTPRINT
from fsd.nmp.config import NmpConfig
from fsd.nmp.model import REG_PER_STEP, NmpOutput
from fsd.nmp.samples import CLASS_NAMES

STRIDE = 4
OFFSET_SCALE_M = 4.0
# Trajectories track the rear axle (the ego-frame origin); collisions are judged at the body centre.
BODY_CENTER_M = sum(EGO_FOOTPRINT[0]) / 2


def cell_centers(cfg: NmpConfig) -> torch.Tensor:
    """(h, w, 2) ego-frame (x, y) of the stride-4 detection cells."""
    h, w = cfg.grid.shape[0] // STRIDE, cfg.grid.shape[1] // STRIDE
    size = STRIDE * cfg.resolution
    rows, cols = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    return torch.stack([cfg.x_range[1] - (rows + 0.5) * size, cfg.y_range[1] - (cols + 0.5) * size], dim=-1)


def detection_targets(boxes: torch.Tensor, box_valid: torch.Tensor, box_class: torch.Tensor, cfg: NmpConfig):
    """Per-cell (cls, reg, reg_mask): a cell is positive when its centre lies in a step-0 vehicle box."""
    centers = cell_centers(cfg).to(boxes.device)                                   # (h, w, 2)
    if boxes.shape[1] == 0:
        h, w = centers.shape[:2]
        steps = boxes.shape[2]
        zeros = boxes.new_zeros
        return zeros((boxes.shape[0], 1, h, w)), zeros((boxes.shape[0], steps * REG_PER_STEP, h, w)), zeros((boxes.shape[0], steps, h, w), dtype=torch.bool)
    detect = torch.tensor([CLASS_NAMES.index(c) for c in cfg.det_classes], device=boxes.device)
    owner = box_valid[..., 0] & torch.isin(box_class, detect)                       # (B, K)
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
    reg = torch.cat([offset, picked[..., 2:4].clamp(min=1e-3).log(), torch.sin(picked[..., 4:]), torch.cos(picked[..., 4:])], dim=-1)
    reg = reg.permute(0, 3, 4, 1, 2).flatten(1, 2)                                  # (B, (T+1)*6, h, w)
    mask = (valid & positive[..., None]).permute(0, 3, 1, 2)                        # (B, T+1, h, w)
    return positive[:, None].float(), reg * mask.repeat_interleave(REG_PER_STEP, dim=1), mask


def detection_loss(out: NmpOutput, targets) -> torch.Tensor:
    cls, reg, mask = targets
    focal = sigmoid_focal_loss(out.cls_logits.float(), cls, alpha=0.25, gamma=2.0, reduction="sum") / cls.sum().clamp(min=1)
    entries = mask.repeat_interleave(REG_PER_STEP, dim=1)
    smooth = F.smooth_l1_loss(out.box_reg.float()[entries], reg[entries], reduction="sum") / entries.sum().clamp(min=1)
    return focal + smooth


def _flat_cells(xy: torch.Tensor, grid: BevGrid) -> torch.Tensor:
    """Row-major cell index of each (x, y), clamped to the grid edge."""
    h, w = grid.shape
    rows = torch.floor((grid.x_range[1] - xy[..., 0]) / grid.resolution).long().clamp(0, h - 1)
    cols = torch.floor((grid.y_range[1] - xy[..., 1]) / grid.resolution).long().clamp(0, w - 1)
    return rows * w + cols


def _gather_steps(volume: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """volume (B, T, H, W), cells (B, N, T) -> (B, N, T) value of step t at each cell."""
    b, t = volume.shape[:2]
    step = torch.arange(t, device=cells.device) * (volume.shape[2] * volume.shape[3])
    return volume.reshape(b, -1).gather(1, (cells + step).flatten(1)).view(cells.shape)


def lookup_costs(cost: torch.Tensor, traj: torch.Tensor, grid: BevGrid) -> torch.Tensor:
    return _gather_steps(cost, _flat_cells(traj[..., :2], grid))


def violations(traj: torch.Tensor, obstacles: torch.Tensor, drivable: torch.Tensor, cfg: NmpConfig) -> torch.Tensor:
    """Rule-violation margin gamma (B, N, T): collision with a future GT box, or leaving the road."""
    body = traj[..., :2] + BODY_CENTER_M * torch.stack([torch.cos(traj[..., 2]), torch.sin(traj[..., 2])], dim=-1)
    cells = _flat_cells(body, cfg.grid)
    collide = _gather_steps(obstacles.float(), cells)
    on_road = drivable.float().flatten(1).gather(1, cells.flatten(1)).view(cells.shape)
    return cfg.gamma_collision * collide + cfg.gamma_offroad * (1.0 - on_road)


def planning_loss(cost: torch.Tensor, gt: torch.Tensor, negatives: torch.Tensor, gamma: torch.Tensor, grid: BevGrid) -> torch.Tensor:
    """Paper eq. 6: max over negatives of sum_t ReLU(c_gt - c_neg + d + gamma), batch mean."""
    c_gt = lookup_costs(cost, gt[:, None], grid)
    c_neg = lookup_costs(cost, negatives, grid)
    margin = torch.linalg.norm(gt[:, None, :, :2] - negatives[..., :2], dim=-1)
    return F.relu(c_gt - c_neg + margin + gamma).sum(dim=-1).max(dim=1).values.mean()
