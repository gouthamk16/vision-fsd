"""Inference: cost volume + detections from one frame, then the cheapest sampled trajectory."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fsd.bev import BevGrid
from fsd.data import LidarFrame, SurroundFrame
from fsd.nmp.config import NmpConfig
from fsd.nmp.losses import OFFSET_SCALE_M, cell_centers, lookup_costs
from fsd.nmp.model import REG_PER_STEP, NeuralMotionPlanner, NmpOutput, load_weights
from fsd.nmp.samples import dense_input, ego_state, sweep_points, voxelize
from fsd.nmp.trajectories import sample_trajectories
from fsd.nuscenes_map import NuScenesMapRenderer
from fsd.object_detection import Box3D

SUPPRESS_RADIUS_M = 2.0  # under a car's 1.9 m width: two parked cars side by side stay two


@dataclass(frozen=True)
class PlanResult:
    plan: np.ndarray          # (T, 3) x, y, yaw of the selected trajectory
    candidates: np.ndarray    # (N, T, 3)
    costs: np.ndarray         # (N,) summed cost per candidate
    cost_volume: np.ndarray   # (T, H, W)
    detections: list[Box3D]
    forecasts: np.ndarray     # (K, T+1, 3) x, y, yaw per detection and step
    v0: float


def select_plan(cost: torch.Tensor, candidates: torch.Tensor, grid: BevGrid) -> tuple[int, torch.Tensor]:
    costs = lookup_costs(cost[None], candidates[None], grid)[0].sum(dim=-1)
    return int(torch.argmin(costs)), costs


def _suppress(xy: np.ndarray, scores: np.ndarray, radius: float) -> np.ndarray:
    """Greedy NMS on centre distance: indices of kept detections, best first."""
    keep, removed = [], np.zeros(len(scores), bool)
    for i in np.argsort(-scores):
        if not removed[i]:
            keep.append(i)
            removed |= np.hypot(*(xy - xy[i]).T) < radius
    return np.array(keep, int)


def decode_detections(out: NmpOutput, cfg: NmpConfig, threshold: float = 0.3) -> tuple[list[list[Box3D]], list[np.ndarray]]:
    """Vehicle boxes at 3x3 score peaks above `threshold`, one per vehicle, with their forecast poses per step.

    Training puts one Gaussian peak at each vehicle's centre, but in bf16 neighbouring cells can still tie and
    each be a 3x3 peak, so peaks closer than SUPPRESS_RADIUS_M are merged into the best-scoring one.
    """
    score = out.cls_logits.float().sigmoid()
    peaks = (score == F.max_pool2d(score, 3, stride=1, padding=1)) & (score > threshold)
    centers = cell_centers(cfg).to(score.device)
    steps = cfg.horizon + 1
    boxes, forecasts = [], []
    for b in range(score.shape[0]):
        rows, cols = torch.nonzero(peaks[b, 0], as_tuple=True)
        reg = out.box_reg[b].float().view(steps, REG_PER_STEP, *score.shape[2:])[:, :, rows, cols]  # (T+1, 6, K)
        xy = centers[rows, cols].T[None] + OFFSET_SCALE_M * reg[:, :2]                              # (T+1, 2, K)
        axis = 0.5 * torch.atan2(reg[:, 6], reg[:, 7])                                              # (T+1, K)
        # The axis is reliable even for a parked car; the full angle only chooses which end is the front.
        flip = torch.cos(torch.atan2(reg[:, 4], reg[:, 5]) - axis) < 0
        yaw = torch.remainder(axis + math.pi * flip + math.pi, 2 * math.pi) - math.pi
        size = reg[0, 2:4].exp()                                                                    # (2, K): w, l
        poses = torch.cat([xy, yaw[:, None]], dim=1).permute(2, 0, 1).cpu().numpy()                  # (K, T+1, 3)
        scores, size = score[b, 0, rows, cols].cpu().numpy(), size.cpu().numpy()
        keep = _suppress(poses[:, 0, :2], scores, SUPPRESS_RADIUS_M)
        boxes.append([
            Box3D("car", np.array([poses[k, 0, 0], poses[k, 0, 1], 0.8]), np.array([size[0, k], size[1, k], 1.6]), float(poses[k, 0, 2]), float(scores[k]))
            for k in keep
        ])
        forecasts.append(poses[keep])
    return boxes, forecasts


class NmpPlanner:
    def __init__(self, weights: Path, cfg: NmpConfig = NmpConfig(), device: str | None = None):
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = NeuralMotionPlanner(cfg)
        load_weights(self.model, torch.load(weights, map_location="cpu", weights_only=True)["model"], weights)
        self.model.to(self.device).eval()
        self.generator = torch.Generator(device=self.device).manual_seed(0)

    @torch.inference_mode()
    def plan(self, frame: SurroundFrame, lidar: LidarFrame, map_renderer: NuScenesMapRenderer) -> PlanResult | None:
        """None until `lidar` holds the full history: training skips such keyframes rather than zero-padding."""
        if len(lidar.sweeps) < self.cfg.history_sweeps:
            return None
        # Exactly the history a training sample holds: ego_state reaches back as far as the frame goes.
        lidar = LidarFrame(lidar.sweeps[: self.cfg.history_sweeps])
        points, sweep = sweep_points(lidar, self.cfg)
        masks = map_renderer.layer_masks(frame, self.cfg.grid, self.cfg.map_layers)
        x = torch.from_numpy(dense_input(voxelize(points, sweep, self.cfg), masks, self.cfg))[None].to(self.device)
        with torch.autocast(self.device.type, torch.bfloat16, enabled=self.device.type == "cuda"):
            out = self.model(x)
        v0, k0 = ego_state(lidar)
        candidates = sample_trajectories(
            torch.tensor([v0], device=self.device), torch.tensor([k0], device=self.device), self.cfg, self.cfg.n_trajectories, generator=self.generator
        )[0]
        cost = out.cost[0].float()
        best, costs = select_plan(cost, candidates, self.cfg.grid)
        boxes, forecasts = decode_detections(out, self.cfg)
        return PlanResult(
            plan=candidates[best].cpu().numpy(),
            candidates=candidates.cpu().numpy(),
            costs=costs.cpu().numpy(),
            cost_volume=cost.cpu().numpy(),
            detections=boxes[0],
            forecasts=forecasts[0],
            v0=v0,
        )
