"""NMP network: PIXOR backbone, detection/forecast head and space-time cost-volume head."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from fsd.nmp.config import NmpConfig

REG_PER_STEP = 8  # dx, dy, log w, log l, sin yaw, cos yaw, sin 2 yaw, cos 2 yaw


@dataclass
class NmpOutput:
    cls_logits: torch.Tensor  # (B, 1, H/4, W/4)
    box_reg: torch.Tensor     # (B, (T+1) * 6, H/4, W/4)
    cost: torch.Tensor        # (B, T, H, W)


def _convs(n: int, c_in: int, c_out: int) -> nn.Sequential:
    layers = []
    for i in range(n):
        layers += [nn.Conv2d(c_in if i == 0 else c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


def _up(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(c_in, c_out, 3, stride=2, padding=1, output_padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        *_convs(1, c_out, c_out),
    )


class NeuralMotionPlanner(nn.Module):
    def __init__(self, cfg: NmpConfig):
        super().__init__()
        self.cfg = cfg
        # Paper: {2, 2, 3, 6, 5} conv layers with {32, 64, 128, 256, 256} filters,
        # max-pool after the first three blocks, blocks 1-4 fused at 1/4 scale.
        self.blocks = nn.ModuleList([
            _convs(2, cfg.in_channels, 32),
            _convs(2, 32, 64),
            _convs(3, 64, 128),
            _convs(6, 128, 256),
        ])
        self.fuse = _convs(5, 32 + 64 + 128 + 256, 256)
        steps = cfg.horizon + 1
        self.cls_head = nn.Sequential(_convs(2, 256, 256), nn.Conv2d(256, 1, 1))
        self.reg_head = nn.Sequential(_convs(2, 256, 256), nn.Conv2d(256, steps * REG_PER_STEP, 1))
        self.cost_head = nn.Sequential(_up(256, 128), _up(128, 64), nn.Conv2d(64, cfg.horizon, 1))
        # Start every cell at ~1% vehicle probability so the focal loss is stable early on.
        nn.init.constant_(self.cls_head[-1].bias, -math.log(99.0))

    def forward(self, x: torch.Tensor) -> NmpOutput:
        x = x.float()
        size = (x.shape[2] // 4, x.shape[3] // 4)
        scales = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            scales.append(F.interpolate(x, size=size, mode="bilinear", align_corners=False))
            if i < 3:
                x = F.max_pool2d(x, 2)
        features = self.fuse(torch.cat(scales, dim=1))
        cost = self.cost_head(features).clamp(-self.cfg.cost_clip, self.cfg.cost_clip)
        return NmpOutput(self.cls_head(features), self.reg_head(features), cost)


def load_weights(model: NeuralMotionPlanner, state: dict, source: object) -> None:
    """Load a checkpoint's weights; a layout mismatch means it predates a head change, which only retraining fixes."""
    try:
        model.load_state_dict(state)
    except RuntimeError as err:
        raise ValueError(f"{source} does not fit this model (written before a head change?): retrain, "
                         "or check out the code it was trained with") from err
