from dataclasses import replace

import torch

from fsd.nmp.config import NmpConfig
from fsd.nmp.model import NeuralMotionPlanner

SMALL = replace(NmpConfig(), x_range=(-8.0, 8.0), y_range=(-4.8, 4.8))  # 80 x 48 cells


def test_output_shapes_small_grid():
    assert SMALL.grid.shape == (80, 48)
    out = NeuralMotionPlanner(SMALL)(torch.zeros(2, SMALL.in_channels, 80, 48, dtype=torch.uint8))
    assert out.cls_logits.shape == (2, 1, 20, 12)
    assert out.box_reg.shape == (2, 42, 20, 12)
    assert out.cost.shape == (2, 6, 80, 48)


def test_cost_is_clipped():
    model = NeuralMotionPlanner(SMALL)
    torch.nn.init.constant_(model.cost_head[-1].bias, 5000.0)
    assert model(torch.zeros(1, SMALL.in_channels, 80, 48)).cost.max() <= 1000.0
