import math
from dataclasses import replace

import numpy as np
import torch

from fsd.nmp.config import NmpConfig
from fsd.nmp.trajectories import rollout, sample_trajectories

CFG = NmpConfig()
TIMES = torch.tensor(CFG.times, dtype=torch.float64)


def _roll(v0, k0, rate, accel):
    t = lambda x: torch.tensor(x, dtype=torch.float64)  # noqa: E731
    return rollout(t(v0), t(k0), t(rate), t(accel), TIMES).numpy()


def test_config_derived_sizes():
    assert CFG.grid.shape == (704, 400)
    assert CFG.z_bins == 27 and CFG.in_channels == 278 and CFG.history_sweeps == 19
    assert np.allclose(CFG.times, [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])


def test_straight_constant_speed_moves_along_x():
    traj = _roll(10.0, 0.0, 0.0, 0.0)
    assert np.allclose(traj[:, 0], 10.0 * CFG.times, atol=1e-3)
    assert np.allclose(traj[:, 1:], 0.0, atol=1e-3)


def test_circle_stays_on_its_radius():
    traj = _roll(10.0, 0.1, 0.0, 0.0)
    assert np.allclose(np.hypot(traj[:, 0], traj[:, 1] - 10.0), 10.0, atol=0.05)


def test_braking_never_reverses():
    x = _roll(2.0, 0.0, 0.0, -5.0)[:, 0]
    assert np.all(np.diff(x) >= 0)
    assert math.isclose(x[-1], 0.4, abs_tol=1e-3)


def test_stationary_ego_with_negative_accel_stays_put():
    assert np.allclose(_roll(0.0, 0.0, 0.0, -3.0), 0.0)


def test_sampler_shapes_and_initial_heading():
    gen = torch.Generator().manual_seed(0)
    out = sample_trajectories(torch.tensor([5.0, 1.0]), torch.tensor([0.0, 0.1]), CFG, 1000, generator=gen)
    assert out.shape == (2, 1000, 6, 3) and torch.isfinite(out).all()
    clothoids = sample_trajectories(torch.tensor([1.0]), torch.tensor([0.1]), replace(CFG, p_straight=0.0, p_circle=0.0), 200, generator=gen)
    # Clothoids start at the ego's curvature: yaw = k0*xi + rate*xi^2/2 with |rate| <= pi/36.
    first = clothoids[0, :, 0]
    xi = torch.hypot(first[:, 0], first[:, 1])  # chord ~ arc at these curvatures
    assert torch.all((first[:, 2] - 0.1 * xi).abs() <= 0.5 * (math.pi / 36) * xi**2 + 1e-3)


def test_free_initial_state_negatives_differ_from_ego_speed():
    gen = torch.Generator().manual_seed(1)
    out = sample_trajectories(torch.tensor([0.0]), torch.tensor([0.0]), replace(CFG, p_straight=1.0), 500, obey_initial_p=0.0, generator=gen)
    assert (out[0, :, -1, 0] > 1.0).float().mean() > 0.5  # redrawn v0 ~ U[0, 15] moves most of them
