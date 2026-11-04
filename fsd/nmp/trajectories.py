"""Physically feasible ego trajectories: curvature linear in arc length, constant acceleration.

The paper samples clothoids (via Fresnel integrals) plus straight lines and
circles. All three are curves whose curvature is linear in arc length,
kappa(xi) = k0 + rate * xi, so one numerical integrator covers them.
"""

from __future__ import annotations

import math

import torch

from fsd.nmp.config import NmpConfig


def _arc_length(v0: torch.Tensor, accel: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Distance travelled by time t with v(t) = max(0, v0 + accel t): no reversing."""
    stop = torch.where(accel < 0, -v0 / accel.clamp(max=-1e-9), torch.full_like(v0, math.inf))
    te = torch.minimum(t, stop)
    return v0 * te + 0.5 * accel * te * te


def rollout(
    v0: torch.Tensor,
    k0: torch.Tensor,
    rate: torch.Tensor,
    accel: torch.Tensor,
    times: torch.Tensor,
    substeps: int = 10,
    max_curvature: float = 0.2,
) -> torch.Tensor:
    """(x, y, yaw) at `times` in the start's ego frame; leading dims of the inputs broadcast."""
    v0, k0, rate, accel = torch.broadcast_tensors(v0, k0, rate, accel)
    step = float(times[0]) / substeps
    n = int(round(float(times[-1]) / step))
    t = step * torch.arange(n + 1, dtype=v0.dtype, device=v0.device)
    xi = _arc_length(v0[..., None], accel[..., None], t)          # (..., n+1)
    dxi = xi[..., 1:] - xi[..., :-1]
    kappa = (k0[..., None] + rate[..., None] * (xi[..., :-1] + 0.5 * dxi)).clamp(-max_curvature, max_curvature)
    dyaw = kappa * dxi
    yaw = torch.cat([torch.zeros_like(xi[..., :1]), torch.cumsum(dyaw, dim=-1)], dim=-1)
    mid = yaw[..., :-1] + 0.5 * dyaw
    x = torch.cat([torch.zeros_like(xi[..., :1]), torch.cumsum(torch.cos(mid) * dxi, dim=-1)], dim=-1)
    y = torch.cat([torch.zeros_like(xi[..., :1]), torch.cumsum(torch.sin(mid) * dxi, dim=-1)], dim=-1)
    pick = torch.round(times.to(v0.device) / step).long()
    return torch.stack([x[..., pick], y[..., pick], yaw[..., pick]], dim=-1)


def sample_trajectories(
    v0: torch.Tensor,
    k0: torch.Tensor,
    cfg: NmpConfig,
    n: int,
    obey_initial_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """(B, n, T, 3) trajectories per batch item, mixing straight, circle and clothoid paths."""
    shape, device = (v0.shape[0], n), v0.device

    def uniform(lo: float, hi: float) -> torch.Tensor:
        return lo + (hi - lo) * torch.rand(shape, generator=generator, device=device)

    v = v0[:, None].expand(shape).clone()
    k = k0[:, None].expand(shape).clone()
    free = torch.rand(shape, generator=generator, device=device) >= obey_initial_p
    v[free] = uniform(*cfg.init_speed_range)[free]
    k[free] = uniform(-cfg.max_curvature, cfg.max_curvature)[free]

    kind = torch.rand(shape, generator=generator, device=device)
    straight = kind < cfg.p_straight
    circle = ~straight & (kind < cfg.p_straight + cfg.p_circle)
    a = uniform(*cfg.clothoid_a_range)
    sign = torch.where(torch.rand(shape, generator=generator, device=device) < 0.5, -1.0, 1.0)
    rate = torch.where(straight | circle, torch.zeros_like(a), sign * math.pi / (a * a))
    k = torch.where(straight, torch.zeros_like(k), torch.where(circle, uniform(-cfg.max_curvature, cfg.max_curvature), k))

    times = torch.tensor(cfg.times, dtype=v.dtype, device=device)
    return rollout(v, k, rate, uniform(*cfg.accel_range), times, max_curvature=cfg.max_curvature)
