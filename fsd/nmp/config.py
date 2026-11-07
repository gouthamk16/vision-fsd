"""Hyper-parameters of the neural motion planner (Zeng et al., CVPR 2019) on nuScenes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fsd.bev import BevGrid
from fsd.ego_motion import EGO_FOOTPRINT
from fsd.nuscenes_map import MAP_LAYERS


@dataclass(frozen=True)
class NmpConfig:
    # Paper ROI: 140.8 x 80 m around the ego at 0.2 m, z from -2 to 3.4 m.
    x_range: tuple[float, float] = (-70.4, 70.4)
    y_range: tuple[float, float] = (-40.0, 40.0)
    resolution: float = 0.2
    z_range: tuple[float, float] = (-2.0, 3.4)
    z_resolution: float = 0.2
    # 10 sweeps over 1 s: every other sweep of the 20 Hz LIDAR_TOP stream.
    n_sweeps: int = 10
    sweep_stride: int = 2
    # 6 future steps, one per keyframe (2 Hz), so 3 s.
    horizon: int = 6
    dt: float = 0.5
    map_layers: tuple[str, ...] = MAP_LAYERS
    det_classes: tuple[str, ...] = ("car", "truck", "bus", "trailer", "construction")
    n_trajectories: int = 1000
    accel_range: tuple[float, float] = (-5.0, 5.0)
    clothoid_a_range: tuple[float, float] = (6.0, 80.0)
    max_curvature: float = 0.2
    p_straight: float = 0.5
    p_circle: float = 0.25
    negative_free_init_p: float = 0.8
    init_speed_range: tuple[float, float] = (0.0, 15.0)
    # Training obstacles grow by a little more than the ego half-width, so the ego body can be tested as a few
    # points on its axis (losses.body_points).
    obstacle_dilation: float = 1.0
    gamma_collision: float = 200.0
    gamma_offroad: float = 100.0
    cost_clip: float = 1000.0
    beta: float = 0.01

    def __post_init__(self):
        cells = self.obstacle_dilation / self.resolution
        if abs(cells - round(cells)) > 1e-6 or self.obstacle_dilation <= EGO_FOOTPRINT[1][1]:
            raise ValueError(f"obstacle_dilation {self.obstacle_dilation} m must be a whole number of {self.resolution} m cells "
                             f"and exceed the ego half-width {EGO_FOOTPRINT[1][1]} m, or the body points miss overlaps")

    @property
    def grid(self) -> BevGrid:
        return BevGrid(self.x_range, self.y_range, self.resolution)

    @property
    def z_bins(self) -> int:
        return int(round((self.z_range[1] - self.z_range[0]) / self.z_resolution))

    @property
    def in_channels(self) -> int:
        return self.n_sweeps * self.z_bins + len(self.map_layers)

    @property
    def times(self) -> np.ndarray:
        return self.dt * np.arange(1, self.horizon + 1)

    @property
    def history_sweeps(self) -> int:
        return (self.n_sweeps - 1) * self.sweep_stride + 1
