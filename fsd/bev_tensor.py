"""2.5D LiDAR BEV tensor: per-cell height statistics in the ego frame.

Channels per cell: point density, min / max / mean height, and height range
(max - min: ~0 on flat road, large on cars and walls). Computed from the
point cloud alone, so it is available on a real vehicle.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from fsd.bev import BevGrid, draw_ego, draw_metric_grid, resample
from fsd.data import SurroundFrame
from fsd.draw import colorize, label_band, with_header


@dataclass(frozen=True)
class BevTensor:
    density: np.ndarray
    max_height: np.ndarray
    min_height: np.ndarray
    mean_height: np.ndarray
    height_range: np.ndarray
    occupied: np.ndarray
    grid: BevGrid


def compute_bev_height_channels(points_ego: np.ndarray, grid: BevGrid, z_range=(-3.0, 5.0)) -> BevTensor:
    points = points_ego[(points_ego[:, 2] >= z_range[0]) & (points_ego[:, 2] <= z_range[1])]
    rows, cols, inside = grid.cells(points)
    h, w = grid.shape
    flat = rows[inside] * w + cols[inside]
    z = points[inside, 2].astype(np.float64)

    count = np.bincount(flat, minlength=h * w).astype(np.float64)
    zsum = np.bincount(flat, weights=z, minlength=h * w)
    zmax = np.full(h * w, -np.inf)
    np.maximum.at(zmax, flat, z)
    zmin = np.full(h * w, np.inf)
    np.minimum.at(zmin, flat, z)

    occupied = count > 0
    zmax = np.where(occupied, zmax, 0.0)
    zmin = np.where(occupied, zmin, 0.0)
    return BevTensor(
        density=count.reshape(h, w),
        max_height=zmax.reshape(h, w),
        min_height=zmin.reshape(h, w),
        mean_height=(zsum / np.maximum(count, 1.0)).reshape(h, w),
        height_range=(zmax - zmin).reshape(h, w),
        occupied=occupied.reshape(h, w),
        grid=grid,
    )


def _panel(values: np.ndarray, tensor: BevTensor, vmin: float, vmax: float, cmap: int, label: str, view: BevGrid) -> np.ndarray:
    color = colorize(values, vmin, vmax, cmap)
    color[~tensor.occupied] = (20, 20, 20)
    # Nearest keeps cells crisp and never blends a real height with the empty fill.
    image = resample(color, tensor.grid, view, cv2.INTER_NEAREST)
    draw_metric_grid(image, view)
    draw_ego(image, view)
    label_band(image, label)
    return image


def render_bev_channels(frame: SurroundFrame, tensor: BevTensor, view: BevGrid, z_range=(-3.0, 5.0)) -> np.ndarray:
    log_density = np.log1p(tensor.density)
    panels = [
        _panel(log_density, tensor, 0.0, float(log_density.max() or 1.0), cv2.COLORMAP_MAGMA, "density (log)", view),
        _panel(tensor.max_height, tensor, *z_range, cv2.COLORMAP_VIRIDIS, "max height", view),
        _panel(tensor.min_height, tensor, *z_range, cv2.COLORMAP_VIRIDIS, "min height", view),
        _panel(tensor.mean_height, tensor, *z_range, cv2.COLORMAP_VIRIDIS, "mean height", view),
        _panel(tensor.height_range, tensor, 0.0, 4.0, cv2.COLORMAP_INFERNO, "height range (0-4 m)", view),
        _panel(tensor.occupied.astype(np.float64), tensor, 0.0, 1.0, cv2.COLORMAP_BONE, "occupied cells", view),
    ]
    sheet = np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:])])
    return with_header(
        sheet,
        f"{frame.scene_name} | sample {frame.sample_index} | 2.5D height tensor",
        f"{int(tensor.occupied.sum())} occupied cells | {tensor.grid.shape[0]}x{tensor.grid.shape[1]} @ {tensor.grid.resolution:.2f} m | heights {z_range[0]:.0f} to {z_range[1]:.0f} m",
    )
