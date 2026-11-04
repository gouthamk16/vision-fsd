"""Ego-frame bird's-eye-view geometry and drawing.

One convention for every BEV raster and view: x (forward) points up the image,
y (left) points left, and pixel (0, 0) is the corner at (x_max, y_max).

Rasters (occupancy, height) are computed on a coarse `BevGrid`; views render
the same metric extent on a finer grid and draw vector overlays (points, boxes,
map, text) directly at that resolution rather than upscaling a coarse image.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from fsd.data import SurroundFrame
from fsd.draw import FONT, colorize, splat, with_header
from fsd.ego_motion import EGO_FOOTPRINT

DARK_CANVAS = (20, 20, 20)
_SHIFT = 4  # cv2 fixed-point bits: sub-pixel accurate anti-aliased geometry


@dataclass(frozen=True)
class BevGrid:
    x_range: tuple[float, float] = (-50.0, 50.0)
    y_range: tuple[float, float] = (-50.0, 50.0)
    resolution: float = 0.25  # metres per cell

    @property
    def shape(self) -> tuple[int, int]:
        return (
            int(round((self.x_range[1] - self.x_range[0]) / self.resolution)),
            int(round((self.y_range[1] - self.y_range[0]) / self.resolution)),
        )

    def with_resolution(self, resolution: float) -> BevGrid:
        return replace(self, resolution=resolution)

    def to_pixels(self, xy: np.ndarray) -> np.ndarray:
        """Continuous (col, row) image coordinates of ego-frame (x, y) points."""
        return np.stack(
            [(self.y_range[1] - xy[:, 1]) / self.resolution, (self.x_range[1] - xy[:, 0]) / self.resolution],
            axis=1,
        )

    def cells(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(rows, cols, inside) cell indices; floor, so points past an edge stay outside."""
        px = np.floor(self.to_pixels(xy)).astype(np.int64)
        h, w = self.shape
        inside = (px[:, 1] >= 0) & (px[:, 1] < h) & (px[:, 0] >= 0) & (px[:, 0] < w)
        return px[:, 1], px[:, 0], inside


def resample(raster: np.ndarray, source: BevGrid, target: BevGrid, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Warp a raster between grids by metric extent; cells outside `source` become 0."""
    s = source.resolution / target.resolution
    dc = (target.y_range[1] - source.y_range[1]) / target.resolution + 0.5 * s - 0.5
    dr = (target.x_range[1] - source.x_range[1]) / target.resolution + 0.5 * s - 0.5
    h, w = target.shape
    affine = np.float32([[s, 0, dc], [0, s, dr]])
    return cv2.warpAffine(raster, affine, (w, h), flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def fixed_point(grid: BevGrid, xy: np.ndarray) -> np.ndarray:
    return np.round(grid.to_pixels(xy) * (1 << _SHIFT)).astype(np.int32).reshape(-1, 1, 2)


def draw_polygon(image: np.ndarray, grid: BevGrid, xy: np.ndarray, color, thickness: int = 1, fill: bool = False) -> None:
    pts = fixed_point(grid, xy)
    if fill:
        cv2.fillPoly(image, [pts], color, cv2.LINE_AA, _SHIFT)
    else:
        cv2.polylines(image, [pts], True, color, thickness, cv2.LINE_AA, _SHIFT)


def draw_arrow(image: np.ndarray, grid: BevGrid, start_xy, end_xy, color, thickness: int = 1) -> None:
    (c0, r0), (c1, r1) = grid.to_pixels(np.array([start_xy, end_xy], dtype=np.float64))
    cv2.arrowedLine(image, (int(c0), int(r0)), (int(c1), int(r1)), color, thickness, cv2.LINE_AA, tipLength=0.3)


def draw_points(image: np.ndarray, grid: BevGrid, xy: np.ndarray, colors: np.ndarray, radius: int = 0) -> None:
    splat(image, np.floor(grid.to_pixels(xy)).astype(np.int32), colors, radius)


def draw_metric_grid(image: np.ndarray, grid: BevGrid, spacing_m: int = 10, dark: bool = True) -> None:
    minor, major = ((48, 48, 48), (95, 95, 95)) if dark else ((222, 222, 222), (175, 175, 175))
    h, w = image.shape[:2]
    scale = float(np.clip(w / 2400.0, 0.35, 0.6))
    for x in range(int(np.ceil(grid.x_range[0] / spacing_m)) * spacing_m, int(grid.x_range[1]) + 1, spacing_m):
        row = int(round((grid.x_range[1] - x) / grid.resolution))
        cv2.line(image, (0, row), (w - 1, row), major if x == 0 else minor, 1)
        cv2.putText(image, f"{x}m", (4, row - 4), FONT, scale, major, 1, cv2.LINE_AA)
    for y in range(int(np.ceil(grid.y_range[0] / spacing_m)) * spacing_m, int(grid.y_range[1]) + 1, spacing_m):
        col = int(round((grid.y_range[1] - y) / grid.resolution))
        cv2.line(image, (col, 0), (col, h - 1), major if y == 0 else minor, 1)


def draw_ego(image: np.ndarray, grid: BevGrid, fill=(0, 200, 0), outline=(255, 255, 255)) -> None:
    (x0, x1), (y0, y1) = EGO_FOOTPRINT
    body = np.array([[x1, y1], [x1, y0], [x0, y0], [x0, y1]])
    draw_polygon(image, grid, body, fill, fill=True)
    draw_polygon(image, grid, body, outline)
    draw_arrow(image, grid, ((x0 + x1) / 2, 0.0), (x1 + 2.0, 0.0), outline, 2)


def canvas(grid: BevGrid, color=DARK_CANVAS) -> np.ndarray:
    return np.full((*grid.shape, 3), color, dtype=np.uint8)


def lidar_canvas(points: np.ndarray, grid: BevGrid, background: np.ndarray | None = None, z_range=(-2.0, 4.0)) -> np.ndarray:
    """Points coloured by height over an optional map background, tallest drawn last."""
    image = canvas(grid) if background is None else background.copy()
    draw_metric_grid(image, grid)
    points = points[(points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])]
    points = points[np.argsort(points[:, 2])]
    draw_points(image, grid, points, colorize(points[:, 2], *z_range, cv2.COLORMAP_TURBO), radius=1)
    return image


def render_lidar_bev(frame: SurroundFrame, points: np.ndarray, grid: BevGrid, background: np.ndarray | None = None) -> np.ndarray:
    image = lidar_canvas(points, grid, background)
    draw_ego(image, grid)
    sweeps = len(np.unique(points[:, 3])) if points.shape[1] > 3 else 1
    return with_header(
        image,
        f"{frame.scene_name} | sample {frame.sample_index} | LiDAR BEV",
        f"{len(points)} points from {sweeps} sweep(s), ego-motion compensated | colour = height (-2 to 4 m) | grid 10 m",
    )
