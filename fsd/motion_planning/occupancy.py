from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class CollisionGrid:
    blocked: np.ndarray
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    resolution: float

    def xy_to_row_col(self, x: float, y: float) -> tuple[int, int] | None:
        """Return the BEV cell for coordinates in (x_min, x_max] and (y_min, y_max]."""
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        if x <= x_min or x > x_max or y <= y_min or y > y_max:
            return None

        row = int((x_max - x) / self.resolution)
        col = int((y_max - y) / self.resolution)
        if row < 0 or col < 0 or row >= self.blocked.shape[0] or col >= self.blocked.shape[1]:
            return None
        return row, col

    def is_blocked_xy(self, x: float, y: float) -> bool:
        row_col = self.xy_to_row_col(x, y)
        if row_col is None:
            return True

        row, col = row_col
        return bool(self.blocked[row, col])

    def footprint_blocked(self, x: float, y: float, radius_m: float) -> bool:
        row_col = self.xy_to_row_col(x, y)
        if row_col is None:
            return True

        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        if x - radius_m < x_min or x + radius_m > x_max:
            return True
        if y - radius_m < y_min or y + radius_m > y_max:
            return True

        row, col = row_col
        radius_cells = math.ceil(radius_m / self.resolution)
        row_start = max(0, row - radius_cells)
        row_stop = min(self.blocked.shape[0], row + radius_cells + 1)
        col_start = max(0, col - radius_cells)
        col_stop = min(self.blocked.shape[1], col + radius_cells + 1)

        radius_sq = radius_m * radius_m
        for candidate_row in range(row_start, row_stop):
            x_hi = x_max - candidate_row * self.resolution
            x_lo = x_hi - self.resolution
            closest_x = min(max(x, x_lo), x_hi)
            for candidate_col in range(col_start, col_stop):
                if not self.blocked[candidate_row, candidate_col]:
                    continue

                y_hi = y_max - candidate_col * self.resolution
                y_lo = y_hi - self.resolution
                closest_y = min(max(y, y_lo), y_hi)
                dx = closest_x - x
                dy = closest_y - y
                if dx * dx + dy * dy <= radius_sq:
                    return True
        return False


def build_collision_grid(
    occupancy_probability: np.ndarray,
    height_range: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: float,
    occupancy_threshold: float = 0.62,
    height_threshold: float = 0.45,
) -> CollisionGrid:
    if occupancy_probability.shape != height_range.shape:
        raise ValueError("occupancy_probability and height_range must have matching shapes")

    blocked = (occupancy_probability >= occupancy_threshold) | (height_range >= height_threshold)
    return CollisionGrid(
        blocked=blocked,
        x_range=x_range,
        y_range=y_range,
        resolution=resolution,
    )
