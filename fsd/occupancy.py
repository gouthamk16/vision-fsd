"""Temporal log-odds occupancy fusion over an ego-frame BEV grid.

Keeps a rolling grid in the current ego frame. Each keyframe the previous grid
is warped into the new ego frame by the ego-pose delta and decayed toward
unknown, then updated with the keyframe sweep: cells holding above-ground
returns are hits, and cells a ray passes through before its first obstacle are
misses (free).
"""

from __future__ import annotations

import cv2
import numpy as np

from fsd.bev import BevGrid, draw_ego, draw_metric_grid, resample
from fsd.data import LidarFrame, Pose, SurroundFrame
from fsd.draw import with_header


class TemporalOccupancyMapper:
    def __init__(
        self,
        grid: BevGrid = BevGrid(),
        ground_height: float = 0.3,
        max_height: float = 3.0,
        logit_hit: float = 0.85,
        logit_miss: float = 0.4,
        clamp: float = 5.0,
        decay: float = 0.97,
        azimuth_bins: int = 720,
    ):
        self.grid = grid
        self.ground_height = ground_height
        self.max_height = max_height
        self.logit_hit = logit_hit
        self.logit_miss = logit_miss
        self.clamp = clamp
        self.decay = decay
        self.azimuth_bins = azimuth_bins
        h, w = grid.shape
        rows, cols = np.mgrid[0:h, 0:w]
        self._cell_xy = np.stack(
            [grid.x_range[1] - (rows + 0.5) * grid.resolution, grid.y_range[1] - (cols + 0.5) * grid.resolution],
            axis=-1,
        )
        self.logodds = np.zeros((h, w), dtype=np.float32)
        self._last_pose: Pose | None = None
        self._scene_token: str | None = None

    def reset(self) -> None:
        self.logodds.fill(0.0)
        self._last_pose = None

    def step(self, frame: SurroundFrame, lidar: LidarFrame, points: np.ndarray) -> np.ndarray:
        """Fuse the keyframe sweep (points with age 0) and return occupancy probability."""
        if frame.scene_token != self._scene_token:
            self.reset()
            self._scene_token = frame.scene_token
        if self._last_pose is not None:
            self.logodds = self._warp(self.logodds, lidar.ego_pose.relative_to(self._last_pose))
        self.logodds *= self.decay

        free, occ = self._evidence(points[points[:, 3] == 0], lidar.calibration.translation[:2])
        self.logodds[free & ~occ] -= self.logit_miss
        self.logodds[occ] += self.logit_hit
        np.clip(self.logodds, -self.clamp, self.clamp, out=self.logodds)
        self._last_pose = lidar.ego_pose
        return 1.0 / (1.0 + np.exp(-self.logodds))

    def _warp(self, grid: np.ndarray, current_to_previous: Pose) -> np.ndarray:
        # Pixel centres: (col, row) <-> (x, y) = (x_max - (row + .5) r, y_max - (col + .5) r).
        r, x_max, y_max = self.grid.resolution, self.grid.x_range[1], self.grid.y_range[1]
        px_to_m = np.array([[0.0, -r, x_max - 0.5 * r], [-r, 0.0, y_max - 0.5 * r], [0.0, 0.0, 1.0]])
        m_to_px = np.linalg.inv(px_to_m)
        se2 = np.eye(3)
        se2[:2, :2] = current_to_previous.rotation[:2, :2]
        se2[:2, 2] = current_to_previous.translation[:2]
        affine = (m_to_px @ se2 @ px_to_m)[:2].astype(np.float32)
        h, w = grid.shape
        # `affine` maps current pixels to previous ones, hence WARP_INVERSE_MAP.
        return cv2.warpAffine(grid, affine, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0.0)

    def _evidence(self, points: np.ndarray, sensor_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = self.grid.shape
        points = points[points[:, 2] <= self.max_height]
        obstacle = points[:, 2] >= self.ground_height
        rows, cols, inside = self.grid.cells(points)
        occ = np.zeros((h, w), dtype=bool)
        keep = inside & obstacle
        occ[rows[keep], cols[keep]] = True

        rel = points[:, :2] - sensor_xy
        bins = self._azimuth_bin(rel)
        ranges = np.hypot(rel[:, 0], rel[:, 1])
        first_hit = np.full(self.azimuth_bins, np.inf)
        np.minimum.at(first_hit, bins[obstacle], ranges[obstacle])
        farthest_seen = np.zeros(self.azimuth_bins)
        np.maximum.at(farthest_seen, bins, ranges)

        cell_rel = self._cell_xy - sensor_xy
        cell_bins = self._azimuth_bin(cell_rel.reshape(-1, 2)).reshape(h, w)
        cell_range = np.hypot(cell_rel[..., 0], cell_rel[..., 1])
        # Stop one cell short of the first obstacle so its own cell is not cleared.
        limit = np.minimum(first_hit - self.grid.resolution, farthest_seen)
        return cell_range < limit[cell_bins], occ

    def _azimuth_bin(self, rel: np.ndarray) -> np.ndarray:
        angle = np.arctan2(rel[:, 1], rel[:, 0])
        return ((angle + np.pi) / (2 * np.pi) * self.azimuth_bins).astype(np.int64) % self.azimuth_bins


def occupancy_canvas(prob: np.ndarray, raster: BevGrid, view: BevGrid) -> np.ndarray:
    """Free = navy, unknown = dark gray, occupied = amber, resampled smoothly to `view`."""
    prob = resample(prob.astype(np.float32) - 0.5, raster, view) + 0.5
    occ = np.clip((prob - 0.5) * 2.0, 0.0, 1.0)[..., None]
    free = np.clip((0.5 - prob) * 2.0, 0.0, 1.0)[..., None]
    unknown = np.array([42, 42, 42], dtype=np.float32)
    image = unknown * (1 - occ - free) + np.array([95, 45, 20], np.float32) * free + np.array([60, 200, 255], np.float32) * occ
    return image.astype(np.uint8)


def render_occupancy_bev(frame: SurroundFrame, prob: np.ndarray, raster: BevGrid, view: BevGrid) -> np.ndarray:
    image = occupancy_canvas(prob, raster, view)
    draw_metric_grid(image, view)
    draw_ego(image, view)
    return with_header(
        image,
        f"{frame.scene_name} | sample {frame.sample_index} | temporal occupancy (log-odds)",
        "occupied = amber   free (ray-cast) = navy   unknown = gray",
    )
