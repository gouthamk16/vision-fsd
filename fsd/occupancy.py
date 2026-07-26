"""Temporal log-odds occupancy fusion over an ego-frame BEV grid.

Maintains a rolling occupancy grid expressed in the current ego frame. Each
keyframe the previous grid is warped into the new ego frame using nuScenes ego
poses, decayed toward "unknown", and updated with fresh LiDAR evidence (hits =
occupied, the polar visible region = free).
"""

from __future__ import annotations

import cv2
import numpy as np

from fsd.bev import lidar_points_to_ego
from fsd.data import LidarFrame, SurroundFrame
from fsd.lidar_projection import load_lidar_points, quaternion_to_rotation_matrix


def _relative_se2(prev_pose: dict, cur_pose: dict) -> tuple[np.ndarray, np.ndarray]:
    """SE(2) mapping a point in the current ego frame to the previous ego frame."""
    rot_prev = quaternion_to_rotation_matrix(prev_pose["rotation"])[:2, :2]
    rot_cur = quaternion_to_rotation_matrix(cur_pose["rotation"])[:2, :2]
    t_prev = np.asarray(prev_pose["translation"][:2], dtype=np.float64)
    t_cur = np.asarray(cur_pose["translation"][:2], dtype=np.float64)
    r_rel = rot_prev.T @ rot_cur
    t_rel = rot_prev.T @ (t_cur - t_prev)
    return r_rel, t_rel


class TemporalOccupancyMapper:
    """Rolling ego-frame log-odds occupancy grid fused across keyframes."""

    def __init__(
        self,
        x_range: tuple[float, float] = (-50.0, 50.0),
        y_range: tuple[float, float] = (-50.0, 50.0),
        z_range: tuple[float, float] = (-2.5, 3.0),
        ground_height: float = 0.3,
        resolution: float = 0.25,
        logit_hit: float = 0.85,
        logit_miss: float = 0.4,
        clamp: float = 5.0,
        decay: float = 0.97,
    ):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.ground_height = ground_height
        self.resolution = resolution
        self.logit_hit = logit_hit
        self.logit_miss = logit_miss
        self.clamp = clamp
        self.decay = decay

        self.h = int(round((x_range[1] - x_range[0]) / resolution))
        self.w = int(round((y_range[1] - y_range[0]) / resolution))
        self.logodds = np.zeros((self.h, self.w), dtype=np.float32)
        self._last_pose: dict | None = None

    def reset(self) -> None:
        self.logodds.fill(0.0)
        self._last_pose = None

    def _pixel_affine(self, prev_pose: dict, cur_pose: dict) -> np.ndarray:
        r_rel, t_rel = _relative_se2(prev_pose, cur_pose)
        se2 = np.eye(3)
        se2[:2, :2] = r_rel
        se2[:2, 2] = t_rel

        res = self.resolution
        x_max = self.x_range[1]
        y_max = self.y_range[1]
        # pixel [col,row,1] -> meter [x,y,1]
        p2m = np.array([[0.0, -res, x_max], [-res, 0.0, y_max], [0.0, 0.0, 1.0]])
        # meter [x,y,1] -> pixel [col,row,1]
        m2p = np.array([[0.0, -1.0 / res, y_max / res], [-1.0 / res, 0.0, x_max / res], [0.0, 0.0, 1.0]])
        full = m2p @ se2 @ p2m
        return full[:2].astype(np.float32)

    def _evidence(self, points_ego: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Height split: ground-level returns -> free, above-ground -> occupied.

        Marking only observed cells (rather than ray-casting free space) keeps
        the ground plane out of the occupied set and avoids treating the road
        as an obstacle.
        """
        free = np.zeros((self.h, self.w), dtype=bool)
        occ = np.zeros((self.h, self.w), dtype=bool)

        x, y, z = points_ego[:, 0], points_ego[:, 1], points_ego[:, 2]
        cols = ((self.y_range[1] - y) / self.resolution).astype(np.int32)
        rows = ((self.x_range[1] - x) / self.resolution).astype(np.int32)
        inb = (rows >= 0) & (rows < self.h) & (cols >= 0) & (cols < self.w)

        ground = inb & (z >= self.z_range[0]) & (z < self.ground_height)
        obstacle = inb & (z >= self.ground_height) & (z <= self.z_range[1])
        free[rows[ground], cols[ground]] = True
        occ[rows[obstacle], cols[obstacle]] = True
        return free, occ

    def step(self, lidar: LidarFrame) -> np.ndarray:
        """Fuse one LiDAR frame and return the current occupancy probability grid."""
        cur_pose = lidar.ego_pose
        if self._last_pose is not None:
            affine = self._pixel_affine(self._last_pose, cur_pose)
            # `affine` already maps current-frame pixels to previous-frame
            # pixels, so use WARP_INVERSE_MAP (cv2 otherwise inverts M itself).
            self.logodds = cv2.warpAffine(
                self.logodds,
                affine,
                (self.w, self.h),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
        self.logodds *= self.decay

        points_ego = lidar_points_to_ego(load_lidar_points(lidar.pointcloud_path), lidar)
        free, occ = self._evidence(points_ego)
        free[occ] = False
        self.logodds[free] += -self.logit_miss
        self.logodds[occ] += self.logit_hit
        np.clip(self.logodds, -self.clamp, self.clamp, out=self.logodds)

        self._last_pose = cur_pose
        return 1.0 / (1.0 + np.exp(-self.logodds))


def render_occupancy_bev(
    frame: SurroundFrame,
    prob: np.ndarray,
    x_range: tuple[float, float] = (-50.0, 50.0),
    y_range: tuple[float, float] = (-50.0, 50.0),
    resolution: float = 0.25,
    scale: int = 2,
) -> np.ndarray:
    """Colour the fused occupancy: free = dark blue, unknown = gray, occupied = warm."""
    h, w = prob.shape
    occ_amt = np.clip((prob - 0.5) * 2.0, 0.0, 1.0)[..., None]
    free_amt = np.clip((0.5 - prob) * 2.0, 0.0, 1.0)[..., None]

    base = np.full((h, w, 3), 90, dtype=np.float32)
    free_color = np.array([70, 35, 20], dtype=np.float32)   # BGR navy
    occ_color = np.array([60, 235, 255], dtype=np.float32)  # BGR amber/white
    canvas = base * (1 - free_amt - occ_amt) + free_color * free_amt + occ_color * occ_amt
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    for x in range(int(np.ceil(x_range[0] / 10) * 10), int(x_range[1]) + 1, 10):
        row = int((x_range[1] - x) / resolution)
        if 0 <= row < h:
            cv2.line(canvas, (0, row), (w - 1, row), (50, 50, 50), 1)
    for y in range(int(np.ceil(y_range[0] / 10) * 10), int(y_range[1]) + 1, 10):
        col = int((y_range[1] - y) / resolution)
        if 0 <= col < w:
            cv2.line(canvas, (col, 0), (col, h - 1), (50, 50, 50), 1)

    row0 = int(x_range[1] / resolution)
    col0 = int(y_range[1] / resolution)
    car_w = max(4, int(2.0 / resolution))
    car_l = max(8, int(4.5 / resolution))
    cv2.rectangle(canvas, (col0 - car_w // 2, row0 - car_l // 2), (col0 + car_w // 2, row0 + car_l // 2), (0, 200, 0), -1)
    cv2.arrowedLine(canvas, (col0, row0), (col0, row0 - car_l), (255, 255, 255), 1, tipLength=0.4)

    header_h = 58
    output = np.full((header_h + h, w, 3), 18, dtype=np.uint8)
    output[header_h:, :] = canvas
    title = f"{frame.scene_name} | sample {frame.sample_index} | temporal occupancy (log-odds)"
    legend = "occupied=amber  free=navy  unknown=gray"
    cv2.putText(output, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(output, legend, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 220, 255), 1, cv2.LINE_AA)

    if scale > 1:
        output = cv2.resize(output, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return output
