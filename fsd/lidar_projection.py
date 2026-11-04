from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from fsd.contact_sheet import camera_sheet
from fsd.data import CameraFrame, LidarFrame, Pose, SurroundFrame
from fsd.draw import colorize, splat
from fsd.ego_motion import EGO_FOOTPRINT

# Returns off the ego body itself: the footprint plus a margin for mirrors and
# the roof rack, below the 1.84 m sensor mount.
_EGO_MASK_MARGIN_M = 0.3
_EGO_MASK_MAX_Z_M = 2.5


def load_lidar_points(path: str | Path) -> np.ndarray:
    """Load nuScenes LIDAR_TOP .pcd.bin points as Nx3 XYZ in the sensor frame."""
    points = np.fromfile(str(path), dtype=np.float32)
    if points.size % 5 != 0:
        raise ValueError(f"Unexpected nuScenes LiDAR point format in {path}: {points.size} floats")
    return points.reshape((-1, 5))[:, :3]


def _off_ego_body(points_ego: np.ndarray) -> np.ndarray:
    (x0, x1), (y0, y1) = EGO_FOOTPRINT
    m = _EGO_MASK_MARGIN_M
    on_body = (
        (points_ego[:, 0] > x0 - m) & (points_ego[:, 0] < x1 + m)
        & (points_ego[:, 1] > y0 - m) & (points_ego[:, 1] < y1 + m)
        & (points_ego[:, 2] < _EGO_MASK_MAX_Z_M)
    )
    return ~on_body


def lidar_points_ego(lidar: LidarFrame) -> np.ndarray:
    """All sweeps of `lidar` in its reference ego frame as (N, 4): x, y, z, age_s.

    Older sweeps are motion-compensated for ego motion only, so moving objects
    smear over the accumulation window.
    """
    chunks = []
    for sweep in lidar.sweeps:
        points = sweep.calibration.apply(load_lidar_points(sweep.path))
        points = points[_off_ego_body(points)]
        points = sweep.ego_pose.relative_to(lidar.ego_pose).apply(points)
        age = np.full((len(points), 1), (lidar.timestamp_us - sweep.timestamp_us) * 1e-6)
        chunks.append(np.hstack([points, age]))
    return np.concatenate(chunks).astype(np.float32)


def project_to_camera(
    points_ego: np.ndarray,
    ego_pose: Pose,
    camera: CameraFrame,
    image_shape: tuple[int, ...],
    min_depth: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project ego-frame points into `camera`. Returns (uv, depth, index of kept points)."""
    cam = camera.calibration.invert(camera.ego_pose.invert(ego_pose.apply(points_ego[:, :3].astype(np.float64))))
    front = np.flatnonzero(cam[:, 2] > min_depth)
    projected = cam[front] @ camera.intrinsic.T
    uv = projected[:, :2] / projected[:, 2:3]
    h, w = image_shape[:2]
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return uv[inside], cam[front[inside], 2], front[inside]


def depth_to_bgr(depth: np.ndarray, max_depth: float = 80.0) -> np.ndarray:
    """Near returns red, far blue."""
    return colorize(-depth, -max_depth, 0.0, cv2.COLORMAP_TURBO)


def render_lidar_projection_sheet(
    frame: SurroundFrame,
    lidar: LidarFrame,
    points_ego: np.ndarray,
    tile_width: int = 640,
    max_depth: float = 80.0,
    point_radius: int = 2,
) -> np.ndarray:
    def annotate(image: np.ndarray, camera: CameraFrame) -> str:
        uv, depth, _ = project_to_camera(points_ego, lidar.ego_pose, camera, image.shape)
        order = np.argsort(-depth)  # far first, so near returns stay on top
        splat(image, uv[order].astype(np.int32), depth_to_bgr(depth[order], max_depth), point_radius)
        return f"{camera.channel} | {len(uv)} LiDAR pts"

    return camera_sheet(
        frame,
        tile_width,
        f"{frame.scene_name} | sample {frame.sample_index} | LiDAR projected into cameras",
        f"{len(lidar.sweeps)} sweep(s), {len(points_ego)} points, coloured by depth (0-{max_depth:.0f} m)",
        annotate,
    )
