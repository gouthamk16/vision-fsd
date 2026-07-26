from __future__ import annotations

import cv2
import numpy as np

from fsd.data import LidarFrame, SurroundFrame
from fsd.lidar_projection import load_lidar_points, transform_points


def lidar_points_to_ego(lidar_points: np.ndarray, lidar: LidarFrame) -> np.ndarray:
    """Transform LiDAR sensor-frame points into the ego-vehicle frame."""
    return transform_points(
        lidar_points,
        lidar.calibrated_sensor["rotation"],
        lidar.calibrated_sensor["translation"],
    )


def _metric_to_pixel(
    points_ego: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_min, x_max = x_range
    y_min, y_max = y_range

    valid = (
        (points_ego[:, 0] >= x_min)
        & (points_ego[:, 0] <= x_max)
        & (points_ego[:, 1] >= y_min)
        & (points_ego[:, 1] <= y_max)
    )
    points = points_ego[valid]

    cols = ((y_max - points[:, 1]) / resolution).astype(np.int32)
    rows = ((x_max - points[:, 0]) / resolution).astype(np.int32)
    return rows, cols, points


def _draw_metric_grid(
    image: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: float,
    spacing_m: int = 10,
) -> None:
    x_min, x_max = x_range
    y_min, y_max = y_range
    height, width = image.shape[:2]

    for x in range(int(np.ceil(x_min / spacing_m) * spacing_m), int(x_max) + 1, spacing_m):
        row = int((x_max - x) / resolution)
        if 0 <= row < height:
            color = (60, 60, 60) if x != 0 else (110, 110, 110)
            cv2.line(image, (0, row), (width - 1, row), color, 1)
            cv2.putText(image, f"{x}m", (6, max(14, row - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    for y in range(int(np.ceil(y_min / spacing_m) * spacing_m), int(y_max) + 1, spacing_m):
        col = int((y_max - y) / resolution)
        if 0 <= col < width:
            color = (60, 60, 60) if y != 0 else (110, 110, 110)
            cv2.line(image, (col, 0), (col, height - 1), color, 1)
            cv2.putText(image, f"{y}m", (max(2, col + 3), height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)


def _draw_ego_vehicle(
    image: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: float,
) -> None:
    _, x_max = x_range
    _, y_max = y_range
    row = int(x_max / resolution)
    col = int(y_max / resolution)

    car_w = max(6, int(2.0 / resolution))
    car_l = max(12, int(4.5 / resolution))
    cv2.rectangle(
        image,
        (col - car_w // 2, row - car_l // 2),
        (col + car_w // 2, row + car_l // 2),
        (220, 220, 240),
        -1,
    )
    cv2.rectangle(
        image,
        (col - car_w // 2, row - car_l // 2),
        (col + car_w // 2, row + car_l // 2),
        (90, 90, 130),
        1,
    )
    cv2.arrowedLine(image, (col, row), (col, row - car_l), (255, 255, 255), 2, tipLength=0.35)


def render_lidar_bev(
    frame: SurroundFrame,
    lidar: LidarFrame,
    x_range: tuple[float, float] = (-50.0, 50.0),
    y_range: tuple[float, float] = (-50.0, 50.0),
    z_range: tuple[float, float] = (-3.0, 5.0),
    resolution: float = 0.25,
    scale: int = 2,
) -> np.ndarray:
    """Render a top-down ego-frame BEV from one LiDAR scan."""
    lidar_points = load_lidar_points(lidar.pointcloud_path)
    points_ego = lidar_points_to_ego(lidar_points, lidar)

    z_min, z_max = z_range
    height = int(round((x_range[1] - x_range[0]) / resolution))
    width = int(round((y_range[1] - y_range[0]) / resolution))
    canvas = np.full((height, width, 3), 20, dtype=np.uint8)
    _draw_metric_grid(canvas, x_range, y_range, resolution)

    valid_z = (points_ego[:, 2] >= z_min) & (points_ego[:, 2] <= z_max)
    rows, cols, visible_points = _metric_to_pixel(points_ego[valid_z], x_range, y_range, resolution)

    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    rows = rows[inside]
    cols = cols[inside]
    visible_points = visible_points[inside]

    if len(visible_points):
        z_norm = np.clip((visible_points[:, 2] - z_min) / (z_max - z_min), 0.0, 1.0)
        colors = cv2.applyColorMap((z_norm * 255).astype(np.uint8).reshape((-1, 1)), cv2.COLORMAP_TURBO)
        canvas[rows, cols] = colors.reshape((-1, 3))

    _draw_ego_vehicle(canvas, x_range, y_range, resolution)

    header_h = 58
    output = np.full((header_h + height, width, 3), 18, dtype=np.uint8)
    output[header_h:, :] = canvas
    title = f"{frame.scene_name} | sample {frame.sample_index} | ego-frame LiDAR BEV"
    stats = f"raw points={len(lidar_points)} | visible={len(visible_points)} | res={resolution:.2f}m/cell"
    cv2.putText(output, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(output, stats, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (190, 220, 255), 1, cv2.LINE_AA)

    if scale > 1:
        output = cv2.resize(output, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return output
