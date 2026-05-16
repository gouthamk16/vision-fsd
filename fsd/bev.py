from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from fsd.data import LidarFrame, NuScenesSceneLoader, SurroundFrame
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


def save_lidar_bev_sequence(
    dataroot: str | Path | None = None,
    scene_index: int = 0,
    scene_name: str | None = None,
    start_sample_index: int = 0,
    max_frames: int | None = 40,
    output_path: str | Path = "outputs/nuscenes_lidar_bev_scene0_40f.mp4",
    fps: float = 2.0,
    resolution: float = 0.25,
    scale: int = 2,
) -> Path:
    loader = NuScenesSceneLoader(dataroot=dataroot)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    rendered = 0
    try:
        for frame, lidar in loader.iter_scene_frames(
            scene_index=scene_index,
            start_sample_index=start_sample_index,
            max_frames=max_frames,
            scene_name=scene_name,
            include_lidar=True,
        ):
            if lidar is None:
                raise RuntimeError("LiDAR frame was not loaded")
            bev = render_lidar_bev(frame, lidar, resolution=resolution, scale=scale)
            rendered += 1

            if writer is None:
                height, width = bev.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
                if not writer.isOpened():
                    raise OSError(f"Could not open video writer: {output}")
            writer.write(bev)
    finally:
        if writer is not None:
            writer.release()

    if rendered == 0:
        raise RuntimeError("No BEV frames were rendered")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render ego-frame LiDAR BEV videos from nuScenes.")
    parser.add_argument("--dataroot", default=None, help="nuScenes root. Defaults to NUSCENES_ROOT or D:/nuscenes.")
    parser.add_argument("--scene-index", type=int, default=0, help="Scene index to render.")
    parser.add_argument("--scene-name", default=None, help="Scene name to render, e.g. scene-0001.")
    parser.add_argument("--start-sample-index", type=int, default=0, help="First key sample index within the scene.")
    parser.add_argument("--frames", type=int, default=40, help="Maximum number of key samples to render.")
    parser.add_argument("--fps", type=float, default=2.0, help="Output video FPS.")
    parser.add_argument("--resolution", type=float, default=0.25, help="BEV grid meters per cell.")
    parser.add_argument("--scale", type=int, default=2, help="Nearest-neighbor scale for saved output.")
    parser.add_argument("--output", default="outputs/nuscenes_lidar_bev_scene0_40f.mp4", help="Output video path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = save_lidar_bev_sequence(
        dataroot=args.dataroot,
        scene_index=args.scene_index,
        scene_name=args.scene_name,
        start_sample_index=args.start_sample_index,
        max_frames=args.frames,
        output_path=args.output,
        fps=args.fps,
        resolution=args.resolution,
        scale=args.scale,
    )
    print(output)


if __name__ == "__main__":
    main()
