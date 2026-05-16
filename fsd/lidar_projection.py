from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from fsd.data import CAMERA_CHANNELS, CameraFrame, LidarFrame, NuScenesSceneLoader, SurroundFrame


def load_lidar_points(path: str | Path) -> np.ndarray:
    """Load nuScenes LIDAR_TOP .pcd.bin points as Nx3 XYZ float32."""
    points = np.fromfile(str(path), dtype=np.float32)
    if points.size % 5 != 0:
        raise ValueError(f"Unexpected nuScenes LiDAR point format in {path}: {points.size} floats")
    return points.reshape((-1, 5))[:, :3]


def quaternion_to_rotation_matrix(quaternion: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    """Convert nuScenes wxyz quaternion to a 3x3 rotation matrix."""
    q = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Cannot convert zero-norm quaternion to rotation matrix")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_points(points: np.ndarray, rotation: list[float], translation: list[float]) -> np.ndarray:
    """Apply source-to-target nuScenes transform to Nx3 row-vector points."""
    rot = quaternion_to_rotation_matrix(rotation)
    trans = np.asarray(translation, dtype=np.float64)
    return points @ rot.T + trans


def inverse_transform_points(points: np.ndarray, rotation: list[float], translation: list[float]) -> np.ndarray:
    """Apply inverse of a source-to-target nuScenes transform to Nx3 row-vector points."""
    rot = quaternion_to_rotation_matrix(rotation)
    trans = np.asarray(translation, dtype=np.float64)
    return (points - trans) @ rot


def lidar_points_to_camera(lidar_points: np.ndarray, lidar: LidarFrame, camera: CameraFrame) -> np.ndarray:
    """Transform LiDAR sensor-frame points into a camera sensor frame."""
    points = transform_points(
        lidar_points,
        lidar.calibrated_sensor["rotation"],
        lidar.calibrated_sensor["translation"],
    )
    points = transform_points(
        points,
        lidar.ego_pose["rotation"],
        lidar.ego_pose["translation"],
    )
    points = inverse_transform_points(
        points,
        camera.ego_pose["rotation"],
        camera.ego_pose["translation"],
    )
    points = inverse_transform_points(
        points,
        camera.calibrated_sensor["rotation"],
        camera.calibrated_sensor["translation"],
    )
    return points


def project_camera_points(
    camera_points: np.ndarray,
    intrinsic: list[list[float]],
    image_shape: tuple[int, int, int],
    min_depth: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-frame XYZ points to image UV pixels and return UV + depth."""
    depth = camera_points[:, 2]
    valid = depth > min_depth
    points = camera_points[valid]
    depth = depth[valid]

    if points.size == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.float32)

    intrinsic_matrix = np.asarray(intrinsic, dtype=np.float64)
    pixels = points @ intrinsic_matrix.T
    pixels = pixels[:, :2] / pixels[:, 2:3]

    height, width = image_shape[:2]
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    return pixels[inside].astype(np.int32), depth[inside].astype(np.float32)


def depth_to_bgr(depth: np.ndarray, max_depth: float = 80.0) -> np.ndarray:
    clipped = np.clip(depth, 0.0, max_depth)
    normalized = (255.0 * (1.0 - clipped / max_depth)).astype(np.uint8)
    return cv2.applyColorMap(normalized.reshape((-1, 1)), cv2.COLORMAP_TURBO)


def draw_projected_points(
    image: np.ndarray,
    uv: np.ndarray,
    depth: np.ndarray,
    point_radius: int = 1,
    max_depth: float = 80.0,
) -> np.ndarray:
    overlay = image.copy()
    if uv.size == 0:
        return overlay

    colors = depth_to_bgr(depth, max_depth=max_depth).reshape((-1, 3))
    for (u, v), color in zip(uv, colors):
        cv2.circle(
            overlay,
            (int(u), int(v)),
            point_radius,
            tuple(int(c) for c in color),
            -1,
            lineType=cv2.LINE_AA,
        )
    return overlay


def render_lidar_projection_sheet(
    frame: SurroundFrame,
    lidar: LidarFrame,
    tile_width: int = 640,
    max_depth: float = 80.0,
    point_radius: int = 2,
) -> np.ndarray:
    lidar_points = load_lidar_points(lidar.pointcloud_path)
    tiles = []

    for channel in CAMERA_CHANNELS:
        camera = frame.cameras[channel]
        image = cv2.imread(str(camera.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read camera image: {camera.image_path}")

        camera_points = lidar_points_to_camera(lidar_points, lidar, camera)
        uv, depth = project_camera_points(
            camera_points,
            camera.camera_intrinsic,
            image.shape,
        )
        height, width = image.shape[:2]
        scale = tile_width / width
        tile_size = (tile_width, int(round(height * scale)))
        tile = cv2.resize(image, tile_size, interpolation=cv2.INTER_AREA)

        if uv.size > 0:
            uv_scaled = (uv * scale).astype(np.int32)
            tile = draw_projected_points(tile, uv_scaled, depth, point_radius=point_radius, max_depth=max_depth)

        label = f"{channel} | {len(uv)} projected LiDAR pts"
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(
            tile,
            label,
            (10, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)

    top = np.hstack(tiles[:3])
    bottom = np.hstack(tiles[3:])
    header_h = 82
    sheet = np.zeros((header_h + top.shape[0] + bottom.shape[0], top.shape[1], 3), dtype=np.uint8)
    sheet[:] = (22, 22, 22)

    header = f"{frame.scene_name} | sample {frame.sample_index} | LiDAR projected into cameras"
    timestamps = f"sample t={frame.timestamp_us} | lidar t={lidar.timestamp_us} | points={len(lidar_points)}"
    cv2.putText(sheet, header, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(sheet, timestamps, (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (190, 220, 255), 1, cv2.LINE_AA)

    y = header_h
    sheet[y:y + top.shape[0], :] = top
    y += top.shape[0]
    sheet[y:y + bottom.shape[0], :] = bottom
    return sheet


def save_lidar_projection_sheet(
    dataroot: str | Path | None = None,
    scene_index: int = 0,
    sample_index: int = 0,
    scene_name: str | None = None,
    output_path: str | Path = "outputs/nuscenes_lidar_projection_scene0_sample0.jpg",
    tile_width: int = 640,
    max_depth: float = 80.0,
    point_radius: int = 1,
) -> Path:
    loader = NuScenesSceneLoader(dataroot=dataroot)
    frame = loader.get_surround_frame(
        scene_index=scene_index,
        sample_index=sample_index,
        scene_name=scene_name,
    )
    lidar = loader.get_lidar_frame(
        scene_index=scene_index,
        sample_index=sample_index,
        scene_name=scene_name,
    )
    sheet = render_lidar_projection_sheet(
        frame,
        lidar,
        tile_width=tile_width,
        max_depth=max_depth,
        point_radius=point_radius,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise OSError(f"Failed to write LiDAR projection sheet: {output}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project nuScenes LIDAR_TOP points into six camera images.")
    parser.add_argument("--dataroot", default=None, help="nuScenes root. Defaults to NUSCENES_ROOT or D:/nuscenes.")
    parser.add_argument("--scene-index", type=int, default=0, help="Scene index to render.")
    parser.add_argument("--scene-name", default=None, help="Scene name to render, e.g. scene-0001.")
    parser.add_argument("--sample-index", type=int, default=0, help="Sample index within the scene.")
    parser.add_argument("--tile-width", type=int, default=640, help="Width of each camera tile in pixels.")
    parser.add_argument("--max-depth", type=float, default=80.0, help="Depth in meters used for color scaling.")
    parser.add_argument("--point-radius", type=int, default=2, help="Projected point radius in pixels.")
    parser.add_argument(
        "--output",
        default="outputs/nuscenes_lidar_projection_scene0_sample0.jpg",
        help="Output image path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = save_lidar_projection_sheet(
        dataroot=args.dataroot,
        scene_index=args.scene_index,
        sample_index=args.sample_index,
        scene_name=args.scene_name,
        output_path=args.output,
        tile_width=args.tile_width,
        max_depth=args.max_depth,
        point_radius=args.point_radius,
    )
    print(output)


if __name__ == "__main__":
    main()
