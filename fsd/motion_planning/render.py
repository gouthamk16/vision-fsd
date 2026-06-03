from __future__ import annotations

import cv2
import numpy as np

from fsd.data import CameraFrame, LidarFrame, SurroundFrame
from fsd.lidar_projection import inverse_transform_points, transform_points
from fsd.motion_planning.state import PlannedTrajectory, PlanningResult


def _trajectory_pixels(
    trajectory: PlannedTrajectory,
    image_shape: tuple[int, ...],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: float,
) -> np.ndarray:
    height, width = image_shape[:2]
    x_max = x_range[1]
    y_max = y_range[1]
    pixels: list[tuple[int, int]] = []
    for point in trajectory.points:
        col = int((y_max - point.y) / resolution)
        row = int((x_max - point.x) / resolution)
        if 0 <= row < height and 0 <= col < width:
            pixels.append((col, row))
    return np.asarray(pixels, dtype=np.int32)


def draw_trajectory_points(
    image: np.ndarray,
    trajectory: PlannedTrajectory,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: float,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    output = image.copy()
    pixels = _trajectory_pixels(trajectory, output.shape, x_range, y_range, resolution)
    if pixels.size == 0:
        return output

    if len(pixels) >= 2:
        cv2.polylines(
            output,
            [pixels.reshape((-1, 1, 2))],
            isClosed=False,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
    radius = max(1, thickness)
    for col, row in pixels:
        cv2.circle(output, (int(col), int(row)), radius, color, -1, lineType=cv2.LINE_AA)
    return output


def _render_occupancy_background(probability: np.ndarray) -> np.ndarray:
    prob = np.asarray(probability, dtype=np.float32)
    occ_amt = np.clip((prob - 0.5) * 2.0, 0.0, 1.0)[..., None]
    free_amt = np.clip((0.5 - prob) * 2.0, 0.0, 1.0)[..., None]

    base = np.full((*prob.shape, 3), 86, dtype=np.float32)
    free_color = np.array([72, 46, 28], dtype=np.float32)
    occ_color = np.array([56, 210, 245], dtype=np.float32)
    canvas = base * (1.0 - free_amt - occ_amt) + free_color * free_amt + occ_color * occ_amt
    return np.clip(canvas, 0, 255).astype(np.uint8)


def _trajectory_ground_points(trajectory: PlannedTrajectory, ground_z: float) -> np.ndarray:
    if not trajectory.points:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(
        [(point.x, point.y, ground_z) for point in trajectory.points],
        dtype=np.float64,
    )


def project_trajectory_to_camera(
    trajectory: PlannedTrajectory,
    ego_pose: dict,
    camera: CameraFrame,
    image_shape: tuple[int, int, int],
    ground_z: float = 0.0,
    min_depth: float = 0.5,
) -> list[np.ndarray]:
    points_ego = _trajectory_ground_points(trajectory, ground_z)
    if points_ego.size == 0:
        return []

    points_global = transform_points(points_ego, ego_pose["rotation"], ego_pose["translation"])
    points_camera_ego = inverse_transform_points(
        points_global,
        camera.ego_pose["rotation"],
        camera.ego_pose["translation"],
    )
    points_camera = inverse_transform_points(
        points_camera_ego,
        camera.calibrated_sensor["rotation"],
        camera.calibrated_sensor["translation"],
    )

    depth = points_camera[:, 2]
    intrinsic = np.asarray(camera.camera_intrinsic, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        projected = points_camera @ intrinsic.T
        pixels = projected[:, :2] / projected[:, 2:3]

    h, w = image_shape[:2]
    valid = (
        np.isfinite(pixels).all(axis=1)
        & (depth > min_depth)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < w)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < h)
    )
    pixel_int = pixels.astype(np.int32, copy=False)

    polylines: list[np.ndarray] = []
    current: list[tuple[int, int]] = []
    for is_valid, pixel in zip(valid, pixel_int):
        if is_valid:
            current.append((int(pixel[0]), int(pixel[1])))
            continue
        if current:
            polylines.append(np.asarray(current, dtype=np.int32))
            current = []
    if current:
        polylines.append(np.asarray(current, dtype=np.int32))
    return polylines


def draw_camera_trajectory_polylines(
    image: np.ndarray,
    polylines: list[np.ndarray],
    color: tuple[int, int, int],
    thickness: int = 4,
) -> np.ndarray:
    output = image.copy()
    for polyline in polylines:
        if len(polyline) >= 2:
            cv2.polylines(
                output,
                [polyline.reshape((-1, 1, 2))],
                isClosed=False,
                color=color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )
        elif len(polyline) == 1:
            cv2.circle(output, tuple(int(v) for v in polyline[0]), thickness, color, -1, cv2.LINE_AA)
    return output


def _resize_width(image: np.ndarray, width: int) -> np.ndarray:
    h, w = image.shape[:2]
    if w == width:
        return image
    height = int(round(h * width / w))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _resize_height(image: np.ndarray, height: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h == height:
        return image
    width = int(round(w * height / h))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def render_planning_camera_result(
    frame: SurroundFrame,
    lidar: LidarFrame,
    result: PlanningResult,
    camera_channel: str = "CAM_FRONT",
    tile_width: int = 720,
    bev_panel_height: int = 260,
) -> np.ndarray:
    camera = frame.cameras[camera_channel]
    camera_image = cv2.imread(str(camera.image_path), cv2.IMREAD_COLOR)
    if camera_image is None:
        raise FileNotFoundError(f"Could not read camera image: {camera.image_path}")

    selected_color = (0, 190, 255) if result.selected.is_emergency_stop else (60, 220, 70)
    polylines = project_trajectory_to_camera(
        trajectory=result.selected,
        ego_pose=lidar.ego_pose,
        camera=camera,
        image_shape=camera_image.shape,
    )
    camera_overlay = draw_camera_trajectory_polylines(
        camera_image,
        polylines,
        color=selected_color,
        thickness=5,
    )
    camera_tile = _resize_width(camera_overlay, tile_width)
    cv2.rectangle(camera_tile, (0, 0), (camera_tile.shape[1], 42), (0, 0, 0), -1)
    status = "emergency_stop" if result.selected.is_emergency_stop else "selected"
    label = (
        f"{frame.scene_name} | sample {frame.sample_index} | "
        f"planner_camera {status} | valid={len(result.valid_candidates)}/{len(result.candidates)}"
    )
    cv2.putText(camera_tile, label, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 1, cv2.LINE_AA)

    bev = render_planning_result(result, scale=1)
    bev_tile = _resize_height(bev, bev_panel_height)
    panel = np.full((bev_panel_height, tile_width, 3), 18, dtype=np.uint8)
    x0 = max(0, (tile_width - bev_tile.shape[1]) // 2)
    x1 = min(tile_width, x0 + bev_tile.shape[1])
    panel[:, x0:x1] = bev_tile[:, : x1 - x0]
    return np.vstack([camera_tile, panel])


def render_planning_result(result: PlanningResult, scale: int = 2) -> np.ndarray:
    world = result.world
    canvas = _render_occupancy_background(world.occupancy_probability)

    for candidate in result.valid_candidates:
        canvas = draw_trajectory_points(
            canvas,
            candidate,
            world.x_range,
            world.y_range,
            world.resolution,
            color=(145, 145, 145),
            thickness=1,
        )

    selected_color = (0, 190, 255) if result.selected.is_emergency_stop else (60, 220, 70)
    canvas = draw_trajectory_points(
        canvas,
        result.selected,
        world.x_range,
        world.y_range,
        world.resolution,
        color=selected_color,
        thickness=2,
    )

    header_h = 42
    output = np.full((header_h + canvas.shape[0], canvas.shape[1], 3), 18, dtype=np.uint8)
    output[header_h:, :] = canvas
    status = "emergency_stop" if result.selected.is_emergency_stop else "selected"
    text = (
        f"{status} | reason={result.reason} | "
        f"valid={len(result.valid_candidates)}/{len(result.candidates)}"
    )
    cv2.putText(output, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)

    if scale > 1:
        return cv2.resize(output, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return output
