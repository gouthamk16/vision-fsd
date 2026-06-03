from pathlib import Path

import cv2
import numpy as np

from fsd.data import CameraFrame, SurroundFrame, LidarFrame
from fsd.motion_planning.render import (
    draw_trajectory_points,
    project_trajectory_to_camera,
    render_planning_camera_result,
)
from fsd.motion_planning.state import EgoState, PlannedTrajectory, PlannerWorld, PlanningResult, TrajectoryPoint


def test_draw_trajectory_points_changes_pixels_on_blank_image():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    trajectory = PlannedTrajectory(
        points=[
            TrajectoryPoint(x=9.0, y=9.0, yaw=0.0, speed_mps=1.0, t=0.0),
            TrajectoryPoint(x=8.0, y=8.0, yaw=0.0, speed_mps=1.0, t=1.0),
        ],
        cost=0.0,
        is_emergency_stop=False,
    )

    rendered = draw_trajectory_points(
        image,
        trajectory,
        x_range=(0.0, 10.0),
        y_range=(0.0, 10.0),
        resolution=1.0,
        color=(0, 255, 0),
        thickness=1,
    )

    assert np.count_nonzero(rendered) > 0
    assert np.count_nonzero(image) == 0


def test_draw_trajectory_points_filters_off_image_points():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    trajectory = PlannedTrajectory(
        points=[
            TrajectoryPoint(x=100.0, y=100.0, yaw=0.0, speed_mps=1.0, t=0.0),
            TrajectoryPoint(x=5.0, y=5.0, yaw=0.0, speed_mps=1.0, t=1.0),
        ],
        cost=0.0,
        is_emergency_stop=False,
    )

    rendered = draw_trajectory_points(
        image,
        trajectory,
        x_range=(0.0, 10.0),
        y_range=(0.0, 10.0),
        resolution=1.0,
        color=(255, 0, 0),
        thickness=1,
    )

    assert rendered[5, 5].tolist() == [255, 0, 0]


def test_draw_trajectory_points_maps_non_symmetric_xy_to_row_col():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    trajectory = PlannedTrajectory(
        points=[
            TrajectoryPoint(x=8.0, y=3.0, yaw=0.0, speed_mps=1.0, t=0.0),
        ],
        cost=0.0,
        is_emergency_stop=False,
    )

    rendered = draw_trajectory_points(
        image,
        trajectory,
        x_range=(0.0, 10.0),
        y_range=(0.0, 10.0),
        resolution=1.0,
        color=(12, 34, 56),
        thickness=1,
    )

    assert rendered[2, 7].tolist() == [12, 34, 56]
    assert rendered[7, 2].tolist() == [0, 0, 0]


def _synthetic_front_camera(image_path: Path) -> CameraFrame:
    return CameraFrame(
        channel="CAM_FRONT",
        image_path=image_path,
        timestamp_us=1,
        sample_data_token="camera",
        calibrated_sensor={
            "translation": [0.0, 0.0, 0.0],
            "rotation": [-0.5, 0.5, -0.5, 0.5],
            "camera_intrinsic": [[20.0, 0.0, 50.0], [0.0, 20.0, 40.0], [0.0, 0.0, 1.0]],
        },
        ego_pose={"translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]},
    )


def _synthetic_frame(image_path: Path) -> SurroundFrame:
    return SurroundFrame(
        scene_token="scene",
        scene_name="scene-0001",
        sample_token="sample",
        sample_index=0,
        timestamp_us=1,
        cameras={"CAM_FRONT": _synthetic_front_camera(image_path)},
    )


def _synthetic_lidar() -> LidarFrame:
    return LidarFrame(
        channel="LIDAR_TOP",
        pointcloud_path=Path("unused.bin"),
        timestamp_us=1,
        sample_data_token="lidar",
        calibrated_sensor={"translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]},
        ego_pose={"translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]},
    )


def test_project_trajectory_to_camera_projects_ego_ground_points(tmp_path):
    image_path = tmp_path / "front.jpg"
    camera = _synthetic_front_camera(image_path)
    trajectory = PlannedTrajectory(
        points=[
            TrajectoryPoint(x=10.0, y=0.0, yaw=0.0, speed_mps=2.0, t=0.0),
            TrajectoryPoint(x=10.0, y=-1.0, yaw=0.0, speed_mps=2.0, t=0.5),
        ],
        cost=0.0,
        is_emergency_stop=False,
    )

    polylines = project_trajectory_to_camera(
        trajectory=trajectory,
        ego_pose={"translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]},
        camera=camera,
        image_shape=(80, 100, 3),
    )

    assert len(polylines) == 1
    assert polylines[0].tolist() == [[50, 40], [52, 40]]


def test_render_planning_camera_result_combines_front_camera_and_bev(tmp_path):
    image_path = tmp_path / "front.jpg"
    image = np.full((80, 100, 3), 30, dtype=np.uint8)
    cv2.imwrite(str(image_path), image)
    frame = _synthetic_frame(image_path)
    lidar = _synthetic_lidar()
    selected = PlannedTrajectory(
        points=[
            TrajectoryPoint(x=10.0, y=0.0, yaw=0.0, speed_mps=2.0, t=0.0),
            TrajectoryPoint(x=10.0, y=-1.0, yaw=0.0, speed_mps=2.0, t=0.5),
        ],
        cost=1.0,
        is_emergency_stop=False,
    )
    world = PlannerWorld(
        ego=EgoState(0.0, 0.0, 0.0, 2.0, 0.0, 1),
        collision_grid=np.zeros((4, 4), dtype=bool),
        occupancy_probability=np.full((4, 4), 0.5, dtype=np.float32),
        height_range=np.zeros((4, 4), dtype=np.float32),
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
        resolution=1.0,
    )
    result = PlanningResult(selected, [selected], [selected], world, "selected")

    rendered = render_planning_camera_result(frame, lidar, result, tile_width=200, bev_panel_height=90)

    assert rendered.shape[1] == 200
    assert rendered.shape[0] > 160
    assert np.count_nonzero(rendered != 30) > 0
