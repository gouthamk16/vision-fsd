from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from fsd.bev_tensor import BevTensor, bev_tensor_from_lidar
from fsd.data import LidarFrame, SurroundFrame
from fsd.motion_planning.ego_motion import estimate_ego_state
from fsd.motion_planning.occupancy import build_collision_grid
from fsd.motion_planning.state import EgoState
from fsd.object_detection import (
    Box3D,
    NuScenesAnnotationLoader,
    PredictionLoader,
    draw_boxes_on_bev,
    ego_xy_to_bev_pixels,
)
from fsd.occupancy import TemporalOccupancyMapper, render_occupancy_bev
from fsd.tracking import GtVelocityTracker, PredictionVelocityTracker, TrackedObject


@dataclass(frozen=True)
class WorldObject:
    box: Box3D
    distance_m: float
    footprint_ego: np.ndarray
    track_id: str = ""
    velocity_ego: np.ndarray | None = None
    speed_mps: float = 0.0
    future_xy_ego: np.ndarray | None = None


@dataclass(frozen=True)
class BevWorldModel:
    frame: SurroundFrame
    ego: EgoState
    occupancy_probability: np.ndarray
    height_tensor: BevTensor
    collision_grid: np.ndarray
    gt_objects: list[WorldObject]
    pred_objects: list[WorldObject]
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    resolution: float


@dataclass(frozen=True)
class WorldModelConfig:
    x_range: tuple[float, float] = (-50.0, 50.0)
    y_range: tuple[float, float] = (-50.0, 50.0)
    resolution: float = 0.25
    occupancy_threshold: float = 0.62
    height_threshold: float = 0.45
    min_lidar_points: int = 1
    score_threshold: float = 0.1


class WorldModelBuilder:
    def __init__(
        self,
        config: WorldModelConfig | None = None,
        annotation_loader: NuScenesAnnotationLoader | None = None,
        prediction_loader: PredictionLoader | None = None,
    ) -> None:
        self.config = config or WorldModelConfig()
        self.annotation_loader = annotation_loader
        self.prediction_loader = prediction_loader
        self.mapper = TemporalOccupancyMapper(
            x_range=self.config.x_range,
            y_range=self.config.y_range,
            resolution=self.config.resolution,
        )
        self.gt_tracker = GtVelocityTracker()
        self.pred_tracker = PredictionVelocityTracker()
        self._previous_pose = None
        self._previous_timestamp_us = None

    def reset(self) -> None:
        self.mapper.reset()
        self.gt_tracker.reset()
        self.pred_tracker.reset()
        self._previous_pose = None
        self._previous_timestamp_us = None

    def step(self, frame: SurroundFrame, lidar: LidarFrame) -> BevWorldModel:
        occupancy = self.mapper.step(lidar)
        tensor = bev_tensor_from_lidar(
            lidar,
            x_range=self.config.x_range,
            y_range=self.config.y_range,
            resolution=self.config.resolution,
        )
        collision = build_collision_grid(
            occupancy,
            tensor.height_range,
            self.config.x_range,
            self.config.y_range,
            self.config.resolution,
            self.config.occupancy_threshold,
            self.config.height_threshold,
        )
        ego = estimate_ego_state(
            lidar.ego_pose,
            lidar.timestamp_us,
            self._previous_pose,
            self._previous_timestamp_us,
        )
        self._previous_pose = lidar.ego_pose
        self._previous_timestamp_us = lidar.timestamp_us

        gt_boxes = []
        if self.annotation_loader is not None:
            gt_boxes = self.annotation_loader.boxes_for_frame(frame, self.config.min_lidar_points)
        gt_tracked = self.gt_tracker.update(frame, gt_boxes)
        pred_boxes = []
        if self.prediction_loader is not None:
            pred_boxes = self.prediction_loader.boxes_for_frame(frame, self.config.score_threshold)
        pred_tracked = self.pred_tracker.update(frame, pred_boxes)

        return BevWorldModel(
            frame=frame,
            ego=ego,
            occupancy_probability=occupancy,
            height_tensor=tensor,
            collision_grid=collision.blocked,
            gt_objects=[_tracked_world_object(t) for t in gt_tracked],
            pred_objects=[_tracked_world_object(t) for t in pred_tracked],
            x_range=self.config.x_range,
            y_range=self.config.y_range,
            resolution=self.config.resolution,
        )


def _world_object(box: Box3D) -> WorldObject:
    return WorldObject(
        box=box,
        distance_m=float(np.linalg.norm(box.center_ego[:2])),
        footprint_ego=box.corners_ego[:, :2].copy(),
    )


def _tracked_world_object(tracked: TrackedObject) -> WorldObject:
    box = tracked.box
    return WorldObject(
        box=box,
        distance_m=float(np.linalg.norm(box.center_ego[:2])),
        footprint_ego=box.corners_ego[:, :2].copy(),
        track_id=tracked.track_id,
        velocity_ego=tracked.velocity_ego,
        speed_mps=tracked.speed_mps,
        future_xy_ego=tracked.future_xy_ego,
    )


def _draw_object_motion(
    image: np.ndarray,
    objects: list[WorldObject],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: float,
    scale: int,
    color: tuple[int, int, int] = (40, 220, 255),
    min_speed_mps: float = 0.5,
) -> int:
    """Overlay velocity arrows and a faded future ghost footprint. Returns moving count."""
    moving = 0
    for obj in objects:
        if obj.velocity_ego is None or obj.speed_mps < min_speed_mps:
            continue
        moving += 1
        center_xy = obj.box.center_ego[:2]
        tip_xy = center_xy + obj.velocity_ego  # 1 s lookahead: arrow length == speed in metres
        arrow = ego_xy_to_bev_pixels(
            np.stack([center_xy, tip_xy]), x_range, y_range, resolution, scale=scale
        )
        cv2.arrowedLine(image, tuple(arrow[0]), tuple(arrow[1]), color, 2, tipLength=0.3, line_type=cv2.LINE_AA)

        if obj.future_xy_ego is not None and len(obj.future_xy_ego):
            ghost = obj.footprint_ego + (obj.future_xy_ego[-1] - center_xy)
            gp = ego_xy_to_bev_pixels(ghost, x_range, y_range, resolution, scale=scale)
            cv2.polylines(image, [gp.reshape((-1, 1, 2))], True, color, 1, cv2.LINE_AA)

        label = f"{obj.speed_mps:.1f}m/s"
        cv2.putText(image, label, (int(arrow[1][0]) + 3, int(arrow[1][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)
    return moving


def render_world_model_bev(model: BevWorldModel, scale: int = 2) -> np.ndarray:
    image = render_occupancy_bev(
        model.frame,
        model.occupancy_probability,
        x_range=model.x_range,
        y_range=model.y_range,
        resolution=model.resolution,
        scale=scale,
    )
    image = draw_boxes_on_bev(
        image,
        [obj.box for obj in model.gt_objects],
        x_range=model.x_range,
        y_range=model.y_range,
        resolution=model.resolution,
        scale=scale,
        color_override=(80, 240, 80),
        title="GT objects",
    )
    image = draw_boxes_on_bev(
        image,
        [obj.box for obj in model.pred_objects],
        x_range=model.x_range,
        y_range=model.y_range,
        resolution=model.resolution,
        scale=scale,
        color_override=(70, 70, 255),
        title="Pred objects",
        label_scores=True,
    )
    moving_gt = _draw_object_motion(
        image, model.gt_objects, x_range=model.x_range, y_range=model.y_range,
        resolution=model.resolution, scale=scale, color=(40, 220, 255),
    )
    moving_pred = _draw_object_motion(
        image, model.pred_objects, x_range=model.x_range, y_range=model.y_range,
        resolution=model.resolution, scale=scale, color=(0, 165, 255),
    )
    blocked = int(model.collision_grid.sum())
    text = (
        f"world model | speed={model.ego.speed_mps:.1f}m/s | blocked={blocked} | "
        f"gt={len(model.gt_objects)}(mv {moving_gt}) | pred={len(model.pred_objects)}(mv {moving_pred})"
    )
    cv2.rectangle(image, (0, 0), (image.shape[1], 58 * scale), (18, 18, 18), -1)
    cv2.putText(image, text, (10, 25 * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(image, "free/unknown/occupied + height collision + object footprints", (10, 48 * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.42 * scale, (190, 220, 255), 1, cv2.LINE_AA)
    return image
