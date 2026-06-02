from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from fsd.bev_tensor import BevTensor, bev_tensor_from_lidar
from fsd.data import LidarFrame, SurroundFrame
from fsd.motion_planning.ego_motion import estimate_ego_state
from fsd.motion_planning.occupancy import build_collision_grid
from fsd.motion_planning.state import EgoState
from fsd.object_detection import Box3D, NuScenesAnnotationLoader, PredictionLoader, draw_boxes_on_bev
from fsd.occupancy import TemporalOccupancyMapper, render_occupancy_bev


@dataclass(frozen=True)
class WorldObject:
    box: Box3D
    distance_m: float
    footprint_ego: np.ndarray


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
        self._previous_pose = None
        self._previous_timestamp_us = None

    def reset(self) -> None:
        self.mapper.reset()
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
        pred_boxes = []
        if self.prediction_loader is not None:
            pred_boxes = self.prediction_loader.boxes_for_frame(frame, self.config.score_threshold)

        return BevWorldModel(
            frame=frame,
            ego=ego,
            occupancy_probability=occupancy,
            height_tensor=tensor,
            collision_grid=collision.blocked,
            gt_objects=[_world_object(box) for box in gt_boxes],
            pred_objects=[_world_object(box) for box in pred_boxes],
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
    blocked = int(model.collision_grid.sum())
    text = (
        f"world model | speed={model.ego.speed_mps:.1f}m/s | "
        f"blocked={blocked} | gt={len(model.gt_objects)} | pred={len(model.pred_objects)}"
    )
    cv2.rectangle(image, (0, 0), (image.shape[1], 58 * scale), (18, 18, 18), -1)
    cv2.putText(image, text, (10, 25 * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(image, "free/unknown/occupied + height collision + object footprints", (10, 48 * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.42 * scale, (190, 220, 255), 1, cv2.LINE_AA)
    return image
