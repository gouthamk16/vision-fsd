"""BEV world model: fused occupancy, ego state and tracked objects with motion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fsd.bev import BevGrid, draw_arrow, draw_ego, draw_metric_grid, draw_polygon
from fsd.data import Pose, SurroundFrame
from fsd.draw import with_header
from fsd.ego_motion import EgoState, estimate_ego_state
from fsd.nuscenes_map import NuScenesMapRenderer
from fsd.object_detection import NuScenesAnnotationLoader, PredictionLoader, draw_boxes_on_bev, place_label
from fsd.occupancy import occupancy_canvas
from fsd.tracking import GtVelocityTracker, PredictionVelocityTracker, TrackedObject

GT_COLOR = (80, 240, 80)
PRED_COLOR = (70, 90, 255)
MOTION_COLOR = (40, 220, 255)


@dataclass(frozen=True)
class BevWorldModel:
    frame: SurroundFrame
    ego: EgoState
    occupancy: np.ndarray
    gt_objects: list[TrackedObject]
    pred_objects: list[TrackedObject]


class WorldModelBuilder:
    def __init__(
        self,
        annotation_loader: NuScenesAnnotationLoader | None = None,
        prediction_loader: PredictionLoader | None = None,
        min_lidar_points: int = 1,
        score_threshold: float = 0.1,
    ) -> None:
        self.annotation_loader = annotation_loader
        self.prediction_loader = prediction_loader
        self.min_lidar_points = min_lidar_points
        self.score_threshold = score_threshold
        self.gt_tracker = GtVelocityTracker()
        self.pred_tracker = PredictionVelocityTracker()
        self._previous: tuple[str, Pose, int] | None = None

    def step(self, frame: SurroundFrame, occupancy: np.ndarray) -> BevWorldModel:
        previous = self._previous if self._previous and self._previous[0] == frame.scene_token else None
        ego = estimate_ego_state(
            frame.ego_pose,
            frame.timestamp_us,
            previous[1] if previous else None,
            previous[2] if previous else None,
        )
        self._previous = (frame.scene_token, frame.ego_pose, frame.timestamp_us)

        gt_boxes = self.annotation_loader.boxes_for_frame(frame, self.min_lidar_points) if self.annotation_loader else []
        pred_boxes = self.prediction_loader.boxes_for_frame(frame, self.score_threshold) if self.prediction_loader else []
        return BevWorldModel(
            frame=frame,
            ego=ego,
            occupancy=occupancy,
            gt_objects=self.gt_tracker.update(frame, gt_boxes),
            pred_objects=self.pred_tracker.update(frame, pred_boxes),
        )


def _draw_motion(image: np.ndarray, view: BevGrid, objects: list[TrackedObject], placed: list) -> int:
    """Velocity arrows (1 s lookahead) and the last predicted footprint. Returns moving count."""
    moving = [obj for obj in objects if len(obj.future_xy_ego)]
    for obj in moving:
        center = obj.box.center_ego[:2]
        draw_arrow(image, view, center, center + obj.velocity_ego, MOTION_COLOR, 2)
        draw_polygon(image, view, obj.box.footprint + (obj.future_xy_ego[-1] - center), MOTION_COLOR, 1)
        col, row = view.to_pixels((center + obj.velocity_ego)[None])[0]
        place_label(image, f"{obj.speed_mps:.1f} m/s", (int(col) + 4, int(row)), MOTION_COLOR, placed)
    return len(moving)


def render_world_model_bev(
    model: BevWorldModel,
    raster: BevGrid,
    view: BevGrid,
    map_renderer: NuScenesMapRenderer | None = None,
) -> np.ndarray:
    image = occupancy_canvas(model.occupancy, raster, view)
    if map_renderer is not None:
        map_renderer.draw(image, model.frame, view, dark=True, fill=False)
    draw_metric_grid(image, view)
    placed: list = []
    draw_boxes_on_bev(image, view, [o.box for o in model.gt_objects], GT_COLOR, labels=False)
    draw_boxes_on_bev(image, view, [o.box for o in model.pred_objects], PRED_COLOR, labels=False)
    moving = _draw_motion(image, view, model.gt_objects + model.pred_objects, placed)
    draw_ego(image, view)
    frame = model.frame
    return with_header(
        image,
        f"{frame.scene_name} | sample {frame.sample_index} | world model | ego {model.ego.speed_mps:.1f} m/s",
        f"GT boxes {len(model.gt_objects)} (green) | predicted {len(model.pred_objects)} (red) | moving {moving}: arrow = 1 s, outline = 3 s",
    )
