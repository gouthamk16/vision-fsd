from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from fsd.bev import BevGrid, draw_arrow, draw_polygon
from fsd.contact_sheet import camera_sheet
from fsd.data import CameraFrame, NuScenesSceneLoader, Pose, SurroundFrame, quaternion_to_rotation_matrix
from fsd.draw import FONT

DETECTION_CLASSES = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.trailer": "trailer",
    "vehicle.construction": "construction",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
}

CLASS_COLORS = {
    "car": (80, 220, 80),
    "truck": (80, 180, 255),
    "bus": (60, 220, 255),
    "trailer": (160, 160, 255),
    "construction": (130, 100, 255),
    "motorcycle": (255, 160, 80),
    "bicycle": (255, 210, 80),
    "pedestrian": (255, 80, 220),
    "barrier": (200, 200, 200),
    "traffic_cone": (60, 140, 255),
}

_BOX_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))


@dataclass(frozen=True)
class Box3D:
    class_name: str
    center_ego: np.ndarray
    size: np.ndarray  # width, length, height (nuScenes order)
    yaw_ego: float
    score: float | None = None
    num_lidar_pts: int = 0
    instance_token: str = ""

    def corners(self) -> np.ndarray:
        """(8, 3) corners: bottom face front-left, front-right, rear-right, rear-left, then the top face."""
        width, length, height = self.size
        x = np.array([1, 1, -1, -1]) * length / 2
        y = np.array([1, -1, -1, 1]) * width / 2
        local = np.concatenate([np.stack([x, y, np.full(4, -height / 2)], 1), np.stack([x, y, np.full(4, height / 2)], 1)])
        c, s = np.cos(self.yaw_ego), np.sin(self.yaw_ego)
        return local @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T + self.center_ego

    @property
    def footprint(self) -> np.ndarray:
        return self.corners()[:4, :2]


def box_from_global(record: dict[str, Any], ego_pose: Pose, class_name: str, score: float | None = None) -> Box3D:
    """Box from a nuScenes global-frame record (translation, size, rotation) into the ego frame."""
    yaw_global_rot = ego_pose.rotation.T @ quaternion_to_rotation_matrix(record["rotation"])
    return Box3D(
        class_name=class_name,
        center_ego=ego_pose.invert(np.asarray(record["translation"], dtype=np.float64)),
        size=np.asarray(record["size"], dtype=np.float64),
        yaw_ego=float(np.arctan2(yaw_global_rot[1, 0], yaw_global_rot[0, 0])),
        score=score,
        num_lidar_pts=int(record.get("num_lidar_pts", 0)),
        instance_token=record.get("instance_token", ""),
    )


class NuScenesAnnotationLoader:
    """Ground-truth boxes for a frame, in the frame's reference ego pose."""

    def __init__(self, scene_loader: NuScenesSceneLoader):
        self.scene_loader = scene_loader
        categories = {c["token"]: c["name"] for c in scene_loader.load_table("category.json")}
        self.category_of_instance = {i["token"]: categories[i["category_token"]] for i in scene_loader.load_table("instance.json")}

    def boxes_for_frame(self, frame: SurroundFrame, min_lidar_points: int = 1) -> list[Box3D]:
        boxes = []
        for annotation in self.scene_loader.annotations(frame.sample_token):
            class_name = DETECTION_CLASSES.get(self.category_of_instance[annotation["instance_token"]])
            if class_name is not None and annotation["num_lidar_pts"] >= min_lidar_points:
                boxes.append(box_from_global(annotation, frame.ego_pose, class_name))
        return boxes


class PredictionLoader:
    """Predicted boxes exported by a detector such as CenterPoint.

    Accepted layouts: {"results": {sample_token: [global-frame nuScenes records]}}
    or {"samples": {sample_token: [ego-frame records]}}, where an ego-frame
    record has center_ego, yaw, size and detection_name / detection_score.
    """

    def __init__(self, prediction_path: str | Path):
        with Path(prediction_path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        self.global_frame = "results" in data
        self.records = data["results"] if self.global_frame else data["samples"]

    def boxes_for_frame(self, frame: SurroundFrame, score_threshold: float = 0.1) -> list[Box3D]:
        boxes = []
        for record in self.records.get(frame.sample_token, []):
            score = float(record["detection_score"])
            if score < score_threshold:
                continue
            name = record["detection_name"]
            if self.global_frame:
                boxes.append(box_from_global(record, frame.ego_pose, name, score))
            else:
                boxes.append(Box3D(name, np.asarray(record["center_ego"], float), np.asarray(record["size"], float), float(record["yaw"]), score))
        return boxes


def draw_boxes_on_bev(image: np.ndarray, grid: BevGrid, boxes: list[Box3D], color=None, labels: bool = True) -> None:
    """Draw footprints, heading ticks and short non-overlapping labels, in place."""
    placed: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=lambda b: float(np.hypot(*b.center_ego[:2]))):
        box_color = color or CLASS_COLORS.get(box.class_name, (255, 255, 255))
        footprint = box.footprint
        draw_polygon(image, grid, footprint, box_color, 2)
        draw_arrow(image, grid, box.center_ego[:2], footprint[:2].mean(axis=0), box_color, 1)
        if labels:
            text = box.class_name if box.score is None else f"{box.class_name} {box.score:.2f}"
            col, row = grid.to_pixels(footprint).min(axis=0)
            place_label(image, text, (int(col), int(row) - 4), box_color, placed)


def place_label(image: np.ndarray, text: str, origin: tuple[int, int], color, placed: list) -> None:
    """Draw `text` unless it would overlap a label already in `placed`."""
    (w, h), _ = cv2.getTextSize(text, FONT, 0.4, 1)
    x, y = origin
    rect = (x, y - h, x + w, y + 2)
    if any(rect[0] < r[2] and r[0] < rect[2] and rect[1] < r[3] and r[1] < rect[3] for r in placed):
        return
    placed.append(rect)
    cv2.putText(image, text, (x, y), FONT, 0.4, color, 1, cv2.LINE_AA)


def project_box_to_camera(box: Box3D, ego_pose: Pose, camera: CameraFrame, image_shape: tuple[int, ...]) -> np.ndarray | None:
    """Pixel corners of `box` in `camera`, or None if any corner is behind it or all are off-image."""
    corners = camera.calibration.invert(camera.ego_pose.invert(ego_pose.apply(box.corners())))
    if np.any(corners[:, 2] <= 0.5):
        return None
    projected = corners @ camera.intrinsic.T
    pixels = projected[:, :2] / projected[:, 2:3]
    h, w = image_shape[:2]
    if not np.any((pixels[:, 0] >= 0) & (pixels[:, 0] < w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)):
        return None
    return pixels.astype(np.int32)


def render_camera_boxes(frame: SurroundFrame, boxes: list[Box3D], title: str, tile_width: int = 640) -> np.ndarray:
    def annotate(image: np.ndarray, camera: CameraFrame) -> str:
        drawn = 0
        for box in boxes:
            pixels = project_box_to_camera(box, frame.ego_pose, camera, image.shape)
            if pixels is None:
                continue
            drawn += 1
            color = CLASS_COLORS.get(box.class_name, (255, 255, 255))
            for i, j in _BOX_EDGES:
                cv2.line(image, tuple(pixels[i]), tuple(pixels[j]), color, 1, cv2.LINE_AA)
            score = "" if box.score is None else f" {box.score:.2f}"
            cv2.putText(image, f"{box.class_name}{score}", tuple(pixels[4]), FONT, 0.4, color, 1, cv2.LINE_AA)
        return f"{camera.channel} | {drawn} boxes"

    return camera_sheet(frame, tile_width, f"{frame.scene_name} | sample {frame.sample_index} | {title}", f"{len(boxes)} boxes", annotate)
