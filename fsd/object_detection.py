from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from fsd.data import CAMERA_CHANNELS, CameraFrame, NuScenesSceneLoader, SurroundFrame, _iter_json_objects
from fsd.lidar_projection import inverse_transform_points, quaternion_to_rotation_matrix, transform_points


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
    "barrier": (220, 220, 220),
    "traffic_cone": (60, 140, 255),
}
CAMERA_GT_COLOR = (60, 255, 60)
CAMERA_PRED_COLOR = (60, 80, 255)


@dataclass(frozen=True)
class Box3D:
    sample_token: str
    annotation_token: str
    class_name: str
    raw_category: str
    center_ego: np.ndarray
    size: np.ndarray
    yaw_ego: float
    corners_ego: np.ndarray
    num_lidar_pts: int
    num_radar_pts: int
    score: float | None = None
    source: str = "gt"
    instance_token: str = ""


class NuScenesAnnotationLoader:
    """Load nuScenes 3D boxes and convert them into ego-frame labels."""

    def __init__(self, scene_loader: NuScenesSceneLoader):
        self.scene_loader = scene_loader
        self.meta_dir = scene_loader.meta_dir
        self.instances = scene_loader._index_by_token(scene_loader._load_table("instance.json"))
        self.categories = scene_loader._index_by_token(scene_loader._load_table("category.json"))
        self._annotation_cache: dict[str, list[dict[str, Any]]] = {}

    def boxes_for_frame(
        self,
        frame: SurroundFrame,
        min_lidar_points: int = 1,
        detection_classes: dict[str, str] | None = None,
    ) -> list[Box3D]:
        detection_classes = detection_classes or DETECTION_CLASSES
        annotations = self._annotations_for_sample(frame.sample_token)
        boxes: list[Box3D] = []

        for annotation in annotations:
            if annotation.get("num_lidar_pts", 0) < min_lidar_points:
                continue

            instance = self.instances[annotation["instance_token"]]
            category = self.categories[instance["category_token"]]["name"]
            class_name = detection_classes.get(category)
            if class_name is None:
                continue

            box = annotation_to_ego_box(
                annotation=annotation,
                ego_pose=frame.ego_pose,
                class_name=class_name,
                raw_category=category,
            )
            boxes.append(box)

        return boxes

    def prefetch_sample_annotations(self, sample_tokens: set[str]) -> None:
        missing = sample_tokens - self._annotation_cache.keys()
        if not missing:
            return

        grouped = {token: [] for token in missing}
        table_path = self.meta_dir / "sample_annotation.json"
        for record in _iter_json_objects(table_path):
            sample_token = record.get("sample_token")
            if sample_token in grouped:
                grouped[sample_token].append(record)

        self._annotation_cache.update(grouped)

    def _annotations_for_sample(self, sample_token: str) -> list[dict[str, Any]]:
        cached = self._annotation_cache.get(sample_token)
        if cached is not None:
            return cached

        found: list[dict[str, Any]] = []
        table_path = self.meta_dir / "sample_annotation.json"
        for record in _iter_json_objects(table_path):
            if record.get("sample_token") == sample_token:
                found.append(record)

        self._annotation_cache[sample_token] = found
        return found


def annotation_to_ego_box(
    annotation: dict[str, Any],
    ego_pose: dict[str, Any],
    class_name: str,
    raw_category: str,
) -> Box3D:
    center_global = np.asarray(annotation["translation"], dtype=np.float64)
    size = np.asarray(annotation["size"], dtype=np.float64)
    width, length, height = size

    local_corners = np.array(
        [
            [length / 2, width / 2, 0.0],
            [length / 2, -width / 2, 0.0],
            [-length / 2, -width / 2, 0.0],
            [-length / 2, width / 2, 0.0],
        ],
        dtype=np.float64,
    )
    box_rotation_global = quaternion_to_rotation_matrix(annotation["rotation"])
    corners_global = local_corners @ box_rotation_global.T + center_global
    corners_ego = inverse_transform_points(
        corners_global,
        ego_pose["rotation"],
        ego_pose["translation"],
    )
    center_ego = inverse_transform_points(
        center_global.reshape(1, 3),
        ego_pose["rotation"],
        ego_pose["translation"],
    )[0]

    ego_rotation_global = quaternion_to_rotation_matrix(ego_pose["rotation"])
    box_rotation_ego = ego_rotation_global.T @ box_rotation_global
    yaw_ego = float(np.arctan2(box_rotation_ego[1, 0], box_rotation_ego[0, 0]))

    return Box3D(
        sample_token=annotation["sample_token"],
        annotation_token=annotation["token"],
        class_name=class_name,
        raw_category=raw_category,
        center_ego=center_ego,
        size=size,
        yaw_ego=yaw_ego,
        corners_ego=corners_ego,
        num_lidar_pts=int(annotation.get("num_lidar_pts", 0)),
        num_radar_pts=int(annotation.get("num_radar_pts", 0)),
        instance_token=annotation.get("instance_token", ""),
    )


def ego_yaw_to_rotation_matrix(yaw: float) -> np.ndarray:
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def box_bottom_corners_ego(center_ego: np.ndarray, size: np.ndarray, yaw_ego: float) -> np.ndarray:
    width, length, _ = size
    local_corners = np.array(
        [
            [length / 2, width / 2, 0.0],
            [length / 2, -width / 2, 0.0],
            [-length / 2, -width / 2, 0.0],
            [-length / 2, width / 2, 0.0],
        ],
        dtype=np.float64,
    )
    return local_corners @ ego_yaw_to_rotation_matrix(yaw_ego).T + center_ego


def box_corners_ego_3d(box: Box3D) -> np.ndarray:
    width, length, height = box.size
    local_corners = np.array(
        [
            [length / 2, width / 2, -height / 2],
            [length / 2, -width / 2, -height / 2],
            [-length / 2, -width / 2, -height / 2],
            [-length / 2, width / 2, -height / 2],
            [length / 2, width / 2, height / 2],
            [length / 2, -width / 2, height / 2],
            [-length / 2, -width / 2, height / 2],
            [-length / 2, width / 2, height / 2],
        ],
        dtype=np.float64,
    )
    return local_corners @ ego_yaw_to_rotation_matrix(box.yaw_ego).T + box.center_ego


def make_ego_box(
    sample_token: str,
    class_name: str,
    center_ego: np.ndarray,
    size: np.ndarray,
    yaw_ego: float,
    score: float | None = None,
    source: str = "prediction",
) -> Box3D:
    center_ego = np.asarray(center_ego, dtype=np.float64)
    size = np.asarray(size, dtype=np.float64)
    return Box3D(
        sample_token=sample_token,
        annotation_token="",
        class_name=class_name,
        raw_category=class_name,
        center_ego=center_ego,
        size=size,
        yaw_ego=float(yaw_ego),
        corners_ego=box_bottom_corners_ego(center_ego, size, yaw_ego),
        num_lidar_pts=0,
        num_radar_pts=0,
        score=score,
        source=source,
    )


class PredictionLoader:
    """Load predicted 3D boxes exported by a detector such as CenterPoint.

    Supported formats:
    1. {"results": {sample_token: [nuScenes global-box records...]}}
    2. {"sample_token": "...", "boxes": [ego-frame records...]}
    3. {"samples": {sample_token: [ego-frame records...]}}
    """

    def __init__(self, prediction_path: str | Path):
        self.path = Path(prediction_path)
        with self.path.open("r", encoding="utf-8") as handle:
            self.data = json.load(handle)

    def boxes_for_frame(self, frame: SurroundFrame, score_threshold: float = 0.1) -> list[Box3D]:
        records = self._records_for_sample(frame.sample_token)
        boxes = []
        for record in records:
            score = _prediction_score(record)
            if score is not None and score < score_threshold:
                continue
            boxes.append(prediction_record_to_ego_box(record, frame))
        return boxes

    def _records_for_sample(self, sample_token: str) -> list[dict[str, Any]]:
        if isinstance(self.data, dict) and "results" in self.data:
            return list(self.data["results"].get(sample_token, []))
        if isinstance(self.data, dict) and "samples" in self.data:
            return list(self.data["samples"].get(sample_token, []))
        if isinstance(self.data, dict) and self.data.get("sample_token") == sample_token:
            return list(self.data.get("boxes", []))
        if isinstance(self.data, list):
            return [record for record in self.data if record.get("sample_token") == sample_token]
        return []


def prediction_record_to_ego_box(record: dict[str, Any], frame: SurroundFrame) -> Box3D:
    class_name = (
        record.get("class_name")
        or record.get("detection_name")
        or record.get("name")
        or record.get("label")
        or "unknown"
    )
    score = _prediction_score(record)
    size = np.asarray(record.get("size") or record.get("dimensions"), dtype=np.float64)

    if "center_ego" in record or "translation_ego" in record:
        center_ego = np.asarray(record.get("center_ego") or record.get("translation_ego"), dtype=np.float64)
        yaw_ego = float(record.get("yaw", record.get("yaw_ego", 0.0)))
    elif "translation" in record:
        center_global = np.asarray(record["translation"], dtype=np.float64)
        center_ego = inverse_transform_points(
            center_global.reshape(1, 3),
            frame.ego_pose["rotation"],
            frame.ego_pose["translation"],
        )[0]
        if "rotation" in record:
            box_rotation_global = quaternion_to_rotation_matrix(record["rotation"])
            ego_rotation_global = quaternion_to_rotation_matrix(frame.ego_pose["rotation"])
            box_rotation_ego = ego_rotation_global.T @ box_rotation_global
            yaw_ego = float(np.arctan2(box_rotation_ego[1, 0], box_rotation_ego[0, 0]))
        else:
            yaw_ego = float(record.get("yaw", 0.0))
    else:
        raise ValueError(f"Prediction record has no supported center field: {record}")

    return make_ego_box(
        sample_token=record.get("sample_token", frame.sample_token),
        class_name=class_name,
        center_ego=center_ego,
        size=size,
        yaw_ego=yaw_ego,
        score=score,
        source="prediction",
    )


def _prediction_score(record: dict[str, Any]) -> float | None:
    value = record.get("score", record.get("detection_score", None))
    return None if value is None else float(value)


def ego_xy_to_bev_pixels(
    xy: np.ndarray,
    x_range: tuple[float, float] = (-50.0, 50.0),
    y_range: tuple[float, float] = (-50.0, 50.0),
    resolution: float = 0.25,
    header_h: int = 58,
    scale: int = 2,
) -> np.ndarray:
    x_max = x_range[1]
    y_max = y_range[1]
    cols = ((y_max - xy[:, 1]) / resolution)
    rows = ((x_max - xy[:, 0]) / resolution) + header_h
    pixels = np.stack([cols, rows], axis=1) * scale
    return pixels.astype(np.int32)


def draw_boxes_on_bev(
    bev_image: np.ndarray,
    boxes: list[Box3D],
    x_range: tuple[float, float] = (-50.0, 50.0),
    y_range: tuple[float, float] = (-50.0, 50.0),
    resolution: float = 0.25,
    header_h: int = 58,
    scale: int = 2,
    color_override: tuple[int, int, int] | None = None,
    title: str = "GT 3D boxes",
    thickness: int = 2,
    label_scores: bool = False,
) -> np.ndarray:
    image = bev_image.copy()
    height, width = image.shape[:2]

    for box in boxes:
        pixels = ego_xy_to_bev_pixels(
            box.corners_ego[:, :2],
            x_range=x_range,
            y_range=y_range,
            resolution=resolution,
            header_h=header_h,
            scale=scale,
        )
        if not np.any(
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < height)
        ):
            continue

        color = color_override or CLASS_COLORS.get(box.class_name, (255, 255, 255))
        cv2.polylines(image, [pixels.reshape((-1, 1, 2))], isClosed=True, color=color, thickness=thickness)

        front_mid = ((pixels[0] + pixels[1]) / 2).astype(np.int32)
        center = ego_xy_to_bev_pixels(
            box.center_ego[:2].reshape(1, 2),
            x_range=x_range,
            y_range=y_range,
            resolution=resolution,
            header_h=header_h,
            scale=scale,
        )[0]
        cv2.arrowedLine(image, tuple(center), tuple(front_mid), color, 2, tipLength=0.3)

        distance = float(np.linalg.norm(box.center_ego[:2]))
        if label_scores and box.score is not None:
            label = f"{box.class_name} {box.score:.2f} {distance:.1f}m"
        elif box.source == "gt":
            label = f"{box.class_name} {box.num_lidar_pts}pts {distance:.1f}m"
        else:
            label = f"{box.class_name} {distance:.1f}m"
        label_pos = (int(center[0]) + 4, int(center[1]) - 4)
        cv2.putText(image, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)

    cv2.putText(
        image,
        f"{title}: {len(boxes)}",
        (10, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return image


def project_ego_box_to_camera(
    box: Box3D,
    frame: SurroundFrame,
    camera: CameraFrame,
    image_shape: tuple[int, int, int],
) -> np.ndarray | None:
    corners_ego = box_corners_ego_3d(box)
    corners_global = transform_points(corners_ego, frame.ego_pose["rotation"], frame.ego_pose["translation"])
    corners_camera_ego = inverse_transform_points(corners_global, camera.ego_pose["rotation"], camera.ego_pose["translation"])
    corners_camera = inverse_transform_points(
        corners_camera_ego,
        camera.calibrated_sensor["rotation"],
        camera.calibrated_sensor["translation"],
    )
    depth = corners_camera[:, 2]
    if np.any(depth <= 0.5):
        return None

    intrinsic = np.asarray(camera.camera_intrinsic, dtype=np.float64)
    projected = corners_camera @ intrinsic.T
    pixels = projected[:, :2] / projected[:, 2:3]
    h, w = image_shape[:2]
    if not np.any((pixels[:, 0] >= 0) & (pixels[:, 0] < w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)):
        return None
    return pixels.astype(np.int32)


def draw_projected_box_on_image(
    image: np.ndarray,
    pixels: np.ndarray,
    color: tuple[int, int, int],
    label: str | None = None,
) -> None:
    edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    for i, j in edges:
        cv2.line(image, tuple(pixels[i]), tuple(pixels[j]), color, 2, cv2.LINE_AA)
    if label:
        anchor = tuple(pixels[4])
        cv2.putText(image, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)


def render_camera_box_sheet(
    frame: SurroundFrame,
    gt_boxes: list[Box3D],
    pred_boxes: list[Box3D],
    tile_width: int = 640,
) -> np.ndarray:
    tiles = []
    for channel in CAMERA_CHANNELS:
        camera = frame.cameras[channel]
        image = cv2.imread(str(camera.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read camera image: {camera.image_path}")

        for box in gt_boxes:
            pixels = project_ego_box_to_camera(box, frame, camera, image.shape)
            if pixels is not None:
                draw_projected_box_on_image(image, pixels, CAMERA_GT_COLOR, box.class_name)

        for box in pred_boxes:
            pixels = project_ego_box_to_camera(box, frame, camera, image.shape)
            if pixels is not None:
                score = "" if box.score is None else f" {box.score:.2f}"
                draw_projected_box_on_image(image, pixels, CAMERA_PRED_COLOR, f"{box.class_name}{score}")

        h, w = image.shape[:2]
        scale = tile_width / w
        tile = cv2.resize(image, (tile_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(
            tile,
            f"{channel} | GT green / pred red",
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
    header_h = 58
    sheet = np.zeros((header_h + top.shape[0] + bottom.shape[0], top.shape[1], 3), dtype=np.uint8)
    sheet[:] = (22, 22, 22)
    cv2.putText(
        sheet,
        f"{frame.scene_name} | sample {frame.sample_index} | 3D boxes projected to cameras",
        (12, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    y = header_h
    sheet[y:y + top.shape[0], :] = top
    y += top.shape[0]
    sheet[y:y + bottom.shape[0], :] = bottom
    return sheet
