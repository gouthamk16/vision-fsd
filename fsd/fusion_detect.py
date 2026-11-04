"""Camera + LiDAR frustum-fusion object detector.

A deployable, label-free detector that runs on this environment (no
MMDetection3D / spconv). Cameras answer *what and where in the image* (YOLO 2D
boxes + class); LiDAR answers *how far and how big*. For each 2D detection we
take the LiDAR points whose projection lands inside the box, keep the nearest
non-ground cluster, and fit a class-sized `Box3D` behind its visible face.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fsd.bev import BevGrid, draw_ego, lidar_canvas
from fsd.contact_sheet import read_image
from fsd.data import CAMERA_CHANNELS, LidarFrame, SurroundFrame
from fsd.draw import with_header
from fsd.lidar_projection import project_to_camera
from fsd.object_detection import Box3D, draw_boxes_on_bev, render_camera_boxes

# COCO class id -> our class name (only the agents we care about).
COCO_TO_CLASS = {0: "pedestrian", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Class size priors as (width, length, height) in metres, nuScenes convention.
SIZE_PRIORS = {
    "car": (1.9, 4.6, 1.7),
    "truck": (2.5, 7.0, 3.0),
    "bus": (2.9, 11.0, 3.5),
    "motorcycle": (0.8, 2.1, 1.4),
    "bicycle": (0.6, 1.8, 1.4),
    "pedestrian": (0.7, 0.7, 1.7),
}


def _fit_footprint(xy: np.ndarray, width: float, length: float) -> tuple[np.ndarray, float]:
    """Centre and yaw of a width x length box whose near face produced `xy`."""
    centroid = xy.mean(axis=0)
    away = centroid / max(float(np.linalg.norm(centroid)), 1e-6)
    if len(xy) < 5 or length - width < 0.5:
        return centroid + away * width / 2, float(np.arctan2(away[1], away[0]))
    _, axes = np.linalg.eigh(np.cov((xy - centroid).T))
    minor, major = axes[:, 0], axes[:, 1]
    normal = minor if minor @ away >= 0 else -minor
    # Seeing much less than the class length means an end face (rear or front):
    # its points spread across the vehicle, so the heading is the face normal.
    end_face = np.ptp((xy - centroid) @ major) < 0.6 * length
    heading = normal if end_face else (major if major @ away >= 0 else -major)
    center = centroid + normal * (length if end_face else width) / 2
    return center, float(np.arctan2(heading[1], heading[0]))


def _box_from_frustum(points: np.ndarray, class_name: str, score: float, ground_height: float, depth_band: float, min_points: int) -> Box3D | None:
    pts = points[points[:, 2] > ground_height]
    if len(pts) < min_points:
        return None
    # A 2D box also catches background: keep returns within a depth band of the
    # closest robust return.
    rng = np.hypot(pts[:, 0], pts[:, 1])
    cluster = pts[rng <= np.percentile(rng, 15) + depth_band]
    if len(cluster) < min_points:
        return None
    width, length, height = SIZE_PRIORS[class_name]
    center_xy, yaw = _fit_footprint(cluster[:, :2], width, length)
    # Rest the box on the ego ground plane (road ~ z=0) rather than the lowest visible return.
    center = np.array([center_xy[0], center_xy[1], height / 2.0])
    return Box3D(class_name, center, np.array([width, length, height]), yaw, score)


def _dedup(boxes: list[Box3D], merge_dist: float = 2.5) -> list[Box3D]:
    """Greedily drop duplicate detections of the same object across cameras."""
    kept: list[Box3D] = []
    for box in sorted(boxes, key=lambda b: -(b.score or 0.0)):
        if not any(o.class_name == box.class_name and np.linalg.norm(o.center_ego[:2] - box.center_ego[:2]) < merge_dist for o in kept):
            kept.append(box)
    return kept


class FrustumFusionDetector:
    """YOLO 2D detection on each camera fused with LiDAR for 3D localisation."""

    def __init__(
        self,
        weights_path: str | Path = "yolo26n.pt",
        device: str | None = None,
        conf: float = 0.35,
        ground_height: float = 0.3,
        depth_band: float = 6.0,
        min_points: int = 6,
    ):
        # Deferred: importing ultralytics pulls in torch and the YOLO registry,
        # which would cost every importer of this module even when no detection
        # view is requested.
        from ultralytics import YOLO

        self.model = YOLO(str(weights_path))
        self.device = device
        self.conf = conf
        self.ground_height = ground_height
        self.depth_band = depth_band
        self.min_points = min_points

    def detect(self, frame: SurroundFrame, lidar: LidarFrame, points_ego: np.ndarray) -> list[Box3D]:
        detections: list[Box3D] = []
        for channel in CAMERA_CHANNELS:
            camera = frame.cameras[channel]
            image = read_image(camera)
            result = self.model.predict(image, conf=self.conf, device=self.device, verbose=False)[0]
            if result.boxes is None or len(result.boxes) == 0:
                continue
            uv, _, kept = project_to_camera(points_ego, lidar.ego_pose, camera, image.shape)
            data = result.boxes.data.cpu().numpy()  # x1, y1, x2, y2, conf, cls
            for x1, y1, x2, y2, score, cls in data:
                class_name = COCO_TO_CLASS.get(int(cls))
                in_box = (uv[:, 0] >= x1) & (uv[:, 0] < x2) & (uv[:, 1] >= y1) & (uv[:, 1] < y2)
                if class_name is None or not in_box.any():
                    continue
                box = _box_from_frustum(points_ego[kept[in_box], :3], class_name, float(score), self.ground_height, self.depth_band, self.min_points)
                if box is not None:
                    detections.append(box)
        return _dedup(detections)


def render_fusion_bev(frame: SurroundFrame, points: np.ndarray, boxes: list[Box3D], view: BevGrid, background: np.ndarray | None = None) -> np.ndarray:
    image = lidar_canvas(points, view, background)
    draw_boxes_on_bev(image, view, boxes)
    draw_ego(image, view)
    return with_header(
        image,
        f"{frame.scene_name} | sample {frame.sample_index} | camera+LiDAR frustum detections",
        f"{len(boxes)} boxes: YOLO 2D box + nearest LiDAR cluster + class size prior",
    )


def render_fusion_cameras(frame: SurroundFrame, boxes: list[Box3D], tile_width: int = 360) -> np.ndarray:
    return render_camera_boxes(frame, boxes, "camera+LiDAR detections", tile_width)
