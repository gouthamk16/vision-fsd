"""Camera + LiDAR frustum-fusion object detector.

A deployable, label-free detector that runs on this environment (no
MMDetection3D / spconv). Cameras answer *what and where in the image* (YOLO 2D
boxes + class); LiDAR answers *how far and how big*. For each 2D detection we
take the LiDAR points whose projection lands inside the box, keep the nearest
non-ground cluster, and turn it into an ego-frame `Box3D`.

This is "frustum" detection without a learned 3D head: the 2D box defines a
viewing frustum, and the LiDAR points inside it localise the object in 3D.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from fsd.bev import lidar_points_to_ego, render_lidar_bev
from fsd.data import CAMERA_CHANNELS, CameraFrame, LidarFrame, SurroundFrame
from fsd.lidar_projection import lidar_points_to_camera, load_lidar_points
from fsd.object_detection import Box3D, draw_boxes_on_bev, make_ego_box, render_camera_box_sheet


# COCO class id -> our class name (only the agents we care about).
COCO_TO_CLASS = {0: "pedestrian", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Class size priors as (width, length, height) in metres — nuScenes convention.
SIZE_PRIORS = {
    "car": (1.9, 4.6, 1.7),
    "truck": (2.5, 7.0, 3.0),
    "bus": (2.9, 11.0, 3.5),
    "motorcycle": (0.8, 2.1, 1.4),
    "bicycle": (0.6, 1.8, 1.4),
    "pedestrian": (0.7, 0.7, 1.7),
}


def _project_lidar(
    points_sensor: np.ndarray,
    points_ego: np.ndarray,
    lidar: LidarFrame,
    camera: CameraFrame,
    image_shape: tuple[int, int],
    min_depth: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (uv, depth, ego_xyz) for LiDAR points visible in this camera."""
    cam = lidar_points_to_camera(points_sensor, lidar, camera)
    depth = cam[:, 2]
    front = depth > min_depth
    cam, depth, ego = cam[front], depth[front], points_ego[front]

    intrinsic = np.asarray(camera.camera_intrinsic, dtype=np.float64)
    proj = cam @ intrinsic.T
    uv = proj[:, :2] / proj[:, 2:3]

    h, w = image_shape[:2]
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return uv[inside], depth[inside], ego[inside]


def _box_from_frustum(
    ego_pts: np.ndarray,
    class_name: str,
    sample_token: str,
    score: float,
    ground_height: float,
    depth_band: float,
    min_points: int,
) -> Box3D | None:
    """Localise one object from the LiDAR points inside a 2D box."""
    nonground = ego_pts[:, 2] > ground_height
    pts = ego_pts[nonground]
    if len(pts) < min_points:
        return None

    # Keep the nearest cluster: a 2D box also catches background, so take the
    # points within a depth band of the closest robust return.
    rng = np.hypot(pts[:, 0], pts[:, 1])
    near = np.percentile(rng, 15)
    cluster = pts[rng <= near + depth_band]
    if len(cluster) < min_points:
        return None

    center_xy = cluster[:, :2].mean(axis=0)
    width, length, height = SIZE_PRIORS[class_name]
    # Rest the box on the ego-frame ground plane (road ~ z=0), so the box bottom
    # sits on the road rather than floating from the lowest visible return.
    center = np.array([center_xy[0], center_xy[1], height / 2.0])

    # Heading from the cluster's dominant horizontal axis (radial fallback).
    xy = cluster[:, :2] - center_xy
    if len(xy) >= 5:
        cov = xy.T @ xy
        eigvals, eigvecs = np.linalg.eigh(cov)
        major = eigvecs[:, int(np.argmax(eigvals))]
        yaw = float(np.arctan2(major[1], major[0]))
    else:
        yaw = float(np.arctan2(center_xy[1], center_xy[0]))

    return make_ego_box(
        sample_token=sample_token,
        class_name=class_name,
        center_ego=center,
        size=(width, length, height),
        yaw_ego=yaw,
        score=score,
        source="prediction",
    )


def _dedup(boxes: list[Box3D], merge_dist: float = 2.5) -> list[Box3D]:
    """Greedily drop duplicate detections of the same object across cameras."""
    kept: list[Box3D] = []
    for box in sorted(boxes, key=lambda b: -(b.score or 0.0)):
        clash = any(
            other.class_name == box.class_name
            and np.linalg.norm(other.center_ego[:2] - box.center_ego[:2]) < merge_dist
            for other in kept
        )
        if not clash:
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

    def detect(self, frame: SurroundFrame, lidar: LidarFrame) -> list[Box3D]:
        points_sensor = load_lidar_points(lidar.pointcloud_path)
        points_ego = lidar_points_to_ego(points_sensor, lidar)

        detections: list[Box3D] = []
        for channel in CAMERA_CHANNELS:
            camera = frame.cameras[channel]
            image = cv2.imread(str(camera.image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Could not read camera image: {camera.image_path}")

            result = self.model.predict(image, conf=self.conf, device=self.device, verbose=False)[0]
            if result.boxes is None or len(result.boxes) == 0:
                continue

            xyxy = result.boxes.xyxy.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()

            uv, _, ego = _project_lidar(points_sensor, points_ego, lidar, camera, image.shape)

            for (x1, y1, x2, y2), c, score in zip(xyxy, cls, confs):
                class_name = COCO_TO_CLASS.get(int(c))
                if class_name is None:
                    continue
                in_box = (uv[:, 0] >= x1) & (uv[:, 0] < x2) & (uv[:, 1] >= y1) & (uv[:, 1] < y2)
                if not in_box.any():
                    continue
                box = _box_from_frustum(
                    ego[in_box],
                    class_name,
                    frame.sample_token,
                    float(score),
                    self.ground_height,
                    self.depth_band,
                    self.min_points,
                )
                if box is not None:
                    detections.append(box)

        return _dedup(detections)


def render_fusion_bev(
    frame: SurroundFrame,
    lidar: LidarFrame,
    boxes: list[Box3D],
    resolution: float = 0.25,
    scale: int = 2,
) -> np.ndarray:
    bev = render_lidar_bev(frame, lidar, resolution=resolution, scale=scale)
    return draw_boxes_on_bev(
        bev,
        boxes,
        resolution=resolution,
        scale=scale,
        title="camera+LiDAR fused detections",
        label_scores=True,
    )


def render_fusion_cameras(frame: SurroundFrame, boxes: list[Box3D], tile_width: int = 360) -> np.ndarray:
    return render_camera_box_sheet(frame, gt_boxes=[], pred_boxes=boxes, tile_width=tile_width)
