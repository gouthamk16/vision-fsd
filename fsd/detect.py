from dotenv import load_dotenv
import numpy as np
import torch
from ultralytics import YOLO
import cv2
import time
from fsd.logging_utils import get_logger

load_dotenv()

# Typical depth extent (meters) per class — depth map only gives the front surface,
# so we use priors to reconstruct the full box for BEV display.
_CLASS_DEPTH = {0: 0.4, 1: 1.8, 2: 4.5, 3: 2.2, 5: 12.0, 7: 8.5}

_BEV_W = 320
_BEV_H = 440
_BEV_SCALE = 5          # px per meter → ~83m forward range, ±32m lateral
_BEV_CAM_OFFSET = 30    # px from bottom edge to camera origin

_CLASS_ABBREV  = {1: "Bi", 2: "Ca", 3: "Mc", 5: "Bu", 7: "Tr"}
_CLASS_COLORS  = {
    1: (200, 130,  80),   # Bicycle  – orange
    2: ( 80, 210,  80),   # Car      – green
    3: ( 80, 130, 220),   # Moto     – blue
    5: (210, 210,  80),   # Bus      – yellow
    7: (200,  80, 200),   # Truck    – magenta
}


class VehicleTracker:
    def __init__(
        self,
        model_name="yolo26n.pt",
        confidence_threshold=0.25,
        tracking_confidence_threshold=0.15,
        display_confidence_threshold=0.25,
        iou_threshold=0.7,
        image_size=960,
        max_detections=100,
        end_to_end=True,
        enable_tracking=True,
        tracker_config="fsd/bytetrack_vehicle.yaml",
        max_track_misses=6,
    ):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.use_half = self.device.startswith("cuda")
        self.model_name = model_name
        self.model = YOLO(self.model_name)
        self.model.to(self.device)
        self.confidence_threshold = confidence_threshold
        self.tracking_confidence_threshold = tracking_confidence_threshold
        self.display_confidence_threshold = display_confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.max_detections = max_detections
        self.end_to_end = end_to_end
        self.enable_tracking = enable_tracking
        self.tracker_config = tracker_config
        self.max_track_misses = max_track_misses
        self.vehicle_classes = [1, 2, 3, 5, 7]
        self.pure_vehicle_classes = self.vehicle_classes
        self.classMap = {0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck", 100: "NA"}
        self.detected_objects = {}
        self.track_memory = {}
        self.logger = get_logger('VehicleTracker')
        self.logger.info(
            f"VehicleTracker initialized with {self.model_name} on {self.device}, "
            f"half={self.use_half}, end_to_end={self.end_to_end}, tracking={self.enable_tracking}."
        )

    def _estimate_3d_box(self, depth_map, box, cls, frame_shape):
        """
        Returns ((X, Y, Z), (W, H, D)) in meters, or None.

        Z  = median depth of the inner 50% of the 2D bbox (front surface depth).
        W,H = back-projected from pixel extents: e.g. W = pixel_w * Z / fx.
        D  = class-prior depth extent; box extends from Z to Z+D along camera axis.
        """
        if depth_map is None:
            return None

        x, y, w, h = box
        fh, fw = frame_shape[:2]
        x1 = max(0, x + int(w * 0.25))
        x2 = min(fw, x + int(w * 0.75))
        y1 = max(0, y + int(h * 0.35))
        y2 = min(fh, y + int(h * 0.85))
        if x2 <= x1 or y2 <= y1:
            return None

        roi = depth_map[y1:y2, x1:x2]
        valid = roi[np.isfinite(roi) & (roi > 0)]
        if valid.size < 5:
            return None

        Z = float(np.median(valid))
        if Z <= 0.1:
            return None

        fx = fy = 1000.0
        cx, cy = fw / 2.0, fh / 2.0
        X = (x + w / 2.0 - cx) * Z / fx
        Y = (y + h / 2.0 - cy) * Z / fy
        W = w * Z / fx
        H = h * Z / fy
        D = _CLASS_DEPTH.get(cls, 4.0)
        return (X, Y, Z), (W, H, D)

    def _draw_bev(self, frame, objects_3d):
        """Render a bird's-eye-view minimap and overlay it on the bottom-right of the frame."""
        panel = np.full((_BEV_H, _BEV_W, 3), 18, dtype=np.uint8)
        cam_x = _BEV_W // 2
        cam_y = _BEV_H - _BEV_CAM_OFFSET

        # FOV cone — ~87° horizontal for fx=1000 (tan(43°) ≈ 0.93)
        fov_tan = 0.93
        cone_reach = cam_y  # extend all the way to the top of the panel
        cv2.line(panel, (cam_x, cam_y),
                 (cam_x - int(cone_reach * fov_tan), cam_y - cone_reach), (42, 42, 42), 1)
        cv2.line(panel, (cam_x, cam_y),
                 (cam_x + int(cone_reach * fov_tan), cam_y - cone_reach), (42, 42, 42), 1)

        # Distance grid lines
        for d in [10, 20, 30, 50, 70]:
            ry = cam_y - int(d * _BEV_SCALE)
            if 0 < ry < _BEV_H:
                cv2.line(panel, (0, ry), (_BEV_W, ry), (40, 40, 40), 1)
                cv2.putText(panel, f"{d}m", (_BEV_W - 30, ry - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (85, 85, 85), 1)

        # Ego centre line (path ahead)
        cv2.line(panel, (cam_x, cam_y - 5), (cam_x, 0), (38, 38, 38), 1)

        # Ego vehicle icon
        ew, eh = 10, 16
        cv2.rectangle(panel,
                      (cam_x - ew // 2, cam_y - eh),
                      (cam_x + ew // 2, cam_y),
                      (170, 170, 210), -1)
        cv2.rectangle(panel,
                      (cam_x - ew // 2, cam_y - eh),
                      (cam_x + ew // 2, cam_y),
                      (100, 100, 130), 1)

        # Detected vehicles
        for center, dims, cls in objects_3d:
            X, _, Z = center
            W, _, D = dims
            bev_cx   = cam_x + int(X * _BEV_SCALE)
            front_py = cam_y - int(Z * _BEV_SCALE)
            back_py  = cam_y - int((Z + D) * _BEV_SCALE)
            wp       = max(6, int(W * _BEV_SCALE))

            # Skip objects entirely outside the panel
            if back_py >= _BEV_H or front_py <= 0 or bev_cx < 0 or bev_cx >= _BEV_W:
                continue

            fy_top = max(0, back_py)
            fy_bot = min(_BEV_H - 1, front_py)
            if fy_top >= fy_bot:
                continue

            col  = _CLASS_COLORS.get(cls, (150, 150, 150))
            dark = tuple(c // 2 for c in col)
            cv2.rectangle(panel, (bev_cx - wp // 2, fy_top), (bev_cx + wp // 2, fy_bot), col, -1)
            cv2.rectangle(panel, (bev_cx - wp // 2, fy_top), (bev_cx + wp // 2, fy_bot), dark, 1)

            # Label: class abbreviation + distance
            abbrev = _CLASS_ABBREV.get(cls, "?")
            lx = max(1, min(bev_cx - wp // 2, _BEV_W - 36))
            ly = max(8, fy_top - 3)
            cv2.putText(panel, f"{abbrev} {Z:.0f}m", (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (230, 230, 230), 1)

        # Panel title + border
        cv2.putText(panel, "BEV", (6, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110, 110, 110), 1)
        cv2.rectangle(panel, (0, 0), (_BEV_W - 1, _BEV_H - 1), (55, 55, 55), 1)

        # Composite onto main frame (bottom-right)
        fh, fw = frame.shape[:2]
        xo, yo = fw - _BEV_W - 10, fh - _BEV_H - 10
        if yo >= 0 and xo >= 0:
            roi = frame[yo:yo + _BEV_H, xo:xo + _BEV_W]
            frame[yo:yo + _BEV_H, xo:xo + _BEV_W] = cv2.addWeighted(roi, 0.15, panel, 0.95, 0)

    def draw_bb(self, frame, bounding_box_coords, inference_time, depth_map=None):
        current_objects = []
        current_track_ids = set()
        self.detected_objects = {}
        self.logger.debug(f'Drawing bounding boxes. Inference time: {inference_time:.4f}s')

        for result in bounding_box_coords:
            for box in result.boxes:
                cls = int(box.cls[-1])
                conf = float(box.conf[-1])
                class_name = self.classMap.get(cls, "Unknown")
                self.detected_objects[class_name] = self.detected_objects.get(class_name, 0) + 1

                if cls not in self.pure_vehicle_classes:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                track_id = int(box.id.item()) if box.id is not None else None
                if conf < self.display_confidence_threshold and track_id is None:
                    continue

                w, h = x2 - x1, y2 - y1
                # Lock class to first assigned for this track — prevents flicker
                # between visually similar classes (e.g. Truck ↔ Bus).
                if track_id is not None and track_id in self.track_memory:
                    cls = self.track_memory[track_id][4]
                obj = (int(x1), int(y1), int(w), int(h), cls, conf, track_id, False)
                current_objects.append(obj)
                if track_id is not None:
                    current_track_ids.add(track_id)
                    self.track_memory[track_id] = obj[:-1] + (0,)

        if self.enable_tracking:
            for track_id in list(self.track_memory.keys()):
                if track_id in current_track_ids:
                    continue
                x, y, w, h, cls, conf, _, misses = self.track_memory[track_id]
                misses += 1
                if misses > self.max_track_misses:
                    del self.track_memory[track_id]
                    continue
                self.track_memory[track_id] = (x, y, w, h, cls, conf, track_id, misses)
                current_objects.append((x, y, w, h, cls, conf, track_id, True))

        objects_3d = []

        for obj in current_objects:
            x, y, w, h, cls, conf, track_id, is_stale = obj
            color = (150, 70, 90) if is_stale else (255, 0, 123)
            thick = 1 if is_stale else 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thick)

            box3d = self._estimate_3d_box(depth_map, (x, y, w, h), cls, frame.shape)
            if box3d is not None:
                center, dims = box3d
                objects_3d.append((center, dims, cls))
                self.logger.debug(
                    f"{self.classMap.get(cls, '?')} ID:{track_id} "
                    f"X:{center[0]:.1f}m Z:{center[2]:.1f}m"
                )
                cv2.putText(frame, f"{center[2]:.1f}m", (x, y + h + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if objects_3d:
            self._draw_bev(frame, objects_3d)

        fps = 1.0 / inference_time if inference_time > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Time: {inference_time*1000:.1f}ms", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Objects: {len(current_objects)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        self.logger.debug(f"Objects detected: {self.detected_objects}")
        return frame

    def detect_bb(self, frame, target_fps=10):
        start_time = time.time()
        self.logger.debug('Tracking frame for object detection')
        try:
            confidence_threshold = (
                self.tracking_confidence_threshold
                if self.enable_tracking
                else self.confidence_threshold
            )
            inference_args = {
                "conf": confidence_threshold,
                "iou": self.iou_threshold,
                "classes": self.pure_vehicle_classes,
                "imgsz": self.image_size,
                "max_det": self.max_detections,
                "device": self.device,
                "half": self.use_half,
                "end2end": self.end_to_end,
                "verbose": False,
            }
            if self.enable_tracking:
                results = self.model.track(
                    frame,
                    persist=True,
                    tracker=self.tracker_config,
                    **inference_args,
                )
            else:
                results = self.model.predict(frame, **inference_args)
            inference_time = time.time() - start_time
            self.logger.debug(f'YOLO Inference completed in {inference_time:.4f}s.')
            return results, inference_time
        except Exception as e:
            self.logger.exception(f'Error during tracking: {e}')
            raise
