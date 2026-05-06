from dotenv import load_dotenv
import torch
from ultralytics import YOLO
import cv2
import time
from fsd.logging_utils import get_logger

load_dotenv()

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
        self.vehicle_classes = [1, 2, 3, 5, 7]  # Bicycle, Car, Motorcycle, Bus, Truck
        self.pure_vehicle_classes = self.vehicle_classes
        self.classMap = {0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck", 100: "NA"}
        self.detected_objects = {}
        self.track_memory = {}
        self.logger = get_logger('VehicleTracker')
        self.logger.info(
            f"VehicleTracker initialized with {self.model_name} on {self.device}, "
            f"half={self.use_half}, end_to_end={self.end_to_end}, tracking={self.enable_tracking}."
        )

    def draw_bb(self, frame, bounding_box_coords, inference_time):
        current_objects = []
        current_track_ids = set()
        self.detected_objects = {}
        self.logger.debug(f'Drawing bounding boxes. Inference time: {inference_time:.4f}s')
        
        for result in bounding_box_coords:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[-1])
                conf = float(box.conf[-1])
                
                class_name = self.classMap.get(cls, "Unknown")
                if class_name not in self.detected_objects:
                    self.detected_objects[class_name] = 0
                self.detected_objects[class_name] += 1
                
                if cls not in self.pure_vehicle_classes:
                    continue
                    
                if cls in self.vehicle_classes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    track_id = int(box.id.item()) if box.id is not None else None
                    if conf < self.display_confidence_threshold and track_id is None:
                        continue

                    w = x2 - x1
                    h = y2 - y1
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

        for obj in current_objects:
            x, y, w, h, cls, conf, track_id, is_stale = obj
            box_color = (150, 70, 90) if is_stale else (255, 0, 123)
            box_thickness = 1 if is_stale else 2
            cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, box_thickness)
            object_class = self.classMap[cls]
            label = f"{object_class} {conf:.2f}"
            if track_id is not None:
                label = f"{label} ID:{track_id}"
            # cv2.putText(frame, label, (x, max(y-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 123), 1)

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
