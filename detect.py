import torch
from ultralytics import YOLO
import cv2
import numpy as np
import time

class VehicleTracker:
    def __init__(self, confidence_threshold=0.4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO('yolo_models/yolov8n.pt')
        self.model.to(self.device)
        self.confidence_threshold = confidence_threshold
        self.vehicle_classes = [0, 1, 2, 3, 5, 7]  # Person, Bicycle, Car, Motorcycle, Bus, Truck
        self.pure_vehicle_classes = [1, 2, 3, 5, 7]  # Excluding Person (0)
        self.classMap = {0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck", 100: "NA"}
        self.vehicle_only = False
        self.detected_objects = {}

    def toggle_vehicle_only(self):
        self.vehicle_only = not self.vehicle_only
        return self.vehicle_only

    def draw_bb(self, frame, bounding_box_coords, inference_time):
        current_objects = []
        self.detected_objects = {}
        
        for result in bounding_box_coords:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[-1])
                conf = float(box.conf[-1])
                
                class_name = self.classMap.get(cls, "Unknown")
                if class_name not in self.detected_objects:
                    self.detected_objects[class_name] = 0
                self.detected_objects[class_name] += 1
                
                # Filter based on vehicle_only mode
                if self.vehicle_only and cls not in self.pure_vehicle_classes:
                    continue
                    
                if cls in self.vehicle_classes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    w = x2 - x1
                    h = y2 - y1
                    current_objects.append((int(x1), int(y1), int(w), int(h), cls, conf))

        for obj in current_objects:
            x, y, w, h, cls, conf = obj
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
            objectClass = self.classMap[cls]
            cv2.putText(frame, f"{objectClass} {conf:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        fps = 1.0 / inference_time if inference_time > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Time: {inference_time*1000:.1f}ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Objects: {len(current_objects)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        mode_text = "Vehicles Only" if self.vehicle_only else "All Objects"
        cv2.putText(frame, f"Mode: {mode_text}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, "Press 'v' to toggle vehicles only", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame
        
    def track(self, frame, target_fps=10):
        start_time = time.time()
        # Removed the problematic line: self.frames.append(frame)
        # Process the frame directly without storing it
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(frame_rgb, verbose=False)
        inference_time = time.time() - start_time
        return results, inference_time

    def create_detection_display(self, width=400, height=300):
        display = np.zeros((height, width, 3), dtype=np.uint8)
        display.fill(50)
        
        cv2.putText(display, "Real-time Detections", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_offset = 60
        for obj_name, count in self.detected_objects.items():
            if obj_name != "Unknown":
                text = f"{obj_name}: {count}"
                cv2.putText(display, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                y_offset += 30
                
        if not self.detected_objects:
            cv2.putText(display, "No objects detected", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            
        return display