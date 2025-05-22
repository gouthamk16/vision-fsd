# Draw bounding boxes for cars
import torch
from ultralytics import YOLO
import cv2
import numpy as np

class VehicleTracker:
    def __init__(self, confidence_threshold=0.4):
        # Initialize YOLO model with GPU support
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.notifier.speak("Detection Initiated")
        # Load YOLO model
        self.model = YOLO('models/yolov8n.pt')  # or 'yolov8n.pt' for less accuracy but faster inference
        self.model.to(self.device)
        self.frames = [np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)]
        # Tracking parameters
        self.confidence_threshold = confidence_threshold
        # Valid vehicle classes in YOLO v8
        self.vehicle_classes = [0, 1, 2, 3, 5, 7]  # car, bus, truck in YOLOv8
        self.classMap = {0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck", 100: "NA"}

    def draw_bb(self, frame, bounding_box_coords):
        current_vehicles = []
        cls = None
        
        for result in bounding_box_coords:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[-1])
                conf = float(box.conf[-1])
                
                if cls in self.vehicle_classes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    w = x2 - x1
                    h = y2 - y1
                    current_vehicles.append((int(x1), int(y1), int(w), int(h)))

        if cls not in self.vehicle_classes:
            cls = 100

        for vehicle in current_vehicles:
            x, y, w, h = vehicle
            # cv2.circle(frame, (x+(w//2), y+(h//2)), 0, (0, 255, 0), 5)
            # cv2.circle(black_frame, (x+(w//2), y+(h//2)), 0, (255, 0, 255), 5)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
            vehicleClass = self.classMap[cls]
            cv2.putText(frame, f"{vehicleClass}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        return frame
        
    def track(self, frame, target_fps=10):
        self.frames.append(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(frame_rgb, verbose=False)
    
        # draw bouding boxes
        return results
