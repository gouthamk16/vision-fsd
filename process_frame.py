import time
import cv2
from detect import VehicleTracker
from extract import FeatureExtractor

class Processor:
    def __init__(self, frame):
        self.raw_frame = frame
        self.processed_frame = None
        self.tracker = VehicleTracker()
        self.feature_extractor = FeatureExtractor(frame)

    def init_process(self):
        total_start_time = time.time()
        
        bb_coords, detection_time = self.tracker.track(frame=self.raw_frame)
        self.processed_frame = self.tracker.draw_bb(frame=self.raw_frame.copy(), bounding_box_coords=bb_coords, inference_time=detection_time)
        
        self.feature_extractor.frame = self.processed_frame
        self.processed_frame, feature_time = self.feature_extractor.process_frame()
        
        total_time = time.time() - total_start_time
        total_fps = 1.0 / total_time if total_time > 0 else 0
        
        cv2.putText(self.processed_frame, f"Total FPS: {total_fps:.1f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(self.processed_frame, f"Total Time: {total_time*1000:.1f}ms", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return self.processed_frame

