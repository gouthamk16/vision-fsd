import time
import cv2
import logging
from fsd.detect import VehicleTracker
from fsd.extract import FeatureExtractor

class Processor:
    def __init__(self, frame):
        self.raw_frame = frame
        self.processed_frame = None
        self.tracker = VehicleTracker()
        self.feature_extractor = FeatureExtractor(frame)
        self.logger = logging.getLogger('Processor')
        self.logger.debug('Processor initialized.')

    def calculate_distance(self):
        total_start_time = time.time()
        self.logger.debug('Starting calculate_distance.')
        try:
            self.feature_extractor.frame = self.raw_frame
            feature_frame, feature_time = self.feature_extractor.process_frame()
            self.logger.debug(f'Feature extraction and matching completed in {feature_time:.4f}s.')
            bb_coords, detection_time = self.tracker.track(frame=self.raw_frame)
            self.logger.debug(f'Object detection completed in {detection_time:.4f}s.')
            self.processed_frame = self.tracker.draw_bb(frame=feature_frame, bounding_box_coords=bb_coords, inference_time=detection_time)
        except Exception as e:
            self.logger.exception(f'Error in calculate_distance: {e}')
            raise
        total_time = time.time() - total_start_time
        total_fps = 1.0 / total_time if total_time > 0 else 0
        cv2.putText(self.processed_frame, f"Total FPS: {total_fps:.1f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(self.processed_frame, f"Total Time: {total_time*1000:.1f}ms", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        self.logger.debug(f'Frame processed in {total_time:.4f}s (Total FPS: {total_fps:.2f})')
        return self.processed_frame