import time
import cv2
import logging
from fsd.detect import VehicleTracker
from fsd.extract import VisualOdometry as FeatureExtractor 

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class FrameProcessor:
    def __init__(self, frame_height, frame_width):
        self.logger = logging.getLogger('FrameProcessor')
        self.tracker = VehicleTracker() 
        self.feature_extractor = FeatureExtractor(focal_length=(1000, 1000), principal_point=(frame_width // 2, frame_height // 2))
        self.logger.info('FrameProcessor initialized successfully.')

    def process(self, frame):
        total_start_time = time.time()
        self.logger.debug('Starting frame processing.')
        
        try:
            feature_frame, _, _, feature_time = self.feature_extractor.process_frame(frame)
            self.logger.debug(f'Feature extraction completed in {feature_time*1000:.2f}ms.')
            
            bb_coords, detection_time = self.tracker.track(frame=frame)
            self.logger.debug(f'Object detection completed in {detection_time*1000:.2f}ms.')
            
            processed_frame = self.tracker.draw_bb(
                frame=feature_frame, 
                bounding_box_coords=bb_coords, 
                inference_time=detection_time
            )

        except Exception as e:
            self.logger.exception(f'Error during frame processing: {e}')
            return frame

        total_time = time.time() - total_start_time
        total_fps = 1.0 / total_time if total_time > 0 else 0
        
        cv2.putText(processed_frame, f"Total FPS: {total_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(processed_frame, f"Total Time: {total_time*1000:.1f}ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        self.logger.debug(f'Frame processed in {total_time*1000:.2f}ms (FPS: {total_fps:.2f})')
        
        return processed_frame
