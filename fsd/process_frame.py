import os
from dotenv import load_dotenv
import time
from fsd.logging_utils import get_logger
from fsd.detect import VehicleTracker
from fsd.depth import MonocularDepthEstimator
from fsd.vision import VisualOdometry as FeatureExtractor

load_dotenv()

class FrameProcessor:
    def __init__(self, frame_height, frame_width):
        self.logger = get_logger('FrameProcessor')
        self.tracker = VehicleTracker()
        try:
            self.depth_estimator = MonocularDepthEstimator()
        except Exception as e:
            self.depth_estimator = None
            self.logger.warning(f"Depth estimator unavailable; continuing without 3D depth labels: {e}")
        self.feature_extractor = FeatureExtractor(focal_length=(1000, 1000), principal_point=(frame_width // 2, frame_height // 2))
        self.logger.info('FrameProcessor initialized successfully.')

    def process(self, frame):
        total_start_time = time.time()
        self.logger.debug('Starting frame processing.')

        try:
            # If the image has an overall darkened hue, then use the image decomposition method to increase vsibility (i.e., nighttime image enhancement)
            # frame = something.enhance_image(frame)
            feature_frame, _, _, feature_time, success = self.feature_extractor.process_frame(frame)
            self.logger.debug(f'Feature extraction completed in {feature_time*1000:.2f}ms.')

            bb_coords, detection_time = self.tracker.detect_bb(frame=frame)
            self.logger.debug(f'Object detection completed in {detection_time*1000:.2f}ms.')

            depth_map = None
            if self.depth_estimator is not None:
                depth_map, depth_time = self.depth_estimator.estimate(frame)
                self.logger.debug(f'Depth estimation completed in {depth_time*1000:.2f}ms.')

            processed_frame = self.tracker.draw_bb(
                frame=feature_frame,
                bounding_box_coords=bb_coords,
                inference_time=detection_time,
                depth_map=depth_map,
            )

        except Exception as e:
            self.logger.exception(f'Error during frame processing: {e}')
            return frame, None

        total_time = time.time() - total_start_time
        total_fps = 1.0 / total_time if total_time > 0 else 0

        self.logger.debug(f'Frame processed in {total_time*1000:.2f}ms (FPS: {total_fps:.2f})')

        return processed_frame, depth_map
