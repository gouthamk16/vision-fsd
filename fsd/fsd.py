import cv2
import os
import time
import logging
from fsd.logging_utils import setup_logging
from fsd.process_frame import Processor

def driver(video_path):
    video_path = video_path
    log_folder = "logs/"
    os.makedirs(log_folder, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(os.path.join(log_folder, f'app_{timestamp}.log'))
    logger = logging.getLogger('driver')

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    total_processing_time = 0
    processor = None

    logger.info(f"Starting video processing...")
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("No more frames to read. Exiting loop.")
            break

        frame_start = time.time()
        if processor is None:
            logger.debug("Initializing Processor for the first frame.")
            processor = Processor(frame)
        else:
            processor.raw_frame = frame
            logger.debug(f"Processing frame {frame_count+1}")
        
        try:
            annotated_frame = processor.calculate_distance()
        except Exception as e:
            logger.exception(f"Error processing frame {frame_count+1}: {e}")
            break
        
        frame_time = time.time() - frame_start
        logger.debug(f"Frame {frame_count+1} processed in {frame_time:.4f}s")
        
        total_processing_time += frame_time
        frame_count += 1
        
        avg_fps = frame_count / total_processing_time if total_processing_time > 0 else 0
        
        cv2.putText(annotated_frame, f"Avg FPS: {avg_fps:.1f}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Video frame", annotated_frame)
        
        if frame_count % 30 == 0:
            logger.info(f"Processed {frame_count} frames, Avg FPS: {avg_fps:.2f}")

        key = cv2.waitKey(15)
        if key == ord('q'):
            logger.info("'q' pressed. Exiting.")
            break

    total_time = time.time() - start_time
    final_avg_fps = frame_count / total_time if total_time > 0 else 0

    logger.info(f"Processing complete!")
    logger.info(f"Total frames: {frame_count}")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Average FPS: {final_avg_fps:.2f}")
    logger.info(f"Processing FPS: {frame_count/total_processing_time:.2f}")

    cap.release()
    cv2.destroyAllWindows()