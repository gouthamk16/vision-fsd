import os
from dotenv import load_dotenv
import cv2
import time
from monocular_vision.logging_utils import setup_logging, get_logger

load_dotenv()
from monocular_vision.process_frame import FrameProcessor

load_dotenv()

def _resize_for_processing(frame, max_width=1920, max_height=1080):
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale == 1.0:
        return frame

    resized_width = int(width * scale)
    resized_height = int(height * scale)
    return cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)


def _make_output_path(video_path, timestamp):
    os.makedirs("outputs", exist_ok=True)
    name = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join("outputs", f"{name}_processed_{timestamp}.mp4")


def _depth_to_bgr(depth_map):
    import numpy as np
    import matplotlib
    lo, hi = depth_map.min(), depth_map.max()
    if hi - lo < 1e-6:
        return None
    norm = ((depth_map - lo) / (hi - lo) * 255).astype(np.uint8)
    cmap = matplotlib.colormaps.get_cmap("Spectral_r")
    rgb = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    return rgb[:, :, ::-1]  # RGB -> BGR for OpenCV


def driver(
    video_path,
    mode="stream",
    output_path=None,
    max_frames=None,
    target_motion_fps=20,
    high_fps_threshold=45,
):
    if mode not in {"stream", "save"}:
        raise ValueError("mode must be either 'stream' or 'save'")

    log_folder = "logs/"
    os.makedirs(log_folder, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(os.path.join(log_folder, f'app_{timestamp}.log'))
    logger = get_logger('driver')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return

    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_stride = 1
    if source_fps and source_fps >= high_fps_threshold:
        frame_stride = max(1, round(source_fps / target_motion_fps))
    output_fps = source_fps if source_fps else target_motion_fps

    source_frame_count = 0
    processed_frame_count = 0
    total_processing_time = 0
    processor = None
    writer = None
    depth_writer = None
    if mode == "save" and output_path is None:
        output_path = _make_output_path(video_path, timestamp)
    depth_output_path = output_path.replace(".mp4", "_depth.mp4") if output_path else None

    logger.info(f"Starting video processing: {video_path} ({mode})")
    logger.info(
        f"Video metadata: {source_width}x{source_height}, "
        f"{source_fps:.2f} FPS, {source_frames} frames, stride={frame_stride}"
    )
    if mode == "save":
        logger.info(
            f"Saving processed video to {output_path} at {output_fps:.2f} FPS "
            f"with {frame_stride} output frame(s) per processed frame"
        )
    start_time = time.time()
    end_of_video = False

    while True:
        if max_frames is not None and processed_frame_count >= max_frames:
            logger.info(f"Reached requested processed frame limit: {max_frames}")
            break

        ret, frame = cap.read()
        if not ret:
            logger.info("No more frames to read. Exiting loop.")
            break

        source_frame_count += 1
        frame = _resize_for_processing(frame)
        frame_start = time.time()
        if processor is None:
            logger.debug("Initializing Processor for the first frame.")
            processor = FrameProcessor(frame_height=frame.shape[0], frame_width=frame.shape[1])
        else:
            logger.info(f"Processing frame {processed_frame_count+1} (source frame {source_frame_count})")
        
        try:
            annotated_frame, depth_map = processor.process(frame)
        except Exception as e:
            logger.exception(f"Error processing frame {processed_frame_count+1}: {e}")
            break
        
        frame_time = time.time() - frame_start
        logger.info(f"Frame {processed_frame_count+1} processed in {frame_time:.4f}s")
        
        total_processing_time += frame_time
        processed_frame_count += 1
        
        avg_fps = processed_frame_count / total_processing_time if total_processing_time > 0 else 0
        
        cv2.putText(annotated_frame, f"Avg FPS: {avg_fps:.1f}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"Frame: {processed_frame_count}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        depth_vis = _depth_to_bgr(depth_map) if depth_map is not None else None

        if mode == "save":
            if writer is None:
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                height, width = annotated_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
                if not writer.isOpened():
                    logger.error(f"Could not open video writer: {output_path}")
                    break
            for _ in range(frame_stride):
                writer.write(annotated_frame)
            if depth_vis is not None:
                if depth_writer is None:
                    dh, dw = depth_vis.shape[:2]
                    depth_writer = cv2.VideoWriter(depth_output_path, cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (dw, dh))
                for _ in range(frame_stride):
                    depth_writer.write(depth_vis)
        else:
            cv2.imshow("Video frame", annotated_frame)
            if depth_vis is not None:
                cv2.imshow("Depth map", depth_vis)
        
        if processed_frame_count % 30 == 0:
            logger.info(f"Processed {processed_frame_count} frames, Avg FPS: {avg_fps:.2f}")

        if mode == "stream":
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("'q' pressed. Exiting.")
                break

        for _ in range(frame_stride - 1):
            if not cap.grab():
                end_of_video = True
                break
            source_frame_count += 1

        if end_of_video:
            logger.info("No more frames to read. Exiting loop.")
            break

    total_time = time.time() - start_time
    final_avg_fps = processed_frame_count / total_time if total_time > 0 else 0

    logger.info(f"Processing complete!")
    logger.info(f"Total source frames read: {source_frame_count}")
    logger.info(f"Total processed frames: {processed_frame_count}")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Average FPS: {final_avg_fps:.2f}")
    if total_processing_time > 0:
        logger.info(f"Processing FPS: {processed_frame_count/total_processing_time:.2f}")
    if output_path:
        logger.info(f"Output video: {output_path}")

    cap.release()
    if writer is not None:
        writer.release()
    if depth_writer is not None:
        depth_writer.release()
        logger.info(f"Depth video: {depth_output_path}")
    if mode == "stream":
        cv2.destroyAllWindows()

    return output_path if mode == "save" else None
