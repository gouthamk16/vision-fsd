import cv2
import os
import time
from process_frame import Processor

video_path = "data/highway_bridge.mp4"
output_folder = "output"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_count = 0
total_processing_time = 0
processor = None

print(f"Starting video processing...")
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_start = time.time()
    if processor is None:
        processor = Processor(frame)
    else:
        processor.raw_frame = frame
        
    annotated_frame = processor.calculate_distance()
    detection_display = processor.get_detection_display()
    
    frame_time = time.time() - frame_start
    
    total_processing_time += frame_time
    frame_count += 1
    
    avg_fps = frame_count / total_processing_time if total_processing_time > 0 else 0
    
    cv2.putText(annotated_frame, f"Avg FPS: {avg_fps:.1f}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Video frame", annotated_frame)
    cv2.imshow("Detection Stats", detection_display)
    
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames, Avg FPS: {avg_fps:.2f}")

    key = cv2.waitKey(15)
    if key == ord('q'):
        break
    elif key == ord('v'):
        vehicle_only = processor.toggle_vehicle_only()
        print(f"Vehicle only mode: {'ON' if vehicle_only else 'OFF'}")

total_time = time.time() - start_time
final_avg_fps = frame_count / total_time if total_time > 0 else 0

print(f"Processing complete!")
print(f"Total frames: {frame_count}")
print(f"Total time: {total_time:.2f}s")
print(f"Average FPS: {final_avg_fps:.2f}")
print(f"Processing FPS: {frame_count/total_processing_time:.2f}")

cap.release()
cv2.destroyAllWindows()