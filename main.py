## Python script to calculate the distance between the vechicle and other surrounding vehicles from a video frame.

import cv2
import os
from process_frame import Processor

video_path = "data/highway_bridge.mp4"
output_folder = "output"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Take the frame, calculate the distance to surrounding vehicles
    # Return the annotated frame
    processor = Processor(frame)
    annotated_frame = processor.calculate_distance()

    cv2.imshow("Video frame", annotated_frame)
    
    # Get the annotated frame for the raw frame
    # Get the camera focal lenght for the frame processed - wip

    key = cv2.waitKey(15)
    if key == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
