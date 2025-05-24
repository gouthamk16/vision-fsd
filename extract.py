import torch
from ultralytics import YOLO
import cv2
import numpy as np

## Functiont to extract the important features in the image using cv2.goodFeaturesToTrack


class FeatureExtractor:
    def __init__(self, frame):
        self.frame = frame

    def extract_features(self):
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(image=gray, maxCorners=300, qualityLevel=0.01, minDistance=10)
        if corners is not None:
            corners = np.intp(corners)
        else:
            corners = np.array([])  # Empty array if no corners found
        edges = cv2.Canny(image=gray, threshold1=100, threshold2=200, L2gradient=True)
        return corners, edges
        
    def process_frame(self):
        corners1, edges1 = self.extract_features()

        for feature in corners1:
            fx, fy = feature.ravel()
            cv2.circle(self.frame, (fx, fy), 3, (255, 0, 0), 2)  # Increased radius to 3 and thickness to 2

        # Draw the edges
        # Draw the edges in red
        edges_colored = np.zeros_like(self.frame)
        edges_colored[edges1 != 0] = [0, 0, 255]  # BGR format - Red
        self.frame = cv2.addWeighted(self.frame, 0.8, edges_colored, 0.2, 0)
        
        print("No of corner features detected: ", len(corners1))
        return self.frame 