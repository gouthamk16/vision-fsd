import cv2
import numpy as np
import time

class FeatureExtractor:
    def __init__(self, frame):
        self.frame = frame

    def extract_features(self):
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(image=gray, maxCorners=300, qualityLevel=0.01, minDistance=10)
        if corners is not None:
            corners = np.intp(corners)
        else:
            corners = np.array([])
        edges = cv2.Canny(image=gray, threshold1=100, threshold2=200, L2gradient=True)
        return corners, edges
        
    def process_frame(self):
        start_time = time.time()
        corners1, edges1 = self.extract_features()

        for feature in corners1:
            fx, fy = feature.ravel()
            cv2.circle(self.frame, (fx, fy), 3, (255, 0, 0), 2)

        edges_colored = np.zeros_like(self.frame)
        edges_colored[edges1 != 0] = [0, 0, 255]
        self.frame = cv2.addWeighted(self.frame, 0.8, edges_colored, 0.2, 0)
        
        processing_time = time.time() - start_time
 
        return self.frame, processing_time