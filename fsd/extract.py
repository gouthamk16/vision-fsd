import cv2
import numpy as np
import time
import logging

class VisualOdometry:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_frame = None
        self.prev_kp = None
        self.prev_desc = None
        self.camera_matrix = np.array([[800, 0, 320],
                                     [0, 800, 240],
                                     [0, 0, 1]], dtype=np.float32)
        self.trajectory = []
        self.current_pose = np.eye(4)
        self.logger = logging.getLogger('VisualOdometry')
        self.logger.debug('VisualOdometry initialized.')
        
    def extract_features(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, desc = self.orb.detectAndCompute(gray, None)
        return kp, desc
    
    def match_features(self, desc1, desc2):
        if desc1 is None or desc2 is None:
            return []
        matches = self.bf.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:min(len(matches), 100)]
        return good_matches
    
    def get_matched_points(self, kp1, kp2, matches):
        if len(matches) < 8:
            return None, None
        
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        return pts1, pts2
    
    def estimate_essential_matrix(self, pts1, pts2):
        E, mask = cv2.findEssentialMat(pts1, pts2, self.camera_matrix, 
                                      method=cv2.RANSAC, prob=0.999, threshold=1.0)
        return E, mask
    
    def recover_pose(self, E, pts1, pts2):
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.camera_matrix)
        return R, t, mask
    
    def update_pose(self, R, t):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        self.current_pose = self.current_pose @ T
        self.trajectory.append(self.current_pose[:3, 3].copy())
    
    def process_frame(self, frame):
        kp, desc = self.extract_features(frame)
        
        if self.prev_frame is None:
            self.prev_frame = frame
            self.prev_kp = kp
            self.prev_desc = desc
            self.logger.debug('First frame processed in VisualOdometry.')
            return frame, 0, 0
        
        start_time = time.time()
        
        matches = self.match_features(self.prev_desc, desc)
        pts1, pts2 = self.get_matched_points(self.prev_kp, kp, matches)
        
        R, t = np.eye(3), np.zeros((3, 1))
        num_matches = len(matches)
        
        if pts1 is not None and pts2 is not None and len(pts1) >= 8:
            E, mask = self.estimate_essential_matrix(pts1, pts2)
            if E is not None:
                R, t, pose_mask = self.recover_pose(E, pts1, pts2)
                self.update_pose(R, t)
        
        annotated_frame = frame.copy()
        for i, match in enumerate(matches[:50]):
            pt1 = tuple(map(int, self.prev_kp[match.queryIdx].pt))
            pt2 = tuple(map(int, kp[match.trainIdx].pt))
            cv2.circle(annotated_frame, pt2, 3, (0, 255, 0), -1)
            cv2.line(annotated_frame, pt1, pt2, (255, 0, 0), 1)
        
        processing_time = time.time() - start_time
        
        self.prev_frame = frame
        self.prev_kp = kp
        self.prev_desc = desc
        
        self.logger.debug(f'Frame processed in {processing_time:.4f}s with {num_matches} matches.')
        
        return annotated_frame, processing_time, num_matches
    

class FeatureExtractor:
    def __init__(self, frame):
        self.frame = frame
        self.vo = VisualOdometry()
        self.logger = logging.getLogger('FeatureExtractor')
        self.logger.debug('FeatureExtractor initialized.')
        self.prev_frame = None
        self.prev_kp = None
        self.prev_desc = None
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
    def extract_features(self):
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(image=gray, maxCorners=500, qualityLevel=0.01, minDistance=5)
        # Extracting keypoints from corner coords
        kps = []
        for c in corners:
            x, y = c.ravel()
            kps.append(cv2.KeyPoint(x=float(x), y=float(y), size=20))
        self.logger.debug(f"Keypoints extracted.")
        # Extracting descriptors from keypoints
        ord = cv2.ORB_create(nfeatures=1000)
        kps, descs = ord.compute(gray, kps)
        self.logger.debug(f"Descriptors extracted.")
        # Extracting edges from the frame
        if corners is not None:
            corners = np.intp(corners)
        else:
            corners = np.array([])
        edges = cv2.Canny(image=gray, threshold1=100, threshold2=200, L2gradient=True)
        return corners, edges, kps, descs

    def get_matched_points(self, kp1, kp2, matches):
        if len(matches) < 8:
            return None, None
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        return pts1, pts2

    def match_features(self, desc1, desc2):
        if desc1 is None or desc2 is None:
            return []
        matches = self.bf.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:min(len(matches), 100)]
        return good_matches
        
    def process_frame(self):
        start_time = time.time()
        
        _, _, kps, descs = self.extract_features()        
        self.logger.debug("Feature extraction completed.")
        
        if self.prev_frame is None:
            self.prev_frame = self.frame
            self.prev_kp = kps
            self.prev_desc = descs
            self.logger.debug('First frame processed in FeatureExtractor.')
            return self.frame, 0  # Return only two values

        matches = self.match_features(self.prev_desc, descs)
        pts1, pts2 = self.get_matched_points(self.prev_kp, kps, matches)
        self.logger.debug(f"Features matched.")

        # R, T = np.eye(3), np.zeros((3, 1)) # Rotation and translation matrices
        num_matches = len(matches)
        self.logger.debug(f"Number of matches: {num_matches}")

        annotated_frame = self.frame.copy()
        for i, match in enumerate(matches[:50]):
            pt1 = tuple(map(int, self.prev_kp[match.queryIdx].pt))
            pt2 = tuple(map(int, kps[match.trainIdx].pt))
            cv2.circle(annotated_frame, pt2, 1, (0, 255, 0), -1)
            cv2.line(annotated_frame, pt1, pt2, (255, 0, 0), 1)

        final_frame = annotated_frame
        processing_time = time.time() - start_time

        self.prev_frame = self.frame
        self.prev_kp = kps
        self.prev_desc = descs
                
        return final_frame, processing_time