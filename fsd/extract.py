import cv2
import numpy as np
import time
import logging


class MotionEstimator:
    def __init__(self, focal_length, principal_point):
        self.K = np.array([[focal_length[0], 0, principal_point[0]],
                           [0, focal_length[1], principal_point[1]],
                           [0, 0, 1]])
        self.logger = logging.getLogger('MotionEstimator')
        self.logger.debug('MotionEstimator initialized.')
        
    def recover_pose(self, pts1, pts2):
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)
        return R, t

class FeatureExtractor:
    def __init__(self, frame):
        self.frame = frame
        self.logger = logging.getLogger('FeatureExtractor')
        self.logger.debug('FeatureExtractor initialized.')
        self.prev_frame = None
        self.prev_kp = None
        self.prev_desc = None
        # Using FLANN matcher for SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
    def extract_features(self):
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        
        # Using SIFT detector and descriptor
        sift = cv2.SIFT_create(
            nfeatures=2000,       
            nOctaveLayers=3,       # Number of layers in each octave
            contrastThreshold=0.04, # Lower threshold = more features
            edgeThreshold=10,      # Higher threshold = fewer edge-like features
            sigma=1.6              # Gaussian blur sigma
        )
        
        kps, descs = sift.detectAndCompute(gray, None)
        self.logger.debug(f"SIFT keypoints extracted: {len(kps) if kps else 0}")
        
        # Convert keypoints to corner format for compatibility
        if kps:
            corners = np.array([[kp.pt[0], kp.pt[1]] for kp in kps], dtype=np.float32)
            corners = corners.reshape(-1, 1, 2)
        else:
            corners = np.array([])
        
        # Extracting edges from the frame
        edges = cv2.Canny(image=gray, threshold1=100, threshold2=200, L2gradient=True)
        
        return corners, edges, kps, descs

    def get_matched_points(self, kp1, kp2, matches):
        if len(matches) < 8:
            return None, None
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        return pts1, pts2

    def match_features(self, desc1, desc2):
        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            return []
        
        try:
            # Using FLANN matcher with Lowe's ratio test 
            matches = self.flann.knnMatch(desc1, desc2, k=2)
            
            # Apply Lowe's ratio test to filter good matches - andi
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance: 
                        good_matches.append(m)
            
            # Sort by distance and take more good matches
            good_matches = sorted(good_matches, key=lambda x: x.distance)
            good_matches = good_matches[:min(len(good_matches), 200)]  
            
            self.logger.debug(f"Good matches found: {len(good_matches)}")
            return good_matches
            
        except Exception as e:
            self.logger.error(f"Error in feature matching: {e}")
            return []
        
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

        num_matches = len(matches)
        self.logger.debug(f"Number of matches: {num_matches}")

        annotated_frame = self.frame.copy()
        # Draw more matches for better visualization
        for i, match in enumerate(matches[:100]):  # Increased from 50 to 100 - still andi
            pt1 = tuple(map(int, self.prev_kp[match.queryIdx].pt))
            pt2 = tuple(map(int, kps[match.trainIdx].pt))
            cv2.circle(annotated_frame, pt2, 2, (0, 255, 0), -1) 
            cv2.line(annotated_frame, pt1, pt2, (255, 0, 0), 1)

        final_frame = annotated_frame
        processing_time = time.time() - start_time

        self.prev_frame = self.frame
        self.prev_kp = kps
        self.prev_desc = descs
                
        return final_frame, processing_time