import cv2
import numpy as np
import time
from monocular_vision.logging_utils import get_logger

class VisualOdometry:
    def __init__(self, focal_length, principal_point):
        self.logger = get_logger('VisualOdometry')
        
        # Camera intrinsic matrix K
        self.K = np.array([[focal_length[0], 0, principal_point[0]],
                           [0, focal_length[1], principal_point[1]],
                           [0, 0, 1]])
        self.logger.debug(f"Camera matrix initialized:\n{self.K}")
        
        # Previous frame data
        self.prev_kp = None
        self.prev_desc = None
        
        # SIFT for feature detection and description
        self.sift = cv2.SIFT_create(
            nfeatures=3000,
            nOctaveLayers=3,
            contrastThreshold=0.04,
            edgeThreshold=10,
            sigma=1.6
        )

        # FLANN for fast feature matching
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=75)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.logger.info('VisualOdometry initialized.')


    def _extract_features(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kps, descs = self.sift.detectAndCompute(gray, None)
        self.logger.debug(f"Extracted {len(kps) if kps else 0} keypoints.")
        return kps, descs


    def _match_features(self, desc1, desc2):
        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            return []
        matches = self.flann.knnMatch(desc1, desc2, k=2)
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        self.logger.debug(f"Found {len(good_matches)} good matches.")
        return good_matches


    def _recover_pose(self, kps, matches):
        R, t = np.identity(3), np.zeros((3, 1))
        
        if len(matches) <= 8:
            self.logger.warning(f"Not enough matches ({len(matches)}) to estimate motion.")
            return [], R, t, False
        
        pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kps[m.trainIdx].pt for m in matches])

        # Avoid processing static scenes by checking pixel movement
        median_flow = np.median(np.linalg.norm(pts2 - pts1, axis=1))
        if median_flow < 2.0:
            self.logger.debug(f"Insufficient motion detected: {median_flow:.2f} pixels")
            return [], R, t, False

        E, mask = cv2.findEssentialMat(pts2, pts1, self.K, 
                                       method=cv2.RANSAC, 
                                       prob=0.999, 
                                       threshold=1.0)
        
        if E is None:
            self.logger.warning("Could not compute Essential Matrix.")
            return [], R, t, False
        
        inlier_count, R, t, _ = cv2.recoverPose(E, pts2, pts1, self.K)
        
        # Validate pose: require sufficient inliers
        if inlier_count < 80:
            self.logger.warning(f"Insufficient inliers for reliable pose: {inlier_count}")
            return [], R, t, False
        
        # Validate pose: check for excessively large rotation
        angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        if angle > np.pi / 6:  # ~30 degrees
            self.logger.warning(f"Large rotation detected: {np.degrees(angle):.1f} degrees")
        
        self.logger.debug(f"Motion recovered. Translation: {t.flatten()}, Rotation: {np.degrees(angle):.2f}°")
        
        # Filter matches using the inlier mask from RANSAC
        inlier_matches = [m for i, m in enumerate(matches) if mask.ravel()[i]]
        self.logger.info(f"Found {len(inlier_matches)} inlier matches out of {len(matches)} total.")
        return inlier_matches, R, t, True
    
    
    def process_frame(self, frame):
        start_time = time.time()
        kps, descs = self._extract_features(frame)
        
        if self.prev_desc is None:
            self.prev_kp, self.prev_desc = kps, descs
            self.logger.debug('First frame processed; storing features.')
            return frame, np.identity(3), np.zeros((3, 1)), 0, True

        matches = self._match_features(self.prev_desc, descs)
        inlier_matches, R, t, success = self._recover_pose(kps, matches)

        annotated_frame = frame.copy()
        for match in inlier_matches:
            pt1 = tuple(map(int, self.prev_kp[match.queryIdx].pt))
            pt2 = tuple(map(int, kps[match.trainIdx].pt))
            cv2.line(annotated_frame, pt1, pt2, (255, 0, 0), 1)
            cv2.circle(annotated_frame, pt2, 2, (0, 255, 0), -1)

        status_color = (0, 255, 0) if success else (0, 0, 255)
        status_text = f"Matches: {len(inlier_matches)}, Success: {success}"
        cv2.putText(annotated_frame, status_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Update state for the next iteration
        self.prev_kp, self.prev_desc = kps, descs
        processing_time = time.time() - start_time
        return annotated_frame, R, t, processing_time, success