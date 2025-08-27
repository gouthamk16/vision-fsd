import os
from dotenv import load_dotenv
import cv2
import numpy as np
import time
from fsd.logging_utils import get_logger

load_dotenv()

class VisualOdometry:
    """
    A Visual Odometry class that extracts features, tracks them across frames,
    and estimates the camera's motion.
    """
    def __init__(self, focal_length, principal_point):
        self.logger = get_logger('VisualOdometry')
        
        # Camera Intrinsics
        self.K = np.array([[focal_length[0], 0, principal_point[0]],
                           [0, focal_length[1], principal_point[1]],
                           [0, 0, 1]])
        self.logger.debug(f"Camera matrix initialized:\n{self.K}")
        self.prev_kp = None
        self.prev_desc = None
        self.frames = []
        
        # Using SIFT for feature detection and description
        self.sift = cv2.SIFT_create(
            nfeatures=2800,
            nOctaveLayers=3,
            contrastThreshold=0.04,
            edgeThreshold=10,
            sigma=1.6
        )

        # Using FLANN for fast feature matching
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=75)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.logger.info('VisualOdometry initialized.')

    def _extract_features(self, frame):
        """Extracts keypoints and descriptors from a frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kps, descs = self.sift.detectAndCompute(gray, None)
        self.logger.debug(f"Extracted {len(kps) if kps else 0} keypoints.")
        return kps, descs

    def _match_features(self, desc1, desc2):
        """Matches descriptors and returns good matches using Lowe's ratio test."""
        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            return []
            
        # k=2 for Lowe's ratio test
        matches = self.flann.knnMatch(desc1, desc2, k=2)
        
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        if len(good_matches) > 0:
            self.logger.debug(f"Found {len(good_matches)} good matches.")
        return good_matches

    def _recover_pose(self, kps, matches):
        """Recovers the camera pose from matched keypoints."""
        R, t = np.identity(3), np.zeros((3, 1))
        if len(matches) > 8:
            # Get coordinates of matched keypoints
            pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kps[m.trainIdx].pt for m in matches])

            # Find Essential Matrix using RANSAC
            E, mask = cv2.findEssentialMat(pts2, pts1, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            
            if E is not None:
                # Recover Rotation and Translation 
                _, R, t, pose_mask = cv2.recoverPose(E, pts2, pts1, self.K)
                if mask is None:
                    mask = pose_mask
                    return matches, R, t
                if mask is not None:
                    mask = mask.ravel()
                    inlier_matches = [m for i, m in enumerate(matches) if mask[i]==1]
                    self.logger.debug(f"Found {len(inlier_matches)} inlier matches.")
                    return inlier_matches, R, t
                self.logger.debug(f"Motion recovered. Translation: {t.flatten()}")
            else:
                self.logger.warning("Could not compute Essential Matrix.")
        else:
            self.logger.warning(f"Not enough matches ({len(matches)}) to estimate motion.")
    
    def triangulate(self, R, t, kps, matches):
        # Projection matrices for the reference, current frame
        projection_matrix_p1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        projection_matrix_p2 = self.K @ np.hstack([R, t])
        # Matched keypoints
        kp1 = np.float32([self.prev_kp[m.queryIdx].pt for m in matches])
        kp2 = np.float32([kps[m.trainIdx].pt for m in matches])
        homogenous_coords = cv2.triangulatePoints(projection_matrix_p1, projection_matrix_p2, kp1, kp2)
        # Convert homogenous coords to cartesian.
        points_3d = (homogenous_coords / homogenous_coords[3])
        points_3d = points_3d[:3, :].T
        return points_3d


    def process_frame(self, frame):
        """
        Processes a new frame to estimate camera motion and returns the annotated frame.
        """
        start_time = time.time()
        self.frames.append(frame)
        
        # Extract features from the current frame
        kps, descs = self._extract_features(frame)
        
        # If this is the first frame, just store its features and return
        if self.prev_desc is None:
            self.prev_kp = kps
            self.prev_desc = descs
            self.logger.debug('First frame processed. Storing features.')
            return frame, np.identity(3), np.zeros((3, 1)), 0

        # Match features with the previous frame
        matches = self._match_features(self.prev_desc, descs)
        # Recover pose if enough matches are found
        inlier_matches, R, t = self._recover_pose(kps, matches)

        # Draw matches on the frame
        annotated_frame = frame.copy()
        # sort inliers by descriptor match distance (best first) for nicer viz
        inlier_matches = sorted(inlier_matches, key=lambda m: m.distance)[:100]
        # Draw lines for the top 100 matches
        for match in inlier_matches:
            pt1 = tuple(map(int, self.prev_kp[match.queryIdx].pt))
            pt2 = tuple(map(int, kps[match.trainIdx].pt))
            cv2.line(annotated_frame, pt1, pt2, (255, 0, 0), 1) 
            cv2.circle(annotated_frame, pt2, 2, (0, 255, 0), -1)  
        
        # Update state for the next frame
        self.prev_kp = kps
        self.prev_desc = descs

        processing_time = time.time() - start_time
        return annotated_frame, R, t, processing_time