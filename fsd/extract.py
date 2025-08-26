import cv2
import numpy as np
import time
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class VisualOdometry:
    """
    A Visual Odometry class that extracts features, tracks them across frames,
    and estimates the camera's motion.
    """
    def __init__(self, focal_length, principal_point):
        self.logger = logging.getLogger('VisualOdometry')
        
        # Camera Intrinsics
        self.K = np.array([[focal_length[0], 0, principal_point[0]],
                           [0, focal_length[1], principal_point[1]],
                           [0, 0, 1]])
        self.logger.debug(f"Camera matrix initialized:\n{self.K}")
        self.prev_kp = None
        self.prev_desc = None
        
        # Using SIFT for feature detection and description
        self.sift = cv2.SIFT_create(
            nfeatures=2000,
            nOctaveLayers=3,
            contrastThreshold=0.04,
            edgeThreshold=10,
            sigma=1.6
        )

        # Using FLANN for fast feature matching
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
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
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        self.logger.debug(f"Found {len(good_matches)} good matches.")
        return good_matches

    def process_frame(self, frame):
        """
        Processes a new frame to estimate camera motion and returns the annotated frame.
        """
        start_time = time.time()
        
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
        R, t = np.identity(3), np.zeros((3, 1))
        if len(matches) > 8:
            # Get coordinates of matched keypoints
            pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kps[m.trainIdx].pt for m in matches])

            # Find Essential Matrix using RANSAC
            E, mask = cv2.findEssentialMat(pts2, pts1, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            
            if E is not None:
                # Recover Rotation and Translation 
                _, R, t, mask = cv2.recoverPose(E, pts2, pts1, self.K)
                self.logger.debug(f"Motion recovered. Translation: {t.flatten()}")
            else:
                self.logger.warning("Could not compute Essential Matrix.")
        else:
            self.logger.warning(f"Not enough matches ({len(matches)}) to estimate motion.")

        # Draw matches on the frame
        annotated_frame = frame.copy()
        # Draw lines for the top 100 matches
        for match in matches[:100]:
            pt1 = tuple(map(int, self.prev_kp[match.queryIdx].pt))
            pt2 = tuple(map(int, kps[match.trainIdx].pt))
            cv2.line(annotated_frame, pt1, pt2, (255, 0, 0), 1) 
            cv2.circle(annotated_frame, pt2, 2, (0, 255, 0), -1)  
        
        # Update state for the next frame
        self.prev_kp = kps
        self.prev_desc = descs

        processing_time = time.time() - start_time
        return annotated_frame, R, t, processing_time