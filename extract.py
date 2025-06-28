import cv2
import numpy as np
import time

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
        
        return annotated_frame, processing_time, num_matches
    
    def draw_trajectory(self, width=400, height=300):
        traj_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        if len(self.trajectory) < 2:
            cv2.putText(traj_img, "Building trajectory...", (10, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            return traj_img
        
        points = np.array(self.trajectory)
        
        if len(points) > 1:
            min_x, max_x = points[:, 0].min(), points[:, 0].max()
            min_z, max_z = points[:, 2].min(), points[:, 2].max()
            
            if max_x - min_x > 0 and max_z - min_z > 0:
                scale_x = (width - 40) / (max_x - min_x)
                scale_z = (height - 40) / (max_z - min_z)
                scale = min(scale_x, scale_z)
            else:
                scale = 1
            
            center_x = width // 2
            center_z = height // 2
            
            for i in range(1, len(points)):
                x1 = int(center_x + (points[i-1][0] - points[0][0]) * scale)
                z1 = int(center_z - (points[i-1][2] - points[0][2]) * scale)
                x2 = int(center_x + (points[i][0] - points[0][0]) * scale)
                z2 = int(center_z - (points[i][2] - points[0][2]) * scale)
                
                cv2.line(traj_img, (x1, z1), (x2, z2), (0, 255, 0), 2)
                cv2.circle(traj_img, (x2, z2), 3, (0, 0, 255), -1)
        
        current_pos = self.current_pose[:3, 3]
        cv2.putText(traj_img, f"X: {current_pos[0]:.2f}", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(traj_img, f"Y: {current_pos[1]:.2f}", (10, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(traj_img, f"Z: {current_pos[2]:.2f}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(traj_img, f"Points: {len(self.trajectory)}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return traj_img

class FeatureExtractor:
    def __init__(self, frame):
        self.frame = frame
        self.vo = VisualOdometry()

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
        
        vo_frame, vo_time, matches = self.vo.process_frame(self.frame)
        
        corners1, edges1 = self.extract_features()

        for feature in corners1:
            fx, fy = feature.ravel()
            cv2.circle(vo_frame, (fx, fy), 3, (255, 0, 0), 2)

        edges_colored = np.zeros_like(vo_frame)
        edges_colored[edges1 != 0] = [0, 0, 255]
        final_frame = cv2.addWeighted(vo_frame, 0.8, edges_colored, 0.2, 0)
        
        processing_time = time.time() - start_time
        
        cv2.putText(final_frame, f"Matches: {matches}", (10, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(final_frame, f"VO Time: {vo_time*1000:.1f}ms", (10, 320), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
 
        return final_frame, processing_time
    
    def get_trajectory_display(self):
        return self.vo.draw_trajectory()