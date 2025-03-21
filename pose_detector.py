from ultralytics import YOLO
import numpy as np
import cv2

class PoseDetector:
    def __init__(self, confidence_threshold=0.5):
        # Load YOLOv8n-pose model
        self.model = YOLO('yolov8n-pose.pt')
        self.confidence_threshold = confidence_threshold
        self.last_detection_time = None
        self.still_duration = 0
    
    def detect_pose(self, frame):
        """
        Detect pose in the given frame and return keypoints
        Returns: keypoints if detected, None otherwise
        """
        try:
            results = self.model(frame, verbose=False)
            if len(results) > 0 and len(results[0].keypoints.data) > 0:
                # Get keypoints for the first person detected
                keypoints = results[0].keypoints.data[0].cpu().numpy()
                if len(keypoints) > 0:  # Ensure we have keypoints
                    return keypoints
            return None
        except Exception as e:
            print(f"Error in pose detection: {e}")
            return None
    
    def draw_pose(self, frame, keypoints):
        """
        Draw the detected pose keypoints and connections for upper body
        """
        if keypoints is None or len(keypoints) == 0:
            return frame
        
        # Define all COCO keypoint connections
        connections = [
            (0, 1), (0, 2),     # Nose to eyes
            (1, 3), (2, 4),     # Eyes to ears
            (5, 6),             # Shoulders
            (5, 7), (7, 9),     # Left arm
            (6, 8), (8, 10),    # Right arm
            (5, 11), (6, 12),   # Shoulders to hips
            (11, 13), (13, 15), # Left leg
            (12, 14), (14, 16), # Right leg
            (11, 12),           # Hips
            (0, 5), (0, 6)      # Nose to shoulders
        ]
        
        # Draw all keypoints (0-16)
        for i in range(17):
            x, y, conf = keypoints[i]
            if conf > self.confidence_threshold:  # Only draw high-confidence points
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
        
        # Draw connections
        for connection in connections:
            start_idx, end_idx = connection
            if (keypoints[start_idx][2] > self.confidence_threshold and 
                keypoints[end_idx][2] > self.confidence_threshold):  # Check confidence
                start_point = (int(keypoints[start_idx][0]), 
                             int(keypoints[start_idx][1]))
                end_point = (int(keypoints[end_idx][0]), 
                           int(keypoints[end_idx][1]))
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
        return frame
