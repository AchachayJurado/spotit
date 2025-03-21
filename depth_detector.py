import cv2
import torch
import numpy as np
from PIL import Image

class DepthDetector:
    def __init__(self):
        # Initialize MiDaS model for depth estimation
        self.model_type =  "MiDaS_small"  #"DPT_Swin2_Tiny_256"
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.midas.to(self.device)
        self.midas.eval()

        # Initialize transforms for MiDaS
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.small_transform

        # Initialize YOLOv8n for person detection
        from ultralytics import YOLO
        self.detector = YOLO('yolov8n.pt')

    def detect_depth(self, frame):
        """
        Estimate depth for a given frame and detect person
        Returns: depth map and person bounding box
        """
        # Transform input for MiDaS
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)

        # Run depth inference
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        
        # Normalize depth map for visualization
        depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
        
        # Get initial region of interest using depth map
        roi = self._get_depth_roi(depth_map)
        if roi is None:
            return depth_map, None
            
        # Crop frame to ROI
        x, y, w, h = roi
        roi_frame = frame[y:y+h, x:x+w]
        
        # Detect person in ROI using YOLO
        person_box = self._detect_person(roi_frame)
        if person_box is None:
            return depth_map, roi  # Fall back to depth-based ROI
            
        # Adjust person box coordinates relative to original frame
        x_rel, y_rel, w_rel, h_rel = person_box
        final_box = (
            x + x_rel,
            y + y_rel,
            w_rel,
            h_rel
        )
        
        return depth_map, final_box
    
    def _get_depth_roi(self, depth_map, threshold=0.7):
        """
        Get initial region of interest using depth map
        """
        # Threshold depth map to find closest regions
        mask = (depth_map < threshold).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Get largest contour (assumed to be the person)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add padding
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(depth_map.shape[1] - x, w + 2*padding)
        h = min(depth_map.shape[0] - y, h + 2*padding)
        
        return (x, y, w, h)
    
    def _detect_person(self, frame):
        """
        Detect person in frame using YOLOv8n
        Returns bounding box if person detected, None otherwise
        """
        try:
            # Run inference with person-only detection
            results = self.detector(frame, classes=[0], conf=0.5, verbose=False)
            
            # Get first person detection
            if len(results) > 0 and len(results[0].boxes.data) > 0:
                box = results[0].boxes.data[0]  # Get first box
                x1, y1, x2, y2 = box[:4].cpu().numpy()  # Get coordinates
                return (
                    int(x1),
                    int(y1),
                    int(x2 - x1),  # width
                    int(y2 - y1)   # height
                )
            return None
        except Exception as e:
            print(f"Error in person detection: {e}")
            return None
    
    def get_colored_depth_map(self, depth_map):
        """
        Convert depth map to colored visualization
        """
        colored = cv2.applyColorMap((depth_map * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    
    def crop_frame(self, frame, box):
        """
        Crop frame using bounding box
        """
        if box is None:
            return frame
            
        x, y, w, h = box
        # Ensure coordinates are within frame bounds
        x = max(0, x)
        y = max(0, y)
        w = min(frame.shape[1] - x, w)
        h = min(frame.shape[0] - y, h)
        
        return frame[y:y+h, x:x+w]
