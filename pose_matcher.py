import numpy as np
from scipy.spatial.distance import euclidean

class PoseMatcher:
    def __init__(self, similarity_threshold=0.85, confidence_threshold=0.7):
        self.similarity_threshold = similarity_threshold
        self.confidence_threshold = confidence_threshold
        
        # Define joint connections and their base weights
        self.angle_joints = [
            # Torso core (highest weight as it's the most stable)
            ([5, 11, 12], 2.0),  # left shoulder - hips
            ([6, 12, 11], 2.0),  # right shoulder - hips
            ([5, 6, 12], 2.0),   # shoulders - right hip
            ([6, 5, 11], 2.0),   # shoulders - left hip
            
            # Arms (high weight as they're key for poses)
            ([5, 7, 9], 1.5),    # left arm
            ([6, 8, 10], 1.5),   # right arm
            ([7, 5, 6], 1.5),    # left shoulder angle
            ([8, 6, 5], 1.5),    # right shoulder angle
            
            # Legs (medium weight)
            ([11, 13, 15], 1.2), # left leg
            ([12, 14, 16], 1.2), # right leg
            ([13, 11, 12], 1.2), # left hip angle
            ([14, 12, 11], 1.2), # right hip angle
            
            # Head (lower weight as it's more variable)
            ([1, 0, 2], 1.0),    # eyes
            ([3, 1, 0], 1.0),    # left eye-ear
            ([4, 2, 0], 1.0),    # right eye-ear
            ([0, 5, 6], 1.0),    # neck
        ]
        
        # Keypoint pairs for distance calculation
        self.keypoint_pairs = [
            (5, 6),   # shoulders
            (11, 12), # hips
            (7, 8),   # elbows
            (9, 10),  # wrists
            (13, 14), # knees
            (15, 16), # ankles
        ]
    
    def _normalize_keypoints(self, keypoints):
        """Normalize keypoints based on torso"""
        # Get torso keypoints
        left_shoulder = keypoints[5][:2]
        right_shoulder = keypoints[6][:2]
        left_hip = keypoints[11][:2]
        right_hip = keypoints[12][:2]
        
        # Calculate torso center and scale
        torso_center = np.mean([left_shoulder, right_shoulder, left_hip, right_hip], axis=0)
        torso_size = np.mean([
            np.linalg.norm(left_shoulder - right_shoulder),
            np.linalg.norm(left_hip - right_hip),
            np.linalg.norm(left_shoulder - left_hip),
            np.linalg.norm(right_shoulder - right_hip)
        ])
        
        # Normalize all keypoints
        normalized_keypoints = keypoints.copy()
        normalized_keypoints[:, :2] = (keypoints[:, :2] - torso_center) / torso_size
        
        return normalized_keypoints
    
    def _calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points"""
        v1 = p1[:2] - p2[:2]
        v2 = p3[:2] - p2[:2]
        
        # Normalize vectors
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm == 0 or v2_norm == 0:
            return 0
            
        v1_normalized = v1 / v1_norm
        v2_normalized = v2 / v2_norm
        
        # Calculate angle using dot product
        dot_product = np.clip(np.dot(v1_normalized, v2_normalized), -1.0, 1.0)
        angle = np.arccos(dot_product)
        
        return np.degrees(angle)
    
    def _get_joint_angles(self, keypoints):
        """Calculate angles for all defined joint connections"""
        angles = []
        confidences = []  # Track confidence for each angle
        
        for (joint_indices, _) in self.angle_joints:
            p1 = keypoints[joint_indices[0]]
            p2 = keypoints[joint_indices[1]]
            p3 = keypoints[joint_indices[2]]
            
            # Calculate minimum confidence for this angle
            min_conf = min(p1[2], p2[2], p3[2])
            confidences.append(min_conf)
            
            # Only calculate angle if confidence is high enough
            if min_conf > self.confidence_threshold:
                angle = self._calculate_angle(p1, p2, p3)
                angles.append(angle)
            else:
                angles.append(0)
        
        return np.array(angles), np.array(confidences)
    
    def _calculate_distance_similarity(self, pose1, pose2):
        """Calculate similarity based on Euclidean distances between corresponding points"""
        total_similarity = 0
        valid_pairs = 0
        
        for pair in self.keypoint_pairs:
            p1_a = pose1[pair[0]]
            p1_b = pose1[pair[1]]
            p2_a = pose2[pair[0]]
            p2_b = pose2[pair[1]]
            
            # Check confidence
            if (min(p1_a[2], p1_b[2], p2_a[2], p2_b[2]) > self.confidence_threshold):
                # Calculate distance between corresponding points
                dist1 = np.linalg.norm(p1_a[:2] - p1_b[:2])
                dist2 = np.linalg.norm(p2_a[:2] - p2_b[:2])
                
                # Convert distance difference to similarity
                similarity = max(0, 1 - abs(dist1 - dist2))
                total_similarity += similarity
                valid_pairs += 1
        
        return total_similarity / valid_pairs if valid_pairs > 0 else 0
    
    def _calculate_weighted_similarity(self, angles1, angles2, confidences):
        """Calculate weighted similarity between two sets of angles"""
        if len(angles1) != len(angles2):
            return 0.0
            
        total_weight = 0
        weighted_similarity = 0
        
        for i, ((_, base_weight), conf) in enumerate(zip(self.angle_joints, confidences)):
            # Skip if confidence is too low
            if conf <= self.confidence_threshold:
                continue
                
            # Adjust weight based on confidence
            weight = base_weight * conf
            
            # Calculate similarity for this angle pair
            angle_diff = abs(angles1[i] - angles2[i])
            angle_similarity = max(0, 1 - (angle_diff / 180.0))
            
            # Add weighted similarity
            weighted_similarity += angle_similarity * weight
            total_weight += weight
        
        return weighted_similarity / total_weight if total_weight > 0 else 0.0
    
    def compute_similarity(self, pose1, pose2):
        """
        Compute similarity between two poses using multiple metrics
        """
        # Check if all keypoints are present and have sufficient confidence
        if (pose1 is None or pose2 is None or 
            len(pose1) != 17 or len(pose2) != 17 or
            not all(kp[2] > self.confidence_threshold for kp in pose1) or
            not all(kp[2] > self.confidence_threshold for kp in pose2)):
            return 0.0
            
        # Normalize keypoints
        norm_pose1 = self._normalize_keypoints(pose1)
        norm_pose2 = self._normalize_keypoints(pose2)
        
        # Calculate angles and confidences
        angles1, confidences1 = self._get_joint_angles(norm_pose1)
        angles2, confidences2 = self._get_joint_angles(norm_pose2)
        
        # Use minimum confidence between poses
        confidences = np.minimum(confidences1, confidences2)
        
        # Calculate angle-based similarity
        angle_similarity = self._calculate_weighted_similarity(angles1, angles2, confidences)
        
        # Calculate distance-based similarity
        distance_similarity = self._calculate_distance_similarity(norm_pose1, norm_pose2)
        
        # Combine similarities (give more weight to angle similarity)
        # Give more weight to distance similarity for easier matching
        final_similarity = 0.4 * angle_similarity + 0.6 * distance_similarity
        
        return float(final_similarity)
    
    def is_pose_matched(self, detected_pose, reference_pose):
        """
        Check if detected pose matches reference pose
        Returns:
            tuple: (bool, float) - (is_matched, similarity_score)
        """
        similarity = self.compute_similarity(detected_pose, reference_pose)
        return similarity >= self.similarity_threshold, similarity
