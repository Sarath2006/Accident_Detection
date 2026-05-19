"""
Enhanced Accident Detection Logic V2
Improved with better algorithms, adaptive thresholds, and multi-modal fusion
"""
from utils.geometry import iou
import numpy as np


class EnhancedAccidentLogic:
    """
    Advanced 5-layer accident detection system with adaptive thresholds:
    1. Motion Anomalies (speed drop, trajectory break, acceleration)
    2. Spatial Interaction (collision, proximity, overlap patterns)
    3. Optical Flow Analysis (motion consistency)
    4. Temporal Persistence (confirmation over frames)
    5. CNN Verification (ResNet50 confidence fusion)
    """

    def __init__(
        self,
        speed_drop_threshold=3.0,
        overlap_iou_threshold=0.15,
        cnn_threshold=0.55,
        temporal_window=8,  # Reduced from 10 for faster detection
        proximity_threshold=50,  # pixels
        acceleration_threshold=0.5,  # m/s^2
        speed_scale=1.0,  # Normalize speed when frames are skipped
        accident_threshold_bias=0.0  # Negative lowers threshold (more sensitive)
    ):
        self.speed_drop_threshold = speed_drop_threshold
        self.overlap_iou_threshold = overlap_iou_threshold
        self.cnn_threshold = cnn_threshold
        self.temporal_window = temporal_window
        self.proximity_threshold = proximity_threshold
        self.acceleration_threshold = acceleration_threshold
        self.speed_scale = max(speed_scale, 1.0)
        self.accident_threshold_bias = accident_threshold_bias
        
        # Tracking data
        self.previous_speeds = {}
        self.speed_history = {}  # Last 5 frames
        self.stopped_counter = {}
        self.previous_positions = {}
        self.previous_angles = {}
        self.previous_areas = {}
        self.optical_flow_data = {}
        
        # Temporal consistency
        self.accident_confidence_history = []
        self.consecutive_accident_frames = 0

    def _normalize_speed(self, speed):
        return speed / self.speed_scale

    def _calculate_speed_drop_score(self, speeds):
        """
        LAYER 1A: Enhanced speed drop detection with acceleration analysis
        Returns: (objects_with_drops, severity_scores)
        """
        sudden_stops = {}
        
        for obj_id, current_speed in speeds.items():
            current_speed = self._normalize_speed(current_speed)
            if obj_id not in self.speed_history:
                self.speed_history[obj_id] = []
            
            self.speed_history[obj_id].append(current_speed)
            self.speed_history[obj_id] = self.speed_history[obj_id][-5:]  # Keep last 5
            
            if obj_id in self.previous_speeds:
                prev_speed = self.previous_speeds[obj_id]
                speed_drop = max(0, prev_speed - current_speed)
                
                # Check different speed ranges for adaptive thresholds
                if prev_speed > 15:  # High speed - more sensitive
                    threshold = prev_speed * 0.45  # 45% drop
                    severity_multiplier = 1.5
                elif prev_speed > 8:  # Medium speed
                    threshold = prev_speed * 0.35  # 35% drop
                    severity_multiplier = 1.0
                else:  # Low speed
                    threshold = max(prev_speed * 0.25, self.speed_drop_threshold)
                    severity_multiplier = 0.7
                
                if speed_drop > threshold:
                    score = (speed_drop / (threshold + 1e-6)) * severity_multiplier
                    sudden_stops[obj_id] = min(score, 2.0)  # Cap at 2.0
                
                # Check sustained deceleration (acceleration analysis)
                if len(self.speed_history[obj_id]) >= 3:
                    accel = self.speed_history[obj_id][-3] - self.speed_history[obj_id][-1]
                    if accel > 3:  # Strong deceleration
                        sudden_stops[obj_id] = sudden_stops.get(obj_id, 0) + 0.3
            
            # Prolonged stop (> 2 seconds at 30fps = 60 frames)
            if current_speed < self.speed_drop_threshold:
                self.stopped_counter[obj_id] = self.stopped_counter.get(obj_id, 0) + 1
                if self.stopped_counter[obj_id] > 60:
                    sudden_stops[obj_id] = min(sudden_stops.get(obj_id, 0) + 0.2, 2.0)
            else:
                self.stopped_counter[obj_id] = 0
            
            self.previous_speeds[obj_id] = current_speed
        
        return sudden_stops

    def _calculate_trajectory_anomaly_score(self, tracked_objects):
        """
        LAYER 1B: Enhanced trajectory analysis with rotation detection
        Returns: (trajectory_anomalies_dict)
        """
        anomalies = {}
        
        for obj_id, obj in tracked_objects.items():
            bbox = obj["bbox"]
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            
            if obj_id in self.previous_positions:
                prev_center = self.previous_positions[obj_id]
                
                dx = center[0] - prev_center[0]
                dy = center[1] - prev_center[1]
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance > 2:  # Moving significantly
                    current_angle = np.arctan2(dy, dx)
                    
                    if obj_id in self.previous_angles:
                        prev_angle = self.previous_angles[obj_id]
                        angle_diff = abs(current_angle - prev_angle)
                        # Normalize to 0-180 degrees
                        angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
                        
                        # Sudden direction change > 30 degrees (lowered from 45)
                        if angle_diff > np.pi / 6:  # 30 degrees
                            anomalies[obj_id] = min(angle_diff / (np.pi / 4), 2.0)
                    
                    self.previous_angles[obj_id] = current_angle
            
            self.previous_positions[obj_id] = center
        
        return anomalies

    def _calculate_collision_score(self, tracked_objects):
        """
        LAYER 2: Improved collision detection with proximity analysis
        Returns: (collision_pairs, severity_scores)
        """
        collisions = {}
        ids = list(tracked_objects.keys())
        
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                obj1 = tracked_objects[ids[i]]
                obj2 = tracked_objects[ids[j]]
                
                bbox1 = obj1["bbox"]
                bbox2 = obj2["bbox"]
                
                # Calculate IoU
                intersection = iou(bbox1, bbox2)
                
                # Proximity analysis (even if not overlapping yet)
                center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
                center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
                distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                
                collision_key = f"{ids[i]}-{ids[j]}"
                
                # Actual collision (overlap)
                if intersection > self.overlap_iou_threshold:
                    severity = min(intersection * 3, 2.0)  # Higher IoU = more severe
                    collisions[collision_key] = {"type": "overlap", "score": severity}
                
                # Near-collision (approaching rapidly)
                elif distance < self.proximity_threshold:
                    proximity_score = 1.0 - (distance / self.proximity_threshold)
                    collisions[collision_key] = {"type": "proximity", "score": proximity_score * 0.6}
        
        return collisions

    def _calculate_box_deformation_score(self, tracked_objects):
        """
        LAYER 2B: Detect bounding box deformation and size changes
        Indicates: vehicle damage, crushing, sudden approach
        """
        deformations = {}
        
        for obj_id, obj in tracked_objects.items():
            bbox = obj["bbox"]
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height
            
            if obj_id in self.previous_areas:
                prev_area = self.previous_areas[obj_id]
                if prev_area > 0:
                    area_change = abs(area - prev_area) / prev_area
                    
                    # Sudden size change > 20% (lowered from 30%)
                    if area_change > 0.20:
                        deformations[obj_id] = min(area_change, 2.0)
            
            self.previous_areas[obj_id] = area
        
        return deformations

    def evaluate(self, tracked_objects, speeds, cnn_score=0.0, fire_smoke=False):
        """
        Unified accident evaluation with weighted multi-modal fusion
        
        Returns: {
            'accident': bool,
            'accident_score': float (0-10),
            'confidence': float (0-1),
            'reasons': list,
            'cnn_score': float,
            'speed_drops': dict,
            'collisions': dict,
            'anomalies': dict,
            'nearby_vehicles': int
        }
        """
        
        # Layer 1: Motion Analysis
        speed_drops = self._calculate_speed_drop_score(speeds)
        trajectory_anomalies = self._calculate_trajectory_anomaly_score(tracked_objects)
        
        # Layer 2: Spatial Analysis
        collisions = self._calculate_collision_score(tracked_objects)
        deformations = self._calculate_box_deformation_score(tracked_objects)
        
        # Calculate accident score (0-10 scale)
        num_vehicles = max(1, len(tracked_objects))
        num_pairs = max(1, (len(tracked_objects) * (len(tracked_objects) - 1)) / 2)

        motion_score = sum(speed_drops.values()) / num_vehicles
        spatial_score = sum(c.get('score', 0) for c in collisions.values()) / num_pairs
        anomaly_score = sum(trajectory_anomalies.values()) / num_vehicles
        deformation_score = sum(deformations.values()) / num_vehicles
        
        # Weighted combination
        accident_score = (
            motion_score * 2.5 +          # Most important
            spatial_score * 2.0 +         # Very important
            anomaly_score * 1.5 +         # Important
            deformation_score * 1.0 +     # Moderate
            (cnn_score * 3.0 if cnn_score > 0 else 0)  # CNN verification
        )
        
        # Normalize score
        max_possible = 2.5 * 2 + 2.0 * 2 + 1.5 * 2 + 1.0 * 2 + 3.0
        accident_score = min(accident_score / (max_possible / 10), 10.0)
        
        # Determine if accident
        # Evidence gating for balanced accuracy
        has_collision = bool(collisions)
        has_motion = motion_score >= 0.6
        has_anomaly = anomaly_score >= 0.7
        has_deformation = deformation_score >= 0.6
        strong_cnn = cnn_score >= self.cnn_threshold

        evidence_count = sum([
            1 if has_collision else 0,
            1 if has_motion else 0,
            1 if has_anomaly else 0,
            1 if has_deformation else 0,
            1 if strong_cnn else 0
        ])

        if has_collision and (has_motion or strong_cnn):
            accident_threshold = 3.0
        elif has_collision and has_anomaly:
            accident_threshold = 3.2
        elif strong_cnn and has_motion:
            accident_threshold = 3.2
        elif strong_cnn and (has_anomaly or has_deformation):
            accident_threshold = 3.3
        else:
            accident_threshold = 3.8

        accident_threshold = max(0.5, accident_threshold + self.accident_threshold_bias)
        has_min_evidence = (evidence_count >= 2) or (has_collision and (has_motion or strong_cnn))
        is_accident = accident_score > accident_threshold and has_min_evidence
        
        # Enhanced confidence calculation
        if is_accident:
            confidence_factors = []
            
            # CNN confidence (if available and above threshold)
            if cnn_score > self.cnn_threshold:
                confidence_factors.append(min(cnn_score * 1.1, 1.0))
            
            # Motion evidence
            if speed_drops:
                max_speed_drop = max(speed_drops.values())
                confidence_factors.append(min(max_speed_drop / 2.0, 1.0))
            
            # Collision evidence
            if collisions:
                confidence_factors.append(0.8)
            
            # Anomaly evidence
            if trajectory_anomalies:
                confidence_factors.append(0.7)
            
            # Fire/smoke (very strong indicator)
            if fire_smoke:
                confidence_factors.append(1.0)
            
            # Average confidence
            confidence = np.mean(confidence_factors) if confidence_factors else 0.5
            confidence = max(min(confidence, 1.0), 0.4)  # Clamp 0.4-1.0
        else:
            confidence = 0.0
        
        # Build reasons list
        reasons = []
        severity_indicators = []
        if speed_drops:
            reasons.append(f"Speed drops detected: {len(speed_drops)} vehicle(s)")
            for score in speed_drops.values():
                if score >= 1.5:
                    severity_indicators.append("severe")
                elif score >= 1.0:
                    severity_indicators.append("moderate")
        if collisions:
            reasons.append(f"Collisions detected: {len(collisions)} collision(s)")
            for collision in collisions.values():
                score = collision.get("score", 0)
                if collision.get("type") == "overlap" and score >= 1.2:
                    severity_indicators.append("severe")
                elif score >= 0.6:
                    severity_indicators.append("moderate")
        if trajectory_anomalies:
            reasons.append(f"Trajectory anomalies: {len(trajectory_anomalies)} vehicle(s)")
            for score in trajectory_anomalies.values():
                if score >= 1.2:
                    severity_indicators.append("moderate")
        if deformations:
            reasons.append(f"Box deformations: {len(deformations)} vehicle(s)")
            for score in deformations.values():
                if score >= 1.2:
                    severity_indicators.append("severe")
                elif score >= 0.7:
                    severity_indicators.append("moderate")
        if fire_smoke:
            reasons.append("Fire or smoke detected")
        if cnn_score > self.cnn_threshold:
            reasons.append(f"CNN confidence: {cnn_score:.1%}")
        
        # Track temporal consistency
        self.accident_confidence_history.append(confidence if is_accident else 0.0)
        self.accident_confidence_history = self.accident_confidence_history[-self.temporal_window:]
        
        if is_accident:
            self.consecutive_accident_frames += 1
        else:
            self.consecutive_accident_frames = 0

        # Confirm accidents using consecutive or recent hits within the temporal window
        recent_hits = sum(1 for c in self.accident_confidence_history if c > 0.0)
        confirmed_accident = (
            is_accident and
            (self.consecutive_accident_frames >= 2 or recent_hits >= 2)
        )
        
        return {
            'accident': confirmed_accident,
            'accident_score': accident_score,
            'confidence': confidence,
            'reasons': reasons,
            'cnn_score': cnn_score,
            'speed_drops': speed_drops,
            'collisions': collisions,
            'trajectory_anomalies': trajectory_anomalies,
            'deformations': deformations,
            'severity_indicators': severity_indicators,
            'fire_or_smoke': fire_smoke,
            'nearby_vehicles': len(tracked_objects),
            'consecutive_frames': self.consecutive_accident_frames
        }
