"""
Sequence Analyzer for Accident Timeline Reconstruction
Chronological analysis of events leading to accident
"""
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class EventType(Enum):
    """Types of events in accident sequence"""
    APPROACH = 'approach'  # Vehicles approaching
    DECELERATION = 'deceleration'  # Braking
    ACCELERATION = 'acceleration'  # Speeding
    LANE_CHANGE = 'lane_change'  # Lane change
    COLLISION = 'collision'  # Impact occurs
    SEPARATION = 'separation'  # Vehicles separating after impact
    POST_IMPACT = 'post_impact'  # Sliding/skidding after impact


@dataclass
class TimelineEvent:
    """Single event in accident timeline"""
    timestamp_ms: float
    frame_number: int
    event_type: EventType
    description: str
    details: Dict
    confidence: float = 0.7


class SequenceAnalyzer:
    """
    Analyzes and reconstructs chronological sequence of accident events
    """
    
    def __init__(self):
        """Initialize sequence analyzer"""
        self.speed_threshold_deceleration = 2.0  # m/s² for deceleration detection
        self.speed_threshold_acceleration = 2.0  # m/s² for acceleration detection
    
    def reconstruct_sequence(self, accident_scene) -> List[Dict]:
        """
        Reconstruct full timeline of accident sequence
        
        Args:
            accident_scene: AccidentScene object with forensic data
        
        Returns:
            List of timeline events in chronological order
        """
        timeline = []
        
        # Analyze speeds for behavioral patterns
        speeds_data = self._analyze_speed_patterns(accident_scene.speeds)
        
        # Detect approach phase
        timeline.extend(self._detect_approach_phase(accident_scene, speeds_data))
        
        # Detect critical maneuvers
        timeline.extend(self._detect_maneuvers(accident_scene, speeds_data))
        
        # Detect collision moment
        collision_events = self._detect_collision_phase(accident_scene)
        timeline.extend(collision_events)
        
        # Detect post-impact phase
        timeline.extend(self._detect_post_impact_phase(accident_scene, speeds_data))
        
        # Sort by frame number
        timeline = sorted(timeline, key=lambda x: x['frame_number'])
        
        # Convert to detailed timeline
        detailed_timeline = self._convert_to_detailed_timeline(timeline, accident_scene)
        
        return detailed_timeline
    
    def _analyze_speed_patterns(self, speeds: Dict) -> Dict:
        """
        Analyze speed patterns throughout video
        
        Args:
            speeds: Dictionary of speeds per frame
        
        Returns:
            Dictionary of speed pattern analysis
        """
        patterns = {
            'constant_speed': [],
            'accelerating': [],
            'decelerating': [],
            'rapid_changes': []
        }
        
        speed_list = sorted(speeds.items())
        
        for i in range(1, len(speed_list)):
            frame1, speeds1 = speed_list[i-1]
            frame2, speeds2 = speed_list[i]
            
            if speeds1 and speeds2:
                for vehicle_id in speeds1.keys():
                    if vehicle_id in speeds2:
                        v1 = speeds1[vehicle_id]
                        v2 = speeds2[vehicle_id]
                        
                        # Calculate acceleration (simple finite difference)
                        acceleration = v2 - v1
                        
                        if abs(acceleration) < 0.5:
                            patterns['constant_speed'].append((frame2, vehicle_id, v2))
                        elif acceleration > self.speed_threshold_acceleration:
                            patterns['accelerating'].append((frame2, vehicle_id, v2))
                        elif acceleration < -self.speed_threshold_deceleration:
                            patterns['decelerating'].append((frame2, vehicle_id, v2))
                        
                        if abs(acceleration) > 5.0:
                            patterns['rapid_changes'].append((frame2, vehicle_id, acceleration))
        
        return patterns
    
    def _detect_approach_phase(self, accident_scene, speeds_data: Dict) -> List[Dict]:
        """
        Detect approach phase before collision
        
        Args:
            accident_scene: AccidentScene object
            speeds_data: Speed pattern analysis
        
        Returns:
            List of approach phase events
        """
        events = []
        
        # Find first frame with both vehicles
        start_frame = None
        for frame_num in sorted(accident_scene.detected_objects.keys()):
            if len(accident_scene.detected_objects[frame_num]) >= 2:
                start_frame = frame_num
                break
        
        if start_frame and start_frame < min(accident_scene.accident_frames or [999999]):
            # Get speeds in approach phase
            approach_frames = list(range(start_frame, start_frame + 30, 5))
            
            for frame_num in approach_frames:
                if frame_num in accident_scene.speeds:
                    speeds_frame = accident_scene.speeds[frame_num]
                    
                    events.append({
                        'frame_number': frame_num,
                        'event_type': EventType.APPROACH.value,
                        'description': 'Vehicles in approach - moving toward collision point',
                        'details': {
                            'vehicle_speeds': speeds_frame,
                            'status': 'pre_collision'
                        },
                        'confidence': 0.95
                    })
        
        return events
    
    def _detect_maneuvers(self, accident_scene, speeds_data: Dict) -> List[Dict]:
        """
        Detect critical maneuvers (braking, acceleration, lane changes)
        
        Args:
            accident_scene: AccidentScene object
            speeds_data: Speed pattern analysis
        
        Returns:
            List of maneuver events
        """
        events = []
        
        # Detect deceleration events
        for frame_num, vehicle_id, speed in speeds_data['decelerating']:
            if frame_num not in (accident_scene.accident_frames or []):
                events.append({
                    'frame_number': frame_num,
                    'event_type': EventType.DECELERATION.value,
                    'description': f'Vehicle {vehicle_id} decelerating/braking',
                    'details': {
                        'vehicle_id': vehicle_id,
                        'speed_ms': float(speed)
                    },
                    'confidence': 0.8
                })
        
        # Detect acceleration events
        for frame_num, vehicle_id, speed in speeds_data['accelerating']:
            if frame_num not in (accident_scene.accident_frames or []):
                events.append({
                    'frame_number': frame_num,
                    'event_type': EventType.ACCELERATION.value,
                    'description': f'Vehicle {vehicle_id} accelerating',
                    'details': {
                        'vehicle_id': vehicle_id,
                        'speed_ms': float(speed)
                    },
                    'confidence': 0.8
                })
        
        return events
    
    def _detect_collision_phase(self, accident_scene) -> List[Dict]:
        """
        Detect collision impact moment
        
        Args:
            accident_scene: AccidentScene object
        
        Returns:
            List of collision phase events
        """
        events = []
        
        for frame_num in accident_scene.accident_frames:
            if frame_num in accident_scene.detected_objects:
                objects = accident_scene.detected_objects[frame_num]
                speeds_frame = accident_scene.speeds.get(frame_num, {})
                
                events.append({
                    'frame_number': frame_num,
                    'event_type': EventType.COLLISION.value,
                    'description': 'Collision impact detected - vehicles collide',
                    'details': {
                        'num_vehicles': len(objects),
                        'speeds': speeds_frame,
                        'impact_angles': accident_scene.impact_angles.get(frame_num, {})
                    },
                    'confidence': 0.99
                })
        
        return events
    
    def _detect_post_impact_phase(self, accident_scene, speeds_data: Dict) -> List[Dict]:
        """
        Detect post-impact phase (sliding, separation)
        
        Args:
            accident_scene: AccidentScene object
            speeds_data: Speed pattern analysis
        
        Returns:
            List of post-impact phase events
        """
        events = []
        
        first_accident_frame = min(accident_scene.accident_frames) if accident_scene.accident_frames else 999999
        
        # Find post-impact frames (after first collision)
        post_impact_frames = [f for f in sorted(accident_scene.detected_objects.keys())
                             if f > first_accident_frame]
        
        for frame_num in post_impact_frames[:10]:  # Look at first 10 frames after
            if frame_num in accident_scene.speeds:
                speeds_frame = accident_scene.speeds[frame_num]
                objects = accident_scene.detected_objects.get(frame_num, {})
                
                # Check if vehicles are separating
                separation = accident_scene.physics_analysis.get('vehicle_separation', {}).get(f'frame_{frame_num}', {})
                
                event_type = EventType.POST_IMPACT.value
                if separation.get('separation_distance_pixels', 0) > 50:
                    event_type = EventType.SEPARATION.value
                
                events.append({
                    'frame_number': frame_num,
                    'event_type': event_type,
                    'description': 'Post-impact phase - vehicles sliding/separating',
                    'details': {
                        'vehicle_speeds': speeds_frame,
                        'separation_distance': separation.get('separation_distance_pixels', 0),
                        'status': 'post_collision'
                    },
                    'confidence': 0.85
                })
        
        return events
    
    def _convert_to_detailed_timeline(
        self,
        timeline: List[Dict],
        accident_scene
    ) -> List[Dict]:
        """
        Convert timeline events to detailed report format
        
        Args:
            timeline: List of timeline events
            accident_scene: AccidentScene object
        
        Returns:
            List of detailed timeline events
        """
        detailed_timeline = []
        
        fps = getattr(accident_scene, "fps", 30)
        
        for event in timeline:
            frame_num = event['frame_number']
            timestamp_seconds = frame_num / fps
            
            detailed_event = {
                'timestamp_s': timestamp_seconds,
                'timestamp_ms': int(timestamp_seconds * 1000),
                'frame_number': frame_num,
                'event_type': event['event_type'],
                'description': event['description'],
                'details': event['details'],
                'confidence': event['confidence'],
                'phase': self._determine_phase(frame_num, accident_scene.accident_frames)
            }
            
            detailed_timeline.append(detailed_event)
        
        return detailed_timeline
    
    def _determine_phase(
        self,
        frame_num: int,
        accident_frames: List[int]
    ) -> str:
        """
        Determine accident phase (pre, impact, post)
        
        Args:
            frame_num: Frame number
            accident_frames: List of accident frame numbers
        
        Returns:
            Phase name
        """
        if not accident_frames:
            return 'pre_collision'
        
        min_accident = min(accident_frames)
        max_accident = max(accident_frames)
        
        if frame_num < min_accident:
            return 'pre_collision'
        elif min_accident <= frame_num <= max_accident:
            return 'collision'
        else:
            return 'post_collision'
