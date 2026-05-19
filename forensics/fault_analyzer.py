"""
Fault Analyzer for Accident Determination
Uses AI and forensic evidence to determine liability
"""
import torch
import numpy as np
from typing import Dict, Tuple
from enum import Enum


class FaultLevel(Enum):
    """Fault determination levels"""
    CLEAR_FAULT = 0.9  # > 90% responsibility
    MAJORITY_FAULT = 0.75  # 75-90% responsibility
    COMPARATIVE_FAULT = 0.5  # 50% responsibility (split)
    MINOR_FAULT = 0.25  # 25-50% responsibility
    NO_FAULT = 0.1  # < 10% responsibility


class FaultAnalyzer:
    """
    AI-powered fault determination system
    Analyzes accident physics, speeds, and behavior patterns
    """
    
    def __init__(self):
        """Initialize fault analyzer"""
        self.speed_sensitivity = {
            'significant_speed_advantage': 10,  # km/h
            'minor_speed_difference': 5,  # km/h
            'speed_threshold': 30  # km/h
        }
        
        self.impact_factors = {
            'head_on': 0.8,  # High fault for head-on
            'rear_end': 0.85,  # Highest fault for rear-end
            'side_impact': 0.6,  # Moderate for side
            't_bone': 0.7,  # T-bone collision
        }
    
    def determine_fault(self, accident_scene, physics_analysis: Dict) -> Dict:
        """
        Determine fault based on accident evidence
        
        Args:
            accident_scene: AccidentScene object with all forensic data
            physics_analysis: Dictionary of physics calculations
        
        Returns:
            Dictionary of fault determination with liability percentages
        """
        fault_analysis = {
            'primary_fault_vehicle': None,
            'liability_percentage': {},
            'fault_reasons': {},
            'speed_analysis': {},
            'impact_analysis': {},
            'behavior_analysis': {},
            'confidence_score': 0.0,
            'recommendation': 'INCONCLUSIVE'
        }
        
        # Analyze speeds
        speed_fault = self._analyze_speed_fault(
            physics_analysis.get('collision_velocity', {}),
            accident_scene.speeds
        )
        fault_analysis['speed_analysis'] = speed_fault
        
        # Analyze impact type
        impact_fault = self._analyze_impact_type(
            physics_analysis.get('collision_velocity', {}),
            accident_scene.impact_angles
        )
        fault_analysis['impact_analysis'] = impact_fault
        
        # Analyze behavior patterns
        behavior_fault = self._analyze_behavior(accident_scene)
        fault_analysis['behavior_analysis'] = behavior_fault
        
        # Determine primary fault
        liability = self._calculate_liability(
            speed_fault,
            impact_fault,
            behavior_fault
        )
        
        fault_analysis['liability_percentage'] = liability
        fault_analysis['primary_fault_vehicle'] = max(
            liability.items(),
            key=lambda x: x[1]
        )[0]
        
        # Generate fault reasons
        fault_analysis['fault_reasons'] = self._generate_fault_reasons(
            liability,
            speed_fault,
            impact_fault,
            behavior_fault
        )
        
        # Calculate confidence
        fault_analysis['confidence_score'] = self._calculate_confidence(
            accident_scene,
            physics_analysis
        )
        
        # Make recommendation
        fault_analysis['recommendation'] = self._make_recommendation(liability)
        
        return fault_analysis
    
    def _analyze_speed_fault(self, collision_velocity: Dict, speeds: Dict) -> Dict:
        """
        Analyze fault based on speed differential
        
        Args:
            collision_velocity: Collision velocity data
            speeds: Speed estimates throughout video
        
        Returns:
            Dictionary of speed-based fault factors
        """
        speed_fault = {
            'speed_differential_fault': {},
            'excessive_speed_fault': {},
            'speed_advantage_analysis': {}
        }
        
        for frame, collision_data in collision_velocity.items():
            v1 = collision_data.get('vehicle_1_velocity_ms', 0)
            v2 = collision_data.get('vehicle_2_velocity_ms', 0)
            
            # Convert to km/h
            v1_kmh = v1 * 3.6
            v2_kmh = v2 * 3.6
            
            # Vehicle traveling faster has more responsibility
            speed_diff = abs(v1_kmh - v2_kmh)
            
            speed_fault['speed_differential_fault'][frame] = {
                'vehicle_1_speed_kmh': float(v1_kmh),
                'vehicle_2_speed_kmh': float(v2_kmh),
                'speed_differential_kmh': float(speed_diff),
                'higher_speed_vehicle': 'vehicle_1' if v1 > v2 else 'vehicle_2',
                'fault_factor': self._speed_to_fault_factor(speed_diff)
            }
            
            # Excessive speed analysis
            posted_speed_limit = 50  # Assume 50 km/h urban speed
            if max(v1_kmh, v2_kmh) > posted_speed_limit * 1.2:
                exceeding_vehicle = 'vehicle_1' if v1_kmh > v2_kmh else 'vehicle_2'
                speed_fault['excessive_speed_fault'][frame] = {
                    'vehicle': exceeding_vehicle,
                    'speed_kmh': float(max(v1_kmh, v2_kmh)),
                    'speed_limit_kmh': posted_speed_limit,
                    'overage_percentage': float((max(v1_kmh, v2_kmh) - posted_speed_limit) / posted_speed_limit * 100)
                }
        
        return speed_fault
    
    def _analyze_impact_type(self, collision_velocity: Dict, impact_angles: Dict) -> Dict:
        """
        Determine fault based on collision type
        
        Args:
            collision_velocity: Collision velocity data
            impact_angles: Impact angle data
        
        Returns:
            Dictionary of impact-based fault factors
        """
        impact_fault = {
            'collision_type': 'UNKNOWN',
            'fault_factor': 0.5,
            'details': {}
        }
        
        # Determine collision type from angle
        for frame, angle_data in impact_angles.items():
            angle = angle_data.get('collision_angle', 0)
            
            if abs(angle) < 30:  # Rear-end (0-30 degrees)
                impact_fault['collision_type'] = 'REAR_END'
                impact_fault['fault_factor'] = self.impact_factors['rear_end']
                impact_fault['details'] = {
                    'description': 'Rear-end collision',
                    'fault_rule': 'Vehicle behind typically at fault',
                    'angle_degrees': float(angle)
                }
            elif 150 < abs(angle) < 180:  # Head-on
                impact_fault['collision_type'] = 'HEAD_ON'
                impact_fault['fault_factor'] = self.impact_factors['head_on']
                impact_fault['details'] = {
                    'description': 'Head-on collision',
                    'fault_rule': 'Fault shared based on lane violation',
                    'angle_degrees': float(angle)
                }
            elif 45 < abs(angle) < 135:  # Side impact
                impact_fault['collision_type'] = 'SIDE_IMPACT'
                impact_fault['fault_factor'] = self.impact_factors['side_impact']
                impact_fault['details'] = {
                    'description': 'Side impact / T-bone collision',
                    'fault_rule': 'Right-of-way determines fault',
                    'angle_degrees': float(angle)
                }
        
        return impact_fault
    
    def _analyze_behavior(self, accident_scene) -> Dict:
        """
        Analyze driver behavior patterns
        
        Args:
            accident_scene: AccidentScene object
        
        Returns:
            Dictionary of behavior-based fault factors
        """
        behavior_fault = {
            'sudden_deceleration': 0.0,
            'sudden_acceleration': 0.0,
            'lane_change': 0.0,
            'braking_response': 0.0,
            'overall_behavior_fault': 0.5
        }
        
        # Analyze speed patterns for sudden changes
        if accident_scene.speeds:
            speeds_list = list(accident_scene.speeds.values())
            
            if len(speeds_list) > 1:
                # Check for sudden deceleration
                speed_changes = []
                for i in range(1, len(speeds_list)):
                    if speeds_list[i] and speeds_list[i-1]:
                        change = list(speeds_list[i].values())[0] - list(speeds_list[i-1].values())[0]
                        speed_changes.append(change)
                
                if speed_changes:
                    avg_deceleration = np.mean([c for c in speed_changes if c < 0])
                    if avg_deceleration < -5:  # Significant deceleration
                        behavior_fault['braking_response'] = 0.3
        
        # Calculate overall behavior fault
        behavior_fault['overall_behavior_fault'] = np.mean([
            behavior_fault['sudden_deceleration'],
            behavior_fault['sudden_acceleration'],
            behavior_fault['lane_change'],
            behavior_fault['braking_response']
        ])
        
        return behavior_fault
    
    def _calculate_liability(
        self,
        speed_fault: Dict,
        impact_fault: Dict,
        behavior_fault: Dict
    ) -> Dict:
        """
        Calculate liability percentage for each vehicle
        
        Args:
            speed_fault: Speed-based fault analysis
            impact_fault: Impact type analysis
            behavior_fault: Behavior analysis
        
        Returns:
            Dictionary of liability percentages
        """
        liability = {
            'vehicle_1': 0.0,
            'vehicle_2': 0.0
        }
        
        # Speed-based liability
        if speed_fault.get('speed_differential_fault'):
            speed_data = list(speed_fault['speed_differential_fault'].values())[0]
            if speed_data['higher_speed_vehicle'] == 'vehicle_1':
                liability['vehicle_1'] += speed_data['fault_factor'] * 0.4
                liability['vehicle_2'] += (1 - speed_data['fault_factor']) * 0.4
            else:
                liability['vehicle_2'] += speed_data['fault_factor'] * 0.4
                liability['vehicle_1'] += (1 - speed_data['fault_factor']) * 0.4
        
        # Impact-based liability (rear-end = vehicle behind at fault)
        if impact_fault['collision_type'] == 'REAR_END':
            # Assuming vehicle_2 hit vehicle_1
            liability['vehicle_2'] += impact_fault['fault_factor'] * 0.4
            liability['vehicle_1'] += (1 - impact_fault['fault_factor']) * 0.4
        else:
            # Equal split for other collision types
            liability['vehicle_1'] += 0.2
            liability['vehicle_2'] += 0.2
        
        # Behavior-based liability
        liability['vehicle_1'] += behavior_fault.get('overall_behavior_fault', 0) * 0.2
        liability['vehicle_2'] += (1 - behavior_fault.get('overall_behavior_fault', 0.5)) * 0.2
        
        # Normalize to 100%
        total = liability['vehicle_1'] + liability['vehicle_2']
        if total > 0:
            liability['vehicle_1'] = min(100, (liability['vehicle_1'] / total) * 100)
            liability['vehicle_2'] = min(100, (liability['vehicle_2'] / total) * 100)
        
        return liability
    
    def _speed_to_fault_factor(self, speed_diff_kmh: float) -> float:
        """
        Convert speed differential to fault factor
        
        Args:
            speed_diff_kmh: Speed difference in km/h
        
        Returns:
            Fault factor (0.0 to 1.0)
        """
        if speed_diff_kmh < 5:
            return 0.4  # Minor speed difference
        elif speed_diff_kmh < 15:
            return 0.6  # Moderate speed difference
        elif speed_diff_kmh < 30:
            return 0.8  # Large speed difference
        else:
            return 0.95  # Extreme speed difference
    
    def _generate_fault_reasons(
        self,
        liability: Dict,
        speed_fault: Dict,
        impact_fault: Dict,
        behavior_fault: Dict
    ) -> Dict:
        """
        Generate human-readable fault reasons
        
        Args:
            liability: Liability percentages
            speed_fault: Speed fault data
            impact_fault: Impact fault data
            behavior_fault: Behavior fault data
        
        Returns:
            Dictionary of fault reasons by vehicle
        """
        reasons = {
            'vehicle_1': [],
            'vehicle_2': []
        }
        
        # Add speed reasons
        if speed_fault.get('speed_differential_fault'):
            speed_data = list(speed_fault['speed_differential_fault'].values())[0]
            if speed_data['higher_speed_vehicle'] == 'vehicle_1':
                reasons['vehicle_1'].append(
                    f"Excessive speed: {speed_data['vehicle_1_speed_kmh']:.1f} km/h"
                )
            else:
                reasons['vehicle_2'].append(
                    f"Excessive speed: {speed_data['vehicle_2_speed_kmh']:.1f} km/h"
                )
        
        # Add impact reasons
        if impact_fault['collision_type'] == 'REAR_END':
            reasons['vehicle_2'].append('Rear-end collision - following too closely')
        elif impact_fault['collision_type'] == 'HEAD_ON':
            reasons['vehicle_1'].append('Potential lane violation')
        
        # Add behavior reasons
        if behavior_fault.get('braking_response', 0) > 0.5:
            reasons['vehicle_1'].append('Inadequate braking response')
        
        return reasons
    
    def _calculate_confidence(self, accident_scene, physics_analysis: Dict) -> float:
        """
        Calculate confidence in fault determination
        
        Args:
            accident_scene: AccidentScene object
            physics_analysis: Physics analysis data
        
        Returns:
            Confidence score (0.0 to 1.0)
        """
        confidence_factors = []
        
        # Confidence from number of tracked frames
        num_frames = len(accident_scene.detected_objects)
        confidence_factors.append(min(1.0, num_frames / 30))  # More frames = higher confidence
        
        # Confidence from impact angle data
        num_angle_measurements = len(accident_scene.impact_angles)
        confidence_factors.append(min(1.0, num_angle_measurements / 10))
        
        # Confidence from physics consistency
        if physics_analysis:
            confidence_factors.append(0.8)
        
        # Average confidence
        return float(np.mean(confidence_factors)) if confidence_factors else 0.5
    
    def _make_recommendation(self, liability: Dict) -> str:
        """
        Make recommendation based on liability
        
        Args:
            liability: Liability percentages
        
        Returns:
            Recommendation string
        """
        v1_liability = liability.get('vehicle_1', 0)
        
        if v1_liability > 80:
            return 'VEHICLE_1_CLEARLY_AT_FAULT'
        elif v1_liability > 60:
            return 'VEHICLE_1_MAJORITY_FAULT'
        elif abs(v1_liability - 50) < 10:
            return 'COMPARATIVE_FAULT_50_50'
        elif v1_liability < 40:
            return 'VEHICLE_2_MAJORITY_FAULT'
        else:
            return 'VEHICLE_2_CLEARLY_AT_FAULT'
