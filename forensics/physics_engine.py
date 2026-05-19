"""
Physics Engine for Accident Reconstruction
Calculates forces, energy, and physical metrics from video data
"""
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class VehiclePhysics:
    """Physics properties of a vehicle"""
    mass_kg: float = 1500  # Average car mass in kg
    length_m: float = 4.5  # Average car length in meters
    width_m: float = 1.8   # Average car width in meters
    height_m: float = 1.5  # Average car height in meters
    friction_coefficient: float = 0.7  # Road friction (dry asphalt)
    
    # Deformation zones (meters)
    crumple_zone_front: float = 0.6
    crumple_zone_rear: float = 0.4
    passenger_compartment_strength: float = 0.3  # Reinforcement factor


class PhysicsEngine:
    """
    Advanced physics calculations for accident reconstruction
    Based on engineering principles and NHTSA standards
    """
    
    def __init__(self):
        """Initialize physics engine with vehicle parameters"""
        self.default_physics = VehiclePhysics()
        self.gravity = 9.81  # m/s^2
        self.energy_loss_coefficient = 0.75  # Energy loss in collision
    
    def analyze(
        self,
        detected_objects: Dict,
        speeds: Dict,
        impact_angles: Dict
    ) -> Dict:
        """
        Complete physics analysis of accident
        
        Args:
            detected_objects: Dictionary of tracked objects per frame
            speeds: Dictionary of speed estimates per frame
            impact_angles: Dictionary of impact angles per frame
        
        Returns:
            Dictionary of physics analysis results
        """
        analysis = {
            'collision_velocity': self._calculate_collision_velocity(speeds),
            'impact_forces': self._calculate_impact_forces(speeds),
            'kinetic_energy': self._calculate_kinetic_energy(speeds),
            'deformation_analysis': self._calculate_deformation(detected_objects),
            'momentum_conservation': self._check_momentum_conservation(speeds),
            'crush_severity': self._calculate_crush_severity(detected_objects),
            'vehicle_separation': self._calculate_separation_distance(detected_objects),
            'skid_marks': self._estimate_skid_marks(speeds),
            'coefficient_of_restitution': self._calculate_restitution(speeds)
        }
        
        return analysis

    def _to_velocity_list(self, frame_speeds: Dict) -> list:
        if not isinstance(frame_speeds, dict):
            return []
        velocities = []
        for value in frame_speeds.values():
            try:
                velocities.append(float(value))
            except (TypeError, ValueError):
                continue
        return velocities
    
    def _calculate_collision_velocity(self, speeds: Dict) -> Dict:
        """
        Calculate velocity at impact
        
        Args:
            speeds: Dictionary of speed estimates
        
        Returns:
            Dictionary of collision velocities
        """
        collision_velocity = {}
        
        for frame_num, frame_speeds in speeds.items():
            velocities = self._to_velocity_list(frame_speeds)
            if velocities:
                collision_velocity[f'frame_{frame_num}'] = {
                    'vehicle_1_velocity_ms': float(velocities[0]) if len(velocities) > 0 else 0.0,
                    'vehicle_2_velocity_ms': float(velocities[1]) if len(velocities) > 1 else 0.0,
                    'relative_velocity_ms': float(abs(velocities[0] - velocities[1])) if len(velocities) > 1 else 0.0,
                    'collision_severity': 'MINOR' if len(velocities) < 2 else 'MINOR' if abs(velocities[0] - velocities[1]) < 5 else 'MODERATE' if abs(velocities[0] - velocities[1]) < 15 else 'SEVERE'
                }
        
        return collision_velocity
    
    def _calculate_impact_forces(self, speeds: Dict) -> Dict:
        """
        Calculate forces during impact (F = ma)
        Using estimated deceleration over crumple zones
        
        Args:
            speeds: Dictionary of speed estimates
        
        Returns:
            Dictionary of impact forces in Newtons
        """
        impact_forces = {}
        
        for frame_num, frame_speeds in speeds.items():
            velocities = self._to_velocity_list(frame_speeds)
            if velocities:
                
                # Assume impact over crumple zone distance
                impact_distance = self.default_physics.crumple_zone_front
                
                forces = {}
                for i, v in enumerate(velocities):
                    if v > 0:
                        # v² = u² + 2as, solve for a: a = (v² - u²) / (2s)
                        # where final velocity = 0, initial = v, distance = impact_distance
                        deceleration = (v ** 2) / (2 * impact_distance)
                        force_newtons = self.default_physics.mass_kg * deceleration
                        forces[f'vehicle_{i+1}'] = force_newtons
                
                impact_forces[f'frame_{frame_num}'] = forces
        
        return impact_forces
    
    def _calculate_kinetic_energy(self, speeds: Dict) -> Dict:
        """
        Calculate kinetic energy (KE = 0.5 * m * v²)
        
        Args:
            speeds: Dictionary of speed estimates
        
        Returns:
            Dictionary of kinetic energies in Joules
        """
        kinetic_energy = {}
        
        for frame_num, frame_speeds in speeds.items():
            if frame_speeds:
                energies = {}
                for obj_id, v in frame_speeds.items():
                    # Convert m/s to more realistic speeds
                    energy = 0.5 * self.default_physics.mass_kg * (v ** 2)
                    energies[f'vehicle_{obj_id}'] = energy
                
                kinetic_energy[f'frame_{frame_num}'] = energies
        
        return kinetic_energy
    
    def _calculate_deformation(self, detected_objects: Dict) -> Dict:
        """
        Estimate vehicle deformation from bounding box changes
        
        Args:
            detected_objects: Dictionary of tracked objects
        
        Returns:
            Dictionary of deformation estimates
        """
        deformation = {}
        
        # Calculate bounding box area changes over time
        for frame_num, objects in detected_objects.items():
            frame_deform = {}
            for obj_id, obj in objects.items():
                if 'bbox' in obj:
                    x1, y1, x2, y2 = obj['bbox']
                    bbox_area = (x2 - x1) * (y2 - y1)
                    frame_deform[f'vehicle_{obj_id}'] = {
                        'bbox_area_pixels': bbox_area,
                        'estimated_compression_percentage': 0  # Would need multiple frames for calculation
                    }
            
            if frame_deform:
                deformation[f'frame_{frame_num}'] = frame_deform
        
        return deformation
    
    def _check_momentum_conservation(self, speeds: Dict) -> Dict:
        """
        Verify momentum conservation before/after collision
        
        Args:
            speeds: Dictionary of speed estimates
        
        Returns:
            Dictionary of momentum analysis
        """
        momentum_analysis = {}
        
        for frame_num, frame_speeds in speeds.items():
            velocities = self._to_velocity_list(frame_speeds)
            if len(velocities) >= 2:
                
                # Calculate momentum (p = m * v)
                p_before = self.default_physics.mass_kg * sum(velocities)
                
                # After collision (estimated)
                p_after = self.default_physics.mass_kg * (velocities[0] if velocities else 0)
                
                momentum_analysis[f'frame_{frame_num}'] = {
                    'momentum_before_kgms': float(p_before),
                    'momentum_after_kgms': float(p_after),
                    'momentum_loss_percentage': float(abs(p_after - p_before) / abs(p_before) * 100) if p_before != 0 else 0
                }
        
        return momentum_analysis
    
    def _calculate_crush_severity(self, detected_objects: Dict) -> Dict:
        """
        Estimate crush severity from vehicle damage
        
        Args:
            detected_objects: Dictionary of tracked objects
        
        Returns:
            Dictionary of crush severity ratings
        """
        crush_severity = {}
        
        for frame_num, objects in detected_objects.items():
            severity_scores = {}
            for obj_id, obj in objects.items():
                # Estimate based on bounding box distortion
                if 'bbox' in obj:
                    x1, y1, x2, y2 = obj['bbox']
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Calculate aspect ratio deviation from normal
                    normal_aspect_ratio = self.default_physics.length_m / self.default_physics.width_m
                    pixel_aspect = width / height if height > 0 else normal_aspect_ratio
                    
                    severity = min(100, abs(pixel_aspect - normal_aspect_ratio) * 50)
                    severity_scores[f'vehicle_{obj_id}'] = {
                        'crush_rating': 'LOW' if severity < 30 else 'MODERATE' if severity < 70 else 'SEVERE',
                        'severity_percentage': float(severity)
                    }
            
            if severity_scores:
                crush_severity[f'frame_{frame_num}'] = severity_scores
        
        return crush_severity
    
    def _calculate_separation_distance(self, detected_objects: Dict) -> Dict:
        """
        Calculate separation distance between vehicles after impact
        
        Args:
            detected_objects: Dictionary of tracked objects
        
        Returns:
            Dictionary of separation distances
        """
        separation = {}
        
        for frame_num, objects in detected_objects.items():
            if len(objects) >= 2:
                object_list = list(objects.values())
                
                # Calculate center positions
                centers = []
                for obj in object_list:
                    if 'bbox' in obj:
                        x1, y1, x2, y2 = obj['bbox']
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        centers.append((cx, cy))
                
                if len(centers) >= 2:
                    # Calculate distances between all pairs
                    distances = []
                    for i in range(len(centers)):
                        for j in range(i+1, len(centers)):
                            dist = np.sqrt((centers[i][0] - centers[j][0])**2 + 
                                         (centers[i][1] - centers[j][1])**2)
                            distances.append(dist)
                    
                    separation[f'frame_{frame_num}'] = {
                        'separation_distance_pixels': float(min(distances)) if distances else 0,
                        'average_separation_pixels': float(np.mean(distances)) if distances else 0
                    }
        
        return separation
    
    def _estimate_skid_marks(self, speeds: Dict) -> Dict:
        """
        Estimate skid mark length from speed and friction
        Using formula: d = v² / (2 * μ * g)
        
        Args:
            speeds: Dictionary of speed estimates
        
        Returns:
            Dictionary of estimated skid mark lengths
        """
        skid_marks = {}
        
        for frame_num, frame_speeds in speeds.items():
            if frame_speeds:
                marks = {}
                for obj_id, v in frame_speeds.items():
                    # d = v² / (2 * μ * g)
                    if v > 0:
                        skid_length_m = (v ** 2) / (2 * self.default_physics.friction_coefficient * self.gravity)
                        marks[f'vehicle_{obj_id}'] = {
                            'estimated_skid_length_m': float(skid_length_m),
                            'confidence': 'MEDIUM'  # Would be higher with actual skid mark detection
                        }
                
                if marks:
                    skid_marks[f'frame_{frame_num}'] = marks
        
        return skid_marks
    
    def _calculate_restitution(self, speeds: Dict) -> Dict:
        """
        Calculate coefficient of restitution (e)
        e = relative velocity after / relative velocity before
        
        Args:
            speeds: Dictionary of speed estimates
        
        Returns:
            Dictionary of restitution coefficients
        """
        restitution = {}
        
        for frame_num, frame_speeds in speeds.items():
            velocities = self._to_velocity_list(frame_speeds)
            if len(velocities) >= 2:
                v_before = velocities[0] - velocities[1]
                # After collision, assume vehicles move together
                v_after = 0.1 * v_before  # Typical for car-to-car collision
                
                e = abs(v_after) / abs(v_before) if v_before != 0 else 0
                
                restitution[f'frame_{frame_num}'] = {
                    'coefficient_of_restitution': float(min(1.0, e)),
                    'collision_type': 'INELASTIC' if e < 0.3 else 'ELASTIC' if e > 0.7 else 'PARTIALLY_ELASTIC'
                }
        
        return restitution


if __name__ == '__main__':
    # Test physics engine
    engine = PhysicsEngine()
    
    # Dummy data for testing
    test_speeds = {
        1: {0: 20.0, 1: 10.0},  # Frame 1: vehicle 0 at 20 m/s, vehicle 1 at 10 m/s
        2: {0: 15.0, 1: 15.0},  # Frame 2: both at 15 m/s (converging)
    }
    
    test_objects = {
        1: {0: {'bbox': (100, 100, 200, 200)}, 1: {'bbox': (300, 100, 400, 200)}},
        2: {0: {'bbox': (110, 105, 210, 205)}, 1: {'bbox': (290, 100, 390, 200)}},
    }
    
    test_angles = {
        2: {'collision_angle': 45.0, 'impact_point': (200, 150)}
    }
    
    results = engine.analyze(test_objects, test_speeds, test_angles)
    
    import json
    print(json.dumps(results, indent=2, default=str))
