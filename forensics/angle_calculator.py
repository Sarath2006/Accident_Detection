"""Angle Calculator for Impact Geometry Analysis"""
import numpy as np
from typing import Dict


class AngleCalculator:
    """Calculate impact angles and collision geometry"""
    
    def calculate_angles(self, objects: Dict) -> Dict:
        """Calculate angles between objects"""
        angles = {
            'collision_angle': 0.0,
            'impact_point': (0, 0)
        }
        
        if len(objects) >= 2:
            # Extract positions
            positions = []
            for obj in objects.values():
                if 'bbox' in obj:
                    x1, y1, x2, y2 = obj['bbox']
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    positions.append((cx, cy))
            
            if len(positions) >= 2:
                # Calculate angle between vectors
                dx = positions[1][0] - positions[0][0]
                dy = positions[1][1] - positions[0][1]
                angle = np.degrees(np.arctan2(dy, dx))
                angles['collision_angle'] = float(angle)
                angles['impact_point'] = ((positions[0][0] + positions[1][0])/2,
                                         (positions[0][1] + positions[1][1])/2)
        
        return angles
