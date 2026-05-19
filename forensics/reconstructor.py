"""
Main Forensic Analysis Reconstructor
Orchestrates all forensic modules for court-admissible accident reconstruction
"""
import torch
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import os

from detection.resnet50_classifier import ResNet50AccidentDetector
from detection.vehicle_detector import VehicleDetector
from tracking.tracker import ObjectTracker
from tracking.speed_estimator import SpeedEstimator

from .physics_engine import PhysicsEngine
from .fault_analyzer import FaultAnalyzer
from .sequence_analyzer import SequenceAnalyzer
from .angle_calculator import AngleCalculator
from .report_generator import ForensicReportGenerator
from .visualization_3d import Visualization3D
from .contributing_factors import ContributingFactorsAnalyzer


@dataclass
class AccidentScene:
    """Represents a complete accident scene with all forensic data"""
    video_path: str
    frame_count: int
    accident_frames: List[int]  # Frame numbers where accidents detected
    detected_objects: Dict  # Tracked vehicle/object data
    speeds: Dict  # Speed estimates
    impact_angles: Dict  # Impact angles between objects
    severity_scores: Dict  # Multi-class severity predictions
    physics_analysis: Dict  # Force and energy calculations
    fault_determination: Dict  # AI fault determination
    environmental_factors: Dict  # Weather, road conditions, etc.
    timeline: List[Dict]  # Chronological sequence of events
    fps: int = 30
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class ForensicReconstructor:
    """
    Main orchestrator for forensic accident reconstruction
    Integrates all analysis modules for court-admissible reports
    """
    
    def __init__(
        self,
        accident_model_path: Optional[str] = None,
        vehicle_model_path: Optional[str] = None,
        device: str = 'cuda'
    ):
        """
        Initialize forensic reconstructor
        
        Args:
            accident_model_path: Path to ResNet50 accident detector model
            vehicle_model_path: Path to YOLO vehicle detector model
            device: 'cuda' or 'cpu'
        """
        self.device = device
        
        # Initialize detection models
        self.accident_detector = ResNet50AccidentDetector(
            num_severity_classes=4,
            pretrained=True
        ).to(device)
        
        if accident_model_path and os.path.exists(accident_model_path):
            self.accident_detector.load_state_dict(
                torch.load(accident_model_path, map_location=device)
            )
        
        self.vehicle_detector = VehicleDetector(
            model_path=vehicle_model_path or 'models/yolov8n.pt',
            conf_threshold=0.5,
            iou_threshold=0.45
        )
        
        # Initialize tracking
        self.tracker = ObjectTracker()
        self.speed_estimator = SpeedEstimator()
        
        # Initialize analysis modules
        self.physics_engine = PhysicsEngine()
        self.fault_analyzer = FaultAnalyzer()
        self.sequence_analyzer = SequenceAnalyzer()
        self.angle_calculator = AngleCalculator()
        self.contributing_factors = ContributingFactorsAnalyzer()
        self.visualization_3d = Visualization3D()
        self.report_generator = ForensicReportGenerator()
    
    def analyze_video(
        self,
        video_path: str,
        output_dir: str = 'forensics_output'
    ) -> Tuple[AccidentScene, str]:
        """
        Complete forensic analysis of accident video
        
        Args:
            video_path: Path to accident video file
            output_dir: Directory to save forensic outputs
        
        Returns:
            Tuple of (AccidentScene, report_path)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"FORENSIC ACCIDENT RECONSTRUCTION ANALYSIS")
        print(f"Video: {os.path.basename(video_path)}")
        print(f"{'='*70}")
        
        # Initialize scene
        accident_scene = AccidentScene(
            video_path=video_path,
            frame_count=0,
            accident_frames=[],
            detected_objects={},
            speeds={},
            impact_angles={},
            severity_scores={},
            physics_analysis={},
            fault_determination={},
            environmental_factors={},
            timeline=[]
        )
        
        # Step 1: Detect and track vehicles
        print("\n[1/5] Vehicle Detection & Tracking...")
        detected_objects, speeds = self._detect_and_track(video_path)
        accident_scene.detected_objects = detected_objects
        accident_scene.speeds = speeds
        
        # Step 2: Calculate impact angles and geometry
        print("[2/5] Geometric Analysis...")
        impact_angles = self._calculate_impact_angles(detected_objects)
        accident_scene.impact_angles = impact_angles
        
        # Step 3: Physics-based force calculations
        print("[3/5] Physics Analysis...")
        physics_analysis = self.physics_engine.analyze(
            detected_objects=detected_objects,
            speeds=speeds,
            impact_angles=impact_angles
        )
        accident_scene.physics_analysis = physics_analysis
        
        # Step 4: Fault determination using AI
        print("[4/5] Fault Determination...")
        fault_determination = self.fault_analyzer.determine_fault(
            accident_scene=accident_scene,
            physics_analysis=physics_analysis
        )
        accident_scene.fault_determination = fault_determination
        
        # Step 5: Timeline and sequence reconstruction
        print("[5/5] Sequence Reconstruction...")
        timeline = self.sequence_analyzer.reconstruct_sequence(accident_scene)
        accident_scene.timeline = timeline
        
        # Generate report
        print("\nGenerating Court-Admissible Report...")
        report_path = self.report_generator.generate_report(
            accident_scene=accident_scene,
            output_dir=output_dir
        )
        
        print(f"\n{'='*70}")
        print(f"✓ Forensic analysis complete")
        print(f"  Report: {report_path}")
        print(f"{'='*70}\n")
        
        return accident_scene, report_path
    
    def _detect_and_track(self, video_path: str) -> Tuple[Dict, Dict]:
        """
        Detect vehicles and estimate speeds throughout video
        
        Args:
            video_path: Path to video file
        
        Returns:
            Tuple of (detected_objects, speeds)
        """
        cap = cv2.VideoCapture(video_path)
        detected_objects = {}
        speeds = {}
        frame_num = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            
            # Detect vehicles
            vehicles = self.vehicle_detector.detect(frame)
            
            # Track objects
            tracked = self.tracker.update(vehicles)
            
            # Estimate speeds
            frame_speeds = self.speed_estimator.compute(tracked)
            
            # Store results
            detected_objects[frame_num] = tracked
            speeds[frame_num] = frame_speeds
        
        cap.release()
        return detected_objects, speeds
    
    def _calculate_impact_angles(self, detected_objects: Dict) -> Dict:
        """
        Calculate impact angles between objects
        
        Args:
            detected_objects: Dictionary of tracked objects per frame
        
        Returns:
            Dictionary of impact angles and collision geometry
        """
        impact_angles = {}
        
        for frame_num, objects in detected_objects.items():
            if len(objects) >= 2:
                # Calculate angles between each pair of objects
                frame_angles = self.angle_calculator.calculate_angles(objects)
                impact_angles[frame_num] = frame_angles
        
        return impact_angles
    
    def batch_analyze(
        self,
        video_list: List[str],
        output_dir: str = 'forensics_output'
    ) -> List[Tuple[AccidentScene, str]]:
        """
        Analyze multiple accident videos
        
        Args:
            video_list: List of video file paths
            output_dir: Base directory for outputs
        
        Returns:
            List of (AccidentScene, report_path) tuples
        """
        results = []
        
        for i, video_path in enumerate(video_list, 1):
            print(f"\n[{i}/{len(video_list)}] Processing {os.path.basename(video_path)}")
            
            try:
                scene, report = self.analyze_video(
                    video_path=video_path,
                    output_dir=os.path.join(output_dir, f"case_{i:03d}")
                )
                results.append((scene, report))
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        return results
    
    def generate_visualization_3d(
        self,
        accident_scene: AccidentScene,
        output_path: str
    ):
        """
        Generate 3D collision visualization
        
        Args:
            accident_scene: AccidentScene object
            output_path: Path to save 3D visualization file
        """
        self.visualization_3d.render_collision(
            accident_scene=accident_scene,
            output_path=output_path
        )
    
    def save_scene_json(
        self,
        accident_scene: AccidentScene,
        output_path: str
    ):
        """
        Save accident scene as JSON for archiving
        
        Args:
            accident_scene: AccidentScene object
            output_path: Path to save JSON file
        """
        scene_dict = {
            'video_path': accident_scene.video_path,
            'timestamp': accident_scene.timestamp,
            'frame_count': accident_scene.frame_count,
            'accident_frames': accident_scene.accident_frames,
            'physics_analysis': accident_scene.physics_analysis,
            'fault_determination': accident_scene.fault_determination,
            'timeline': accident_scene.timeline
        }
        
        with open(output_path, 'w') as f:
            json.dump(scene_dict, f, indent=2, default=str)


if __name__ == '__main__':
    # Example usage
    reconstructor = ForensicReconstructor()
    
    # Analyze single video
    video_path = "path/to/accident_video.mp4"
    scene, report = reconstructor.analyze_video(video_path)
    
    # Generate 3D visualization
    reconstructor.generate_visualization_3d(
        scene,
        output_path="forensics_output/collision_3d.html"
    )
