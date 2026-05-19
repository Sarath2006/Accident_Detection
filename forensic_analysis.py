"""
Forensic Analysis System
Generates court-admissible forensic reconstruction reports from accident detection data
"""
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from config.settings import Settings
from input.video_reader import VideoReader
from detection.vehicle_detector import VehicleDetector
from detection.resnet50_classifier import ResNet50AccidentDetector
from tracking.tracker import ObjectTracker
from tracking.speed_estimator import SpeedEstimator
from forensics.reconstructor import ForensicReconstructor, AccidentScene
from forensics.report_generator import ForensicReportGenerator
from forensics.physics_engine import PhysicsEngine
from forensics.fault_analyzer import FaultAnalyzer
from forensics.sequence_analyzer import SequenceAnalyzer
from forensics.angle_calculator import AngleCalculator
from forensics.contributing_factors import ContributingFactorsAnalyzer
from analysis.enhanced_accident_logic import EnhancedAccidentLogic
from intelligence.hazard_engine import HazardEngine


class ForensicAnalysisSystem:
    """Complete forensic analysis pipeline"""
    
    def __init__(self, device='cuda', meters_per_pixel=None, calibration_label=None):
        """Initialize forensic analysis system"""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.settings = Settings()
        self.meters_per_pixel = meters_per_pixel
        self.calibration_label = calibration_label
        
        print(f"[FORENSICS] Initializing on device: {self.device}")
        
        # Load models
        self.accident_detector = ResNet50AccidentDetector(
            num_severity_classes=4,
            pretrained=True
        ).to(self.device)
        
        # Load pretrained weights if available
        model_path = 'models/resnet50_accident_best.pth'
        if os.path.exists(model_path):
            try:
                self.accident_detector.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )
                print(f"[FORENSICS] Loaded accident detector from {model_path}")
            except Exception as e:
                print(f"[FORENSICS] Warning: Could not load model weights: {e}")
        
        self.vehicle_detector = VehicleDetector(model_path='models/yolov8n.pt')
        self.tracker = ObjectTracker()
        self.speed_estimator = SpeedEstimator()

        self.accident_logic = EnhancedAccidentLogic(
            speed_drop_threshold=3.0,
            overlap_iou_threshold=0.15,
            cnn_threshold=0.55,
            temporal_window=8,
            proximity_threshold=50,
            speed_scale=self.settings.FRAME_SKIP,
            accident_threshold_bias=self.settings.FORENSIC_ACCIDENT_THRESHOLD_BIAS
        )
        self.hazard_engine = HazardEngine()
        
        # Forensics components
        self.physics_engine = PhysicsEngine()
        self.fault_analyzer = FaultAnalyzer()
        self.sequence_analyzer = SequenceAnalyzer()
        self.angle_calculator = AngleCalculator()
        self.contributing_factors = ContributingFactorsAnalyzer()
        self.report_generator = ForensicReportGenerator()
        
        # Image preprocessing
        from torchvision import transforms
        self.accident_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def analyze_video(self, video_path: str, output_dir: str = 'forensics_output') -> dict:
        """
        Perform complete forensic analysis on a video
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save forensic report
        
        Returns:
            Dictionary with forensic analysis results
        """
        print(f"\n{'='*80}")
        print(f"FORENSIC ANALYSIS: {os.path.basename(video_path)}")
        print(f"{'='*80}\n")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize analysis data structures
        accident_frames = []
        detected_objects_timeline = {}
        speeds_timeline = {}
        impact_angles = {}
        severity_scores = {}
        frame_count = 0
        accident_frame_details = []
        
        # Read video
        video_reader = VideoReader(video_path)
        print(f"[FORENSICS] Processing {video_reader.total_frames} frames...")
        
        fps = video_reader.fps if video_reader.fps > 0 else self.settings.DEFAULT_FPS
        frame_skip = max(self.settings.FRAME_SKIP, 1)
        if self.meters_per_pixel and self.meters_per_pixel > 0:
            speed_scale = self.meters_per_pixel * (fps / frame_skip)
            speed_unit = "m/s"
        else:
            speed_scale = 1.0
            speed_unit = "px/frame"

        with tqdm(total=video_reader.total_frames, desc="Forensic Analysis", unit="frames") as pbar:
            for frame in video_reader:
                frame_count += 1
                pbar.update(1)
                
                # Skip frames for performance
                if frame_count % self.settings.FRAME_SKIP != 0:
                    continue
                
                # Detect vehicles
                vehicles = self.vehicle_detector.detect(frame)
                
                # Classify accident
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = self.accident_transform(frame_rgb).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    accident_pred = self.accident_detector.predict(frame_tensor)
                
                accident_confidence = accident_pred.get('accident_confidence', 0.0)
                if isinstance(accident_confidence, torch.Tensor):
                    accident_confidence = float(accident_confidence.item())
                
                # Track objects
                tracked_objects = self.tracker.update(vehicles)
                speeds_raw = self.speed_estimator.compute(tracked_objects)
                speeds = {
                    obj_id: float(speed) * speed_scale
                    for obj_id, speed in speeds_raw.items()
                }
                
                # Record timeline with dictionary keys
                detected_objects_timeline[frame_count] = tracked_objects.copy() if tracked_objects else {}
                speeds_timeline[frame_count] = speeds.copy() if speeds else {}
                
                # Enhanced accident logic with hazard decision
                accident_event = self.accident_logic.evaluate(
                    tracked_objects=tracked_objects,
                    speeds=speeds_raw,
                    cnn_score=accident_confidence,
                    fire_smoke=False
                )
                hazard = self.hazard_engine.decide(accident_event)

                # Calculate impact angles if collision detected
                if hazard.get("status") == "ACCIDENT":
                    accident_frames.append(frame_count)
                    severity = hazard.get('severity', 'MINOR')
                    severity_scores[frame_count] = {
                        'confidence': hazard.get('confidence', 0.0),
                        'severity': severity
                    }

                    # Record accident frame details
                    accident_frame_details.append({
                        'frame_number': frame_count,
                        'timestamp_s': frame_count / float(fps),
                        'event_type': 'collision',
                        'description': 'Collision impact detected',
                        'details': {
                            'vehicles_count': len(tracked_objects),
                            'speeds': speeds.copy() if speeds else {},
                            'speed_unit': speed_unit
                        },
                        'severity': severity,
                        'confidence': hazard.get('confidence', 0.0)
                    })

                    # Calculate angles between vehicles
                    if len(tracked_objects) >= 2:
                        angles = self.angle_calculator.calculate_angles(
                            tracked_objects
                        )
                        impact_angles[frame_count] = angles
        
        video_reader.release()
        
        no_accident = False
        if not accident_frames:
            print("[FORENSICS] No accidents detected in video. Generating report anyway.")
            no_accident = True
        
        print(f"\n[FORENSICS] {len(accident_frames)} accident frame(s) detected")
        
        # Physics analysis
        print("[FORENSICS] Performing physics analysis...")
        try:
            physics_analysis = self.physics_engine.analyze(
                detected_objects_timeline,
                speeds_timeline,
                impact_angles
            )
        except Exception as e:
            print(f"[FORENSICS] Physics analysis warning: {e}")
            physics_analysis = {'status': 'incomplete', 'error': str(e)}
        
        # Sequence analysis
        print("[FORENSICS] Analyzing accident sequence...")
        
        # Contributing factors analysis
        print("[FORENSICS] Identifying contributing factors...")
        factors = {
            'primary_accident_frame': accident_frames[0] if accident_frames else None,
            'total_accident_frames': len(accident_frames),
            'accident_cluster_start': min(accident_frames) if accident_frames else None,
            'accident_cluster_end': max(accident_frames) if accident_frames else None,
            'peak_severity': max([detail.get('severity', 'MINOR') for detail in accident_frame_details], 
                                default='MINOR'),
            'peak_confidence': max([detail['confidence'] for detail in accident_frame_details], 
                                  default=0.0),
            'speed_unit': speed_unit,
            'meters_per_pixel': self.meters_per_pixel,
            'calibration_label': self.calibration_label
        }
        
        # Fault determination
        print("[FORENSICS] Determining fault liability...")
        accident_scene = AccidentScene(
            video_path=video_path,
            fps=fps,
            frame_count=frame_count,
            accident_frames=accident_frames,
            detected_objects=detected_objects_timeline,
            speeds=speeds_timeline,
            impact_angles=impact_angles,
            severity_scores=severity_scores,
            physics_analysis=physics_analysis,
            fault_determination={},
            environmental_factors={
                'speed_unit': speed_unit,
                'meters_per_pixel': self.meters_per_pixel,
                'calibration_label': self.calibration_label
            },
            timeline=[]
        )
        
        try:
            fault_analysis = self.fault_analyzer.determine_fault(
                accident_scene,
                physics_analysis
            )
        except Exception as e:
            print(f"[FORENSICS] Fault analysis warning: {e}")
            fault_analysis = {'status': 'incomplete', 'error': str(e)}
        
        accident_scene.fault_determination = fault_analysis

        # Build timeline after physics/fault data are available
        timeline = self.sequence_analyzer.reconstruct_sequence(accident_scene)
        if not timeline:
            timeline = accident_frame_details
        accident_scene.timeline = timeline
        
        # Generate report
        print("[FORENSICS] Generating forensic report...")
        try:
            report_path = self.report_generator.generate_report(
                accident_scene,
                output_dir
            )
            print(f"\n[FORENSICS] ✓ Forensic report generated: {report_path}")
        except Exception as e:
            print(f"[FORENSICS] Report generation warning: {e}")
            report_path = None
        
        return {
            'status': 'NO_ACCIDENT' if no_accident else 'SUCCESS',
            'video_path': video_path,
            'accident_frames': accident_frames,
            'report_path': report_path,
            'physics_analysis': physics_analysis,
            'fault_analysis': fault_analysis,
            'timeline': timeline,
            'contributing_factors': factors
        }
    
    def analyze_batch(self, video_paths: list, output_dir: str = 'forensics_output') -> list:
        """
        Analyze multiple videos for forensics
        
        Args:
            video_paths: List of video file paths
            output_dir: Output directory for reports
        
        Returns:
            List of analysis results
        """
        results = []
        
        for video_path in video_paths:
            try:
                result = self.analyze_video(video_path, output_dir)
                results.append(result)
            except Exception as e:
                print(f"[FORENSICS] Error analyzing {video_path}: {e}")
                results.append({
                    'status': 'ERROR',
                    'video_path': video_path,
                    'error': str(e)
                })
        
        return results


def main():
    """Main forensic analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Forensic Accident Analysis")
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to video file or folder with videos"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="forensics_output",
        help="Output directory for forensic reports"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for analysis"
    )
    parser.add_argument(
        "--calib-meters",
        type=float,
        default=None,
        help="Real-world meters for calibration reference"
    )
    parser.add_argument(
        "--calib-pixels",
        type=float,
        default=None,
        help="Pixel length for calibration reference"
    )
    parser.add_argument(
        "--calib-label",
        type=str,
        default=None,
        help="Label for calibration reference (e.g., 'lane width')"
    )
    
    args = parser.parse_args()

    if args.calib_meters is None or args.calib_pixels is None:
        args.calib_meters = args.calib_meters or Settings.DEFAULT_CALIB_METERS
        args.calib_pixels = args.calib_pixels or Settings.DEFAULT_CALIB_PIXELS
        args.calib_label = args.calib_label or Settings.DEFAULT_CALIB_LABEL
    
    # Initialize system
    meters_per_pixel = None
    if args.calib_meters and args.calib_pixels and args.calib_pixels > 0:
        meters_per_pixel = args.calib_meters / args.calib_pixels

    forensic_system = ForensicAnalysisSystem(
        device=args.device,
        meters_per_pixel=meters_per_pixel,
        calibration_label=args.calib_label
    )
    
    # Get video paths
    video_paths = []
    if os.path.isfile(args.video):
        video_paths = [args.video]
    elif os.path.isdir(args.video):
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
        for root, dirs, files in os.walk(args.video):
            for file in files:
                if file.lower().endswith(video_extensions):
                    video_paths.append(os.path.join(root, file))
    
    if not video_paths:
        print("[ERROR] No videos found!")
        return
    
    print(f"[INFO] Found {len(video_paths)} video(s) for forensic analysis\n")
    
    # Analyze videos
    results = forensic_system.analyze_batch(video_paths, args.output)
    
    # Summary
    print(f"\n{'='*80}")
    print("FORENSIC ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
    no_accident_count = sum(1 for r in results if r['status'] == 'NO_ACCIDENT')
    error_count = sum(1 for r in results if r['status'] == 'ERROR')
    
    print(f"Total Videos: {len(results)}")
    print(f"  ✓ Accidents Analyzed: {success_count}")
    print(f"  ○ No Accidents: {no_accident_count}")
    print(f"  ✗ Errors: {error_count}")
    print(f"\nReports saved to: {args.output}")


if __name__ == "__main__":
    main()
