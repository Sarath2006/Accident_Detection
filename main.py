"""
Enhanced Accident Detection with Improved Logic V2
Best accuracy with multi-modal fusion and adaptive thresholds
"""
import os
import csv
import cv2
import argparse
from collections import deque
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from config.settings import Settings
from input.video_reader import VideoReader

from preprocessing.frame_extractor import FrameExtractor
from preprocessing.resize import FrameResizer
from preprocessing.denoise import FrameDenoiser
from preprocessing.stabilize import VideoStabilizer
from preprocessing.brightness import BrightnessNormalizer

from detection.vehicle_detector import VehicleDetector
from detection.resnet50_classifier import ResNet50AccidentDetector
from detection.fire_smoke_detector import FireSmokeDetector

from tracking.tracker import ObjectTracker
from tracking.speed_estimator import SpeedEstimator

from analysis.enhanced_accident_logic import EnhancedAccidentLogic
from intelligence.hazard_engine import HazardEngine

from utils.visualization import Visualizer
from database.db_manager import DatabaseManager
from forensic_analysis import ForensicAnalysisSystem

# Import simple report generator
try:
    from simple_report import SimpleReportGenerator
except ImportError:
    # Fallback if simple_report not available
    class SimpleReportGenerator:
        def __init__(self):
            self.results = []
        def add_video_result(self, *args, **kwargs):
            pass
        def print_report(self):
            pass
        def save_report(self, filename):
            return filename


def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced AI Accident Detection")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to video file or folder with videos"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save per-frame debug scores and selected frames"
    )
    parser.add_argument(
        "--debug-output",
        type=str,
        default="debug_output",
        help="Output directory for debug artifacts"
    )
    parser.add_argument(
        "--debug-score-threshold",
        type=float,
        default=2.5,
        help="Save frames when accident score >= this value"
    )
    parser.add_argument(
        "--debug-cnn-threshold",
        type=float,
        default=0.3,
        help="Save frames when CNN confidence >= this value"
    )
    parser.add_argument(
        "--debug-save-every",
        type=int,
        default=0,
        help="Save every Nth processed frame (0 disables)"
    )
    parser.add_argument(
        "--forensics",
        dest="forensics",
        action="store_true",
        help="Generate forensic reports after detection"
    )
    parser.add_argument(
        "--no-forensics",
        dest="forensics",
        action="store_false",
        help="Skip forensic reports"
    )
    parser.add_argument(
        "--forensics-output",
        type=str,
        default="forensics_output",
        help="Output directory for forensic reports"
    )
    parser.set_defaults(forensics=True)
    return parser.parse_args()


def resolve_video_paths(dataset_path):
    """Get all video files from path"""
    video_extensions = (".mp4", ".avi", ".mov", ".mkv")
    video_files = []

    if os.path.isfile(dataset_path):
        if dataset_path.lower().endswith(video_extensions):
            return [dataset_path]
        return []

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(video_extensions):
                video_files.append(os.path.join(root, file))

    return sorted(video_files)


def assess_video_quality(video_path, sample_count=5):
    """Estimate basic video quality to tune preprocessing and thresholds."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "low_quality": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur": 0.0,
            "frames": 0
        }

    brightness_values = []
    contrast_values = []
    blur_values = []
    frames = 0

    while frames < sample_count:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_values.append(float(np.mean(gray)))
        contrast_values.append(float(np.std(gray)))
        blur_values.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        frames += 1

    cap.release()

    if frames == 0:
        return {
            "low_quality": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur": 0.0,
            "frames": 0
        }

    avg_brightness = float(np.mean(brightness_values))
    avg_contrast = float(np.mean(contrast_values))
    avg_blur = float(np.mean(blur_values))

    low_light = avg_brightness < 60
    overexposed = avg_brightness > 190
    low_contrast = avg_contrast < 35
    blurry = avg_blur < 100
    low_quality = sum([low_light, overexposed, low_contrast, blurry]) >= 2

    return {
        "low_quality": low_quality,
        "brightness": avg_brightness,
        "contrast": avg_contrast,
        "blur": avg_blur,
        "frames": frames
    }


def main():
    args = parse_args()
    
    print("\n" + "="*80)
    print("ENHANCED ACCIDENT DETECTION V2 - BEST LOGIC")
    print("="*80 + "\n")

    # Initialize
    settings = Settings()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}\n")

    # Models
    print("[INFO] Loading models...")
    accident_detector = ResNet50AccidentDetector(num_severity_classes=4, pretrained=True)
    accident_detector = accident_detector.to(device)
    
    model_path = 'models/resnet50_accident_best.pth'
    if os.path.exists(model_path):
        try:
            accident_detector.load_state_dict(torch.load(model_path, map_location=device))
            print(f"[INFO] Loaded pretrained model from {model_path}")
        except Exception as e:
            print(f"[WARNING] Could not load model: {e}")
    
    accident_detector.eval()

    fire_detector = None
    if os.path.exists(settings.FIRE_MODEL_PATH):
        try:
            fire_detector = FireSmokeDetector(
                model_path=settings.FIRE_MODEL_PATH,
                device=str(device),
                threshold=settings.FIRE_PROB_THRESHOLD
            )
            print(f"[INFO] Loaded fire model from {settings.FIRE_MODEL_PATH}")
        except Exception as e:
            print(f"[WARNING] Fire model not available: {e}")
    else:
        print(f"[WARNING] Fire model not found at {settings.FIRE_MODEL_PATH}")

    # Preprocessing
    extractor = FrameExtractor()
    resizer = FrameResizer(target_size=(640, 480))
    
    hazard_engine = HazardEngine()
    visualizer = Visualizer()
    report = SimpleReportGenerator()

    # Database
    os.makedirs(args.output, exist_ok=True)
    db = DatabaseManager('database/accident_detection.db')

    if args.debug:
        os.makedirs(args.debug_output, exist_ok=True)
        os.makedirs(os.path.join(args.debug_output, "frames"), exist_ok=True)

    # Preprocessing transform
    accident_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Process videos
    video_paths = resolve_video_paths(args.dataset)
    print(f"[INFO] Found {len(video_paths)} video(s)\n")

    if not video_paths:
        print("[ERROR] No videos found!")
        return

    os.makedirs(f"{args.output}/accident_frames", exist_ok=True)

    for video_path in video_paths:
        video_name = os.path.basename(video_path)
        video_stem = os.path.splitext(video_name)[0].replace(" ", "_")
        category = os.path.basename(os.path.dirname(video_path))

        quality = assess_video_quality(video_path)
        if quality["low_quality"]:
            denoiser = FrameDenoiser(method="median")
            brightness = BrightnessNormalizer(clip_limit=3.0, tile_grid_size=(8, 8))
            yolo_conf = 0.25
            yolo_iou = 0.45
            accident_logic = EnhancedAccidentLogic(
                speed_drop_threshold=2.5,
                overlap_iou_threshold=0.12,
                cnn_threshold=0.5,
                temporal_window=8,
                proximity_threshold=70,
                speed_scale=settings.FRAME_SKIP
            )
        else:
            denoiser = FrameDenoiser(method="gaussian")
            brightness = BrightnessNormalizer(clip_limit=2.0, tile_grid_size=(8, 8))
            yolo_conf = 0.35
            yolo_iou = 0.5
            accident_logic = EnhancedAccidentLogic(
                speed_drop_threshold=3.0,
                overlap_iou_threshold=0.15,
                cnn_threshold=0.55,
                temporal_window=8,
                proximity_threshold=50,
                speed_scale=settings.FRAME_SKIP
            )

        stabilizer = VideoStabilizer()
        vehicle_detector = VehicleDetector(
            model_path='models/yolov8n.pt',
            conf_threshold=yolo_conf,
            iou_threshold=yolo_iou
        )
        tracker = ObjectTracker()
        speed_estimator = SpeedEstimator()

        print(f"\n[DETECTION] Processing: {video_name} | Category: {category}")
        print("="*80)

        video_accidents = []
        max_severity = "NORMAL"
        max_confidence = 0.0
        vehicles_involved = set()
        accident_frame_path = None
        frame_count = 0
        accident_frame_num = 0

        fire_window = deque(maxlen=settings.FIRE_TEMPORAL_WINDOW)
        fire_detected_any = False
        max_fire_confidence = 0.0
        fire_frame_path = None

        fire_threshold = settings.FIRE_PROB_THRESHOLD
        if quality["low_quality"]:
            fire_threshold = min(0.9, fire_threshold + 0.1)

        debug_file = None
        debug_writer = None
        if args.debug:
            debug_path = os.path.join(args.debug_output, f"{video_stem}_debug.csv")
            debug_file = open(debug_path, "w", newline="", encoding="utf-8")
            debug_writer = csv.writer(debug_file)
            debug_writer.writerow([
                "frame",
                "cnn_confidence",
                "fire_confidence",
                "fire_detected",
                "accident_score",
                "hazard_status",
                "severity",
                "confidence",
                "vehicles",
                "speed_drops",
                "collisions",
                "trajectory_anomalies",
                "deformations",
                "evidence_count",
                "consecutive_frames"
            ])

        video_reader = VideoReader(video_path)
        pbar = tqdm(total=video_reader.total_frames, desc=f"Processing {video_name}", unit="frames")

        with torch.no_grad():
            for frame in video_reader:
                frame_count += 1
                pbar.update(1)

                # Skip frames for performance
                if frame_count % settings.FRAME_SKIP != 0:
                    continue

                # Preprocessing
                frame = extractor.extract(frame)
                frame = resizer.apply(frame)
                frame = denoiser.apply(frame)
                frame = stabilizer.apply(frame)
                frame = brightness.apply(frame)

                if frame is None:
                    continue

                # Vehicle detection
                vehicles = vehicle_detector.detect(frame)
                
                # ResNet50 accident classification
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = accident_transform(frame_rgb).unsqueeze(0).to(device)
                accident_pred = accident_detector.predict(frame_tensor)
                
                cnn_confidence = accident_pred.get('accident_confidence', 0.0)
                if isinstance(cnn_confidence, torch.Tensor):
                    cnn_confidence = float(cnn_confidence.item())

                fire_score = 0.0
                fire_smoke = False
                if fire_detector is not None:
                    fire_score = float(fire_detector.predict(frame_rgb))
                    fire_window.append(fire_score >= fire_threshold)
                    fire_hits = sum(1 for hit in fire_window if hit)
                    fire_smoke = fire_hits >= settings.FIRE_MIN_HITS
                    if fire_score > max_fire_confidence:
                        max_fire_confidence = fire_score

                # Tracking and speed
                tracked_objects = tracker.update(vehicles)
                speeds = speed_estimator.compute(tracked_objects)

                # ENHANCED ACCIDENT LOGIC
                accident_event = accident_logic.evaluate(
                    tracked_objects=tracked_objects,
                    speeds=speeds,
                    cnn_score=cnn_confidence,
                    fire_smoke=fire_smoke
                )

                # Final decision
                hazard = hazard_engine.decide(accident_event)

                if fire_smoke and hazard.get("status") != "ACCIDENT":
                    hazard = {
                        "status": "FIRE",
                        "severity": "SEVERE",
                        "confidence": max(hazard.get("confidence", 0.0), fire_score),
                        "reasons": ["fire_detected"],
                        "accident_score": accident_event.get("accident_score", 0.0),
                        "cnn_score": cnn_confidence
                    }

                if fire_smoke:
                    fire_detected_any = True

                if args.debug and debug_writer:
                    evidence_count = sum([
                        1 if accident_event.get("collisions") else 0,
                        1 if accident_event.get("speed_drops") else 0,
                        1 if accident_event.get("trajectory_anomalies") else 0,
                        1 if accident_event.get("deformations") else 0,
                        1 if accident_event.get("cnn_score", 0.0) >= accident_logic.cnn_threshold else 0
                    ])
                    debug_writer.writerow([
                        frame_count,
                        f"{cnn_confidence:.4f}",
                        f"{fire_score:.4f}",
                        int(fire_smoke),
                        f"{accident_event.get('accident_score', 0.0):.4f}",
                        hazard.get("status", ""),
                        hazard.get("severity", ""),
                        f"{hazard.get('confidence', 0.0):.4f}",
                        len(tracked_objects),
                        len(accident_event.get("speed_drops", {})),
                        len(accident_event.get("collisions", {})),
                        len(accident_event.get("trajectory_anomalies", {})),
                        len(accident_event.get("deformations", {})),
                        evidence_count,
                        accident_event.get("consecutive_frames", 0)
                    ])

                # Track vehicles involved
                for obj_id in tracked_objects.keys():
                    if hazard["status"] == "ACCIDENT":
                        obj = tracked_objects[obj_id]
                        vehicles_involved.add(f"{obj.get('class', 'unknown')}#{obj_id}")

                # Record accidents
                if hazard["status"] == "ACCIDENT":
                    severity = hazard.get("severity", "MINOR")
                    confidence = hazard.get("confidence", 0.0)
                    score = accident_event.get("accident_score", 0.0)
                    
                    video_accidents.append({
                        "frame": frame_count,
                        "severity": severity,
                        "confidence": confidence,
                        "score": score,
                        "reasons": accident_event.get("reasons", [])
                    })

                    # Track worst case
                    severity_order = {"MINOR": 1, "MODERATE": 2, "SEVERE": 3}
                    if severity_order.get(severity, 0) > severity_order.get(max_severity, 0):
                        max_severity = severity
                        max_confidence = confidence
                        accident_frame_num = frame_count
                        
                        # Save frame
                        frame_filename = f"{video_name.replace('.mp4', '')}_frame_{frame_count}.jpg"
                        frame_path = f"{args.output}/accident_frames/{frame_filename}"
                        cv2.imwrite(frame_path, frame)
                        accident_frame_path = frame_path
                        
                        print(f"  [ALERT] Accident detected at frame {frame_count}: {severity} "
                              f"(Confidence: {confidence:.1%}, Score: {score:.1f}/10)")

                if fire_smoke and fire_frame_path is None:
                    fire_filename = f"{video_name.replace('.mp4', '')}_fire_{frame_count}.jpg"
                    fire_path = f"{args.output}/accident_frames/{fire_filename}"
                    cv2.imwrite(fire_path, frame)
                    fire_frame_path = fire_path
                    print(f"  [ALERT] Fire detected at frame {frame_count} (Confidence: {fire_score:.1%})")

                # Visualization
                frame_display = frame.copy()
                frame_display = visualizer.draw_raw_detections(frame_display, vehicles)
                frame_display = visualizer.draw_detections(frame_display, tracked_objects, speeds)
                frame_display = visualizer.draw_hazard(frame_display, hazard)
                
                # Add metrics
                cv2.putText(frame_display, f"Frame: {frame_count} | Score: {accident_event.get('accident_score', 0):.1f}", 
                           (20, frame_display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if args.debug:
                    should_save = False
                    if hazard.get("status") == "ACCIDENT":
                        should_save = True
                    if accident_event.get("accident_score", 0.0) >= args.debug_score_threshold:
                        should_save = True
                    if cnn_confidence >= args.debug_cnn_threshold:
                        should_save = True
                    if accident_event.get("collisions"):
                        should_save = True
                    if args.debug_save_every > 0 and frame_count % args.debug_save_every == 0:
                        should_save = True

                    if should_save:
                        debug_frame_name = f"{video_stem}_frame_{frame_count}.jpg"
                        debug_frame_path = os.path.join(args.debug_output, "frames", debug_frame_name)
                        cv2.imwrite(debug_frame_path, frame_display)
                
                cv2.imshow("Enhanced Accident Detection", frame_display)
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    break

        video_reader.release()
        pbar.close()

        if debug_file:
            debug_file.close()

        # Store results
        if fire_detected_any and max_severity == "NORMAL":
            max_severity = "SEVERE"
            max_confidence = max(max_confidence, max_fire_confidence)
            if not accident_frame_path and fire_frame_path:
                accident_frame_path = fire_frame_path

        accident_detected = 1 if (max_severity != "NORMAL" or fire_detected_any) else 0
        vehicles_str = ",".join(list(vehicles_involved)) if vehicles_involved else "NONE"

        db.insert_video_accident(
            video_name=video_name,
            category=category,
            accident_detected=accident_detected,
            severity=max_severity,
            confidence=max_confidence,
            fire_smoke=1 if fire_detected_any else 0,
            vehicles_involved=vehicles_str,
            estimated_deaths=0,
            accident_frame_path=accident_frame_path or ""
        )

        # Summary
        print(f"\n[SUMMARY] {video_name}:")
        print(f"  Status: {max_severity}")
        print(f"  Confidence: {max_confidence:.1%}")
        print(f"  Accidents detected: {len(video_accidents)}")
        print(f"  Vehicles involved: {vehicles_str}")
        if fire_detected_any:
            print(f"  Fire Detected: YES ({max_fire_confidence:.1%})")
            print("  Action: IMMEDIATE ACTION NEEDED")
        
        # Add to report
        report.add_video_result(
            video_name=video_name,
            category=category,
            accident_detected=bool(accident_detected),
            severity=max_severity,
            confidence=max_confidence,
            accident_frame_path=accident_frame_path or "",
            vehicles_involved=list(vehicles_involved),
            fire_detected=fire_detected_any,
            fire_confidence=max_fire_confidence,
            action_required=fire_detected_any
        )

    # Finalize
    db.close()
    cv2.destroyAllWindows()
    
    report.print_report()
    report.save_report("accident_detection_report_v2.txt")

    if args.forensics:
        print("\n" + "=" * 80)
        print("FORENSIC ANALYSIS (POST-DETECTION)")
        print("=" * 80 + "\n")
        forensic_system = ForensicAnalysisSystem(device=args.device)
        forensic_system.analyze_batch(video_paths, args.forensics_output)
    
    print("\n" + "="*80)
    print("DETECTION COMPLETE")
    print("="*80)
    print(f"Report: accident_detection_report_v2.txt")
    print(f"Frames: {args.output}/accident_frames/")
    print(f"Database: database/accident_detection.db")


if __name__ == "__main__":
    main()
