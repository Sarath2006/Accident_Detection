# AI Accident Detection + Forensics + Fire (Planned) - Architecture V2

## Purpose
This document is the single source of truth for the full system architecture. It includes:
- What is implemented now
- What models are used in each part
- The full data flow (pictorial)
- Fire detection module (planned for later)

---

## Pictorial Architecture (End-to-End Data Flow)

```
INPUT VIDEO
    |
    v
[1] Frame Extractor (preprocessing/frame_extractor.py)
    |
    v
[2] Image Enhancement
    |-- resize.py
    |-- denoise.py
    |-- stabilize.py
    |-- brightness.py
    |
    v
[3] Vehicle Detection (YOLOv8)
    |-- detection/vehicle_detector.py
    |-- model: models/yolov8n.pt
    |
    v
[4] Multi-Object Tracking
    |-- tracking/tracker.py
    |
    v
[5] Speed Estimation
    |-- tracking/speed_estimator.py
    |
    v
[6] Accident Classification (ResNet50)
    |-- detection/resnet50_classifier.py
    |-- model: models/resnet50_accident_best.pth
    |-- outputs: accident_confidence, severity_class, severity_score
    |
    v
[7] Fire/Smoke Detection (PLANNED)
    |-- detection/fire_smoke_detector.py (to add)
    |-- model: models/yolov8m-fire.pt (to add)
    |-- outputs: fire_probability
    |
    v
[8] Accident Decision Logic
    |-- analysis/accident_logic.py
    |-- signals: speed_drop + overlap + resnet50_confidence + fire_probability
    |
    v
[9] Hazard Engine
    |-- intelligence/hazard_engine.py
    |
    v
[10] Outputs
    |-- accident frames -> outputs/accident_frames/
    |-- report -> accident_detection_report.txt
    |-- database -> database/accident_detection.db
    |
    v
[11] Forensic Reconstruction (optional)
    |-- forensics/reconstructor.py
    |-- physics_engine.py
    |-- fault_analyzer.py
    |-- sequence_analyzer.py
    |-- angle_calculator.py
    |-- report_generator.py (PDF)
    |-- visualization_3d.py
    |-- contributing_factors.py
```

---

## What Is Implemented Now

### Implemented Modules
- Video processing and frame extraction
- Preprocessing (resize, denoise, stabilize, brightness)
- YOLOv8 vehicle detection
- Tracking + speed estimation
- ResNet50 accident detection with severity
- Accident decision logic (multi-signal fusion)
- Report generation (text)
- Database storage (SQLite)
- Forensics reconstruction pipeline (optional)

### Implemented Models
| Model | Purpose | File | Used In |
|---|---|---|---|
| ResNet50 | Accident detection + severity | models/resnet50_accident_best.pth | detection/resnet50_classifier.py |
| YOLOv8n | Vehicle detection | models/yolov8n.pt | detection/vehicle_detector.py |

---

## Fire Detection Module (Planned)

Fire detection will be added as a strong safety signal to confirm accidents and detect dangerous events even when collision is not obvious.

### Planned Model
| Model | Purpose | File | Used In |
|---|---|---|---|
| YOLOv8m (fire/smoke) | Fire/smoke detection | models/yolov8m-fire.pt | detection/fire_smoke_detector.py |

### Planned Initialization
```python
from detection.fire_smoke_detector import FireSmokeDetector

fire_detector = FireSmokeDetector(
    model_path="models/yolov8m-fire.pt",
    device=device,
    threshold=0.7
)
```

### Planned Inference
```python
fire_score = fire_detector.predict(frame_rgb)
fire_detected = fire_score >= FIRE_PROB_THRESHOLD
```

### Planned Decision Logic Update
```python
if (accident_confidence > 0.6 and
    (speed_drop or overlap or fire_detected) and
    temporal_confirmed):
    decision = "ACCIDENT"
```

---

## Detailed Module Architecture (Every Part)

### 1) Input and Preprocessing
- File: input/video_reader.py
- File: preprocessing/frame_extractor.py
- File: preprocessing/resize.py
- File: preprocessing/denoise.py
- File: preprocessing/stabilize.py
- File: preprocessing/brightness.py

Purpose:
- Extract frames
- Normalize size and brightness
- Reduce noise and camera shake

### 2) Vehicle Detection
- File: detection/vehicle_detector.py
- Model: models/yolov8n.pt

Purpose:
- Find vehicles (car, bus, truck, motorcycle)
- Provide bounding boxes and class labels

### 3) Tracking and Speed Estimation
- File: tracking/tracker.py
- File: tracking/speed_estimator.py

Purpose:
- Assign IDs to vehicles
- Track movement across frames
- Estimate speed in pixels/frame

### 4) Accident AI (ResNet50)
- File: detection/resnet50_classifier.py
- Model: models/resnet50_accident_best.pth

Outputs:
- Accident probability (confidence)
- Severity class (Normal, Minor, Moderate, Severe)
- Severity score (0-100)

### 5) Accident Logic (Fusion)
- File: analysis/accident_logic.py

Signals Used:
- Speed drops
- Collision overlap (IoU)
- Trajectory breaks
- ResNet50 confidence
- Fire detection (planned)

### 6) Hazard Engine
- File: intelligence/hazard_engine.py

Purpose:
- Converts accident_event into risk level
- Categorizes hazard severity

### 7) Reporting
- File: simple_report.py

Purpose:
- Builds accident_detection_report.txt
- Stores video summary, confidence, severity, frames

### 8) Database Storage
- File: database/db_manager.py

Purpose:
- Stores results in accident_detection.db
- Two tables: video_accidents, accident_events

### 9) Forensic Reconstruction (Optional)
- File: forensics/reconstructor.py
- File: forensics/physics_engine.py
- File: forensics/fault_analyzer.py
- File: forensics/sequence_analyzer.py
- File: forensics/angle_calculator.py
- File: forensics/report_generator.py
- File: forensics/visualization_3d.py
- File: forensics/contributing_factors.py

Purpose:
- Physics calculations (force, energy, momentum)
- Fault determination (liability)
- Timeline reconstruction
- Accident vehicle speed summary (per vehicle and peak speed)
- Court-admissible PDF report
- 3D visualization

---

## Forensics Math Formulas

These formulas are used inside the forensics modules to compute physics metrics:

- **Relative velocity:** $v_{rel} = |v_1 - v_2|$
- **Kinetic energy:** $KE = \frac{1}{2} m v^2$
- **Impact deceleration:** $a = \frac{v^2}{2 s}$ (assuming stop distance $s$)
- **Impact force:** $F = m a$

Notes:
- $v$ is estimated from tracking (pixels/frame) and can be calibrated to m/s.
- $m$ uses a default vehicle mass from the physics engine.
- $s$ uses a default crumple zone distance.

---

## Model Usage by Pipeline Stage

| Stage | Model | Output | Why Needed |
|---|---|---|---|
| Vehicle Detection | YOLOv8n | boxes + class | Find vehicles and track them |
| Accident Detection | ResNet50 | confidence + severity | Classify accidents from frames |
| Fire Detection (planned) | YOLOv8m (fire/smoke) | fire probability | Confirm accidents and hazards |

---

## Outputs

- Accident frames: outputs/accident_frames/
- Text report: accident_detection_report.txt
    - Includes accident confidence, severity, and vehicle speed at event time
- Database: database/accident_detection.db
- Forensic PDF: generated via forensics/report_generator.py
    - Includes accident vehicle speed summary (per vehicle and peak speed)
- 3D visualization: generated via forensics/visualization_3d.py

---

## Configuration (Key Settings)

From config/settings.py:
- ACCIDENT_PROB_THRESHOLD
- FRAME_SKIP
- SPEED_DROP_THRESHOLD_PIXELS
- OVERLAP_IOU_THRESHOLD
- TEMPORAL_WINDOW

Planned additions:
- FIRE_MODEL_PATH
- FIRE_PROB_THRESHOLD

---

## Summary

- Accident detection is fully implemented with ResNet50 + YOLOv8 + tracking.
- Forensics module is complete and can generate court-admissible reports.
- Fire detection is planned and will be integrated as a strong evidence signal.
- This architecture is the full blueprint for the system now and next.
