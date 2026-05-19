# Methodology

This section describes the complete methodology of the proposed accident and hazard detection framework. The system integrates deep learning, tracking, motion analysis, and forensic reconstruction to deliver a unified, real-time traffic safety solution. The pipeline is designed for video surveillance input and can be extended with a dedicated fire/smoke detector.

---

## 1. System Overview
The system processes road surveillance videos and produces: (1) accident detection with severity, (2) hazard classification, and (3) forensic evidence generation. The method combines:
- YOLOv8 vehicle detection
- Multi-object tracking and speed estimation
- ResNet50 accident classification with severity prediction
- Multi-signal fusion for final accident decision
- Optional forensic reconstruction (physics, fault, timeline)
- Planned fire/smoke detection using YOLOv8m

---

## 2. Data Preparation
The training dataset uses a folder-based image structure:
- `datasets/Accident/` (label = 1)
- `datasets/NonAccident/` (label = 0)

Images are resized to 224x224 for ResNet50 input. Augmentation is applied during training:
- Random rotation (10 degrees)
- Random horizontal flip
- Color jitter (brightness, contrast)
- Normalization with ImageNet mean and std

The dataset is split 80% for training and 20% for validation.

---

## 3. Vehicle Detection (YOLOv8n)
Vehicles are detected in each frame using YOLOv8n. The detector identifies classes: car, bus, truck, motorbike.

Key settings:
- Confidence threshold: 0.4
- IoU threshold: 0.5

Detected bounding boxes are passed to the tracking module.

---

## 4. Multi-Object Tracking
Multi-object tracking assigns persistent IDs to vehicles across frames. This enables:
- Stable vehicle trajectories
- Speed estimation
- Collision proximity and overlap checks

Tracking output is used for motion analysis and accident confirmation.

---

## 5. Speed Estimation
Speed is estimated from the displacement of tracked vehicle centers between frames (pixels/frame). These values are used to detect sudden deceleration or prolonged stops, which are strong accident indicators.

---

## 6. Accident Classification (ResNet50)
A ResNet50 backbone is used for multi-task learning with three heads:
1. Accident classification (binary)
2. Severity classification (4 classes: Normal, Minor, Moderate, Severe)
3. Severity regression (0-100 score)

Loss function:

$$
L = 0.5L_{acc} + 0.3L_{sev} + 0.2L_{reg}
$$

Optimization:
- Optimizer: AdamW
- Learning rate: 1e-4
- Weight decay: 1e-5
- Scheduler: ReduceLROnPlateau (factor 0.5, patience 5)

---

## 7. Multi-Signal Accident Decision Logic
Final accident decisions use a fusion of multiple signals:
- Speed drops (motion anomaly)
- Collision overlap and proximity (spatial interaction)
- Trajectory anomalies and box deformation
- ResNet50 accident confidence

Evidence is combined into an accident score. Temporal confirmation requires consistent evidence across multiple frames to reduce false positives.

---

## 8. Hazard Engine
The hazard engine converts accident events into a structured risk level. It uses accident confidence and severity indicators to classify hazard severity for reporting and alerts.

---

## 9. Forensic Reconstruction
The forensic subsystem generates physics-based evidence and a court-admissible report. It computes:

- Relative velocity:

$$
 v_{rel} = |v_1 - v_2|
$$

- Kinetic energy:

$$
 KE = \frac{1}{2}mv^2
$$

- Deceleration (assuming stop distance s):

$$
 a = \frac{v^2}{2s}
$$

- Impact force:

$$
 F = ma
$$

The forensic pipeline includes:
- Physics analysis
- Fault determination with liability percentages
- Timeline reconstruction
- PDF report generation
- Optional 3D visualization

---

## 10. Fire and Smoke Detection (Planned Extension)
Fire detection will be added using YOLOv8m fine-tuned for fire/smoke. The planned fire signal will be fused with accident evidence for stronger hazard confirmation. If fire is detected with a high accident confidence, severity is raised and the hazard is escalated.

---

## 11. Outputs
The system produces:
- Accident frames in `outputs/accident_frames/`
- Text report: `accident_detection_report.txt`
- SQLite database: `database/accident_detection.db`
- Forensic PDF report: `forensics_output/`

---

## 12. Execution
Training:
```
python train.py --epochs 10 --batch-size 8
```

Inference:
```
python main.py --dataset vidoes/
```
