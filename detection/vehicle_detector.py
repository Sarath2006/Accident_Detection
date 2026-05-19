"""Rewritten vehicle detector that WORKS with tensor parsing"""
import os
import cv2
import torch
from ultralytics import YOLO


class VehicleDetector:
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.5, vehicle_classes=None):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.temp_dir = "temp_frames"
        os.makedirs(self.temp_dir, exist_ok=True)
        
        if vehicle_classes is None:
            vehicle_classes = {"car", "bus", "truck", "motorbike", "motorcycle"}
        self.vehicle_classes = {c.lower() for c in vehicle_classes}
        
        # Get YOLO model class names
        if hasattr(self.model, 'names'):
            self.class_names = self.model.names
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'names'):
            self.class_names = self.model.model.names
        else:
            # Default COCO class names
            self.class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

    def detect(self, frame):
        if frame is None:
            return []

        # Get original frame dimensions
        orig_h, orig_w = frame.shape[:2]

        # Save frame temporarily
        temp_path = os.path.join(self.temp_dir, "frame_tmp.jpg")
        cv2.imwrite(temp_path, frame)

        try:
            # Run inference
            results = self.model(temp_path, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False, imgsz=640)
            
            # Clean up
            try:
                os.remove(temp_path)
            except:
                pass
            
            # Extract detections using proper YOLOv8 API
            detections = []
            
            if len(results) > 0:
                result = results[0]  # ultralytics.engine.results.Results object
                
                # Use the boxes attribute (correct YOLOv8 API)
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    boxes = result.boxes
                    
                    # Get all detections at once
                    xyxy = boxes.xyxy.cpu().numpy()  # Bounding boxes [x1, y1, x2, y2]
                    confs = boxes.conf.cpu().numpy()  # Confidences
                    classes = boxes.cls.cpu().numpy()  # Class IDs
                    
                    for i in range(len(boxes)):
                        cls_id = int(classes[i])
                        confidence = float(confs[i])
                        
                        # Get class name
                        cls_name = self.class_names.get(cls_id, "")
                        
                        # Filter vehicles only
                        if cls_name and cls_name.lower() in self.vehicle_classes:
                            x1, y1, x2, y2 = xyxy[i]
                            
                            detections.append({
                                "class": cls_name,
                                "confidence": confidence,
                                "bbox": (int(x1), int(y1), int(x2), int(y2))
                            })
            
            return detections
            
        except Exception as e:
            print(f"[ERROR] Detection failed: {e}")
            import traceback
            traceback.print_exc()
            try:
                os.remove(temp_path)
            except:
                pass
            return []
