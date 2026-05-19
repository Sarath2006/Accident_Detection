"""
Fire/Smoke detector wrapper for YOLOv8m classification or detection.
"""
from typing import Optional

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise ImportError("ultralytics is required for fire detection") from exc


class FireSmokeDetector:
    """Detects fire/smoke in a frame using a YOLOv8m model."""

    def __init__(self, model_path: str, device: str = "cpu", threshold: float = 0.7):
        self.model_path = model_path
        self.device = device
        self.threshold = threshold
        self.model = YOLO(model_path)

    def _get_fire_indices(self, names):
        fire_keys = {"fire", "smoke", "flame"}
        indices = []
        for idx, name in names.items():
            if str(name).lower() in fire_keys:
                indices.append(idx)
        return indices

    def predict(self, frame_rgb) -> float:
        """
        Predict fire probability for a single RGB frame.

        Returns:
            float: fire probability in [0, 1]
        """
        results = self.model.predict(frame_rgb, verbose=False, device=self.device)
        if not results:
            return 0.0

        result = results[0]
        names = getattr(result, "names", {})
        fire_indices = self._get_fire_indices(names)

        # Classification model
        probs = getattr(result, "probs", None)
        if probs is not None and hasattr(probs, "data"):
            data = probs.data
            if fire_indices:
                return float(max(data[i] for i in fire_indices))
            # If class names are missing, use top-1 as fire score
            return float(data.max())

        # Detection model
        boxes = getattr(result, "boxes", None)
        if boxes is not None and hasattr(boxes, "cls"):
            fire_scores = []
            for cls_id, conf in zip(boxes.cls, boxes.conf):
                if int(cls_id) in fire_indices:
                    fire_scores.append(float(conf))
            return float(max(fire_scores)) if fire_scores else 0.0

        return 0.0

    def is_fire(self, frame_rgb) -> bool:
        return self.predict(frame_rgb) >= self.threshold
