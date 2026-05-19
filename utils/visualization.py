import cv2


class Visualizer:
    """
    Handles all visualization and overlays
    """

    def draw_detections(self, frame, tracked_objects, speeds):
        """Draw tracked objects with ID, speed, and colored boxes"""
        for object_id, obj in tracked_objects.items():
            x1, y1, x2, y2 = obj["bbox"]
            label = obj["class"]

            speed = speeds.get(object_id, 0.0)

            # Use yellow/gold for tracked objects to differentiate from raw detections
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            # Create tracking info label
            track_text = f"ID:{object_id} {label} {speed:.1f}px/s"
            cv2.putText(
                frame,
                track_text,
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )

        return frame

    def draw_raw_detections(self, frame, detections):
        """Draw raw YOLO detections directly with clear labels and colors"""
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["class"]
            confidence = det["confidence"]

            # Use bright colors for different vehicle types
            color_map = {
                "car": (0, 255, 0),      # Green
                "truck": (0, 165, 255),  # Orange
                "bus": (255, 0, 0),      # Blue
                "motorcycle": (255, 0, 255),  # Magenta
                "motorbike": (255, 0, 255)
            }
            color = color_map.get(label.lower(), (255, 255, 0))  # Default: Cyan

            # Draw bounding box (thicker line)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Create label background
            label_text = f"{label.upper()} {confidence:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
            
            # Draw label text
            cv2.putText(
                frame,
                label_text,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),  # Black text on colored background
                2
            )

        return frame

    def draw_hazard(self, frame, hazard):
        if hazard["status"] == "ACCIDENT":
            text = f"ACCIDENT | {hazard['severity']}"
            color = (0, 0, 255)
        else:
            text = "NORMAL"
            color = (0, 255, 0)

        cv2.putText(
            frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            3
        )

        return frame
