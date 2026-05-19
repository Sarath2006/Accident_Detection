import os

class Settings:
    """
    Central configuration file for the AI Accident Detection System
    """

    # ---------------- Project Roots ----------------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # ----------- INPUT -----------
    DEFAULT_FPS = 30
    FRAME_SIZE = (640, 640)
    FRAME_SKIP = 2  # Process every Nth frame (1=all frames, 2=every 2nd frame, etc.)

    # ----------- MODEL PATHS -----------
    MODELS_DIR = os.path.join(BASE_DIR, "models")

    YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n.pt")
    ACCIDENT_CNN_PATH = os.path.join(MODELS_DIR, "resnet50_accident_best.pth")  # ResNet50 model
    FIRE_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8m-fire.pt")

    # ----------- YOLO SETTINGS -----------
    YOLO_CONF_THRESHOLD = 0.4
    YOLO_IOU_THRESHOLD = 0.5
    VEHICLE_CLASSES = ["car", "bus", "truck", "motorbike"]

    # ----------- ACCIDENT DETECTION SETTINGS -----------
    ACCIDENT_PROB_THRESHOLD = 0.6  # ResNet50 confidence threshold
    FIRE_PROB_THRESHOLD = 0.85  # Fire classification threshold (INCREASED - reduce false positives)
    FIRE_TEMPORAL_WINDOW = 12  # Frames for fire confirmation window (INCREASED)
    FIRE_MIN_HITS = 4  # Minimum hits within window to confirm fire (INCREASED)

    # ----------- TRACKING SETTINGS -----------
    MAX_LOST_FRAMES = 10
    SPEED_DROP_THRESHOLD = 0.7   # sudden speed reduction ratio

    # ----------- ACCIDENT LOGIC -----------
    STOP_TIME_THRESHOLD = 15     # frames
    OVERLAP_IOU_THRESHOLD = 0.15  # LOWERED - More sensitive to overlaps
    SPEED_DROP_THRESHOLD_PIXELS = 3.0  # Lowered for more sensitive detection

    # ----------- OUTPUT / DEBUG -----------
    SHOW_VISUALS = True
    LOG_LEVEL = "INFO"

    # ----------- FORENSICS CALIBRATION -----------
    DEFAULT_CALIB_LABEL = "lane width"
    DEFAULT_CALIB_METERS = 3.5
    DEFAULT_CALIB_PIXELS = 420

    # ----------- FORENSICS SENSITIVITY -----------
    FORENSIC_ACCIDENT_THRESHOLD_BIAS = -1.0