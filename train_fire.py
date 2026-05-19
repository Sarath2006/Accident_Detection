"""
Train YOLOv8m fire/smoke classification model.
Dataset format (classification):

datasets/fire/
  train/
    Fire/
    NonFire/
  val/
    Fire/
    NonFire/
"""
import argparse
import os
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8m fire/smoke classifier")
    parser.add_argument("--data", type=str, default="datasets/fire", help="Dataset root folder")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=224, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="cuda or cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.data):
        raise SystemExit(f"Dataset not found: {args.data}")

    print("[INFO] Training YOLOv8m fire/smoke classifier")
    print(f"[INFO] Dataset: {args.data}")

    model = YOLO("yolov8m-cls.pt")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs_fire",
        name="yolov8m_fire_cls"
    )

    # Export best model to project models folder
    best_path = os.path.join("runs_fire", "yolov8m_fire_cls", "weights", "best.pt")
    if os.path.exists(best_path):
        os.makedirs("models", exist_ok=True)
        target = os.path.join("models", "yolov8m-fire.pt")
        os.replace(best_path, target)
        print(f"[OK] Saved fire model to: {target}")
    else:
        print("[WARN] Best model not found. Check training outputs.")


if __name__ == "__main__":
    main()
