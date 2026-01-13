"""
Run inference with a trained YOLOv8 model on an image folder or video.

Examples:
  python src/predict.py --weights runs/detect/train/weights/best.pt --source test_images/
  python src/predict.py --weights runs/detect/train/weights/best.pt --source construction_site.mp4
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True, help="Path to best.pt")
    ap.add_argument("--source", type=str, required=True, help="Image/dir/video source")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    model = YOLO(args.weights)
    # Ultralytics will write outputs under runs/detect/predict* when save=True.
    model.predict(source=args.source, conf=args.conf, save=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

