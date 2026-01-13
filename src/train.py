"""
Train + evaluate YOLOv8 on dataset/data.yaml using a pretrained YOLOv8 nano model.

This is equivalent to the CLI command in the prompt, but in Python:
  yolo task=detect mode=train model=yolov8n.pt data=dataset/data.yaml epochs=50 imgsz=640 batch=16
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="dataset/data.yaml")
    ap.add_argument("--model", type=str, default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--project", type=str, default="runs/detect")
    ap.add_argument("--name", type=str, default="train")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        exist_ok=True,
        seed=42,
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    if best.exists():
        YOLO(str(best)).val(data=args.data)
        print(f"[OK] best.pt: {best}")
    else:
        print(f"[WARN] best.pt not found at expected path: {best}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

