"""
Sanity checks for the prepared dataset/ folder.

Checks:
  - Expected folder structure exists.
  - Every image has a corresponding label .txt (empty is allowed).
  - Every label row is YOLO format: cls x_center y_center w h
  - All coords are normalized to [0, 1].
  - Class ids are only {0,1,2}.
"""

from __future__ import annotations

import argparse
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_CLASSES = {0, 1, 2}


def _iter_images(images_dir: Path) -> list[Path]:
    if not images_dir.is_dir():
        return []
    return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def _check_label_file(label_path: Path) -> tuple[int, int]:
    """
    Returns: (num_objects, num_errors)
    """
    if not label_path.exists():
        return (0, 1)

    n_obj = 0
    n_err = 0
    for i, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            n_err += 1
            continue

        try:
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            n_err += 1
            continue

        if cls not in ALLOWED_CLASSES:
            n_err += 1
            continue

        # YOLO normalized coords
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
            n_err += 1
            continue
        if w <= 0.0 or h <= 0.0:
            n_err += 1
            continue

        n_obj += 1

    return (n_obj, n_err)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, default="dataset", help="Path to dataset/ directory.")
    args = ap.parse_args()

    ds = Path(args.dataset_dir).expanduser().resolve()

    required = [
        ds / "images" / "train",
        ds / "images" / "val",
        ds / "labels" / "train",
        ds / "labels" / "val",
        ds / "data.yaml",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("[FAIL] Missing required paths:")
        for p in missing:
            print(f"  - {p}")
        return 2

    total_imgs = 0
    total_objs = 0
    total_errs = 0
    missing_labels = 0

    for split in ["train", "val"]:
        images_dir = ds / "images" / split
        labels_dir = ds / "labels" / split
        imgs = _iter_images(images_dir)
        if not imgs:
            print(f"[WARN] No images found in {images_dir}")
            continue

        for img in imgs:
            total_imgs += 1
            label = labels_dir / f"{img.stem}.txt"
            n_obj, n_err = _check_label_file(label)
            total_objs += n_obj
            total_errs += n_err
            if not label.exists():
                missing_labels += 1

    if missing_labels:
        print(f"[FAIL] Missing label files for {missing_labels} images.")
        print("       Create empty .txt files for background images (no objects).")
        return 2

    if total_errs:
        print(f"[FAIL] Found {total_errs} label errors across {total_imgs} images.")
        print("       Fix labels or re-run src/prepare_dataset.py.")
        return 2

    print("[OK] Dataset looks good.")
    print(f"[OK] Images: {total_imgs} | Total objects: {total_objs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

