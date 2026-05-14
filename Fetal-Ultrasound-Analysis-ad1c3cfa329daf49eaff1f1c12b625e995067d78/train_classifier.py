"""
Train a YOLOv8 classification model for fetal ultrasound plane recognition.

Use this when your dataset is folder-labeled classification data, not bounding-box
detection data.

Expected layout:
    dataset_cls/
      train/fetal_femur/*.jpg
      val/fetal_femur/*.jpg

Output:
    runs/classify/train/weights/best.pt
    runs/classify/train/weights/last.pt

Usage:
    python train_classifier.py
    python train_classifier.py --data dataset_cls
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BASE_DIR = Path(__file__).resolve().parent


def auto_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(8, cpu_count - 1))


def count_images(root: Path) -> int:
    return sum(
        1 for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_dataset(data_root: Path) -> None:
    if not data_root.exists():
        raise FileNotFoundError(f"Classification dataset not found: {data_root}")
    for split in ("train", "val"):
        split_dir = data_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing required folder: {split_dir}")
        classes = [path for path in split_dir.iterdir() if path.is_dir()]
        if len(classes) < 2:
            raise ValueError(f"{split_dir} must contain at least two class folders.")
        if count_images(split_dir) == 0:
            raise ValueError(f"No images found in {split_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 fetal ultrasound classifier")
    parser.add_argument("--data", default="dataset_cls", help="YOLO classification dataset root")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--cache", default="True", help="Use YOLO cache=True/False")
    parser.add_argument("--workers", default="auto", help="Worker count or 'auto'")
    parser.add_argument("--project", default=str(BASE_DIR / "runs" / "classify"))
    parser.add_argument("--name", default="train")
    parser.add_argument("--copy-best", default="", help="Optional path to copy best.pt after training")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = (BASE_DIR / args.data).resolve() if not Path(args.data).is_absolute() else Path(args.data)
    validate_dataset(data_root)
    workers = auto_workers() if str(args.workers).lower() == "auto" else int(args.workers)
    cache = str(args.cache).lower() in {"1", "true", "yes", "y"}

    device = 0 if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.total_memory / (1024 ** 3):.2f} GB)")
    else:
        print("Device: CPU")
    print(f"Data: {data_root}")
    print(f"Output: {Path(args.project) / args.name}")

    model = YOLO("yolov8n-cls.pt")
    results = model.train(
        data=str(data_root),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        cache=cache,
        workers=workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        device=device,
        plots=True,
        save=True,
    )

    save_dir = Path(getattr(results, "save_dir", Path(args.project) / args.name))
    best = save_dir / "weights" / "best.pt"
    print(f"Best classifier saved at: {best}")
    if args.copy_best and best.exists():
        shutil.copy2(best, args.copy_best)
        print(f"Copied best classifier to: {Path(args.copy_best).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
