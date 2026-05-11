"""
Train a YOLOv8 classification model for fetal ultrasound analysis.

Expected dataset layout:
    dataset/
      train/
        Fetal brain/
        Fetal abdomen/
        ...
      val/
        Fetal brain/
        Fetal abdomen/
        ...
      test/
        Fetal brain/
        Fetal abdomen/
        ...

Usage:
    python train_model.py --data dataset --epochs 50
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(root: Path) -> int:
    return sum(
        1
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_dataset(data_dir: Path) -> None:
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {data_dir}\n"
            "Create it with Phase1_Data_Preparation.ipynb or copy a YOLO "
            "classification dataset here first."
        )

    required_splits = ["train", "val"]
    missing = [split for split in required_splits if not (data_dir / split).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Missing dataset split folder(s): {', '.join(missing)}\n"
            f"Expected at least: {data_dir / 'train'} and {data_dir / 'val'}"
        )

    train_classes = [
        path.name for path in sorted((data_dir / "train").iterdir()) if path.is_dir()
    ]
    if len(train_classes) < 2:
        raise ValueError(
            "YOLO classification training needs at least two class folders inside "
            f"{data_dir / 'train'}."
        )

    train_count = count_images(data_dir / "train")
    val_count = count_images(data_dir / "val")
    if train_count == 0 or val_count == 0:
        raise ValueError(
            f"Dataset has no usable images. Found train={train_count}, val={val_count}."
        )

    print("Dataset ready")
    print(f"  Path        : {data_dir}")
    print(f"  Classes     : {len(train_classes)}")
    print(f"  Train images: {train_count}")
    print(f"  Val images  : {val_count}")
    if (data_dir / "test").is_dir():
        print(f"  Test images : {count_images(data_dir / 'test')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 fetal classifier")
    parser.add_argument("--data", default="dataset", help="YOLO classification dataset root")
    parser.add_argument("--model", default="yolov8s-cls.pt", help="Starting model weights")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=224, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--project", default="runs/classify", help="Output runs folder")
    parser.add_argument("--name", default="fetal_yolov8s", help="Run name")
    parser.add_argument("--copy-best", default="best.pt", help="Where to copy the best weights")
    parser.add_argument("--validate-only", action="store_true", help="Only check dataset/training setup")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data).resolve()
    validate_dataset(data_dir)

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Device      : {device}")
    if device == "cpu":
        print("Note        : CUDA is not available, so training will be slow on this machine.")

    if args.validate_only:
        return

    model = YOLO(args.model)
    results = model.train(
        data=str(data_dir),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=device,
        pretrained=True,
        patience=8,
        save=True,
        plots=True,
    )

    save_dir = Path(getattr(results, "save_dir", Path(args.project) / args.name))
    best_weights = save_dir / "weights" / "best.pt"
    if best_weights.exists() and args.copy_best:
        destination = Path(args.copy_best)
        shutil.copy2(best_weights, destination)
        print(f"Copied best weights to: {destination.resolve()}")
    else:
        print(f"Training finished. Best weights expected at: {best_weights}")


if __name__ == "__main__":
    main()
