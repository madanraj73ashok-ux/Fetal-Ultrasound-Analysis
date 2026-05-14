"""
Advanced YOLOv8 detection training entrypoint for fetal ultrasound analysis.

Default output:
    runs/detect/train/results.csv
    runs/detect/train/confusion_matrix.png
    runs/detect/train/PR_curve.png
    runs/detect/train/F1_curve.png
    runs/detect/train/weights/best.pt
    runs/detect/train/weights/last.pt

Usage:
    python train_model.py
    python train_model.py --data data.yaml --epochs 50
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
RUN_DIR = BASE_DIR / "runs" / "detect" / "train"
STATUS_FILE = BASE_DIR / "training_status.json"
LOG_FILE = BASE_DIR / "training.log"


class Tee:
    """Write training output to terminal and a log file continuously."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def auto_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(8, cpu_count - 1))


def gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {
            "device": "cpu",
            "cuda": False,
            "name": "CPU",
            "vram_gb": None,
        }

    props = torch.cuda.get_device_properties(0)
    return {
        "device": 0,
        "cuda": True,
        "name": props.name,
        "vram_gb": round(props.total_memory / (1024 ** 3), 2),
    }


def write_status(**updates) -> None:
    status = {}
    if STATUS_FILE.exists():
        try:
            status = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            status = {}
    status.update(updates)
    status["updated_at"] = datetime.now().isoformat(timespec="seconds")
    STATUS_FILE.write_text(json.dumps(status, indent=2), encoding="utf-8")


def validate_training_inputs(data_yaml: Path) -> None:
    dataset_root = BASE_DIR / "dataset"
    classification_like = (
        (dataset_root / "train").is_dir()
        and any(path.is_dir() for path in (dataset_root / "train").iterdir())
        and not (dataset_root / "images" / "train").is_dir()
        and not (dataset_root / "labels" / "train").is_dir()
    )
    if classification_like:
        raise ValueError(
            "Classification-format dataset detected at dataset/train/<class_name>/.\n"
            "Do not train YOLO detection on folder-labeled data.\n"
            "Run this instead:\n"
            "  python convert_classification_dataset.py --input dataset --output dataset_cls\n"
            "  python train_classifier.py --data dataset_cls"
        )
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"data.yaml not found: {data_yaml}\n"
            "Create a YOLO detection dataset first, then run check_dataset.py."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 fetal detector")
    parser.add_argument("--data", default="data.yaml", help="YOLO detection data.yaml")
    parser.add_argument("--model", default="yolov8s.pt", help="Starting YOLOv8 detection weights")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--cache", default="True", help="Use YOLO cache=True/False")
    parser.add_argument("--workers", default="auto", help="Worker count or 'auto'")
    parser.add_argument("--project", default=str(BASE_DIR / "runs" / "detect"))
    parser.add_argument("--name", default="train")
    parser.add_argument("--copy-best", default=str(BASE_DIR / "best.pt"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_yaml = (BASE_DIR / args.data).resolve() if not Path(args.data).is_absolute() else Path(args.data)
    log_handle = LOG_FILE.open("a", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_handle)
    sys.stderr = Tee(sys.__stderr__, log_handle)

    started_at = datetime.now().isoformat(timespec="seconds")
    info = gpu_info()
    workers = auto_workers() if str(args.workers).lower() == "auto" else int(args.workers)
    cache = str(args.cache).lower() in {"1", "true", "yes", "y"}

    write_status(
        running=True,
        completed=False,
        success=False,
        stopped=False,
        message="Training starting...",
        started_at=started_at,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        cache=cache,
        workers=workers,
        gpu=info,
        run_dir=str(RUN_DIR),
    )

    try:
        validate_training_inputs(data_yaml)
        print("=" * 72)
        print("YOLOv8 Fetal Ultrasound Detection Training")
        print("=" * 72)
        print(f"Data       : {data_yaml}")
        print(f"Model      : {args.model}")
        print(f"Device     : {info['name']}" + (f" ({info['vram_gb']} GB)" if info["cuda"] else ""))
        print(f"Epochs     : {args.epochs}")
        print(f"Image size : {args.imgsz}")
        print(f"Batch      : {args.batch}")
        print(f"Workers    : {workers}")
        print(f"Cache      : {cache}")
        print(f"Run dir    : {RUN_DIR}")
        print("=" * 72)

        model = YOLO(args.model)
        write_status(message="Training running...", running=True)
        results = model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            cache=cache,
            workers=workers,
            project=args.project,
            name=args.name,
            exist_ok=True,
            device=info["device"],
            plots=True,
            save=True,
            verbose=True,
        )

        save_dir = Path(getattr(results, "save_dir", RUN_DIR))
        best_weights = save_dir / "weights" / "best.pt"
        last_weights = save_dir / "weights" / "last.pt"
        copied_to = None
        if best_weights.exists() and args.copy_best:
            destination = Path(args.copy_best)
            shutil.copy2(best_weights, destination)
            copied_to = str(destination.resolve())
            print(f"Copied best model to prediction path: {copied_to}")

        write_status(
            running=False,
            completed=True,
            success=True,
            stopped=False,
            message="Training completed successfully.",
            completed_at=datetime.now().isoformat(timespec="seconds"),
            run_dir=str(save_dir),
            best_model=str(best_weights) if best_weights.exists() else None,
            last_model=str(last_weights) if last_weights.exists() else None,
            copied_to=copied_to,
        )
        print("Training completed successfully.")
        return 0
    except KeyboardInterrupt:
        write_status(
            running=False,
            completed=True,
            success=False,
            stopped=True,
            message="Training stopped by user.",
            completed_at=datetime.now().isoformat(timespec="seconds"),
        )
        print("Training stopped by user.")
        return 130
    except Exception as exc:
        write_status(
            running=False,
            completed=True,
            success=False,
            stopped=False,
            message=f"Training failed: {exc}",
            completed_at=datetime.now().isoformat(timespec="seconds"),
        )
        print(f"Training failed: {exc}")
        return 1
    finally:
        time.sleep(0.2)
        log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
