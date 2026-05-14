"""
Inspect and convert fetal ultrasound classification datasets for YOLOv8-cls.

This script does NOT create detection labels or fake bounding boxes.

Input classification layout:
    dataset/
      train/fetal_femur/*.jpg
      val/fetal_femur/*.jpg

Output YOLO classification layout:
    dataset_cls/
      train/fetal_femur/*.jpg
      val/fetal_femur/*.jpg
      test/fetal_femur/*.jpg  (if present)

Usage:
    python convert_classification_dataset.py
    python convert_classification_dataset.py --input dataset --output dataset_cls
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
REQUIRED_CLASSES = (
    "fetal_femur",
    "fetal_abdomen",
    "fetal_brain",
    "fetal_thorax",
    "fetal_skull",
    "fetal_spine",
)
CLASS_ALIASES = {
    "femur": "fetal_femur",
    "fetal femur": "fetal_femur",
    "fetal-femur": "fetal_femur",
    "abdomen": "fetal_abdomen",
    "fetal abdomen": "fetal_abdomen",
    "fetal-abdomen": "fetal_abdomen",
    "brain": "fetal_brain",
    "fetal brain": "fetal_brain",
    "fetal-brain": "fetal_brain",
    "thorax": "fetal_thorax",
    "fetal thorax": "fetal_thorax",
    "fetal-thorax": "fetal_thorax",
    "skull": "fetal_skull",
    "fetal skull": "fetal_skull",
    "fetal-skull": "fetal_skull",
    "spine": "fetal_spine",
    "fetal spine": "fetal_spine",
    "fetal-spine": "fetal_spine",
}


def normalize_class_name(name: str) -> str:
    cleaned = name.strip().lower().replace("-", "_").replace(" ", "_")
    return CLASS_ALIASES.get(name.strip().lower(), CLASS_ALIASES.get(cleaned, cleaned))


def image_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(
        file for file in path.rglob("*")
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )


def detect_dataset_type(root: Path) -> str:
    if (root / "images" / "train").is_dir() and (root / "labels" / "train").is_dir():
        return "detection"
    if (root / "train").is_dir() and any(path.is_dir() for path in (root / "train").iterdir()):
        return "classification"
    return "missing_or_unknown"


def copy_split(input_root: Path, output_root: Path, split: str) -> Counter:
    counts: Counter[str] = Counter()
    split_root = input_root / split
    if not split_root.exists():
        return counts

    for class_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
        class_name = normalize_class_name(class_dir.name)
        destination_dir = output_root / split / class_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        for source in image_files(class_dir):
            destination = destination_dir / source.name
            if destination.exists():
                destination = destination_dir / f"{source.stem}_{abs(hash(source))}{source.suffix}"
            shutil.copy2(source, destination)
            counts[class_name] += 1
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert classification dataset for YOLOv8-cls")
    parser.add_argument("--input", default="dataset", help="Input classification dataset root")
    parser.add_argument("--output", default="dataset_cls", help="Output YOLO classification dataset root")
    parser.add_argument("--overwrite", action="store_true", help="Delete output folder before converting")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    dataset_type = detect_dataset_type(input_root)
    print(f"Detected dataset type: {dataset_type}")

    if dataset_type == "missing_or_unknown":
        print(f"Dataset not found or unknown format: {input_root}")
        return 1
    if dataset_type == "detection":
        print("This is already a YOLO detection dataset. Do not convert it to classification.")
        return 0

    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    total: Counter[str] = Counter()
    for split in SPLITS:
        counts = copy_split(input_root, output_root, split)
        total.update(counts)
        print(f"{split}: {sum(counts.values())} images")
        for class_name in REQUIRED_CLASSES:
            print(f"  {class_name}: {counts.get(class_name, 0)}")

    missing = [class_name for class_name in REQUIRED_CLASSES if total.get(class_name, 0) == 0]
    if missing:
        print("Warning: these required classes were not found:")
        for class_name in missing:
            print(f"  - {class_name}")

    print(f"YOLO classification dataset saved to: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
