"""
Validate a YOLOv8 detection dataset before training.

Expected layout:
    dataset/
      images/train
      images/val
      images/test
      labels/train
      labels/val
      labels/test
      data.yaml

Usage:
    python check_dataset.py
    python check_dataset.py --dataset dataset
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
EXPECTED_CLASSES = {
    "Fetal abdomen",
    "Fetal brain",
    "Fetal femur",
    "Fetal thorax",
    "Maternal cervix",
    "Trans-cerebellum",
    "Trans-thalamic",
    "Trans-ventricular",
    "Other",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check YOLOv8 detection dataset")
    parser.add_argument(
        "--dataset",
        default="dataset",
        help="Dataset root folder containing images, labels, and data.yaml",
    )
    return parser.parse_args()


def read_yaml(path: Path) -> tuple[dict, list[str]]:
    errors: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception as exc:
        return {}, [f"{path}: could not read YAML: {exc}"]

    if not isinstance(data, dict):
        errors.append(f"{path}: YAML root must be a mapping/object.")
        return {}, errors
    return data, errors


def normalize_names(raw_names) -> list[str]:
    if isinstance(raw_names, dict):
        return [str(raw_names[key]) for key in sorted(raw_names, key=lambda item: int(item))]
    if isinstance(raw_names, list):
        return [str(name) for name in raw_names]
    return []


def resolve_yaml_path(dataset_root: Path, yaml_root_value, split_value) -> Path | None:
    if not split_value:
        return None

    split_path = Path(str(split_value))
    if split_path.is_absolute():
        return split_path

    if yaml_root_value:
        yaml_root = Path(str(yaml_root_value))
        if not yaml_root.is_absolute():
            yaml_root = (dataset_root / yaml_root).resolve()
        return (yaml_root / split_path).resolve()

    return (dataset_root / split_path).resolve()


def validate_structure(dataset_root: Path) -> list[str]:
    errors: list[str] = []
    required = [dataset_root / "data.yaml"]
    for split in SPLITS:
        required.append(dataset_root / "images" / split)
        required.append(dataset_root / "labels" / split)

    for path in required:
        if not path.exists():
            errors.append(f"Missing required path: {path}")
        elif path.suffix != ".yaml" and not path.is_dir():
            errors.append(f"Required path is not a directory: {path}")
    return errors


def validate_yaml(dataset_root: Path, class_count: int) -> tuple[list[str], list[str]]:
    yaml_path = dataset_root / "data.yaml"
    data, errors = read_yaml(yaml_path)
    if errors:
        return [], errors

    names = normalize_names(data.get("names"))
    if not names:
        errors.append(f"{yaml_path}: missing or invalid 'names' list/dict.")

    if "nc" in data and int(data["nc"]) != len(names):
        errors.append(f"{yaml_path}: nc={data['nc']} does not match names count={len(names)}.")

    if class_count and names and class_count > len(names):
        errors.append(
            f"{yaml_path}: labels reference class id {class_count - 1}, "
            f"but names only has {len(names)} classes."
        )

    name_set = set(names)
    if names and name_set != EXPECTED_CLASSES:
        missing = sorted(EXPECTED_CLASSES - name_set)
        extra = sorted(name_set - EXPECTED_CLASSES)
        if missing:
            errors.append(f"{yaml_path}: missing expected classes: {', '.join(missing)}")
        if extra:
            errors.append(f"{yaml_path}: unexpected classes: {', '.join(extra)}")

    for split in SPLITS:
        if split not in data:
            errors.append(f"{yaml_path}: missing '{split}' entry.")
            continue
        resolved = resolve_yaml_path(dataset_root, data.get("path"), data.get(split))
        expected = (dataset_root / "images" / split).resolve()
        if resolved != expected:
            errors.append(
                f"{yaml_path}: {split} points to '{resolved}', expected '{expected}'."
            )

    return names, errors


def validate_label_file(label_path: Path, class_limit: int | None) -> tuple[list[int], list[str]]:
    class_ids: list[int] = []
    errors: list[str] = []

    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        return class_ids, [f"{label_path}: could not read label file: {exc}"]

    for line_number, line in enumerate(lines, start=1):
        text = line.strip()
        if not text:
            continue
        parts = text.split()
        if len(parts) != 5:
            errors.append(
                f"{label_path}:{line_number}: expected 5 values "
                "class_id x_center y_center width height, got "
                f"{len(parts)}."
            )
            continue

        try:
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"{label_path}:{line_number}: non-numeric label value.")
            continue

        if class_id < 0:
            errors.append(f"{label_path}:{line_number}: class_id cannot be negative.")
        elif class_limit is not None and class_id >= class_limit:
            errors.append(
                f"{label_path}:{line_number}: class_id {class_id} is outside "
                f"valid range 0-{class_limit - 1}."
            )
        else:
            class_ids.append(class_id)

        for name, value in zip(("x_center", "y_center", "width", "height"), values):
            if not 0 <= value <= 1:
                errors.append(
                    f"{label_path}:{line_number}: {name}={value} must be between 0 and 1."
                )

        if values[2] <= 0 or values[3] <= 0:
            errors.append(
                f"{label_path}:{line_number}: width and height must be greater than 0."
            )

    return class_ids, errors


def collect_split(dataset_root: Path, split: str, class_limit: int | None) -> dict:
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    images = sorted(
        path for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ) if image_dir.exists() else []

    missing_labels: list[str] = []
    invalid_labels: list[str] = []
    distribution: Counter[int] = Counter()

    for image_path in images:
        relative = image_path.relative_to(image_dir)
        label_path = (label_dir / relative).with_suffix(".txt")
        if not label_path.exists():
            missing_labels.append(str(label_path))
            continue

        class_ids, errors = validate_label_file(label_path, class_limit)
        distribution.update(class_ids)
        invalid_labels.extend(errors)

    return {
        "images": len(images),
        "missing_labels": missing_labels,
        "invalid_labels": invalid_labels,
        "distribution": distribution,
    }


def print_summary(results: dict, class_names: list[str], structure_errors: list[str], yaml_errors: list[str]) -> int:
    all_missing = []
    all_invalid = []
    total_distribution: Counter[int] = Counter()

    print("\nDataset Summary")
    print("=" * 60)
    for split in SPLITS:
        split_result = results.get(split, {})
        print(f"{split} image count: {split_result.get('images', 0)}")
        all_missing.extend(split_result.get("missing_labels", []))
        all_invalid.extend(split_result.get("invalid_labels", []))
        total_distribution.update(split_result.get("distribution", Counter()))

    print(f"missing labels: {len(all_missing)}")
    print(f"invalid labels: {len(all_invalid)}")

    print("\nClass Distribution")
    print("=" * 60)
    if class_names:
        for class_id, class_name in enumerate(class_names):
            print(f"{class_id}: {class_name}: {total_distribution.get(class_id, 0)}")
    elif total_distribution:
        for class_id in sorted(total_distribution):
            print(f"{class_id}: {total_distribution[class_id]}")
    else:
        print("No labeled objects found.")

    errors = structure_errors + yaml_errors + all_missing + all_invalid
    if errors:
        print("\nProblems")
        print("=" * 60)
        for problem in errors:
            print(f"- {problem}")
        return 1

    print("\nDataset is ready for YOLOv8 training.")
    return 0


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset).resolve()

    structure_errors = validate_structure(dataset_root)
    names, yaml_errors = validate_yaml(dataset_root, 0) if (dataset_root / "data.yaml").exists() else ([], [])
    class_limit = len(names) if names else None

    results = {}
    for split in SPLITS:
        results[split] = collect_split(dataset_root, split, class_limit)

    return print_summary(results, names, structure_errors, yaml_errors)


if __name__ == "__main__":
    raise SystemExit(main())
