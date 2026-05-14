"""
Create a synthetic sample fetal ultrasound dataset for YOLOv8 training demo.

Generates both detection (dataset/) and classification (dataset_cls/) formats
with ultrasound-like grayscale images for 9 fetal anatomy classes.

Usage:
    python create_sample_dataset.py
"""

from __future__ import annotations

import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

BASE_DIR = Path(__file__).resolve().parent

# ── Classes matching the project's expected classes ──
CLASSES = [
    "Fetal abdomen",
    "Fetal brain",
    "Fetal femur",
    "Fetal thorax",
    "Maternal cervix",
    "Trans-cerebellum",
    "Trans-thalamic",
    "Trans-ventricular",
    "Other",
]

# Folder-safe names for classification dataset
FOLDER_NAMES = {
    "Fetal abdomen": "fetal_abdomen",
    "Fetal brain": "fetal_brain",
    "Fetal femur": "fetal_femur",
    "Fetal thorax": "fetal_thorax",
    "Maternal cervix": "maternal_cervix",
    "Trans-cerebellum": "trans_cerebellum",
    "Trans-thalamic": "trans_thalamic",
    "Trans-ventricular": "trans_ventricular",
    "Other": "other",
}

# Per-class visual parameters to make each class look distinct
CLASS_STYLES = {
    "Fetal abdomen": {"shape": "ellipse", "base_gray": 35, "inner": True, "inner_shape": "circle"},
    "Fetal brain": {"shape": "ellipse", "base_gray": 30, "inner": True, "inner_shape": "wavy"},
    "Fetal femur": {"shape": "line", "base_gray": 25, "inner": False, "inner_shape": None},
    "Fetal thorax": {"shape": "ellipse", "base_gray": 32, "inner": True, "inner_shape": "ribs"},
    "Maternal cervix": {"shape": "funnel", "base_gray": 40, "inner": False, "inner_shape": None},
    "Trans-cerebellum": {"shape": "ellipse", "base_gray": 28, "inner": True, "inner_shape": "dual_lobe"},
    "Trans-thalamic": {"shape": "ellipse", "base_gray": 30, "inner": True, "inner_shape": "midline"},
    "Trans-ventricular": {"shape": "ellipse", "base_gray": 30, "inner": True, "inner_shape": "ventricle"},
    "Other": {"shape": "random", "base_gray": 45, "inner": False, "inner_shape": None},
}

IMG_SIZE = 224
TRAIN_PER_CLASS = 25
VAL_PER_CLASS = 8
TEST_PER_CLASS = 5


def add_ultrasound_noise(img_array: np.ndarray, intensity: float = 0.15) -> np.ndarray:
    """Add speckle noise typical of ultrasound images."""
    noise = np.random.rayleigh(intensity * 255, img_array.shape)
    noisy = img_array.astype(np.float64) + noise * 0.3
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def add_scan_lines(draw: ImageDraw.Draw, width: int, height: int) -> None:
    """Add faint horizontal scan lines like real ultrasound."""
    for y in range(0, height, random.randint(3, 6)):
        alpha = random.randint(5, 20)
        draw.line([(0, y), (width, y)], fill=alpha, width=1)


def add_sector_mask(img: Image.Image) -> Image.Image:
    """Add a dark sector/cone mask to simulate ultrasound probe field of view."""
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    cx, cy = w // 2, -h // 6
    radius = int(h * 1.4)
    angle_spread = random.randint(55, 75)
    bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
    draw.pieslice(bbox, 90 - angle_spread, 90 + angle_spread, fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=8))
    result = Image.composite(img, Image.new("L", (w, h), 0), mask)
    return result


def draw_structure(draw: ImageDraw.Draw, style: dict, cx: int, cy: int,
                   rx: int, ry: int, brightness: int) -> tuple:
    """Draw the main anatomical structure and return bounding box (x1,y1,x2,y2)."""
    shape = style["shape"]
    x1, y1, x2, y2 = cx - rx, cy - ry, cx + rx, cy + ry

    if shape == "ellipse":
        draw.ellipse([x1, y1, x2, y2], fill=brightness, outline=brightness + 30)
        if style["inner"]:
            draw_inner(draw, style["inner_shape"], cx, cy, rx, ry, brightness)

    elif shape == "line":
        # Femur = thick bright line with rounded ends
        angle = random.uniform(-0.4, 0.4)
        length = rx * 2
        lx1 = cx - int(length / 2 * math.cos(angle))
        ly1 = cy - int(length / 2 * math.sin(angle))
        lx2 = cx + int(length / 2 * math.cos(angle))
        ly2 = cy + int(length / 2 * math.sin(angle))
        draw.line([(lx1, ly1), (lx2, ly2)], fill=brightness + 50, width=random.randint(6, 12))
        draw.ellipse([lx1 - 8, ly1 - 8, lx1 + 8, ly1 + 8], fill=brightness + 40)
        draw.ellipse([lx2 - 8, ly2 - 8, lx2 + 8, ly2 + 8], fill=brightness + 40)
        pad = 12
        x1 = min(lx1, lx2) - pad
        y1 = min(ly1, ly2) - pad
        x2 = max(lx1, lx2) + pad
        y2 = max(ly1, ly2) + pad

    elif shape == "funnel":
        # Cervix = funnel/trapezoid shape
        top_w = rx // 2
        draw.polygon([
            (cx - top_w, cy - ry),
            (cx + top_w, cy - ry),
            (cx + rx, cy + ry),
            (cx - rx, cy + ry),
        ], fill=brightness, outline=brightness + 20)

    elif shape == "random":
        # Other = random blobs
        for _ in range(random.randint(2, 5)):
            ox = cx + random.randint(-rx, rx)
            oy = cy + random.randint(-ry, ry)
            r = random.randint(8, 25)
            draw.ellipse([ox - r, oy - r, ox + r, oy + r], fill=brightness + random.randint(-10, 30))

    return (max(0, x1), max(0, y1), min(IMG_SIZE, x2), min(IMG_SIZE, y2))


def draw_inner(draw: ImageDraw.Draw, inner_type: str, cx: int, cy: int,
               rx: int, ry: int, brightness: int) -> None:
    """Draw inner detail structures for different anatomical classes."""
    if inner_type == "circle":
        # Abdomen inner circle (stomach bubble)
        ir = min(rx, ry) // 3
        draw.ellipse([cx - ir, cy - ir, cx + ir, cy + ir], fill=brightness - 15,
                     outline=brightness + 20)

    elif inner_type == "wavy":
        # Brain folds
        for i in range(3):
            offset = (i - 1) * (ry // 3)
            draw.arc([cx - rx + 10, cy + offset - 8, cx + rx - 10, cy + offset + 8],
                     0, 180, fill=brightness + 25, width=2)

    elif inner_type == "ribs":
        # Thorax rib lines
        for i in range(4):
            yy = cy - ry // 2 + i * (ry // 3)
            draw.line([(cx - rx + 15, yy), (cx + rx - 15, yy)],
                      fill=brightness + 35, width=2)

    elif inner_type == "dual_lobe":
        # Cerebellum = two lobes
        lobe_r = rx // 3
        draw.ellipse([cx - rx // 2 - lobe_r, cy - lobe_r,
                      cx - rx // 2 + lobe_r, cy + lobe_r],
                     fill=brightness + 15, outline=brightness + 30)
        draw.ellipse([cx + rx // 2 - lobe_r, cy - lobe_r,
                      cx + rx // 2 + lobe_r, cy + lobe_r],
                     fill=brightness + 15, outline=brightness + 30)

    elif inner_type == "midline":
        # Thalamic midline
        draw.line([(cx, cy - ry + 5), (cx, cy + ry - 5)],
                  fill=brightness + 40, width=2)
        draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=brightness + 20)

    elif inner_type == "ventricle":
        # Ventricular = bright CSF-filled space
        vw = rx // 2
        vh = ry // 3
        draw.ellipse([cx - vw, cy - vh, cx + vw, cy + vh],
                     fill=brightness + 30, outline=brightness + 45)


def generate_image(class_name: str, seed: int) -> tuple:
    """Generate one synthetic ultrasound image and return (PIL Image, bbox_normalized)."""
    random.seed(seed)
    np.random.seed(seed % (2**31))

    style = CLASS_STYLES[class_name]
    base_gray = style["base_gray"] + random.randint(-8, 8)

    # Create base dark image with gradient
    arr = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    for y in range(IMG_SIZE):
        for x in range(IMG_SIZE):
            dist = math.sqrt((x - IMG_SIZE / 2) ** 2 + (y - IMG_SIZE / 2) ** 2)
            fade = max(0, base_gray - int(dist * 0.12))
            arr[y, x] = fade + random.randint(0, 6)

    img = Image.fromarray(arr, mode="L")
    draw = ImageDraw.Draw(img)

    # Add scan lines
    add_scan_lines(draw, IMG_SIZE, IMG_SIZE)

    # Main structure position with slight randomization
    cx = IMG_SIZE // 2 + random.randint(-20, 20)
    cy = IMG_SIZE // 2 + random.randint(-15, 15)
    rx = random.randint(35, 55)
    ry = random.randint(30, 50)
    brightness = base_gray + random.randint(40, 80)

    # Draw the anatomical structure
    x1, y1, x2, y2 = draw_structure(draw, style, cx, cy, rx, ry, brightness)

    # Add ultrasound artifacts
    arr = np.array(img)
    arr = add_ultrasound_noise(arr, intensity=random.uniform(0.1, 0.2))
    img = Image.fromarray(arr, mode="L")

    # Apply sector mask
    img = add_sector_mask(img)

    # Gaussian blur for realism
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

    # Normalize bbox to 0-1 for YOLO format
    bbox_x_center = ((x1 + x2) / 2) / IMG_SIZE
    bbox_y_center = ((y1 + y2) / 2) / IMG_SIZE
    bbox_w = (x2 - x1) / IMG_SIZE
    bbox_h = (y2 - y1) / IMG_SIZE

    # Clamp
    bbox_x_center = max(0.01, min(0.99, bbox_x_center))
    bbox_y_center = max(0.01, min(0.99, bbox_y_center))
    bbox_w = max(0.05, min(0.98, bbox_w))
    bbox_h = max(0.05, min(0.98, bbox_h))

    return img, (bbox_x_center, bbox_y_center, bbox_w, bbox_h)


def create_data_yaml(dataset_dir: Path) -> None:
    """Create YOLO data.yaml for detection training."""
    yaml_content = f"""# Fetal Ultrasound Detection Dataset
# Auto-generated sample dataset for demo/training

path: {dataset_dir.resolve()}
train: images/train
val: images/val
test: images/test

nc: {len(CLASSES)}
names:
"""
    for i, name in enumerate(CLASSES):
        yaml_content += f"  {i}: {name}\n"

    (dataset_dir / "data.yaml").write_text(yaml_content, encoding="utf-8")

    # Also copy to project root for convenience (train_model.py looks for data.yaml in BASE_DIR)
    root_yaml = BASE_DIR / "data.yaml"
    root_content = yaml_content.replace(f"path: {dataset_dir.resolve()}", f"path: {dataset_dir.resolve()}")
    root_yaml.write_text(root_content, encoding="utf-8")
    print(f"  Created {dataset_dir / 'data.yaml'}")
    print(f"  Created {root_yaml} (project root copy)")


def main() -> int:
    print("=" * 60)
    print("Creating Sample Fetal Ultrasound Dataset")
    print("=" * 60)

    det_dir = BASE_DIR / "dataset"
    cls_dir = BASE_DIR / "dataset_cls"

    # Clean up existing
    for d in (det_dir, cls_dir):
        if d.exists():
            print(f"  Removing existing {d.name}/...")
            shutil.rmtree(d)

    # Create directory structure
    for split in ("train", "val", "test"):
        (det_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (det_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for cls_name in CLASSES:
            folder = FOLDER_NAMES[cls_name]
            (cls_dir / split / folder).mkdir(parents=True, exist_ok=True)

    # Generate images
    splits = {"train": TRAIN_PER_CLASS, "val": VAL_PER_CLASS, "test": TEST_PER_CLASS}
    total = 0

    for class_id, class_name in enumerate(CLASSES):
        folder = FOLDER_NAMES[class_name]
        print(f"\n  [{class_id}] {class_name}:")

        for split, count in splits.items():
            for i in range(count):
                seed = hash(f"{class_name}_{split}_{i}") & 0xFFFFFFFF
                img, bbox = generate_image(class_name, seed)

                filename = f"{folder}_{split}_{i:03d}.jpg"

                # Save detection format
                img_path = det_dir / "images" / split / filename
                lbl_path = det_dir / "labels" / split / f"{folder}_{split}_{i:03d}.txt"
                img.convert("RGB").save(img_path, "JPEG", quality=90)
                lbl_path.write_text(
                    f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n",
                    encoding="utf-8",
                )

                # Save classification format
                cls_path = cls_dir / split / folder / filename
                img.convert("RGB").save(cls_path, "JPEG", quality=90)

                total += 1

            print(f"    {split}: {count} images")

    # Create data.yaml
    print()
    create_data_yaml(det_dir)

    print(f"\n{'=' * 60}")
    print(f"  Total images generated: {total}")
    print(f"  Detection dataset:      {det_dir}")
    print(f"  Classification dataset: {cls_dir}")
    print(f"  data.yaml:              {det_dir / 'data.yaml'}")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
