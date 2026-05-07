# ============================================================
# inference.py — Fetal Image Analysis Inference Pipeline
# ============================================================
# Project : Fetal Image Analysis Using Deep Learning
# Author  : TB Solutions
# Phase   : 3 — Inference Pipeline
# Model   : YOLOv8s Classification (BCNatal dataset, 9 classes)
#
# Usage:
#   from inference import load_model, predict, preprocess,
#                         annotate_image, batch_predict
#
#   model = load_model("best.pt")
#   result = predict("ultrasound.png", model)
#   print(result)
# ============================================================

import os
import io
import time
import base64
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore")

# ---- Lazy imports (loaded when needed) ----
_torch = None
_YOLO = None


def _import_torch():
    """Lazy import for torch to speed up module load."""
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _import_yolo():
    """Lazy import for ultralytics YOLO."""
    global _YOLO
    if _YOLO is None:
        from ultralytics import YOLO
        _YOLO = YOLO
    return _YOLO


# ============================================================
# CLINICAL METADATA
# ============================================================
# Maps each anatomical class to a clinical description.
# These are used in the prediction output for context.

CLINICAL_NOTES = {
    "Fetal abdomen": (
        "Fetal abdominal plane detected. This view is used to measure "
        "abdominal circumference (AC), a key biometric for estimating "
        "fetal weight and assessing growth."
    ),
    "Fetal brain": (
        "Fetal brain plane detected. Brain imaging is critical for "
        "evaluating structural development and identifying abnormalities "
        "such as ventriculomegaly or neural tube defects."
    ),
    "Fetal femur": (
        "Fetal femur plane detected. Femur length (FL) measurement is a "
        "standard biometric used for gestational age estimation and growth "
        "monitoring."
    ),
    "Fetal thorax": (
        "Fetal thorax plane detected. Thoracic imaging evaluates lung "
        "development, cardiac structures, and detects conditions like "
        "pleural effusion or diaphragmatic hernia."
    ),
    "Maternal cervix": (
        "Maternal cervix plane detected. Cervical length measurement "
        "is important for assessing the risk of preterm birth."
    ),
    "Trans-cerebellum": (
        "Trans-cerebellar plane detected. This view shows the posterior "
        "fossa and is used to measure transcerebellar diameter (TCD) "
        "for gestational age assessment."
    ),
    "Trans-thalamic": (
        "Trans-thalamic plane detected. This is the standard view for "
        "measuring biparietal diameter (BPD) and head circumference (HC), "
        "key fetal biometrics."
    ),
    "Trans-ventricular": (
        "Trans-ventricular plane detected. This view is used to assess "
        "lateral ventricle width, important for detecting ventriculomegaly "
        "and other brain anomalies."
    ),
    "Other": (
        "Non-standard or unclassified fetal ultrasound plane. "
        "The image does not clearly match any of the 8 standard "
        "anatomical views. Manual review recommended."
    ),
}

# Default class presumed normal (all known anatomical planes)
NORMAL_CLASSES = {
    "Fetal abdomen", "Fetal brain", "Fetal femur", "Fetal thorax",
    "Maternal cervix", "Trans-cerebellum", "Trans-thalamic",
    "Trans-ventricular",
}

# Model version tag
MODEL_VERSION = "fetal_yolov8s_v1"


# ============================================================
# 1. MODEL LOADING
# ============================================================

def load_model(model_path, device=None):
    """
    Load a YOLOv8 classification model from a .pt file.

    Args:
        model_path (str): Path to the best.pt model file.
        device (str|int|None): Device to load on.
            None = auto-detect (GPU if available, else CPU).

    Returns:
        model: Loaded YOLO model, warmed up and ready.

    Raises:
        FileNotFoundError: If model_path does not exist.
        RuntimeError: If model fails to load.
    """
    YOLO = _import_yolo()
    torch = _import_torch()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Auto-detect device
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    # Load model
    model = YOLO(model_path)

    # Warm up with a dummy image
    dummy = np.zeros((224, 224, 3), dtype=np.uint8)
    model.predict(source=dummy, imgsz=224, device=device, verbose=False)

    return model


def get_model_info(model):
    """
    Extract model metadata.

    Returns:
        dict with task, classes, input_size, device info.
    """
    torch = _import_torch()

    class_names = model.names  # dict {idx: name}
    device_str = "GPU" if torch.cuda.is_available() else "CPU"

    return {
        "task": "classification",
        "classes": list(class_names.values()),
        "num_classes": len(class_names),
        "input_size": 224,
        "device": device_str,
        "model_version": MODEL_VERSION,
    }


# ============================================================
# 2. PREPROCESSING
# ============================================================

def preprocess(image_input, target_size=224):
    """
    Preprocess an image for YOLOv8 classification inference.

    Accepts multiple input types for flexibility:
        - str: file path to an image
        - PIL.Image.Image: a PIL image object
        - np.ndarray: a NumPy array (H x W x C, BGR or RGB)
        - str (base64): a base64-encoded image string

    Processing steps:
        1. Load/decode the image
        2. Convert to RGB
        3. Apply CLAHE contrast enhancement (improves ultrasound)
        4. Resize to target_size x target_size
        5. Normalize pixel values to [0, 1]

    Args:
        image_input: Image in any supported format.
        target_size (int): Output size (square).

    Returns:
        tuple: (preprocessed_np_array, original_pil_image)
            - preprocessed: np.ndarray, shape (H, W, 3), float32, [0,1]
            - original: PIL.Image in RGB
    """
    # ---- Step 1: Load image based on input type ----
    if isinstance(image_input, str):
        # Could be a file path or a base64 string
        if os.path.exists(image_input):
            pil_img = Image.open(image_input)
        else:
            # Try decoding as base64
            try:
                # Strip data URI prefix if present
                if "," in image_input:
                    image_input = image_input.split(",", 1)[1]
                img_bytes = base64.b64decode(image_input)
                pil_img = Image.open(io.BytesIO(img_bytes))
            except Exception as e:
                raise ValueError(
                    f"Input string is not a valid file path or base64: {e}"
                )
    elif isinstance(image_input, Image.Image):
        pil_img = image_input
    elif isinstance(image_input, np.ndarray):
        # Assume BGR if 3-channel (OpenCV convention)
        if len(image_input.shape) == 3 and image_input.shape[2] == 3:
            rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        elif len(image_input.shape) == 2:
            rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        else:
            rgb = image_input
        pil_img = Image.fromarray(rgb.astype(np.uint8))
    elif isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input))
    else:
        raise TypeError(
            f"Unsupported input type: {type(image_input)}. "
            "Expected str (path/base64), PIL.Image, np.ndarray, or bytes."
        )

    # ---- Step 2: Convert to RGB ----
    original_pil = pil_img.convert("RGB")

    # ---- Step 3: CLAHE enhancement ----
    img_np = np.array(original_pil)
    # Convert to LAB for CLAHE on luminance channel
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # ---- Step 4: Resize ----
    resized = cv2.resize(
        enhanced, (target_size, target_size),
        interpolation=cv2.INTER_AREA
    )

    # ---- Step 5: Normalize to [0, 1] ----
    normalized = resized.astype(np.float32) / 255.0

    return normalized, original_pil


# ============================================================
# 3. CORE INFERENCE
# ============================================================

def predict(image_input, model, top_k=3, conf_threshold=0.5,
            target_size=224, device=None):
    """
    Run classification inference on a single image.

    Args:
        image_input: Image in any format accepted by preprocess().
        model: Loaded YOLO model (from load_model()).
        top_k (int): Number of top predictions to return.
        conf_threshold (float): Below this confidence, flag as uncertain.
        target_size (int): Inference input size.
        device: Device override (None = auto).

    Returns:
        dict: Structured prediction result with keys:
            top_prediction, confidence, top_k_predictions,
            status, clinical_note, processing_time_ms,
            image_size, model_version
    """
    torch = _import_torch()

    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    start_time = time.time()

    # ---- Preprocess ----
    preprocessed, original_pil = preprocess(image_input, target_size)

    # ---- Run YOLO inference ----
    # YOLO accepts PIL, numpy, or path — we pass the preprocessed array
    # But for best results, pass original and let YOLO handle its own resize
    results = model.predict(
        source=original_pil,
        imgsz=target_size,
        device=device,
        verbose=False,
    )

    result = results[0]
    probs = result.probs
    class_names = model.names  # {idx: name}

    # ---- Extract top-K predictions ----
    probs_np = probs.data.cpu().numpy()
    top_k_actual = min(top_k, len(probs_np))
    top_k_indices = np.argsort(probs_np)[::-1][:top_k_actual]

    top_k_predictions = []
    for idx in top_k_indices:
        top_k_predictions.append({
            "class": class_names[int(idx)],
            "confidence": round(float(probs_np[idx]), 4),
        })

    # Top prediction
    top_pred = top_k_predictions[0]
    top_class = top_pred["class"]
    top_conf = top_pred["confidence"]

    # ---- Determine Normal/Abnormal status ----
    if top_conf < conf_threshold:
        status = "Uncertain — Low Confidence"
    elif top_class in NORMAL_CLASSES:
        status = "Normal"
    elif top_class == "Other":
        status = "Review Needed — Non-standard View"
    else:
        status = "Normal"

    # ---- Clinical note ----
    clinical_note = CLINICAL_NOTES.get(
        top_class,
        f"Detected anatomical plane: {top_class}. "
        "Consult a medical professional for interpretation."
    )

    # ---- Processing time ----
    elapsed_ms = (time.time() - start_time) * 1000

    return {
        "top_prediction": top_class,
        "confidence": round(top_conf, 4),
        "top_k_predictions": top_k_predictions,
        "status": status,
        "clinical_note": clinical_note,
        "processing_time_ms": round(elapsed_ms, 1),
        "image_size": [target_size, target_size],
        "model_version": MODEL_VERSION,
    }


# ============================================================
# 4. ANNOTATED OUTPUT GENERATOR
# ============================================================

def annotate_image(original_image, prediction_dict):
    """
    Create a visually annotated version of the ultrasound image.

    Draws:
        - Prediction label (top-left)
        - Confidence bar (horizontal, teal fill)
        - Status badge with colored border (green=Normal, red=Abnormal)
        - Top-3 predictions overlay
        - TB Solutions watermark (bottom-right, subtle)

    Args:
        original_image: PIL.Image or path to image.
        prediction_dict: dict output from predict().

    Returns:
        PIL.Image: Annotated image.
    """
    # Load image if needed
    if isinstance(original_image, str):
        img = Image.open(original_image).convert("RGB")
    elif isinstance(original_image, np.ndarray):
        img = Image.fromarray(
            cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        )
    elif isinstance(original_image, Image.Image):
        img = original_image.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(original_image)}")

    # Resize to standard display size
    display_size = (448, 448)
    img = img.resize(display_size, Image.LANCZOS)

    draw = ImageDraw.Draw(img)
    W, H = display_size

    # ---- Try to load a font (fall back to default) ----
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_med   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except (IOError, OSError):
        try:
            font_large = ImageFont.truetype("arial.ttf", 18)
            font_med   = ImageFont.truetype("arial.ttf", 14)
            font_small = ImageFont.truetype("arial.ttf", 11)
        except (IOError, OSError):
            font_large = ImageFont.load_default()
            font_med   = ImageFont.load_default()
            font_small = ImageFont.load_default()

    top_pred = prediction_dict["top_prediction"]
    confidence = prediction_dict["confidence"]
    status = prediction_dict["status"]
    top_k = prediction_dict["top_k_predictions"]

    # ---- Colors ----
    TEAL = (13, 110, 110)        # #0D6E6E
    TEAL_LIGHT = (46, 172, 172)  # #2EACAC
    GREEN = (34, 139, 34)
    RED = (224, 92, 92)          # #E05C5C
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (180, 180, 180)
    BG_DARK = (0, 0, 0, 160)    # Semi-transparent black

    is_normal = "Normal" in status

    # ---- 1. Status badge border ----
    border_color = GREEN if is_normal else RED
    border_width = 4
    for i in range(border_width):
        draw.rectangle(
            [i, i, W - 1 - i, H - 1 - i],
            outline=border_color
        )

    # ---- 2. Top label bar (semi-transparent background) ----
    bar_height = 56
    overlay = Image.new("RGBA", (W, bar_height), (0, 0, 0, 180))
    img_rgba = img.convert("RGBA")
    img_rgba.paste(overlay, (0, 0), overlay)
    img = img_rgba.convert("RGB")
    draw = ImageDraw.Draw(img)

    # Status indicator dot
    dot_color = GREEN if is_normal else RED
    draw.ellipse([12, 10, 28, 26], fill=dot_color)

    # Prediction text
    label_text = f"{top_pred}"
    draw.text((34, 8), label_text, fill=WHITE, font=font_large)

    # Confidence percentage
    conf_text = f"{confidence * 100:.1f}%"
    draw.text((W - 70, 8), conf_text, fill=TEAL_LIGHT, font=font_large)

    # Status text
    status_short = "Normal" if is_normal else "Review"
    draw.text((34, 32), status_short, fill=dot_color, font=font_small)

    # ---- 3. Confidence bar ----
    bar_y = bar_height + 2
    bar_bg_w = W - 8
    bar_fill_w = int(bar_bg_w * confidence)
    draw.rectangle([4, bar_y, 4 + bar_bg_w, bar_y + 6], fill=(60, 60, 60))
    draw.rectangle([4, bar_y, 4 + bar_fill_w, bar_y + 6], fill=TEAL_LIGHT)

    # ---- 4. Top-K predictions overlay (bottom-left) ----
    bottom_overlay = Image.new("RGBA", (W, 70), (0, 0, 0, 160))
    img_rgba = img.convert("RGBA")
    img_rgba.paste(bottom_overlay, (0, H - 70), bottom_overlay)
    img = img_rgba.convert("RGB")
    draw = ImageDraw.Draw(img)

    y_offset = H - 65
    for i, pred in enumerate(top_k[:3]):
        cls_name = pred["class"]
        cls_conf = pred["confidence"]
        text = f"#{i+1} {cls_name}: {cls_conf*100:.1f}%"
        color = WHITE if i == 0 else GRAY
        draw.text((10, y_offset + i * 18), text, fill=color, font=font_small)

    # ---- 5. TB Solutions watermark ----
    watermark = "TB Solutions"
    draw.text((W - 100, H - 18), watermark, fill=(255, 255, 255, 80),
              font=font_small)

    return img


# ============================================================
# 5. BATCH INFERENCE
# ============================================================

def batch_predict(image_folder, model, output_folder=None,
                  save_annotated=True, conf_threshold=0.5,
                  target_size=224):
    """
    Run inference on all images in a folder.

    Args:
        image_folder (str): Path to folder containing images.
        model: Loaded YOLO model.
        output_folder (str|None): Where to save annotated images.
            If None, annotated images are not saved.
        save_annotated (bool): Whether to save annotated versions.
        conf_threshold (float): Confidence threshold for status.
        target_size (int): Inference input size.

    Returns:
        tuple: (results_df, summary_dict)
            - results_df: pd.DataFrame with one row per image
            - summary_dict: aggregated statistics
    """
    IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

    # Collect image paths
    image_paths = []
    if os.path.isdir(image_folder):
        for root, _, files in os.walk(image_folder):
            for f in sorted(files):
                if os.path.splitext(f)[1].lower() in IMG_EXTS:
                    image_paths.append(os.path.join(root, f))
    else:
        raise NotADirectoryError(f"Not a directory: {image_folder}")

    if not image_paths:
        raise ValueError(f"No images found in: {image_folder}")

    # Create output folder
    if output_folder and save_annotated:
        os.makedirs(output_folder, exist_ok=True)

    # Process each image
    results = []
    batch_start = time.time()

    for i, img_path in enumerate(image_paths):
        try:
            # Predict
            pred = predict(
                img_path, model,
                conf_threshold=conf_threshold,
                target_size=target_size,
            )

            # Determine true class from folder name
            parent_dir = os.path.basename(os.path.dirname(img_path))
            true_class = parent_dir if parent_dir != os.path.basename(image_folder) else "Unknown"

            record = {
                "filename": os.path.basename(img_path),
                "filepath": img_path,
                "true_class": true_class,
                "predicted_class": pred["top_prediction"],
                "confidence": pred["confidence"],
                "status": pred["status"],
                "processing_time_ms": pred["processing_time_ms"],
                "correct": true_class == pred["top_prediction"],
            }
            results.append(record)

            # Save annotated image
            if output_folder and save_annotated:
                annotated = annotate_image(img_path, pred)
                out_name = f"annotated_{os.path.basename(img_path)}"
                out_path = os.path.join(output_folder, out_name)
                annotated.save(out_path, quality=95)

        except Exception as e:
            results.append({
                "filename": os.path.basename(img_path),
                "filepath": img_path,
                "true_class": "Unknown",
                "predicted_class": "ERROR",
                "confidence": 0.0,
                "status": f"Error: {str(e)}",
                "processing_time_ms": 0,
                "correct": False,
            })

        # Progress update every 50 images
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(image_paths)} images...")

    batch_elapsed = time.time() - batch_start

    # Build DataFrame
    results_df = pd.DataFrame(results)

    # Build summary
    total = len(results_df)
    correct = results_df["correct"].sum() if "correct" in results_df else 0
    per_class = results_df["predicted_class"].value_counts().to_dict()
    avg_conf_per_class = (
        results_df.groupby("predicted_class")["confidence"]
        .mean()
        .to_dict()
    )
    normal_count = results_df[
        results_df["status"].str.contains("Normal", na=False)
    ].shape[0]
    abnormal_count = total - normal_count
    avg_time = results_df["processing_time_ms"].mean()

    summary = {
        "total_images": total,
        "correct_predictions": int(correct),
        "accuracy": round(correct / total, 4) if total > 0 else 0,
        "per_class_count": per_class,
        "avg_confidence_per_class": {
            k: round(v, 4) for k, v in avg_conf_per_class.items()
        },
        "normal_count": normal_count,
        "abnormal_count": abnormal_count,
        "total_time_seconds": round(batch_elapsed, 2),
        "avg_time_per_image_ms": round(avg_time, 1),
        "throughput_images_per_sec": round(total / batch_elapsed, 1)
        if batch_elapsed > 0 else 0,
    }

    return results_df, summary


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def image_to_base64(image_input):
    """Convert an image to base64 string."""
    if isinstance(image_input, str):
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise TypeError(f"Unsupported type: {type(image_input)}")

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_image(b64_string):
    """Convert a base64 string to PIL Image."""
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(img_bytes))


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetal Image Analysis — Inference Pipeline"
    )
    parser.add_argument("image", help="Path to ultrasound image")
    parser.add_argument(
        "--model", default="best.pt",
        help="Path to YOLOv8 model file (default: best.pt)"
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="Number of top predictions (default: 3)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--annotate", action="store_true",
        help="Save annotated image to disk"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path for annotated image"
    )

    args = parser.parse_args()

    print("Loading model...")
    model = load_model(args.model)
    info = get_model_info(model)
    print(f"Model: {info['model_version']} | "
          f"Classes: {info['num_classes']} | "
          f"Device: {info['device']}")

    print(f"\nRunning inference on: {args.image}")
    result = predict(
        args.image, model,
        top_k=args.top_k,
        conf_threshold=args.threshold,
    )

    print(f"\n{'='*50}")
    print(f"Prediction : {result['top_prediction']}")
    print(f"Confidence : {result['confidence']*100:.1f}%")
    print(f"Status     : {result['status']}")
    print(f"Time       : {result['processing_time_ms']:.1f} ms")
    print(f"{'='*50}")
    print(f"\nClinical Note: {result['clinical_note']}")

    if args.annotate:
        output_path = args.output or f"annotated_{os.path.basename(args.image)}"
        _, orig = preprocess(args.image)
        annotated = annotate_image(orig, result)
        annotated.save(output_path)
        print(f"\nAnnotated image saved: {output_path}")
