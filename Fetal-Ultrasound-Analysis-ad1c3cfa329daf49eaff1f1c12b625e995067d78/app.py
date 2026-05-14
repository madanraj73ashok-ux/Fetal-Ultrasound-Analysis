# ============================================================
# app.py — Flask REST API for Fetal Image Analysis
# ============================================================
# Project : Fetal Image Analysis Using Deep Learning
# Author  : TB Solutions
# Phase   : 3 — Inference Pipeline (API Server)
#
# Endpoints:
#   POST /predict           — image file → JSON prediction
#   POST /predict/base64    — base64 JSON → JSON prediction
#   POST /predict/annotated — image file → annotated JPEG
#   GET  /health            — health check
#   GET  /classes           — list of class names
#
# Usage:
#   python app.py
#   → Server runs at http://localhost:5000
# ============================================================

import os
import io
import time
import logging
import json
import base64
import csv
from datetime import datetime
from pathlib import Path

from flask import (
    Flask, request, jsonify, send_file
)
from flask_cors import CORS
from PIL import Image

# Import our inference module
from inference import (
    load_model, get_model_info, predict,
    preprocess, annotate_image, MODEL_VERSION, EXPECTED_CLASS_NAMES
)

# ============================================================
# CONFIGURATION
# ============================================================

# Model path — configurable via environment variable
MODEL_PATH = os.environ.get("FETAL_MODEL_PATH", "best.pt")
BASE_DIR = Path(__file__).resolve().parent
DETECTION_MODEL_PATH = BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"
CLASSIFICATION_MODEL_PATH = BASE_DIR / "runs" / "classify" / "train" / "weights" / "best.pt"
EVALUATION_UNAVAILABLE_MESSAGE = (
    "Real evaluation results are not available. Run YOLO validation using a "
    "labeled dataset."
)
YOLO_EVAL_DIR = BASE_DIR / "runs" / "detect" / "train"
YOLO_EVAL_RESULTS = YOLO_EVAL_DIR / "results.csv"
YOLO_EVAL_CONFUSION = YOLO_EVAL_DIR / "confusion_matrix.png"
YOLO_EVAL_WEIGHTS = YOLO_EVAL_DIR / "weights" / "best.pt"
YOLO_CLS_EVAL_DIR = BASE_DIR / "runs" / "classify" / "train"
YOLO_CLS_RESULTS = YOLO_CLS_EVAL_DIR / "results.csv"
YOLO_CLS_CONFUSION = YOLO_CLS_EVAL_DIR / "confusion_matrix.png"
YOLO_CLS_WEIGHTS = YOLO_CLS_EVAL_DIR / "weights" / "best.pt"
YOLO_EVAL_COMMAND = (
    "yolo detect val model=runs/detect/train/weights/best.pt data=data.yaml"
)
YOLO_CLS_EVAL_COMMAND = (
    "yolo classify val model=runs/classify/train/weights/best.pt data=dataset_cls"
)

# Server settings
HOST = os.environ.get("FETAL_HOST", "0.0.0.0")
PORT = int(os.environ.get("FETAL_PORT", 5000))

# Limits
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}

NOT_ULTRASOUND_LABEL = "Not ultrasound"
VALID_GROUND_TRUTH_LABELS = {
    "Fetal brain",
    "Fetal abdomen",
    "Fetal femur",
    NOT_ULTRASOUND_LABEL,
}

# ============================================================
# APP INITIALIZATION
# ============================================================

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Enable CORS for Antigravity frontend
CORS(app, resources={r"/*": {"origins": "*"}})

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---- Model is loaded lazily so the demo UI is reachable immediately. ----
model = None
model_info = None
model_load_error = None
active_model_type = None


def choose_model_path():
    """Prefer trained detection, then trained classification, then configured fallback."""
    env_path = os.environ.get("FETAL_MODEL_PATH")
    if env_path:
        path = Path(env_path)
        return str(path), "Custom"
    if DETECTION_MODEL_PATH.exists():
        return str(DETECTION_MODEL_PATH), "Detection"
    if CLASSIFICATION_MODEL_PATH.exists():
        return str(CLASSIFICATION_MODEL_PATH), "Classification"
    return MODEL_PATH, "Unknown"


def set_prediction_model_path(model_path):
    """Point future predictions at a newly trained model."""
    global MODEL_PATH, model, model_info, model_load_error, active_model_type
    MODEL_PATH = str(model_path)
    active_model_type = "Detection" if "runs\\detect" in MODEL_PATH or "runs/detect" in MODEL_PATH else "Classification"
    model = None
    model_info = None
    model_load_error = None
    logger.info(f"Prediction model path updated: {MODEL_PATH}")


def ensure_model_loaded():
    """Load the model once, on demand, before prediction endpoints use it."""
    global MODEL_PATH, model, model_info, model_load_error, active_model_type

    if model is not None:
        return True

    MODEL_PATH, active_model_type = choose_model_path()
    logger.info(f"Loading model from: {MODEL_PATH}")
    try:
        model = load_model(MODEL_PATH)
        model_info = get_model_info(model)
        model_load_error = None
        logger.info(f"Model loaded: {model_info['model_version']} | "
                    f"{model_info['num_classes']} classes | "
                    f"Device: {model_info['device']}")
        return True
    except Exception as e:
        logger.error(f"FATAL: Failed to load model: {e}")
        model = None
        model_info = None
        model_load_error = str(e)
        return False


try:
    from training_dashboard import register_training_routes
    register_training_routes(app, BASE_DIR, on_model_updated=set_prediction_model_path)
except Exception as exc:
    logger.warning(f"Training dashboard routes were not registered: {exc}")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def allowed_file(filename):
    """Check if file extension is in the allowed list."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def _safe_div(numerator, denominator):
    """Return zero when a metric denominator is empty."""
    return numerator / denominator if denominator else 0


def calculate_metrics_from_matrix(tp, tn, fp, fn):
    """
    Calculate percentages from confusion-matrix counts.

    The UI/API should display these derived values instead of hardcoded
    percentages.
    """
    total = tp + tn + fp + fn
    accuracy = _safe_div(tp + tn, total)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    return {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2),
    }


def normalize_ground_truth(value):
    """Return a supported ground-truth label or None."""
    if value in VALID_GROUND_TRUTH_LABELS:
        return value
    return None


def event_matrix_from_labels(actual_label, predicted_label, is_ultrasound):
    """
    Build one confusion-matrix event from actual and predicted labels.

    For this beginner demo:
    - Correct fetal structure prediction = TP
    - Wrong fetal structure prediction = FN
    - Correctly rejected non-ultrasound = TN
    - Non-ultrasound classified as fetal = FP
    """
    counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    actual_label = normalize_ground_truth(actual_label)
    if actual_label is None:
        return counts

    predicted_positive = is_ultrasound and predicted_label != NOT_ULTRASOUND_LABEL
    actual_positive = actual_label != NOT_ULTRASOUND_LABEL

    if actual_positive and predicted_label == actual_label:
        counts["TP"] = 1
    elif actual_positive:
        counts["FN"] = 1
    elif predicted_positive:
        counts["FP"] = 1
    else:
        counts["TN"] = 1

    return counts


def add_evaluation_fields(result, actual_label, predicted_label, is_ultrasound):
    """Attach labels for traceability without exposing fake dataset metrics."""
    result.update({
        "actual_label": normalize_ground_truth(actual_label),
        "predicted_label": predicted_label,
        "is_ultrasound": bool(is_ultrasound),
    })
    result.setdefault("model_type", active_model_type or "Unknown")
    return result


def invalid_input_result(actual_label=None):
    """Return a safe non-ultrasound prediction payload."""
    result = {
        "top_prediction": NOT_ULTRASOUND_LABEL,
        "confidence": 0,
        "top_k_predictions": [],
        "status": "Invalid input",
        "clinical_note": "Invalid input: Please upload a fetal ultrasound image.",
        "processing_time_ms": 0,
        "image_size": [0, 0],
        "model_version": MODEL_VERSION if model_info else "fetal_yolov8s_v1",
        "model_type": active_model_type or "Unknown",
    }
    return add_evaluation_fields(
        result,
        actual_label,
        NOT_ULTRASOUND_LABEL,
        is_ultrasound=False,
    )


def model_missing_response():
    """Return an error response when no trained model is available."""
    return {
        "error": "Model not found",
        "detail": (
            "Model not found. Please train or upload a trained model. "
            "Run: python train_model.py (detection) or "
            "python train_classifier.py (classification)"
        ),
        "model_load_error": model_load_error,
    }


def is_likely_fetal_ultrasound(img):
    """
    Lightweight content gate for the demo/API.

    Fetal ultrasound scans are usually monochrome with low color saturation.
    This rejects obvious documents, menus, selfies, and other colorful images
    before the classifier can produce a misleading medical-looking label.
    """
    sample = img.convert("RGB")
    sample.thumbnail((192, 192))
    pixels = list(sample.getdata())
    if not pixels:
        return False

    gray_like = 0
    saturation_sum = 0.0

    for r, g, b in pixels:
        max_c = max(r, g, b)
        min_c = min(r, g, b)
        if abs(r - g) <= 22 and abs(g - b) <= 22 and abs(r - b) <= 22:
            gray_like += 1
        saturation_sum += (max_c - min_c) / max_c if max_c else 0

    total = len(pixels)
    gray_ratio = gray_like / total
    avg_saturation = saturation_sum / total

    return gray_ratio >= 0.62 and avg_saturation <= 0.22


def validate_ultrasound_content(img):
    """Return a Flask error response if the uploaded image is not ultrasound-like."""
    if is_likely_fetal_ultrasound(img):
        return None

    return jsonify({
        "error": "Invalid ultrasound image",
        "detail": (
            "Please upload a fetal ultrasound scan. The selected image looks "
            "like a color photo, document, menu, or other non-ultrasound image."
        ),
    }), 400


def validate_image_request():
    """
    Validate incoming image upload request.

    Returns:
        tuple: (pil_image, error_response)
            If valid: (PIL.Image, None)
            If invalid: (None, Flask response)
    """
    if "image" not in request.files:
        return None, jsonify({
            "error": "No image file provided",
            "detail": "Send a file with key 'image' in multipart/form-data"
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return None, jsonify({
            "error": "Empty filename",
            "detail": "The uploaded file has no name"
        }), 400

    if not allowed_file(file.filename):
        return None, jsonify({
            "error": "Invalid file type",
            "detail": f"Allowed types: {', '.join(ALLOWED_EXTENSIONS)}",
            "received": file.filename,
        }), 400

    try:
        img = Image.open(file.stream).convert("RGB")
        return img, None
    except Exception as e:
        return None, jsonify({
            "error": "Could not read image",
            "detail": str(e),
        }), 400


def decode_base64_image(b64_string):
    """Decode a base64 image string into a PIL image."""
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def find_first_existing(patterns):
    """Return the first matching project file for a list of glob patterns."""
    for pattern in patterns:
        matches = sorted(BASE_DIR.glob(pattern), key=lambda path: len(path.parts))
        for match in matches:
            if match.is_file():
                return match
    return None


def find_dataset_folders():
    """Report whether common labeled dataset split folders exist."""
    candidates = [BASE_DIR / "dataset", BASE_DIR / "data" / "dataset"]
    found = {}
    for root in candidates:
        for split in ("train", "val", "valid", "test"):
            split_path = root / split
            if split_path.exists() and split_path.is_dir():
                found[split] = str(split_path.relative_to(BASE_DIR))
    return found


def parse_float(value):
    """Parse numeric metric values from YOLO CSV cells."""
    if value is None:
        return None
    text = str(value).strip().replace("%", "")
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number * 100 if 0 <= number <= 1 else number


def latest_results_metrics(results_csv):
    """Read real YOLO validation metrics from the last row of results.csv."""
    with results_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}

    row = rows[-1]

    def first_metric(names):
        for name in names:
            for key, value in row.items():
                if key and key.strip() == name:
                    parsed = parse_float(value)
                    if parsed is not None:
                        return round(parsed, 2)
        return None

    precision = first_metric([
        "metrics/precision(B)", "metrics/precision", "precision", "Precision",
    ])
    recall = first_metric([
        "metrics/recall(B)", "metrics/recall", "recall", "Recall",
    ])
    accuracy = first_metric([
        "metrics/accuracy_top1", "metrics/accuracy", "accuracy", "Accuracy",
        "top1_acc",
    ])
    top5 = first_metric([
        "metrics/accuracy_top5", "accuracy_top5", "top5_acc",
    ])
    map50 = first_metric([
        "metrics/mAP50(B)", "metrics/mAP50", "mAP50",
    ])
    map5095 = first_metric([
        "metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50-95",
    ])

    f1_score = None
    if precision is not None and recall is not None:
        denominator = precision + recall
        f1_score = round(2 * precision * recall / denominator, 2) if denominator else 0

    metrics = {}
    if accuracy is not None:
        metrics["accuracy"] = accuracy
    if top5 is not None:
        metrics["top5_accuracy"] = top5
    if map50 is not None:
        metrics["map50"] = map50
    if map5095 is not None:
        metrics["map50_95"] = map5095
    if precision is not None:
        metrics["precision"] = precision
    if recall is not None:
        metrics["recall"] = recall
    if f1_score is not None:
        metrics["f1_score"] = f1_score
    return metrics


def evaluation_status_payload():
    """Build a dashboard payload from detection and classification artifacts."""
    detect_results_csv = YOLO_EVAL_RESULTS if YOLO_EVAL_RESULTS.exists() else None
    detect_confusion = YOLO_EVAL_CONFUSION if YOLO_EVAL_CONFUSION.exists() else None
    cls_results_csv = YOLO_CLS_RESULTS if YOLO_CLS_RESULTS.exists() else None
    cls_confusion = YOLO_CLS_CONFUSION if YOLO_CLS_CONFUSION.exists() else None
    data_yaml = find_first_existing([
        "data.yaml",
        "dataset/data.yaml",
        "data/dataset/data.yaml",
    ])
    best_weights = YOLO_EVAL_WEIGHTS if YOLO_EVAL_WEIGHTS.exists() else None
    cls_weights = YOLO_CLS_WEIGHTS if YOLO_CLS_WEIGHTS.exists() else None
    dataset_folders = find_dataset_folders()

    files = {
        "expected_results_csv": "runs/detect/train/results.csv",
        "expected_confusion_matrix": "runs/detect/train/confusion_matrix.png",
        "expected_best_pt": "runs/detect/train/weights/best.pt",
        "best_pt": str(best_weights.relative_to(BASE_DIR)) if best_weights else None,
        "results_csv": str(detect_results_csv.relative_to(BASE_DIR)) if detect_results_csv else None,
        "confusion_matrix": str(detect_confusion.relative_to(BASE_DIR)) if detect_confusion else None,
        "classification_expected_results_csv": "runs/classify/train/results.csv",
        "classification_expected_confusion_matrix": "runs/classify/train/confusion_matrix.png",
        "classification_expected_best_pt": "runs/classify/train/weights/best.pt",
        "classification_best_pt": (
            str(cls_weights.relative_to(BASE_DIR)) if cls_weights else None
        ),
        "classification_results_csv": (
            str(cls_results_csv.relative_to(BASE_DIR)) if cls_results_csv else None
        ),
        "classification_confusion_matrix": (
            str(cls_confusion.relative_to(BASE_DIR)) if cls_confusion else None
        ),
        "data_yaml": str(data_yaml.relative_to(BASE_DIR)) if data_yaml else None,
        "dataset_folders": dataset_folders,
    }

    metrics = {}
    classification_metrics = {}
    if detect_results_csv:
        try:
            metrics = latest_results_metrics(detect_results_csv)
        except Exception as exc:
            logger.warning(f"Could not parse detection evaluation results: {exc}")
    if cls_results_csv:
        try:
            classification_metrics = latest_results_metrics(cls_results_csv)
        except Exception as exc:
            logger.warning(f"Could not parse classification evaluation results: {exc}")

    available = bool(metrics or detect_confusion or classification_metrics or cls_confusion)
    return {
        "available": available,
        "message": None if available else EVALUATION_UNAVAILABLE_MESSAGE,
        "metrics": metrics,
        "classification_metrics": classification_metrics,
        "files": files,
        "command": YOLO_EVAL_COMMAND,
        "classification_command": YOLO_CLS_EVAL_COMMAND,
        "confusion_matrix_url": (
            "/evaluation/confusion-matrix" if detect_confusion else None
        ),
        "classification_confusion_matrix_url": (
            "/evaluation/classification-confusion-matrix" if cls_confusion else None
        ),
    }


# ============================================================
# ENDPOINTS
# ============================================================

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """
    POST /predict
    Input: multipart/form-data with 'image' file
    Output: JSON prediction result
    """
    img, error = validate_image_request()
    if error:
        return error

    actual_label = normalize_ground_truth(request.form.get("ground_truth"))
    start_time = time.time()

    try:
        if not is_likely_fetal_ultrasound(img):
            result = invalid_input_result(actual_label)
            return jsonify(result), 200

        if not ensure_model_loaded():
            return jsonify(model_missing_response()), 503

        result = predict(img, model)
        result = add_evaluation_fields(
            result,
            actual_label,
            result["top_prediction"],
            is_ultrasound=True,
        )
        elapsed = (time.time() - start_time) * 1000

        logger.info(
            f"POST /predict | "
            f"{request.files['image'].filename} | "
            f"Pred: {result['top_prediction']} "
            f"({result['confidence']*100:.1f}%) | "
            f"{elapsed:.0f}ms"
        )

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            "error": "Prediction failed",
            "detail": str(e),
        }), 500


@app.route("/", methods=["GET"])
def demo_page():
    """Serve the browser demo UI at the API root."""
    demo_path = os.path.join(os.path.dirname(__file__), "fetal_analysis_demo.html")
    if not os.path.exists(demo_path):
        return jsonify({
            "error": "Demo UI not found",
            "detail": "fetal_analysis_demo.html is missing",
        }), 404

    return send_file(demo_path, mimetype="text/html")


@app.route("/evaluation", methods=["GET"])
def evaluation_page():
    """Serve the developer evaluation dashboard."""
    return demo_page()


@app.route("/dataset/status", methods=["GET"])
def dataset_status_endpoint():
    """Report dataset presence for the UI."""
    dataset_det = BASE_DIR / "dataset"
    dataset_cls = BASE_DIR / "dataset_cls"
    data_yaml = find_first_existing(["data.yaml", "dataset/data.yaml"])
    det_splits = {}
    cls_splits = {}
    for split in ("train", "val", "test"):
        det_img = dataset_det / "images" / split
        det_lbl = dataset_det / "labels" / split
        if det_img.exists():
            count = sum(1 for f in det_img.iterdir() if f.is_file())
            det_splits[split] = count
        cls_dir = dataset_cls / split
        if cls_dir.exists():
            classes = [d.name for d in cls_dir.iterdir() if d.is_dir()]
            count = sum(1 for f in cls_dir.rglob("*") if f.is_file())
            cls_splits[split] = {"classes": classes, "images": count}

    has_detection = bool(det_splits)
    has_classification = bool(cls_splits)
    return jsonify({
        "detection_dataset": {
            "available": has_detection,
            "path": str(dataset_det),
            "data_yaml": str(data_yaml) if data_yaml else None,
            "splits": det_splits,
        },
        "classification_dataset": {
            "available": has_classification,
            "path": str(dataset_cls),
            "splits": cls_splits,
        },
        "message": None if (has_detection or has_classification)
            else "Dataset not found. Please add dataset before training.",
    }), 200


@app.route("/evaluation/status", methods=["GET"])
def evaluation_status_endpoint():
    """Return real evaluation metrics when local artifacts exist."""
    return jsonify(evaluation_status_payload()), 200


@app.route("/evaluation/confusion-matrix", methods=["GET"])
def evaluation_confusion_matrix_endpoint():
    """Serve the real confusion-matrix image if it exists."""
    payload = evaluation_status_payload()
    path = payload["files"].get("confusion_matrix")
    if not path:
        return jsonify({
            "error": "Confusion matrix not found",
            "detail": EVALUATION_UNAVAILABLE_MESSAGE,
        }), 404
    return send_file(BASE_DIR / path, mimetype="image/png")


@app.route("/evaluation/classification-confusion-matrix", methods=["GET"])
def evaluation_classification_confusion_matrix_endpoint():
    """Serve the real classification confusion matrix image if it exists."""
    payload = evaluation_status_payload()
    path = payload["files"].get("classification_confusion_matrix")
    if not path:
        return jsonify({
            "error": "Classification confusion matrix not found",
            "detail": EVALUATION_UNAVAILABLE_MESSAGE,
        }), 404
    return send_file(BASE_DIR / path, mimetype="image/png")


@app.route("/predict/base64", methods=["POST"])
def predict_base64_endpoint():
    """
    POST /predict/base64
    Input: JSON {"image": "<base64-encoded-string>"}
    Output: JSON prediction result
    """
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({
            "error": "No image data provided",
            "detail": "Send JSON with key 'image' containing base64 string"
        }), 400

    start_time = time.time()

    try:
        b64_string = data["image"]
        actual_label = normalize_ground_truth(data.get("ground_truth"))
        img = decode_base64_image(b64_string)
        if not is_likely_fetal_ultrasound(img):
            result = invalid_input_result(actual_label)
            return jsonify(result), 200

        if not ensure_model_loaded():
            return jsonify(model_missing_response()), 503

        result = predict(img, model)
        result = add_evaluation_fields(
            result,
            actual_label,
            result["top_prediction"],
            is_ultrasound=True,
        )
        elapsed = (time.time() - start_time) * 1000

        logger.info(
            f"POST /predict/base64 | "
            f"Pred: {result['top_prediction']} "
            f"({result['confidence']*100:.1f}%) | "
            f"{elapsed:.0f}ms"
        )

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Base64 prediction error: {e}")
        return jsonify({
            "error": "Prediction failed",
            "detail": str(e),
        }), 500


@app.route("/predict/annotated", methods=["POST"])
def predict_annotated_endpoint():
    """
    POST /predict/annotated
    Input: multipart/form-data with 'image' file
    Output: Annotated JPEG image
            Prediction JSON in X-Prediction-* response headers
    """
    if not ensure_model_loaded():
        return jsonify({
            "error": "Model not loaded",
            "detail": model_load_error,
        }), 500

    img, error = validate_image_request()
    if error:
        return error
    content_error = validate_ultrasound_content(img)
    if content_error:
        return content_error

    start_time = time.time()

    try:
        result = predict(img, model)
        annotated = annotate_image(img, result)

        # Save annotated image to bytes
        img_buffer = io.BytesIO()
        annotated.save(img_buffer, format="JPEG", quality=95)
        img_buffer.seek(0)

        elapsed = (time.time() - start_time) * 1000

        logger.info(
            f"POST /predict/annotated | "
            f"{request.files['image'].filename} | "
            f"Pred: {result['top_prediction']} "
            f"({result['confidence']*100:.1f}%) | "
            f"{elapsed:.0f}ms"
        )

        response = send_file(
            img_buffer,
            mimetype="image/jpeg",
            as_attachment=False,
            download_name="annotated_result.jpg"
        )

        # Add prediction data as custom headers
        response.headers["X-Prediction-Class"] = result["top_prediction"]
        response.headers["X-Prediction-Confidence"] = str(result["confidence"])
        response.headers["X-Prediction-Status"] = result["status"]
        response.headers["X-Processing-Time-Ms"] = str(result["processing_time_ms"])

        return response

    except Exception as e:
        logger.error(f"Annotated prediction error: {e}")
        return jsonify({
            "error": "Annotated prediction failed",
            "detail": str(e),
        }), 500


@app.route("/health", methods=["GET"])
def health_endpoint():
    """
    GET /health
    Output: Server and model health status
    """
    status = {
        "status": "ok",
        "model": "loaded" if model is not None else "not loaded",
        "model_type": active_model_type or choose_model_path()[1],
        "model_path": MODEL_PATH,
        "timestamp": datetime.now().isoformat(),
        "version": "1.0",
    }

    if model_info:
        status["classes"] = model_info["classes"]
        status["num_classes"] = model_info["num_classes"]
        status["device"] = model_info["device"]
        status["model_version"] = model_info["model_version"]
    elif model_load_error:
        status["detail"] = model_load_error
        status["classes"] = EXPECTED_CLASS_NAMES
        status["num_classes"] = len(EXPECTED_CLASS_NAMES)

    return jsonify(status), 200


@app.route("/classes", methods=["GET"])
def classes_endpoint():
    """
    GET /classes
    Output: List of all class names the model can predict
    """
    if not ensure_model_loaded():
        return jsonify({
            "error": "Model not loaded",
            "detail": model_load_error,
            "classes": EXPECTED_CLASS_NAMES,
            "count": len(EXPECTED_CLASS_NAMES),
        }), 503

    return jsonify({
        "classes": model_info["classes"],
        "count": model_info["num_classes"],
    }), 200


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(413)
def too_large(e):
    """Handle files exceeding MAX_CONTENT_LENGTH."""
    return jsonify({
        "error": "File too large",
        "detail": f"Maximum file size is {MAX_CONTENT_LENGTH // (1024*1024)} MB",
    }), 413


@app.errorhandler(404)
def not_found(e):
    """Handle unknown endpoints."""
    return jsonify({
        "error": "Endpoint not found",
        "detail": "Available endpoints: /predict, /predict/base64, "
                  "/predict/annotated, /health, /classes",
    }), 404


@app.errorhandler(405)
def method_not_allowed(e):
    """Handle wrong HTTP method."""
    return jsonify({
        "error": "Method not allowed",
        "detail": str(e),
    }), 405


@app.errorhandler(500)
def internal_error(e):
    """Handle unexpected server errors."""
    logger.error(f"Internal error: {e}")
    return jsonify({
        "error": "Internal server error",
        "detail": "An unexpected error occurred. Check server logs.",
    }), 500


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Fetal Image Analysis - API Server")
    print("=" * 60)
    print(f"  Host   : {HOST}")
    print(f"  Port   : {PORT}")
    print(f"  Model  : {MODEL_PATH}")

    if model_info:
        print(f"  Task   : {model_info['task']}")
        print(f"  Classes: {model_info['num_classes']}")
        print(f"  Device : {model_info['device']}")
    else:
        print("  Model  : lazy-loaded on first prediction request")

    print("=" * 60)
    print(f"  Endpoints:")
    print(f"    POST /predict           - image -> JSON")
    print(f"    POST /predict/base64    - base64 -> JSON")
    print(f"    POST /predict/annotated - image -> annotated JPEG")
    print(f"    GET  /health            — health check")
    print(f"    GET  /classes           — class list")
    print("=" * 60)

    app.run(host=HOST, port=PORT, debug=False)
