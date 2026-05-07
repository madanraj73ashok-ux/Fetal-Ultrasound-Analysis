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
from datetime import datetime

from flask import (
    Flask, request, jsonify, send_file
)
from flask_cors import CORS
from PIL import Image

# Import our inference module
from inference import (
    load_model, get_model_info, predict,
    preprocess, annotate_image, MODEL_VERSION
)

# ============================================================
# CONFIGURATION
# ============================================================

# Model path — configurable via environment variable
MODEL_PATH = os.environ.get("FETAL_MODEL_PATH", "best.pt")

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

app = Flask(__name__)
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

# ---- Load model at startup (thread-safe, loaded once) ----
logger.info(f"Loading model from: {MODEL_PATH}")
try:
    model = load_model(MODEL_PATH)
    model_info = get_model_info(model)
    logger.info(f"Model loaded: {model_info['model_version']} | "
                f"{model_info['num_classes']} classes | "
                f"Device: {model_info['device']}")
except Exception as e:
    logger.error(f"FATAL: Failed to load model: {e}")
    model = None
    model_info = None


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
    """Attach actual/predicted labels and one-event matrix metrics."""
    counts = event_matrix_from_labels(actual_label, predicted_label, is_ultrasound)
    result.update(calculate_metrics_from_matrix(
        counts["TP"], counts["TN"], counts["FP"], counts["FN"]
    ))
    result.update({
        "actual_label": normalize_ground_truth(actual_label),
        "predicted_label": predicted_label,
        "is_ultrasound": bool(is_ultrasound),
        "event_matrix": counts,
    })
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
    }
    return add_evaluation_fields(
        result,
        actual_label,
        NOT_ULTRASOUND_LABEL,
        is_ultrasound=False,
    )


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
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    img, error = validate_image_request()
    if error:
        return error

    actual_label = normalize_ground_truth(request.form.get("ground_truth"))
    start_time = time.time()

    try:
        if not is_likely_fetal_ultrasound(img):
            result = invalid_input_result(actual_label)
            return jsonify(result), 200

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


@app.route("/predict/base64", methods=["POST"])
def predict_base64_endpoint():
    """
    POST /predict/base64
    Input: JSON {"image": "<base64-encoded-string>"}
    Output: JSON prediction result
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

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
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

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
        "status": "ok" if model is not None else "error",
        "model": "loaded" if model is not None else "not loaded",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0",
    }

    if model_info:
        status["classes"] = model_info["classes"]
        status["num_classes"] = model_info["num_classes"]
        status["device"] = model_info["device"]
        status["model_version"] = model_info["model_version"]

    return jsonify(status), 200 if model else 503


@app.route("/classes", methods=["GET"])
def classes_endpoint():
    """
    GET /classes
    Output: List of all class names the model can predict
    """
    if model_info is None:
        return jsonify({"error": "Model not loaded"}), 503

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
    print("🩺 Fetal Image Analysis — API Server")
    print("=" * 60)
    print(f"  Host   : {HOST}")
    print(f"  Port   : {PORT}")
    print(f"  Model  : {MODEL_PATH}")

    if model_info:
        print(f"  Task   : {model_info['task']}")
        print(f"  Classes: {model_info['num_classes']}")
        print(f"  Device : {model_info['device']}")
    else:
        print("  ⚠️  Model failed to load!")

    print("=" * 60)
    print(f"  Endpoints:")
    print(f"    POST /predict           — image → JSON")
    print(f"    POST /predict/base64    — base64 → JSON")
    print(f"    POST /predict/annotated — image → annotated JPEG")
    print(f"    GET  /health            — health check")
    print(f"    GET  /classes           — class list")
    print("=" * 60)

    app.run(host=HOST, port=PORT, debug=False)
