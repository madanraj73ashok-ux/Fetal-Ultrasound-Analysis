# Fetal Image Analysis — Inference Package

## TB Solutions | ECE Microproject

---

## Overview

This package provides a **production-ready inference pipeline** for the Fetal Image Analysis system. 
It classifies fetal ultrasound images into 9 anatomical planes using a YOLOv8s model trained on the 
BCNatal dataset.

### Classes Detected

| # | Class | Description |
|---|-------|-------------|
| 1 | Fetal abdomen | Abdominal circumference measurement plane |
| 2 | Fetal brain | Brain structural assessment view |
| 3 | Fetal femur | Femur length measurement plane |
| 4 | Fetal thorax | Thoracic/cardiac evaluation view |
| 5 | Maternal cervix | Cervical length assessment view |
| 6 | Trans-cerebellum | Posterior fossa / TCD measurement |
| 7 | Trans-thalamic | BPD and HC measurement plane |
| 8 | Trans-ventricular | Lateral ventricle assessment |
| 9 | Other | Non-standard / unclassified plane |

---

## Files

| File | Description |
|------|-------------|
| `inference.py` | Core inference module (preprocess, predict, annotate, batch) |
| `app.py` | Flask REST API server |
| `requirements.txt` | Python dependencies |
| `best.pt` | Trained YOLOv8s model weights (add separately) |
| `test_images/` | Sample fetal ultrasound images for testing |

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Place Your Model

Copy `best.pt` from Google Drive (`/FetalNet/best.pt`) into this directory.

### 3. Run as Python Module

```python
from inference import load_model, predict

# Load model
model = load_model("best.pt")

# Predict on an image
result = predict("ultrasound.png", model)
print(result["top_prediction"])     # e.g., "Fetal Brain"
print(result["confidence"])         # e.g., 0.967
print(result["status"])             # "Normal" or "Review Needed"
print(result["clinical_note"])      # Clinical description
```

### 4. Run CLI

```bash
python inference.py ultrasound.png --model best.pt --annotate
```

### 5. Run API Server

```bash
python app.py
```

Server starts at `http://localhost:5000`. Endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Upload image → get JSON prediction |
| POST | `/predict/base64` | Send base64 → get JSON prediction |
| POST | `/predict/annotated` | Upload image → get annotated JPEG |
| GET | `/health` | Server health check |
| GET | `/classes` | List of class names |

### 6. Test API with cURL

```bash
# Health check
curl http://localhost:5000/health

# Predict
curl -X POST -F "image=@ultrasound.png" http://localhost:5000/predict

# Predict with base64
curl -X POST -H "Content-Type: application/json" \
  -d '{"image": "<base64-string>"}' \
  http://localhost:5000/predict/base64

# Get annotated image
curl -X POST -F "image=@ultrasound.png" \
  http://localhost:5000/predict/annotated --output annotated.jpg
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FETAL_MODEL_PATH` | `best.pt` | Path to model weights |
| `FETAL_HOST` | `0.0.0.0` | Server host |
| `FETAL_PORT` | `5000` | Server port |

---

## Deploying on Antigravity

For the Phase 4 web demo:

1. Upload `inference.py`, `app.py`, and `best.pt` to your Antigravity project
2. The Antigravity frontend (HTML/JS) will call the Flask API endpoints
3. Ensure CORS is enabled (already configured in `app.py`)
4. The frontend sends images, receives JSON predictions, and displays results

---

## Performance

- **Single inference:** ~30-60ms (GPU), ~100-200ms (CPU)
- **Batch throughput:** ~15-30 images/second (GPU)
- **Model size:** ~12 MB (best.pt)
- **Input size:** 224×224 pixels

---

## Disclaimer

This system is for **educational/research purposes only**. It is NOT a medical device 
and should NOT be used for clinical diagnosis. Always consult qualified medical 
professionals for clinical decisions.

---

*TB Solutions — Fetal Image Analysis Using Deep Learning*
