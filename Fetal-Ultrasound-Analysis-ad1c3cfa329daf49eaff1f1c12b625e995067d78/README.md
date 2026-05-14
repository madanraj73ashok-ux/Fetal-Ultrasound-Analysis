# Fetal Ultrasound Analysis Using Deep Learning (YOLOv8)

> **College Microproject — ECE Department**
> Manakula Vinayagar Institute of Technology

⚕️ *This system is for educational and research demonstration only. It is not for clinical diagnosis.*

---

## Overview

This project implements an **AI-powered prenatal diagnostic assistant** that automatically classifies fetal ultrasound images into anatomical planes using **YOLOv8** deep learning. It supports both **detection** (bounding boxes) and **classification** (plane recognition) modes.

### Features
- **Upload & Predict** — Upload ultrasound images and get instant AI classification
- **Model Evaluation** — View real training metrics (accuracy, precision, recall, F1, confusion matrix)
- **Training Dashboard** — Start/stop/monitor YOLOv8 training from the browser
- **Dataset Checker** — Validate dataset structure before training
- **Dual Model Support** — YOLOv8 Detection + YOLOv8 Classification

### Classes Detected

| # | Class | Clinical Use |
|---|-------|-------------|
| 1 | Fetal abdomen | Abdominal circumference measurement |
| 2 | Fetal brain | Neurodevelopmental assessment |
| 3 | Fetal femur | Gestational age estimation |
| 4 | Fetal thorax | Cardiac/lung evaluation |
| 5 | Maternal cervix | Preterm birth risk assessment |
| 6 | Trans-cerebellum | Posterior fossa assessment |
| 7 | Trans-thalamic | BPD/HC measurement |
| 8 | Trans-ventricular | Ventricle assessment |
| 9 | Other | Non-standard plane |

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## Project Structure

```
├── app.py                          # Flask web server (main entry point)
├── inference.py                    # Core inference pipeline (preprocess, predict, annotate)
├── training_dashboard.py           # Training management routes
├── train_model.py                  # YOLOv8 detection training script
├── train_classifier.py             # YOLOv8 classification training script
├── check_dataset.py                # Dataset validation tool
├── convert_classification_dataset.py  # Convert folder-labeled dataset for YOLO-cls
├── fetal_analysis_demo.html        # Main web UI (all pages)
├── static/
│   ├── style.css                   # Application stylesheet
│   └── app.js                      # Frontend JavaScript
├── templates/
│   └── training.html               # Training dashboard page
├── best.pt                         # Trained model weights
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
├── Phase1_Data_Preparation.ipynb   # Data preparation documentation
├── Phase2_Model_Training.ipynb     # Training documentation
└── Phase3_Inference_Pipeline.ipynb # Inference documentation
```

---

## Dataset Format

### Detection Dataset (Bounding Boxes)

```
dataset/
├── images/
│   ├── train/         # Training images (.jpg, .png)
│   ├── val/           # Validation images
│   └── test/          # Test images
├── labels/
│   ├── train/         # YOLO format labels (.txt)
│   ├── val/
│   └── test/
└── data.yaml          # Class names and paths
```

**Label format** (each `.txt` file): `class_id x_center y_center width height` (normalized 0-1)

### Classification Dataset (Folder-Labeled)

```
dataset_cls/
├── train/
│   ├── fetal_brain/   # Images of fetal brain
│   ├── fetal_femur/   # Images of fetal femur
│   └── ...
├── val/
│   ├── fetal_brain/
│   └── ...
└── test/              # Optional
```

### Validate Dataset

```bash
python check_dataset.py --dataset dataset
```

### Convert Classification Dataset

```bash
python convert_classification_dataset.py --input dataset --output dataset_cls
```

---

## Training Steps

### Detection Training

```bash
python train_model.py --data data.yaml --epochs 50 --imgsz 640 --batch 8
```

Output: `runs/detect/train/weights/best.pt`

### Classification Training

```bash
python train_classifier.py --data dataset_cls --epochs 50 --imgsz 224 --batch 16
```

Output: `runs/classify/train/weights/best.pt`

### Training via Browser

1. Open **http://localhost:5000/training**
2. Click **Start Training**
3. Monitor progress, loss curves, and metrics in real-time
4. Best model is automatically loaded for predictions

---

## Prediction Steps

### Via Web UI
1. Open **http://localhost:5000**
2. Click **Prediction** in the navbar
3. Upload a fetal ultrasound image
4. Click **Analyze Image**
5. View results: detected structure, confidence, clinical note

### Via API

```bash
# Health check
curl http://localhost:5000/health

# Predict from file
curl -X POST -F "image=@ultrasound.png" http://localhost:5000/predict

# Predict from base64
curl -X POST -H "Content-Type: application/json" \
  -d '{"image": "<base64-string>"}' \
  http://localhost:5000/predict/base64
```

### Via Python

```python
from inference import load_model, predict

model = load_model("best.pt")
result = predict("ultrasound.png", model)
print(result["top_prediction"])   # e.g., "Fetal brain"
print(result["confidence"])       # e.g., 0.967
```

---

## Evaluation Steps

### Generate Evaluation Metrics

After training, run YOLO validation:

```bash
# Detection
yolo detect val model=runs/detect/train/weights/best.pt data=data.yaml

# Classification
yolo classify val model=runs/classify/train/weights/best.pt data=dataset_cls
```

This generates `results.csv` and `confusion_matrix.png` which the Evaluation page reads automatically.

### View in Browser

Open **http://localhost:5000** → click **Evaluation** to see real metrics.

---

## Deployment Steps

### Local (Development)

```bash
pip install -r requirements.txt
python app.py
# Server runs at http://localhost:5000
```

### Production (Linux/Cloud)

```bash
pip install -r requirements.txt
gunicorn app:app --bind 0.0.0.0:5000 --workers 2
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FETAL_MODEL_PATH` | `best.pt` | Path to model weights |
| `FETAL_HOST` | `0.0.0.0` | Server host |
| `FETAL_PORT` | `5000` | Server port |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **"Model not found"** | Train a model first or place `best.pt` in the project root |
| **"Dataset not found"** | Add dataset to `dataset/` or `dataset_cls/` folder |
| **Server won't start** | Run `pip install -r requirements.txt` first |
| **CUDA out of memory** | Reduce `--batch` size (e.g., `--batch 4`) |
| **Slow inference** | CPU is expected to be slower; use GPU if available |
| **Evaluation page empty** | Run YOLO validation command to generate metrics |
| **Import errors** | Ensure you're using Python 3.9+ with all requirements installed |

---

## Screenshots for Project Report

Take these screenshots for your project documentation:

1. **Home Page** — Shows project overview and how-it-works flow
2. **Prediction Page (before)** — Upload zone with no image
3. **Prediction Page (after)** — With analyzed result showing class, confidence, and clinical note
4. **Evaluation Page** — Real metrics cards and confusion matrix
5. **Training Dashboard** — During or after training showing progress and graphs
6. **Dataset Checker** — Dataset status with splits info
7. **About Page** — Team info and project details
8. **Mobile View** — Responsive layout on mobile width

---

## Team

| Role | Name |
|------|------|
| Team Members | JAI RASIGA.G.K, SANGAMITHIRAI.K, RAJADIVYA.V |
| Project Guide | MR. S.MOHANRAJ AP/ECE |
| Department | Electronics & Communication Engineering |
| Institution | Manakula Vinayagar Institute of Technology |

---

## Disclaimer

This system is for **educational and research demonstration only**. It is NOT a medical device and should NOT be used for clinical diagnosis. Always consult qualified medical professionals for clinical decisions.

---

*Fetal Image Analysis Using Deep Learning — YOLOv8*
