"""Flask routes for live YOLOv8 training management."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import torch
from flask import jsonify, render_template, send_file


TRAINING_PROCESS: subprocess.Popen | None = None
TRAINING_LOCK = threading.Lock()


def _parse_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _percent(value):
    if value is None:
        return None
    return round(value * 100 if 0 <= value <= 1 else value, 2)


def _gpu_status() -> dict:
    if not torch.cuda.is_available():
        return {"cuda": False, "name": "CPU", "vram_gb": None, "device": "cpu"}
    props = torch.cuda.get_device_properties(0)
    return {
        "cuda": True,
        "name": props.name,
        "vram_gb": round(props.total_memory / (1024 ** 3), 2),
        "device": 0,
    }


def _read_status(status_file: Path) -> dict:
    if not status_file.exists():
        return {}
    try:
        return json.loads(status_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _tail_log(log_file: Path, max_lines: int = 80) -> list[str]:
    if not log_file.exists():
        return []
    try:
        return log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-max_lines:]
    except OSError:
        return []


def _first(row: dict, names: list[str]):
    for name in names:
        for key, value in row.items():
            if key and key.strip() == name:
                return _parse_float(value)
    return None


def _read_results(results_csv: Path) -> tuple[dict, list[dict]]:
    if not results_csv.exists():
        return {}, []
    with results_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}, []

    history: list[dict] = []
    for index, row in enumerate(rows, start=1):
        epoch = _first(row, ["epoch"]) or index
        box_loss = _first(row, ["train/box_loss", "val/box_loss"])
        cls_loss = _first(row, ["train/cls_loss", "val/cls_loss"])
        dfl_loss = _first(row, ["train/dfl_loss", "val/dfl_loss"])
        loss_parts = [value for value in (box_loss, cls_loss, dfl_loss) if value is not None]
        loss = round(sum(loss_parts), 4) if loss_parts else None
        precision = _first(row, ["metrics/precision(B)", "metrics/precision"])
        recall = _first(row, ["metrics/recall(B)", "metrics/recall"])
        map50 = _first(row, ["metrics/mAP50(B)", "metrics/mAP50"])
        map5095 = _first(row, ["metrics/mAP50-95(B)", "metrics/mAP50-95"])
        f1 = None
        if precision is not None and recall is not None:
            denominator = precision + recall
            f1 = (2 * precision * recall / denominator) if denominator else 0
        history.append({
            "epoch": int(epoch),
            "loss": loss,
            "precision": _percent(precision),
            "recall": _percent(recall),
            "map50": _percent(map50),
            "map50_95": _percent(map5095),
            "f1": _percent(f1),
        })

    latest = history[-1]
    return latest, history


def register_training_routes(app, base_dir: Path, on_model_updated=None) -> None:
    """Register training dashboard routes on the existing Flask app."""

    status_file = base_dir / "training_status.json"
    log_file = base_dir / "training.log"
    run_dir = base_dir / "runs" / "detect" / "train"
    results_csv = run_dir / "results.csv"
    artifacts = {
        "results": run_dir / "results.png",
        "confusion": run_dir / "confusion_matrix.png",
        "pr": run_dir / "PR_curve.png",
        "f1": run_dir / "F1_curve.png",
    }

    def process_running() -> bool:
        global TRAINING_PROCESS
        return TRAINING_PROCESS is not None and TRAINING_PROCESS.poll() is None

    def build_status() -> dict:
        status = _read_status(status_file)
        latest, history = _read_results(results_csv)
        running = process_running()
        epochs = int(status.get("epochs") or 50)
        current_epoch = int(latest.get("epoch") or 0)
        progress = round(min(100, (current_epoch / epochs) * 100), 1) if epochs else 0
        completed = bool(status.get("completed")) or (current_epoch >= epochs and current_epoch > 0)

        if completed and status.get("success") and on_model_updated:
            best = run_dir / "weights" / "best.pt"
            if best.exists() and not status.get("prediction_model_reloaded"):
                on_model_updated(best)
                status["prediction_model_reloaded"] = True
                status_file.write_text(json.dumps(status, indent=2), encoding="utf-8")

        artifact_urls = {
            name: f"/training/artifact/{name}" for name, path in artifacts.items() if path.exists()
        }
        return {
            "running": running,
            "completed": completed,
            "success": bool(status.get("success")),
            "message": status.get("message") or ("Training running..." if running else "Training idle."),
            "epoch": current_epoch,
            "epochs": epochs,
            "progress": progress,
            "loss": latest.get("loss"),
            "precision": latest.get("precision"),
            "recall": latest.get("recall"),
            "map50": latest.get("map50"),
            "map50_95": latest.get("map50_95"),
            "f1": latest.get("f1"),
            "gpu": status.get("gpu") or _gpu_status(),
            "history": history,
            "logs": _tail_log(log_file),
            "run_dir": str(run_dir),
            "best_model": str(run_dir / "weights" / "best.pt"),
            "last_model": str(run_dir / "weights" / "last.pt"),
            "artifacts": artifact_urls,
        }

    @app.route("/training", methods=["GET"])
    def training_page():
        return render_template("training.html")

    @app.route("/training/status", methods=["GET"])
    def training_status():
        return jsonify(build_status()), 200

    @app.route("/training/start", methods=["POST"])
    def training_start():
        global TRAINING_PROCESS
        with TRAINING_LOCK:
            if process_running():
                return jsonify({"ok": False, "message": "Training is already running."}), 409

            status_file.write_text(json.dumps({
                "running": True,
                "completed": False,
                "success": False,
                "message": "Training queued...",
                "epochs": 50,
                "gpu": _gpu_status(),
            }, indent=2), encoding="utf-8")
            log_file.write_text("", encoding="utf-8")
            command = [sys.executable, str(base_dir / "train_model.py")]
            TRAINING_PROCESS = subprocess.Popen(
                command,
                cwd=str(base_dir),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )
        return jsonify({"ok": True, "message": "Training started."}), 200

    @app.route("/training/stop", methods=["POST"])
    def training_stop():
        global TRAINING_PROCESS
        with TRAINING_LOCK:
            if not process_running():
                return jsonify({"ok": False, "message": "Training is not running."}), 409
            TRAINING_PROCESS.terminate()
            status = _read_status(status_file)
            status.update({
                "running": False,
                "completed": True,
                "success": False,
                "stopped": True,
                "message": "Training stopped by user.",
            })
            status_file.write_text(json.dumps(status, indent=2), encoding="utf-8")
        return jsonify({"ok": True, "message": "Training stopped."}), 200

    @app.route("/training/open-results", methods=["POST"])
    def training_open_results():
        run_dir.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(str(run_dir))
        return jsonify({"ok": True, "path": str(run_dir)}), 200

    @app.route("/training/artifact/<name>", methods=["GET"])
    def training_artifact(name: str):
        path = artifacts.get(name)
        if not path or not path.exists():
            return jsonify({"error": "Artifact not found"}), 404
        return send_file(path, mimetype="image/png")
