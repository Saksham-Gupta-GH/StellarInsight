from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from backend.starfinder.domain import DomainSpec, validate_payload
from backend.starfinder.schema import LABEL_ID_TO_NAME, PREDICTION_WARNING_MESSAGE


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT_DIR = _project_root()
STATIC_DIR = ROOT_DIR / "static"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"


def _load_domain_spec() -> DomainSpec:
    domain_path = ARTIFACTS_DIR / "domain.json"
    if not domain_path.exists():
        raise FileNotFoundError(
            f"Missing domain spec at {domain_path}. Run `python3 -m backend.train_model`."
        )
    payload = json.loads(domain_path.read_text(encoding="utf-8"))
    return DomainSpec.from_dict(payload)


def _load_model():
    model_path = ARTIFACTS_DIR / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model at {model_path}. Run `python3 -m backend.train_model`."
        )
    return joblib.load(model_path)


domain_spec = _load_domain_spec()
model = _load_model()


app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/")


@app.get("/health")
def health() -> Any:
    return jsonify({"status": "ok"})


@app.get("/")
def index() -> Any:
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.get("/schema")
def schema() -> Any:
    return jsonify(
        {
            "colors": sorted(domain_spec.allowed_colors),
            "spectral_classes": sorted(domain_spec.allowed_spectral_classes),
            "numeric_ranges": domain_spec.numeric_ranges,
            "label_map": LABEL_ID_TO_NAME,
        }
    )


@app.post("/predict")
def predict() -> Any:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"in_domain": False, "message": "Invalid JSON payload."}), 400

    validation = validate_payload(payload=payload, domain_spec=domain_spec)
    if not validation.ok:
        if validation.out_of_domain:
            return (
                jsonify(
                    {
                        "in_domain": False,
                        "message": PREDICTION_WARNING_MESSAGE,
                    }
                ),
                200,
            )
        return jsonify({"in_domain": False, "message": validation.message}), 400

    proba = model.predict_proba(validation.features)[0]
    pred_type = int(np.argmax(proba))
    confidence = float(proba[pred_type])

    return jsonify(
        {
            "in_domain": True,
            "prediction": {
                "type_id": pred_type,
                "type_name": LABEL_ID_TO_NAME[str(pred_type)],
                "confidence": confidence,
            },
        }
    )


@app.get("/assets/<path:filename>")
def assets(filename: str) -> Any:
    return send_from_directory(str(STATIC_DIR / "assets"), filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8001")))
