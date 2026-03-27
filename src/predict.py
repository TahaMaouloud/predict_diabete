from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import DECISION_THRESHOLD, FEATURE_COLUMNS, MODEL_PATH


def load_artifact(model_path: Path | str = MODEL_PATH) -> dict[str, Any]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Modele non trouve: {path}. Lancez d'abord scripts/train.py."
        )
    artifact: dict[str, Any] = joblib.load(path)
    return artifact


def build_input_row(patient_data: dict[str, float]) -> pd.DataFrame:
    missing = [c for c in FEATURE_COLUMNS if c not in patient_data]
    if missing:
        raise ValueError(
            "Champs manquants pour la prediction: " + ", ".join(missing)
        )
    row = pd.DataFrame([patient_data], columns=FEATURE_COLUMNS)
    return row


def predict_risk(
    artifact: dict[str, Any],
    patient_data: dict[str, float],
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, float | str]:
    model = artifact["model"]
    input_row = build_input_row(patient_data)
    risk_prob = float(model.predict_proba(input_row)[:, 1][0])
    prediction = int(risk_prob >= threshold)

    return {
        "risk_probability": risk_prob,
        "prediction": prediction,
        "risk_level": "Risque eleve" if prediction == 1 else "Risque faible",
    }

