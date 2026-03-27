from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# Allow `python scripts/train.py` from project root without package install.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATASET_PATH,
    DECISION_THRESHOLD,
    FEATURE_COLUMNS,
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.data import load_dataset, split_features_target
from src.explainability import compute_global_importance, compute_reference_values
from src.modeling import evaluate_on_test, train_best_model


def save_artifacts(
    model,
    best_model_name: str,
    leaderboard: pd.DataFrame,
    metrics: dict[str, float],
    reference_values: dict[str, float],
    global_importance: pd.DataFrame,
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "best_model_name": best_model_name,
        "threshold": DECISION_THRESHOLD,
        "reference_values": reference_values,
        "global_importance": global_importance.to_dict(orient="records"),
        "cv_leaderboard": leaderboard.to_dict(orient="records"),
        "test_metrics": metrics,
    }
    joblib.dump(artifact, MODEL_PATH)

    metrics_payload = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(DATASET_PATH),
        "best_model_name": best_model_name,
        "decision_threshold": DECISION_THRESHOLD,
        "cv_leaderboard": artifact["cv_leaderboard"],
        "test_metrics": metrics,
    }
    with Path(METRICS_PATH).open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    print("[1/5] Chargement du dataset...")
    df = load_dataset(DATASET_PATH)
    x, y = split_features_target(df)

    print("[2/5] Separation train/test...")
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("[3/5] Entrainement + selection du meilleur modele...")
    model, best_model_name, leaderboard = train_best_model(
        x_train=x_train,
        y_train=y_train,
        feature_columns=FEATURE_COLUMNS,
    )

    print("[4/5] Evaluation et explicabilite...")
    test_metrics = evaluate_on_test(model, x_test, y_test, threshold=DECISION_THRESHOLD)
    reference_values = compute_reference_values(x_train)
    global_importance = compute_global_importance(model, x_test, y_test)

    print("[5/5] Sauvegarde des artefacts...")
    save_artifacts(
        model=model,
        best_model_name=best_model_name,
        leaderboard=leaderboard,
        metrics=test_metrics,
        reference_values=reference_values,
        global_importance=global_importance,
    )

    print("\nEntrainement termine.")
    print(f"Meilleur modele: {best_model_name}")
    for metric_name, metric_value in test_metrics.items():
        print(f"- {metric_name}: {metric_value:.4f}")
    print(f"\nModele: {MODEL_PATH}")
    print(f"Metriques: {METRICS_PATH}")


if __name__ == "__main__":
    main()
