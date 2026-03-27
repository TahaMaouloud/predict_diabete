from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATASET_PATH, FEATURE_COLUMNS, TARGET_COLUMN


ZERO_IS_MISSING_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


def load_dataset(dataset_path: Path | str = DATASET_PATH) -> pd.DataFrame:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable: {path}. "
            "Ajoutez le fichier Kaggle/UCI puis relancez l'entrainement."
        )

    df = pd.read_csv(path)
    expected = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = sorted(expected.difference(df.columns))
    if missing:
        raise ValueError(
            "Colonnes manquantes dans le dataset: "
            + ", ".join(missing)
            + "."
        )

    df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()

    # Force numeric parsing for robust handling of Kaggle/UCI variants.
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ZERO_IS_MISSING_COLUMNS:
        df[col] = df[col].replace(0, np.nan)

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"La colonne cible '{TARGET_COLUMN}' contient des valeurs invalides."
        )

    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int).copy()
    return x, y
