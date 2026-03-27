from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE


def compute_reference_values(x_train: pd.DataFrame) -> dict[str, float]:
    return x_train.median(numeric_only=True).to_dict()


def compute_global_importance(
    model: Pipeline,
    x_eval: pd.DataFrame,
    y_eval: pd.Series,
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        x_eval,
        y_eval,
        n_repeats=20,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=-1,
    )
    importance_df = pd.DataFrame(
        {
            "feature": x_eval.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values(by="importance_mean", ascending=False)
    return importance_df.reset_index(drop=True)


def compute_local_impacts(
    model: Pipeline,
    input_row: pd.DataFrame,
    reference_values: dict[str, float],
) -> tuple[float, pd.DataFrame]:
    """
    Calculates per-feature impact using a counterfactual one-feature-at-a-time method:
    replace one feature with its reference median and measure probability delta.
    """
    current_proba = float(model.predict_proba(input_row)[:, 1][0])
    impacts: list[dict[str, Any]] = []

    for feature, reference in reference_values.items():
        modified = input_row.copy()
        modified[feature] = reference
        ref_proba = float(model.predict_proba(modified)[:, 1][0])
        impacts.append(
            {
                "feature": feature,
                "impact_on_risk": current_proba - ref_proba,
            }
        )

    impacts_df = pd.DataFrame(impacts)
    impacts_df["abs_impact"] = impacts_df["impact_on_risk"].abs()
    impacts_df = impacts_df.sort_values(by="abs_impact", ascending=False)
    return current_proba, impacts_df.reset_index(drop=True)

