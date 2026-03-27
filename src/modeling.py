from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import DECISION_THRESHOLD, RANDOM_STATE


def build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("num", numeric_pipeline, feature_columns)],
        remainder="drop",
    )


def get_candidate_models() -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
    }


def evaluate_models_cv(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    feature_columns: list[str],
) -> pd.DataFrame:
    preprocessor = build_preprocessor(feature_columns)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    specificity_scorer = make_scorer(recall_score, pos_label=0)

    rows: list[dict[str, float | str]] = []
    for model_name, estimator in get_candidate_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", estimator),
            ]
        )
        scores = cross_validate(
            pipeline,
            x_train,
            y_train,
            cv=cv,
            scoring={
                "auc_roc": "roc_auc",
                "sensitivity": "recall",
                "specificity": specificity_scorer,
            },
            n_jobs=-1,
        )
        rows.append(
            {
                "model": model_name,
                "cv_auc_roc_mean": float(np.mean(scores["test_auc_roc"])),
                "cv_sensitivity_mean": float(np.mean(scores["test_sensitivity"])),
                "cv_specificity_mean": float(np.mean(scores["test_specificity"])),
            }
        )

    leaderboard = pd.DataFrame(rows).sort_values(
        by="cv_auc_roc_mean",
        ascending=False,
    )
    return leaderboard.reset_index(drop=True)


def train_best_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    feature_columns: list[str],
) -> tuple[Pipeline, str, pd.DataFrame]:
    leaderboard = evaluate_models_cv(x_train, y_train, feature_columns)
    best_model_name = str(leaderboard.iloc[0]["model"])
    best_estimator = get_candidate_models()[best_model_name]
    preprocessor = build_preprocessor(feature_columns)

    best_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", best_estimator),
        ]
    )
    best_pipeline.fit(x_train, y_train)
    return best_pipeline, best_model_name, leaderboard


def evaluate_on_test(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, float]:
    y_proba = model.predict_proba(x_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "test_auc_roc": float(roc_auc_score(y_test, y_proba)),
        "test_sensitivity": float(recall_score(y_test, y_pred, pos_label=1)),
        "test_specificity": float(recall_score(y_test, y_pred, pos_label=0)),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

