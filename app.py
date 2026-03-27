from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import DECISION_THRESHOLD, FEATURE_COLUMNS, MODEL_PATH
from src.explainability import compute_local_impacts
from src.predict import build_input_row, load_artifact


st.set_page_config(
    page_title="Depistage Diabete Type 2",
    page_icon="D2",
    layout="wide",
)

st.title("Application d'aide au depistage du diabete de type 2")
st.caption(
    "Outil decisionnel base sur des donnees cliniques (Pima Indians Diabetes Dataset)."
)


@st.cache_resource
def get_artifact():
    return load_artifact(MODEL_PATH)


def render_metrics(artifact: dict) -> None:
    st.subheader("Performance du modele")
    metrics = artifact.get("test_metrics", {})
    if not metrics:
        st.info("Metriques indisponibles. Relancez l'entrainement.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("AUC-ROC", f"{metrics.get('test_auc_roc', 0):.3f}")
    c2.metric("Sensibilite", f"{metrics.get('test_sensitivity', 0):.3f}")
    c3.metric("Specificite", f"{metrics.get('test_specificity', 0):.3f}")


def render_input_form() -> dict[str, float]:
    with st.form("patient_form"):
        st.subheader("Parametres patient")
        c1, c2 = st.columns(2)
        with c1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
            glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
            blood_pressure = st.number_input(
                "BloodPressure", min_value=0, max_value=200, value=70
            )
            skin_thickness = st.number_input(
                "SkinThickness", min_value=0, max_value=100, value=20
            )
        with c2:
            insulin = st.number_input("Insulin", min_value=0, max_value=900, value=79)
            bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=28.0)
            pedigree = st.number_input(
                "DiabetesPedigreeFunction",
                min_value=0.05,
                max_value=3.0,
                value=0.47,
                step=0.01,
            )
            age = st.number_input("Age", min_value=18, max_value=120, value=33)

        submitted = st.form_submit_button("Estimer le risque")

    if not submitted:
        return {}

    return {
        "Pregnancies": float(pregnancies),
        "Glucose": float(glucose),
        "BloodPressure": float(blood_pressure),
        "SkinThickness": float(skin_thickness),
        "Insulin": float(insulin),
        "BMI": float(bmi),
        "DiabetesPedigreeFunction": float(pedigree),
        "Age": float(age),
    }


def render_prediction(artifact: dict, patient_data: dict[str, float]) -> None:
    model = artifact["model"]
    input_row = build_input_row(patient_data)
    probability = float(model.predict_proba(input_row)[:, 1][0])
    prediction = int(probability >= DECISION_THRESHOLD)

    st.subheader("Resultat")
    st.metric("Score de risque", f"{probability * 100:.1f}%")
    if prediction == 1:
        st.error("Risque eleve (selon le seuil de decision)")
    else:
        st.success("Risque faible (selon le seuil de decision)")

    st.caption(
        "Ce score est une aide a la decision et ne remplace pas un diagnostic medical."
    )

    reference_values = artifact.get("reference_values", {})
    if reference_values:
        _, impacts_df = compute_local_impacts(model, input_row, reference_values)
        st.subheader("Facteurs influents (patient courant)")
        display_df = impacts_df[["feature", "impact_on_risk"]].copy()
        display_df["impact_on_risk"] = display_df["impact_on_risk"] * 100
        st.bar_chart(
            display_df.set_index("feature"),
            x_label="Feature",
            y_label="Impact sur le risque (%)",
        )


def render_global_importance(artifact: dict) -> None:
    st.subheader("Importance globale des variables")
    importance = artifact.get("global_importance", [])
    if not importance:
        st.info("Importance globale indisponible.")
        return

    importance_df = pd.DataFrame(importance)
    if "feature" not in importance_df.columns or "importance_mean" not in importance_df.columns:
        st.info("Format d'importance globale invalide.")
        return
    st.dataframe(
        importance_df[["feature", "importance_mean", "importance_std"]],
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    if not MODEL_PATH.exists():
        st.warning(
            "Modele non trouve. Placez d'abord le dataset dans "
            "`data/raw/diabetes.csv`, puis executez `python scripts/train.py`."
        )
        return

    artifact = get_artifact()
    st.sidebar.subheader("Modele actif")
    st.sidebar.write(f"Nom: `{artifact.get('best_model_name', 'inconnu')}`")
    st.sidebar.write(f"Seuil: `{artifact.get('threshold', DECISION_THRESHOLD):.2f}`")
    st.sidebar.write("Features:")
    st.sidebar.write(", ".join(FEATURE_COLUMNS))

    render_metrics(artifact)
    patient_data = render_input_form()
    if patient_data:
        render_prediction(artifact, patient_data)
    st.divider()
    render_global_importance(artifact)


if __name__ == "__main__":
    main()
