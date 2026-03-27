# Application decisionnelle de depistage du diabete type 2

Application PFE (Python + Machine Learning + Streamlit) pour estimer le risque de diabete de type 2 a partir de donnees cliniques.

## Stack
- Python
- pandas / NumPy
- scikit-learn
- Streamlit
- matplotlib / seaborn

## Architecture (3 couches)
1. Donnees
- Chargement et preparation des donnees (`src/data.py`)
- Dataset attendu dans `data/raw/diabetes.csv`

2. Logique metier ML
- Entrainement, comparaison de modeles, evaluation (`src/modeling.py`)
- Explicabilite globale + locale (`src/explainability.py`)
- Sauvegarde du modele et metriques (`models/`)

3. Presentation
- Interface web Streamlit (`app.py`)

## Installation
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset attendu
- Nom de fichier: `diabetes.csv`
- Emplacement: `data/raw/diabetes.csv`
- Colonnes attendues:
  - `Pregnancies`
  - `Glucose`
  - `BloodPressure`
  - `SkinThickness`
  - `Insulin`
  - `BMI`
  - `DiabetesPedigreeFunction`
  - `Age`
  - `Outcome`

## Entrainement
```bash
python scripts/train.py
```

Sorties:
- Modele: `models/diabetes_model.joblib`
- Metriques: `models/training_metrics.json`

## Lancer l'application web
```bash
streamlit run app.py
```

## Indicateurs suivis
- AUC-ROC
- Sensibilite
- Specificite
- Accuracy / Precision / F1

## Notes
- Le zero est traite comme valeur manquante pour `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`.
- L'application fournit une aide a la decision interpretable (impacts locaux par variable), pas un diagnostic medical.

