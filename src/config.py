from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"

DATASET_PATH = DATA_DIR / "diabetes.csv"
MODEL_PATH = MODELS_DIR / "diabetes_model.joblib"
METRICS_PATH = MODELS_DIR / "training_metrics.json"

FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET_COLUMN = "Outcome"

RANDOM_STATE = 42
TEST_SIZE = 0.2
DECISION_THRESHOLD = 0.5

