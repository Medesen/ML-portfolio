import os
from pathlib import Path

# Anchor all paths to a configurable project root.
# Prefer CHURN_ROOT env var, else use current working directory.
_env_root = os.getenv("CHURN_ROOT")
ROOT_DIR = Path(_env_root).resolve() if _env_root else Path.cwd()
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
ARFF_PATH = DATA_DIR / "dataset.arff"
CSV_PATH = DATA_DIR / "dataset.csv"
MODEL_PATH = MODELS_DIR / "churn_model.joblib"
TARGET_COLUMN = "churn"
LOG_PATH = ROOT_DIR / "churn_prediction.log"
PII_CANDIDATE_COLUMNS = ["customerid", "customer_id", "accountid", "account_id"]
