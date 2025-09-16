#!/usr/bin/env python3
"""
Customer Churn Prediction Tool

This script provides functionality for training a churn prediction model and
using it to make predictions on new data.

Usage (preferred):
    - For training: python -m src.main train [--pos-label <label>]
    - For prediction: python -m src.main predict --input <input_data_path>

Legacy (if running directly as a script from the file's directory):
    - For training: python main.py train
    - For prediction: python main.py predict --input <input_data_path>
"""


import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import subprocess
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Use non-interactive backend automatically when running headless (no DISPLAY)
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    # When executed as a module: python -m src.ml_service.main
    from .convert_arff_to_csv import convert_arff_to_csv
except Exception:
    # Fallback when running the file directly: python main.py
    from convert_arff_to_csv import convert_arff_to_csv


# Constants
# Anchor all paths to the project root so execution is robust to the CWD.
# Assuming this file lives at <project_root>/src/main.py
ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
ARFF_PATH = DATA_DIR / "dataset.arff"
CSV_PATH = DATA_DIR / "dataset.csv"
MODEL_PATH = MODELS_DIR / "churn_model.joblib"
TARGET_COLUMN = "churn"  # Adjust based on your dataset
LOG_PATH = ROOT_DIR / "churn_prediction.log"
PII_CANDIDATE_COLUMNS = ["customerid", "customer_id", "accountid", "account_id"]


def ensure_dirs():
    """Ensure necessary directories exist"""
    MODELS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)


def ensure_csv_exists():
    """Convert ARFF to CSV if needed"""
    if not CSV_PATH.exists() and ARFF_PATH.exists():
        logging.getLogger(__name__).info("Converting ARFF to CSV...")
        convert_arff_to_csv(str(ARFF_PATH), str(CSV_PATH))
    elif not CSV_PATH.exists():
        logging.getLogger(__name__).error(f"Neither {CSV_PATH} nor {ARFF_PATH} found")
        sys.exit(1)


def load_data():
    """Load data from CSV file"""
    ensure_csv_exists()
    try:
        # Normalize common NA tokens so dtype inference works (e.g., '?' in ARFF-derived CSVs)
        na_tokens = ["?", "NA", "N/A", "na", "n/a", "None", "null", "Null", "NULL", ""]
        df = pd.read_csv(CSV_PATH, na_values=na_tokens, keep_default_na=True)
        logging.getLogger(__name__).info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        logging.getLogger(__name__).exception(f"Error loading data: {str(e)}")
        sys.exit(1)


def _deep_update(base: dict, updates: dict) -> dict:
    """Recursively update nested dicts."""
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_run_config(config_path: str = None) -> dict:
    """Load run configuration from JSON file, with sensible defaults."""
    defaults = {
        "random_state": 42,
        "splits": {"test_size": 0.2, "val_size": 0.25},
        "grid_search": {
            "cv": 3,
            "scoring": "roc_auc",
            "param_grid": {
                "classifier__n_estimators": [50, 100],
                "classifier__max_depth": [None, 10, 20],
            },
        },
    }
    if not config_path:
        return defaults
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        merged = _deep_update(defaults, user_cfg)
        logging.getLogger(__name__).info(f"Loaded config from {config_path}")
        return merged
    except Exception as e:
        logging.getLogger(__name__).warning(
            f"Failed to load config from {config_path}: {e}. Using defaults."
        )
        return defaults


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file's contents."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_git_sha(root_dir: Path) -> str:
    """Return current git SHA if repo; else empty string."""
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root_dir))
            .decode()
            .strip()
        )
        return sha
    except Exception:
        return ""


def split_data(df, training=True, positive_label=None):
    """Split features and target; model pipeline handles preprocessing.

    Deterministic label encoding: map positive class to 1 and negative to 0.
    Update POSITIVE_CLASS label below if your dataset differs.
    """
    # Normalize target column name case-insensitively
    df = df.copy()
    if TARGET_COLUMN not in df.columns:
        lower_map = {c.lower(): c for c in df.columns}
        if TARGET_COLUMN.lower() in lower_map:
            original = lower_map[TARGET_COLUMN.lower()]
            df = df.rename(columns={original: TARGET_COLUMN})
        else:
            logging.getLogger(__name__).warning(
                f"Target column '{TARGET_COLUMN}' not found in data."
            )
            if training:
                logging.getLogger(__name__).error(
                    f"Cannot train without target column. Available columns: {list(df.columns)}"
                )
                sys.exit(1)

    # Split features and target
    if TARGET_COLUMN in df.columns:
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN] if training else None
    else:
        X = df
        y = None

    # Deterministic target encoding to {0,1}
    if training and y is not None:
        # Determine positive label
        if positive_label is None:
            # Auto-detect from common labels
            if y.dtype == "object" or str(y.dtype) == "category":
                candidates = ["Churn", "Yes", "True", "1"]
                y_str = y.astype(str)
                for cand in candidates:
                    if cand in set(y_str.unique()):
                        positive_label = cand
                        break
                if positive_label is None:
                    # Fallback: majority class becomes negative, minority positive
                    counts = y_str.value_counts()
                    positive_label = counts.idxmin()
                y = (y_str == str(positive_label)).astype(int)
            else:
                positive_label = 1
                y = (y == 1).astype(int)
        else:
            # Use provided positive label
            if y.dtype == "object" or str(y.dtype) == "category":
                # Validate provided label exists
                if str(positive_label) not in set(y.astype(str).unique()):
                    available = set(y.astype(str).unique())
                    print(
                        f"Error: Provided --pos-label '{positive_label}' not found in "
                        f"target values: {available}"
                    )
                    sys.exit(1)
                y = (y.astype(str) == str(positive_label)).astype(int)
            else:
                y = (y == positive_label).astype(int)

        # Validate both classes present
        unique_vals = set(pd.Series(y).unique())
        if unique_vals != {0, 1}:
            print(
                f"Error: After encoding, target classes are {unique_vals}. "
                f"Need both 0 and 1 to train."
            )
            sys.exit(1)

    return (X, y) if training else X


def train_model(positive_label=None, show_plots=False, config: dict = None):
    """Train the churn prediction model"""
    logging.getLogger(__name__).info("Starting model training...")
    if config is None:
        config = load_run_config()

    # Versioned run identifier and artifact paths
    run_id = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    diagnostics_path = MODELS_DIR / f"model_diagnostics_{run_id}.txt"
    cm_val_path = MODELS_DIR / f"confusion_matrix_val_{run_id}.png"
    feat_path = MODELS_DIR / f"feature_importance_{run_id}.png"
    roc_val_path = MODELS_DIR / f"roc_curve_val_{run_id}.png"
    cm_test_path = MODELS_DIR / f"confusion_matrix_test_{run_id}.png"
    roc_test_path = MODELS_DIR / f"roc_curve_test_{run_id}.png"
    versioned_model_path = MODELS_DIR / f"churn_model_{run_id}.joblib"

    # Load and preprocess data
    df = load_data()
    # Normalize target column case for training-time metadata
    if TARGET_COLUMN not in df.columns:
        lower_map = {c.lower(): c for c in df.columns}
        if TARGET_COLUMN.lower() in lower_map:
            df = df.rename(columns={lower_map[TARGET_COLUMN.lower()]: TARGET_COLUMN})
    # Determine effective positive label for metadata (mirrors split_data logic)
    effective_pos_label = positive_label
    if effective_pos_label is None and TARGET_COLUMN in df.columns:
        y_raw = df[TARGET_COLUMN]
        if y_raw.dtype == "object" or str(y_raw.dtype) == "category":
            candidates = ["Churn", "Yes", "True", "1"]
            y_str = y_raw.astype(str)
            for cand in candidates:
                if cand in set(y_str.unique()):
                    effective_pos_label = cand
                    break
            if effective_pos_label is None:
                counts = y_str.value_counts()
                effective_pos_label = counts.idxmin()
        else:
            effective_pos_label = 1
    # Now encode with the same effective positive label
    X, y = split_data(df, training=True, positive_label=effective_pos_label)

    # Split data into train/validation/test sets with stratification
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=config["splits"]["test_size"],
        random_state=config["random_state"],
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=config["splits"]["val_size"],
        random_state=config["random_state"],
        stratify=y_trainval,
    )

    # Identify numeric and categorical columns from training data
    numeric_features = X.select_dtypes(include=[np.number]).columns
    categorical_features = X.select_dtypes(include=["object", "category"]).columns

    # Create preprocessing pipelines
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Create model pipeline with preprocessing
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=config["random_state"])),
        ]
    )

    # Define hyperparameter grid
    param_grid = config["grid_search"]["param_grid"]

    # Perform grid search
    logging.getLogger(__name__).info("Performing hyperparameter tuning...")
    grid_search = GridSearchCV(
        model,
        param_grid=param_grid,
        cv=config["grid_search"]["cv"],
        n_jobs=-1,
        verbose=1,
        scoring=config["grid_search"]["scoring"],
    )
    grid_search.fit(X_train, y_train)

    # Get best model
    best_model = grid_search.best_estimator_

    # Evaluate on validation set
    y_pred = best_model.predict(X_val)
    # Use the pipeline's classes_ to locate the positive class index
    classes = best_model.named_steps["classifier"].classes_
    if 1 not in set(classes):
        print(f"Error: Trained classifier classes {classes} do not include positive class '1'.")
        sys.exit(1)
    # We trained with y in {0,1}, so positive class is 1
    pos_index = int(np.where(classes == 1)[0][0])
    y_pred_proba = best_model.predict_proba(X_val)[:, pos_index]

    # Print evaluation metrics
    logging.getLogger(__name__).info("\nModel Performance:")
    logging.getLogger(__name__).info(f"Accuracy (val): {accuracy_score(y_val, y_pred):.4f}")
    logging.getLogger(__name__).info(f"ROC AUC (val): {roc_auc_score(y_val, y_pred_proba):.4f}")
    logging.getLogger(__name__).info(
        "\nClassification Report (val):\n" + classification_report(y_val, y_pred)
    )

    # Save model diagnostics to file
    # diagnostics_path already versioned above
    with open(diagnostics_path, "w") as f:
        f.write("MODEL DIAGNOSTICS REPORT\n")
        f.write("=======================\n\n")
        f.write(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}\n")
        f.write(f"ROC AUC: {roc_auc_score(y_val, y_pred_proba):.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_val, y_pred))
        f.write("\nConfusion Matrix:\n")
        f.write(f"{confusion_matrix(y_val, y_pred)}\n\n")
        f.write("Best Model Parameters:\n")
        f.write(f"{best_model['classifier'].get_params()}\n\n")

        # Feature analysis
        if hasattr(best_model["classifier"], "feature_importances_"):
            importances = best_model["classifier"].feature_importances_
            indices = np.argsort(importances)[::-1]
            f.write("Feature Importance Ranking:\n")
            for i, idx in enumerate(indices[:20]):  # Top 20 features
                f.write(f"{i+1}. Feature #{idx}: {importances[idx]:.4f}\n")

    logging.getLogger(__name__).info(f"Detailed diagnostics saved to: {diagnostics_path}")

    # Confusion Matrix Visualization (validation)
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Churned", "Churned"],
        yticklabels=["Not Churned", "Churned"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (Validation)")
    plt.tight_layout()
    plt.savefig(cm_val_path)
    logging.getLogger(__name__).info(f"Confusion matrix (val) plot saved to: {cm_val_path}")
    if show_plots:
        plt.show()
    else:
        plt.close()

    # Feature Importance
    if hasattr(best_model["classifier"], "feature_importances_"):
        # Get feature importances from the trained model
        importances = best_model["classifier"].feature_importances_

        # Use fitted preprocessor to get readable feature names --------
        pre = best_model.named_steps["preprocessor"]

        # Extract the original column lists from the fitted ColumnTransformer
        numeric_cols, categorical_cols = [], []
        for name, trans, cols in pre.transformers_:
            if name == "num":
                numeric_cols = list(cols)
            elif name == "cat":
                categorical_cols = list(cols)

        # Get expanded one-hot names for categorical columns
        try:
            ohe = pre.named_transformers_["cat"].named_steps["onehot"]
            cat_feature_names = ohe.get_feature_names_out(categorical_cols)
        except Exception:
            # Fallback (shouldn't happen with a fitted OneHotEncoder)
            cat_feature_names = categorical_cols

        feature_names = list(numeric_cols) + list(cat_feature_names)

        # Safety fallback if there is a length mismatch
        if len(feature_names) != len(importances):
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        # Sort by importance
        indices = np.argsort(importances)[::-1]
        top_n = min(20, len(indices))
        top_idx = indices[:top_n]
        features = [feature_names[i] for i in top_idx]
        importance_values = importances[top_idx]

        # Create feature importance plot (path defined above)
        plt.figure(figsize=(12, 8))
        plt.title("Feature Importance Rankings", fontsize=14)

        # Create horizontal bar chart
        plt.barh(range(top_n), importance_values, color="steelblue", alpha=0.8)
        plt.yticks(range(top_n), features, fontsize=11)
        plt.xlabel("Relative Importance", fontsize=12)

        # Add value labels
        for i, v in enumerate(importance_values):
            plt.text(v + 0.01, i, f"{v:.4f}", va="center", fontsize=10)

        plt.grid(axis="x", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(feat_path)
        logging.getLogger(__name__).info(f"Feature importance plot saved to: {feat_path}")
        if show_plots:
            plt.show()
        else:
            plt.close()

        # Save readable feature importance analysis to file
        with open(diagnostics_path, "a") as f:
            f.write("\n\nFEATURE IMPORTANCE ANALYSIS\n")
            f.write("=========================\n\n")
            f.write("Top Features (after preprocessing):\n")
            for rank, i in enumerate(top_idx, 1):
                f.write(f"  {rank}. {feature_names[i]}: {importances[i]:.6f}\n")

    # ROC Curve (validation)
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Validation)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_val_path)
    logging.getLogger(__name__).info(f"ROC (val) curve saved to: {roc_val_path}")
    if show_plots:
        plt.show()
    else:
        plt.close()

    # Save training schema/metadata for inference validation and lineage
    try:
        metadata = {
            "target_column": TARGET_COLUMN,
            "positive_label": effective_pos_label,
            "numeric_columns": list(numeric_features),
            "categorical_columns": list(categorical_features),
            "feature_columns": list(numeric_features) + list(categorical_features),
            "run_id": run_id,
            "data_csv": str(CSV_PATH),
            "data_sha256": compute_file_sha256(CSV_PATH) if CSV_PATH.exists() else "",
            "git_sha": get_git_sha(ROOT_DIR),
            "artifacts": {
                "diagnostics": str(diagnostics_path),
                "confusion_matrix_val": str(cm_val_path),
                "roc_curve_val": str(roc_val_path),
                "feature_importance": str(feat_path),
                "confusion_matrix_test": str(cm_test_path),
                "roc_curve_test": str(roc_test_path),
            },
            "model_path": str(MODELS_DIR / "churn_model.joblib"),
            "model_versioned_path": str(versioned_model_path),
        }
        meta_path = MODELS_DIR / "model_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(metadata, mf, indent=2)
        logging.getLogger(__name__).info(f"Saved model metadata to: {meta_path}")
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to save model metadata: {e}")

    # Evaluate on test set
    y_test_pred = best_model.predict(X_test)
    classes_test = best_model.named_steps["classifier"].classes_
    if 1 not in set(classes_test):
        logging.getLogger(__name__).error(
            f"Trained classifier classes {classes_test} do not include positive class '1'."
        )
        sys.exit(1)
    pos_index_test = int(np.where(classes_test == 1)[0][0])
    y_test_pred_proba = best_model.predict_proba(X_test)[:, pos_index_test]

    # Append test metrics to diagnostics
    with open(diagnostics_path, "a") as f:
        f.write("\nTEST SET PERFORMANCE\n")
        f.write("=====================\n\n")
        f.write(f"Accuracy (test): {accuracy_score(y_test, y_test_pred):.4f}\n")
        f.write(f"ROC AUC (test): {roc_auc_score(y_test, y_test_pred_proba):.4f}\n\n")
        f.write("Classification Report (test):\n")
        f.write(classification_report(y_test, y_test_pred))
        f.write("\nConfusion Matrix (test):\n")
        f.write(f"{confusion_matrix(y_test, y_test_pred)}\n")

    # Confusion Matrix (test) path defined above
    plt.figure(figsize=(8, 6))
    cm_test = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(
        cm_test,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Churned", "Churned"],
        yticklabels=["Not Churned", "Churned"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (Test)")
    plt.tight_layout()
    plt.savefig(cm_test_path)
    logging.getLogger(__name__).info(f"Confusion matrix (test) plot saved to: {cm_test_path}")
    if show_plots:
        plt.show()
    else:
        plt.close()

    # ROC Curve (test) path defined above
    plt.figure(figsize=(8, 6))
    fpr_t, tpr_t, _ = roc_curve(y_test, y_test_pred_proba)
    roc_auc_t = auc(fpr_t, tpr_t)
    plt.plot(fpr_t, tpr_t, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc_t:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_test_path)
    logging.getLogger(__name__).info(f"ROC (test) curve saved to: {roc_test_path}")
    if show_plots:
        plt.show()
    else:
        plt.close()

    # Save the model (stable + versioned copy)
    logging.getLogger(__name__).info(f"Saving model to {MODEL_PATH} and {versioned_model_path}...")
    joblib.dump(best_model, MODEL_PATH)
    try:
        joblib.dump(best_model, versioned_model_path)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to save versioned model: {e}")

    # Save environment requirements snapshot
    try:
        reqs = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode()
        req_path = MODELS_DIR / f"requirements_{run_id}.txt"
        with open(req_path, "w", encoding="utf-8") as rf:
            rf.write(reqs)
        logging.getLogger(__name__).info(f"Saved environment requirements to: {req_path}")
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to capture environment requirements: {e}")

    # Save the effective config used for this run
    try:
        cfg_path = MODELS_DIR / f"run_config_{run_id}.json"
        with open(cfg_path, "w", encoding="utf-8") as cf:
            json.dump(config, cf, indent=2)
        logging.getLogger(__name__).info(f"Saved run config to: {cfg_path}")
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to save run config: {e}")

    logging.getLogger(__name__).info("Training completed successfully!")
    return best_model


def predict(input_path=None, drop_pii=False):
    """Make predictions using the trained model"""
    if not MODEL_PATH.exists():
        logging.getLogger(__name__).error(
            f"Model not found at {MODEL_PATH}. Please train the model first."
        )
        return

    # Load the trained model
    logging.getLogger(__name__).info(f"Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)

    # Load input data
    try:
        if input_path:
            na_tokens = ["?", "NA", "N/A", "na", "n/a", "None", "null", "Null", "NULL", ""]
            df = pd.read_csv(input_path, na_values=na_tokens, keep_default_na=True)
        else:
            # If no input path is provided, prompt user for input
            logging.getLogger(__name__).warning(
                "No input data provided. Please enter values for each feature:"
            )
            # Here you would implement an interactive feature input
            # For simplicity, we'll just exit
            logging.getLogger(__name__).warning(
                "Interactive input not implemented. Please provide an input CSV file."
            )
            return
    except Exception as e:
        logging.getLogger(__name__).exception(f"Error loading input data: {str(e)}")
        return

    # Preprocess input data
    # Prepare input features (model pipeline will handle preprocessing)
    X = split_data(df, training=False)

    # Align columns to training schema
    meta_path = MODELS_DIR / "model_metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as mf:
                metadata = json.load(mf)
            expected = metadata.get("feature_columns", [])
            if expected:
                missing = [c for c in expected if c not in X.columns]
                extra = [c for c in X.columns if c not in expected]
                if missing:
                    print(f"Note: Adding missing columns with NA for inference: {missing}")
                    for c in missing:
                        X[c] = np.nan
                if extra:
                    print(f"Note: Dropping unexpected columns not seen in training: {extra}")
                    X = X.drop(columns=extra)
                # Reorder to match training
                X = X[expected]
        except Exception as e:
            print(f"Warning: Failed to load/parse model metadata for alignment: {e}")
    else:
        print("Warning: Model metadata not found; proceeding without schema alignment.")

    # Make predictions
    # Compute probabilities with respect to the positive class (1)
    classes = model.named_steps["classifier"].classes_
    if 1 not in set(classes):
        logging.getLogger(__name__).error(
            f"Loaded classifier classes {classes} do not include positive class '1'."
        )
        logging.getLogger(__name__).error(
            "If you retrained with a different encoding, please re-run training "
            "or update the pipeline."
        )
        return
    pos_index = int(np.where(classes == 1)[0][0])
    predictions_proba = model.predict_proba(X)[:, pos_index]
    predictions = model.predict(X)
    class_labels = predictions  # predictions are already 0/1 aligned with training

    # Create results DataFrame
    results = df.copy()
    # Optionally remove likely PII columns from output
    if drop_pii:
        lower_cols = {c.lower(): c for c in results.columns}
        to_drop = [lower_cols[c] for c in lower_cols.keys() if c in set(PII_CANDIDATE_COLUMNS)]
        if to_drop:
            logging.getLogger(__name__).info(f"Dropping likely PII columns from output: {to_drop}")
            results = results.drop(columns=to_drop)
    results["churn_probability"] = predictions_proba
    results["predicted_churn"] = class_labels

    # Display and save results
    logging.getLogger(__name__).info(
        "\nChurn Predictions (preview):\n"
        + str(results[["predicted_churn", "churn_probability"]].head(10))
    )

    # Save predictions
    output_path = ROOT_DIR / "predictions.csv"
    results.to_csv(output_path, index=False)
    logging.getLogger(__name__).info(f"Full predictions saved to {output_path}")


def main():
    """Main function to parse arguments and run appropriate tasks"""
    parser = argparse.ArgumentParser(description="Customer Churn Prediction Tool")
    parser.add_argument("action", choices=["train", "predict"], help="Action to perform")
    parser.add_argument(
        "--pos-label", help="Positive class label for training (e.g., Yes, Churn, 1)"
    )
    parser.add_argument(
        "--show-plots", action="store_true", help="Force display plots interactively"
    )
    parser.add_argument(
        "--no-show-plots", action="store_true", help="Force disable interactive plot display"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument("--input", help="Path to input data for prediction (required for predict)")
    parser.add_argument(
        "--no-pii", action="store_true", help="Drop likely PII columns from prediction outputs"
    )
    parser.add_argument("--config", help="Path to JSON config file for training options")
    args = parser.parse_args()

    ensure_dirs()

    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, args.log_level))
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, args.log_level))
    ch.setFormatter(formatter)
    # Rotating file handler
    fh = RotatingFileHandler(LOG_PATH, maxBytes=5_000_000, backupCount=3)
    fh.setLevel(getattr(logging, args.log_level))
    fh.setFormatter(formatter)
    # Avoid duplicate handlers if main() is called more than once
    logger.handlers = []
    logger.addHandler(ch)
    logger.addHandler(fh)

    if args.action == "train":
        # Default to showing plots when a DISPLAY is available unless explicitly disabled
        default_show = bool(os.environ.get("DISPLAY"))
        show_flag = args.show_plots or (default_show and not args.no_show_plots)
        cfg = load_run_config(args.config)
        train_model(positive_label=args.pos_label, show_plots=show_flag, config=cfg)
    elif args.action == "predict":
        predict(args.input, drop_pii=args.no_pii)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
