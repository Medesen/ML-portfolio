#!/usr/bin/env python3
"""
Customer Churn Prediction Tool

This script provides functionality for training a churn prediction model and
using it to make predictions on new data.

Usage:
    - For training: python -m ml_portfolio_churn.main train [--pos-label <label>]
    - For prediction: python -m ml_portfolio_churn.main predict --input <input_data_path>
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
from pydantic import BaseModel, Field, ValidationError, field_validator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
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

# Local module imports
from .constants import (
    CSV_PATH,
    DATA_DIR,
    LOG_PATH,
    MODEL_PATH,
    MODELS_DIR,
    PII_CANDIDATE_COLUMNS,
    ROOT_DIR,
    TARGET_COLUMN,
)
from .data_io import load_data
from .features import align_features_to_training_schema, split_data


def ensure_dirs():
    """Ensure necessary directories exist"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


"""Main CLI entrypoint and training/prediction logic."""


# (imports already performed above)


class GridSearchConfig(BaseModel):
    cv: int = Field(default=3, ge=2)
    scoring: str = Field(default="roc_auc")
    param_grid: dict = Field(
        default_factory=lambda: {
            "classifier__n_estimators": [50, 100],
            "classifier__max_depth": [None, 10, 20],
        }
    )


class SplitsConfig(BaseModel):
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    val_size: float = Field(default=0.25, gt=0.0, lt=1.0)

    @field_validator("val_size")
    @classmethod
    def validate_val_size(cls, v: float) -> float:
        # Ensure that after splitting, each split can sustain CV; keep simple constraints here
        if not (0.0 < v < 1.0):
            raise ValueError("val_size must be in (0,1)")
        return v


class RunConfig(BaseModel):
    random_state: int = 42
    splits: SplitsConfig = SplitsConfig()
    grid_search: GridSearchConfig = GridSearchConfig()


def load_run_config(config_path: str = None) -> RunConfig:
    """Load and validate run configuration from JSON using pydantic with sensible defaults."""
    if not config_path:
        return RunConfig()
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        cfg = RunConfig.model_validate(user_cfg)
        logging.getLogger(__name__).info(f"Loaded config from {config_path}")
        return cfg
    except (OSError, json.JSONDecodeError, ValidationError) as e:
        logging.getLogger(__name__).warning(
            f"Failed to load/validate config from {config_path}: {e}. Using defaults."
        )
        return RunConfig()


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


# split_data imported from local features module


def train_model(
    positive_label=None,
    show_plots=False,
    config: dict = None,
    use_mlflow: bool = False,
    mlflow_uri: str | None = None,
    registered_model_name: str | None = None,
    model_stage: str | None = None,
    calibration: str = "none",
    threshold_objective: str = "f1",
    min_precision: float | None = None,
    top_k_ratio: float | None = None,
    class_weight: str | None = None,
):
    """Train the churn prediction model"""
    logging.getLogger(__name__).info("Starting model training...")
    # Ensure required directories exist (robust when called directly from tests)
    ensure_dirs()
    if config is None:
        config = load_run_config()
    elif isinstance(config, dict):
        try:
            config = RunConfig.model_validate(config)
        except ValidationError as e:
            logging.getLogger(__name__).warning(
                f"Provided config dict failed validation ({e}); using defaults."
            )
            config = RunConfig()

    # Versioned run identifier and artifact paths
    # Use timezone-aware UTC timestamp
    try:
        run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    except AttributeError:
        # Fallback for older Python versions
        run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
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
        test_size=config.splits.test_size,
        random_state=config.random_state,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=config.splits.val_size,
        random_state=config.random_state,
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
            (
                "classifier",
                RandomForestClassifier(
                    random_state=config.random_state,
                    class_weight=class_weight if class_weight else None,
                ),
            ),
        ]
    )

    # Define hyperparameter grid
    param_grid = config.grid_search.param_grid

    # Perform grid search
    logging.getLogger(__name__).info("Performing hyperparameter tuning...")
    grid_search = GridSearchCV(
        model,
        param_grid=param_grid,
        cv=config.grid_search.cv,
        n_jobs=-1,
        verbose=1,
        scoring=config.grid_search.scoring,
    )
    grid_search.fit(X_train, y_train)

    # Get best model
    best_model = grid_search.best_estimator_

    # Keep references for feature importance even if we calibrate later
    pre_for_importance = best_model.named_steps["preprocessor"]
    orig_classifier = best_model.named_steps["classifier"]

    # Optional probability calibration using validation set
    if calibration in {"platt", "isotonic"}:
        method = "sigmoid" if calibration == "platt" else "isotonic"
        X_val_trans = pre_for_importance.transform(X_val)
        calibrator = CalibratedClassifierCV(estimator=orig_classifier, cv="prefit", method=method)
        calibrator.fit(X_val_trans, y_val)
        best_model = Pipeline(
            steps=[("preprocessor", pre_for_importance), ("classifier", calibrator)]
        )

    # Evaluate on validation set (probabilities for threshold tuning)
    # Use the pipeline's classes_ to locate the positive class index
    classes = best_model.named_steps["classifier"].classes_
    if 1 not in set(classes):
        logging.getLogger(__name__).error(
            f"Trained classifier classes {classes} do not include positive class '1'."
        )
        raise ValueError(f"Trained classifier classes {classes} do not include positive class '1'.")
    # We trained with y in {0,1}, so positive class is 1
    pos_index = int(np.where(classes == 1)[0][0])
    y_pred_proba = best_model.predict_proba(X_val)[:, pos_index]
    # Decision threshold tuning will set decision_threshold; labels computed after tuning

    # Determine decision threshold (tuning on validation set)
    decision_threshold = 0.5
    if threshold_objective == "f1":
        # Sweep thresholds to maximize F1
        candidate_ts = np.unique(np.concatenate([[0.0, 1.0], y_pred_proba]))
        best_f1 = -1.0
        for t in candidate_ts:
            preds = (y_pred_proba >= t).astype(int)
            f1 = f1_score(y_val, preds)
            if f1 > best_f1:
                best_f1 = f1
                decision_threshold = float(t)
    elif threshold_objective == "precision_constrained_recall":
        # Maximize recall subject to precision >= min_precision
        if min_precision is None:
            min_precision = 0.8
        candidate_ts = np.unique(np.concatenate([[0.0, 1.0], y_pred_proba]))

        best_recall = -1.0
        for t in candidate_ts:
            preds = (y_pred_proba >= t).astype(int)
            p = precision_score(y_val, preds, zero_division=0)
            r = recall_score(y_val, preds, zero_division=0)
            if p >= min_precision and r > best_recall:
                best_recall = r
                decision_threshold = float(t)
    elif threshold_objective == "top_k":
        # Choose threshold at top-k ratio of validation probabilities
        if not top_k_ratio:
            top_k_ratio = 0.1
        k = max(1, int(len(y_val) * float(top_k_ratio)))
        sorted_probs = np.sort(y_pred_proba)[::-1]
        decision_threshold = float(sorted_probs[k - 1])

    # Apply tuned threshold for label predictions
    y_pred = (y_pred_proba >= decision_threshold).astype(int)

    # Print evaluation metrics
    logging.getLogger(__name__).info("\nModel Performance:")
    logging.getLogger(__name__).info(f"Accuracy (val): {accuracy_score(y_val, y_pred):.4f}")
    logging.getLogger(__name__).info(f"ROC AUC (val): {roc_auc_score(y_val, y_pred_proba):.4f}")
    logging.getLogger(__name__).info(f"Decision threshold (val-tuned): {decision_threshold:.4f}")
    # PR metrics
    pr_auc = average_precision_score(y_val, y_pred_proba)
    p, r, _ = precision_recall_curve(y_val, y_pred_proba)
    logging.getLogger(__name__).info(f"PR AUC (val): {pr_auc:.4f}")
    logging.getLogger(__name__).info(
        "\nClassification Report (val):\n" + classification_report(y_val, y_pred)
    )

    # Save model diagnostics to file
    # diagnostics_path already versioned above
    with open(diagnostics_path, "w") as f:
        f.write("MODEL DIAGNOSTICS REPORT\n")
        f.write("=======================\n\n")
        f.write(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}\n")
        f.write(f"ROC AUC: {roc_auc_score(y_val, y_pred_proba):.4f}\n")
        f.write(f"PR AUC: {pr_auc:.4f}\n")
        f.write(f"Decision threshold: {decision_threshold:.4f}\n\n")
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
    if hasattr(orig_classifier, "feature_importances_"):
        # Get feature importances from the trained model
        importances = orig_classifier.feature_importances_

        # Use fitted preprocessor to get readable feature names --------
        pre = pre_for_importance

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

    # Compute reference statistics for drift detection
    reference_stats = {}
    try:
        # Numerical feature statistics (on training data)
        num_stats = {}
        for col in numeric_features:
            num_stats[col] = {
                "mean": float(X_train[col].mean()),
                "std": float(X_train[col].std()),
                "min": float(X_train[col].min()),
                "max": float(X_train[col].max()),
                "median": float(X_train[col].median()),
            }
        reference_stats["numerical"] = num_stats

        # Categorical feature statistics (value distributions)
        cat_stats = {}
        for col in categorical_features:
            value_counts = X_train[col].value_counts(normalize=True, dropna=False).to_dict()
            # Convert to string keys for JSON serialization
            cat_stats[col] = {str(k): float(v) for k, v in value_counts.items()}
        reference_stats["categorical"] = cat_stats

        # Target distribution (on training data)
        target_dist = pd.Series(y_train).value_counts(normalize=True).to_dict()
        reference_stats["target_distribution"] = {str(k): float(v) for k, v in target_dist.items()}

        logging.getLogger(__name__).info("Computed reference statistics for drift detection")
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to compute reference statistics: {e}")
        reference_stats = {}

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
            "decision_threshold": decision_threshold,
            "threshold_objective": threshold_objective,
            "reference_statistics": reference_stats,
        }
        meta_path = MODELS_DIR / "model_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(metadata, mf, indent=2)
        logging.getLogger(__name__).info(f"Saved model metadata to: {meta_path}")
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to save model metadata: {e}")

    # Evaluate on test set
    classes_test = best_model.named_steps["classifier"].classes_
    if 1 not in set(classes_test):
        logging.getLogger(__name__).error(
            f"Trained classifier classes {classes_test} do not include positive class '1'."
        )
        raise ValueError(
            f"Trained classifier classes {classes_test} do not include positive class '1'."
        )
    pos_index_test = int(np.where(classes_test == 1)[0][0])
    y_test_pred_proba = best_model.predict_proba(X_test)[:, pos_index_test]
    # Apply validation-tuned threshold to test probabilities for label predictions
    y_test_pred_thresholded = (y_test_pred_proba >= decision_threshold).astype(int)

    # (MLflow logging moved below after artifacts are created)

    # Append test metrics to diagnostics
    with open(diagnostics_path, "a") as f:
        f.write("\nTEST SET PERFORMANCE\n")
        f.write("=====================\n\n")
        f.write(f"Accuracy (test): {accuracy_score(y_test, y_test_pred_thresholded):.4f}\n")
        f.write(f"ROC AUC (test): {roc_auc_score(y_test, y_test_pred_proba):.4f}\n\n")
        f.write("Classification Report (test):\n")
        f.write(classification_report(y_test, y_test_pred_thresholded))
        f.write("\nConfusion Matrix (test):\n")
        f.write(f"{confusion_matrix(y_test, y_test_pred_thresholded)}\n")

    # Confusion Matrix (test) path defined above
    plt.figure(figsize=(8, 6))
    cm_test = confusion_matrix(y_test, y_test_pred_thresholded)
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

    # Optional: log to MLflow after all artifacts are created (and diagnostics include test)
    if use_mlflow:
        try:
            import mlflow
            import mlflow.sklearn

            if not mlflow_uri:
                mlflow_uri = str(ROOT_DIR / "mlruns")
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment("churn_prediction")
            with mlflow.start_run() as run:
                # Params - convert Pydantic model to dict for access
                cfg_dict = config.model_dump() if hasattr(config, "model_dump") else config
                mlflow.log_params(
                    {
                        "random_state": cfg_dict.get("random_state"),
                        "cv": cfg_dict.get("grid_search", {}).get("cv"),
                        "scoring": cfg_dict.get("grid_search", {}).get("scoring"),
                    }
                )
                # Best classifier params (subset)
                best_params = best_model["classifier"].get_params()
                for p in ["n_estimators", "max_depth", "random_state"]:
                    if p in best_params:
                        mlflow.log_param(f"best_{p}", best_params[p])
                mlflow.log_param("class_weight", class_weight or "none")

                # Metrics
                mlflow.log_metric("val_accuracy", float(accuracy_score(y_val, y_pred)))
                mlflow.log_metric("val_roc_auc", float(roc_auc_score(y_val, y_pred_proba)))
                mlflow.log_metric("val_pr_auc", float(pr_auc))
                mlflow.log_metric(
                    "test_accuracy", float(accuracy_score(y_test, y_test_pred_thresholded))
                )
                mlflow.log_metric("test_roc_auc", float(roc_auc_score(y_test, y_test_pred_proba)))
                mlflow.log_param("decision_threshold", decision_threshold)
                mlflow.log_param("threshold_objective", threshold_objective)

                # Artifacts (now exist on disk)
                for p in [
                    diagnostics_path,
                    cm_val_path,
                    roc_val_path,
                    cm_test_path,
                    roc_test_path,
                ]:
                    try:
                        if Path(p).exists():
                            mlflow.log_artifact(str(p))
                    except Exception:
                        pass

                # Model with signature and input example
                try:
                    from mlflow.models.signature import infer_signature

                    input_example = X_val.head(5)
                    # Use predict_proba as output to capture probability schema
                    output_example = best_model.predict_proba(input_example)
                    signature = infer_signature(input_example, output_example)
                    mlflow.sklearn.log_model(
                        best_model,
                        artifact_path="model",
                        signature=signature,
                        input_example=input_example,
                    )
                except Exception:
                    # Fall back to logging without signature if inference fails
                    mlflow.sklearn.log_model(best_model, artifact_path="model")

                # Optional: register model in MLflow Model Registry and set stage
                try:
                    if registered_model_name:
                        from mlflow.tracking import MlflowClient

                        client = MlflowClient()
                        model_uri = f"runs:/{run.info.run_id}/model"
                        registered = client.register_model(
                            model_uri=model_uri, name=registered_model_name
                        )
                        if model_stage and model_stage.lower() in {"staging", "production"}:
                            client.transition_model_version_stage(
                                name=registered_model_name,
                                version=registered.version,
                                stage=model_stage.capitalize(),
                                archive_existing_versions=True,
                            )
                except Exception as reg_e:
                    logging.getLogger(__name__).warning(
                        f"Model registry step skipped due to error: {reg_e}"
                    )
        except Exception as e:
            logging.getLogger(__name__).warning(f"MLflow logging skipped due to error: {e}")

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
            # Persist validated config as JSON for provenance
            json.dump(json.loads(config.model_dump_json()), cf, indent=2)
        logging.getLogger(__name__).info(f"Saved run config to: {cfg_path}")
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to save run config: {e}")

    logging.getLogger(__name__).info("Training completed successfully!")
    return best_model


def predict(
    input_path=None,
    drop_pii=False,
    threshold_override: float | None = None,
    output_path: str | None = None,
):
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

    # Basic inference-time checks: missing required columns
    meta_path = MODELS_DIR / "model_metadata.json"
    expected = None
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as mf:
                metadata = json.load(mf)
            expected = metadata.get("feature_columns", [])
        except Exception:
            expected = None
    if expected:
        missing_required = [c for c in expected if c not in X.columns]
        if missing_required:
            logging.getLogger(__name__).warning(
                f"Input is missing expected columns: {missing_required}. They will be added as NA."
            )

    # Align columns to training schema
    X = align_features_to_training_schema(X)

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
    # Use CLI override if provided; otherwise fall back to metadata, else 0.5
    threshold = 0.5
    if threshold_override is not None:
        try:
            threshold = float(threshold_override)
        except Exception:
            pass
    meta_path = MODELS_DIR / "model_metadata.json"
    if threshold_override is None and meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as mf:
                metadata = json.load(mf)
            threshold = float(metadata.get("decision_threshold", threshold))
        except Exception:
            pass
    predictions = (predictions_proba >= threshold).astype(int)
    class_labels = predictions

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
    # Use provided output path, otherwise default to a timestamped file to avoid overwriting
    if output_path:
        out_path = Path(output_path)
        # If a relative path is given, make it relative to ROOT_DIR
        if not out_path.is_absolute():
            out_path = ROOT_DIR / out_path
    else:
        try:
            ts = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
        except AttributeError:
            ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = ROOT_DIR / f"predictions_{ts}.csv"
    results.to_csv(out_path, index=False)
    logging.getLogger(__name__).info(f"Full predictions saved to {out_path}")


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
    parser.add_argument(
        "--threshold",
        type=float,
        help="Override decision threshold for prediction (0-1); defaults to metadata or 0.5",
    )
    parser.add_argument(
        "--output",
        help=(
            "Path to write predictions CSV "
            "(default: predictions_{timestamp}.csv under CHURN_ROOT)"
        ),
    )
    parser.add_argument("--config", help="Path to JSON config file for training options")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow tracking/logging")
    parser.add_argument(
        "--mlflow-uri",
        help="MLflow tracking URI (defaults to ./mlruns if omitted)",
    )
    parser.add_argument(
        "--registered-model-name",
        help="If provided, register the run's model under this MLflow Model Registry name",
    )
    parser.add_argument(
        "--model-stage",
        choices=["Staging", "Production"],
        help="Optional stage to transition the registered model to (Staging or Production)",
    )
    parser.add_argument(
        "--calibration",
        choices=["none", "platt", "isotonic"],
        default="none",
        help="Probability calibration method (uses validation set)",
    )
    parser.add_argument(
        "--threshold-objective",
        choices=["f1", "precision_constrained_recall", "top_k"],
        default="f1",
        help="Objective for decision threshold tuning",
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        help="Minimum precision constraint for precision_constrained_recall",
    )
    parser.add_argument(
        "--top-k-ratio",
        type=float,
        help="Top-k ratio (0-1) for top_k threshold objective",
    )
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="none",
        help="Class weight strategy for classifier (handles imbalance)",
    )
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

    try:
        if args.action == "train":
            # Default to showing plots when a DISPLAY is available unless explicitly disabled
            default_show = bool(os.environ.get("DISPLAY"))
            show_flag = args.show_plots or (default_show and not args.no_show_plots)
            cfg = load_run_config(args.config)
            train_model(
                positive_label=args.pos_label,
                show_plots=show_flag,
                config=cfg,
                use_mlflow=args.mlflow,
                mlflow_uri=args.mlflow_uri,
                registered_model_name=args.registered_model_name,
                model_stage=args.model_stage,
                calibration=args.calibration,
                threshold_objective=args.threshold_objective,
                min_precision=args.min_precision,
                top_k_ratio=args.top_k_ratio,
                class_weight=None if args.class_weight == "none" else args.class_weight,
            )
        elif args.action == "predict":
            predict(
                args.input,
                drop_pii=args.no_pii,
                threshold_override=args.threshold,
                output_path=args.output,
            )
        else:
            parser.print_help()
    except Exception as exc:
        logging.getLogger(__name__).exception(f"Unhandled error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
