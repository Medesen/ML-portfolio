"""Drift detection utilities for monitoring feature and prediction distributions."""

from typing import Any, Dict

import numpy as np
import pandas as pd


def compute_numerical_drift(
    current_data: pd.DataFrame, reference_stats: Dict[str, Dict[str, float]], threshold: float = 0.1
) -> Dict[str, Any]:
    """Compute drift metrics for numerical features.

    Uses relative difference in mean and standard deviation as drift indicators.

    Args:
        current_data: DataFrame with current data
        reference_stats: Dictionary of reference statistics (mean, std, etc.) per feature
        threshold: Threshold for flagging drift (default 0.1 = 10% change)

    Returns:
        Dictionary with drift metrics per feature
    """
    drift_metrics = {}

    for col, ref_stats in reference_stats.items():
        if col not in current_data.columns:
            drift_metrics[col] = {"status": "missing", "message": "Column not in current data"}
            continue

        current_mean = float(current_data[col].mean())
        current_std = float(current_data[col].std())

        ref_mean = ref_stats.get("mean", 0)
        ref_std = ref_stats.get("std", 1)

        # Relative difference in mean (avoid division by zero)
        mean_diff = abs(current_mean - ref_mean) / (abs(ref_mean) + 1e-10)

        # Relative difference in std
        std_diff = abs(current_std - ref_std) / (abs(ref_std) + 1e-10)

        # Detect drift if either mean or std changed significantly
        drifted = (mean_diff > threshold) or (std_diff > threshold)

        drift_metrics[col] = {
            "status": "drifted" if drifted else "stable",
            "current_mean": current_mean,
            "reference_mean": ref_mean,
            "mean_diff_pct": float(mean_diff * 100),
            "current_std": current_std,
            "reference_std": ref_std,
            "std_diff_pct": float(std_diff * 100),
        }

    return drift_metrics


def compute_categorical_drift(
    current_data: pd.DataFrame, reference_stats: Dict[str, Dict[str, float]], threshold: float = 0.1
) -> Dict[str, Any]:
    """Compute drift metrics for categorical features.

    Uses Population Stability Index (PSI) as drift metric.

    Args:
        current_data: DataFrame with current data
        reference_stats: Dictionary of reference value distributions per feature
        threshold: PSI threshold for flagging drift (default 0.1)

    Returns:
        Dictionary with drift metrics per feature
    """
    drift_metrics = {}

    for col, ref_dist in reference_stats.items():
        if col not in current_data.columns:
            drift_metrics[col] = {"status": "missing", "message": "Column not in current data"}
            continue

        # Get current distribution
        current_dist = current_data[col].value_counts(normalize=True, dropna=False).to_dict()
        current_dist = {str(k): float(v) for k, v in current_dist.items()}

        # Compute PSI (Population Stability Index)
        psi = 0.0
        all_categories = set(list(ref_dist.keys()) + list(current_dist.keys()))

        for category in all_categories:
            current_pct = current_dist.get(str(category), 1e-10)
            ref_pct = ref_dist.get(str(category), 1e-10)

            # PSI formula: (current - reference) * ln(current / reference)
            psi += (current_pct - ref_pct) * np.log((current_pct + 1e-10) / (ref_pct + 1e-10))

        # PSI interpretation: <0.1 no change, 0.1-0.25 moderate, >0.25 significant
        drifted = psi > threshold

        # Find new categories
        new_categories = [cat for cat in current_dist.keys() if cat not in ref_dist]

        drift_metrics[col] = {
            "status": "drifted" if drifted else "stable",
            "psi": float(psi),
            "new_categories": new_categories,
            "num_categories_current": len(current_dist),
            "num_categories_reference": len(ref_dist),
        }

    return drift_metrics


def compute_prediction_drift(
    predictions: np.ndarray, reference_dist: Dict[str, float], threshold: float = 0.1
) -> Dict[str, Any]:
    """Compute drift in prediction distribution.

    Args:
        predictions: Array of predictions (0/1)
        reference_dist: Reference distribution from training
        threshold: Threshold for flagging drift

    Returns:
        Dictionary with drift metrics
    """
    current_dist = pd.Series(predictions).value_counts(normalize=True).to_dict()
    current_dist = {str(k): float(v) for k, v in current_dist.items()}

    # Compute difference in positive class rate
    current_positive_rate = current_dist.get("1", 0.0)
    ref_positive_rate = reference_dist.get("1", 0.0)

    diff = abs(current_positive_rate - ref_positive_rate)
    drifted = diff > threshold

    return {
        "status": "drifted" if drifted else "stable",
        "current_positive_rate": current_positive_rate,
        "reference_positive_rate": ref_positive_rate,
        "absolute_difference": float(diff),
        "relative_difference_pct": float(diff / (ref_positive_rate + 1e-10) * 100),
    }


def detect_drift(
    data: pd.DataFrame,
    metadata: Dict[str, Any],
    predictions: np.ndarray = None,
    numerical_threshold: float = 0.1,
    categorical_threshold: float = 0.1,
    prediction_threshold: float = 0.1,
) -> Dict[str, Any]:
    """Main drift detection function.

    Args:
        data: Current data to check for drift
        metadata: Model metadata with reference statistics
        predictions: Optional predictions to check for prediction drift
        numerical_threshold: Threshold for numerical drift (default 10%)
        categorical_threshold: PSI threshold for categorical drift (default 0.1)
        prediction_threshold: Threshold for prediction drift (default 10%)

    Returns:
        Comprehensive drift report
    """
    reference_stats = metadata.get("reference_statistics", {})

    if not reference_stats:
        return {
            "status": "no_baseline",
            "message": (
                "No reference statistics available. "
                "Retrain with latest code to enable drift detection."
            ),
        }

    drift_report = {
        "status": "analyzed",
        "sample_size": len(data),
        "numerical_features": {},
        "categorical_features": {},
        "overall_drift": False,
    }

    # Numerical drift
    if "numerical" in reference_stats:
        num_drift = compute_numerical_drift(
            data, reference_stats["numerical"], threshold=numerical_threshold
        )
        drift_report["numerical_features"] = num_drift

        # Check if any numerical feature drifted
        num_drifted = any(v.get("status") == "drifted" for v in num_drift.values())
        drift_report["overall_drift"] = drift_report["overall_drift"] or num_drifted

    # Categorical drift
    if "categorical" in reference_stats:
        cat_drift = compute_categorical_drift(
            data, reference_stats["categorical"], threshold=categorical_threshold
        )
        drift_report["categorical_features"] = cat_drift

        # Check if any categorical feature drifted
        cat_drifted = any(v.get("status") == "drifted" for v in cat_drift.values())
        drift_report["overall_drift"] = drift_report["overall_drift"] or cat_drifted

    # Prediction drift (if predictions provided)
    if predictions is not None and "target_distribution" in reference_stats:
        pred_drift = compute_prediction_drift(
            predictions, reference_stats["target_distribution"], threshold=prediction_threshold
        )
        drift_report["prediction_drift"] = pred_drift
        drift_report["overall_drift"] = drift_report["overall_drift"] or (
            pred_drift.get("status") == "drifted"
        )

    return drift_report
