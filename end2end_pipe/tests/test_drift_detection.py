import numpy as np
import pandas as pd
from ml_portfolio_churn.drift import (
    compute_categorical_drift,
    compute_numerical_drift,
    compute_prediction_drift,
    detect_drift,
)


def test_numerical_drift_stable():
    """Test that similar distributions are detected as stable."""
    current = pd.DataFrame(
        {
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )

    reference = {
        "feature1": {"mean": 3.0, "std": 1.5},
        "feature2": {"mean": 30.0, "std": 15.0},
    }

    result = compute_numerical_drift(current, reference, threshold=0.1)

    assert result["feature1"]["status"] == "stable"
    assert result["feature2"]["status"] == "stable"


def test_numerical_drift_detected():
    """Test that significant distribution changes are detected."""
    current = pd.DataFrame(
        {
            "feature1": [10.0, 20.0, 30.0, 40.0, 50.0],  # mean shifted from 3 to 30
        }
    )

    reference = {
        "feature1": {"mean": 3.0, "std": 1.5},
    }

    result = compute_numerical_drift(current, reference, threshold=0.1)

    assert result["feature1"]["status"] == "drifted"
    assert result["feature1"]["mean_diff_pct"] > 10  # >10% change


def test_numerical_drift_missing_column():
    """Test handling of missing columns."""
    current = pd.DataFrame(
        {
            "feature1": [1.0, 2.0, 3.0],
        }
    )

    reference = {
        "feature1": {"mean": 2.0, "std": 1.0},
        "feature2": {"mean": 5.0, "std": 2.0},  # Not in current data
    }

    result = compute_numerical_drift(current, reference, threshold=0.1)

    assert result["feature1"]["status"] in ["stable", "drifted"]
    assert result["feature2"]["status"] == "missing"


def test_categorical_drift_stable():
    """Test that similar categorical distributions are stable."""
    current = pd.DataFrame(
        {
            "cat1": ["a", "a", "b", "b", "c"],
        }
    )

    reference = {
        "cat1": {"a": 0.4, "b": 0.4, "c": 0.2},
    }

    result = compute_categorical_drift(current, reference, threshold=0.1)

    assert result["cat1"]["status"] == "stable"
    assert result["cat1"]["psi"] < 0.1


def test_categorical_drift_detected():
    """Test that significant categorical changes are detected."""
    current = pd.DataFrame(
        {
            "cat1": ["a"] * 10,  # All 'a', very different from reference
        }
    )

    reference = {
        "cat1": {"a": 0.2, "b": 0.5, "c": 0.3},
    }

    result = compute_categorical_drift(current, reference, threshold=0.1)

    assert result["cat1"]["status"] == "drifted"
    assert result["cat1"]["psi"] > 0.1


def test_categorical_drift_new_categories():
    """Test detection of new categories."""
    current = pd.DataFrame(
        {
            "cat1": ["a", "b", "d"],  # 'd' is new
        }
    )

    reference = {
        "cat1": {"a": 0.5, "b": 0.5},
    }

    result = compute_categorical_drift(current, reference, threshold=0.1)

    assert "d" in result["cat1"]["new_categories"]


def test_prediction_drift_stable():
    """Test stable prediction distribution."""
    predictions = np.array([0, 0, 1, 1, 0, 0, 1, 1])  # 50% positive
    reference = {"0": 0.5, "1": 0.5}

    result = compute_prediction_drift(predictions, reference, threshold=0.1)

    assert result["status"] == "stable"
    assert abs(result["absolute_difference"]) < 0.1


def test_prediction_drift_detected():
    """Test detection of prediction distribution shift."""
    predictions = np.array([1, 1, 1, 1, 1, 1, 0, 0])  # 75% positive vs 50% reference
    reference = {"0": 0.5, "1": 0.5}

    result = compute_prediction_drift(predictions, reference, threshold=0.1)

    assert result["status"] == "drifted"
    assert result["absolute_difference"] > 0.1


def test_detect_drift_no_baseline():
    """Test drift detection with no baseline statistics."""
    data = pd.DataFrame({"a": [1, 2, 3]})
    metadata = {}  # No reference_statistics

    result = detect_drift(data, metadata)

    assert result["status"] == "no_baseline"


def test_detect_drift_full_analysis():
    """Test full drift analysis with numerical and categorical features."""
    current = pd.DataFrame(
        {
            "num1": [1.0, 2.0, 3.0, 4.0],
            "cat1": ["a", "a", "b", "b"],
        }
    )

    metadata = {
        "reference_statistics": {
            "numerical": {
                "num1": {"mean": 2.5, "std": 1.2, "min": 1.0, "max": 4.0, "median": 2.5},
            },
            "categorical": {
                "cat1": {"a": 0.5, "b": 0.5},
            },
            "target_distribution": {"0": 0.5, "1": 0.5},
        }
    }

    result = detect_drift(data=current, metadata=metadata)

    assert result["status"] == "analyzed"
    assert "numerical_features" in result
    assert "categorical_features" in result
    assert result["sample_size"] == 4


def test_detect_drift_with_predictions():
    """Test drift analysis including prediction drift."""
    current = pd.DataFrame(
        {
            "num1": [1.0, 2.0, 3.0, 4.0],
        }
    )

    predictions = np.array([0, 0, 1, 1])

    metadata = {
        "reference_statistics": {
            "numerical": {
                "num1": {"mean": 2.5, "std": 1.2, "min": 1.0, "max": 4.0, "median": 2.5},
            },
            "target_distribution": {"0": 0.5, "1": 0.5},
        }
    }

    result = detect_drift(data=current, metadata=metadata, predictions=predictions)

    assert "prediction_drift" in result
    assert result["prediction_drift"]["status"] in ["stable", "drifted"]


def test_categorical_drift_psi_calculation():
    """Test PSI calculation accuracy."""
    # PSI formula: sum((current - ref) * ln(current / ref))
    current = pd.DataFrame(
        {
            "cat": ["a"] * 6 + ["b"] * 4,  # 60% a, 40% b
        }
    )

    reference = {
        "cat": {"a": 0.5, "b": 0.5},  # 50% a, 50% b
    }

    result = compute_categorical_drift(current, reference, threshold=0.1)

    # Manually calculate expected PSI
    # (0.6 - 0.5) * ln(0.6/0.5) + (0.4 - 0.5) * ln(0.4/0.5)
    # = 0.1 * ln(1.2) + (-0.1) * ln(0.8)
    # = 0.1 * 0.1823 - 0.1 * 0.2231
    # ≈ 0.0041

    assert result["cat"]["psi"] < 0.1  # Should be small change
    assert result["cat"]["status"] == "stable"


def test_numerical_drift_zero_std():
    """Test handling of zero standard deviation."""
    current = pd.DataFrame(
        {
            "const": [5.0, 5.0, 5.0, 5.0],  # Constant feature
        }
    )

    reference = {
        "const": {"mean": 5.0, "std": 0.0},
    }

    # Should not crash
    result = compute_numerical_drift(current, reference, threshold=0.1)

    assert result["const"]["status"] in ["stable", "drifted"]
