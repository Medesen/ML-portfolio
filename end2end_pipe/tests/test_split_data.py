import pandas as pd
import pytest
from ml_portfolio_churn.features import split_data

# With package installed in CI, no sys.path hacks are required


def test_split_data_encodes_positive_label_yes():
    df = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4],
            "feature_b": ["x", "y", "x", "z"],
            "Churn": ["Yes", "No", "No", "Yes"],
        }
    )

    X, y = split_data(df, training=True, positive_label="Yes")

    assert list(X.columns) == ["feature_a", "feature_b"]
    assert list(y) == [1, 0, 0, 1]


def test_split_data_case_insensitive_target_column():
    df = pd.DataFrame(
        {
            "feature": [10, 20],
            "CHURN": ["No", "Yes"],
        }
    )

    X, y = split_data(df, training=True, positive_label="Yes")
    assert list(X.columns) == ["feature"]
    assert list(y) == [0, 1]


def test_split_data_raises_if_pos_label_missing():
    df = pd.DataFrame(
        {
            "f": [0, 1, 2],
            "Churn": ["No", "No", "No"],
        }
    )

    with pytest.raises(ValueError):
        split_data(df, training=True, positive_label="Yes")


def test_split_data_requires_both_classes_when_training():
    df = pd.DataFrame(
        {
            "f": [0, 1, 2],
            "Churn": ["Yes", "Yes", "Yes"],
        }
    )

    with pytest.raises(ValueError):
        split_data(df, training=True, positive_label="Yes")


def test_split_data_inference_returns_features_only():
    df = pd.DataFrame(
        {
            "a": [1, 2],
            "b": ["x", "y"],
        }
    )

    X = split_data(df, training=False)
    assert list(X.columns) == ["a", "b"]
