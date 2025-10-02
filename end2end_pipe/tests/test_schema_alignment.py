import json

import pandas as pd
from ml_portfolio_churn.features import align_features_to_training_schema


def test_align_adds_missing_columns():
    """Test that missing columns are added with NaN."""
    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    metadata = {"feature_columns": ["a", "b", "c"]}

    aligned = align_features_to_training_schema(X, metadata=metadata)

    assert list(aligned.columns) == ["a", "b", "c"]
    assert aligned["c"].isna().all()


def test_align_drops_extra_columns():
    """Test that unexpected columns are dropped."""
    X = pd.DataFrame({"a": [1, 2], "b": [3, 4], "extra": [5, 6]})
    metadata = {"feature_columns": ["a", "b"]}

    aligned = align_features_to_training_schema(X, metadata=metadata)

    assert list(aligned.columns) == ["a", "b"]
    assert "extra" not in aligned.columns


def test_align_reorders_columns():
    """Test that columns are reordered to match training schema."""
    X = pd.DataFrame({"c": [1, 2], "a": [3, 4], "b": [5, 6]})
    metadata = {"feature_columns": ["a", "b", "c"]}

    aligned = align_features_to_training_schema(X, metadata=metadata)

    assert list(aligned.columns) == ["a", "b", "c"]


def test_align_handles_missing_and_extra_columns():
    """Test combined scenario: missing, extra, and reordering."""
    X = pd.DataFrame({"feature_2": [10, 20], "extra_col": [99, 88], "feature_1": [1, 2]})
    metadata = {"feature_columns": ["feature_1", "feature_2", "feature_3"]}

    aligned = align_features_to_training_schema(X, metadata=metadata)

    assert list(aligned.columns) == ["feature_1", "feature_2", "feature_3"]
    assert aligned["feature_3"].isna().all()
    assert "extra_col" not in aligned.columns
    assert aligned["feature_1"].tolist() == [1, 2]
    assert aligned["feature_2"].tolist() == [10, 20]


def test_align_no_metadata_returns_unchanged():
    """Test that with empty metadata dict, DataFrame is returned unchanged."""
    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    # Pass empty dict to indicate no metadata (None attempts to load from disk)
    aligned = align_features_to_training_schema(X, metadata={})

    pd.testing.assert_frame_equal(aligned, X)


def test_align_empty_feature_columns_returns_unchanged():
    """Test that with empty feature_columns, DataFrame is returned unchanged."""
    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    metadata = {"feature_columns": []}

    aligned = align_features_to_training_schema(X, metadata=metadata)

    pd.testing.assert_frame_equal(aligned, X)


def test_align_loads_from_file_when_no_metadata_provided(tmp_path, monkeypatch):
    """Test that metadata is loaded from file when not provided."""
    # Create temporary models directory with metadata
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    metadata = {"feature_columns": ["col_a", "col_b"]}
    meta_path = models_dir / "model_metadata.json"
    meta_path.write_text(json.dumps(metadata))

    # Monkeypatch MODELS_DIR
    monkeypatch.setattr("ml_portfolio_churn.features.MODELS_DIR", models_dir)

    X = pd.DataFrame({"col_b": [1, 2], "col_a": [3, 4]})
    aligned = align_features_to_training_schema(X, metadata=None)

    assert list(aligned.columns) == ["col_a", "col_b"]


def test_align_handles_missing_metadata_file_gracefully(tmp_path, monkeypatch):
    """Test graceful handling when metadata file doesn't exist."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    monkeypatch.setattr("ml_portfolio_churn.features.MODELS_DIR", models_dir)

    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    aligned = align_features_to_training_schema(X, metadata=None)

    # Should return unchanged when no metadata available
    pd.testing.assert_frame_equal(aligned, X)
