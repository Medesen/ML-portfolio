import pandas as pd
from ml_portfolio_churn.main import train_model

# With package installed in CI, no sys.path hacks are required


def test_smoke_train(tmp_path, monkeypatch):
    # Create a small synthetic dataset
    n = 20
    df = pd.DataFrame(
        {"num": list(range(n)), "cat": ["a", "b"] * (n // 2), "churn": (["Yes", "No"] * (n // 2))}
    )

    # Write dataset to the project data directory (isolated via monkeypatch)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv_path = data_dir / "dataset.csv"
    df.to_csv(csv_path, index=False)

    # Point the module's DATA_DIR and CSV_PATH to our temp dataset
    monkeypatch.setattr("ml_portfolio_churn.main.DATA_DIR", data_dir)
    monkeypatch.setattr("ml_portfolio_churn.main.CSV_PATH", csv_path)
    # Ensure the data loader reads from the same temporary CSV
    monkeypatch.setattr("ml_portfolio_churn.data_io.CSV_PATH", csv_path)

    # Run a minimal training with larger splits so stratification works on tiny data
    config = {
        "random_state": 42,
        # Generous sizes so each split has >= 2 per class
        "splits": {"test_size": 0.4, "val_size": 0.5},
        "grid_search": {
            "cv": 2,
            "scoring": "roc_auc",
            "param_grid": {"classifier__n_estimators": [10], "classifier__max_depth": [None]},
        },
    }
    model = train_model(positive_label="Yes", show_plots=False, config=config)

    assert model is not None
