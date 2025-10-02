import json
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.timeout(60)
def test_cli_train_smoke(tmp_path: Path):
    # Arrange: create tiny dataset under CHURN_ROOT/data
    root = tmp_path
    data_dir = root / "data"
    models_dir = root / "models"
    data_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)

    df = pd.DataFrame(
        {
            "num": [0, 1, 2, 3, 4, 5, 6, 7],
            "cat": ["a", "b", "a", "b", "a", "b", "a", "b"],
            "churn": ["Yes", "No", "No", "Yes", "No", "No", "Yes", "No"],
        }
    )
    (data_dir / "dataset.csv").write_text(df.to_csv(index=False))

    # Minimal config for fast run
    cfg = {
        "random_state": 42,
        # Larger train split so each class has enough members for CV
        "splits": {"test_size": 0.25, "val_size": 0.25},
        "grid_search": {
            "cv": 2,
            "scoring": "roc_auc",
            "param_grid": {"classifier__n_estimators": [10], "classifier__max_depth": [None]},
        },
    }
    cfg_path = root / "config.json"
    cfg_path.write_text(json.dumps(cfg))

    # Prefer churn CLI if available; otherwise use python -m ml_portfolio_churn.main
    churn_cmd = shutil.which("churn")
    env = os.environ.copy()
    env["CHURN_ROOT"] = str(root)

    if churn_cmd:
        cmd = [
            churn_cmd,
            "train",
            "--pos-label",
            "Yes",
            "--config",
            str(cfg_path),
            "--no-show-plots",
        ]
    else:
        cmd = [
            shutil.which("python") or "python",
            "-m",
            "ml_portfolio_churn.main",
            "train",
            "--pos-label",
            "Yes",
            "--config",
            str(cfg_path),
            "--no-show-plots",
        ]

    # Act
    proc = subprocess.run(
        cmd, cwd=str(root), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )

    # Assert
    assert proc.returncode == 0, proc.stdout.decode()
    assert (models_dir / "churn_model.joblib").exists(), "Model artifact not created"
