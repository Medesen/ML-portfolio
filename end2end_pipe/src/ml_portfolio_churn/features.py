import json
import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pandera.pandas as pa

from .constants import MODELS_DIR, TARGET_COLUMN


def _validate_inference_schema(df: pd.DataFrame) -> None:
    try:
        # Explicit but permissive schema: require at least some columns and allow extras.
        # If you know expected feature names/types, add them under columns={...}.
        schema = pa.DataFrameSchema(
            columns={},
            coerce=True,
            strict=False,
            checks=[pa.Check(lambda d: d.shape[1] > 0, error="input has no columns")],
        )
        schema.validate(df, lazy=True)
    except pa.errors.SchemaError as e:
        logging.getLogger(__name__).warning(f"Input schema validation warnings/errors: {e}")


def split_data(df: pd.DataFrame, training: bool = True, positive_label: Optional[str] = None):
    """Split features and target; model pipeline handles preprocessing.

    Deterministic label encoding: map positive class to 1 and negative to 0.
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
                    f"Cannot train without target column. " f"Available columns: {list(df.columns)}"
                )
                raise ValueError(
                    f"Target column '{TARGET_COLUMN}' not found; "
                    f"available columns: {list(df.columns)}"
                )

    # Split features and target
    if TARGET_COLUMN in df.columns:
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN] if training else None
    else:
        X = df
        y = None

    # Basic schema checks (training time)
    if training:
        null_fractions = X.isna().mean().sort_values(ascending=False)
        high_null = [c for c, frac in null_fractions.items() if frac > 0.7]
        if high_null:
            logging.getLogger(__name__).warning(
                f"Columns with >70% missing values (consider dropping): {high_null}"
            )
        constant_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
        if constant_cols:
            logging.getLogger(__name__).warning(
                f"Constant columns detected (low information): {constant_cols}"
            )
    else:
        # Inference-time: validate/coerce basic schema (loose checks)
        _validate_inference_schema(X)

    # Deterministic target encoding to {0,1}
    if training and y is not None:
        if positive_label is None:
            if y.dtype == "object" or str(y.dtype) == "category":
                candidates = ["Churn", "Yes", "True", "1"]
                y_str = y.astype(str)
                for cand in candidates:
                    if cand in set(y_str.unique()):
                        positive_label = cand
                        break
                if positive_label is None:
                    counts = y_str.value_counts()
                    positive_label = counts.idxmin()
                y = (y_str == str(positive_label)).astype(int)
            else:
                positive_label = 1
                y = (y == 1).astype(int)
        else:
            if y.dtype == "object" or str(y.dtype) == "category":
                if str(positive_label) not in set(y.astype(str).unique()):
                    unique_vals = set(y.astype(str).unique())
                    logging.getLogger(__name__).error(
                        f"Provided --pos-label '{positive_label}' "
                        f"not found in target values: {unique_vals}"
                    )
                    raise ValueError(
                        f"Provided --pos-label '{positive_label}' "
                        f"not found in target values: {unique_vals}"
                    )
                y = (y.astype(str) == str(positive_label)).astype(int)
            else:
                # Numeric target: coerce provided positive_label to numeric if given as string
                coerced = positive_label
                if isinstance(positive_label, str):
                    try:
                        # Try int first, then float fallback
                        coerced = int(float(positive_label))
                    except Exception:
                        try:
                            coerced = float(positive_label)
                        except Exception:
                            # Leave as-is; comparison will likely fail and raise below
                            coerced = positive_label
                y = (y == coerced).astype(int)

        unique_vals = set(pd.Series(y).unique())
        if unique_vals != {0, 1}:
            logging.getLogger(__name__).error(
                f"After encoding, target classes are {unique_vals}. Need both 0 and 1 to train."
            )
            raise ValueError(
                f"After encoding, target classes are {unique_vals}. Need both 0 and 1 to train."
            )

    return (X, y) if training else X


def align_features_to_training_schema(
    X: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """Align inference features to match training schema.

    Adds missing columns with NaN, drops unexpected columns, and reorders to match training.

    Args:
        X: Feature DataFrame from inference data
        metadata: Optional metadata dict; if None, will attempt to load from MODELS_DIR

    Returns:
        Aligned DataFrame with same schema as training data
    """
    if metadata is None:
        meta_path = MODELS_DIR / "model_metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as mf:
                    metadata = json.load(mf)
            except Exception as e:
                logging.getLogger(__name__).warning(
                    f"Failed to load/parse model metadata for alignment: {e}"
                )
                return X
        else:
            logging.getLogger(__name__).warning(
                "Model metadata not found; proceeding without schema alignment."
            )
            return X

    expected = metadata.get("feature_columns", [])
    if not expected:
        return X

    missing = [c for c in expected if c not in X.columns]
    extra = [c for c in X.columns if c not in expected]

    if missing:
        logging.getLogger(__name__).info(f"Adding missing columns with NA for inference: {missing}")
        for c in missing:
            X[c] = np.nan

    if extra:
        logging.getLogger(__name__).info(
            f"Dropping unexpected columns not seen in training: {extra}"
        )
        X = X.drop(columns=extra)

    # Reorder to match training
    X = X[expected]

    return X
