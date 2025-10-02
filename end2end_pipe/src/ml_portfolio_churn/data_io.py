import logging
from typing import Optional

import pandas as pd

from .constants import ARFF_PATH, CSV_PATH
from .convert_arff_to_csv import convert_arff_to_csv


def ensure_csv_exists() -> None:
    """Convert ARFF to CSV if needed."""
    if not CSV_PATH.exists() and ARFF_PATH.exists():
        logging.getLogger(__name__).info("Converting ARFF to CSV...")
        convert_arff_to_csv(str(ARFF_PATH), str(CSV_PATH))
    elif not CSV_PATH.exists():
        logging.getLogger(__name__).error(f"Neither {CSV_PATH} nor {ARFF_PATH} found")
        raise FileNotFoundError(f"Neither {CSV_PATH} nor {ARFF_PATH} found")


def load_data(path: Optional[str] = None) -> pd.DataFrame:
    """Load data from CSV file, normalizing common NA tokens."""
    if path is None:
        ensure_csv_exists()
        data_path = CSV_PATH
    else:
        data_path = path
    try:
        na_tokens = ["?", "NA", "N/A", "na", "n/a", "None", "null", "Null", "NULL", ""]
        df = pd.read_csv(data_path, na_values=na_tokens, keep_default_na=True)
        logging.getLogger(__name__).info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        logging.getLogger(__name__).exception(f"Error loading data: {str(e)}")
        raise
