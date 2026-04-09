import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ['price']   # minimum required — price must exist


def load_data(filepath: str) -> pd.DataFrame:
   
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    try:
        df = pd.read_csv(filepath)

        if df.empty:
            raise ValueError("Dataset is empty.")

        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                raise ValueError(
                    f"Required column '{col}' missing from dataset."
                )

        logger.info(
            f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns"
        )
        logger.info(f"Columns: {df.columns.tolist()}")
        return df

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise