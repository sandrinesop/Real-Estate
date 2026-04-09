import logging
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def split_features_target(df: pd.DataFrame,
                           target_col: str = 'price'):
   
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in dataset.")

    x = df.drop(target_col, axis=1)
    y = df[target_col]
    feature_cols = x.columns.tolist()

    logger.info(f"Features: {len(feature_cols)} columns")
    logger.info(f"Target: {target_col} — range: ${y.min():,.0f} to ${y.max():,.0f}")
    return x, y, feature_cols


def split_train_test(x, y,
                     test_size: float = 0.2,
                     stratify_col: str = 'property_type_Condo',
                     random_state: int = 42):
   
    stratify = None
    if stratify_col and stratify_col in x.columns:
        stratify = x[stratify_col]
        logger.info(f"Stratifying split on: {stratify_col}")
    else:
        logger.warning(
            f"Stratify column '{stratify_col}' not found. "
            "Splitting without stratification."
        )

    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=test_size,
        stratify=stratify,
        random_state=random_state
    )

    logger.info(
        f"Split complete — Train: {len(x_train)}, Test: {len(x_test)}"
    )
    return x_train, x_test, y_train, y_test


def save_feature_columns(feature_cols: list,
                          path: str = "models/feature_columns.pkl"):

    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(feature_cols, path)
    logger.info(f"Feature columns saved to {path}")


def preprocess_pipeline(filepath: str,
                         test_size: float = 0.2):
 
    from src.data_loader import load_data

    df = load_data(filepath)
    x, y, feature_cols = split_features_target(df)
    x_train, x_test, y_train, y_test = split_train_test(
        x, y, test_size=test_size
    )
    save_feature_columns(feature_cols)

    logger.info("Preprocessing pipeline complete.")
    return x_train, x_test, y_train, y_test, feature_cols