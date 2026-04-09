import logging
import os
import joblib
import pandas as pd

logger = logging.getLogger(__name__)
MODELS_DIR = "models"


def load_model(model_name: str = "random_forest"):
  
    path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at {path}. Please run main.py first."
        )
    model = joblib.load(path)
    logger.info(f"Model loaded: {model_name}")
    return model


def load_feature_columns():
    path = os.path.join(MODELS_DIR, "feature_columns.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Feature columns not found. Please run main.py first."
        )
    cols = joblib.load(path)
    logger.info(f"Feature columns loaded: {len(cols)} features")
    return cols


def validate_input(input_dict: dict, feature_cols: list):
    
    # Check for price field accidentally included
    if 'price' in input_dict:
        raise ValueError(
            "'price' should not be in input — it is the target to predict."
        )

    # Check that numeric values are reasonable
    if 'year_built' in input_dict:
        year = input_dict['year_built']
        if not (1800 <= year <= 2025):
            raise ValueError(f"year_built {year} seems invalid.")

    logger.info("Input validation passed.")


def predict_price(input_dict: dict,
                  model_name: str = "random_forest") -> dict:
  
    model = load_model(model_name)
    feature_cols = load_feature_columns()

    validate_input(input_dict, feature_cols)

    # Build input DataFrame aligned to training columns
    input_df = pd.DataFrame([input_dict])
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_cols]

    predicted_price = model.predict(input_df)[0]

    result = {
        "predicted_price": round(float(predicted_price), 2),
        "model_used": model_name
    }
    logger.info(
        f"Predicted price: ${predicted_price:,.0f} "
        f"using {model_name}"
    )
    return result