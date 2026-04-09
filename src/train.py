import logging
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)
MODELS_DIR = "models"


def train_linear_regression(x_train, y_train) -> LinearRegression:
   
    try:
        model = LinearRegression()
        model.fit(x_train, y_train)
        logger.info(
            f"Linear Regression trained. "
            f"Intercept: ${model.intercept_:,.2f}"
        )
        return model
    except Exception as e:
        logger.error(f"Linear Regression training failed: {e}")
        raise


def train_random_forest(x_train, y_train,
                         n_estimators: int = 200) -> RandomForestRegressor:
 
    try:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            criterion='absolute_error',
            random_state=42,
            n_jobs=-1    # use all CPU cores for faster training
        )
        model.fit(x_train, y_train)
        logger.info(
            f"Random Forest trained. "
            f"Trees: {n_estimators}"
        )
        return model
    except Exception as e:
        logger.error(f"Random Forest training failed: {e}")
        raise


def save_model(model, filename: str):
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
        path = os.path.join(MODELS_DIR, filename)
        joblib.dump(model, path)
        logger.info(f"Model saved: {path}")
    except Exception as e:
        logger.error(f"Failed to save model '{filename}': {e}")
        raise


def train_all_models(x_train, y_train) -> dict:
    
    models = {
        "linear_regression": train_linear_regression(x_train, y_train),
        "random_forest": train_random_forest(x_train, y_train),
    }
    for name, model in models.items():
        save_model(model, f"{name}.pkl")

    logger.info("All models trained and saved.")
    return models