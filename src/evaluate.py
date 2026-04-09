import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error,
                              mean_squared_error, r2_score)

logger = logging.getLogger(__name__)


def evaluate_model(model, x_train, y_train,
                   x_test, y_test,
                   model_name: str = "Model") -> dict:
    
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_r2 = r2_score(y_test, test_pred)

    logger.info(f"{model_name} | Train MAE: ${train_mae:,.0f}")
    logger.info(f"{model_name} | Test  MAE: ${test_mae:,.0f}")
    logger.info(f"{model_name} | RMSE: ${test_rmse:,.0f}")
    logger.info(f"{model_name} | R²:   {test_r2:.4f}")

    print(f"\n{'='*50}")
    print(f"  {model_name} Evaluation")
    print(f"{'='*50}")
    print(f"  Train MAE : ${train_mae:,.0f}")
    print(f"  Test  MAE : ${test_mae:,.0f}")
    print(f"  Test  RMSE: ${test_rmse:,.0f}")
    print(f"  Test  R²  : {test_r2:.4f}")

    # Check against success criteria
    if test_mae < 70000:
        print(f"  ✅ SUCCESS: MAE ${test_mae:,.0f} is below $70,000 target!")
    else:
        print(f"  ⚠️  MAE ${test_mae:,.0f} exceeds $70,000 target.")

    return {
        "train_mae": train_mae,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_r2": test_r2
    }


def plot_residuals(model, x_test, y_test,
                   model_name: str = "Model",
                   save_path: str = None):
   
    y_pred = model.predict(x_test)
    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5, color='steelblue', s=30)
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    axes[0].set_xlabel('Predicted Price ($)')
    axes[0].set_ylabel('Residual (Actual - Predicted)')
    axes[0].set_title(f'{model_name} — Residual Plot')
    axes[0].grid(True, alpha=0.3)

    # Actual vs Predicted
    axes[1].scatter(y_test, y_pred, alpha=0.5, color='coral', s=30)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[1].plot([min_val, max_val], [min_val, max_val],
                 'r--', linewidth=1.5, label='Perfect Prediction')
    axes[1].set_xlabel('Actual Price ($)')
    axes[1].set_ylabel('Predicted Price ($)')
    axes[1].set_title(f'{model_name} — Actual vs Predicted')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        logger.info(f"Residual plot saved: {save_path}")

    return fig


def plot_feature_importance(model, feature_cols: list,
                             top_n: int = 15,
                             save_path: str = None):
   
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model has no feature_importances_ attribute.")
        return None

    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values('Importance', ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(feat_df['Feature'], feat_df['Importance'],
            color='steelblue')
    ax.set_title(f'Top {top_n} Feature Importances — Random Forest')
    ax.set_xlabel('Importance Score')
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        logger.info(f"Feature importance plot saved: {save_path}")

    return fig


def compare_models(results: dict):
    print(f"\n{'='*60}")
    print("  Model Comparison")
    print(f"{'='*60}")
    print(f"  {'Model':<30} {'Test MAE':>12} {'R²':>8}")
    print(f"  {'-'*50}")
    for name, metrics in results.items():
        print(
            f"  {name:<30} "
            f"${metrics['test_mae']:>10,.0f} "
            f"{metrics['test_r2']:>8.4f}"
        )