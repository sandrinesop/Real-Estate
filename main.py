from src.utils import setup_logging, ensure_directories
from src.preprocessing import preprocess_pipeline
from src.train import train_all_models
from src.evaluate import (evaluate_model, plot_residuals,
                           plot_feature_importance, compare_models)


def main():
    setup_logging()
    ensure_directories()

    # Step 1: Preprocess
    print("\n🔄 Loading and splitting data...")
    x_train, x_test, y_train, y_test, feature_cols = preprocess_pipeline(
        "data/final.csv"
    )
    print(f"   Train size: {len(x_train)}, Test size: {len(x_test)}")
    print(f"   Features: {len(feature_cols)} columns")
    print(f"   Price range: ${y_train.min():,.0f} – ${y_train.max():,.0f}")

    # Step 2: Train both models
    print("\n🏗️  Training models...")
    models = train_all_models(x_train, y_train)

    # Step 3: Evaluate both models
    print("\n📊 Evaluating models...")
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(
            model, x_train, y_train,
            x_test, y_test,
            model_name=name
        )

    # Step 4: Compare models
    compare_models(results)

    # Step 5: Generate plots
    print("\n📈 Generating evaluation plots...")
    for name, model in models.items():
        plot_residuals(
            model, x_test, y_test,
            model_name=name,
            save_path=f"logs/{name}_residuals.png"
        )

    # Feature importance for Random Forest
    plot_feature_importance(
        models['random_forest'],
        feature_cols,
        save_path="logs/feature_importance.png"
    )

    print("\n✅ Pipeline complete. Models saved to models/")


if __name__ == "__main__":
    main()