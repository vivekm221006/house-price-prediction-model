"""
Model Evaluation Module - IMPROVED
Evaluates models with comprehensive metrics including MAPE.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name="Model"):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    def mape(y_true, y_pred):
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    metrics = {
        'Model': model_name,
        'Train MAE': mean_absolute_error(y_train, y_train_pred),
        'Train RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'Train R²': r2_score(y_train, y_train_pred),
        'Test MAE': mean_absolute_error(y_test, y_test_pred),
        'Test RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'Test R²': r2_score(y_test, y_test_pred),
        'Test MAPE (%)': mape(y_test.values, y_test_pred),
        'Predictions': y_test_pred
    }
    return metrics


def evaluate_all_models(models, X_train, y_train, X_test, y_test):
    print("=" * 60)
    print("EVALUATING ALL MODELS")
    print("=" * 60)

    results = []
    predictions = {}

    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test, model_name)
        predictions[model_name] = metrics.pop('Predictions')
        results.append(metrics)

        print(f"  Test R²:   {metrics['Test R²']:.4f}")
        print(f"  Test RMSE: {metrics['Test RMSE']:.4f}")
        print(f"  Test MAE:  {metrics['Test MAE']:.4f}")
        print(f"  Test MAPE: {metrics['Test MAPE (%)']:.2f}%")

    results_df = pd.DataFrame(results)
    return results_df, predictions


def save_metrics(results_df, filepath='outputs/model_metrics.csv'):
    results_df.drop(columns=['Predictions'], errors='ignore').to_csv(filepath, index=False)
    print(f"Metrics saved to {filepath}")
