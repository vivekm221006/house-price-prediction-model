"""
Training Pipeline - runs full training and saves models + metrics.
Run: python train_pipeline.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, ".")

from src.data_loader import load_dataset
from src.feature_engineering import create_interaction_features
from src.model_training import train_all_models, save_model
from src.model_evaluation import evaluate_all_models

os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("data", exist_ok=True)

print("Loading dataset...")
df = load_dataset()

X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

print("Applying feature engineering...")
X = create_interaction_features(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

models, cv_results = train_all_models(X_train, y_train, run_cv=True)

results_df, predictions = evaluate_all_models(models, X_train, y_train, X_test, y_test)

# Save metrics
metrics_to_save = results_df.copy()
metrics_to_save.to_csv("outputs/model_metrics.csv", index=False)

# Save CV results
with open("outputs/cv_results.json", "w") as f:
    json.dump(cv_results, f, indent=2)

# Save feature names for UI
feature_names = list(X.columns)
with open("outputs/feature_names.json", "w") as f:
    json.dump(feature_names, f)

# Save all models
for name, model in models.items():
    safe_name = name.lower().replace(" ", "_")
    save_model(model, f"outputs/models/{safe_name}.pkl")

# Save best model name
best_model_name = results_df.sort_values("Test R²", ascending=False).iloc[0]["Model"]
with open("outputs/best_model.txt", "w") as f:
    f.write(best_model_name)

print(f"\n✅ Training complete. Best model: {best_model_name}")
print(results_df[["Model", "Test R²", "Test RMSE", "Test MAE", "Test MAPE (%)"]].to_string(index=False))
