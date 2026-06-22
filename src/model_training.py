"""
Model Training Module - IMPROVED
Trains multiple regression models including XGBoost.
Uses cross-validation and removes capped values.
"""

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
import joblib
import os

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available — skipping XGBoost model")


def remove_capped_targets(X, y, cap_value=5.0):
    """Remove samples where target is capped at max value."""
    mask = y < cap_value
    removed = len(y) - mask.sum()
    print(f"Removed {removed} capped samples ({removed/len(y)*100:.1f}%) with y >= {cap_value}")
    return X[mask], y[mask]


def cross_validate_model(model, X, y, cv=5):
    """Run cross-validation and return mean R² and std."""
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=-1)
    return scores.mean(), scores.std()


def train_all_models(X_train, y_train, random_state=42, run_cv=True):
    """Train all models with optional cross-validation."""
    print("=" * 60)
    print("TRAINING ALL MODELS (WITH DE-SATURATION)")
    print("=" * 60)

    # Remove capped targets
    X_tr, y_tr = remove_capped_targets(X_train, y_train)

    models_config = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.05, max_iter=10000),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=random_state,
        ),
    }

    if XGBOOST_AVAILABLE:
        models_config["XGBoost"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            verbosity=0,
        )

    cv_results = {}
    trained_models = {}

    for name, model in models_config.items():
        print(f"\nTraining {name}...")
        if run_cv:
            cv_mean, cv_std = cross_validate_model(model, X_tr, y_tr, cv=5)
            cv_results[name] = {"cv_r2_mean": cv_mean, "cv_r2_std": cv_std}
            print(f"  CV R² = {cv_mean:.4f} ± {cv_std:.4f}")
        model.fit(X_tr, y_tr)
        trained_models[name] = model

    print("\nALL MODELS TRAINED SUCCESSFULLY")
    return trained_models, cv_results


def save_model(model, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved → {filepath}")


def load_model(filepath):
    return joblib.load(filepath)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_loader import load_dataset
    from src.feature_engineering import create_interaction_features
    from sklearn.model_selection import train_test_split

    df = load_dataset()
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]
    X = create_interaction_features(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models, cv_results = train_all_models(X_train, y_train)

    for name, model in models.items():
        safe_name = name.lower().replace(" ", "_")
        save_model(model, f"outputs/models/{safe_name}.pkl")

    print("\nCV Results:")
    for name, res in cv_results.items():
        print(f"  {name}: {res['cv_r2_mean']:.4f} ± {res['cv_r2_std']:.4f}")
