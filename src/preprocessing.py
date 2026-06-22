"""
Data Preprocessing Module - IMPROVED
Handles cleaning, outlier removal, encoding, and scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def handle_missing_values(data):
    missing = data.isnull().sum()
    if missing.sum() == 0:
        print("No missing values found.")
        return data

    numerical_cols = data.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if data[col].isnull().sum() > 0:
            data[col] = data[col].fillna(data[col].median())
            print(f"  Filled {col} with median")

    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if data[col].isnull().sum() > 0:
            mode = data[col].mode()
            data[col] = data[col].fillna(mode[0] if len(mode) > 0 else 'Unknown')
            print(f"  Filled {col} with mode")

    return data


def remove_outliers_iqr(X, y, multiplier=3.0):
    """
    Remove extreme outliers using IQR on target variable.
    More conservative than default (3x IQR) to keep real expensive homes.
    """
    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1
    mask = (y >= Q1 - multiplier * IQR) & (y <= Q3 + multiplier * IQR)
    removed = (~mask).sum()
    print(f"Removed {removed} outlier samples via IQR (multiplier={multiplier})")
    return X[mask], y[mask]


def split_features_target(data, target_column='MedHouseVal'):
    y = data[target_column]
    X = data.drop(columns=[target_column])
    print(f"Features: {X.shape[1]} | Samples: {X.shape[0]}")
    return X, y


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    return X_train_scaled, X_test_scaled, scaler


def split_train_test(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def preprocess_data(data, target_column='MedHouseVal', test_size=0.2, random_state=42):
    print("=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)

    data = handle_missing_values(data)
    X, y = split_features_target(data, target_column)
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size, random_state)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    print("PREPROCESSING COMPLETE")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
