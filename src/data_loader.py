"""
Data Loading Module - IMPROVED
Loads California Housing dataset.
"""

import os
import pandas as pd
import numpy as np


LOCAL_CSV = "data/housing_data_raw.csv"

def load_dataset(random_state=42):
    """Load the California Housing dataset. Uses local CSV if sklearn download fails."""
    np.random.seed(random_state)

    # Try sklearn first
    try:
        from sklearn.datasets import fetch_california_housing
        housing = fetch_california_housing(as_frame=True)
        data = housing.frame
        print(f"Dataset loaded from sklearn: {data.shape[0]} rows, {data.shape[1]} columns")
        return data
    except Exception:
        pass

    # Fallback: local CSV
    if os.path.exists(LOCAL_CSV):
        data = pd.read_csv(LOCAL_CSV)
        print(f"Dataset loaded from local CSV: {data.shape[0]} rows, {data.shape[1]} columns")
        return data

    raise RuntimeError("Could not load dataset. Ensure data/housing_data_raw.csv exists.")


def save_dataset(data, filepath='data/housing_data.csv'):
    data.to_csv(filepath, index=False)
    print(f"Dataset saved to {filepath}")
