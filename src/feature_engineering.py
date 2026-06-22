"""
Feature Engineering Module - IMPROVED
Richer interaction features + log transforms + location clusters
"""

import pandas as pd
import numpy as np

EPSILON = 1e-5


def create_interaction_features(X):
    """
    Create interaction features between key variables.
    Enhanced with log transforms, ratios, and location features.
    """
    X_enhanced = X.copy()

    # --- Room/Bedroom ratios ---
    if 'AveRooms' in X.columns and 'AveBedrms' in X.columns:
        X_enhanced['RoomsPerBedroom'] = X['AveRooms'] / (X['AveBedrms'] + EPSILON)

    if 'AveRooms' in X.columns and 'AveOccup' in X.columns:
        X_enhanced['RoomsPerPerson'] = X['AveRooms'] / (X['AveOccup'] + EPSILON)

    if 'Population' in X.columns and 'AveOccup' in X.columns:
        X_enhanced['HouseholdsPerPopulation'] = 1 / (X['AveOccup'] + EPSILON)

    # --- NEW: Log transforms to reduce skew ---
    if 'MedInc' in X.columns:
        X_enhanced['LogMedInc'] = np.log1p(X['MedInc'])

    if 'Population' in X.columns:
        X_enhanced['LogPopulation'] = np.log1p(X['Population'])

    if 'AveOccup' in X.columns:
        X_enhanced['LogAveOccup'] = np.log1p(X['AveOccup'])

    # --- NEW: Income × Room interaction ---
    if 'MedInc' in X.columns and 'AveRooms' in X.columns:
        X_enhanced['IncomeRooms'] = X['MedInc'] * X['AveRooms']

    # --- NEW: Population density (bedrooms per person) ---
    if 'AveBedrms' in X.columns and 'AveOccup' in X.columns:
        X_enhanced['BedrmsPerPerson'] = X['AveBedrms'] / (X['AveOccup'] + EPSILON)

    # --- NEW: Location cluster (distance to major CA cities) ---
    if 'Latitude' in X.columns and 'Longitude' in X.columns:
        # Distance to San Francisco (~37.77, -122.42)
        X_enhanced['DistToSF'] = np.sqrt(
            (X['Latitude'] - 37.77)**2 + (X['Longitude'] - (-122.42))**2
        )
        # Distance to Los Angeles (~34.05, -118.24)
        X_enhanced['DistToLA'] = np.sqrt(
            (X['Latitude'] - 34.05)**2 + (X['Longitude'] - (-118.24))**2
        )
        # Distance to San Diego (~32.72, -117.16)
        X_enhanced['DistToSD'] = np.sqrt(
            (X['Latitude'] - 32.72)**2 + (X['Longitude'] - (-117.16))**2
        )
        # Min coast distance (rough approximation via longitude)
        X_enhanced['CoastProximity'] = X['Longitude'] + 122  # closer to 0 = more coastal

    return X_enhanced
