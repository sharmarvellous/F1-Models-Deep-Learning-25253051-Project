"""
HOW TO USE THE F1 MONZA LAP TIME PREDICTION MODEL
==================================================

1. Load the model and scalers:
import joblib
import pickle
import numpy as np
import pandas as pd

model = joblib.load('gradient_boosting_model.pkl')
feature_scaler = joblib.load('feature_scaler.pkl')
target_scaler = joblib.load('target_scaler.pkl')

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)
with open('feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)
with open('categorical_mappings.pkl', 'rb') as f:
    cat_maps = pickle.load(f)

2. Prepare a new lap's features:
# Example lap data
new_lap = {
    'LapNumber': 25,
    'Stint': 2,
    'StintLap': 15,
    'TyreAge': 15,
    'Position': 5,
    'MaxSpeed': 340.5,
    'SpeedStd': 12.3,
    'ThrottleMean': 78.2,
    'BrakeMean': 15.4,
    'RpmMean': 11500,
    'SectorVar': 0.45,
    'PrevThrottle': 77.8,
    'LapTimeDelta': 0.12,
    'SpeedDelta': 1.2,
    'RollingAvg_3': 86.4,
    'RollingAvgSpeed_3': 240.5,
    'RollingStd_3': 0.8,
    'SpeedThrottleInteraction': 340.5 * 78.2,
    'NormLapNumber': 0.5,
    'NormTyreAge': 0.6,
    'TireAgePosition': 15 * (1/5),
    'StintEfficiency': 15/2,
    'SpeedEfficiency': 340.5/86.4,
    'ThrottleBrakeRatio': 78.2/15.4,
    'ExpectedDegradation': 15 * 0.1,  # MEDIUM compound
    'TyreAge_sq': 225,
    'LapNumber_sq': 625,
    'Position_sq': 25,
    'Driver': 'VER',  # Will be one-hot encoded
    'Compound': 'MEDIUM'  # Will be one-hot encoded
}

3. Preprocess the new lap:
# Convert to DataFrame
new_lap_df = pd.DataFrame([new_lap])

# One-hot encode categorical variables
for driver in cat_maps['drivers']:
    new_lap_df[f'Driver_{driver}'] = (new_lap['Driver'] == driver).astype(int)
for compound in cat_maps['compounds']:
    new_lap_df[f'Compound_{compound}'] = (new_lap['Compound'] == compound).astype(int)

# Select features in correct order
X_new = new_lap_df[feature_names].values

# Scale features
X_new_scaled = feature_scaler.transform(X_new)

4. Make prediction:
pred_scaled = model.predict(X_new_scaled)
pred_seconds = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]

print(f"Predicted lap time: {pred_seconds:.2f} seconds")

5. For pit laps, add pit time:
# If it's a pit lap
if is_pit_lap:
    pit_time = 20.0  # seconds for pit stop
    final_lap_time = pred_seconds + pit_time
