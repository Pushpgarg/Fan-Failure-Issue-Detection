from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import ADASYN
import pandas as pd
import numpy as np
import os
import joblib

# === Step 1: Load dataset and feature engineering ===
data = pd.read_csv('fan_dataset.csv')
data.drop(columns=['Timestamp'], inplace=True)

# Add new feature: vibration_total
data['vibration_total'] = np.sqrt(data['Vibration_X']**2 + data['Vibration_Y']**2 + data['Vibration_Z']**2)

# === Step 2: Define column groups ===
failure_cols = ['Overheating', 'Misalignment', 'Bearing Failure', 'Imbalance', 'Electrical Issue']
categorical_cols = ['Equipment Type', 'Location']
numerical_cols = list(set(data.columns) - set(failure_cols) - set(categorical_cols))

# === Step 3: Setup preprocessor ===
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(drop='first'), categorical_cols)
], remainder='passthrough')

# Fit encoder once and save for app.py
preprocessor.fit(data.drop(columns=failure_cols))
joblib.dump(preprocessor, "encoder.pkl")
print("✅ Saved encoder.pkl")

# === Step 4: Define ADASYN preparation function ===
def prepare_failure_dataset(data, target_label):
    """
    Creates a synthetic dataset for a given failure label using ADASYN.
    Returns (X, y) where X is a DataFrame and y is a Series.
    """
    X = data.drop(columns=failure_cols)
    y = data[target_label]

    # One-hot encode
    X_encoded = preprocessor.transform(X)
    cat_features = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols))
    all_features = cat_features + numerical_cols
    X_encoded_df = pd.DataFrame(X_encoded, columns=all_features)

    # Apply ADASYN
    try:
        sampler = ADASYN(sampling_strategy='minority', random_state=42, n_neighbors=5)
        X_resampled_np, y_resampled = sampler.fit_resample(X_encoded_df, y)

        # Convert back to DataFrame
        X_resampled_df = pd.DataFrame(X_resampled_np, columns=all_features)

        # Fix one-hot encoded columns (convert float back to int 0/1)
        for col in cat_features:
            if col in X_resampled_df.columns:
                X_resampled_df[col] = (X_resampled_df[col] > 0.5).astype(int)

        return X_resampled_df, pd.Series(y_resampled, name=target_label)

    except ValueError as e:
        print(f"⚠️ Skipping {target_label} due to error: {e}")
        return X_encoded_df, y  # fallback: return unbalanced original

# === Step 5: Create synthetic datasets ===
os.makedirs("synthetic_datasets", exist_ok=True)

for failure_label in failure_cols:
    X, y = prepare_failure_dataset(data, failure_label)

    dataset = X.copy()
    dataset[failure_label] = y

    filename = f"synthetic_datasets/dataset_{failure_label.replace(' ', '_')}.csv"
    dataset.to_csv(filename, index=False)
    print(f"✅ Saved {filename} — Shape: {dataset.shape} | Positives: {y.sum()}")
