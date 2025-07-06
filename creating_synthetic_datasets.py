from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import ADASYN
import pandas as pd
import numpy as np
import os

# loading data and feature engineering
data = pd.read_csv('fan_dataset.csv')
data.drop(columns=['Timestamp'], inplace=True)
data['vibration_total'] = np.sqrt(data['Vibration_X']**2 + data['Vibration_Y']**2 + data['Vibration_Z']**2)

# creating list of columns
failure_cols = ['Overheating', 'Misalignment', 'Bearing Failure', 'Imbalance', 'Electrical Issue']
categorical_cols = ['Equipment Type', 'Location']
numerical_cols = list(set(data.columns) - set(failure_cols) - set(categorical_cols))

# creating column trandformers for categorical encoding
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(drop='first'), categorical_cols)
], remainder='passthrough')

# function for creating different datasets
def prepare_failure_dataset(data, target_label):
    """
    Creates a synthetic dataset for a given failure label using ADASYN.
    """
    X = data.drop(columns=failure_cols)
    y = data[target_label]

    # One-hot encode
    X_encoded = preprocessor.fit_transform(X)
    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    all_features = list(cat_features) + numerical_cols
    X_encoded_df = pd.DataFrame(X_encoded, columns=all_features)

    # Apply ADASYN
    try:
        sampler = ADASYN(sampling_strategy='minority', random_state=42, n_neighbors=5)
        X_resampled, y_resampled = sampler.fit_resample(X_encoded_df, y)
        return pd.DataFrame(X_resampled, columns=all_features), pd.Series(y_resampled, name=target_label)
    except ValueError as e:
        print(f"⚠️ Skipping {target_label} due to error: {e}")
        return X_encoded_df, y  # return original if ADASYN fails
    

# calling the function
os.makedirs("synthetic_datasets", exist_ok=True)
for failure_label in failure_cols:
    X, y = prepare_failure_dataset(data, failure_label)
    
    dataset = X.copy()
    dataset[failure_label] = y

    filename = f"synthetic_datasets/dataset_{failure_label.replace(' ', '_')}.csv"
    dataset.to_csv(filename, index=False)

    # Step 4: Confirm
    print(f"✅ Saved {filename} — Shape: {dataset.shape} | Positives: {y.sum()}")
