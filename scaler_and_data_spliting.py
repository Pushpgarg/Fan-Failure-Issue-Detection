import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Create folders if they don't exist
os.makedirs("scalers", exist_ok=True)
os.makedirs("split_data", exist_ok=True)

failure_cols = ['Overheating', 'Misalignment', 'Bearing Failure', 'Imbalance', 'Electrical Issue']

# Loop through each failure dataset
for failure_label in failure_cols:
    path = f"synthetic_datasets/dataset_{failure_label.replace(' ', '_')}.csv"
    df = pd.read_csv(path)

    # Split X and y
    X = df.drop(columns=[failure_label])
    y = df[failure_label]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale only numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    scaler_path = f"scalers/scaler_{failure_label.replace(' ', '_')}.pkl"
    joblib.dump(scaler, scaler_path)

    # Save split data
    label_folder = f"split_data/{failure_label.replace(' ', '_')}"
    os.makedirs(label_folder, exist_ok=True)

    pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(f"{label_folder}/X_train.csv", index=False)
    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(f"{label_folder}/X_test.csv", index=False)
    y_train.to_csv(f"{label_folder}/y_train.csv", index=False)
    y_test.to_csv(f"{label_folder}/y_test.csv", index=False)

    print(f"✅ {failure_label}: Scaler saved, train/test split done → Total: {len(df)}")
