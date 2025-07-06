import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Folder to save trained models
os.makedirs("trained_models", exist_ok=True)

failure_cols = ['Overheating', 'Misalignment', 'Bearing Failure', 'Imbalance', 'Electrical Issue']

for failure_label in failure_cols:
    label = failure_label.replace(" ", "_")
    print(f"\n=== 🚀 Training model for: {failure_label} ===")

    # Paths
    base_path = f"split_data/{label}"
    X_train_path = f"{base_path}/X_train.csv"
    X_test_path = f"{base_path}/X_test.csv"
    y_train_path = f"{base_path}/y_train.csv"
    y_test_path = f"{base_path}/y_test.csv"

    # Load data
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path).squeeze()
    y_test = pd.read_csv(y_test_path).squeeze()

    # Train model
    model = RandomForestClassifier(
        n_estimators=40,
        class_weight='balanced',
        random_state=42,
        max_depth=11
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred_train = model.predict(X_train)
    print("=== 📊 Training Evaluation ===")
    print(classification_report(y_train, y_pred_train))
    y_pred_test = model.predict(X_test)
    print("=== 📊 Testing Evaluation ===")
    print(classification_report(y_test, y_pred_test))

    # Save model
    model_path = f"trained_models/model_{label}.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Model saved to {model_path}")
