import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
import numpy as np

# Create folder for trained models
os.makedirs("trained_models", exist_ok=True)

failure_cols = ['Overheating', 'Misalignment', 'Bearing Failure', 'Imbalance', 'Electrical Issue']

for failure_label in failure_cols:
    label = failure_label.replace(" ", "_")
    print(f"\n=== 🚀 Tuning model for: {failure_label} ===")

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

    # Optuna objective
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1 
        }

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        f1_scores = []
        overfit_penalties = []

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            clf = RandomForestClassifier(**params)
            clf.fit(X_tr, y_tr)

            y_tr_pred = clf.predict(X_tr)
            y_val_pred = clf.predict(X_val)

            train_f1 = f1_score(y_tr, y_tr_pred, average='weighted')
            val_f1 = f1_score(y_val, y_val_pred, average='weighted')

            overfit_penalties.append(abs(train_f1 - val_f1))
            f1_scores.append(val_f1)

        avg_f1 = np.mean(f1_scores)
        avg_overfit = np.mean(overfit_penalties)

        return avg_f1 - avg_overfit  # high validation F1, low overfitting


    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    print(f"✅ Best trial: {study.best_trial.params}")

    # Train final model on best params
    best_params = study.best_trial.params
    best_params["class_weight"] = "balanced"
    best_params["random_state"] = 42

    final_model = RandomForestClassifier(**best_params)
    final_model.fit(X_train, y_train)

    # Evaluate
    print("=== 📊 Training Evaluation ===")
    y_pred_train = final_model.predict(X_train)
    print(classification_report(y_train, y_pred_train))

    print("=== 📊 Testing Evaluation ===")
    y_pred_test = final_model.predict(X_test)
    print(classification_report(y_test, y_pred_test))

    # Save model
    model_path = f"trained_models/model_{label}.pkl"
    joblib.dump(final_model, model_path)
    print(f"✅ Model saved to {model_path}")
