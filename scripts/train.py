"""
Train a RandomForest churn classifier and log the run to MLflow.

Usage (from project root):
    python scripts/train.py

Environment variables:
    MLFLOW_TRACKING_URI  – defaults to http://localhost:5000
    DATA_PATH            – defaults to data/churn.csv
    MODEL_OUTPUT_PATH    – defaults to app/model/model.pkl
"""
import os
import pickle

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
DATA_PATH = os.getenv("DATA_PATH", "data/churn.csv")
MODEL_PATH = os.getenv("MODEL_OUTPUT_PATH", "app/model/model.pkl")

FEATURES = ["monthly_spend", "tenure_months", "support_tickets", "login_frequency"]
TARGET = "churned"

# Hyperparameters (easy to swap for a grid-search later)
PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "min_samples_leaf": 5,
    "class_weight": "balanced",
    "random_state": 42,
}


def load_data(path: str):
    df = pd.read_csv(path)
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    }


def main():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("churn-prediction")

    with mlflow.start_run():
        X_train, X_test, y_train, y_test = load_data(DATA_PATH)

        model = RandomForestClassifier(**PARAMS)
        model.fit(X_train, y_train)

        metrics = evaluate(model, X_test, y_test)

        mlflow.log_params(PARAMS)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        print("Metrics:", metrics)

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        print(f"Model saved → {MODEL_PATH}")


if __name__ == "__main__":
    main()
