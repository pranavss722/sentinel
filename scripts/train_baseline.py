"""Train and register the baseline XGBoost model in MLflow."""
from __future__ import annotations

import os

import mlflow
import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

MODEL_NAME = "ad-click-baseline"


def generate_synthetic_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic binary classification dataset: 10K rows, 10 features."""
    X, y = make_classification(
        n_samples=10_000,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        random_state=42,
    )
    return X, y


def train_model(X: np.ndarray, y: np.ndarray) -> xgb.XGBClassifier:
    """Train an XGBoost binary classifier."""
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X, y)
    return model


def main() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    X, y = generate_synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run(run_name="baseline-xgb"):
        model = train_model(X_train, y_train)

        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, proba),
        }
        mlflow.log_metrics(metrics)
        mlflow.log_params(model.get_params())

        model_info = mlflow.xgboost.log_model(model, artifact_path="model")

        result = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=MODEL_NAME,
        )

        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias(MODEL_NAME, "champion", result.version)

        print(f"Registered {MODEL_NAME} v{result.version} as champion")
        print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
