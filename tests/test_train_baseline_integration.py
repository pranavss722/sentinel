"""Integration tests for train_baseline.py."""
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from scripts.train_baseline import generate_synthetic_data, train_model, main


def test_generate_synthetic_data_is_reproducible():
    X1, y1 = generate_synthetic_data()
    X2, y2 = generate_synthetic_data()
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_train_model_produces_valid_probabilities():
    X, y = generate_synthetic_data()
    model = train_model(X, y)
    proba = model.predict_proba(X[:10])
    assert proba.shape == (10, 2)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_train_model_achieves_minimum_auc():
    X, y = generate_synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = train_model(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    assert auc >= 0.85, f"AUC {auc:.4f} is below 0.85 minimum"


def test_main_registers_champion_alias():
    mock_client_instance = MagicMock()
    mock_register_result = MagicMock()
    mock_register_result.version = "1"

    mock_model_info = MagicMock()
    mock_model_info.model_uri = "runs:/abc123/model"

    with patch("scripts.train_baseline.mlflow") as mock_mlflow:
        mock_mlflow.tracking.MlflowClient.return_value = mock_client_instance
        mock_mlflow.start_run.return_value.__enter__ = MagicMock()
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
        mock_mlflow.xgboost.log_model.return_value = mock_model_info
        mock_mlflow.register_model.return_value = mock_register_result

        main()

        mock_mlflow.set_tracking_uri.assert_called_once()
        mock_mlflow.register_model.assert_called_once_with(
            model_uri="runs:/abc123/model",
            name="ad-click-baseline",
        )
        mock_client_instance.set_registered_model_alias.assert_called_once_with(
            "ad-click-baseline", "champion", "1"
        )
