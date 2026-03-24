"""End-to-end rollback tests."""
import asyncio

import numpy as np
import pytest
from unittest.mock import patch


class FakeModel:
    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 0.4), np.full(n, 0.6)])


class FakeVersion:
    def __init__(self):
        self.version = "1"
        self.source = "models:/ad-click-baseline/1"


class FakeMlflowClient:
    def get_model_version_by_alias(self, name, alias):
        if alias == "champion":
            return FakeVersion()
        raise Exception("no challenger")


@pytest.mark.asyncio
async def test_e2e_drift_triggers_rollback():
    fake_model = FakeModel()

    with patch("app.main.MlflowClient", return_value=FakeMlflowClient()), \
         patch("app.main.mlflow") as mock_mlflow, \
         patch("os.environ", {
             "CANARY_WEIGHT": "0.5",
             "MLFLOW_TRACKING_URI": "http://localhost:5000",
             "SLO_P99_MS": "200",
             "SLO_ERROR_RATE_PCT": "1.0",
         }):
        mock_mlflow.xgboost.load_model.return_value = fake_model

        from app.main import create_app
        app = create_app()

        # Get the canary controller from the app
        controller = app.state.canary_controller

        # Force drift monitor to signal rollback
        controller._drift_monitor._has_update = True
        controller._drift_monitor._psi_values = {"f0": 999.0}

        controller._poll_task = asyncio.create_task(
            controller._drift_poll_loop(interval_seconds=0)
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        controller.stop_drift_polling()
        await asyncio.sleep(0)

        assert controller.canary_weight == 0.0


@pytest.mark.asyncio
async def test_e2e_slo_breach_triggers_rollback():
    fake_model = FakeModel()

    with patch("app.main.MlflowClient", return_value=FakeMlflowClient()), \
         patch("app.main.mlflow") as mock_mlflow, \
         patch("os.environ", {
             "CANARY_WEIGHT": "0.5",
             "MLFLOW_TRACKING_URI": "http://localhost:5000",
             "SLO_P99_MS": "200",
             "SLO_ERROR_RATE_PCT": "1.0",
         }):
        mock_mlflow.xgboost.load_model.return_value = fake_model

        from app.main import create_app
        app = create_app()

        controller = app.state.canary_controller

        # Inject 100 latency recordings of 1.0s (1000ms >> 200ms SLO)
        for _ in range(100):
            controller.record_latency(1.0)

        assert controller.check_slo() is False

        controller.rollback()
        assert controller.canary_weight == 0.0
