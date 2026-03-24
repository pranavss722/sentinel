"""Tests for main app wiring."""
import pytest
from unittest.mock import patch
import numpy as np
from httpx import AsyncClient, ASGITransport


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
async def test_app_health_endpoint():
    fake_model = FakeModel()

    with patch("app.main.MlflowClient", return_value=FakeMlflowClient()), \
         patch("app.main.mlflow") as mock_mlflow:
        mock_mlflow.xgboost.load_model.return_value = fake_model

        from app.main import create_app
        app = create_app()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_app_predict_endpoint():
    fake_model = FakeModel()

    with patch("app.main.MlflowClient", return_value=FakeMlflowClient()), \
         patch("app.main.mlflow") as mock_mlflow:
        mock_mlflow.xgboost.load_model.return_value = fake_model

        from app.main import create_app
        app = create_app()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/predict", json={"features": [1.0] * 10})
        assert resp.status_code == 200
        body = resp.json()
        assert body["model"] == "champion"
        assert "score" in body


@pytest.mark.asyncio
async def test_app_has_metrics_endpoint():
    fake_model = FakeModel()

    with patch("app.main.MlflowClient", return_value=FakeMlflowClient()), \
         patch("app.main.mlflow") as mock_mlflow:
        mock_mlflow.xgboost.load_model.return_value = fake_model

        from app.main import create_app
        app = create_app()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/metrics")
        assert resp.status_code == 200
        assert "prediction_requests_total" in resp.text
