"""Health and predict endpoint tests."""
import pytest
import numpy as np
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI

from app.router import create_router


class FakeModel:
    def __init__(self, pred_value: float = 0.5):
        self._pred_value = pred_value

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([
            np.full(n, 1 - self._pred_value),
            np.full(n, self._pred_value),
        ])


class FakeRegistry:
    def __init__(self, champion_pred=0.5, challenger_pred=None):
        self._champion = FakeModel(champion_pred)
        self._challenger = FakeModel(challenger_pred) if challenger_pred else None

    def get_champion(self):
        return self._champion

    def get_challenger(self):
        return self._challenger


class FakeDriftMonitor:
    def should_rollback(self):
        return False


def make_app(canary_weight=0.0, champion_pred=0.5):
    from app.canary import CanaryController

    registry = FakeRegistry(champion_pred=champion_pred)
    drift_monitor = FakeDriftMonitor()
    controller = CanaryController(
        registry=registry,
        drift_monitor=drift_monitor,
        canary_weight=canary_weight,
    )
    router = create_router(controller)
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.mark.asyncio
async def test_health_returns_200():
    app = make_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_health_response_body():
    app = make_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    body = resp.json()
    assert body["status"] == "ok"


@pytest.mark.asyncio
async def test_predict_returns_200():
    app = make_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/predict", json={"features": [1.0] * 10})
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_predict_returns_score_and_model():
    app = make_app(champion_pred=0.7)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/predict", json={"features": [1.0] * 10})
    body = resp.json()
    assert "score" in body
    assert "model" in body
    assert body["model"] == "champion"
    assert body["score"] == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_predict_records_latency():
    app = make_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/predict", json={"features": [1.0] * 10})
