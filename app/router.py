"""Prediction and health-check routes."""
from __future__ import annotations

import time

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

from app.canary import CanaryController
from app.metrics import prediction_requests_total, prediction_latency_seconds


class PredictRequest(BaseModel):
    features: list[float]


class PredictResponse(BaseModel):
    score: float
    model: str


def create_router(canary_controller: CanaryController) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    async def health():
        return {"status": "ok"}

    @router.post("/predict", response_model=PredictResponse)
    async def predict(req: PredictRequest):
        prediction_requests_total.inc()
        start = time.perf_counter()

        payload = np.array([req.features])
        result = canary_controller.route_request(payload)

        latency = time.perf_counter() - start
        prediction_latency_seconds.observe(latency)
        canary_controller.record_latency(latency)
        canary_controller.record_request(error=False)

        return PredictResponse(score=result["score"], model=result["model"])

    return router
