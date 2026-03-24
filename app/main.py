"""FastAPI application entry point."""
from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from mlflow.tracking import MlflowClient
from prometheus_client import generate_latest

from app.canary import CanaryController
from app.drift import DriftMonitor
from app.model_registry import ModelRegistry
from app.router import create_router

FEATURE_NAMES = [f"f{i}" for i in range(10)]
MODEL_NAME = "ad-click-baseline"


def create_app() -> FastAPI:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    canary_weight = float(os.environ.get("CANARY_WEIGHT", "0.0"))
    slo_p99_ms = float(os.environ.get("SLO_P99_MS", "200"))
    slo_error_rate_pct = float(os.environ.get("SLO_ERROR_RATE_PCT", "1.0"))

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    registry = ModelRegistry(
        client=client,
        model_name=MODEL_NAME,
        loader_fn=mlflow.pyfunc.load_model,
    )

    drift_monitor = DriftMonitor(
        feature_names=FEATURE_NAMES,
        psi_threshold=0.2,
        kl_threshold=0.1,
    )

    canary_controller = CanaryController(
        registry=registry,
        drift_monitor=drift_monitor,
        canary_weight=canary_weight,
        slo_p99_ms=slo_p99_ms,
        slo_error_rate_pct=slo_error_rate_pct,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        canary_controller._poll_task = asyncio.create_task(
            canary_controller._drift_poll_loop(interval_seconds=60)
        )
        yield
        canary_controller.stop_drift_polling()

    app = FastAPI(title="Sentinel", lifespan=lifespan)
    app.state.canary_controller = canary_controller
    router = create_router(canary_controller)
    app.include_router(router)

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(generate_latest(), media_type="text/plain")

    return app


app = create_app()
