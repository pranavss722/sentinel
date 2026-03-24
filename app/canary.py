"""Canary controller — Bernoulli traffic splitting and SLO-based rollback."""
from __future__ import annotations

import asyncio
import logging
import random
from collections import deque
from typing import Any

import numpy as np

from app.metrics import canary_rollback_total, route_decisions_total

logger = logging.getLogger(__name__)

CANARY_STAGES = [0.01, 0.10, 0.50, 1.0]


class CanaryController:
    """Routes prediction requests between champion and challenger models."""

    def __init__(
        self,
        registry: Any,
        drift_monitor: Any,
        canary_weight: float = 0.0,
        slo_p99_ms: float = 200.0,
        slo_error_rate_pct: float = 1.0,
    ) -> None:
        self._registry = registry
        self._drift_monitor = drift_monitor
        self.canary_weight = canary_weight
        self._slo_p99_ms = slo_p99_ms
        self._slo_error_rate_pct = slo_error_rate_pct
        self._latencies: deque[float] = deque(maxlen=10_000)
        self._errors: deque[bool] = deque(maxlen=10_000)
        self._stage_index = self._current_stage_index()
        self._poll_task: asyncio.Task | None = None

    def _current_stage_index(self) -> int:
        for i, stage in enumerate(CANARY_STAGES):
            if abs(self.canary_weight - stage) < 1e-9:
                return i
        return -1

    def route_request(self, payload: np.ndarray) -> dict:
        """Route a single request via Bernoulli draw on canary_weight."""
        challenger = self._registry.get_challenger()
        use_challenger = (
            challenger is not None
            and random.random() < self.canary_weight
        )

        if use_challenger:
            model = challenger
            model_label = "challenger"
        else:
            model = self._registry.get_champion()
            model_label = "champion"

        route_decisions_total.labels(model=model_label).inc()
        proba = model.predict_proba(payload)
        score = float(proba[0, 1])

        return {"model": model_label, "score": score}

    def record_latency(self, latency_s: float) -> None:
        self._latencies.append(latency_s)

    def record_request(self, error: bool = False) -> None:
        self._errors.append(error)

    def check_slo(self) -> bool:
        """Return True if SLO holds over recorded window."""
        if self._latencies:
            p99_s = float(np.percentile(list(self._latencies), 99))
            if p99_s * 1000 > self._slo_p99_ms:
                return False

        if self._errors:
            error_rate = sum(self._errors) / len(self._errors) * 100
            if error_rate > self._slo_error_rate_pct:
                return False

        return True

    def rollback(self) -> None:
        """Reset canary weight to 0 — all traffic to champion."""
        self.canary_weight = 0.0
        self._stage_index = -1
        canary_rollback_total.inc()

    def advance_stage(self) -> None:
        """Move to the next canary stage if one exists."""
        next_index = self._stage_index + 1
        if next_index < len(CANARY_STAGES):
            self._stage_index = next_index
            self.canary_weight = CANARY_STAGES[self._stage_index]

    async def _drift_poll_loop(self, interval_seconds: int) -> None:
        while True:
            if self._drift_monitor.should_rollback():
                self.rollback()
                logger.warning("Auto-rollback triggered: drift threshold exceeded")
            await asyncio.sleep(interval_seconds)

    def stop_drift_polling(self) -> None:
        """Cancel the drift polling task cleanly."""
        if self._poll_task is not None and not self._poll_task.done():
            self._poll_task.cancel()
