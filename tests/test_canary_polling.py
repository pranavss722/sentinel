"""Tests for CanaryController drift polling loop."""
import asyncio
from unittest.mock import patch

import numpy as np
import pytest

from app.metrics import canary_rollback_total


class FakeModel:
    def __init__(self, name: str):
        self.name = name

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])


class FakeRegistry:
    def __init__(self):
        self._champion = FakeModel("champion")
        self._challenger = FakeModel("challenger")

    def get_champion(self):
        return self._champion

    def get_challenger(self):
        return self._challenger


class FakeDriftMonitor:
    def __init__(self, should_roll: bool = False):
        self._should_roll = should_roll

    def should_rollback(self) -> bool:
        return self._should_roll



def make_controller(canary_weight=0.5, drift_rollback=False):
    from app.canary import CanaryController
    return CanaryController(
        registry=FakeRegistry(),
        drift_monitor=FakeDriftMonitor(drift_rollback),
        canary_weight=canary_weight,
        slo_p99_ms=200.0,
        slo_error_rate_pct=1.0,
    )


@pytest.mark.asyncio
async def test_polling_calls_rollback_when_drift_detected():
    controller = make_controller(canary_weight=0.5, drift_rollback=True)
    before = canary_rollback_total._value.get()

    controller._poll_task = asyncio.create_task(
        controller._drift_poll_loop(interval_seconds=0)
    )
    # Yield control so the background task runs at least one iteration
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    controller.stop_drift_polling()

    assert controller.canary_weight == 0.0
    assert canary_rollback_total._value.get() > before


@pytest.mark.asyncio
async def test_polling_does_not_rollback_when_no_drift():
    controller = make_controller(canary_weight=0.5, drift_rollback=False)

    controller._poll_task = asyncio.create_task(
        controller._drift_poll_loop(interval_seconds=0)
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    controller.stop_drift_polling()

    assert controller.canary_weight == 0.5


@pytest.mark.asyncio
async def test_stop_polling_cancels_task():
    controller = make_controller(canary_weight=0.5, drift_rollback=False)

    controller._poll_task = asyncio.create_task(
        controller._drift_poll_loop(interval_seconds=0)
    )
    await asyncio.sleep(0)
    controller.stop_drift_polling()
    # Yield so the cancellation propagates
    await asyncio.sleep(0)

    task = controller._poll_task
    assert task is not None
    assert task.done() or task.cancelled()

    # Set drift to True after stopping — weight should NOT change
    controller._drift_monitor._should_roll = True
    controller.canary_weight = 0.5
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert controller.canary_weight == 0.5


@pytest.mark.asyncio
async def test_polling_never_calls_get_event_loop():
    """CanaryController must not call the deprecated asyncio.get_event_loop()."""
    controller = make_controller(canary_weight=0.5, drift_rollback=False)

    with patch("app.canary.asyncio.get_event_loop") as mock_get_loop:
        controller._poll_task = asyncio.create_task(
            controller._drift_poll_loop(interval_seconds=0)
        )
        await asyncio.sleep(0)
        controller.stop_drift_polling()
        await asyncio.sleep(0)

        mock_get_loop.assert_not_called()
